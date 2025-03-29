"""Traj definitions."""

from typing import Optional, Dict

import torch
from torch import nn
from egoallo.config import CONFIG_FILE
from jaxtyping import Bool
from jaxtyping import Float
from torch import Tensor
from egoallo.type_stubs import SmplFamilyModelType
from .base_traj import BaseDenoiseTraj
import dataclasses
from egoallo.config import make_cfg
from egoallo.utils.setup_logger import setup_logger

local_config_file = CONFIG_FILE
CFG = make_cfg(config_name="defaults", config_file=local_config_file, cli_args=[])

logger = setup_logger(output=None, name=__name__)


@dataclasses.dataclass
class JointsOnlyTraj(BaseDenoiseTraj):
    """Denoising trajectory that only predicts joint positions."""

    joints: Float[Tensor, "*batch timesteps 22 3"]
    """3D joint positions."""

    def __init__(
        self,
        joints: Float[Tensor, "*batch timesteps 22 3"],
        **kwargs,  # Ignore other parameters
    ):
        # TODO: Remove this once we have a proper constructor.
        """Initialize JointsOnlyTraj with just joint positions.

        Args:
            joints: Joint positions tensor of shape (*batch, timesteps, 22, 3)
            **kwargs: Additional arguments that will be ignored
        """
        raise DeprecationWarning(
            "JointsOnlyTraj is deprecated. Use AbsoluteDenoiseTraj instead.",
        )
        self.joints = joints

    @property
    def loss_weights(self) -> dict[str, float]:
        # Default loss weights for absolute mode
        absolute_weights = {
            "joints": 3.0,
        }
        return absolute_weights

    def compute_loss(
        self,
        other: "JointsOnlyTraj",
        mask: Bool[Tensor, "batch time"],
        weight_t: Float[Tensor, "batch"],
    ) -> dict[str, Float[Tensor, ""]]:
        """Compute loss between this trajectory and another using joint positions only."""
        batch, time = mask.shape[:2]

        loss_terms = {
            "joints": self._weight_and_mask_loss(
                ((self.joints - other.joints) ** 2).reshape(batch, time, -1),
                mask,
                weight_t,
            ),
        }

        return loss_terms

    @staticmethod
    def get_packed_dim(include_hands: bool = False) -> int:
        """Get dimension of packed state vector.

        Args:
            include_hands: Whether to include hand rotations (unused in this class).

        Returns:
            Total dimension of packed state vector.
        """
        # 22 joints * 3 coordinates per joint
        return CFG.smplh.num_joints * 3

    def pack(self) -> Float[Tensor, "*batch timesteps d_state"]:
        """Pack trajectory into a single flattened vector."""
        return self.joints.reshape(*self.joints.shape[:-2], -1)

    @classmethod
    def unpack(
        cls,
        x: Float[Tensor, "*batch timesteps d_state"],
        include_hands: bool = False,
        project_rotmats: bool = False,
    ) -> "JointsOnlyTraj":
        """Unpack trajectory from a single flattened vector."""
        (*batch, time, d_state) = x.shape
        assert d_state == cls.get_packed_dim(include_hands)

        joints = x.reshape(*batch, time, CFG.smplh.num_joints, 3)
        return cls(joints=joints)

    def apply_to_body(self, body_model: "SmplFamilyModelType") -> "SmplFamilyModelType":
        """Apply the trajectory data to a SMPL-H body model.

        Note: This is not implemented for JointsOnlyTraj since we don't have
        the necessary parameters to fully pose the SMPL-H model.
        """
        raise NotImplementedError(
            "JointsOnlyTraj does not support applying to SMPL-H body model",
        )

    def encode(
        self,
        encoders: nn.ModuleDict,
        batch: int,
        time: int,
    ) -> Float[Tensor, "batch time d_latent"]:
        """Encode joints-only trajectory into latent space."""
        return encoders["joints"](self.joints.reshape((batch, time, -1)))

    def _compute_metrics(
        self,
        other: "JointsOnlyTraj",
        body_model: Optional["SmplFamilyModelType"] = None,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, float]:
        """Compute metrics between this trajectory and another.
        Only computes joint position based metrics since this class only has joint data.
        """
        # TEMPORARY_FIX: import BodyEvaluator lazily to avoid circular imports
        from egoallo.evaluation.body_evaluator import BodyEvaluator

        assert self.check_shapes(other), f"{self.check_shapes(other)}"
        metrics = {}

        metrics["mpjpe"] = float(
            BodyEvaluator.compute_mpjpe(
                label_root_pos=other.joints[..., 0, :],  # [batch, T, 3]
                label_joint_pos=other.joints[..., 1:22, :],  # [batch, T, 21, 3]
                pred_root_pos=self.joints[..., 0, :],  # [batch, T, 3]
                pred_joint_pos=self.joints[..., 1:22, :],  # [batch, T, 21, 3]
                per_frame_procrustes_align=False,
                device=device,
            ).mean(),
        )

        metrics["pampjpe"] = float(
            BodyEvaluator.compute_mpjpe(
                label_root_pos=other.joints[..., 0, :],  # [batch, T, 3]
                label_joint_pos=other.joints[..., 1:22, :],  # [batch, T, 21, 3]
                pred_root_pos=self.joints[..., 0, :],  # [batch, T, 3]
                pred_joint_pos=self.joints[..., 1:22, :],  # [batch, T, 21, 3]
                per_frame_procrustes_align=True,
                device=device,
            ).mean(),
        )

        return metrics

    @classmethod
    def get_modality_dict(cls, include_hands: bool = False) -> dict[str, int]:
        """Get dictionary of modalities and their dimensions.

        For JointsOnlyTraj, only joint positions are included.

        Args:
            include_hands: Not used for this trajectory type

        Returns:
            Dictionary mapping modality names to their dimensions
        """
        num_smplh_jnts = CFG.smplh.num_joints
        return {
            "joints": num_smplh_jnts * 3,  # x,y,z coordinates for each joint
        }
