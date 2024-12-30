"""Network definitions."""

import dataclasses
from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import cached_property, cache
from typing import (
    Any,
    Generic,
    Literal,
    TypeVar,
    Union,
    assert_never,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor

from math import ceil
from pathlib import Path

import numpy as np
import torch
import typeguard
from einops import rearrange
from jaxtyping import Bool, Float, Int, jaxtyped
from rotary_embedding_torch import RotaryEmbedding
from torch import Tensor, nn

from egoallo.config import CONFIG_FILE, make_cfg
from egoallo.types import JointCondMode, DenoiseTrajType

from .fncsmpl import SmplhModel, SmplhShapedAndPosed
from .tensor_dataclass import TensorDataclass
from .transforms import SE3, SO3

local_config_file = CONFIG_FILE
CFG = make_cfg(config_name="defaults", config_file=local_config_file, cli_args=[])

from egoallo.utils.setup_logger import setup_logger

logger = setup_logger(output=None, name=__name__)


def get_kinematic_chain(use_smplh: bool = True) -> list[tuple[int, int]]:
    """Get kinematic chain for SMPL-H model based on SMPL_JOINT_NAMES."""
    # Define parent-child relationships for joints based on SMPL_JOINT_NAMES order:
    # ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    # 'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    # 'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    # 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand']
    # Each tuple is (child_idx, parent_idx)
    kinematic_chain = [
        (1, 0),  # left_hip -> pelvis
        (2, 0),  # right_hip -> pelvis
        (3, 0),  # spine1 -> pelvis
        (4, 1),  # left_knee -> left_hip
        (5, 2),  # right_knee -> right_hip
        (6, 3),  # spine2 -> spine1
        (7, 4),  # left_ankle -> left_knee
        (8, 5),  # right_ankle -> right_knee
        (9, 6),  # spine3 -> spine2
        (10, 7),  # left_foot -> left_ankle
        (11, 8),  # right_foot -> right_ankle
        (12, 9),  # neck -> spine3
        (13, 12),  # left_collar -> neck
        (14, 12),  # right_collar -> neck
        (15, 12),  # head -> neck
        (16, 13),  # left_shoulder -> left_collar
        (17, 14),  # right_shoulder -> right_collar
        (18, 16),  # left_elbow -> left_shoulder
        (19, 17),  # right_elbow -> right_shoulder
        (20, 18),  # left_wrist -> left_elbow
        (21, 19),  # right_wrist -> right_elbow
        (22, 20),  # left_hand -> left_wrist
        (23, 21),  # right_hand -> right_wrist
    ]
    if use_smplh:
        return kinematic_chain
    else:
        return kinematic_chain[:-2]


@jaxtyped(typechecker=typeguard.typechecked)
def project_rotmats_via_svd(
    rotmats: Float[Tensor, "*batch 3 3"],
) -> Float[Tensor, "*batch 3 3"]:
    u, s, vh = torch.linalg.svd(rotmats)
    del s
    return torch.einsum("...ij,...jk->...ik", u, vh)


T = TypeVar("T", bound="BaseDenoiseTraj")


@dataclasses.dataclass
class DenoisingConfig:
    """Configuration for denoising.

    This class handles configuration for both absolute and velocity-based denoising,
    and provides factory methods for creating appropriate trajectory objects.
    """

    denoising_mode: Literal["absolute", "velocity", "joints_only"] = "absolute"
    temporal_window: int = 2
    use_acceleration: bool = False
    loss_weights: dict[str, float] | None = None
    joint_cond_mode: JointCondMode = "absrel"
    include_hands: bool = False
    def __post_init__(self):
        if self.loss_weights is None:
            # Default loss weights for absolute mode
            absolute_weights = {
                "betas": 0.05,
                "body_rotmats": 1.00,
                "contacts": 0.05,
                "hand_rotmats": 0.00,
                "R_world_root": 0.5,
                "t_world_root": 0.5,
                "joints": 0.25,
                "foot_skating": 0.1,
                "velocity": 0.01,
            }

            # Additional weights for velocity mode
            velocity_weights = {
                # Remove R_world_root and t_world_root from absolute weights
                **{
                    k: v
                    for k, v in absolute_weights.items()
                    if k not in ["R_world_root", "t_world_root"]
                },
                "R_world_root_vel": 1.0,
                "t_world_root_vel": 1.0,
                "R_world_root_acc": 0.5,
                "t_world_root_acc": 0.5,
            }

            # Weights for joints-only mode
            joints_only_weights = {
                "joints": 100.0,
            }

            self.loss_weights = (
                velocity_weights if self.is_velocity_mode()
                else joints_only_weights if self.denoising_mode == "joints_only"
                else absolute_weights
            )

        # Set mode based on joint_cond_mode if not explicitly set
        if self.denoising_mode == "absolute" and self.is_velocity_joint_cond():
            self.denoising_mode = "velocity"

    def is_velocity_joint_cond(self) -> bool:
        """Check if the joint conditioning mode is velocity-based."""
        return self.joint_cond_mode in ("vel_acc", "vel_acc_plus")

    def is_velocity_mode(self) -> bool:
        """Check if we're using velocity-based denoising."""
        return self.denoising_mode == "velocity" or self.is_velocity_joint_cond()

    def create_trajectory(
        self, *args, **kwargs
    ) -> DenoiseTrajType:
        """Factory method to create appropriate trajectory object based on configuration."""
        if self.denoising_mode == "joints_only":
            return JointsOnlyTraj(*args, **kwargs)
        elif self.is_velocity_mode():
            return VelocityDenoiseTraj(*args, **kwargs)
        return AbsoluteDenoiseTraj(*args, **kwargs)

    def get_d_state(self) -> int:
        """Get the dimension of the state vector."""
        if self.denoising_mode == "joints_only":
            return JointsOnlyTraj.get_packed_dim(include_hands=self.include_hands)
        elif self.denoising_mode == "velocity":
            return VelocityDenoiseTraj.get_packed_dim(include_hands=self.include_hands)
        elif self.denoising_mode == "absolute":
            return AbsoluteDenoiseTraj.get_packed_dim(include_hands=self.include_hands)
        else:
            raise ValueError(f"Invalid denoising mode: {self.denoising_mode}")

    @classmethod
    def from_joint_cond_mode(
        cls, joint_cond_mode: JointCondMode, include_hands: bool, **kwargs
    ) -> "DenoisingConfig":
        """Create config from joint conditioning mode.

        Args:
            joint_cond_mode: The joint conditioning mode to use
            **kwargs: Additional configuration parameters

        Returns:
            Configured DenoisingConfig instance
        """
        if joint_cond_mode == "joints_only":
            mode = "joints_only"
        else:
            mode = "velocity" if joint_cond_mode in ("vel_acc", "vel_acc_plus") else "absolute"
        return cls(denoising_mode=mode, joint_cond_mode=joint_cond_mode, include_hands=include_hands, **kwargs)

    def unpack_traj(
        self,
        x: Float[Tensor, "*batch timesteps d_state"],
        include_hands: bool = False,
        project_rotmats: bool = False,
    ) -> "BaseDenoiseTraj[Any]":
        """Unpack trajectory data using appropriate trajectory class based on configuration."""
        if self.denoising_mode == "joints_only":
            return JointsOnlyTraj.unpack(x)
        elif self.is_velocity_mode():
            return VelocityDenoiseTraj.unpack(
                x, include_hands=include_hands, project_rotmats=project_rotmats
            )
        return AbsoluteDenoiseTraj.unpack(
            x, include_hands=include_hands, project_rotmats=project_rotmats
        )

    def fetch_modality_dict(self, include_hands: bool = False) -> dict[str, int]:
        """Get dictionary of modalities and their dimensions based on configuration."""
        num_smplh_jnts = CFG.smplh.num_joints

        if self.denoising_mode == "joints_only":
            return {
                "joints": num_smplh_jnts * 3,  # x,y,z coordinates for each joint
            }

        # Common modalities for both absolute and velocity modes
        modality_dims = {
            "betas": 16,
            "body_rotmats": (num_smplh_jnts - 1) * 9,
            "contacts": num_smplh_jnts,
        }

        # Add hand rotations if specified
        if include_hands:
            modality_dims["hand_rotmats"] = 30 * 9

        # Add mode-specific modalities
        if self.is_velocity_mode():
            # Velocity mode uses temporal offsets
            modality_dims.update(
                {
                    "R_world_root_tm1_t": 9,  # 3x3 rotation matrix
                    "t_world_root_tm1_t": 3,  # 3D translation vector
                }
            )
        else:
            # Absolute mode uses direct positions
            modality_dims.update(
                {
                    "R_world_root": 9,  # 3x3 rotation matrix
                    "t_world_root": 3,  # 3D translation vector
                }
            )

        return modality_dims

    @jaxtyped(typechecker=typeguard.typechecked)
    def from_ego_data(
        self,
        ego_data: "EgoTrainingData",
        include_hands: bool = True,
    ) -> DenoiseTrajType:
        """Convert EgoTrainingData instance to appropriate DenoiseTraj based on config.
        
        Args:
            ego_data: Input EgoTrainingData instance
            include_hands: Whether to include hand data in the output trajectory
            
        Returns:
            Appropriate trajectory object based on denoising mode
        """
        *batch, time, _ = ego_data.T_world_root.shape

        # Extract rotation and translation from T_world_root
        R_world_root = SO3(ego_data.T_world_root[..., :4]).as_matrix()
        t_world_root = ego_data.T_world_root[..., 4:7]

        # Convert body quaternions to rotation matrices
        body_rotmats = SO3(ego_data.body_quats).as_matrix()

        # Handle hand data if present
        hand_rotmats = None
        if ego_data.hand_quats is not None and include_hands:
            hand_rotmats = SO3(ego_data.hand_quats).as_matrix()

        # Create appropriate trajectory based on denoising mode
        if self.denoising_mode == "joints_only":
            return JointsOnlyTraj(
                joints=ego_data.joints_wrt_world,
            )
        elif self.is_velocity_mode():
            # For velocity mode, create VelocityDenoiseTraj
            traj = VelocityDenoiseTraj(
                betas=ego_data.betas.expand((*batch, time, 16)),
                body_rotmats=body_rotmats,
                contacts=ego_data.contacts,
                hand_rotmats=hand_rotmats,
                R_world_root=R_world_root,
                t_world_root=t_world_root,
            )
            # VelocityDenoiseTraj will compute temporal offsets in __post_init__
            return traj
        else:
            # For absolute mode, create AbsoluteDenoiseTraj
            return AbsoluteDenoiseTraj(
                betas=ego_data.betas.expand((*batch, time, 16)),
                body_rotmats=body_rotmats,
                contacts=ego_data.contacts,
                hand_rotmats=hand_rotmats,
                R_world_root=R_world_root,
                t_world_root=t_world_root,
            )


class BaseDenoiseTraj(TensorDataclass, ABC, Generic[T]):
    """Abstract base class for denoising trajectories."""

    @abstractmethod
    def pack(self) -> Float[Tensor, "*batch timesteps d_state"]:
        """Pack trajectory into a single flattened vector."""
        pass

    @classmethod
    @abstractmethod
    def unpack(
        cls,
        x: Float[Tensor, "*batch timesteps d_state"],
        include_hands: bool = False,
        project_rotmats: bool = False,
    ) -> T:
        """Unpack trajectory from a single flattened vector."""
        pass

    def _weight_and_mask_loss(
        self,
        loss_per_step: Float[Tensor, "batch time d"],
        bt_mask: Bool[Tensor, "batch time"],
        weight_t: Float[Tensor, "batch"],
        bt_mask_sum: Float[Tensor, ""] | None = None,
    ) -> Float[Tensor, ""]:
        """Weight and mask per-timestep losses (squared errors)."""
        batch, time, d = loss_per_step.shape
        assert bt_mask.shape == (batch, time)
        assert weight_t.shape == (batch,)

        if bt_mask_sum is None:
            bt_mask_sum = torch.sum(bt_mask)

        return (
            torch.sum(
                torch.sum(
                    torch.mean(loss_per_step, dim=-1) * bt_mask,
                    dim=-1,
                )
                * weight_t
            )
            / bt_mask_sum
        )

    @abstractmethod
    def compute_loss(
        self,
        other: T,
        mask: Bool[Tensor, "batch time"],
        weight_t: Float[Tensor, "batch"],
    ) -> dict[str, Float[Tensor, ""]]:
        """Compute loss between this trajectory and another."""
        pass

    @abstractmethod
    def apply_to_body(self, body_model: SmplhModel) -> SmplhShapedAndPosed:
        """Apply the trajectory data to a SMPL-H body model."""
        pass

    @abstractmethod
    def encode(
        self,
        encoders: nn.ModuleDict,
        batch: int,
        time: int,
    ) -> Float[Tensor, "batch time d_latent"]:
        """Encode trajectory into latent representation.
        
        Args:
            encoders: Dictionary of encoder networks
            batch: Batch size
            time: Sequence length
            
        Returns:
            Encoded representation of shape (batch, time, d_latent)
        """
        pass


@dataclasses.dataclass
class JointsOnlyTraj(BaseDenoiseTraj):
    """Denoising trajectory that only predicts joint positions."""

    joints: Float[Tensor, "*batch timesteps 22 3"]
    """3D joint positions."""

    def __init__(
        self,
        joints: Float[Tensor, "*batch timesteps 22 3"],
        **kwargs # Ignore other parameters
    ):
        # TODO: Remove this once we have a proper constructor.
        """Initialize JointsOnlyTraj with just joint positions.
        
        Args:
            joints: Joint positions tensor of shape (*batch, timesteps, 22, 3)
            **kwargs: Additional arguments that will be ignored
        """
        self.joints = joints

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
                weight_t
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

    def apply_to_body(self, body_model: SmplhModel) -> SmplhShapedAndPosed:
        """Apply the trajectory data to a SMPL-H body model.
        
        Note: This is not implemented for JointsOnlyTraj since we don't have
        the necessary parameters to fully pose the SMPL-H model.
        """
        raise NotImplementedError(
            "JointsOnlyTraj does not support applying to SMPL-H body model"
        )

    def encode(
        self,
        encoders: nn.ModuleDict,
        batch: int,
        time: int,
    ) -> Float[Tensor, "batch time d_latent"]:
        """Encode joints-only trajectory into latent space."""
        return encoders["joints"](self.joints.reshape((batch, time, -1)))


@dataclasses.dataclass
class AbsoluteDenoiseTraj(BaseDenoiseTraj):
    """Denoising trajectory with absolute pose representation."""

    betas: Float[Tensor, "*batch timesteps 16"]
    """Body shape parameters. We don't really need the timesteps axis here,
    it's just for convenience."""

    body_rotmats: Float[Tensor, "*batch timesteps 21 3 3"]
    """Local orientations for each body joint."""

    contacts: Float[Tensor, "*batch timesteps 22"]
    """Contact boolean for each joint."""

    hand_rotmats: Float[Tensor, "*batch timesteps 30 3 3"] | None
    """Local orientations for each body joint."""

    R_world_root: Float[Tensor, "*batch timesteps 3 3"]
    """Global rotation matrix of the root joint."""

    t_world_root: Float[Tensor, "*batch timesteps 3"]
    """Global translation vector of the root joint."""

    def compute_loss(
        self,
        other: "AbsoluteDenoiseTraj",
        mask: Bool[Tensor, "batch time"],
        weight_t: Float[Tensor, "batch"],
    ) -> dict[str, Float[Tensor, ""]]:
        """Compute loss between this trajectory and another using absolute representations."""
        batch, time = mask.shape[:2]

        loss_terms = {
            "betas": self._weight_and_mask_loss(
                (self.betas - other.betas) ** 2, mask, weight_t
            ),
            "body_rotmats": self._weight_and_mask_loss(
                (self.body_rotmats - other.body_rotmats).reshape(batch, time, -1) ** 2,
                mask,
                weight_t,
            ),
            "R_world_root": self._weight_and_mask_loss(
                (self.R_world_root - other.R_world_root).reshape(batch, time, -1) ** 2,
                mask,
                weight_t,
            ),
            "t_world_root": self._weight_and_mask_loss(
                (self.t_world_root - other.t_world_root) ** 2, mask, weight_t
            ),
        }

        if self.hand_rotmats is not None and other.hand_rotmats is not None:
            loss_terms["hand_rotmats"] = self._weight_and_mask_loss(
                (self.hand_rotmats - other.hand_rotmats).reshape(batch, time, -1) ** 2,
                mask,
                weight_t,
            )

        return loss_terms

    @staticmethod
    def get_packed_dim(include_hands: bool) -> int:
        """Get dimension of packed state vector.

        Args:
            include_hands: Whether to include hand rotations in packed dimension.

        Returns:
            Total dimension of packed state vector.
        """
        # 16 (betas) + 21*9 (body_rotmats) + 21 (contacts) + 9 (R_world_root) + 3 (t_world_root)
        num_smplh_jnts = CFG.smplh.num_joints
        packed_dim = 16 + (num_smplh_jnts - 1) * 9 + (num_smplh_jnts) + 9 + 3
        if include_hands:
            packed_dim += 30 * 9  # hand_rotmats
        return packed_dim

    def apply_to_body(self, body_model: SmplhModel) -> SmplhShapedAndPosed:
        """Apply the trajectory data to a SMPL-H body model."""
        device = self.betas.device
        dtype = self.betas.dtype
        # assert self.hand_rotmats is not None

        shaped = body_model.with_shape(
            self.betas
        )  # betas averges across timestep dimensions.
        T_world_root = SE3.from_rotation_and_translation(
            SO3.from_matrix(self.R_world_root),
            self.t_world_root,
        ).parameters()

        posed = shaped.with_pose_decomposed(
            T_world_root=T_world_root,
            body_quats=SO3.from_matrix(self.body_rotmats).wxyz,
            left_hand_quats=SO3.from_matrix(self.hand_rotmats[..., :15, :, :]).wxyz
            if self.hand_rotmats is not None
            else None,
            right_hand_quats=SO3.from_matrix(self.hand_rotmats[..., 15:30, :, :]).wxyz
            if self.hand_rotmats is not None
            else None,
        )
        # posed = shaped.with_pose(
        #     T_world_root=T_world_root,
        #     local_quats=SO3.from_matrix(
        #         torch.cat([self.body_rotmats, self.hand_rotmats], dim=-3)
        #     ).wxyz,
        # )
        return posed

    @jaxtyped(typechecker=typeguard.typechecked)
    def pack(self) -> Float[Tensor, "*batch timesteps d_state"]:
        """Pack trajectory into a single flattened vector."""
        (*batch, time, num_joints, _, _) = self.body_rotmats.shape
        assert num_joints == 21

        # Create list of tensors to pack
        tensors_to_pack = [
            self.betas.reshape((*batch, time, -1)),
            self.body_rotmats.reshape((*batch, time, -1)),
            self.contacts.reshape((*batch, time, -1)),
            self.R_world_root.reshape((*batch, time, -1)),
            self.t_world_root.reshape((*batch, time, -1)),
        ]

        if self.hand_rotmats is not None:
            tensors_to_pack.append(self.hand_rotmats.reshape((*batch, time, -1)))

        return torch.cat(tensors_to_pack, dim=-1)

    @classmethod
    @jaxtyped(typechecker=typeguard.typechecked)
    def unpack(
        cls,
        x: Float[Tensor, "*batch timesteps d_state"],
        include_hands: bool = False,
        project_rotmats: bool = False,
    ) -> "AbsoluteDenoiseTraj":
        """Unpack trajectory from a single flattened vector."""
        (*batch, time, d_state) = x.shape
        assert d_state == cls.get_packed_dim(include_hands)

        if include_hands:
            (
                betas,
                body_rotmats_flat,
                contacts,
                R_world_root,
                t_world_root,
                hand_rotmats_flat,
            ) = torch.split(
                x,
                [
                    16,
                    (CFG.smplh.num_joints - 1) * 9,
                    CFG.smplh.num_joints,
                    9,
                    3,
                    30 * 9,
                ],
                dim=-1,
            )
            body_rotmats = body_rotmats_flat.reshape(
                (*batch, time, (CFG.smplh.num_joints - 1), 3, 3)
            )
            hand_rotmats = hand_rotmats_flat.reshape((*batch, time, 30, 3, 3))
        else:
            betas, body_rotmats_flat, contacts, R_world_root, t_world_root = (
                torch.split(
                    x,
                    [16, (CFG.smplh.num_joints - 1) * 9, CFG.smplh.num_joints, 9, 3],
                    dim=-1,
                )
            )
            body_rotmats = body_rotmats_flat.reshape(
                (*batch, time, (CFG.smplh.num_joints - 1), 3, 3)
            )
            hand_rotmats = None

        if project_rotmats:
            body_rotmats = project_rotmats_via_svd(body_rotmats)
            if hand_rotmats is not None:
                hand_rotmats = project_rotmats_via_svd(hand_rotmats)

        R_world_root = R_world_root.reshape(*batch, time, 3, 3)
        return cls(
            betas=betas,
            body_rotmats=body_rotmats,
            contacts=contacts,
            hand_rotmats=hand_rotmats,
            R_world_root=R_world_root,
            t_world_root=t_world_root,
        )

    def encode(self, encoders: nn.ModuleDict, batch: int, time: int) -> Float[Tensor, "batch time d_latent"]:
        """Encode absolute trajectory into latent space."""
        encoded = (
            encoders["betas"](self.betas.reshape((batch, time, -1)))
            + encoders["body_rotmats"](self.body_rotmats.reshape((batch, time, -1)))
            + encoders["contacts"](self.contacts)
            + encoders["R_world_root"](self.R_world_root.reshape((batch, time, -1)))
            + encoders["t_world_root"](self.t_world_root)
        )
        if self.hand_rotmats is not None:
            encoded = encoded + encoders["hand_rotmats"](self.hand_rotmats.reshape((batch, time, -1)))
        return encoded

@dataclasses.dataclass
class VelocityDenoiseTraj(BaseDenoiseTraj):
    """Denoising trajectory with velocity-based representation."""

    betas: Float[Tensor, "*batch timesteps 16"]
    """Body shape parameters. We don't really need the timesteps axis here,
    it's just for convenience."""

    body_rotmats: Float[Tensor, "*batch timesteps 21 3 3"]
    """Local orientations for each body joint."""

    contacts: Float[Tensor, "*batch timesteps 22"]
    """Contact boolean for each joint."""

    hand_rotmats: Float[Tensor, "*batch timesteps 30 3 3"] | None
    """Local orientations for each body joint."""

    R_world_root: Float[Tensor, "*batch timesteps 3 3"]
    """Global rotation matrix of the root joint."""

    t_world_root: Float[Tensor, "*batch timesteps 3"]
    """Global translation vector of the root joint."""

    R_world_root_tm1_t: Float[Tensor, "*batch timesteps 3 3"] | None = None
    """Relative rotation between consecutive frames (t-1 to t)."""

    t_world_root_tm1_t: Float[Tensor, "*batch timesteps 3"] | None = None
    """Relative translation between consecutive frames (t-1 to t)."""

    R_world_root_acc: Float[Tensor, "*batch timesteps 3 3"] | None = None
    """Acceleration of rotation between consecutive frames."""

    t_world_root_acc: Float[Tensor, "*batch timesteps 3"] | None = None
    """Acceleration of translation between consecutive frames."""

    def __post_init__(self) -> None:
        self._compute_temporal_offsets()

    @staticmethod
    def get_packed_dim(include_hands: bool) -> int:
        """Get dimension of packed state vector.

        Args:
            include_hands: Whether to include hand rotations in packed dimension.

        Returns:
            Total dimension of packed state vector.
        """
        # 16 (betas) + 21*9 (body_rotmats) + 21 (contacts) + 9 (R_world_root_tm1_t) + 3 (t_world_root_tm1_t)
        num_smplh_jnts = CFG.smplh.num_joints
        packed_dim = 16 + (num_smplh_jnts - 1) * 9 + num_smplh_jnts + 9 + 3
        if include_hands:
            packed_dim += 30 * 9  # hand_rotmats
        return packed_dim

    def _compute_temporal_offsets(self) -> None:
        """Compute relative rotations and translations between consecutive frames."""
        batch_shape = self.R_world_root.shape[:-2]
        device = self.R_world_root.device
        dtype = self.R_world_root.dtype

        # Compute relative rotations using SO3
        R_curr = SO3.from_matrix(self.R_world_root[:, 1:])
        R_prev = SO3.from_matrix(self.R_world_root[:, :-1])
        R_rel = R_curr.multiply(R_prev.inverse())
        self.R_world_root_tm1_t = torch.cat(
            [
                torch.eye(3, device=device, dtype=dtype).expand(
                    *batch_shape[:-1], 1, 3, 3
                ),
                R_rel.as_matrix(),
            ],
            dim=-3,
        )

        # Compute relative translations using SE3
        T_curr = SE3.from_rotation_and_translation(
            SO3.from_matrix(self.R_world_root), self.t_world_root
        )
        self.t_world_root_tm1_t = torch.zeros_like(self.t_world_root)
        self.t_world_root_tm1_t[:, 1:] = (
            T_curr.translation()[:, 1:] - T_curr.translation()[:, :-1]
        )

        # Compute accelerations
        # For rotations, multiply consecutive relative rotations
        R_rel_curr = SO3.from_matrix(self.R_world_root_tm1_t[:, 2:])
        R_rel_prev = SO3.from_matrix(self.R_world_root_tm1_t[:, 1:-1])
        self.R_world_root_acc = torch.zeros_like(self.R_world_root)
        self.R_world_root_acc[:, 2:] = R_rel_curr.multiply(
            R_rel_prev.inverse()
        ).as_matrix()

        # For translations, compute difference of consecutive relative translations
        self.t_world_root_acc = torch.zeros_like(self.t_world_root)
        self.t_world_root_acc[:, 2:] = (
            self.t_world_root_tm1_t[:, 2:] - self.t_world_root_tm1_t[:, 1:-1]
        )

    def apply_to_body(self, body_model: SmplhModel) -> SmplhShapedAndPosed:
        """Apply the trajectory data to a SMPL-H body model.

        This method first reconstructs absolute positions from temporal offsets,
        then applies them to the body model.
        """
        device = self.betas.device
        dtype = self.betas.dtype
        batch_shape = self.R_world_root_tm1_t.shape[:-2]
        time = self.R_world_root_tm1_t.shape[-3]

        # Initialize absolute positions with identity rotation and zero translation
        R_world_root = (
            torch.eye(3, device=device, dtype=dtype).expand(*batch_shape, 3, 3).clone()
        )
        t_world_root = torch.zeros((*batch_shape, 3), device=device, dtype=dtype)

        # Reconstruct absolute positions from temporal offsets
        for t in range(1, time):
            R_world_root[..., t, :, :] = (
                SO3.from_matrix(self.R_world_root_tm1_t[..., t, :, :])
                .multiply(SO3.from_matrix(R_world_root[..., t - 1, :, :]))
                .as_matrix()
            )
            t_world_root[..., t, :] = (
                t_world_root[..., t - 1, :] + self.t_world_root_tm1_t[..., t, :]
            )

        # Create SMPL-H model with shape parameters
        shaped = body_model.with_shape(self.betas)

        # Convert rotations and translations to SE3 parameters
        T_world_root = SE3.from_rotation_and_translation(
            SO3.from_matrix(R_world_root),
            t_world_root,
        ).parameters()

        # Apply pose to the model
        posed = shaped.with_pose_decomposed(
            T_world_root=T_world_root,
            body_quats=SO3.from_matrix(self.body_rotmats).wxyz,
            left_hand_quats=SO3.from_matrix(self.hand_rotmats[..., :15, :, :]).wxyz
            if self.hand_rotmats is not None
            else None,
            right_hand_quats=SO3.from_matrix(self.hand_rotmats[..., 15:30, :, :]).wxyz
            if self.hand_rotmats is not None
            else None,
        )

        return posed

    @staticmethod
    def get_packed_dim(include_hands: bool) -> int:
        """Get dimension of packed state vector.

        Args:
            include_hands: Whether to include hand rotations in packed dimension.

        Returns:
            Total dimension of packed state vector.
        """
        # 16 (betas) + 21*9 (body_rotmats) + 22 (contacts) + 9 (R_world_root_tm1_t) + 3 (t_world_root_tm1_t)
        num_smplh_jnts = CFG.smplh.num_joints
        packed_dim = 16 + (num_smplh_jnts - 1) * 9 + num_smplh_jnts + 9 + 3
        if include_hands:
            packed_dim += 30 * 9  # hand_rotmats
        return packed_dim

    @jaxtyped(typechecker=typeguard.typechecked)
    def pack(self) -> Float[Tensor, "*batch timesteps d_state"]:
        """Pack trajectory into a single flattened vector.
        Only packs the temporal offset attributes and other necessary components."""
        (*batch, time, num_joints, _, _) = self.body_rotmats.shape
        assert num_joints == 21

        # Create list of tensors to pack
        tensors_to_pack = [
            self.betas.reshape((*batch, time, -1)),
            self.body_rotmats.reshape((*batch, time, -1)),
            self.contacts.reshape((*batch, time, -1)),
            self.R_world_root_tm1_t.reshape(
                (*batch, time, -1)
            ),  # Pack temporal offsets instead
            self.t_world_root_tm1_t.reshape(
                (*batch, time, -1)
            ),  # Pack temporal offsets instead
        ]

        if self.hand_rotmats is not None:
            tensors_to_pack.append(self.hand_rotmats.reshape((*batch, time, -1)))

        return torch.cat(tensors_to_pack, dim=-1)

    @classmethod
    @jaxtyped(typechecker=typeguard.typechecked)
    def unpack(
        cls,
        x: Float[Tensor, "*batch timesteps d_state"],
        include_hands: bool = False,
        project_rotmats: bool = False,
    ) -> "VelocityDenoiseTraj":
        """Unpack trajectory from a single flattened vector.
        Reconstructs absolute positions from temporal offsets."""
        (*batch, time, d_state) = x.shape
        assert d_state == cls.get_packed_dim(include_hands)

        if include_hands:
            (
                betas,
                body_rotmats_flat,
                contacts,
                R_world_root_tm1_t_flat,
                t_world_root_tm1_t,
                hand_rotmats_flat,
            ) = torch.split(
                x,
                [
                    16,
                    (CFG.smplh.num_joints - 1) * 9,
                    CFG.smplh.num_joints,
                    9,
                    3,
                    30 * 9,
                ],
                dim=-1,
            )
            body_rotmats = body_rotmats_flat.reshape(
                (*batch, time, (CFG.smplh.num_joints - 1), 3, 3)
            )
            hand_rotmats = hand_rotmats_flat.reshape((*batch, time, 30, 3, 3))
        else:
            (
                betas,
                body_rotmats_flat,
                contacts,
                R_world_root_tm1_t_flat,
                t_world_root_tm1_t,
            ) = torch.split(
                x,
                [
                    16,
                    (CFG.smplh.num_joints - 1) * 9,
                    CFG.smplh.num_joints,
                    9,
                    3,
                ],
                dim=-1,
            )
            body_rotmats = body_rotmats_flat.reshape(
                (*batch, time, (CFG.smplh.num_joints - 1), 3, 3)
            )
            hand_rotmats = None

        # Reshape relative rotation matrices
        R_world_root_tm1_t = R_world_root_tm1_t_flat.reshape(*batch, time, 3, 3)

        if project_rotmats:
            body_rotmats = project_rotmats_via_svd(body_rotmats)
            R_world_root_tm1_t = project_rotmats_via_svd(R_world_root_tm1_t)
            if hand_rotmats is not None:
                hand_rotmats = project_rotmats_via_svd(hand_rotmats)

        # Initialize absolute positions with identity rotation and zero translation
        device = x.device
        dtype = x.dtype
        R_world_root = (
            torch.eye(3, device=device, dtype=dtype).expand(*batch, time, 3, 3).clone()
        )
        t_world_root = torch.zeros((*batch, time, 3), device=device, dtype=dtype)

        # Reconstruct absolute positions from temporal offsets
        for t in range(1, time):
            R_world_root[:, t] = (
                SO3.from_matrix(R_world_root_tm1_t[:, t])
                .multiply(SO3.from_matrix(R_world_root[:, t - 1]))
                .as_matrix()
            )
            t_world_root[:, t] = t_world_root[:, t - 1] + t_world_root_tm1_t[:, t]

        return cls(
            betas=betas,
            body_rotmats=body_rotmats,
            contacts=contacts,
            hand_rotmats=hand_rotmats,
            R_world_root=R_world_root,
            t_world_root=t_world_root,
            R_world_root_tm1_t=R_world_root_tm1_t,
            t_world_root_tm1_t=t_world_root_tm1_t,
        )

    def compute_loss(
        self,
        other: "VelocityDenoiseTraj",
        mask: Bool[Tensor, "batch time"],
        weight_t: Float[Tensor, "batch"],
    ) -> dict[str, Float[Tensor, ""]]:
        """Compute loss between this trajectory and another using velocity-based representations."""
        batch, time = mask.shape[:2]

        vel_mask = torch.zeros_like(mask, dtype=torch.bool)
        vel_mask[:, 1:] = mask[:, 1:] & mask[:, :-1]
        acc_mask = torch.zeros_like(mask, dtype=torch.bool)
        acc_mask[:, 2:] = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]

        loss_terms = {
            # Beta loss remains absolute since it's shape parameters
            "betas": self._weight_and_mask_loss(
                (self.betas - other.betas) ** 2, mask, weight_t
            ),
            # Body rotations loss (could be made relative if needed)
            "body_rotmats": self._weight_and_mask_loss(
                (self.body_rotmats - other.body_rotmats).reshape(batch, time, -1) ** 2,
                mask,
                weight_t,
            ),
            # Relative rotation loss
            "R_world_root_vel": self._weight_and_mask_loss(
                (self.R_world_root_tm1_t - other.R_world_root_tm1_t).reshape(
                    batch, time, -1
                )
                ** 2,
                vel_mask,
                weight_t,
            ),
            # Relative translation loss
            "t_world_root_vel": self._weight_and_mask_loss(
                (self.t_world_root_tm1_t - other.t_world_root_tm1_t) ** 2,
                vel_mask,
                weight_t,
            ),
            # Acceleration losses
            "R_world_root_acc": self._weight_and_mask_loss(
                (self.R_world_root_acc - other.R_world_root_acc).reshape(
                    batch, time, -1
                )
                ** 2,
                acc_mask,
                weight_t,
            ),
            "t_world_root_acc": self._weight_and_mask_loss(
                (self.t_world_root_acc - other.t_world_root_acc) ** 2,
                acc_mask,
                weight_t,
            ),
        }

        if self.hand_rotmats is not None and other.hand_rotmats is not None:
            loss_terms["hand_rotmats"] = self._weight_and_mask_loss(
                (self.hand_rotmats - other.hand_rotmats).reshape(batch, time, -1) ** 2,
                mask,
                weight_t,
            )

        return loss_terms

    def encode(self, encoders: nn.ModuleDict, batch: int, time: int) -> Float[Tensor, "batch time d_latent"]:
        """Encode trajectory into latent space."""
        encoded = (
            encoders["betas"](self.betas.reshape((batch, time, -1)))
            + encoders["body_rotmats"](self.body_rotmats.reshape((batch, time, -1)))
            + encoders["contacts"](self.contacts)
            + encoders["R_world_root_tm1_t"](self.R_world_root_tm1_t.reshape((batch, time, -1)))
            + encoders["t_world_root_tm1_t"](self.t_world_root_tm1_t)
        )
        if self.hand_rotmats is not None:
            encoded = encoded + encoders["hand_rotmats"](self.hand_rotmats.reshape((batch, time, -1)))
        return encoded


@jaxtyped(typechecker=typeguard.typechecked)
@dataclass
class EgoDenoiserConfig():
    # Basic parameters
    max_t: int = 1000
    fourier_enc_freqs: int = 3
    d_latent: int = 512
    d_feedforward: int = 2048
    d_noise_emb: int = 1024
    num_heads: int = 4
    encoder_layers: int = 6
    decoder_layers: int = 6
    dropout_p: float = 0.0

    # MAE parameters
    mask_ratio: float = 0.75  # Ratio of joints to mask during training
    include_hands: bool = False

    # Model settings
    activation: Literal["gelu", "relu"] = "gelu"
    positional_encoding: Literal["transformer", "rope"] = "transformer"
    noise_conditioning: Literal["token", "film"] = "token"
    xattn_mode: Literal["kv_from_cond_q_from_x", "kv_from_x_q_from_cond"] = (
        "kv_from_cond_q_from_x"
    )

    # Joint position conditioning settings
    joint_cond_mode: JointCondMode = "vel_acc"
    joint_emb_dim: int = 8

    # Add SMPL-H model path configuration
    smplh_npz_path: Path = Path("data/smplh/neutral/model.npz")

    # Add new config parameter
    use_fourier_in_masked_joints: bool = (
        True  # Whether to apply Fourier encoding in make_cond_with_masked_joints
    )

    # Add new config parameter
    use_joint_embeddings: bool = (
        True  # Whether to use joint index embeddings in conditioning
    )

    @cached_property
    def d_cond(self) -> int:
        """Dimensionality of conditioning vector."""
        num_joints = CFG.smplh.num_joints  # Assuming num_joints is 22
        spatial_dim = 4  # x, y, z, visibility

        # Only include joint embedding dimensions if enabled
        joint_emb_contribution = (
            (self.joint_emb_dim * num_joints) if self.use_joint_embeddings else 0
        )

        if self.joint_cond_mode == "vel_acc":
            # velocities (22*4) + accelerations (22*4) + index_embeddings (22*16 if enabled)
            d_cond = (spatial_dim * num_joints * 2) + joint_emb_contribution
        elif self.joint_cond_mode == "vel_acc_plus":
            # velocities (22*4) + accelerations (22*4) + index_embeddings (22*16 if enabled)
            d_cond = (spatial_dim * num_joints * 2) + joint_emb_contribution
        elif self.joint_cond_mode == "absolute":
            # joints_with_vis (22*4) + index_embeddings (22*16 if enabled)
            d_cond = (spatial_dim * num_joints) + joint_emb_contribution
        elif self.joint_cond_mode == "absrel_jnts":
            # first_joint (4) + local_coords (21*4) + index_embeddings (22*16 if enabled)
            d_cond = (
                spatial_dim + (spatial_dim * (num_joints - 1)) + joint_emb_contribution
            )
        elif self.joint_cond_mode == "absrel" or self.joint_cond_mode == "joints_only":
            # abs_pos (22*4) + rel_pos (22*4) + index_embeddings (22*16 if enabled)
            d_cond = (
                (spatial_dim * num_joints)
                + (spatial_dim * num_joints)
                + joint_emb_contribution
            )
        elif self.joint_cond_mode == "absrel_global_deltas":
            # joints_with_vis (22*spatial_dim) + index_embeddings (22*16 if enabled) + r_mat (9) + t (3)
            d_cond = (spatial_dim * num_joints) + joint_emb_contribution + 9 + 3
        else:
            assert_never(self.joint_cond_mode)

        # Apply Fourier encoding multiplier if enabled
        if self.use_fourier_in_masked_joints:
            d_cond = d_cond * (2 * self.fourier_enc_freqs + 1)

        return d_cond

    @jaxtyped(typechecker=typeguard.typechecked)
    def make_cond_with_masked_joints(
        self,
        joints: Float[Tensor, "batch time 22 3"],
        visible_joints_mask: Bool[Tensor, "batch time 22"],
    ) -> Float[Tensor, "batch time d_cond"]:
        batch, time = visible_joints_mask.shape[:2]
        device = joints.device
        dtype = joints.dtype

        # !joints must be masked to prevent further motion information from being used
        masked_joints = joints.clone()
        masked_joints[~visible_joints_mask] = 0

        # Create joint embeddings if enabled
        if self.use_joint_embeddings:
            joint_embeddings = nn.Embedding(
                CFG.smplh.num_joints, self.joint_emb_dim
            ).to(device)
            all_indices = torch.arange(CFG.smplh.num_joints, device=device)
            index_embeddings = joint_embeddings(all_indices).expand(batch, time, -1, -1)
        else:
            index_embeddings = None

        if self.joint_cond_mode == "vel_acc_plus":
            # TODO: this should be deprectaed now, as it lacks logical soundness now.
            # 1. Compute hierarchical joint representation
            # Root (pelvis) as base
            root_pos = masked_joints[..., 0:1, :]

            # Get relative positions to parent joints using kinematic chain
            # This removes redundant global motion information
            rel_to_parent = torch.zeros_like(masked_joints)
            kinematic_chain = get_kinematic_chain(
                use_smplh=True
            )  # Define this based on SMPL skeleton
            for joint_idx, parent_idx in kinematic_chain:
                rel_to_parent[..., joint_idx, :] = (
                    masked_joints[..., joint_idx, :] - masked_joints[..., parent_idx, :]
                )

            # Define the rel_to_parent mask
            rel_to_parent_visible_mask: Float[Tensor, "batch time 22"] = (
                torch.zeros_like(visible_joints_mask)
            )
            # Handle pelvis (root) joint separately
            rel_to_parent_visible_mask[..., 0] = visible_joints_mask[..., 0]
            # For all other joints, require both joint and parent to be visible
            for joint_idx, parent_idx in kinematic_chain:
                rel_to_parent_visible_mask[..., joint_idx] = (
                    visible_joints_mask[..., joint_idx]
                    & visible_joints_mask[..., parent_idx]
                )

            # 2. Compute motion features and masks using relative parent joint positions
            # Calculate velocity mask (both current and previous frame must have valid relative parent positions)
            vel_mask = torch.zeros_like(rel_to_parent_visible_mask)  # [batch, time, 22]
            vel_mask[:, 1:] = (
                rel_to_parent_visible_mask[:, 1:] & rel_to_parent_visible_mask[:, :-1]
            )

            # Calculate acceleration mask (current and two previous frames must have valid relative parent positions)
            acc_mask = torch.zeros_like(rel_to_parent_visible_mask)  # [batch, time, 22]
            acc_mask[:, 2:] = (
                rel_to_parent_visible_mask[:, 2:]
                & rel_to_parent_visible_mask[:, 1:-1]
                & rel_to_parent_visible_mask[:, :-2]
            )

            # Velocity (first order temporal difference of relative parent positions)
            velocity = torch.zeros_like(rel_to_parent)  # [batch, time, 22, 3]
            velocity[:, 1:] = rel_to_parent[:, 1:] - rel_to_parent[:, :-1]
            velocity = velocity * vel_mask.unsqueeze(-1)  # [batch, time, 22, 3]

            # Acceleration (second order temporal difference of relative parent positions)
            accel = torch.zeros_like(rel_to_parent)  # [batch, time, 22, 3]
            accel[:, 2:] = velocity[:, 2:] - velocity[:, 1:-1]
            accel = accel * acc_mask.unsqueeze(-1)  # [batch, time, 22, 3]

            # 3. Compute motion attention scores based on relative motion
            motion_magnitude = torch.norm(velocity, dim=-1)  # [batch, time-1, 22]
            attention_scores = torch.sigmoid(
                motion_magnitude - motion_magnitude.mean(dim=1, keepdim=True)
            )  # [batch, time-1, 22]

            # 4. Create decorrelated features
            components = [
                # Global root motion and its configuration
                root_pos.reshape(batch, time, -1),
                # Local joint configurations and visibility
                rel_to_parent.reshape(batch, time, -1),
                rel_to_parent_visible_mask.reshape(batch, time, -1),
                # Motion features with attention weights
                (velocity * attention_scores[..., None]).reshape(batch, time, -1),
                (accel * attention_scores[..., None]).reshape(batch, time, -1),
                # Motion feature masks
                vel_mask.reshape(batch, time, -1),
                acc_mask.reshape(batch, time, -1),
                # Optional semantic information
                index_embeddings.reshape(batch, time, -1)
                if self.use_joint_embeddings
                else None,
            ]
            components = [c for c in components if c is not None]

            # Pad temporal features to match time dimension
            pad_size = time - components[2].shape[1]
            components[2] = F.pad(components[2], (0, 0, 0, pad_size))
            pad_size = time - components[3].shape[1]
            components[3] = F.pad(components[3], (0, 0, 0, pad_size))

            cond = torch.cat(components, dim=-1)

        elif self.joint_cond_mode == "vel_acc":
            # Calculate velocity mask (both current and previous frame must be visible)
            vel_mask = torch.zeros_like(visible_joints_mask)
            vel_mask[:, 1:] = visible_joints_mask[:, 1:] & visible_joints_mask[:, :-1]

            # Calculate acceleration mask (current and two previous frames must be visible)
            acc_mask = torch.zeros_like(visible_joints_mask)
            acc_mask[:, 2:] = (
                visible_joints_mask[:, 2:]
                & visible_joints_mask[:, 1:-1]
                & visible_joints_mask[:, :-2]
            )

            # Calculate velocities and mask invisible parts
            velocities = torch.zeros_like(joints)
            velocities[:, 1:] = joints[:, 1:] - joints[:, :-1]
            velocities = velocities * vel_mask.unsqueeze(
                -1
            )  # Mask out invisible velocities

            # Calculate accelerations and mask invisible parts
            accelerations = torch.zeros_like(joints)
            accelerations[:, 2:] = velocities[:, 2:] - velocities[:, 1:-1]
            accelerations = accelerations * acc_mask.unsqueeze(
                -1
            )  # Mask out invisible accelerations

            # Combine with visibility masks
            velocities_with_vis = torch.cat(
                [velocities, vel_mask.unsqueeze(-1).to(dtype)], dim=-1
            )

            accelerations_with_vis = torch.cat(
                [accelerations, acc_mask.unsqueeze(-1).to(dtype)], dim=-1
            )

            # Combine all components
            components = [
                velocities_with_vis.reshape(batch, time, -1),
                accelerations_with_vis.reshape(batch, time, -1),
            ]
            if self.use_joint_embeddings:
                components.append(index_embeddings.reshape(batch, time, -1))

            cond = torch.cat(components, dim=-1)

        elif self.joint_cond_mode == "absolute":
            # joints_with_vis (22*4) + index_embeddings (22*16 if enabled)
            joints_with_vis = torch.cat(
                [masked_joints, visible_joints_mask.unsqueeze(-1).to(dtype)], dim=-1
            )
            components = [joints_with_vis.reshape(batch, time, -1)]
            if self.use_joint_embeddings:
                components.append(index_embeddings.reshape(batch, time, -1))
            cond = torch.cat(components, dim=-1)

        elif self.joint_cond_mode == "absrel_jnts":
            joints_with_vis = torch.cat(
                [masked_joints, visible_joints_mask.unsqueeze(-1).to(dtype)], dim=-1
            )
            first_joint = joints_with_vis[..., 0:1, :]
            other_joints = joints_with_vis[..., 1:, :]
            local_coords = other_joints - first_joint[..., :3].expand_as(
                other_joints[..., :3]
            )
            local_coords = torch.cat(
                [
                    local_coords[..., :3],
                    other_joints[..., 3:],
                ],
                dim=-1,
            )

            components = [
                first_joint.reshape(batch, time, -1),
                local_coords.reshape(batch, time, -1),
            ]
            if self.use_joint_embeddings:
                components.append(index_embeddings.reshape(batch, time, -1))
            cond = torch.cat(components, dim=-1)

        elif self.joint_cond_mode == "absrel" or self.joint_cond_mode == "joints_only":
            joints_with_vis = torch.cat(
                [masked_joints, visible_joints_mask.unsqueeze(-1).to(dtype)], dim=-1
            )
            abs_pos = joints_with_vis
            rel_pos = torch.zeros_like(joints_with_vis)
            rel_pos[:, 1:] = joints_with_vis[:, 1:] - joints_with_vis[:, :-1]

            components = [
                abs_pos.reshape(batch, time, -1),
                rel_pos.reshape(batch, time, -1),
            ]
            if self.use_joint_embeddings:
                components.append(index_embeddings.reshape(batch, time, -1))
            cond = torch.cat(components, dim=-1)

        elif self.joint_cond_mode == "absrel_global_deltas":
            joints_with_vis = torch.cat(
                [masked_joints, visible_joints_mask.unsqueeze(-1).to(dtype)], dim=-1
            )
            first_frame = joints_with_vis[:, 0]
            current_frames = joints_with_vis

            first_centroid = first_frame.mean(dim=1, keepdim=True)
            current_centroids = current_frames.mean(dim=2, keepdim=True)

            first_centered = first_frame - first_centroid
            current_centered = current_frames - current_centroids

            first_centered_exp = first_centered.unsqueeze(1).expand(-1, time, -1, -1)
            H = first_centered_exp.transpose(-2, -1) @ current_centered

            U, _, Vh = torch.linalg.svd(H)
            r_mat = Vh.transpose(-2, -1) @ U.transpose(-2, -1)

            det = torch.linalg.det(r_mat)
            reflection_fix = (
                torch.eye(3, device=r_mat.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(batch, time, -1, -1)
                .clone()
            )
            reflection_fix[..., 2, 2] = det
            r_mat = r_mat @ reflection_fix

            t = current_centroids.squeeze(2) - torch.einsum(
                "btij,bkj->bti", r_mat, first_centroid
            )

            components = [
                torch.cat([joints_with_vis, index_embeddings], dim=-1).reshape(
                    batch, time, -1
                ),
                r_mat.reshape(batch, time, 9),
                t.reshape(batch, time, 3),
            ]
            if self.use_joint_embeddings:
                components.append(index_embeddings.reshape(batch, time, -1))
            cond = torch.cat(components, dim=-1)

        else:
            assert_never(self.joint_cond_mode)

        # Apply Fourier encoding if enabled
        if self.use_fourier_in_masked_joints:
            cond = fourier_encode(cond, freqs=self.fourier_enc_freqs)

        assert cond.shape == (batch, time, self.d_cond)
        return cond


class EgoDenoiser(nn.Module):
    """Denoising network for human motion.

    Inputs are noisy trajectory, conditioning information, and timestep.
    Output is denoised trajectory.
    """

    def __init__(self, config: EgoDenoiserConfig, modality_dims: dict[str, int]):
        super().__init__()

        self.config = config
        self.body_model = SmplhModel.load(config.smplh_npz_path)

        Activation = {"gelu": nn.GELU, "relu": nn.ReLU}[config.activation]

        self.encoders = nn.ModuleDict(
            {
                k: nn.Sequential(
                    nn.Linear(modality_dim, config.d_latent),
                    Activation(),
                    nn.Linear(config.d_latent, config.d_latent),
                    Activation(),
                    nn.Linear(config.d_latent, config.d_latent),
                )
                for k, modality_dim in modality_dims.items()
            }
        )
        self.decoders = nn.ModuleDict(
            {
                k: nn.Sequential(
                    nn.Linear(config.d_latent, config.d_latent),
                    nn.LayerNorm(normalized_shape=config.d_latent),
                    Activation(),
                    nn.Linear(config.d_latent, config.d_latent),
                    Activation(),
                    nn.Linear(config.d_latent, modality_dim),
                )
                for k, modality_dim in modality_dims.items()
            }
        )

        # Helpers for converting between input dimensionality and latent dimensionality.
        self.latent_from_cond = nn.Linear(config.d_cond, config.d_latent)

        # Noise embedder.
        self.noise_emb = nn.Embedding(
            # index 0 will be t=1
            # index 999 will be t=1000
            num_embeddings=config.max_t,
            embedding_dim=config.d_noise_emb,
        )
        self.noise_emb_token_proj = (
            nn.Linear(config.d_noise_emb, config.d_latent, bias=False)
            if config.noise_conditioning == "token"
            else None
        )

        # Encoder / decoder layers.
        # Inputs are conditioning (current noise level, observations); output
        # is encoded conditioning information.
        self.encoder_layers = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        d_latent=config.d_latent,
                        d_noise_emb=config.d_noise_emb,
                        d_feedforward=config.d_feedforward,
                        n_heads=config.num_heads,
                        dropout_p=config.dropout_p,
                        activation=config.activation,
                        include_xattn=False,  # No conditioning for encoder.
                        use_rope_embedding=config.positional_encoding == "rope",
                        use_film_noise_conditioning=config.noise_conditioning == "film",
                        xattn_mode=config.xattn_mode,
                    )
                )
                for _ in range(config.encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        d_latent=config.d_latent,
                        d_noise_emb=config.d_noise_emb,
                        d_feedforward=config.d_feedforward,
                        n_heads=config.num_heads,
                        dropout_p=config.dropout_p,
                        activation=config.activation,
                        include_xattn=True,  # Include conditioning for the decoder.
                        use_rope_embedding=config.positional_encoding == "rope",
                        use_film_noise_conditioning=config.noise_conditioning == "film",
                        xattn_mode=config.xattn_mode,
                    )
                )
                for _ in range(config.decoder_layers)
            ]
        )

    @jaxtyped(typechecker=typeguard.typechecked)
    def forward(
        self,
        x_t_unpacked: Union[VelocityDenoiseTraj, AbsoluteDenoiseTraj, JointsOnlyTraj],
        t: Int[Tensor, "batch"],
        project_output_rotmats: bool,
        joints: Float[Tensor, "batch time num_joints 3"],
        visible_joints_mask: Bool[Tensor, "batch time 22"],
        mask: Bool[Tensor, "batch time"] | None,
        cond_dropout_keep_mask: Bool[Tensor, "batch"] | None = None,
    ) -> Float[Tensor, "batch time state_dim"]:
        """Forward pass with MAE-style masking."""
        config = self.config

        (batch, time, num_body_joints, _) = joints.shape
        assert num_body_joints == 22

        # Encode the trajectory into a single vector per timestep.
        x_t_encoded = x_t_unpacked.encode(self.encoders, batch, time)

        # Embed the diffusion noise level.
        assert t.shape == (batch,)
        noise_emb = self.noise_emb(t - 1)
        assert noise_emb.shape == (batch, config.d_noise_emb)

        cond = config.make_cond_with_masked_joints(
            joints=joints,
            visible_joints_mask=visible_joints_mask,
        )
        # Randomly drop out conditioning information; this serves as a
        # regularizer that aims to improve sample diversity.
        if cond_dropout_keep_mask is not None:
            assert cond_dropout_keep_mask.shape == (batch,)
            cond = cond * cond_dropout_keep_mask[:, None, None]

        # Prepare encoder and decoder inputs.
        if config.positional_encoding == "rope":
            pos_enc = 0
        elif config.positional_encoding == "transformer":
            pos_enc = make_positional_encoding(
                d_latent=config.d_latent,
                length=time,
                dtype=cond.dtype,
            )[None, ...].to(x_t_encoded.device)
            assert pos_enc.shape == (1, time, config.d_latent)
        else:
            assert_never(config.positional_encoding)

        encoder_out = self.latent_from_cond(cond) + pos_enc
        decoder_out = x_t_encoded + pos_enc

        # Append the noise embedding to the encoder and decoder inputs.
        # This is weird if we're using rotary embeddings!
        if self.noise_emb_token_proj is not None:
            noise_emb_token = self.noise_emb_token_proj(noise_emb)
            assert noise_emb_token.shape == (batch, config.d_latent)
            encoder_out = torch.cat([noise_emb_token[:, None, :], encoder_out], dim=1)
            decoder_out = torch.cat([noise_emb_token[:, None, :], decoder_out], dim=1)
            assert (
                encoder_out.shape
                == decoder_out.shape
                == (batch, time + 1, config.d_latent)
            )
            num_tokens = time + 1
        else:
            num_tokens = time

        # Compute attention mask. This needs to be a fl
        if mask is None:
            attn_mask = None
        else:
            assert mask.shape == (batch, time)
            assert mask.dtype == torch.bool
            if self.noise_emb_token_proj is not None:  # Account for noise token.
                mask = torch.cat([mask.new_ones((batch, 1)), mask], dim=1)
            # Last two dimensions of mask are (query, key). We're masking out only keys;
            # it's annoying for the softmax to mask out entire rows without getting NaNs.
            attn_mask = mask[:, None, None, :].repeat(1, 1, num_tokens, 1)
            assert attn_mask.shape == (batch, 1, num_tokens, num_tokens)
            assert attn_mask.dtype == torch.bool

        # Forward pass through transformer.
        for layer in self.encoder_layers:
            encoder_out = layer(encoder_out, attn_mask, noise_emb=noise_emb)
        for layer in self.decoder_layers:
            decoder_out = layer(
                decoder_out, attn_mask, noise_emb=noise_emb, cond=encoder_out
            )

        # Remove the extra token corresponding to the noise embedding.
        if self.noise_emb_token_proj is not None:
            decoder_out = decoder_out[:, 1:, :]
        assert isinstance(decoder_out, Tensor)
        assert decoder_out.shape == (batch, time, config.d_latent)

        packed_output = torch.cat(
            [
                # Project rotation matrices for body_rotmats via SVD,
                (
                    project_rotmats_via_svd(
                        modality_decoder(decoder_out).reshape((-1, 3, 3))
                    ).reshape(
                        (batch, time, {"body_rotmats": 21, "hand_rotmats": 30}[key] * 9)
                    )
                    # if enabled,
                    if project_output_rotmats
                    and key in ("body_rotmats", "hand_rotmats")
                    # otherwise, just decode normally.
                    else modality_decoder(decoder_out)
                )
                for key, modality_decoder in self.decoders.items()
            ],
            dim=-1,
        )
        
        # Return packed output.
        return packed_output


@jaxtyped(typechecker=typeguard.typechecked)
@cache
def make_positional_encoding(
    d_latent: int, length: int, dtype: torch.dtype
) -> Float[Tensor, "length d_latent"]:
    """Computes standard Transformer positional encoding."""
    pe = torch.zeros(length, d_latent, dtype=dtype)
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_latent, 2).float() * (-np.log(10000.0) / d_latent)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    assert pe.shape == (length, d_latent)
    return pe


@jaxtyped(typechecker=typeguard.typechecked)
def fourier_encode(
    x: Float[Tensor, "*batch channels"], freqs: int
) -> Float[Tensor, "*batch channels_plus_2_mul_freqs_mul_channels"]:
    """Apply Fourier encoding to a tensor."""
    *batch_axes, x_dim = x.shape
    coeffs = 2.0 ** torch.arange(freqs, device=x.device)  # shape: (freqs,)
    scaled = (x[..., None] * coeffs).reshape(
        (*batch_axes, x_dim * freqs)
    )  # shape: (*batch_axes, x_dim * freqs)
    return torch.cat(
        [
            x,
            torch.sin(torch.cat([scaled, scaled + torch.pi / 2.0], dim=-1)),
        ],
        dim=-1,
    )  # shape: (*batch_axes, x_dim + 2 * freqs * x_dim)


@dataclass(frozen=True)
class TransformerBlockConfig:
    d_latent: int
    d_noise_emb: int
    d_feedforward: int
    n_heads: int
    dropout_p: float
    activation: Literal["gelu", "relu"]
    include_xattn: bool
    use_rope_embedding: bool
    use_film_noise_conditioning: bool
    xattn_mode: Literal["kv_from_cond_q_from_x", "kv_from_x_q_from_cond"]


class TransformerBlock(nn.Module):
    """An even-tempered Transformer block."""

    def __init__(self, config: TransformerBlockConfig) -> None:
        super().__init__()
        self.sattn_qkv_proj = nn.Linear(
            config.d_latent, config.d_latent * 3, bias=False
        )
        self.sattn_out_proj = nn.Linear(config.d_latent, config.d_latent, bias=False)

        self.layernorm1 = nn.LayerNorm(config.d_latent)
        self.layernorm2 = nn.LayerNorm(config.d_latent)

        assert config.d_latent % config.n_heads == 0
        self.rotary_emb = (
            RotaryEmbedding(
                config.d_latent // config.n_heads,
                learned_freq=False,
            )
            if config.use_rope_embedding
            else None
        )

        if config.include_xattn:
            self.xattn_kv_proj = nn.Linear(
                config.d_latent, config.d_latent * 2, bias=False
            )
            self.xattn_q_proj = nn.Linear(config.d_latent, config.d_latent, bias=False)
            self.xattn_layernorm = nn.LayerNorm(config.d_latent)
            self.xattn_out_proj = nn.Linear(
                config.d_latent, config.d_latent, bias=False
            )

        self.norm_no_learnable = nn.LayerNorm(
            config.d_feedforward, elementwise_affine=False, bias=False
        )
        self.activation = {"gelu": nn.GELU, "relu": nn.ReLU}[config.activation]()
        self.dropout = nn.Dropout(config.dropout_p)

        self.mlp0 = nn.Linear(config.d_latent, config.d_feedforward)
        self.mlp_film_cond_proj = (
            zero_module(
                nn.Linear(config.d_noise_emb, config.d_feedforward * 2, bias=False)
            )
            if config.use_film_noise_conditioning
            else None
        )
        self.mlp1 = nn.Linear(config.d_feedforward, config.d_latent)
        self.config = config

    @jaxtyped(typechecker=typeguard.typechecked)
    def forward(
        self,
        x: Float[Tensor, "batch tokens d_latent"],
        attn_mask: Bool[Tensor, "batch 1 tokens tokens"] | None,
        noise_emb: Float[Tensor, "batch d_noise_emb"],
        cond: Float[Tensor, "batch tokens d_latent"] | None = None,
    ) -> Float[Tensor, "batch tokens d_latent"]:
        config = self.config
        (batch, time, d_latent) = x.shape

        # Self-attention.
        # We put layer normalization after the residual connection.
        x = self.layernorm1(x + self._sattn(x, attn_mask))

        # Include conditioning.
        if config.include_xattn:
            assert cond is not None
            x = self.xattn_layernorm(x + self._xattn(x, attn_mask, cond=cond))

        mlp_out = x
        mlp_out = self.mlp0(mlp_out)
        mlp_out = self.activation(mlp_out)

        # FiLM-style conditioning.
        if self.mlp_film_cond_proj is not None:
            scale, shift = torch.chunk(
                self.mlp_film_cond_proj(noise_emb), chunks=2, dim=-1
            )
            assert scale.shape == shift.shape == (batch, config.d_feedforward)
            mlp_out = (
                self.norm_no_learnable(mlp_out) * (1.0 + scale[:, None, :])
                + shift[:, None, :]
            )

        mlp_out = self.dropout(mlp_out)
        mlp_out = self.mlp1(mlp_out)

        x = self.layernorm2(x + mlp_out)
        assert x.shape == (batch, time, d_latent)
        return x

    def _sattn(self, x: Tensor, attn_mask: Tensor | None) -> Tensor:
        """Multi-head self-attention."""
        config = self.config
        q, k, v = rearrange(
            self.sattn_qkv_proj(x),
            "b t (qkv nh dh) -> qkv b nh t dh",
            qkv=3,
            nh=config.n_heads,
        )
        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q, seq_dim=-2)
            k = self.rotary_emb.rotate_queries_or_keys(k, seq_dim=-2)
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=config.dropout_p, attn_mask=attn_mask
        )
        x = self.dropout(x)
        x = rearrange(x, "b nh t dh -> b t (nh dh)", nh=config.n_heads)
        x = self.sattn_out_proj(x)
        return x

    def _xattn(self, x: Tensor, attn_mask: Tensor | None, cond: Tensor) -> Tensor:
        """Multi-head cross-attention."""
        config = self.config
        k, v = rearrange(
            self.xattn_kv_proj(
                {
                    "kv_from_cond_q_from_x": cond,
                    "kv_from_x_q_from_cond": x,
                }[self.config.xattn_mode]
            ),
            "b t (qk nh dh) -> qk b nh t dh",
            qk=2,
            nh=config.n_heads,
        )
        q = rearrange(
            self.xattn_q_proj(
                {
                    "kv_from_cond_q_from_x": x,
                    "kv_from_x_q_from_cond": cond,
                }[self.config.xattn_mode]
            ),
            "b t (nh dh) -> b nh t dh",
            nh=config.n_heads,
        )
        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q, seq_dim=-2)
            k = self.rotary_emb.rotate_queries_or_keys(k, seq_dim=-2)
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=config.dropout_p, attn_mask=attn_mask
        )
        x = rearrange(x, "b nh t dh -> b t (nh dh)")
        x = self.xattn_out_proj(x)

        return x


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module

