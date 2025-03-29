"""DenoiseTraj definitions for AADecomposition."""

import dataclasses
from typing import Dict
from typing import Optional
from typing import TYPE_CHECKING

import torch
from egoallo.config import CONFIG_FILE
from egoallo.config import make_cfg
from jaxtyping import Bool
from jaxtyping import Float
from torch import nn
from torch import Tensor
import typeguard
from jaxtyping import jaxtyped
from egoallo.constants import SmplFamilyMetaModelZoo, SmplFamilyMetaModelName

# Move type imports inside TYPE_CHECKING block to avoid circular imports
if TYPE_CHECKING:
    from egoallo.type_stubs import SmplFamilyModelType

from egoallo.transforms import SE3, SO3
from egoallo.utils.setup_logger import setup_logger
from .base_traj import BaseDenoiseTraj

local_config_file = CONFIG_FILE
CFG = make_cfg(config_name="defaults", config_file=local_config_file, cli_args=[])

logger = setup_logger(output=None, name=__name__)


@dataclasses.dataclass
@jaxtyped(typechecker=typeguard.typechecked)
class AbsoluteDenoiseTrajAADecomp(BaseDenoiseTraj):
    from egoallo.data.dataclass_aadecomp import EgoTrainingDataAADecomp

    """Denoising trajectory with absolute pose representation."""

    betas: Float[Tensor, "*batch timesteps 10"]
    """Body shape parameters. We don't really need the timesteps axis here,
    it's just for convenience."""

    cos_sin_phis: Float[Tensor, "*batch timesteps 23 2"]
    """Local orientations for each body joint. network predicts cos, sin phis"""

    contacts: Float[Tensor, "*batch timesteps 24"]
    """Contact boolean for each joint."""

    hand_rotmats: Float[Tensor, "*batch timesteps 30 3 3"] | None
    """Local orientations for each body joint."""

    joints_wrt_world: Float[Tensor, "*batch timesteps 24 3"] | None
    """Joint positions in world frame."""

    visible_joints_mask: Float[Tensor, "*batch timesteps 24"] | None
    """Mask for visible joints."""

    metadata: EgoTrainingDataAADecomp.MetaData = dataclasses.field(
        default_factory=EgoTrainingDataAADecomp.MetaData,
    )
    """Metadata for the trajectory."""

    @property
    def t_world_root(self) -> Float[Tensor, "*batch timesteps 3"]:
        return self.joints_wrt_world[..., 0, :]

    @property
    def R_world_root(self) -> Float[Tensor, "*batch timesteps 3 3"]:
        smpl = SmplFamilyMetaModelZoo[SmplFamilyMetaModelName].load(
            self.metadata.smpl_family_model_basedir,
        )
        t_world_root = self.joints_wrt_world[..., 0, :]
        output = smpl.model.hybrik(
            betas=self.betas,
            phis=self.cos_sin_phis,
            pose_skeleton=self.joints_wrt_world,
            transl=t_world_root,
        )
        full_pose = output.rot_mats
        return full_pose[..., 0, :, :]

    def compute_loss(
        self,
        other: "AbsoluteDenoiseTrajAADecomp",
        mask: Bool[Tensor, "batch time"],
        weight_t: Float[Tensor, "batch"],
    ) -> dict[str, Float[Tensor, ""]]:
        """Compute loss between this trajectory and another using absolute representations."""
        batch, time = mask.shape[:2]

        loss_terms = {
            "betas": self._weight_and_mask_loss(
                (self.betas - other.betas) ** 2,
                mask,
                weight_t,
            ),
            "body_twists": self._weight_and_mask_loss(
                (self.cos_sin_phis - other.cos_sin_phis).reshape(batch, time, -1) ** 2,
                mask,
                weight_t,
            ),
            "joints_wrt_world": self._weight_and_mask_loss(
                (self.joints_wrt_world - other.joints_wrt_world).reshape(
                    batch,
                    time,
                    -1,
                )
                ** 2,
                mask,
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

    @staticmethod
    def get_packed_dim(include_hands: bool) -> int:
        """Get dimension of packed state vector.

        Args:
            include_hands: Whether to include hand rotations in packed dimension.

        Returns:
            Total dimension of packed state vector.
        """
        # 10 (betas) + 23*2 (body_twists) + 24 (contacts) + 24*3 (joints_wrt_world) + 3 (t_world_root)
        packed_dim = 10 + 23 * 2 + 24 + 24 * 3
        if include_hands:
            packed_dim += 30 * 9  # hand_rotmats
        return packed_dim

    def apply_to_body(self, body_model: "SmplFamilyModelType") -> "SmplFamilyModelType":
        """Apply the trajectory data to a SMPL-H body model."""
        # assert self.hand_rotmats is not None
        shaped = body_model.with_shape(
            betas=self.betas,
        )

        posed = shaped.with_pose_decomposed_twist_angles(
            transl=self.t_world_root,
            phis=self.cos_sin_phis,
            pose_skeleton=self.joints_wrt_world,
            global_orient=None,
        )

        return posed

    @jaxtyped(typechecker=typeguard.typechecked)
    def pack(self) -> Float[Tensor, "*batch timesteps d_state"]:
        """Pack trajectory into a single flattened vector."""
        (*batch, time, num_joints, _, _) = self.joints_wrt_world.shape
        assert num_joints == 24

        # Create list of tensors to pack
        tensors_to_pack = [
            self.betas.reshape((*batch, time, -1)),
            self.cos_sin_phis.reshape((*batch, time, -1)),
            self.contacts.reshape((*batch, time, -1)),
            self.t_world_root.reshape((*batch, time, -1)),
        ]

        if self.hand_rotmats is not None:
            tensors_to_pack.append(self.hand_rotmats.reshape((*batch, time, -1)))

        return torch.cat(tensors_to_pack, dim=-1)

    @classmethod
    def get_modality_dict(cls, include_hands: bool = False) -> dict[str, int]:
        """Get dictionary of modalities and their dimensions.

        For AbsoluteDenoiseTrajAADecomp, includes shape parameters, angle-axis rotations,
        contacts, and global joint positions.

        Args:
            include_hands: Whether to include hand rotations in the dictionary

        Returns:
            Dictionary mapping modality names to their dimensions
        """
        # Base modalities for AA decomposition mode
        modality_dims = {
            "betas": 10,  # Shape parameters (10 for SMPL vs 16 for SMPL-H)
            "cos_sin_phis": 23 * 2,  # Cosine and sine of twist angles for 23 joints
            "contacts": 24,  # Contact boolean for 24 joints
            "joints_wrt_world": 24 * 3,  # 3D coordinates for 24 joints
        }

        # Add hand rotations if specified
        if include_hands:
            modality_dims["hand_rotmats"] = 30 * 9

        return modality_dims

    @classmethod
    @jaxtyped(typechecker=typeguard.typechecked)
    def unpack(
        cls,
        x: Float[Tensor, "*batch timesteps d_state"],
        include_hands: bool = False,
        project_rotmats: bool = False,
    ) -> "AbsoluteDenoiseTrajAADecomp":
        """Unpack trajectory from a single flattened vector."""
        (*batch, time, d_state) = x.shape
        assert d_state == cls.get_packed_dim(include_hands)

        if include_hands:
            (
                betas,
                cos_sin_phis,
                contacts,
                hand_rotmats_flat,
            ) = torch.split(
                x,
                [
                    16,
                    23 * 2,
                    24,
                    30 * 9,
                ],
                dim=-1,
            )
            hand_rotmats = hand_rotmats_flat.reshape((*batch, time, 30, 3, 3))
        else:
            (
                betas,
                cos_sin_phis,
                contacts,
            ) = torch.split(
                x,
                [16, 23 * 2, 24],
                dim=-1,
            )
            hand_rotmats = None

        return cls(
            betas=betas,
            cos_sin_phis=cos_sin_phis,
            contacts=contacts,
            hand_rotmats=hand_rotmats,
            joints_wrt_world=None,  # Set to None since we don't have joints data when unpacking
            visible_joints_mask=None,  # Set to None since we don't have visibility data when unpacking
        )

    def encode(
        self,
        encoders: nn.ModuleDict,
        batch: int,
        time: int,
    ) -> Float[Tensor, "batch time d_latent"]:
        """Encode absolute trajectory into latent space."""
        encoded = (
            encoders["betas"](self.betas.reshape((batch, time, -1)))
            + encoders["cos_sin_phis"](self.cos_sin_phis.reshape((batch, time, -1)))
            + encoders["contacts"](self.contacts)
        )
        if self.hand_rotmats is not None:
            encoded = encoded + encoders["hand_rotmats"](
                self.hand_rotmats.reshape((batch, time, -1)),
            )
        return encoded

    def _compute_metrics(
        self,
        other: "AbsoluteDenoiseTrajAADecomp",
        body_model: Optional["SmplFamilyModelType"] = None,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, float]:
        """Compute metrics between this trajectory and another.
        Computes all relevant metrics since this class has complete pose data.
        """

        other = other.to(device)
        self = self.to(device)  # noqa
        body_model = body_model.to(device)

        self_has_nan = (
            self.reduce(
                lambda x, y: x.isnan().sum().item()
                if isinstance(x, torch.Tensor)
                else x + y.isnan().sum().item()
                if isinstance(y, torch.Tensor)
                else y,
            )
            > 0
        )
        other_has_nan = (
            other.reduce(
                lambda x, y: x.isnan().sum().item()
                if isinstance(x, torch.Tensor)
                else x + y.isnan().sum().item()
                if isinstance(y, torch.Tensor)
                else y,
            )
            > 0
        )
        if self_has_nan or other_has_nan:
            logger.warning(
                f"NaN values found in trajectory: {self_has_nan}, {other_has_nan}, skipping metrics computation",
            )
            # with warnings.catch_warnings():
            # warnings.filterwarnings("ignore", message=".*NaN values found in trajectory.*", category=RuntimeWarning)
            # raise RuntimeWarning(f"NaN values found in trajectory: {self_has_nan}, {other_has_nan}, skipping metrics computation")
            return {}

        # TEMPORARY_FIX: import BodyEvaluator lazily to avoid circular imports
        from egoallo.evaluation.body_evaluator import BodyEvaluator

        assert self.check_shapes(other), (
            f"self's shpae: {self.check_shapes(other)}, other's shape: {other.check_shapes(self)}"
        )

        metrics = {}

        assert body_model is not None
        gt_shaped = body_model.with_shape(other.betas)

        gt_posed = gt_shaped.with_pose_decomposed_twist_angles(
            transl=other.t_world_root,
            phis=other.cos_sin_phis,
            pose_skeleton=other.joints_wrt_world,
            global_orient=None,
        )
        gt_pose_skeletons = gt_posed.pose_skeleton
        gt_Ts_world_joint = SE3.from_rotation_and_translation(
            rotation=SO3.from_matrix(gt_posed.rot_mats),
            translation=gt_posed.pose_skeleton,
        ).parameters()

        pred_shaped = body_model.with_shape(self.betas)
        pred_posed = pred_shaped.with_pose_decomposed_twist_angles(
            transl=self.t_world_root,
            phis=self.cos_sin_phis,
            pose_skeleton=self.joints_wrt_world,
            global_orient=None,
        )
        pred_pose_skeletons = pred_posed.pose_skeleton
        pred_Ts_world_joint = SE3.from_rotation_and_translation(
            rotation=SO3.from_matrix(pred_posed.rot_mats),
            translation=pred_posed.pose_skeleton,
        ).parameters()

        num_samples, num_timesteps = self.betas.shape[:-1]
        # Body shape error
        metrics["betas_error"] = float(
            BodyEvaluator.compute_masked_error(
                gt=other.betas.reshape(*other.betas.shape[:-1], -1),  # N, T, 16
                pred=self.betas.reshape(*self.betas.shape[:-1], -1),  # N, T, 16
                device=device,
            ),
        )

        # Foot metrics
        metrics["foot_skate"] = float(
            BodyEvaluator.compute_foot_skate(
                pred_Ts_world_joint=pred_Ts_world_joint[..., 1:24, :],
                device=device,
            ).mean(),
        )

        metrics["foot_contact"] = float(
            BodyEvaluator.compute_foot_contact(
                pred_Ts_world_joint=pred_Ts_world_joint[..., 1:24, :],
                device=device,
            ).mean(),
        )

        metrics["mpjpe"] = float(
            BodyEvaluator.compute_mpjpe(
                label_root_pos=gt_pose_skeletons[..., 0, :],  # [batch, T, 3]
                label_joint_pos=gt_pose_skeletons[
                    ...,
                    1:24,
                    :3,
                ],  # [batch, T, 23, 3]
                pred_root_pos=pred_pose_skeletons[..., 0, :],  # [batch, T, 3]
                pred_joint_pos=pred_pose_skeletons[
                    ...,
                    1:24,
                    :3,
                ],  # [batch, T, 23, 3]
                per_frame_procrustes_align=False,
                device=device,
            ).mean(),
        )

        metrics["pampjpe"] = float(
            BodyEvaluator.compute_mpjpe(
                label_root_pos=gt_pose_skeletons[..., 0, :],  # [batch, T, 3]
                label_joint_pos=gt_pose_skeletons[
                    ...,
                    1:24,
                    :3,
                ],  # [batch, T, 23, 3]
                pred_root_pos=pred_pose_skeletons[..., 0, :],  # [batch, T, 3]
                pred_joint_pos=pred_pose_skeletons[
                    ...,
                    1:24,
                    :3,
                ],  # [batch, T, 21, 3]
                per_frame_procrustes_align=True,
                device=device,
            ).mean(),
        )

        metrics["head_ori"] = float(
            BodyEvaluator.compute_head_ori(
                label_Ts_world_joint=gt_Ts_world_joint[
                    ...,
                    1:24,
                    :,
                ],  # [batch, T, 21, 7]
                pred_Ts_world_joint=pred_Ts_world_joint[
                    ...,
                    1:24,
                    :,
                ],  # [batch, T, 21, 7]
                device=device,
            ).mean(),
        )

        metrics["head_trans"] = float(
            BodyEvaluator.compute_head_trans(
                label_Ts_world_joint=gt_Ts_world_joint[
                    ...,
                    1:24,
                    :,
                ],  # [batch, T, 21, 7]
                pred_Ts_world_joint=pred_Ts_world_joint[
                    ...,
                    1:24,
                    :,
                ],  # [batch, T, 21, 7]
                device=device,
            ).mean(),
        )

        metrics["foot_skate"] = float(
            BodyEvaluator.compute_foot_skate(
                pred_Ts_world_joint=pred_Ts_world_joint[
                    ...,
                    1:24,
                    :,
                ],  # [batch, T, 21, 7]
                device=device,
            ).mean(),
        )

        metrics["foot_contact"] = float(
            BodyEvaluator.compute_foot_contact(
                pred_Ts_world_joint=pred_Ts_world_joint[
                    ...,
                    1:24,
                    :,
                ],  # [batch, T, 21, 7]
                device=device,
            ).mean(),
        )

        del other
        del body_model

        return metrics
