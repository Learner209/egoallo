"""DenoiseTraj definitions for AADecomposition."""

import dataclasses
from typing import Dict
from typing import Optional
from typing import TYPE_CHECKING
from typing import Self

import torch
from egoallo.config import CONFIG_FILE
from egoallo.config import make_cfg
from jaxtyping import Bool
from jaxtyping import Float
from torch import nn
from torch import Tensor
import typeguard
from jaxtyping import jaxtyped
from egoallo.constants import SmplFamilyMetaModelZoo
from egoallo.type_stubs import EgoTrainingDataType
from egoallo.tensor_dataclass_batch_plugins import TensorDataclassBatchPlugin

# Move type imports inside TYPE_CHECKING block to avoid circular imports
if TYPE_CHECKING:
    from egoallo.type_stubs import SmplFamilyModelType

from egoallo.transforms import SE3, SO3
from egoallo.utils.setup_logger import setup_logger
from .base_traj import BaseDenoiseTraj
from egoallo.mapping import SMPL_PARENTS as smpl_parent_indices

local_config_file = CONFIG_FILE
CFG = make_cfg(config_name="defaults", config_file=local_config_file, cli_args=[])

logger = setup_logger(output=None, name=__name__)


@dataclasses.dataclass
@jaxtyped(typechecker=typeguard.typechecked)
class AbsoluteDenoiseTrajAADecomp(BaseDenoiseTraj):
    from egoallo.data.dataclass_aadecomp import EgoTrainingDataAADecomp

    """Denoising trajectory with absolute pose representation."""

    betas: Float[Tensor, "*batch timesteps {self.num_betas}"]
    """Body shape parameters. We don't really need the timesteps axis here,
    it's just for convenience."""

    cos_sin_phis: Float[Tensor, "*batch timesteps {self.num_joints}-1 2"]
    """Local orientations for each body joint. network predicts cos, sin phis"""

    contacts: Float[Tensor, "*batch timesteps {self.num_joints}"]
    """Contact boolean for each joint."""

    hand_rotmats: Float[Tensor, "*batch timesteps 30 3 3"] | None
    """Local orientations for each body joint."""

    joints_wrt_world: Float[Tensor, "*batch timesteps {self.num_joints} 3"]
    """Joint positions in world frame."""

    visible_joints_mask: Bool[Tensor, "*batch timesteps {self.num_joints}"]
    """Mask for visible joints."""

    """Metadata for the trajectory."""
    metadata: EgoTrainingDataAADecomp.MetaData = dataclasses.field(
        default_factory=EgoTrainingDataAADecomp.MetaData,
    )

    @property
    def num_betas(self) -> int:
        return 10

    @property
    def num_joints(self) -> int:
        return 24

    @property
    def t_world_root(self) -> Float[Tensor, "*batch timesteps 3"]:
        return self.joints_wrt_world[..., 0, :]

    @property
    def R_world_root(self) -> Float[Tensor, "*batch timesteps 3 3"]:
        device, _ = self.joints_wrt_world.device, self.joints_wrt_world.dtype
        smpl = (
            SmplFamilyMetaModelZoo[self.metadata.smpl_family_meta_model_name]
            .load(
                self.metadata.smpl_family_model_basedir,
            )
            .to(device)
        )
        t_world_root = self.joints_wrt_world[..., 0, :]

        batch_dims = self.joints_wrt_world.shape[:-2]
        flattened_obj = TensorDataclassBatchPlugin.flatten_obj(self, batch_dims)
        flattened_t_world_root, _ = TensorDataclassBatchPlugin.flatten_batch_dims(
            t_world_root,
            batch_dims,
        )

        output = smpl.model.hybrik(
            betas=flattened_obj.betas,
            phis=flattened_obj.cos_sin_phis,
            pose_skeleton=flattened_obj.joints_wrt_world,
            transl=flattened_t_world_root,
        )
        full_pose = output.rot_mats
        full_pose = TensorDataclassBatchPlugin.unflatten_batch_dims(
            full_pose,
            batch_dims,
        )
        return full_pose[..., 0, :, :]

    @property
    def loss_weights(self) -> dict[str, float]:
        # Default loss weights for absolute mode
        absolute_weights = {
            "betas": 0.2,
            "body_twists": 1.0,
            "contacts": 0.1,
            "hand_rotmats": 0.00,
            "joints_wrt_world": 4.0,
            "foot_skating": 0.3,
        }
        return absolute_weights

    def compute_loss(
        self,
        other: "AbsoluteDenoiseTrajAADecomp",
        mask: Bool[Tensor, "batch time"],
        weight_t: Float[Tensor, "batch"],
    ) -> dict[str, Float[Tensor, ""]]:
        """Compute loss between this trajectory and another using absolute representations."""
        batch, time = mask.shape[:2]
        device = mask.device

        loss_terms = {
            "betas": self._weight_and_mask_loss(
                (self.betas - other.betas) ** 2,
                mask,
                weight_t,
            ),
            # "body_twists": self._weight_and_mask_loss(
            #     (self.cos_sin_phis - other.cos_sin_phis).reshape(batch, time, -1) ** 2,
            #     mask,
            #     weight_t,
            # ),
            "contacts": self._weight_and_mask_loss(
                (self.contacts - other.contacts).reshape(batch, time, -1) ** 2,
                mask,
                weight_t,
            ),
        }

        pred_joints = self.joints_wrt_world
        gt_joints = other.joints_wrt_world

        joint_loss = (pred_joints - gt_joints) ** 2  # (b, t, 22, 3)

        if self.visible_joints_mask is not None:
            occ_jts_loss = (
                joint_loss * (~self.visible_joints_mask[..., None])
            ).reshape(batch, time, -1)
            occ_jts_loss = self._weight_and_mask_loss(occ_jts_loss, mask, weight_t)
            vis_jts_loss = (joint_loss * (self.visible_joints_mask[..., None])).reshape(
                batch,
                time,
                -1,
            )
            vis_jts_loss = self._weight_and_mask_loss(vis_jts_loss, mask, weight_t)
        else:
            logger.warning(
                "No visible joints mask found, using all joints for loss calculation, there should be no scenarios when visible_joints_mask is None",
            )
            occ_jts_loss = torch.zeros((batch, time), device=device)
            vis_jts_loss = self._weight_and_mask_loss(joint_loss, mask, weight_t)

        pred_t_parent_joint = (
            pred_joints[..., 1:, :]
            - pred_joints[..., 1:, :][..., smpl_parent_indices[1:], :]
        )
        gt_t_parent_joint = (
            gt_joints[..., 1:, :]
            - gt_joints[..., 1:, :][..., smpl_parent_indices[1:], :]
        )
        jts_relative_loss = ((pred_t_parent_joint - gt_t_parent_joint) ** 2).reshape(
            batch,
            time,
            -1,
        )
        jts_relative_loss = self._weight_and_mask_loss(
            jts_relative_loss,
            mask,
            weight_t,
        )

        # Foot skating loss
        foot_indices = [7, 8, 10, 11]  # Indices for foot joints
        foot_positions = pred_joints[..., foot_indices, :]  # (batch, time, 4, 3)
        foot_velocities = (
            foot_positions[:, 1:] - foot_positions[:, :-1]
        )  # (batch, time-1, 4, 3)

        # Get foot contacts from x_0_pred
        foot_contacts = self.contacts[..., foot_indices]  # (batch, time, 4)
        foot_skating_mask = (
            foot_contacts[:, 1:] * mask[:, 1:, None]
        ).bool()  # (batch, time-1, 4)

        # Compute foot skating loss for each foot joint
        foot_skating_losses = []
        for i in range(len(foot_indices)):
            foot_loss = self._weight_and_mask_loss(
                foot_velocities[..., i, :].pow(2),  # (batch, time-1, 3)
                bt_mask=foot_skating_mask[..., i],  # (batch, time-1)
                weight_t=weight_t,
                bt_mask_sum=torch.maximum(
                    torch.sum(foot_skating_mask[..., i]) * 3,  # Multiply by 3 for x,y,z
                    torch.tensor(1, device=device),
                ),
            )
            foot_skating_losses.append(foot_loss)

        # Average foot skating losses
        foot_skating_loss = torch.stack(foot_skating_losses).mean()

        loss_terms.update(
            {
                "occ": occ_jts_loss,
                "vis": vis_jts_loss,
                "foot_skating": foot_skating_loss,
                "jts_relative": jts_relative_loss,
            },
        )

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
        return sum(
            AbsoluteDenoiseTrajAADecomp.get_modality_dict(
                include_hands=include_hands,
            ).values(),
        )

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
        (*batch, time, _, _) = self.joints_wrt_world.shape

        # Create list of tensors to pack
        tensors_to_pack = [
            self.betas.reshape((*batch, time, -1)),
            self.cos_sin_phis.reshape((*batch, time, -1)),
            self.contacts.reshape((*batch, time, -1)),
            self.joints_wrt_world.reshape((*batch, time, -1)),
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
            "betas": 10,  # Shape parameters (self.num_betas for SMPL vs 16 for SMPL-H)
            "cos_sin_phis": 23 * 2,  # Cosine and sine of twist angles for 23 joints
            "contacts": 24,  # Contact boolean for self.num_joints joints
            "joints_wrt_world": 24 * 3,  # 3D coordinates for self.num_joints joints
        }

        # Add hand rotations if specified
        if include_hands:
            modality_dims["hand_rotmats"] = 30 * 9

        return modality_dims

    @classmethod
    # @jaxtyped(typechecker=typeguard.typechecked)
    def unpack(
        cls,
        x: Float[Tensor, "*batch timesteps d_state"],
        metadata: "EgoTrainingDataType.MetaData",
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
                joints_wrt_world_flat,
                hand_rotmats_flat,
            ) = torch.split(
                x,
                [
                    10,
                    23 * 2,
                    24,
                    24 * 3,
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
                joints_wrt_world_flat,
            ) = torch.split(
                x,
                [10, 23 * 2, 24, 24 * 3],
                dim=-1,
            )
            hand_rotmats = None
        cos_sin_phis = cos_sin_phis.reshape((*batch, time, 23, 2))
        joints_wrt_world = joints_wrt_world_flat.reshape((*batch, time, 24, 3))

        return cls(
            betas=betas,
            cos_sin_phis=cos_sin_phis,
            contacts=contacts,
            hand_rotmats=hand_rotmats,
            joints_wrt_world=joints_wrt_world,
            visible_joints_mask=torch.ones_like(
                joints_wrt_world[..., 0],
                dtype=torch.bool,
            ),  # Set to None since we don't have visibility data when unpacking
            metadata=metadata,
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
            + encoders["contacts"](self.contacts.reshape((batch, time, -1)))
            + encoders["joints_wrt_world"](
                self.joints_wrt_world.reshape((batch, time, -1)),
            )
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
        Args:
            self, other: leading dimension is (num_samples, num_timesteps)
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
                pred_Ts_world_joint=pred_Ts_world_joint[..., 1 : self.num_joints, :],
                device=device,
            ).mean(),
        )

        metrics["foot_contact"] = float(
            BodyEvaluator.compute_foot_contact(
                pred_Ts_world_joint=pred_Ts_world_joint[..., 1 : self.num_joints, :],
                device=device,
            ).mean(),
        )

        metrics["mpjpe"] = float(
            BodyEvaluator.compute_mpjpe(
                label_root_pos=gt_pose_skeletons[..., 0, :],  # [batch, T, 3]
                label_joint_pos=gt_pose_skeletons[
                    ...,
                    1 : self.num_joints,
                    :3,
                ],  # [batch, T, 23, 3]
                pred_root_pos=pred_pose_skeletons[..., 0, :],  # [batch, T, 3]
                pred_joint_pos=pred_pose_skeletons[
                    ...,
                    1 : self.num_joints,
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
                    1 : self.num_joints,
                    :3,
                ],  # [batch, T, 23, 3]
                pred_root_pos=pred_pose_skeletons[..., 0, :],  # [batch, T, 3]
                pred_joint_pos=pred_pose_skeletons[
                    ...,
                    1 : self.num_joints,
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
                    1 : self.num_joints,
                    :,
                ],  # [batch, T, 21, 7]
                pred_Ts_world_joint=pred_Ts_world_joint[
                    ...,
                    1 : self.num_joints,
                    :,
                ],  # [batch, T, 21, 7]
                device=device,
            ).mean(),
        )

        metrics["head_trans"] = float(
            BodyEvaluator.compute_head_trans(
                label_Ts_world_joint=gt_Ts_world_joint[
                    ...,
                    1 : self.num_joints,
                    :,
                ],  # [batch, T, 21, 7]
                pred_Ts_world_joint=pred_Ts_world_joint[
                    ...,
                    1 : self.num_joints,
                    :,
                ],  # [batch, T, 21, 7]
                device=device,
            ).mean(),
        )

        metrics["foot_skate"] = float(
            BodyEvaluator.compute_foot_skate(
                pred_Ts_world_joint=pred_Ts_world_joint[
                    ...,
                    1 : self.num_joints,
                    :,
                ],  # [batch, T, 21, 7]
                device=device,
            ).mean(),
        )

        metrics["foot_contact"] = float(
            BodyEvaluator.compute_foot_contact(
                pred_Ts_world_joint=pred_Ts_world_joint[
                    ...,
                    1 : self.num_joints,
                    :,
                ],  # [batch, T, 21, 7]
                device=device,
            ).mean(),
        )

        del other
        del body_model

        return metrics

    @jaxtyped(typechecker=typeguard.typechecked)
    def _rotate(self, radian: Float[Tensor, "*batch 1"]) -> Self:
        assert self.metadata.stage == "preprocessed", (
            "Only preprocessed data is supported for rotation. since preprocessing aligns data's xy to zeros. and rotation is applied only on yaw(rpy zyx convention.)"
        )

        batch_dims = radian.shape[:-1]
        so3_rot = SO3.from_z_radians(radian)
        expanded_rot = SO3(
            wxyz=so3_rot.wxyz[..., None, :, :],
        )  # [*batch, 1, 3, 3], expand rotation to this shape to be compatible with [*batch, timesteps, 22, 3], the SO3.apply func better has target param and SO3 instance has the same shape.
        assert self.joints_wrt_world.shape[:-3] == batch_dims
        self.joints_wrt_world = expanded_rot.apply(
            self.joints_wrt_world,
        )  # [*batch, timesteps, 22, 3]

        return self

    @jaxtyped(typechecker=typeguard.typechecked)
    def postprocess(
        self,
        height_from_floor: Float[Tensor, "*batch timesteps 1"],
        initial_xy: Float[Tensor, "*batch 1 2"],
        rotate_radian: Optional[Float[Tensor, "*batch 1"]] = None,
    ) -> "AbsoluteDenoiseTrajAADecomp":
        assert self.metadata.stage == "preprocessed"

        device = self.joints_wrt_world.device
        dtype = self.joints_wrt_world.dtype

        if rotate_radian is not None:
            self._rotate(
                rotate_radian.to(dtype=dtype, device=device) * -1,
            )

        self.joints_wrt_world = torch.cat(
            [
                self.joints_wrt_world[..., :2],
                self.joints_wrt_world[..., 2:3]
                + height_from_floor.unsqueeze(-2),  # [*batch, timesteps, 22, 1]
                self.joints_wrt_world[..., 3:],
            ],
            dim=-1,
        )

        self.joints_wrt_world = torch.cat(
            [
                self.joints_wrt_world[..., :2] + initial_xy.unsqueeze(-2).to(device),
                self.joints_wrt_world[..., 2:],
            ],
            dim=-1,
        )

        return self

    @jaxtyped(typechecker=typeguard.typechecked)
    def preprocess(
        self,
        visible_joints_mask: Float[Tensor, "*batch timesteps {self.num_joints}"],
        height_from_floor: Float[Tensor, "*batch timesteps 1"],
        initial_xy: Float[Tensor, "*batch 2"],
        rotate_radian: Optional[Float[Tensor, "1"]] = None,
    ) -> "EgoTrainingDataType":
        """

        Modifies the current EgoTrainingData instance by:
        1. Aligning x,y coordinates to the first frame
        2. Subtracting floor height from z coordinates
        Modifies positions in-place to save memory.
        Returns self for method chaining.
        3. Set where joints is invalid to all zeros, indicated by visible_joints_mask.
        """
        assert self.metadata.stage == "raw"

        expanded_xy = initial_xy.view(
            *initial_xy.shape[:-1],
            1,
            1,
            2,
        )  # Add dims for broadcasting

        self.joints_wrt_world = torch.cat(
            [
                self.joints_wrt_world[..., :2] - expanded_xy,
                self.joints_wrt_world[..., 2:],
            ],
            dim=-1,
        )

        self.joints_wrt_world = torch.cat(
            [
                self.joints_wrt_world[..., :2],
                self.joints_wrt_world[..., 2:3]
                - height_from_floor.unsqueeze(-2),  # [*batch, timesteps, 22, 1]
                self.joints_wrt_world[..., 3:],
            ],
            dim=-1,
        )

        if rotate_radian is not None:
            self._rotate(rotate_radian)
            self.metadata.rotate_radian = rotate_radian

        # Set where joints are invalid to all -1.
        self.joints_wrt_world = torch.where(
            visible_joints_mask.unsqueeze(-1),
            self.joints_wrt_world,
            torch.ones_like(self.joints_wrt_world) * -1,
        )

        return self
