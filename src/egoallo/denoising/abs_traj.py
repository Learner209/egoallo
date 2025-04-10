"""Traj definitions."""

from typing import TypeVar
from typing import Optional, Dict
from egoallo.middleware.third_party.HybrIK.hybrik.models.layers.smplh.fncsmplh import (
    inverse_kinematics_rotation_only,
)
from egoallo.utils.ego_geom import project_rotmats_via_svd

from egoallo.transforms import SE3, SO3
import torch
from torch import nn
from egoallo.config import CONFIG_FILE
from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import jaxtyped
from torch import Tensor
import typeguard
from egoallo.type_stubs import SmplFamilyModelType
from .base_traj import BaseDenoiseTraj
import dataclasses
from egoallo.config import make_cfg
from egoallo.utils.setup_logger import setup_logger
from egoallo.constants import (
    SmplFamilyMetaModelZoo,
)
from egoallo.type_stubs import SmplFamilyModelTypeLiteral

from egoallo.type_stubs import EgoTrainingDataType
from egoallo.tensor_dataclass_batch_plugins import TensorDataclassBatchPlugin
from egoallo.utils.transformation import orth6d_to_rotMat

local_config_file = CONFIG_FILE
CFG = make_cfg(config_name="defaults", config_file=local_config_file, cli_args=[])

logger = setup_logger(output=None, name=__name__)


T = TypeVar("T", bound="BaseDenoiseTraj")


@dataclasses.dataclass
@jaxtyped(typechecker=typeguard.typechecked)
class AbsoluteDenoiseTraj(BaseDenoiseTraj):
    from egoallo.data.dataclass import EgoTrainingData

    """Denoising trajectory with absolute pose representation."""

    betas: Float[Tensor, "*batch timesteps 16"]
    """Body shape parameters. We don't really need the timesteps axis here,
    it's just for convenience."""

    R_world_joints: Float[Tensor, "*batch timesteps 21 6"]
    """Global rotation rot6d of each body joint."""

    contacts: Float[Tensor, "*batch timesteps 22"]
    """Contact boolean for each joint."""

    hand_rotmats: Float[Tensor, "*batch timesteps 30 3 3"] | None
    """Local orientations for each body joint."""

    root_rot6d: Float[Tensor, "*batch timesteps 6"]
    """rotdd vec for root."""

    t_world_root: Float[Tensor, "*batch timesteps 3"]
    """Global translation vector of the root joint."""

    joints_wrt_world: Float[Tensor, "*batch timesteps 22 3"] | None
    """Joint positions in world frame."""

    visible_joints_mask: Float[Tensor, "*batch timesteps 22"] | None
    """Mask for visible joints."""

    metadata: EgoTrainingData.MetaData = dataclasses.field(
        default_factory=EgoTrainingData.MetaData,
    )
    """Metadata for the trajectory."""

    @property
    @jaxtyped(typechecker=typeguard.typechecked)
    def R_world_root(self) -> Float[Tensor, "*batch timesteps 3 3"]:
        batch_dims = self.root_rot6d.shape[:-1]
        flattened_root_rot6d, _ = TensorDataclassBatchPlugin.flatten_batch_dims(
            self.root_rot6d,
            batch_dims,
        )
        flattened_R_world_root = orth6d_to_rotMat(flattened_root_rot6d)[..., :, :3]
        return TensorDataclassBatchPlugin.unflatten_batch_dims(
            flattened_R_world_root,
            batch_dims,
        )

    @jaxtyped(typechecker=typeguard.typechecked)
    def body_rotmats(
        self,
        body_model: "SmplFamilyModelType",
        return_rot_6d: bool = False,
    ) -> Float[Tensor, "*batch timesteps 21 3 3"]:
        num_joints = 22
        device, _dtype = self.betas.device, self.betas.dtype

        body_model_name: "SmplFamilyModelTypeLiteral" = "SmplhModel"
        body_model = (
            SmplFamilyMetaModelZoo[body_model_name]
            .load(
                self.metadata.smpl_family_model_basedir,
                gender=self.metadata.gender,
                num_joints=num_joints,
            )
            .to(device)
        )

        flattened_R_world_joints, _ = TensorDataclassBatchPlugin.flatten_batch_dims(
            self.R_world_joints,
            self.R_world_joints.shape[:-1],
        )

        Rs_parent_joint = inverse_kinematics_rotation_only(
            self.R_world_root,
            (orth6d_to_rotMat(flattened_R_world_joints)[..., :, :3]).reshape(
                self.R_world_joints.shape[:-1] + (3, 3),
            ),
            body_model.parent_indices[
                : num_joints - 1
            ],  # body model's parent indices already excludes root.
        )

        return Rs_parent_joint

    @jaxtyped(typechecker=typeguard.typechecked)
    def compute_loss(
        self,
        other: "AbsoluteDenoiseTraj",
        mask: Bool[Tensor, "batch time"],
        weight_t: Float[Tensor, "batch"],
    ) -> dict[str, Float[Tensor, ""]]:
        """Compute loss between this trajectory and another using absolute representations."""
        batch, time = mask.shape[:2]
        num_joints = self.joints_wrt_world.shape[-2]
        device = mask.device

        body_model_name: "SmplFamilyModelTypeLiteral" = "SmplhModel"
        body_model = (
            SmplFamilyMetaModelZoo[body_model_name]
            .load(
                self.metadata.smpl_family_model_basedir,
                gender=self.metadata.gender,
                num_joints=num_joints,
            )
            .to(device)
        )

        loss_terms = {
            "betas": self._weight_and_mask_loss(
                (self.betas - other.betas) ** 2,
                mask,
                weight_t,
            ),
            "root_rot6d": self._weight_and_mask_loss(
                (self.root_rot6d - other.root_rot6d).reshape(batch, time, -1) ** 2,
                mask,
                weight_t,
            )
            * num_joints,
            "R_world_joints": self._weight_and_mask_loss(
                (self.R_world_joints - other.R_world_joints).reshape(batch, time, -1)
                ** 2,
                mask,
                weight_t,
            ),
            "t_world_root": self._weight_and_mask_loss(
                (self.t_world_root - other.t_world_root) ** 2,
                mask,
                weight_t,
            )
            * num_joints,
        }

        if self.hand_rotmats is not None and other.hand_rotmats is not None:
            loss_terms["hand_rotmats"] = self._weight_and_mask_loss(
                (self.hand_rotmats - other.hand_rotmats).reshape(batch, time, -1) ** 2,
                mask,
                weight_t,
            )

        x_0_pred_posed = self.apply_to_body(
            body_model,
        )  # (b, t, 22, 3)
        pred_joints = torch.cat(
            [
                x_0_pred_posed.T_world_root[..., 4:7].unsqueeze(dim=-2),
                x_0_pred_posed.Ts_world_joint[..., : num_joints - 1, 4:7],
            ],
            dim=-2,
        )  # (b, t, 22, 3)
        assert pred_joints.shape == (batch, time, num_joints, 3)

        # Get ground truth joints from training batch

        gt_joints = other.joints_wrt_world  # (b, t, 22, 3)
        assert gt_joints.shape == (batch, time, num_joints, 3)

        # Calculate joint position loss with masking
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
                # empirically, invisible joints loss should be more important than visible joints loss.
                "occ": occ_jts_loss,
                "vis": vis_jts_loss,
                "foot_skating": foot_skating_loss,
            },
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
        num_smplh_jnts = 22
        packed_dim = 16 + (num_smplh_jnts - 1) * 6 + (num_smplh_jnts) + 6 + 3
        if include_hands:
            packed_dim += 30 * 9  # hand_rotmats
        return packed_dim

    def apply_to_body(self, body_model: "SmplFamilyModelType") -> "SmplFamilyModelType":
        """Apply the trajectory data to a SMPL-H body model."""
        # assert self.hand_rotmats is not None
        *batch, time, _ = self.betas.shape
        device, dtype = self.betas.device, self.betas.dtype

        shaped = body_model.with_shape(
            self.betas,
        )  # betas averges across timestep dimensions.
        T_world_root = SE3.from_rotation_and_translation(
            SO3.from_matrix(self.R_world_root),
            self.t_world_root,
        ).parameters()

        left_hand_quats = (
            SO3.identity(device=device, dtype=dtype).wxyz.repeat(*batch, time, 15, 1)
            if self.hand_rotmats is None
            else None
        )
        right_hand_quats = (
            SO3.identity(device=device, dtype=dtype).wxyz.repeat(*batch, time, 15, 1)
            if self.hand_rotmats is None
            else None
        )
        posed = shaped.with_pose_decomposed(
            T_world_root=T_world_root,
            body_quats=SO3.from_matrix(self.body_rotmats(body_model)).wxyz,
            left_hand_quats=left_hand_quats,
            right_hand_quats=right_hand_quats,
        )

        return posed

    @jaxtyped(typechecker=typeguard.typechecked)
    def pack(self) -> Float[Tensor, "*batch timesteps d_state"]:
        """Pack trajectory into a single flattened vector."""
        (*batch, time, num_joints, _) = self.R_world_joints.shape
        assert num_joints == 21

        # Create list of tensors to pack
        tensors_to_pack = [
            self.betas.reshape((*batch, time, -1)),
            self.R_world_joints.reshape((*batch, time, -1)),
            self.contacts.reshape((*batch, time, -1)),
            self.root_rot6d.reshape((*batch, time, -1)),
            self.t_world_root.reshape((*batch, time, -1)),
        ]

        if self.hand_rotmats is not None:
            tensors_to_pack.append(self.hand_rotmats.reshape((*batch, time, -1)))

        return torch.cat(tensors_to_pack, dim=-1)

    @classmethod
    # @jaxtyped(typechecker=typeguard.typechecked)
    def unpack(
        cls,
        x: Float[Tensor, "*batch timesteps d_state"],
        metadata: "EgoTrainingDataType.MetaData",
        include_hands: bool = False,
        project_rotmats: bool = False,
    ) -> "AbsoluteDenoiseTraj":
        """Unpack trajectory from a single flattened vector."""
        (*batch, time, d_state) = x.shape
        assert d_state == cls.get_packed_dim(include_hands)
        num_joints = 22

        if include_hands:
            (
                betas,
                R_world_joints_flat,
                contacts,
                root_rot6d,
                t_world_root,
                hand_rotmats_flat,
            ) = torch.split(
                x,
                [
                    16,
                    (num_joints - 1) * 6,
                    num_joints,
                    6,
                    3,
                    30 * 9,
                ],
                dim=-1,
            )
            R_world_joints = R_world_joints_flat.reshape(
                (*batch, time, (num_joints - 1), 6),
            )
            hand_rotmats = hand_rotmats_flat.reshape((*batch, time, 30, 3, 3))
        else:
            (
                betas,
                R_world_joints_flat,
                contacts,
                root_rot6d,
                t_world_root,
            ) = torch.split(
                x,
                [16, (num_joints - 1) * 6, num_joints, 6, 3],
                dim=-1,
            )
            R_world_joints = R_world_joints_flat.reshape(
                (*batch, time, (num_joints - 1), 6),
            )
            hand_rotmats = None

        if project_rotmats and hand_rotmats is not None:
            if hand_rotmats is not None:
                hand_rotmats = project_rotmats_via_svd(hand_rotmats)

        return cls(
            betas=betas,
            R_world_joints=R_world_joints,
            contacts=contacts,
            hand_rotmats=hand_rotmats,
            t_world_root=t_world_root,
            root_rot6d=root_rot6d,
            joints_wrt_world=None,  # Set to None since we don't have joints data when unpacking
            visible_joints_mask=None,  # Set to None since we don't have visibility data when unpacking
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
            + encoders["R_world_joints"](self.R_world_joints.reshape((batch, time, -1)))
            + encoders["contacts"](self.contacts)
            + encoders["root_rot6d"](self.root_rot6d.reshape((batch, time, -1)))
            + encoders["t_world_root"](self.t_world_root)
        )
        if self.hand_rotmats is not None:
            encoded = encoded + encoders["hand_rotmats"](
                self.hand_rotmats.reshape((batch, time, -1)),
            )
        return encoded

    def _compute_metrics(
        self,
        other: "AbsoluteDenoiseTraj",
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
        gt_posed = gt_shaped.with_pose_decomposed(
            T_world_root=SE3.from_rotation_and_translation(
                SO3.from_matrix(other.R_world_root),
                other.t_world_root,
            )
            .parameters()
            .to(device),
            body_quats=SO3.from_matrix(other.body_rotmats).wxyz.to(device),
        )
        pred_shaped = body_model.with_shape(self.betas)
        pred_posed = pred_shaped.with_pose_decomposed(
            T_world_root=SE3.from_rotation_and_translation(
                SO3.from_matrix(self.R_world_root),
                self.t_world_root,
            )
            .parameters()
            .to(device),
            body_quats=SO3.from_matrix(self.body_rotmats).wxyz.to(device),
        )

        num_samples, num_timesteps = self.betas.shape[:-1]
        # Body shape error
        metrics["betas_error"] = float(
            BodyEvaluator.compute_masked_error(
                gt=other.betas.reshape(*other.betas.shape[:-1], -1),  # N, T, 16
                pred=self.betas.reshape(*self.betas.shape[:-1], -1),  # N, T, 16
                device=device,
            ),
        )

        # Body rotation error
        metrics["R_world_joints_error"] = float(
            BodyEvaluator.compute_masked_error(
                gt=other.R_world_joints.reshape(
                    *other.R_world_joints.shape[:-3],
                    -1,
                ),  # N, T, 207
                pred=self.R_world_joints.reshape(
                    *self.R_world_joints.shape[:-3],
                    -1,
                ),  # N, T, 207
                device=device,
            ),
        )

        # Root transform errors
        metrics["root_rot6d_error"] = float(
            BodyEvaluator.compute_masked_error(
                gt=other.root_rot6d.reshape(
                    *other.root_rot6d.shape[:-2],
                    -1,
                ),  # N, T, 9
                pred=self.root_rot6d.reshape(
                    *self.root_rot6d.shape[:-2],
                    -1,
                ),  # N, T, 9
                device=device,
            ),
        )

        metrics["t_world_root_error"] = float(
            BodyEvaluator.compute_masked_error(
                gt=other.t_world_root.reshape(*other.t_world_root.shape[:-1], -1),
                pred=self.t_world_root.reshape(*self.t_world_root.shape[:-1], -1),
                device=device,
            )
            * 1000,
        )

        # Foot metrics
        metrics["foot_skate"] = float(
            BodyEvaluator.compute_foot_skate(
                pred_Ts_world_joint=pred_posed.Ts_world_joint[..., :21, :],
                device=device,
            ).mean(),
        )

        metrics["foot_contact"] = float(
            BodyEvaluator.compute_foot_contact(
                pred_Ts_world_joint=pred_posed.Ts_world_joint[..., :21, :],
                device=device,
            ).mean(),
        )

        metrics["mpjpe"] = float(
            BodyEvaluator.compute_mpjpe(
                label_root_pos=gt_posed.T_world_root[..., :3],  # [batch, T, 3]
                label_joint_pos=gt_posed.Ts_world_joint[
                    ...,
                    :21,
                    4:,
                ],  # [batch, T, 21, 3]
                pred_root_pos=pred_posed.T_world_root[..., :3],  # [batch, T, 3]
                pred_joint_pos=pred_posed.Ts_world_joint[
                    ...,
                    :21,
                    4:,
                ],  # [batch, T, 21, 3]
                per_frame_procrustes_align=False,
                device=device,
            ).mean(),
        )

        metrics["pampjpe"] = float(
            BodyEvaluator.compute_mpjpe(
                label_root_pos=gt_posed.T_world_root[..., :3],  # [batch, T, 3]
                label_joint_pos=gt_posed.Ts_world_joint[
                    ...,
                    :21,
                    4:,
                ],  # [batch, T, 21, 3]
                pred_root_pos=pred_posed.T_world_root[..., :3],  # [batch, T, 3]
                pred_joint_pos=pred_posed.Ts_world_joint[
                    ...,
                    :21,
                    4:,
                ],  # [batch, T, 21, 3]
                per_frame_procrustes_align=True,
                device=device,
            ).mean(),
        )

        metrics["head_ori"] = float(
            BodyEvaluator.compute_head_ori(
                label_Ts_world_joint=gt_posed.Ts_world_joint[
                    ...,
                    :21,
                    :,
                ],  # [batch, T, 21, 7]
                pred_Ts_world_joint=pred_posed.Ts_world_joint[
                    ...,
                    :21,
                    :,
                ],  # [batch, T, 21, 7]
                device=device,
            ).mean(),
        )

        metrics["head_trans"] = float(
            BodyEvaluator.compute_head_trans(
                label_Ts_world_joint=gt_posed.Ts_world_joint[
                    ...,
                    :21,
                    :,
                ],  # [batch, T, 21, 7]
                pred_Ts_world_joint=pred_posed.Ts_world_joint[
                    ...,
                    :21,
                    :,
                ],  # [batch, T, 21, 7]
                device=device,
            ).mean(),
        )

        metrics["foot_skate"] = float(
            BodyEvaluator.compute_foot_skate(
                pred_Ts_world_joint=pred_posed.Ts_world_joint[
                    ...,
                    :21,
                    :,
                ],  # [batch, T, 21, 7]
                device=device,
            ).mean(),
        )

        metrics["foot_contact"] = float(
            BodyEvaluator.compute_foot_contact(
                pred_Ts_world_joint=pred_posed.Ts_world_joint[
                    ...,
                    :21,
                    :,
                ],  # [batch, T, 21, 7]
                device=device,
            ).mean(),
        )

        del other
        del body_model

        return metrics

    @classmethod
    def get_modality_dict(cls, include_hands: bool = False) -> dict[str, int]:
        """Get dictionary of modalities and their dimensions.

        For AbsoluteDenoiseTraj, includes shape parameters, joint rotations, contacts,
        global root position and orientation.

        Args:
            include_hands: Whether to include hand rotations in the dictionary

        Returns:
            Dictionary mapping modality names to their dimensions
        """
        # Base modalities for absolute mode
        modality_dims = {
            "betas": 16,
            "R_world_joints": 21 * 6,
            "contacts": 22,
            "root_rot6d": 6,  # 3x3 rotation matrix
            "t_world_root": 3,  # 3D translation vector
        }

        # Add hand rotations if specified
        if include_hands:
            modality_dims["hand_rotmats"] = 30 * 9

        return modality_dims
