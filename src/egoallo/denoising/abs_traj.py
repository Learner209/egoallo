"""Traj definitions."""

from typing import TypeVar
from typing import Optional, Dict
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

local_config_file = CONFIG_FILE
CFG = make_cfg(config_name="defaults", config_file=local_config_file, cli_args=[])

logger = setup_logger(output=None, name=__name__)


T = TypeVar("T", bound="BaseDenoiseTraj")


@dataclasses.dataclass
class AbsoluteDenoiseTraj(BaseDenoiseTraj):
    from egoallo.data.dataclass import EgoTrainingData

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

    joints_wrt_world: Float[Tensor, "*batch timesteps 22 3"] | None
    """Joint positions in world frame."""

    visible_joints_mask: Float[Tensor, "*batch timesteps 22"] | None
    """Mask for visible joints."""

    metadata: EgoTrainingData.MetaData = dataclasses.field(
        default_factory=EgoTrainingData.MetaData,
    )
    """Metadata for the trajectory."""

    @property
    def loss_weights(self) -> dict[str, float]:
        # Default loss weights for absolute mode
        absolute_weights = {
            "betas": 0.2,
            "body_rotmats": 1.0,
            "contacts": 0.1,
            "hand_rotmats": 0.00,
            "R_world_root": 2.0,
            "t_world_root": 2.0,
            "joints": 3.0,
            "foot_skating": 0.3,
            "velocity": 0.1,
        }
        return absolute_weights

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
                (self.betas - other.betas) ** 2,
                mask,
                weight_t,
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
                (self.t_world_root - other.t_world_root) ** 2,
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
        # 16 (betas) + 21*9 (body_rotmats) + 21 (contacts) + 9 (R_world_root) + 3 (t_world_root)
        num_smplh_jnts = CFG.smplh.num_joints
        packed_dim = 16 + (num_smplh_jnts - 1) * 9 + (num_smplh_jnts) + 9 + 3
        if include_hands:
            packed_dim += 30 * 9  # hand_rotmats
        return packed_dim

    def apply_to_body(self, body_model: "SmplFamilyModelType") -> "SmplFamilyModelType":
        """Apply the trajectory data to a SMPL-H body model."""
        # assert self.hand_rotmats is not None

        shaped = body_model.with_shape(
            self.betas,
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
                (*batch, time, (CFG.smplh.num_joints - 1), 3, 3),
            )
            hand_rotmats = hand_rotmats_flat.reshape((*batch, time, 30, 3, 3))
        else:
            (
                betas,
                body_rotmats_flat,
                contacts,
                R_world_root,
                t_world_root,
            ) = torch.split(
                x,
                [16, (CFG.smplh.num_joints - 1) * 9, CFG.smplh.num_joints, 9, 3],
                dim=-1,
            )
            body_rotmats = body_rotmats_flat.reshape(
                (*batch, time, (CFG.smplh.num_joints - 1), 3, 3),
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
            + encoders["body_rotmats"](self.body_rotmats.reshape((batch, time, -1)))
            + encoders["contacts"](self.contacts)
            + encoders["R_world_root"](self.R_world_root.reshape((batch, time, -1)))
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
        metrics["body_rotmats_error"] = float(
            BodyEvaluator.compute_masked_error(
                gt=other.body_rotmats.reshape(
                    *other.body_rotmats.shape[:-3],
                    -1,
                ),  # N, T, 207
                pred=self.body_rotmats.reshape(
                    *self.body_rotmats.shape[:-3],
                    -1,
                ),  # N, T, 207
                device=device,
            ),
        )

        # Root transform errors
        metrics["R_world_root_error"] = float(
            BodyEvaluator.compute_masked_error(
                gt=other.R_world_root.reshape(
                    *other.R_world_root.shape[:-2],
                    -1,
                ),  # N, T, 9
                pred=self.R_world_root.reshape(
                    *self.R_world_root.shape[:-2],
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
        num_smplh_jnts = CFG.smplh.num_joints

        # Base modalities for absolute mode
        modality_dims = {
            "betas": 16,
            "body_rotmats": (num_smplh_jnts - 1) * 9,
            "contacts": num_smplh_jnts,
            "R_world_root": 9,  # 3x3 rotation matrix
            "t_world_root": 3,  # 3D translation vector
        }

        # Add hand rotations if specified
        if include_hands:
            modality_dims["hand_rotmats"] = 30 * 9

        return modality_dims
