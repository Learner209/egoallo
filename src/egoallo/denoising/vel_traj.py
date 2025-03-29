"""Traj definitions."""

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
        return velocity_weights

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
        batch_shape = self.R_world_root.shape[:-3]  # [batch, T, 3, 3]
        device = self.R_world_root.device
        dtype = self.R_world_root.dtype

        # Compute relative rotations using SO3
        R_curr = SO3.from_matrix(self.R_world_root[..., 1:, :, :])  # [batch, T-1, 3, 3]
        R_prev = SO3.from_matrix(
            self.R_world_root[..., :-1, :, :],
        )  # [batch, T-1, 3, 3]
        R_rel = R_curr.multiply(R_prev.inverse())  # [batch, T-1, 3, 3]
        self.R_world_root_tm1_t = torch.cat(
            [
                torch.eye(3, device=device, dtype=dtype).expand(*batch_shape, 1, 3, 3),
                R_rel.as_matrix(),
            ],
            dim=-3,
        )

        # Compute relative translations using SE3
        T_curr = SE3.from_rotation_and_translation(
            SO3.from_matrix(self.R_world_root),
            self.t_world_root,
        )  # [batch, T, 3, 3]
        self.t_world_root_tm1_t = torch.zeros_like(self.t_world_root)  # [batch, T, 3]
        self.t_world_root_tm1_t[..., 1:, :] = (
            T_curr.translation()[..., 1:, :] - T_curr.translation()[..., :-1, :]
        )

        # Compute accelerations
        # For rotations, multiply consecutive relative rotations
        R_rel_curr = SO3.from_matrix(self.R_world_root_tm1_t[..., 2:, :, :])
        R_rel_prev = SO3.from_matrix(self.R_world_root_tm1_t[..., 1:-1, :, :])
        self.R_world_root_acc = torch.zeros_like(self.R_world_root)
        self.R_world_root_acc[..., 2:, :, :] = R_rel_curr.multiply(
            R_rel_prev.inverse(),
        ).as_matrix()

        # For translations, compute difference of consecutive relative translations
        self.t_world_root_acc = torch.zeros_like(self.t_world_root)
        self.t_world_root_acc[..., 2:, :] = (
            self.t_world_root_tm1_t[..., 2:, :] - self.t_world_root_tm1_t[..., 1:-1, :]
        )

    def apply_to_body(self, body_model: "SmplFamilyModelType") -> "SmplFamilyModelType":
        """Apply the trajectory data to a SMPL-H body model.

        This method first reconstructs absolute positions from temporal offsets,
        then applies them to the body model.
        """
        device = self.betas.device
        dtype = self.betas.dtype
        batch_shape = self.R_world_root_tm1_t.shape[:-3]  # [batch, T, 3, 3]
        time = self.R_world_root_tm1_t.shape[-3]

        # Initialize absolute positions with identity rotation and zero translation
        R_world_root = (
            torch.eye(3, device=device, dtype=dtype)
            .expand(*batch_shape, time, 3, 3)
            .clone()
        )
        t_world_root = torch.zeros((*batch_shape, time, 3), device=device, dtype=dtype)

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
                (*batch, time, -1),
            ),  # Pack temporal offsets instead
            self.t_world_root_tm1_t.reshape(
                (*batch, time, -1),
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
                (*batch, time, (CFG.smplh.num_joints - 1), 3, 3),
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
                (*batch, time, (CFG.smplh.num_joints - 1), 3, 3),
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
                (self.betas - other.betas) ** 2,
                mask,
                weight_t,
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
                    batch,
                    time,
                    -1,
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
                    batch,
                    time,
                    -1,
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

    def encode(
        self,
        encoders: nn.ModuleDict,
        batch: int,
        time: int,
    ) -> Float[Tensor, "batch time d_latent"]:
        """Encode trajectory into latent space."""
        encoded = (
            encoders["betas"](self.betas.reshape((batch, time, -1)))
            + encoders["body_rotmats"](self.body_rotmats.reshape((batch, time, -1)))
            + encoders["contacts"](self.contacts)
            + encoders["R_world_root_tm1_t"](
                self.R_world_root_tm1_t.reshape((batch, time, -1)),
            )
            + encoders["t_world_root_tm1_t"](self.t_world_root_tm1_t)
        )
        if self.hand_rotmats is not None:
            encoded = encoded + encoders["hand_rotmats"](
                self.hand_rotmats.reshape((batch, time, -1)),
            )
        return encoded

    def _compute_metrics(
        self,
        other: "VelocityDenoiseTraj",
        body_model: Optional["SmplFamilyModelType"] = None,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, float]:
        """Compute metrics between this trajectory and another.
        Focuses on velocity-based metrics while still computing absolute pose errors.
        """
        metrics = {}
        assert self.check_shapes(other), f"{self.check_shapes(other)}"

        # First reconstruct absolute poses
        pred_posed: "SmplFamilyModelType" = self.apply_to_body(body_model=body_model)
        gt_posed: "SmplFamilyModelType" = other.apply_to_body(body_model=body_model)

        # Body shape error (absolute)
        # TEMPORARY_FIX: import BodyEvaluator lazily to avoid circular imports
        from egoallo.evaluation.body_evaluator import BodyEvaluator

        # Body shape error

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
            ),
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

        return metrics

    @classmethod
    def get_modality_dict(cls, include_hands: bool = False) -> dict[str, int]:
        """Get dictionary of modalities and their dimensions.

        For VelocityDenoiseTraj, includes shape parameters, joint rotations, contacts,
        and relative root transformations between consecutive frames.

        Args:
            include_hands: Whether to include hand rotations in the dictionary

        Returns:
            Dictionary mapping modality names to their dimensions
        """
        num_smplh_jnts = CFG.smplh.num_joints

        # Base modalities for velocity mode
        modality_dims = {
            "betas": 16,
            "body_rotmats": (num_smplh_jnts - 1) * 9,
            "contacts": num_smplh_jnts,
            "R_world_root_tm1_t": 9,  # 3x3 rotation matrix
            "t_world_root_tm1_t": 3,  # 3D translation vector
        }

        # Add hand rotations if specified
        if include_hands:
            modality_dims["hand_rotmats"] = 30 * 9

        return modality_dims
