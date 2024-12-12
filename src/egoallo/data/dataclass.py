from pathlib import Path

import numpy as np
import torch
import torch.utils.data
import typeguard
from jaxtyping import Bool, Float, jaxtyped, Array
from beartype import beartype
from torch import Tensor

from .. import fncsmpl, fncsmpl_extensions
from .. import transforms as tf
from ..tensor_dataclass import TensorDataclass
from typing import Optional

from ..network import EgoDenoiserConfig
from ..viz.smpl_viewer import visualize_ego_training_data as viz_ego_data

from jaxtyping import Float, jaxtyped
from typeguard import typechecked
from dataclasses import dataclass


@jaxtyped(typechecker=typeguard.typechecked)
class EgoTrainingData(TensorDataclass):
    """Dictionary of tensors we use for EgoAllo training."""

    T_world_root: Float[Tensor, "*batch timesteps 7"]
    """Transformation from the world frame to the root frame at each timestep."""

    contacts: Float[Tensor, "*batch timesteps 22"]
    """Contact boolean for each joint."""

    betas: Float[Tensor, "*batch 1 16"]
    """Body shape parameters."""

    # Excluded because not needed.
    joints_wrt_world: Float[Tensor, "*batch timesteps 22 3"]
    """Joint positions relative to the world frame."""
    # @property
    # def joints_wrt_world(self) -> Tensor:
    #     return tf.SE3(self.T_world_cpf[..., None, :]) @ self.joints_wrt_cpf

    body_quats: Float[Tensor, "*batch timesteps 21 4"]
    """Local orientations for each body joint."""

    T_world_cpf: Float[Tensor, "*batch timesteps 7"]
    """Transformation from the world frame to the central pupil frame at each timestep."""

    height_from_floor: Float[Tensor, "*batch timesteps 1"]
    """Distance from CPF to floor at each timestep."""

    joints_wrt_cpf: Float[Tensor, "*batch timesteps 22 3"]
    """Joint positions relative to the central pupil frame."""

    mask: Bool[Tensor, "*batch timesteps"]
    """Mask to support variable-length sequence."""

    hand_quats: Float[Tensor, "*batch timesteps 30 4"] | None
    """Local orientations for each hand joint."""

    visible_joints_mask: Bool[Tensor, "*batch timesteps 22"] | None
    """Boolean mask indicating which joints are visible (not masked)"""

    # visible_joints: Float[Tensor, "*batch timesteps 21 3"] | None
    # """Joint positions relative to the central pupil frame for visible joints."""

    @staticmethod
    def load_from_npz(
        body_model: fncsmpl.SmplhModel,
        path: Path,
        include_hands: bool,
    ) -> "EgoTrainingData":
        """Load a single trajectory from a (processed_30fps) npz file."""
        raw_fields = {
            k: torch.from_numpy(v.astype(np.float32) if v.dtype == np.float64 else v)
            for k, v in np.load(path, allow_pickle=True).items()
            if v.dtype in (np.float32, np.float64)
        }

        timesteps = raw_fields["root_orient"].shape[0]
        assert raw_fields["root_orient"].shape == (timesteps, 3)
        assert raw_fields["pose_body"].shape == (timesteps, 63)
        assert raw_fields["pose_hand"].shape == (timesteps, 90)
        assert raw_fields["joints"].shape == (timesteps, 22, 3)

        T_world_root = torch.cat(
            [
                tf.SO3.exp(raw_fields["root_orient"]).wxyz,
                raw_fields["joints"][:, 0, :],
            ],
            dim=-1,
        )
        body_quats = tf.SO3.exp(raw_fields["pose_body"].reshape(timesteps, 21, 3)).wxyz
        hand_quats = tf.SO3.exp(raw_fields["pose_hand"].reshape(timesteps, 30, 3)).wxyz

        device = body_model.weights.device
        # import ipdb; ipdb.set_trace()
        shaped = body_model.with_shape(raw_fields["betas"].unsqueeze(0).to(device))

        # Batch the SMPL body model operations, this can be pretty memory-intensive...
        posed = shaped.with_pose_decomposed(
            T_world_root=T_world_root.to(device), body_quats=body_quats.to(device)
        )
        T_world_cpf = (
            tf.SE3(posed.Ts_world_joint[:, 14, :])  # T_world_head
            @ tf.SE3(fncsmpl_extensions.get_T_head_cpf(shaped))
        ).parameters()
        assert T_world_cpf.shape == (timesteps, 7)

        return EgoTrainingData(
            T_world_root=T_world_root.cpu(),
            contacts=raw_fields["contacts"].cpu(),  # Root is no longer a joint.
            betas=raw_fields["betas"].unsqueeze(0).cpu(),
            joints_wrt_world=raw_fields["joints"].cpu(),  # root is included.
            body_quats=body_quats.cpu(),
            # CPF frame stuff.
            T_world_cpf=T_world_cpf.cpu(),
            # Get translational z coordinate from wxyz_xyz.
            height_from_floor=T_world_cpf[:, 6:7].cpu(),
            joints_wrt_cpf=(
                # unsqueeze so both shapes are (timesteps, joints, dim)
                tf.SE3(T_world_cpf[:, None, :]).inverse()
                @ raw_fields["joints"].to(T_world_cpf.device)
            ).cpu(),
            mask=torch.ones((timesteps,), dtype=torch.bool),
            hand_quats=hand_quats.cpu() if include_hands else None,
            visible_joints_mask=None,
        )

    @staticmethod
    def visualize_ego_training_data(
        data: "EgoTrainingData",
        body_model: fncsmpl.SmplhModel,
        output_path: str = "output.mp4",
    ):
        viz_ego_data(
            data,
            body_model=body_model,
            output_path=output_path,
        )


def collate_dataclass[T](batch: list[T]) -> T:
    """Collate function that works for dataclasses."""
    keys = vars(batch[0]).keys()
    return type(batch[0])(
        **{k: torch.stack([getattr(b, k) for b in batch]) for k in keys}
    )
