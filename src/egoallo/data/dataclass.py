from pathlib import Path

import numpy as np
import torch
import torch.utils.data
import typeguard
from jaxtyping import Bool, Float, jaxtyped, Array
from egoallo.transforms import SO3, SE3
from torch import Tensor

from .. import fncsmpl, fncsmpl_extensions
from .. import transforms as tf
from ..tensor_dataclass import TensorDataclass
from typing import Optional

from ..network import EgoDenoiseTraj, EgoDenoiserConfig
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
        assert raw_fields["contacts"].shape == (timesteps, 52) or raw_fields[
            "contacts"
        ].shape == (timesteps, 22)
        assert raw_fields["joints"].shape == (timesteps, 22, 3)
        if raw_fields["betas"].shape[0] == 10:
            raw_fields["betas"] = torch.cat([raw_fields["betas"], torch.zeros(6)])
        assert raw_fields["betas"].shape[0] == 16

        device = body_model.weights.device

        T_world_root = torch.cat(
            [
                tf.SO3.exp(raw_fields["root_orient"]).wxyz,
                raw_fields["joints"][:, 0, :],
            ],
            dim=-1,
        ).to(device)

        body_quats = tf.SO3.exp(
            raw_fields["pose_body"].reshape(timesteps, 21, 3)
        ).wxyz.to(device)
        hand_quats = tf.SO3.exp(
            raw_fields["pose_hand"].reshape(timesteps, 30, 3)
        ).wxyz.to(device)

        shaped = body_model.with_shape(raw_fields["betas"].unsqueeze(0).to(device))

        # Batch the SMPL body model operations, this can be pretty memory-intensive...
        posed = shaped.with_pose_decomposed(
            T_world_root=T_world_root.to(device), body_quats=body_quats.to(device)
        )

        # Get initial x,y position offset (ignoring z)
        initial_xy = T_world_root[1, 4:6]  # First frame x,y position

        # Align positions by subtracting x,y offset only
        T_world_root_aligned = T_world_root.clone()
        T_world_root_aligned[..., 4:6] = T_world_root[..., 4:6] - initial_xy

        # Align joints_wrt_world (x,y only)
        joints_wrt_world_aligned = raw_fields["joints"].clone()
        joints_wrt_world_aligned[..., :2] = (
            raw_fields["joints"][..., :2].to(device) - initial_xy
        )

        # Align T_world_cpf (only x,y translation component)
        T_world_cpf = (
            tf.SE3(posed.Ts_world_joint[:, 14, :])  # T_world_head
            @ tf.SE3(fncsmpl_extensions.get_T_head_cpf(shaped))
        ).parameters()

        T_world_cpf_aligned = T_world_cpf.clone()
        T_world_cpf_aligned[..., 4:6] = T_world_cpf[..., 4:6] - initial_xy

        return EgoTrainingData(
            T_world_root=T_world_root_aligned.cpu(),
            contacts=raw_fields["contacts"][:, :22].cpu(),  # root is included.
            betas=raw_fields["betas"].unsqueeze(0).cpu(),
            joints_wrt_world=joints_wrt_world_aligned.cpu(),  # root is included.
            body_quats=body_quats.cpu(),
            # CPF frame stuff.
            T_world_cpf=T_world_cpf_aligned.cpu(),
            height_from_floor=T_world_cpf_aligned[:, 6:7].cpu(),
            joints_wrt_cpf=(
                # unsqueeze so both shapes are (timesteps, joints, dim)
                tf.SE3(T_world_cpf_aligned[:, None, :]).inverse()
                @ joints_wrt_world_aligned.to(T_world_cpf_aligned.device)
            ).cpu(),
            mask=torch.ones((timesteps,), dtype=torch.bool),
            hand_quats=hand_quats.cpu() if include_hands else None,
            visible_joints_mask=None,
        )

    @staticmethod
    def visualize_ego_training_data(
        data: EgoDenoiseTraj,
        body_model: fncsmpl.SmplhModel,
        output_path: str = "output.mp4",
    ):
        viz_ego_data(
            data,
            body_model=body_model,
            output_path=output_path,
        )

    def to_denoise_traj(self, include_hands: bool = True) -> EgoDenoiseTraj:
        """Convert EgoTrainingData instance to EgoDenoiseTraj instance."""
        *batch, time, _ = self.T_world_root.shape

        # Extract rotation and translation from T_world_root
        R_world_root = SO3(self.T_world_root[..., :4]).as_matrix()
        t_world_root = self.T_world_root[..., 4:7]

        # Convert body quaternions to rotation matrices
        body_rotmats = SO3(self.body_quats).as_matrix()

        # Handle hand data if present
        hand_rotmats = None
        if self.hand_quats is not None and include_hands:
            hand_rotmats = SO3(self.hand_quats).as_matrix()

        # Create and return EgoDenoiseTraj instance
        return EgoDenoiseTraj(
            betas=self.betas.expand((*batch, time, 16)),
            body_rotmats=body_rotmats,
            contacts=self.contacts,
            hand_rotmats=hand_rotmats,
            R_world_root=R_world_root,
            t_world_root=t_world_root,
        )


def collate_dataclass[T](batch: list[T]) -> T:
    """Collate function that works for dataclasses."""
    keys = vars(batch[0]).keys()
    return type(batch[0])(
        **{k: torch.stack([getattr(b, k) for b in batch]) for k in keys}
    )


def collate_tensor_only_dataclass[T](batch: list[T]) -> T:
    """Collate function that only stacks tensor attributes in dataclasses.

    This is a more flexible version that:
    1. Only stacks torch.Tensor and np.ndarray attributes
    2. Ignores non-tensor attributes (preserves first item's value)
    3. Handles None values and optional fields
    4. Supports nested dataclasses

    Args:
        batch: List of dataclass instances to collate

    Returns:
        Collated dataclass with stacked tensor attributes

    Example:
        >>> @dataclass
        >>> class Data:
        >>>     x: torch.Tensor
        >>>     y: Optional[torch.Tensor] = None
        >>>     z: str = "test"
        >>> batch = [Data(x=torch.ones(3), y=None, z="a"),
        >>>         Data(x=torch.zeros(3), y=torch.ones(2), z="b")]
        >>> result = collate_tensor_only_dataclass(batch)
        >>> # result.x: tensor([[1,1,1], [0,0,0]])
        >>> # result.y: None
        >>> # result.z: "a"
    """
    if not batch:
        raise ValueError("Empty batch")

    # Get first item as reference
    first = batch[0]
    if not hasattr(first, "__dataclass_fields__"):
        raise TypeError("Expected dataclass instance")

    # Initialize output dict
    collated = {}

    # Get all field names from first item
    fields = vars(first).keys()

    for key in fields:
        # Get values for this field from all items
        values = [getattr(item, key) for item in batch]

        # Handle first non-None value as reference
        ref_val = next((v for v in values if v is not None), None)

        if ref_val is None:
            # If all values are None, keep as None
            collated[key] = None

        elif isinstance(ref_val, (torch.Tensor, np.ndarray)):
            # Stack tensors/arrays, filtering out None values
            valid_values = [v for v in values if v is not None]
            if valid_values:
                if isinstance(ref_val, torch.Tensor):
                    collated[key] = torch.stack(valid_values)
                else:
                    collated[key] = np.stack(valid_values)
            else:
                collated[key] = None

        elif hasattr(ref_val, "__dataclass_fields__"):
            # Recursively handle nested dataclasses
            valid_values = [v for v in values if v is not None]
            if valid_values:
                collated[key] = collate_tensor_only_dataclass(valid_values)
            else:
                collated[key] = None

        else:
            # For non-tensor fields, keep first item's value
            collated[key] = values[0]

    return type(first)(**collated)
