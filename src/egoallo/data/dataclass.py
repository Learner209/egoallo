from pathlib import Path
from typing import Union, assert_never, TYPE_CHECKING
import dataclasses
import numpy as np
import torch
import torch.utils.data
import typeguard
from jaxtyping import Bool, Float, jaxtyped, Array
from egoallo.transforms import SO3, SE3
from egoallo import network
from torch import Tensor

if TYPE_CHECKING:
    from egoallo.types import DenoiseTrajType

from .. import fncsmpl, fncsmpl_extensions
from .. import transforms as tf
from ..tensor_dataclass import TensorDataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..network import EgoDenoiserConfig
from ..viz.smpl_viewer import visualize_ego_training_data as viz_ego_data

from jaxtyping import Float, jaxtyped
from typeguard import typechecked
from dataclasses import dataclass


@jaxtyped(typechecker=typeguard.typechecked)
class EgoTrainingData(TensorDataclass):
    """Dictionary of tensors we use for EgoAllo training."""
    # NOTE: if the attr is tensor/np.ndarray type, then it must has a leading batch dimension, whether it can be broadcasted or not.
    # NOTE: since the `tensor_dataclass` will convert the tensor to a single element tensor, we need to make sure the leading dimension is always there.

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

    take_name: str = ""
    """Name of the take."""

    frame_keys: tuple[int, ...] = ()
    """Keys of the frames in the npz file."""

    initial_xy: Float[Tensor, "*batch 2"] = torch.FloatTensor([0.0, 0.0])
    """Initial x,y position offset from visible joints in first frame"""

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
        initial_xy = T_world_root[0, 4:6]  # First frame x,y position

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
        data: "DenoiseTrajType",
        body_model: fncsmpl.SmplhModel,
        output_path: str = "output.mp4",
        **kwargs,
    ):
        viz_ego_data(
            data,
            body_model=body_model,
            output_path=output_path,
            **kwargs,
        )

    @jaxtyped(typechecker=typeguard.typechecked)
    def align_to_first_frame(self) -> "EgoTrainingData":
        """
        Modifies the current EgoTrainingData instance by aligning x,y coordinates to the first frame.
        Modifies positions in-place to save memory.
        Returns self for method chaining.
        """
        # Get initial x,y position offset from visible joints in first frame
        if self.visible_joints_mask is not None:
            # Use mean of visible joints as reference point
            # Get first frame joints and mask
            first_frame_joints = self.joints_wrt_world[..., 0, :, :]  # [*batch, 22, 3] 
            first_frame_mask = self.visible_joints_mask[..., 0, :]  # [*batch, 22]
            
            # Select only visible joints
            visible_joints = first_frame_joints[first_frame_mask]  # [num_visible, 3]
            initial_xy = visible_joints[..., :2].mean(dim=0)  # [2]
        else:
            # If no visibility mask, use mean of all joints in first frame
            initial_xy = self.joints_wrt_world[..., 0, :, :2].mean(dim=-2)  # [*batch, 2]
        
        # Store initial offset 
        self.initial_xy = initial_xy
        assert isinstance(self.initial_xy, torch.Tensor) and self.initial_xy.shape[-1] == 2

        # Modify positions in-place by subtracting x,y offset
        # Expand initial_xy to match broadcast dimensions
        expanded_xy = initial_xy.view(*initial_xy.shape[:-1], 1, 1, 2)  # Add dims for broadcasting
        
        self.T_world_root[..., 4:6].sub_(initial_xy) # [*batch, timesteps, 2]
        self.joints_wrt_world[..., :2].sub_(expanded_xy) # [*batch, timesteps, 22, 2]
        self.T_world_cpf[..., 4:6].sub_(initial_xy) # [*batch, timesteps, 2]

        return self
