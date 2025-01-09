from pathlib import Path
from typing import Union, assert_never, TYPE_CHECKING
import dataclasses
import numpy as np
import torch
import torch.utils.data
from torch.utils.data.dataloader import _InfiniteConstantSampler
import typeguard
from jaxtyping import Bool, Float, jaxtyped, Array
from egoallo.transforms import SO3, SE3
from torch import Tensor

if TYPE_CHECKING:
    from egoallo.types import DenoiseTrajType

from .. import fncsmpl, fncsmpl_extensions
from .. import transforms as tf
from ..tensor_dataclass import TensorDataclass
from typing import Optional, TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from ..network import EgoDenoiserConfig
    from ..network import AbsoluteDenoiseTraj
    from egoallo.types import DenoiseTrajType
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

    joints_wrt_world: Float[Tensor, "*batch timesteps 22 3"]
    """Joint positions relative to the world frame."""

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

    @dataclass
    class MetaData:
        """Metadata about the trajectory."""
        take_name: str = ""
        """Name of the take."""

        frame_keys: tuple[int, ...] = ()
        """Keys of the frames in the npz file."""

        initial_xy: Float[Tensor, "*batch 2"] = torch.FloatTensor([0.0, 0.0])
        """Initial x,y position offset from visible joints in first frame"""

        stage: Literal["raw", "preprocessed", "postprocessed"] = "raw"
        """Processing stage of the data: 'raw' (before preprocessing), 'preprocessed' (between pre/post), or 'postprocessed' (after postprocessing)."""

        scope: Literal["train", "test"] = "train"
        """Scope of the data: 'train' or 'test'."""

    metadata: MetaData = dataclasses.field(default_factory=MetaData)
    """Metadata about the trajectory."""

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

        joints_wrt_world = raw_fields["joints"]

        # Align T_world_cpf (only x,y translation component)
        T_world_cpf = (
            tf.SE3(posed.Ts_world_joint[:, 14, :])  # T_world_head
            @ tf.SE3(fncsmpl_extensions.get_T_head_cpf(shaped))
        ).parameters()

		# METADATA can be omittted and set as default param.
        return EgoTrainingData(
            T_world_root=T_world_root.cpu(),
            contacts=raw_fields["contacts"][:, :22].cpu(),  # root is included.
            betas=raw_fields["betas"].unsqueeze(0).cpu(),
            joints_wrt_world=joints_wrt_world.cpu(),  # root is included.
            body_quats=body_quats.cpu(),
            # CPF frame stuff.
            T_world_cpf=T_world_cpf.cpu(),
            height_from_floor=T_world_cpf[:, 6:7].cpu(),
            joints_wrt_cpf=(
                # unsqueeze so both shapes are (timesteps, joints, dim)
                tf.SE3(T_world_cpf[:, None, :]).inverse()
                @ joints_wrt_world.to(T_world_cpf.device)
            ).cpu(),
            mask=torch.ones((timesteps,), dtype=torch.bool),
            hand_quats=hand_quats.cpu() if include_hands else None,
            visible_joints_mask=None,
            metadata=EgoTrainingData.MetaData( # default metadata.
                take_name=path.name,
                frame_keys=tuple(),  # Convert to tuple of ints
                stage="raw",
                scope="test",
            ),
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
    def preprocess(self) -> "EgoTrainingData":
        """
        Modifies the current EgoTrainingData instance by:
        1. Aligning x,y coordinates to the first frame
        2. Subtracting floor height from z coordinates
        Modifies positions in-place to save memory.
        Returns self for method chaining.
        """
        assert self.metadata.stage == "raw"
        # Get initial preprocessed x,y position offset from visible joints in first frame
        if self.visible_joints_mask is not None:
            # Find first frame with at least one visible joint
            *B, T, J, _ = self.joints_wrt_world.shape  # Get temporal dimension
            for t in range(T):
                frame_joints = self.joints_wrt_world[..., t, :, :]  # [*batch, 22, 3]
                frame_mask = self.visible_joints_mask[..., t, :]  # [*batch, 22]
                visible_joints = frame_joints[frame_mask]  # [num_visible, 3]
                
                if len(visible_joints) > 0:  # At least one joint is visible
                    initial_xy = visible_joints[..., :2].mean(dim=0)  # [2]
                    break
            else:
                raise RuntimeError("No frames found with visible joints")
        else:
            # raise RuntimeWarning("No visibility mask found, using mean of all joints in first frame")
            # If no visibility mask, use mean of all joints in first frame
            initial_xy = self.joints_wrt_world[..., 0, :, :2].mean(dim=-2)  # [*batch, 2]
        
        # Store initial offset 
        self.metadata.initial_xy = initial_xy
        assert isinstance(self.metadata.initial_xy, torch.Tensor) and self.metadata.initial_xy.shape[-1] == 2

        # Modify positions in-place by subtracting x,y offset
        # Expand initial_xy to match broadcast dimensions
        expanded_xy = initial_xy.view(*initial_xy.shape[:-1], 1, 1, 2)  # Add dims for broadcasting
        
        self.T_world_root[..., 4:6].sub_(initial_xy) # [*batch, timesteps, 2]
        self.joints_wrt_world[..., :2].sub_(expanded_xy) # [*batch, timesteps, 22, 2]
        self.T_world_cpf[..., 4:6].sub_(initial_xy) # [*batch, timesteps, 2]

        # Subtract floor height using existing height_from_floor attribute
        self.joints_wrt_world[..., :, :, 2:3].sub_(self.height_from_floor.unsqueeze(-2)) # [*batch, timesteps, 22, 1]
        self.T_world_root[..., 6:7].sub_(self.height_from_floor) # [*batch, timesteps, 1]
        self.T_world_cpf[..., 6:7].sub_(self.height_from_floor) # [*batch, timesteps, 1]

        self.metadata.stage = "preprocessed"

        return self

    @jaxtyped(typechecker=typeguard.typechecked)
    def postprocess(self) -> "EgoTrainingData":
        """
        Modifies the current EgoTrainingData instance by:
        1. Adding floor height to z coordinates
        """
        assert self.metadata.stage == "preprocessed"
        self.joints_wrt_world[..., :, :, 2:3].add_(self.height_from_floor.unsqueeze(-2)) # [*batch, timesteps, 22, 1]
        self.T_world_root[..., :, 6:7].add_(self.height_from_floor) # [*batch, timesteps, 1]
        self.T_world_cpf[..., :, 6:7].add_(self.height_from_floor) # [*batch, timesteps, 1]
        # Add initial x,y position offset
        # Expand initial_xy to match broadcast dimensions like in preprocess()
        expanded_xy = self.metadata.initial_xy.view(*self.metadata.initial_xy.shape[:-1], 1, 1, 2)  # Add dims for broadcasting
        
        device = self.T_world_root.device
        self.T_world_root[..., 4:6].add_(self.metadata.initial_xy.unsqueeze(-2).to(device)) # [*batch, timesteps, 2]
        self.joints_wrt_world[..., :2].add_(expanded_xy.to(device)) # [*batch, timesteps, 22, 2]
        self.T_world_cpf[..., 4:6].add_(self.metadata.initial_xy.unsqueeze(-2).to(device)) # [*batch, timesteps, 2]

        self.metadata.stage = "postprocessed"

        return self

    def _post_process(self, traj: "DenoiseTrajType") -> "DenoiseTrajType":
        from egoallo.network import AbsoluteDenoiseTraj
        assert self.metadata.stage == "postprocessed"
        assert traj.metadata.stage == "raw", "Only raw data is supported for postprocessing."
        assert isinstance(traj, AbsoluteDenoiseTraj), "Only AbsoluteDenoiseTraj is supported for postprocessing."
        # postprocess the DenoiseTrajType

        # 1. t_world_root, t_world_cpf
        device = traj.t_world_root.device
        traj.t_world_root[..., :, 2:3].add_(self.height_from_floor.to(device)) # [*batch, timesteps, 3]
        traj.t_world_root[..., :, :2].add_(self.metadata.initial_xy.unsqueeze(-2).to(device)) # [*batch, timesteps, 3]

		# 2. assign joints_wrt_world and visible_joints_mask
        assert traj.joints_wrt_world is None and traj.visible_joints_mask is None, f"joints_wrt_world and visible_joints_mask should be None for postprocessing."
        traj.joints_wrt_world = self.joints_wrt_world.clone()
        if self.visible_joints_mask is not None:
            # assert self.metadata.scope == "train", "visible_joints_mask should only be set for train data."
            traj.visible_joints_mask = self.visible_joints_mask.clone()
        else:
            assert self.metadata.scope == "test", "visible_joints_mask shouldn't be set for test data."
            traj.visible_joints_mask = torch.ones_like(traj.joints_wrt_world, dtype=torch.float)

        # 3. assign metadata
        traj.metadata = self.metadata
        return traj
        
