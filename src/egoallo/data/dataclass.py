from pathlib import Path
from typing import TYPE_CHECKING
from typing import Self
import dataclasses
from dataclasses import dataclass
import numpy as np
import torch.utils.data
from jaxtyping import Bool, Float
from egoallo.transforms import SO3, SE3
from torch import Tensor
from typing import Generator


if TYPE_CHECKING:
    from egoallo.types import DenoiseTrajType
    from egoallo.types import DatasetType

from .. import fncsmpl_library as fncsmpl
from .. import fncsmpl_extensions_library as fncsmpl_extensions
from .. import transforms as tf
from ..tensor_dataclass import TensorDataclass
from typing import Optional, TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from egoallo.types import DenoiseTrajType
# from ..viz.smpl_viewer import SMPLViewer
from ..viz.smpl_pyrender_viewer import SMPLViewer

from egoallo.setup_logger import setup_logger

logger = setup_logger(output=None, name=__name__)


# @jaxtyped(typechecker=typeguard.typechecked)
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

        take_name: tuple[str, ...] | tuple[tuple[str, ...], ...] = ()
        """Name of the take."""

        frame_keys: tuple[int, ...] = ()
        """Keys of the frames in the npz file."""

        initial_xy: Float[Tensor, "*batch 2"] = torch.FloatTensor([0.0, 0.0])
        """Initial x,y position offset from visible joints in first frame"""

        stage: Literal["raw", "preprocessed", "postprocessed"] = "raw"
        """Processing stage of the data: 'raw' (before preprocessing), 'preprocessed' (between pre/post), or 'postprocessed' (after postprocessing)."""

        scope: Literal["train", "test"] = "train"
        """Scope of the data: 'train' or 'test'."""

        original_invalid_joints: Optional[Float[Tensor, "*batch timesteps 22 3"]] = None
        """Original values of invalid joints before zeroing"""

        aux_joints_wrt_world_placeholder: Optional[
            Float[Tensor, "*batch timesteps 22 3"]
        ] = None
        """Placeholder for auxiliary joints, used in EgoExoDataset helper."""

        aux_visible_joints_mask_placeholder: Optional[
            Float[Tensor, "*batch timesteps 22"]
        ] = None
        """Placeholder for auxiliary joints, used in EgoExoDataset helper."""

        dataset_type: "DatasetType" = "AdaptiveAmassHdf5Dataset"
        """Type of dataset the trajectory belongs to."""

        rotate_radian: Optional[Float[Tensor, "1"]] = None
        """Rotation radian for trajectory augmentation."""

    metadata: MetaData = dataclasses.field(default_factory=MetaData)
    """Metadata about the trajectory."""

    @staticmethod
    def load_from_npz(
        smplh_model_path: Path,
        data_path: Path,
        include_hands: bool,
        device: torch.device,
    ) -> Generator[tuple["EgoTrainingData", tuple[int, int]], None, None]:
        """Load a single trajectory from a (processed_30fps) npz file."""
        raw_fields = {
            k: torch.from_numpy(v.astype(np.float32) if v.dtype == np.float64 else v)
            for k, v in np.load(data_path, allow_pickle=True).items()
            if v.dtype in (np.float32, np.float64)
        }

        timesteps = raw_fields["root_orient"].shape[0]

        # preprocessing

        betas = (
            raw_fields["betas"]
            if raw_fields["betas"].ndim == 2
            else raw_fields["betas"][None]
        )
        # If betas is 10-dimensional, pad with zeros to make it 16-dimensional
        if betas.shape[-1] == 10:
            padding = torch.zeros(*betas.shape[:-1], 6, dtype=betas.dtype)
            betas = torch.cat([betas, padding], dim=-1)
        assert betas.shape == (1, 16), (
            f"Expected betas shape (1, 16), got {betas.shape}"
        )

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

        T_world_root = torch.cat(
            [
                tf.SO3.exp(raw_fields["root_orient"]).wxyz,
                raw_fields["trans"],
            ],
            dim=-1,
        ).to(device)

        body_quats = tf.SO3.exp(
            raw_fields["pose_body"].reshape(timesteps, 21, 3),
        ).wxyz.to(device)
        hand_quats = tf.SO3.exp(
            raw_fields["pose_hand"].reshape(timesteps, 30, 3),
        ).wxyz.to(device)

        window_size = 30000
        for i in range(0, timesteps, window_size):
            end_idx = min(i + window_size, timesteps)
            batch_size = end_idx - i
            body_model = fncsmpl.SmplhModel.load(
                smplh_model_path,
                use_pca=False,
                batch_size=batch_size,
            ).to(device)

            shaped_batch = body_model.with_shape(
                raw_fields["betas"].unsqueeze(0).repeat(batch_size, 1).to(device),
            )

            T_world_root_batch = T_world_root[i:end_idx].to(device)
            body_quats_batch = body_quats[i:end_idx].to(device)

            posed_batch = shaped_batch.with_pose_decomposed(
                T_world_root=T_world_root_batch,
                body_quats=body_quats_batch,
            )

            # Align T_world_cpf (only x,y translation component)
            T_world_cpf = (
                tf.SE3(posed_batch.Ts_world_joint[:, 14, :])  # T_world_head
                @ tf.SE3(fncsmpl_extensions.get_T_head_cpf(shaped_batch))
            ).parameters()

            joints_wrt_world = raw_fields["joints"]

            # let take_name be the stem of data_path suffixed with window start and end like: "_window_{start}_{end}" using pathlib functionliaty.
            take_name = (
                Path(data_path).stem + "_window_{start}_{end}" + Path(data_path).suffix
            )

            # METADATA can be omittted and set as default param.
            yield (
                EgoTrainingData(
                    T_world_root=T_world_root[i:end_idx],
                    contacts=raw_fields["contacts"][
                        i:end_idx,
                        :22,
                    ],  # root is included.
                    betas=raw_fields["betas"].unsqueeze(0),
                    joints_wrt_world=joints_wrt_world[
                        i:end_idx,
                        :,
                    ],  # root is included.
                    body_quats=body_quats[i:end_idx],
                    # CPF frame stuff.
                    T_world_cpf=T_world_cpf,
                    height_from_floor=T_world_cpf[:, 6:7],
                    joints_wrt_cpf=(
                        # unsqueeze so both shapes are (timesteps, joints, dim)
                        tf.SE3(T_world_cpf[:, None, :]).inverse()
                        @ joints_wrt_world[i:end_idx].to(T_world_cpf.device)
                    ),
                    mask=torch.ones((end_idx - i,), dtype=torch.bool),
                    hand_quats=hand_quats[i:end_idx] if include_hands else None,
                    visible_joints_mask=None,
                    metadata=EgoTrainingData.MetaData(  # default metadata.
                        take_name=(take_name,),
                        frame_keys=tuple(),  # Convert to tuple of ints
                        stage="raw",
                        scope="test",
                    ),
                ),
                (i, end_idx),
            )

            del shaped_batch
            del posed_batch
            del body_model
            del T_world_root_batch
            del body_quats_batch
            del T_world_cpf

            import gc

            gc.collect()

    @staticmethod
    def visualize_ego_training_data(
        data: "DenoiseTrajType",
        smplh_model_path: Path = Path(
            "assets/smpl_based_model/smplh/SMPLH_NEUTRAL.pkl",
        ),
        output_path: str = "output.mp4",
        **kwargs,
    ):
        viewer = SMPLViewer(**kwargs)
        viewer.render_sequence(data, smplh_model_path, output_path, online_render=False)

    # @jaxtyped(typechecker=typeguard.typechecked)
    def preprocess(self, _rotate_radian: None | Tensor = None) -> "EgoTrainingData":
        """
        Modifies the current EgoTrainingData instance by:
        1. Aligning x,y coordinates to the first frame
        2. Subtracting floor height from z coordinates
        Modifies positions in-place to save memory.
        Returns self for method chaining.
        3. Set where joints is invalid to all zeros, indicated by visible_joints_mask.
        """
        assert self.metadata.stage == "raw"
        # Get initial preprocessed x,y position offset from visible joints in first frame
        # FIXME: there is chance that after exerting temporal_dim_mask, thre won't exist a timestep s.t. visible_joints has at least one joint visible for all batch..
        if self.visible_joints_mask is not None:
            # Find first frame with at least one visible joint
            *B, T, J, _ = self.joints_wrt_world.shape  # Get temporal dimension
            for t in range(T):
                frame_joints = self.joints_wrt_world[..., t, :, :]  # [*batch, 22, 3]
                frame_mask = self.visible_joints_mask[..., t, :]  # [*batch, 22]

                # Expand frame_mask to match batch dimensions
                frame_mask = frame_mask.view(*B, -1)  # [*batch, 22]

                # Get visible joints while preserving batch dimensions
                visible_joints_mask = frame_mask.unsqueeze(-1).expand(
                    *B,
                    -1,
                    3,
                )  # [*batch, 22, 3]
                visible_joints = torch.where(
                    visible_joints_mask,
                    frame_joints,
                    torch.zeros_like(frame_joints),
                )

                # Check if any joints are visible in each batch element
                has_visible = frame_mask.any(dim=-1)  # [*batch]

                if has_visible.all():  # all batch elements have visible joints
                    # Calculate mean only over visible joints, preserving batch dims
                    sums = visible_joints.sum(dim=-2)  # [*batch, 3]
                    counts = frame_mask.sum(dim=-1, keepdim=True)  # [*batch, 1]
                    initial_xy = (sums[..., :2] / counts).clone()  # [*batch, 2]
                    break
            else:
                raise RuntimeError("No frames found with visible joints")
        else:
            # raise RuntimeWarning("No visibility mask found, using mean of all joints in first frame")
            # If no visibility mask, use mean of all joints in first frame
            initial_xy = self.joints_wrt_world[..., 0, :, :2].mean(
                dim=-2,
            )  # [*batch, 2]

        # Store initial offset
        self.metadata.initial_xy = initial_xy
        assert (
            isinstance(self.metadata.initial_xy, torch.Tensor)
            and self.metadata.initial_xy.shape[-1] == 2
            and not torch.isnan(self.metadata.initial_xy).any()
        )

        # Modify positions in-place by subtracting x,y offset
        # Expand initial_xy to match broadcast dimensions
        expanded_xy = initial_xy.view(
            *initial_xy.shape[:-1],
            1,
            1,
            2,
        ).clone()  # Add dims for broadcasting

        # FIXME: the in-place operations just won't work, indictaed by the increasing loss and finally nan values.
        # FIXME: and the problem only occurs at the in-place operations with self.T_world_root, not others?
        # self.T_world_root[..., 4:6].sub_(initial_xy) # [*batch, timesteps, 2]
        # self.joints_wrt_world[..., :2].sub_(expanded_xy) # [*batch, timesteps, 22, 2]
        # self.T_world_cpf[..., 4:6].sub_(initial_xy) # [*batch, timesteps, 2]

        self.T_world_root = torch.cat(
            [
                self.T_world_root[..., :4],
                self.T_world_root[..., 4:6] - initial_xy,
                self.T_world_root[..., 6:],
            ],
            dim=-1,
        )
        self.joints_wrt_world = torch.cat(
            [
                self.joints_wrt_world[..., :2] - expanded_xy,
                self.joints_wrt_world[..., 2:],
            ],
            dim=-1,
        )
        self.T_world_cpf = torch.cat(
            [
                self.T_world_cpf[..., :4],
                self.T_world_cpf[..., 4:6] - initial_xy,
                self.T_world_cpf[..., 6:],
            ],
            dim=-1,
        )

        # Subtract floor height using existing height_from_floor attribute
        # self.joints_wrt_world[..., :, :, 2:3].sub_(self.height_from_floor.unsqueeze(-2)) # [*batch, timesteps, 22, 1]
        # self.T_world_root[..., 6:7].sub_(self.height_from_floor) # [*batch, timesteps, 1]
        # self.T_world_cpf[..., 6:7].sub_(self.height_from_floor) # [*batch, timesteps, 1]

        self.joints_wrt_world = torch.cat(
            [
                self.joints_wrt_world[..., :2],
                self.joints_wrt_world[..., 2:3]
                - self.height_from_floor.unsqueeze(-2),  # [*batch, timesteps, 22, 1]
            ],
            dim=-1,
        )
        self.T_world_root = torch.cat(
            [
                self.T_world_root[..., :6],
                self.T_world_root[..., 6:7] - self.height_from_floor,
                self.T_world_root[..., 7:],
            ],
            dim=-1,
        )
        self.T_world_cpf = torch.cat(
            [
                self.T_world_cpf[..., :6],
                self.T_world_cpf[..., 6:7] - self.height_from_floor,
                self.T_world_cpf[..., 7:],
            ],
            dim=-1,
        )

        if _rotate_radian is not None:
            self._rotate(_rotate_radian)
            self.metadata.rotate_radian = _rotate_radian

        if self.visible_joints_mask is not None:
            # Store original values of invalid joints before zeroing
            self.metadata.original_invalid_joints = torch.where(
                ~self.visible_joints_mask.unsqueeze(-1),
                self.joints_wrt_world,
                torch.zeros_like(self.joints_wrt_world),
            )
            # Set where joints are invalid to all -1.
            self.joints_wrt_world = torch.where(
                self.visible_joints_mask.unsqueeze(-1),
                self.joints_wrt_world,
                torch.ones_like(self.joints_wrt_world) * -1,
            )

        self.metadata.stage = "preprocessed"

        return self

    # @jaxtyped(typechecker=typeguard.typechecked)
    def postprocess(self) -> "EgoTrainingData":
        """
        Modifies the current EgoTrainingData instance by:
        """
        assert self.metadata.stage == "preprocessed"
        device = self.T_world_root.device
        dtype = self.T_world_root.dtype

        # Restore original values of invalid joints if they exist.
        if (
            self.metadata.original_invalid_joints is not None
            and self.visible_joints_mask is not None
        ):
            self.joints_wrt_world = torch.where(
                self.visible_joints_mask.unsqueeze(-1),
                self.joints_wrt_world,
                self.metadata.original_invalid_joints.to(device),
            )
            self.metadata.original_invalid_joints = None  # Clear stored values

        if self.metadata.rotate_radian is not None:
            # rad = SO3(self.metadata.rotate_radian.to(dtype=dtype, device=device)).inverse().
            self._rotate(
                self.metadata.rotate_radian.to(dtype=dtype, device=device) * -1,
            )

        # self.joints_wrt_world[..., :, :, 2:3].add_(self.height_from_floor.unsqueeze(-2)) # [*batch, timesteps, 22, 1]
        # self.T_world_root[..., :, 6:7].add_(self.height_from_floor) # [*batch, timesteps, 1]
        # self.T_world_cpf[..., :, 6:7].add_(self.height_from_floor) # [*batch, timesteps, 1]

        self.joints_wrt_world = torch.cat(
            [
                self.joints_wrt_world[..., :2],
                self.joints_wrt_world[..., 2:3]
                + self.height_from_floor.unsqueeze(-2),  # [*batch, timesteps, 22, 1]
                self.joints_wrt_world[..., 3:],
            ],
            dim=-1,
        )
        self.T_world_root = torch.cat(
            [
                self.T_world_root[..., :6],
                self.T_world_root[..., 6:7] + self.height_from_floor,
                self.T_world_root[..., 7:],
            ],
            dim=-1,
        )
        self.T_world_cpf = torch.cat(
            [
                self.T_world_cpf[..., :6],
                self.T_world_cpf[..., 6:7] + self.height_from_floor,
                self.T_world_cpf[..., 7:],
            ],
            dim=-1,
        )

        # Add initial x,y position offset
        # Expand initial_xy to match broadcast dimensions like in preprocess()
        expanded_xy = self.metadata.initial_xy.view(
            *self.metadata.initial_xy.shape[:-1],
            1,
            1,
            2,
        )  # Add dims for broadcasting

        # self.T_world_root[..., 4:6].add_(self.metadata.initial_xy.unsqueeze(-2).to(device)) # [*batch, timesteps, 2]
        # self.joints_wrt_world[..., :2].add_(expanded_xy.to(device)) # [*batch, timesteps, 22, 2]
        # self.T_world_cpf[..., 4:6].add_(self.metadata.initial_xy.unsqueeze(-2).to(device)) # [*batch, timesteps, 2]

        self.T_world_root = torch.cat(
            [
                self.T_world_root[..., :4],
                self.T_world_root[..., 4:6]
                + self.metadata.initial_xy.unsqueeze(-2).to(device),
                self.T_world_root[..., 6:],
            ],
            dim=-1,
        )

        self.joints_wrt_world = torch.cat(
            [
                self.joints_wrt_world[..., :2] + expanded_xy.to(device),
                self.joints_wrt_world[..., 2:],
            ],
            dim=-1,
        )
        self.T_world_cpf = torch.cat(
            [
                self.T_world_cpf[..., :4],
                self.T_world_cpf[..., 4:6]
                + self.metadata.initial_xy.unsqueeze(-2).to(device),
                self.T_world_cpf[..., 6:],
            ],
            dim=-1,
        )

        self.metadata.stage = "postprocessed"

        return self

    def _post_process(self, traj: "DenoiseTrajType") -> "DenoiseTrajType":
        """
        Postprocess the DenoiseTrajType.
        1. If the traj has been already rotated, rotate back.
        2. Add initial x,y position offset.
        3. Add initial height offset.
        """
        from egoallo.network import AbsoluteDenoiseTraj

        assert self.metadata.stage == "postprocessed"
        assert traj.metadata.stage == "raw", (
            "Only raw data is supported for postprocessing."
        )
        assert isinstance(traj, AbsoluteDenoiseTraj), (
            "Only AbsoluteDenoiseTraj is supported for postprocessing."
        )
        # postprocess the DenoiseTrajType
        device = traj.t_world_root.device

        # NOTE: remember to rotate back traj's root translation since the network operates on the rotated traj.
        if self.metadata.rotate_radian is not None:
            dtype, device = traj.t_world_root.dtype, traj.t_world_root.device

            assert self.metadata.rotate_radian.shape[0] == traj.R_world_root.shape[0]
            assert self.metadata.rotate_radian.shape[0] == traj.t_world_root.shape[0]

            inv_so3_rot = SO3.from_z_radians(
                theta=self.metadata.rotate_radian.to(dtype=dtype, device=device) * -1,
            )
            # Rotate translation
            traj.t_world_root = inv_so3_rot.apply(target=traj.t_world_root)
            # Rotate rotation matrices
            traj.R_world_root = inv_so3_rot.multiply(
                SO3.from_matrix(traj.R_world_root),
            ).as_matrix()

        traj.t_world_root = torch.cat(
            [
                traj.t_world_root[..., :, :2]
                + self.metadata.initial_xy.unsqueeze(-2).to(device),
                traj.t_world_root[..., :, 2:3] + self.height_from_floor.to(device),
                traj.t_world_root[..., :, 3:],
            ],
            dim=-1,
        )
        return traj

    def _set_traj(self, traj: "DenoiseTrajType") -> "DenoiseTrajType":
        """
        Set the trajectory for postprocessing.
        Set the joints_wrt_world and visible_joints_mask.
        Set the metadata.
        """
        assert traj.joints_wrt_world is None and traj.visible_joints_mask is None, (
            "joints_wrt_world and visible_joints_mask should be None for postprocessing."
        )
        traj.joints_wrt_world = self.joints_wrt_world.clone()
        if self.visible_joints_mask is not None:
            # assert self.metadata.scope == "train", "visible_joints_mask should only be set for train data."
            traj.visible_joints_mask = self.visible_joints_mask.clone()
        else:
            assert self.metadata.scope == "test", (
                "visible_joints_mask shouldn't be set for test data."
            )
            traj.visible_joints_mask = torch.ones_like(
                traj.joints_wrt_world[..., 0],
                dtype=torch.float,
            )

        # 3. assign metadata
        traj.metadata = self.metadata
        return traj

    # def __post_init__(self):
    #     """Validate that no tensor attributes contain NaN values."""
    #     for field in dataclasses.fields(self):
    #         # Skip non-tensor fields
    #         if field.name == "metadata":
    #             continue

    #         value = getattr(self, field.name)
    #         if value is not None:  # Handle optional fields
    #             if torch.isnan(value).any():
    #                 raise ValueError(f"NaN values detected in {field.name}")

    def _rotate(self, radian: Tensor) -> Self:
        # assert self.metadata.stage == "preprocessed", "Only preprocessed data is supported for rotation. since preprocessing aligns data's xy to zeros. and rotation is applied only on yaw(rpy zyx convention.)"

        so3_rot = SO3.from_z_radians(radian)
        # 1. rotate T_world_cpf
        self.T_world_cpf = SE3.from_rotation_and_translation(  # [*batch, timesteps, 7]
            rotation=so3_rot.multiply(SO3(self.T_world_cpf[..., :4])),
            translation=so3_rot.apply(self.T_world_cpf[..., 4:]),
        ).parameters()
        # 2. rotate T_world_root
        self.T_world_root = SE3.from_rotation_and_translation(  # [*batch, timesteps, 7]
            rotation=so3_rot.multiply(SO3(self.T_world_root[..., :4])),
            translation=so3_rot.apply(self.T_world_root[..., 4:]),
        ).parameters()
        # 3. rotate joints_wrt_world
        expanded_rot = SO3(wxyz=so3_rot.wxyz.unsqueeze(-2))
        self.joints_wrt_world = expanded_rot.apply(
            self.joints_wrt_world,
        )  # [*batch, timesteps, 22, 3]

        return self

    def __getitem__(self, index) -> Self:
        """Implements native Python slicing for TensorDataclass.

        Supports numpy/torch-style indexing including:
        - Single index: data[0]
        - Multiple indices: data[0,1]
        - Slices: data[0:10]
        - Mixed indexing: data[0, :10, 2:4]
        - Ellipsis: data[..., 0]

        Args:
            index: Index specification. Can be int, slice, tuple, or ellipsis.
            recursive_depth: How deep to recurse into nested structures. -1 means unlimited.

        Returns:
            A new TensorDataclass with sliced data.

        Examples:
            >>> data = TensorDataclass(...)
            >>> # Single index
            >>> first_item = data[0]
            >>> # Multiple indices
            >>> specific_item = data[0, 10]
            >>> # Slice
            >>> first_ten = data[:10]
            >>> # Mixed indexing
            >>> subset = data[0, :10, 2:4]
            >>> # Limit recursion depth
            >>> shallow_slice = data[0, recursive_depth=1]
        """
        # Convert single index to tuple for uniform handling
        if not isinstance(index, tuple):
            index = (index,)

        def _getitem_impl[GetItemT](val: GetItemT, idx: tuple, depth: int) -> GetItemT:
            if depth == 0:
                return val

            if isinstance(val, torch.Tensor):
                try:
                    return val[idx]
                except IndexError as e:
                    raise IndexError(
                        f"Invalid index {idx} for tensor of shape {val.shape}",
                    ) from e
            elif isinstance(val, TensorDataclass):
                # Don't slice betas since it's a per-sequence attribute
                vars_dict = vars(val)
                if "betas" in vars_dict:
                    # Keep original betas tensor
                    vars_dict["betas"] = val.betas
                return type(val)(
                    **{
                        k: _getitem_impl(v, idx, depth - 1) if k != "betas" else v
                        for k, v in vars_dict.items()
                    },
                )
            elif isinstance(val, (list, tuple)):
                return type(val)(_getitem_impl(v, idx, depth - 1) for v in val)
            elif isinstance(val, dict):
                assert type(val) is dict  # No subclass support
                return {k: _getitem_impl(v, idx, depth - 1) for k, v in val.items()}  # type: ignore
            else:
                return val

        # ! Only slicing the highest level of attributes in the dataclass.
        return _getitem_impl(self, index, 2)
