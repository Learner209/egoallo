from pathlib import Path
from typing import TYPE_CHECKING
from typing import Self
import dataclasses
from dataclasses import dataclass
import numpy as np
import torch.utils.data
from jaxtyping import Bool, Float
from egoallo.type_stubs import EgoTrainingDataType
from egoallo.transforms import SO3, SE3
from torch import Tensor
from typing import Generator

from egoallo.type_stubs import DenoiseTrajTypeLiteral


if TYPE_CHECKING:
    from egoallo.type_stubs import DenoiseTrajType
    from egoallo.type_stubs import DatasetType

from .. import transforms as tf
from ..tensor_dataclass import TensorDataclass
from typing import Optional, TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from egoallo.type_stubs import DenoiseTrajType
# from ..viz.smpl_viewer import SMPLViewer
from ..viz.smpl_pyrender_viewer import SMPLViewer

from egoallo.setup_logger import setup_logger
import typeguard
from jaxtyping import jaxtyped
from egoallo.viz.hybrik_twist_angle_visualizer import InteractiveSMPLViewer

logger = setup_logger(output=None, name=__name__)


@jaxtyped(typechecker=typeguard.typechecked)
class EgoTrainingDataAADecomp(TensorDataclass):
    """Dictionary of tensors we use for EgoAllo training with AA decomposition."""

    # NOTE: if the attr is tensor/np.ndarray type, then it must has a leading batch dimension, whether it can be broadcasted or not.
    # NOTE: since the `tensor_dataclass` will convert the tensor to a single element tensor, we need to make sure the leading dimension is always there.

    T_world_root: Float[Tensor, "*batch timesteps 7"]
    """Transformation from the world frame to the root frame at each timestep."""

    contacts: Float[Tensor, "*batch timesteps 52"]
    """Contact boolean for each joint."""

    betas: Float[Tensor, "*batch 1 10"]
    """Body shape parameters. Default to 10 when using smplx model."""

    joints_wrt_world: Float[Tensor, "*batch timesteps 24 3"]
    """Joint positions relative to the world frame."""

    body_quats: Float[Tensor, "*batch timesteps 21 4"]
    """Local orientations for each body joint."""

    height_from_floor: Float[Tensor, "*batch timesteps 1"]
    """Distance from CPF to floor at each timestep."""

    mask: Bool[Tensor, "*batch timesteps"]
    """Mask to support variable-length sequence."""

    hand_quats: Float[Tensor, "*batch timesteps 30 4"] | None
    """Local orientations for each hand joint."""

    visible_joints_mask: Bool[Tensor, "*batch timesteps 24"]
    """Boolean mask indicating which joints are visible (not masked)"""

    body_twists: Float[Tensor, "*batch timesteps 23 1"]
    """Twist parameters for body joints."""

    @dataclass
    class MetaData:
        """Metadata about the trajectory."""

        smpl_family_model_basedir: Path
        """Base directory of the smpl family model."""

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

        original_invalid_joints: Optional[Float[Tensor, "*batch timesteps 24 3"]] = None
        """Original values of invalid joints before zeroing"""

        aux_joints_wrt_world_placeholder: Optional[
            Float[Tensor, "*batch timesteps 24 3"]
        ] = None
        """Placeholder for auxiliary joints, used in EgoExoDataset helper."""

        aux_visible_joints_mask_placeholder: Optional[
            Float[Tensor, "*batch timesteps 24"]
        ] = None
        """Placeholder for auxiliary joints, used in EgoExoDataset helper."""

        dataset_type: "DatasetType" = "AdaptiveAmassHdf5Dataset"
        """Type of dataset the trajectory belongs to."""

        rotate_radian: Optional[Float[Tensor, "1"]] = None
        """Rotation radian for trajectory augmentation."""

    metadata: MetaData = dataclasses.field(default_factory=MetaData)
    """Metadata about the trajectory."""

    def to_denoise_traj(
        self,
        denoising_mode: "DenoiseTrajTypeLiteral",
        include_hands: bool = True,
        smpl_family_model_basedir: Path = None,
    ) -> "DenoiseTrajType":
        """Convert EgoTrainingDataAADecomp to appropriate DenoiseTraj based on denoising mode.

        This method implements the conversion logic from EgoTrainingDataAADecomp to various
        DenoiseTraj subclasses based on the specified denoising mode.

        Args:
            denoising_mode: The denoising mode to determine which trajectory type to create
            include_hands: Whether to include hand data in the output trajectory

        Returns:
            Appropriate trajectory object based on denoising mode
        """
        assert denoising_mode == "AbsoluteDenoiseTrajAADecomp"
        from egoallo.denoising import AbsoluteDenoiseTrajAADecomp

        # Extract rotation and translation from T_world_root
        *batch, time, _ = self.T_world_root.shape
        _R_world_root = SO3(self.T_world_root[..., :4]).as_matrix()
        _t_world_root = self.T_world_root[..., 4:7]

        # Handle hand data if present and requested
        hand_rotmats = None
        if self.hand_quats is not None and include_hands:
            hand_rotmats = SO3(self.hand_quats).as_matrix()

        # For absolute mode, create AbsoluteDenoiseTraj
        cos_sin_phis = torch.cat(
            [torch.cos(self.body_twists), torch.sin(self.body_twists)],
            dim=-1,
        )
        return AbsoluteDenoiseTrajAADecomp(
            betas=self.betas.expand((*batch, time, -1)),
            cos_sin_phis=cos_sin_phis,
            contacts=self.contacts[..., :24],
            hand_rotmats=hand_rotmats,
            joints_wrt_world=self.joints_wrt_world,
            visible_joints_mask=self.visible_joints_mask,
            metadata=EgoTrainingDataAADecomp.MetaData(
                take_name=self.metadata.take_name,
                frame_keys=self.metadata.frame_keys,
                scope=self.metadata.scope,
                stage="raw",
                smpl_family_model_basedir=self.metadata.smpl_family_model_basedir,
            ),
        )

    @staticmethod
    def load_from_npz(
        smpl_family_model_dir: Path,
        data_path: Path,
        include_hands: bool,
        device: torch.device,
    ) -> Generator[tuple["EgoTrainingDataType", tuple[int, int]], None, None]:
        """Load a single trajectory from a (processed_30fps) npz file."""
        # Import needed modules
        import sys
        import os
        from pathlib import Path

        # Add project root to path to help with imports
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../.."),
        )
        if project_root not in sys.path:
            sys.path.append(project_root)

        gender = np.load(data_path, allow_pickle=True)["gender"].item()
        assert gender in ["male", "female", "neutral"]

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
            padding = torch.zeros(
                *betas.shape[:-1],
                6,
                dtype=betas.dtype,
                device=betas.device,
            )
            betas = torch.cat([betas, padding], dim=-1)
        assert betas.shape == (1, 16), (
            f"Expected betas shape (1, 16), got {betas.shape}"
        )

        assert raw_fields["root_orient"].shape == (timesteps, 3)
        assert raw_fields["pose_body"].shape == (timesteps, 63)
        assert raw_fields["pose_hand"].shape == (timesteps, 90)
        assert raw_fields["contacts"].shape == (timesteps, 52)
        assert raw_fields["joints"].shape == (timesteps, 52, 3)

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

            body_twists = None
            # Get twists using forward_get_twist - use a try-except block to handle potential import/initialization issues
            smpl_aadecomp_model = None
            smplx_output = None

            from egoallo.constants import (
                SmplFamilyMetaModelZoo,
                SmplFamilyMetaModelName,
            )

            assert SmplFamilyMetaModelName == "SmplModelAADecomp"
            smpl_aadecomp_model = (
                SmplFamilyMetaModelZoo[SmplFamilyMetaModelName]
                .load(smpl_family_model_dir, gender=gender, num_joints=24)
                .to(device)
            )

            # smplx_aadecomp_model = SmplxModelAADecomp.load(
            #     smpl_family_model_dir, gender=gender,
            # ).to(device)

            # Convert data to format expected by SMPLXLayer
            global_orient = (
                raw_fields["root_orient"][i:end_idx]
                .reshape(batch_size, 3)
                .clone()
                .to(device)
            )
            body_pose = (
                raw_fields["pose_body"][i:end_idx]
                .reshape(batch_size, 21, 3)
                .clone()
                .to(device)
            )

            # left_hand_pose = (
            #     raw_fields["pose_hand"][i:end_idx, :45]
            #     .reshape(batch_size, 15, 3)
            #     .clone()
            #     .to(device)
            # )
            # right_hand_pose = (
            #     raw_fields["pose_hand"][i:end_idx, 45:]
            #     .reshape(batch_size, 15, 3)
            #     .clone()
            #     .to(device)
            # )

            transl = (
                raw_fields["trans"][i:end_idx].reshape(batch_size, 3).clone().to(device)
            )
            betas_batch = betas.repeat(batch_size, 1).clone().to(device)

            # test on `SmplxModelAADecomp` class first, which is the original impl. of forward_get_twist func.
            # body_twists = smplx_aadecomp_model.model.forward_get_twist(
            #     betas=betas_batch[..., :11],  # type: ignore
            #     global_orient=SO3.exp(global_orient).as_matrix().reshape(batch_size, 1, 3, 3),  # type: ignore
            #     body_pose=SO3.exp(body_pose).as_matrix().reshape(batch_size, 21, 3, 3),  # type: ignore
            #     left_hand_pose=SO3.exp(left_hand_pose).as_matrix().reshape(batch_size, 15, 3, 3),  # type: ignore
            #     right_hand_pose=SO3.exp(right_hand_pose).as_matrix().reshape(batch_size, 15, 3, 3),  # type: ignore
            #     transl=transl,  # type: ignore
            #     expression=None,
            #     jaw_pose=None,
            #     leye_pose=None,
            #     reye_pose=None,
            #     full_pose=None,
            # )

            smpl_body_pose = torch.cat(
                [
                    SO3.exp(body_pose).as_matrix().reshape(batch_size, 21, 3, 3),
                    torch.eye(3).unsqueeze(0).repeat(batch_size, 2, 1, 1).to(device),
                ],
                dim=1,
            )
            # smpl_body_pose[..., 21, :3, :3] = SO3.exp(left_hand_pose[..., 0, :3]).as_matrix().reshape(batch_size, 3, 3)
            # smpl_body_pose[..., 22, :3, :3] = SO3.exp(right_hand_pose[..., 0, :3]).as_matrix().reshape(batch_size, 3, 3)

            body_twists = smpl_aadecomp_model.model.forward_get_twist(
                betas=betas_batch[..., :10],  # type: ignore
                global_orient=SO3.exp(global_orient)
                .as_matrix()
                .reshape(batch_size, 1, 3, 3),  # type: ignore
                body_pose=smpl_body_pose,  # type: ignore
            )

            # test on `forward_simple_with_pose_decomposed` func.
            # smplx_output = smplx_aadecomp_model.model.forward_simple_with_pose_decomposed(
            #     betas=betas_batch[..., :11],  # type: ignore
            #     global_orient=SO3.exp(global_orient).as_matrix().reshape(batch_size, 1, 3, 3),  # type: ignore
            #     body_pose=SO3.exp(body_pose).as_matrix().reshape(batch_size, 21, 3, 3),  # type: ignore
            #     left_hand_pose=SO3.exp(left_hand_pose).as_matrix().reshape(batch_size, 15, 3, 3),  # type: ignore
            #     right_hand_pose=SO3.exp(right_hand_pose).as_matrix().reshape(batch_size, 15, 3, 3),  # type: ignore
            #     transl=transl,  # type: ignore
            #     expression=None,
            #     jaw_pose=None,
            #     leye_pose=None,
            #     reye_pose=None,
            #     return_verts=True,
            #     use_pose_mean=False
            # )

            # We need extended joint positions for HybrIK smpl integration.
            smpl_jnts = raw_fields["joints"][..., :24, :]

            cos_sin_phis = torch.cat(
                [torch.cos(body_twists), torch.sin(body_twists)],
                dim=-1,
            )

            logger.info("Creating InteractiveSMPLViewer instance...")

            ps_vis = False
            if ps_vis:
                ind = 25
                viewer = InteractiveSMPLViewer(
                    smpl_aadecomp_model=smpl_aadecomp_model,
                    pose_skeleton=smpl_jnts[ind] * 1,
                    betas=betas[0, :10],
                    transl=transl[ind],
                    initial_phis=cos_sin_phis[ind],
                    # global_orient=SO3.exp(global_orient[ind]).as_matrix().reshape(3, 3),
                    global_orient=None,
                    device=device,
                    num_hybrik_joints=24,  # Standard for SMPL output from hybrik
                    leaf_thetas=None,
                    coordinate_transform=True,
                )
                viewer.show()

            # Run hybrik function
            smpl_model_output = smpl_aadecomp_model.model.hybrik(
                betas=betas_batch[..., :10],  # type: ignore
                pose_skeleton=smpl_jnts,  # type: ignore
                phis=cos_sin_phis,  # type: ignore
                transl=transl,  # type: ignore
                # global_orient=SO3.exp(global_orient).as_matrix().reshape(batch_size, 3, 3), # Setting global orient to None as `batch_get_pelvis_orient` or `batch_get_pelvis_orient_svd` would guess it from pose_skeletons.
            )
            # breakpoint()

            # let take_name be the stem of data_path suffixed with window start and end
            take_name = (
                Path(data_path).stem + f"_window_{i}_{end_idx}" + Path(data_path).suffix
            )

            # Temporary fix: Update T_world_root to align with pose_skeleton.
            T_world_root[i:end_idx, ..., 4:7] = smpl_model_output.joints[
                i:end_idx,
                ...,
                0,
                :,
            ]

            # Create the EgoTrainingDataAADecomp instance
            data_dict = {
                "T_world_root": T_world_root[i:end_idx],
                "contacts": raw_fields["contacts"][i:end_idx],
                "betas": raw_fields["betas"][..., :10].unsqueeze(0),
                "joints_wrt_world": smpl_model_output.joints[i:end_idx],
                "body_quats": body_quats[i:end_idx],
                "height_from_floor": T_world_root[i:end_idx, 6:7],
                "mask": torch.ones((end_idx - i,), dtype=torch.bool),
                "hand_quats": hand_quats[i:end_idx] if include_hands else None,
                "visible_joints_mask": torch.ones_like(
                    smpl_model_output.joints[i:end_idx, ..., 0],
                    dtype=torch.bool,
                ),
                "body_twists": body_twists,
                "metadata": EgoTrainingDataAADecomp.MetaData(
                    take_name=(take_name,),
                    frame_keys=tuple(),
                    stage="raw",
                    scope="test",
                    smpl_family_model_basedir=smpl_family_model_dir,
                ),
            }

            ego_data = EgoTrainingDataAADecomp(**data_dict)

            # Yield the created instance and the window indices
            yield (ego_data, (i, end_idx))

            del body_twists
            if smpl_aadecomp_model is not None:
                del smpl_aadecomp_model
            if smplx_output is not None:
                del smplx_output

            import gc

            gc.collect()

    @staticmethod
    def visualize_ego_training_data(
        data: "DenoiseTrajType",
        smpl_family_model_basedir: Path | None = None,
        output_path: str = "output.mp4",
        online_render: bool = False,
        **kwargs,
    ):
        viewer = SMPLViewer(**kwargs)
        viewer.render_sequence(
            data,
            smpl_family_model_basedir,
            output_path,
            online_render=online_render,
        )

    @jaxtyped(typechecker=typeguard.typechecked)
    def preprocess(self, _rotate_radian: None | Tensor = None) -> "EgoTrainingDataType":
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
        )  # Add dims for broadcasting

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

        self.joints_wrt_world = torch.cat(
            [
                self.joints_wrt_world[..., :2],
                self.joints_wrt_world[..., 2:3]
                - self.height_from_floor.unsqueeze(-2),  # [*batch, timesteps, 22, 1]
                self.joints_wrt_world[..., 3:],
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

    @jaxtyped(typechecker=typeguard.typechecked)
    def postprocess(self) -> "EgoTrainingDataType":
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
        # Add initial x,y position offset
        # Expand initial_xy to match broadcast dimensions like in preprocess()
        expanded_xy = self.metadata.initial_xy.view(
            *self.metadata.initial_xy.shape[:-1],
            1,
            1,
            2,
        )  # Add dims for broadcasting

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

        self.metadata.stage = "postprocessed"

        return self

    def _post_process(self, traj: "DenoiseTrajType") -> "DenoiseTrajType":
        """
        No-op.
        """
        return traj

    def _set_traj(self, traj: "DenoiseTrajType") -> "DenoiseTrajType":
        """
        Set the trajectory for postprocessing.
        Set the joints_wrt_world and visible_joints_mask.
        Set the metadata.
        """
        # assert traj.joints_wrt_world is None and traj.visible_joints_mask is None, (
        #     "joints_wrt_world and visible_joints_mask should be None for postprocessing."
        # )
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
