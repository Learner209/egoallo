from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    assert_never,
    cast,
    TYPE_CHECKING,
)

import h5py
import numpy as np
import torch
import torch.utils.data
from torch import Tensor
from jaxtyping import Float, Bool, jaxtyped
from egoallo.config import CONFIG_FILE, make_cfg
from egoallo.data.dataclass import EgoTrainingData
from egoallo.utils.setup_logger import setup_logger
from egoallo.mapping import (
    EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS,
    EGOEXO4D_BODYPOSE_TO_SMPLH_INDICES,
)
import json
import polyscope as ps
from egoallo.utils.aria_utils.ps_vis import draw_coco_kinematic_tree

if TYPE_CHECKING:
    from egoallo.config.inference.inference_egoexo import EgoExoInferenceConfig

local_config_file = CONFIG_FILE
CFG = make_cfg(config_name="defaults", config_file=local_config_file, cli_args=[])
logger = setup_logger(output=None, name=__name__)


class EgoExoDataset(torch.utils.data.Dataset[EgoTrainingData]):
    """Dataset for loading EgoExo bodypose data.

    Loads data in the format specified by bodypose_dataloader and returns
    EgoTrainingData objects compatible with the training pipeline.
    """

    def __init__(
        self,
        config: "EgoExoInferenceConfig",
        cache_files: bool = True,
    ) -> None:
        """Initialize dataset.

        Args:
            config: Training configuration object
            cache_files: Whether to cache loaded data in memory
        """
        self.config = config
        self._slice_strategy = config.dataset_slice_strategy
        self._subseq_len = config.subseq_len
        self._mask_ratio = config.mask_ratio

        # Load annotation files
        self._anno_dirs = config.bodypose_anno_dir
        self._groups = self._initialize_groups()
        # Create mapping from index to take_uid for efficient lookup
        self._group_names = {
            i: take_uid for i, take_uid in enumerate(sorted(self._groups.keys()))
        }

        # Setup caching
        self._cache: Optional[Dict[str, Dict[str, Any]]] = {} if cache_files else None

        # Calculate total length
        if self._slice_strategy == "full_sequence":
            self._length = len(self._groups)
        else:
            total_frames = sum(self._get_sequence_length(g) for g in self._groups)
            self._length = total_frames // self._subseq_len

    def _initialize_groups(self) -> Dict[str, Any]:
        """Initialize dataset groups from annotation directory."""
        groups = {}
        assert (
            self._anno_dirs is not None
        ), "Annotation directories are not set, please check your config."
        for anno_dir in self._anno_dirs:
            for anno_path in anno_dir.rglob("*.json"):
                # Load annotation to check validity
                data = self._load_annotation(anno_path)
                if data is not None and len(data) >= self._subseq_len:
                    groups = {**groups, **data}

        assert len(groups) > 0, f"No valid annotations found in {self._anno_dirs}"
        return groups

    def _load_annotation(self, anno_path: Path) -> Optional[Dict[str, Any]]:
        """Load and validate a single annotation file.

        Args:
            anno_path: Path to annotation JSON file

        Returns:
            Dictionary containing validated annotation data or None if invalid.
            The returned data is organized by frame indices with metadata preserved.
        """
        # try:

        with open(anno_path, "r") as f:
            data = json.load(f)

        # Each file should contain at least one take
        if not data:
            return None

        # Process each take in the file
        processed_data = {}
        for take_uid, take_data in data.items():
            # Each take must have metadata
            if "metadata" not in take_data:
                logger.warning(f"MEATADATA key is not found in the {take_uid}")
                continue

            # Initialize take data with metadata
            metadata = take_data["metadata"]
            processed_data[take_uid] = {
                "metadata": {
                    "take_uid": take_uid,
                    "take_name": metadata.get("take_name"),
                    **metadata,
                }
            }

            # Process frame data
            for frame_idx, frame_data in take_data.items():
                if frame_idx == "metadata":
                    continue

                # Convert frame index to integer
                frame_num = int(frame_idx)

                # Validate required fields for each frame
                required_fields = [
                    "body_3d_world",
                    "body_3d_cam",
                    "body_valid_3d",
                    "ego_camera_intrinsics",
                    "ego_camera_extrinsics",
                ]

                if not all(field in frame_data for field in required_fields):
                    assert False, f"Missing required fields in frame {frame_num} of take {take_uid}"

                processed_data[take_uid][frame_num] = {
                    "body_3d_world": frame_data["body_3d_world"],
                    "body_3d_cam": frame_data["body_3d_cam"],
                    "body_valid_3d": frame_data["body_valid_3d"],
                    "ego_camera_intrinsics": frame_data["ego_camera_intrinsics"],
                    "ego_camera_extrinsics": frame_data["ego_camera_extrinsics"],
                }

            # Ensure we have at least some valid frame data for this take
            if len(processed_data[take_uid]) <= 1:  # Only metadata
                assert (
                    False
                ), f"No valid frame data found for take {take_uid} in {anno_path}"

        return processed_data

        # except Exception as e:
        #     logger.warning(f"Error loading {anno_path}: {e}")
        #     return None

    def _get_sequence_length(self, group: str) -> int:
        """Get length of sequence for a group."""
        data = self._groups[group]
        return len(data) - 1 if data else 0  # exclude the 'metadata' key

    def __len__(self) -> int:
        return len(self._group_names)

    def __getitem__(self, index: int) -> EgoTrainingData:
        """Get a single item from the dataset.

        Args:
            index: Index of item to get

        Returns:
            EgoTrainingData object containing the sequence data
        """
        if self._slice_strategy == "full_sequence":
            group = self._group_names[index]
            data = self._groups[group]
            start_t, end_t = 0, len(data) - 1
        else:
            assert_never(self._slice_strategy)
            # # Find which group and slice based on index
            # group_idx = 0
            # remaining_idx = index
            # while group_idx < len(self._groups):
            #     seq_len = self._get_sequence_length(self._group_names[group_idx])
            #     num_slices = seq_len // self._subseq_len
            #     if remaining_idx < num_slices:
            #         break
            #     remaining_idx -= num_slices
            #     group_idx += 1

            # group = self._group_names[group_idx]
            # data = self._groups[group]
            # start_t = remaining_idx * self._subseq_len
            # end_t = start_t + self._subseq_len
        # Load slice of data
        seq_len = end_t - start_t  # +1 for exclusive end
        frame_keys: tuple[int, ...] = tuple(sorted([k for k in data.keys() if isinstance(k, int)]))
        slice_data = [
            data[frame_keys[t]] for t in range(start_t, end_t)
        ]  # Skip metadata key and get frame data

        # Convert to tensors
        return_smplh_joints = True
        joints_world, joints_cam, visible_mask = self._process_joints(
            slice_data,
            ground_height=data["metadata"]["ground_height"],
            return_smplh_joints=return_smplh_joints,
            num_joints=22 if return_smplh_joints else 17,
            debug_vis=False,
        )
        masked_joints = joints_world.clone()
        masked_joints[~visible_mask] = 0
        # T_world_root = self._process_camera_poses(slice_data)
        take_name = f"name_{data['metadata']['take_name']}_uid_{data['metadata']['take_uid']}_t{start_t}_{end_t}"

        # Create EgoTrainingData object
        ret =  EgoTrainingData(
            joints_wrt_world=masked_joints,
            joints_wrt_cpf=joints_cam,
            T_world_root=torch.zeros((seq_len, 7)),
            T_world_cpf=torch.zeros((seq_len, 7)),
            visible_joints_mask=visible_mask,
            mask=torch.ones(seq_len, dtype=torch.bool),
            take_name=take_name,
            # Add other required fields with appropriate defaults
            betas=torch.zeros((1, 16)),  # Default betas
            body_quats=torch.zeros((seq_len, 21, 4)),  # Default body quaternions
            hand_quats=torch.zeros((seq_len, 30, 4)),  # No hand data
            contacts=torch.zeros((seq_len, 22)),  # Default contacts
            height_from_floor=torch.zeros((seq_len, 1)),  # Default height
            frame_keys=frame_keys, # type: ignore
        )
        ret = ret.align_to_first_frame()
        return ret

    def _process_joints(
        self,
        data: List[Dict[str, Any]],
        ground_height: float,
        return_smplh_joints: bool = True,
        num_joints: int = 22,
        debug_vis: bool = False,
    ) -> Tuple[
        Float[Tensor, "timesteps {num_joints} 3"],
        Float[Tensor, "timesteps {num_joints} 3"],
        Bool[Tensor, "timesteps {num_joints}"],
    ]:
        """Process joint data from annotations.

        Args:
            data: List of frame dictionaries containing body pose data
            return_smplh_joints: If True, converts joints from EgoExo4D (17 joints) to SMPLH format (22 body joints).
                Invalid mappings will be filled with zeros.
            debug_vis: If True, visualize joints using polyscope (for debugging)

        Returns:
            Tuple of:
            - joints_world: World coordinate joint positions (timesteps x J x 3) where J is 17 for EgoExo4D or 22 for SMPLH
            - joints_cam: Camera coordinate joint positions (timesteps x J x 3) where J is 17 for EgoExo4D or 22 for SMPLH
            - visible: Joint visibility mask (timesteps x J) where J is 17 for EgoExo4D or 22 for SMPLH
        """
        joints_world = []
        joints_cam = []
        visible = []

        for frame in data:
            # Convert lists to numpy first to handle NaN values
            world_pos = np.array(frame["body_3d_world"], dtype=np.float32)
            cam_pos = np.array(frame["body_3d_cam"], dtype=np.float32)
            vis = np.array(frame["body_valid_3d"], dtype=bool)

            # Zero out invalid joints
            world_pos[~vis] = np.nan
            cam_pos[~vis] = np.nan

            # Subtract ground height from world positions
            world_pos[..., 2] -= ground_height  # Subtract from z-coordinate

            joints_world.append(world_pos)
            joints_cam.append(cam_pos)
            visible.append(vis)

        joints_world_tensor = torch.tensor(joints_world, dtype=torch.float32)
        joints_cam_tensor = torch.tensor(joints_cam, dtype=torch.float32)
        visible_tensor = torch.tensor(visible, dtype=torch.bool)

        if (
            return_smplh_joints
        ):  # Initialize SMPLH tensors with NaN for positions and False for visibility
            T = joints_world_tensor.shape[0]
            smplh_world = torch.full((T, 22, 3), float("nan"), dtype=torch.float32)
            smplh_cam = torch.full((T, 22, 3), float("nan"), dtype=torch.float32)
            smplh_visible = torch.zeros((T, 22), dtype=torch.bool)

            # Map joints using EGOEXO4D_BODYPOSE_TO_SMPLH_INDICES
            for smplh_idx, ego_idx in enumerate(EGOEXO4D_BODYPOSE_TO_SMPLH_INDICES):
                if ego_idx != -1:
                    # Valid mapping - copy data
                    smplh_world[:, smplh_idx] = joints_world_tensor[:, ego_idx]
                    smplh_cam[:, smplh_idx] = joints_cam_tensor[:, ego_idx]
                    smplh_visible[:, smplh_idx] = visible_tensor[:, ego_idx]
                # Invalid mappings (-1) are already handled by initialization

            joints_world_tensor = smplh_world
            joints_cam_tensor = smplh_cam
            visible_tensor = smplh_visible

        # breakpoint()
        if debug_vis:
            ps.init()
            # Visualize first frame joints in world coordinates
            draw_coco_kinematic_tree(
                joints_world_tensor[0].detach().cpu().numpy(),
                coco_cfg=CFG.plotly.coco_kinematic_tree,
                curve_network_id="world_joints",
            )
            # Visualize first frame joints in camera coordinates
            draw_coco_kinematic_tree(
                joints_cam_tensor[0].detach().cpu().numpy(),
                coco_cfg=CFG.plotly.coco_kinematic_tree,
                curve_network_id="camera_joints",
            )
            ps.show()

        return joints_world_tensor, joints_cam_tensor, visible_tensor

    def _process_camera_poses(self, data: List[Dict[str, Any]]) -> torch.Tensor:
        """Extract camera poses from annotations."""
        poses = []
        for frame in data:
            if "camera_pose" in frame:
                pose = frame["camera_pose"]
                poses.append(
                    [
                        pose["qw"],
                        pose["qx"],
                        pose["qy"],
                        pose["qz"],
                        pose["tx"],
                        pose["ty"],
                        pose["tz"],
                    ]
                )
            else:
                poses.append([1, 0, 0, 0, 0, 0, 0])  # Identity pose

        return torch.tensor(poses, dtype=torch.float32)

    def _get_mask_ratio(self) -> float:
        """Get mask ratio for MAE training."""
        if self.config.random_sample_mask_ratio:
            return np.random.uniform(self.config.mask_ratio / 4, self.config.mask_ratio)
        return self.config.mask_ratio
