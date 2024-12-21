from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast, TYPE_CHECKING

import h5py
import numpy as np
import torch
import torch.utils.data
from egoallo.config import CONFIG_FILE, make_cfg
from egoallo.data.dataclass import EgoTrainingData
from egoallo.utils.setup_logger import setup_logger
from egoallo.mapping import EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS
if TYPE_CHECKING:
	from egoallo.config.train.train_config import EgoAlloTrainConfig

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
        config: "EgoAlloTrainConfig",
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
        self._anno_dir = Path(config.bodypose_anno_dir)
        self._groups = self._initialize_groups()
        
        # Setup caching
        self._cache: Optional[Dict[str, Dict[str, Any]]] = {} if cache_files else None
        
        # Calculate total length
        if self._slice_strategy == "full_sequence":
            self._length = len(self._groups)
        else:
            total_frames = sum(self._get_sequence_length(g) for g in self._groups)
            self._length = total_frames // self._subseq_len

    def _initialize_groups(self) -> List[str]:
        """Initialize dataset groups from annotation directory."""
        groups = []
        for anno_path in self._anno_dir.glob("*.json"):
            # Load annotation to check validity
            data = self._load_annotation(anno_path)
            if data is not None and len(data) >= self._subseq_len:
                groups.append(anno_path.stem)
        
        assert len(groups) > 0, f"No valid annotations found in {self._anno_dir}"
        return groups

    def _load_annotation(self, anno_path: Path) -> Optional[Dict[str, Any]]:
        """Load and validate a single annotation file."""
        try:
            with open(anno_path, "r") as f:
                data = json.load(f)
                
            # Basic validation
            if not data or "annotation3D" not in data[0]:
                return None
                
            return data
        except Exception as e:
            logger.warning(f"Error loading {anno_path}: {e}")
            return None

    def _get_sequence_length(self, group: str) -> int:
        """Get length of sequence for a group."""
        data = self._get_cached_data(group)
        return len(data) if data else 0

    def _get_cached_data(self, group: str) -> Dict[str, Any]:
        """Get data for group, using cache if enabled."""
        if self._cache is not None:
            if group not in self._cache:
                data = self._load_annotation(self._anno_dir / f"{group}.json")
                self._cache[group] = data if data else {}
            return self._cache[group]
        return self._load_annotation(self._anno_dir / f"{group}.json") or {}

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> EgoTrainingData:
        """Get a single item from the dataset.
        
        Args:
            index: Index of item to get
            
        Returns:
            EgoTrainingData object containing the sequence data
        """
        if self._slice_strategy == "full_sequence":
            group = self._groups[index]
            data = self._get_cached_data(group)
            start_t, end_t = 0, len(data)
        else:
            # Find which group and slice based on index
            group_idx = 0
            remaining_idx = index
            while group_idx < len(self._groups):
                seq_len = self._get_sequence_length(self._groups[group_idx])
                num_slices = seq_len // self._subseq_len
                if remaining_idx < num_slices:
                    break
                remaining_idx -= num_slices
                group_idx += 1
            
            group = self._groups[group_idx]
            data = self._get_cached_data(group)
            start_t = remaining_idx * self._subseq_len
            end_t = start_t + self._subseq_len

        # Load slice of data
        seq_len = end_t - start_t
        slice_data = data[start_t:end_t]
        
        # Convert to tensors
        joints_world, joints_cam, visible_mask = self._process_joints(slice_data)
        T_world_root = self._process_camera_poses(slice_data)
        
        # Create mask for MAE training
        mask_ratio = self._get_mask_ratio()
        num_joints = joints_world.shape[1]
        num_masked = int(num_joints * mask_ratio)
        
        visible_joints_mask = torch.ones((seq_len, num_joints), dtype=torch.bool)
        rand_indices = torch.randperm(num_joints)
        masked_indices = rand_indices[:num_masked]
        visible_joints_mask[:, masked_indices] = False

        # Create EgoTrainingData object
        return EgoTrainingData(
            joints_wrt_world=joints_world,
            joints_wrt_cpf=joints_cam,
            T_world_root=T_world_root,
            visible_joints_mask=visible_joints_mask,
            mask=torch.ones(seq_len, dtype=torch.bool),
            # Add other required fields with appropriate defaults
            betas=torch.zeros((1, 16)),  # Default betas
            body_quats=torch.zeros((seq_len, 21, 4)),  # Default body quaternions
            hand_quats=None,  # No hand data
            contacts=torch.zeros((seq_len, 2)),  # Default contacts
            height_from_floor=torch.zeros((seq_len, 1)),  # Default height
        )

    def _process_joints(
        self, data: List[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process joint data from annotations."""
        # Extract 3D joints and visibility
        joints_world = []
        joints_cam = [] 
        visible = []
        
        for frame in data:
            anno3d = frame["annotation3D"]
            world_pos = []
            cam_pos = []
            vis = []
            
            for joint in EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS:
                if joint in anno3d:
                    world_pos.append([
                        anno3d[joint]["x"],
                        anno3d[joint]["y"], 
                        anno3d[joint]["z"]
                    ])
                    # Get camera space coordinates if available
                    cam_pos.append([
                        anno3d[joint].get("x_cam", 0),
                        anno3d[joint].get("y_cam", 0),
                        anno3d[joint].get("z_cam", 0)
                    ])
                    vis.append(1)
                else:
                    world_pos.append([0, 0, 0])
                    cam_pos.append([0, 0, 0])
                    vis.append(0)
                    
            joints_world.append(world_pos)
            joints_cam.append(cam_pos)
            visible.append(vis)
            
        return (
            torch.tensor(joints_world, dtype=torch.float32),
            torch.tensor(joints_cam, dtype=torch.float32),
            torch.tensor(visible, dtype=torch.bool)
        )

    def _process_camera_poses(self, data: List[Dict[str, Any]]) -> torch.Tensor:
        """Extract camera poses from annotations."""
        poses = []
        for frame in data:
            if "camera_pose" in frame:
                pose = frame["camera_pose"]
                poses.append([
                    pose["qw"], pose["qx"], pose["qy"], pose["qz"],
                    pose["tx"], pose["ty"], pose["tz"]
                ])
            else:
                poses.append([1, 0, 0, 0, 0, 0, 0])  # Identity pose
                
        return torch.tensor(poses, dtype=torch.float32)

    def _get_mask_ratio(self) -> float:
        """Get mask ratio for MAE training."""
        if self.config.random_sample_mask_ratio:
            return np.random.uniform(self.config.mask_ratio / 4, self.config.mask_ratio)
        return self.config.mask_ratio
