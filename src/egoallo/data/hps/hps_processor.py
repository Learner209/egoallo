"""HPS dataset processor using functional SMPL-H implementation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
from torch import Tensor
from jaxtyping import Float, Bool

from egoallo.setup_logger import setup_logger
from egoallo.data.motion_processing import MotionProcessor
from egoallo.fncsmpl import SmplhModel, SmplhShaped, SmplhShapedAndPosed
from egoallo.transforms import SE3, SO3
from egoallo.data.dataclass import EgoTrainingData

logger = setup_logger(output="logs/hps_processor", name=__name__)

class HPSProcessor:
    """Process HPS dataset sequences using functional SMPL-H."""
    
    def __init__(
        self,
        hps_dir: str,
        smplh_dir: str,
        output_dir: str,
        fps: int = 30,
        include_hands: bool = True,
        device: str = "cuda",
    ) -> None:
        """Initialize HPS processor.
        
        Args:
            hps_dir: Path to HPS dataset root
            smplh_dir: Path to SMPL model files
            output_dir: Output directory for processed sequences
            fps: Target frames per second
            include_hands: Whether to include hand poses
            device: Device to use for processing
        """
        self.hps_dir = Path(hps_dir)
        self.smplh_dir = Path(smplh_dir)
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.include_hands = include_hands
        self.device = device
        
        # Initialize motion processor
        self.motion_processor = MotionProcessor()
        
        # Load SMPL-H models
        self.body_model = SmplhModel.load(
            self.smplh_dir / "model.npz"
        ).to(self.device)

        # Joint indices for contact detection
        self.joint_indices = {
            "left_ankle": 7,
            "right_ankle": 8,
            "left_toe": 10,
            "right_toe": 11
        }

    def _load_camera_trajectory(
        self, sequence_name: str
    ) -> tuple[list[dict[str, float]], float, float]:
        """Load camera trajectory from HPS format."""
        camera_path = self.hps_dir / "head_camera_localizations" / f"{sequence_name}.json"
        with open(camera_path, 'r') as f:
            trajectory_data = json.load(f)
        
        # Convert timestamps to frame indices
        trajectory_data = {
            int(k): {**v, "time": float(k)/self.fps} 
            for k, v in trajectory_data.items() 
            if v is not None
        }
        
        first_key, last_key = min(trajectory_data.keys()), max(trajectory_data.keys())
        return (
            list(trajectory_data.values()),
            trajectory_data[first_key]["time"],
            trajectory_data[last_key]["time"]
        )

    def _convert_rotations(
        self,
        root_orient: Float[Tensor, "... 3"],
        body_pose: Float[Tensor, "... 63"],
        hand_pose: Float[Tensor, "... 90"],
        trans: Float[Tensor, "... 3"]
    ) -> tuple[Float[Tensor, "... 7"], Float[Tensor, "... 21 4"], Float[Tensor, "... 30 4"]]:
        """Convert rotation representations to EgoTrainingData format."""
        # Convert root orientation and translation to SE(3)
        T_world_root = SE3.from_rotation_and_translation(
            rotation=SO3.exp(root_orient),
            translation=trans
        ).parameters()  # (..., 7)

        # Convert body pose to quaternions (21 joints)
        body_rots = body_pose.reshape(*body_pose.shape[:-1], 21, 3)
        body_quats = SO3.exp(body_rots).wxyz  # (..., 21, 4)

        # Convert hand poses to quaternions (30 joints total)
        hand_rots = hand_pose.reshape(*hand_pose.shape[:-1], 30, 3)
        hand_quats = SO3.exp(hand_rots).wxyz  # (..., 30, 4)

        return T_world_root, body_quats, hand_quats

    def process_sequence(
        self, sequence_name: str, min_frames: int = 30
    ) -> Optional[EgoTrainingData]:
        """Process a single HPS sequence into EgoTrainingData format."""
        # Load sequence data
        seq_path = self.hps_dir / "hps_smpl" / f"{sequence_name}.pkl"
        with open(seq_path, "rb") as f:
            seq_data = np.load(f, allow_pickle=True)
        
        # Load subject betas
        subject = sequence_name.split("_")[0]
        with open(self.hps_dir / "hps_betas" / f"{subject}.json", "r") as f:
            betas_data = json.load(f)
        betas = torch.tensor(betas_data, device=self.device)

        # Convert sequence data to tensors
        poses = torch.from_numpy(seq_data["poses"]).float().to(self.device)
        trans = torch.from_numpy(seq_data["trans"]).float().to(self.device)
        
        num_frames = len(poses)
        if num_frames < min_frames:
            logger.warning(f"Sequence too short: {num_frames} frames")
            return None

        # Split pose parameters
        root_orient = poses[:, :3]  # (N, 3)
        body_pose = poses[:, 3:66]  # (N, 63)
        hand_pose = poses[:, 66:]  # (N, 90)

        # Convert rotations to required format
        T_world_root, body_quats, hand_quats = self._convert_rotations(
            root_orient, body_pose, hand_pose, trans
        )

        # Process through SMPL-H pipeline
        shaped = self.body_model.with_shape(betas[None])
        posed = shaped.with_pose_decomposed(
            T_world_root=T_world_root,
            body_quats=body_quats
        )

        # Get joint positions
        joints = torch.cat([
            posed.T_world_root[..., None, 4:7],  # Root position
            posed.Ts_world_joint[..., 4:7]  # Other joint positions
        ], dim=-2)

        # Process floor height and contacts
        floor_height = self.motion_processor.detect_floor_height(
            joints.cpu().numpy(),
            list(self.joint_indices.values())
        )
        
        # Adjust heights
        trans[:, 2] -= floor_height
        joints[..., 2] -= floor_height
        
        # Get central pupil frame transforms
        T_world_cpf = (
            tf.SE3(posed.Ts_world_joint[:, 14, :])  # T_world_head
            @ tf.SE3(fncsmpl_extensions.get_T_head_cpf(shaped))
        ).parameters()

        # Compute CPF frame-to-frame transform
        T_cpf_tm1_cpf_t = (
            tf.SE3(T_world_cpf[:-1, :]).inverse() @ tf.SE3(T_world_cpf[1:, :])
        ).parameters()

        # Compute joints in CPF frame
        joints_wrt_cpf = (
            tf.SE3(T_world_cpf[:, None, :]).inverse()
            @ joints
        )

        # Create EgoTrainingData instance
        ego_data = EgoTrainingData(
            T_world_root=T_world_root[1:],
            contacts=torch.ones((num_frames-1, 21), dtype=torch.float32),  # Placeholder contacts
            betas=betas[None],
            body_quats=body_quats[1:],
            T_cpf_tm1_cpf_t=T_cpf_tm1_cpf_t,
            T_world_cpf=T_world_cpf[1:],
            height_from_floor=T_world_cpf[1:, 6:7],
            joints_wrt_cpf=joints_wrt_cpf[1:],
            mask=torch.ones((num_frames-1,), dtype=torch.bool),
            hand_quats=hand_quats[1:] if self.include_hands else None
        )

        return ego_data

    def save_sequence(
        self, ego_data: EgoTrainingData, output_path: Path
    ) -> None:
        """Save processed sequence as NPZ file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to numpy arrays
        save_dict = {
            k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in vars(ego_data).items()
        }
        
        np.savez_compressed(output_path, **save_dict)
        logger.info(f"Saved processed sequence to {output_path}")