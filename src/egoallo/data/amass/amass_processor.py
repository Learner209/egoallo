"""AMASS dataset processor using functional SMPL-H implementation."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np
import torch
from torch import Tensor
from jaxtyping import Float

from egoallo.setup_logger import setup_logger
from egoallo.data.motion_processing import MotionProcessor
from egoallo.fncsmpl import SmplMesh, SmplhModel, SmplhShaped, SmplhShapedAndPosed
from egoallo.transforms import SE3, SO3

logger = setup_logger(output="logs/amass_processor", name=__name__)

class AMASSProcessor:
    """Process AMASS dataset sequences using functional SMPL-H implementation."""
    
    def __init__(
        self,
        amass_dir: str,
        smplh_dir: str,
        output_dir: str,
        fps: int = 30,
        include_velocities: bool = True,
        include_align_rot: bool = True,
        device: str = "cuda",
    ) -> None:
        """Initialize AMASS processor.
        
        Args:
            amass_dir: Path to AMASS dataset root
            smplh_dir: Path to SMPL model files
            output_dir: Output directory for processed sequences
            fps: Target frames per second
            include_velocities: Whether to compute velocities
            include_align_rot: Whether to compute alignment rotations
            device: Device to use for processing
        """
        self.amass_dir = Path(amass_dir)
        self.smplh_dir = Path(smplh_dir)
        self.output_dir = Path(output_dir)
        self.target_fps = fps
        self.include_velocities = include_velocities
        self.include_align_rot = include_align_rot
        self.device = device
        
        # Initialize motion processor
        self.motion_processor = MotionProcessor()
        
        # Joint indices for SMPL model
        self.joint_indices = {
            "left_ankle": 7,
            "right_ankle": 8,
            "left_toe": 10,
            "right_toe": 11
        }
        
        # Load SMPL-H models for each gender
        self.body_models = {}
        for gender in ['male', 'female', 'neutral']:
            model_path = self.smplh_dir / f"{gender}/model.npz"
            self.body_models[gender] = SmplhModel.load(model_path)

    def _convert_rotations(
        self, 
        root_orient: Float[Tensor, "... 3"], 
        body_pose: Float[Tensor, "... 63"],
        hand_pose: Float[Tensor, "... 90"],
        trans: Float[Tensor, "... 3"]
    ) -> tuple[Float[Tensor, "... 7"], Float[Tensor, "... 21 4"], Float[Tensor, "... 15 4"], Float[Tensor, "... 15 4"]]:
        """Convert rotation representations."""
        # Convert root orientation and translation to SE(3)
        T_world_root = SE3.from_rotation_and_translation(
            rotation=SO3.exp(root_orient),
            translation=trans
        ).parameters()  # (..., 7)

        # Convert body pose to quaternions (21 joints)
        body_rots = body_pose.reshape(*body_pose.shape[:-1], 21, 3)
        body_quats = SO3.exp(body_rots).wxyz  # (..., 21, 4)

        # Convert hand poses to quaternions (15 joints each)
        left_hand_rots = hand_pose[..., :45].reshape(*hand_pose.shape[:-1], 15, 3)
        right_hand_rots = hand_pose[..., 45:].reshape(*hand_pose.shape[:-1], 15, 3)
        left_hand_quats = SO3.exp(left_hand_rots).wxyz  # (..., 15, 4)
        right_hand_quats = SO3.exp(right_hand_rots).wxyz  # (..., 15, 4)

        return T_world_root, body_quats, left_hand_quats, right_hand_quats

    def process_sequence(
        self, seq_path: Path, min_frames: int = 30
    ) -> Optional[Dict[str, Any]]:
        """Process a single AMASS sequence."""
        # Load sequence data
        seq_data = dict(np.load(seq_path, allow_pickle=True))
        
        # Get sequence info
        gender = str(seq_data.get('gender', 'invalid'))
        fps = int(seq_data['mocap_framerate'])
        
        # Get poses and shape
        poses = torch.from_numpy(seq_data['poses']).float().to(self.device)  # (N, 156)
        trans = torch.from_numpy(seq_data['trans']).float().to(self.device)  # (N, 3)
        betas = torch.from_numpy(seq_data['betas'][:16]).float().to(self.device)  # (16,)
        
        num_frames = len(poses)
        if num_frames < min_frames:
            logger.warning(f"Sequence too short: {num_frames} frames")
            return None

        # Split pose parameters
        root_orient = poses[:, :3]  # (N, 3)
        body_pose = poses[:, 3:66]  # (N, 63)
        hand_pose = poses[:, 66:]  # (N, 90)

        # Convert rotations to required format
        T_world_root, body_quats, left_hand_quats, right_hand_quats = self._convert_rotations(
            root_orient, body_pose, hand_pose, trans
        )

        # Process through SMPL-H pipeline
        body_model: SmplhModel = self.body_models[gender].to(self.device)
        shaped: SmplhShaped = body_model.with_shape(betas[None])  # Add batch dim
        posed: SmplhShapedAndPosed = shaped.with_pose_decomposed(
            T_world_root=T_world_root,
            body_quats=body_quats,
            left_hand_quats=left_hand_quats,
            right_hand_quats=right_hand_quats
        )
        # mesh: SmplMesh = posed.lbs()

        # Extract joint positions (22 SMPL-H joints)
        joints = torch.cat([
            posed.T_world_root[..., None, 4:7],  # Root position
            posed.Ts_world_joint[..., 3:6]  # Other joint positions
        ], dim=-2).detach().cpu().numpy()  # (N, 22, 3)

        # Process floor height and contacts
        floor_height, contacts = self.motion_processor.process_floor_and_contacts(
            joints, self.joint_indices
        )

        # Adjust heights
        trans[:, 2] -= floor_height
        joints[..., 2] -= floor_height

        # Compute velocities if requested
        velocities = None
        if self.include_velocities:
            dt = 1.0 / fps
            joint_vel = np.stack([
                self.motion_processor.compute_joint_velocity(joints[:, i])
                for i in range(joints.shape[1])
            ], axis=1)
            
            trans_vel = self.motion_processor.compute_joint_velocity(
                trans.cpu().numpy()
            )
            
            root_orient_mat = SO3.exp(root_orient).as_matrix().detach().cpu().numpy()
            root_ang_vel = self.motion_processor.compute_angular_velocity(
                root_orient_mat, dt
            )
            
            velocities = {
                'joints': joint_vel,
                'trans': trans_vel,
                'root_orient': root_ang_vel
            }

        # Compute alignment rotations if requested
        align_rot = None
        if self.include_align_rot:
            forward_dir = joints[:, 1] - joints[:, 0]  # Pelvis to spine
            forward_dir = forward_dir / np.linalg.norm(forward_dir)
            align_rot = self.motion_processor.compute_alignment_rotation(forward_dir)

        # Prepare output data
        sequence_data = {
            'poses': poses.cpu().numpy(),
            'trans': trans.cpu().numpy(),
            'betas': betas.cpu().numpy(),
            'gender': gender,
            'fps': fps,
            'joints': joints,
            'contacts': contacts,
            'hand_pose': hand_pose.cpu().numpy(),
            'root_orient': root_orient.cpu().numpy(),
            'body_pose': body_pose.cpu().numpy(),
        }

        if velocities is not None:
            sequence_data['velocities'] = velocities
        if align_rot is not None:
            sequence_data['align_rot'] = align_rot

        return sequence_data

    def save_sequence(
        self, sequence_data: Dict[str, Any], output_path: Path
    ) -> None:
        """Save processed sequence data."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **sequence_data)
        logger.info(f"Saved processed sequence to {output_path}")