"""AMASS dataset processor using common motion utilities."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
from smplx import SMPLH
from tqdm import tqdm

from egoallo.setup_logger import setup_logger
from egoallo.data.motion_processing import MotionProcessor

logger = setup_logger(output="logs/amass_processor", name=__name__)

class AMASSProcessor:
    """Process AMASS dataset sequences for training and evaluation."""
    
    def __init__(
        self,
        amass_dir: str,
        smpl_dir: str,
        output_dir: str,
        fps: int = 30,
        include_velocities: bool = True,
        include_align_rot: bool = True,
        device: str = "cuda",
    ) -> None:
        """Initialize AMASS processor.
        
        Args:
            amass_dir: Path to AMASS dataset root
            smpl_dir: Path to SMPL model files
            output_dir: Output directory for processed sequences
            fps: Target frames per second
            include_velocities: Whether to compute velocities
            include_align_rot: Whether to compute alignment rotations
            device: Device to use for processing
        """
        self.amass_dir = Path(amass_dir)
        self.smpl_dir = Path(smpl_dir)
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
        
        # Initialize SMPLH body models
        self.body_models = {}
        for gender in ['male', 'female', 'neutral']:
            model_path = os.path.join(str(self.smpl_dir), f'{gender}/model.npz')
            self.body_models[gender] = SMPLH(
                model_path=model_path,
                gender=gender,
                batch_size=1,
                num_betas=16,
                use_pca=False,
                flat_hand_mean=True,
            ).to(self.device)
    
    def process_sequence(
        self, seq_path: Path, min_frames: int = 30
    ) -> Optional[Dict[str, Any]]:
        """Process a single AMASS sequence.
        
        Args:
            seq_path: Path to sequence npz file
            min_frames: Minimum number of frames required
            
        Returns:
            Dictionary of processed sequence data
        """
        # Load sequence data
        seq_data = dict(np.load(seq_path, allow_pickle=True))
        
        # Get sequence info
        gender = str(seq_data.get('gender', 'neutral'))
        fps = int(seq_data['mocap_framerate'])
        
        # Split poses into components
        poses = seq_data['poses']  # (N, 156) - Full SMPLH poses
        trans = seq_data['trans']  # (N, 3) 
        betas = seq_data['betas'][:16]  # (16,) - Take first 16 shape params
        
        num_frames = len(poses)
        if num_frames < min_frames:
            logger.warning(f"Sequence too short: {num_frames} frames")
            return None
            
        # Split pose parameters
        root_orient = poses[:, :3]  # (N, 3) - Global root orientation
        body_pose = poses[:, 3:66]  # (N, 63) - Body joint rotations (21 joints)
        hand_pose = poses[:, 66:]  # (N, 90) - Hand joint rotations
        
        # Convert to tensors
        root_orient = torch.from_numpy(root_orient).float().to(self.device)  # (N, 3)
        body_pose = torch.from_numpy(body_pose).float().to(self.device)  # (N, 63) 
        hand_pose = torch.from_numpy(hand_pose).float().to(self.device)  # (N, 90)
        trans = torch.from_numpy(trans).float().to(self.device)  # (N, 3)
        betas = torch.from_numpy(betas).float().to(self.device)  # (16,)
        
        # Forward pass through SMPLH
        body_model = self.body_models[gender]
        body_output = body_model(
            betas=betas.expand(num_frames, -1),  # (N, 16)
            global_orient=root_orient,  # (N, 3)
            body_pose=body_pose,  # (N, 63)
            hand_pose=hand_pose,  # (N, 90)
            transl=trans,  # (N, 3)
            return_verts=True
        )
        
        # Get joint positions
        joints = body_output.joints.detach().cpu().numpy()  # (N, J, 3)
        
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
            
            # Joint velocities (N-2, J, 3)
            joint_vel = np.stack([
                self.motion_processor.compute_joint_velocity(joints[:, i])
                for i in range(joints.shape[1])
            ], axis=1)
            
            # Translation velocities (N-2, 3)
            trans_vel = self.motion_processor.compute_joint_velocity(
                trans.cpu().numpy()
            )
            
            # Angular velocities (N-2, 3)
            root_orient_mat = body_output.global_orient.detach().cpu().numpy()
            root_ang_vel = self.motion_processor.compute_angular_velocity(
                root_orient_mat, dt
            )
            
            velocities = {
                'joints': joint_vel,  # (N-2, J, 3)
                'trans': trans_vel,  # (N-2, 3)
                'root_orient': root_ang_vel  # (N-2, 3)
            }
        
        # Compute alignment rotations if requested
        align_rot = None
        if self.include_align_rot:
            forward_dir = joints[:, 1] - joints[:, 0]  # Pelvis to spine
            forward_dir = forward_dir / np.linalg.norm(forward_dir)
            align_rot = self.motion_processor.compute_alignment_rotation(forward_dir)
        
        # Prepare sequence data
        sequence_data = {
            'poses': poses.cpu().numpy(),  # (N, 156)
            'trans': trans.cpu().numpy(),  # (N, 3)
            'betas': betas.cpu().numpy(),  # (16,)
            'gender': gender,  # str
            'fps': fps,  # float
            'joints': joints,  # (N, J, 3)
            'contacts': contacts,  # (N, J)
            'hand_pose': hand_pose.cpu().numpy(),  # (N, 90)
            'root_orient': root_orient.cpu().numpy(),  # (N, 3)
            'body_pose': body_pose.cpu().numpy(),  # (N, 63)
        }
        
        if velocities is not None:
            sequence_data['velocities'] = velocities
        if align_rot is not None:
            sequence_data['align_rot'] = align_rot
            
        return sequence_data
    
    def save_sequence(
        self, sequence_data: Dict[str, Any], output_path: Path
    ) -> None:
        """Save processed sequence data.
        
        Args:
            sequence_data: Dictionary of sequence data
            output_path: Path to save the data
        """
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as compressed npz
        np.savez_compressed(output_path, **sequence_data)
        logger.info(f"Saved processed sequence to {output_path}") 