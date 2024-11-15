from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from smplx import SMPLX
import trimesh
from sklearn.cluster import DBSCAN

from egoallo.setup_logger import setup_logger
from egoallo.data.motion_processing import MotionProcessor

logger = setup_logger(output="logs/rich_processor", name=__name__)

class RICHDataProcessor:
    """Process RICH dataset sequences for training and evaluation.
    
    This class handles loading and preprocessing of RICH dataset sequences,
    including SMPL-X parameters, human-scene contact annotations, and
    floor height detection.
    """
    
    def __init__(
        self,
        rich_data_dir: str,
        smplx_model_dir: str,
        output_dir: str,
        fps: int = 30,
        include_contact: bool = True,
        use_pca: bool = True,
        device: str = "cuda",
    ) -> None:
        """Initialize the RICH data processor.
        
        Args:
            rich_data_dir: Path to RICH dataset root directory
            smplx_model_dir: Path to SMPL-X model files
            output_dir: Output directory for processed sequences
            fps: Target frames per second for sequences
            include_contact: Whether to include contact annotations
            use_pca: Whether to use PCA for hand poses
            device: Device to use for processing
        """
        self.rich_data_dir = Path(rich_data_dir)
        self.smplx_model_dir = Path(smplx_model_dir)
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.include_contact = include_contact
        self.use_pca = use_pca
        self.device = device

        # Constants for contact and floor detection
        self.floor_vel_thresh = 0.005
        self.floor_height_offset = 0.01
        self.contact_vel_thresh = 0.005
        self.contact_toe_height_thresh = 0.04
        self.contact_ankle_height_thresh = 0.08

        # Load gender and image extension mappings
        with open(self.rich_data_dir / "resource/gender.json", "r") as f:
            self.gender_mapping = json.load(f)
        with open(self.rich_data_dir / "resource/imgext.json", "r") as f:
            self.img_ext_mapping = json.load(f)

        # Initialize SMPL-X models for each gender
        self.body_models = {}
        for gender in ["male", "female", "neutral"]:
            self.body_models[gender] = SMPLX(
                model_path=str(self.smplx_model_dir),
                gender=gender,
                num_pca_comps=12 if use_pca else 45,
                flat_hand_mean=False,
                create_expression=True,
                create_jaw_pose=True,
            ).to(device)

        # Initialize motion processor
        self.motion_processor = MotionProcessor(
            floor_vel_thresh=self.floor_vel_thresh,
            floor_height_offset=self.floor_height_offset,
            contact_vel_thresh=self.contact_vel_thresh,
            contact_toe_height_thresh=self.contact_toe_height_thresh,
            contact_ankle_height_thresh=self.contact_ankle_height_thresh
        )
        
        # Joint indices mapping
        self.joint_indices = {
            "left_toe": 10,
            "right_toe": 11,
            "left_ankle": 8,
            "right_ankle": 9
        }

    def process_frame_data(
        self, split: str, seq_name: str, frame_id: int
    ) -> Tuple[Dict[str, Union[np.ndarray, torch.Tensor]], Optional[np.ndarray], Optional[np.ndarray]]:
        """Load and process SMPL-X parameters and contact data for a single frame.
        
        Args:
            split: Dataset split (train/val/test)
            seq_name: Name of the sequence
            frame_id: Frame ID to load
            
        Returns:
            Tuple containing:
                - Dictionary of SMPL-X parameters
                - Contact vertex indices (if include_contact=True)
                - Contact displacement vectors (if include_contact=True)
        """
        scene_name, sub_id, _ = seq_name.split("_")
        gender = self.gender_mapping[f"{int(sub_id)}"]
        
        # Load SMPL-X parameters
        params_path = (
            self.rich_data_dir / "data/bodies" / split / seq_name / 
            f"{frame_id:05d}" / f"{sub_id}.pkl"
        )
        with open(params_path, "rb") as f:
            body_params = np.load(f, allow_pickle=True)
            
        # Convert parameters to tensors
        body_params_tensor = {
            k: torch.from_numpy(v).float().to(self.device)
            for k, v in body_params.items()
        }
        
        # Process hand poses if using PCA
        if self.use_pca:
            body_model = self.body_models[gender]
            for side in ['left', 'right']:
                hand_pose = body_params_tensor[f'{side}_hand_pose']
                hand_components = getattr(body_model, f'np_{side}_hand_components')
                hand_components = torch.from_numpy(hand_components).float().to(self.device)
                body_params_tensor[f'{side}_hand_pose'] = torch.einsum(
                    'bi,ij->bj', [hand_pose, hand_components]
                )
        
        contact_verts = contact_vecs = None
        if self.include_contact:
            # Load contact annotations if available
            contact_path = (
                self.rich_data_dir / "data/human_scene_contact" / split / 
                seq_name / f"{frame_id:05d}" / f"{sub_id}.pkl"
            )
            if contact_path.exists():
                with open(contact_path, "rb") as f:
                    contact_data = np.load(f, allow_pickle=True)
                contact_verts = np.where(contact_data["contact"] > 0.0)[0]
                contact_vecs = contact_data["s2m_dist_id"]
                
        return body_params_tensor, contact_verts, contact_vecs

    def process_sequence(self, split: str, seq_name: str, output_path: Path) -> Optional[Dict[str, Any]]:
        """Process a complete sequence."""
        if output_path.exists():
            logger.info(f"Skipping {seq_name} - already processed")
            return None
            
        scene_name, sub_id, _ = seq_name.split('_')
        gender = self.gender_mapping[f'{int(sub_id)}']
        body_model = self.body_models[gender]
        
        # Get frame IDs
        seq_dir = self.rich_data_dir / "data/bodies" / split / seq_name
        frame_ids = sorted([
            int(d.name) for d in seq_dir.iterdir() 
            if d.is_dir() and d.name.isdigit()
        ])
        
        # Process each frame
        processed_frames = []
        contact_info = []
        joints_sequence = []
        
        for frame_id in frame_ids:
            # Load and process frame data
            body_params, contact_verts, contact_vecs = self.process_frame_data(
                split, seq_name, frame_id
            )
            
            # Forward pass through SMPL-X
            model_output = body_model(
                betas=body_params['betas'],
                global_orient=body_params['global_orient'],
                body_pose=body_params['body_pose'],
                left_hand_pose=body_params['left_hand_pose'],
                right_hand_pose=body_params['right_hand_pose'],
                return_verts=True
            )
            
            # Store joints for floor height detection
            joints_sequence.append(model_output.joints.detach().cpu().numpy())
            
            # Store processed frame data
            processed_frames.append(body_params)
            if self.include_contact:
                contact_info.append({
                    'contact_verts': contact_verts,
                    'contact_vecs': contact_vecs
                })
        
        # Convert joints sequence to numpy array
        joints_sequence = np.stack(joints_sequence)
        
        # Replace floor height detection with common implementation
        floor_height, contact_labels = self.motion_processor.process_floor_and_contacts(
            joints_sequence, self.joint_indices
        )
        
        # Use floor height for translation adjustment
        for frame_data in processed_frames:
            if 'transl' in frame_data:
                frame_data['transl'][:, 2] -= floor_height
        
        # Prepare sequence data
        sequence_data = {
            'body_params': processed_frames,
            'frame_ids': frame_ids,
            'gender': gender,
            'fps': self.fps
        }
        
        if self.include_contact:
            sequence_data.update({
                'contact_verts': [info['contact_verts'] for info in contact_info],
                'contact_vecs': [info['contact_vecs'] for info in contact_info]
            })
            
        return sequence_data

    def save_sequence(self, sequence_data: Dict[str, Any], output_path: Path) -> None:
        """Save processed sequence data to NPZ file.
        
        Args:
            sequence_data: Dictionary containing processed sequence data
            output_path: Path to save the NPZ file
        """
        # Convert tensors to numpy arrays
        save_data = {
            "body_params": [
                {k: v.cpu().numpy() for k, v in params.items()}
                for params in sequence_data["body_params"]
            ],
            "frame_ids": np.array(sequence_data["frame_ids"]),
            "gender": sequence_data["gender"],
            "fps": sequence_data["fps"]
        }
        
        # Add contact data if included
        if self.include_contact:
            save_data.update({
                "contact_verts": np.array(sequence_data["contact_verts"], dtype=object),
                "contact_vecs": np.array(sequence_data["contact_vecs"], dtype=object)
            })
            
        # Save compressed NPZ file
        np.savez_compressed(output_path, **save_data)
        logger.info(f"Saved processed sequence to {output_path}") 