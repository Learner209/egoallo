from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from jaxtyping import Float

from egoallo.setup_logger import setup_logger
from egoallo.data.motion_processing import MotionProcessor
from egoallo.fncsmpl import SmplhModel, SmplhShaped, SmplhShapedAndPosed
from egoallo.transforms import SE3, SO3
import pickle

logger = setup_logger(output="logs/rich_processor", name=__name__)

class RICHDataProcessor:
    """Process RICH dataset sequences using functional SMPL-H."""
    
    def __init__(
        self,
        rich_data_dir: str,
        smplh_model_dir: str,
        output_dir: str,
        fps: int = 30,
        include_contact: bool = True,
        device: str = "cuda",
    ) -> None:
        self.rich_data_dir = Path(rich_data_dir)
        self.smplh_model_dir = Path(smplh_model_dir)
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.include_contact = include_contact
        self.device = device

        # Load gender mapping
        with open(self.rich_data_dir / "resource/gender.json", "r") as f:
            self.gender_mapping = json.load(f)

        # Load image extension mapping
        with open(self.rich_data_dir / "resource/imgext.json", "r") as f:
            self.img_ext_mapping = json.load(f)

        # Initialize SMPL-H models for each gender
        self.body_models = {}
        for gender in ["male", "female", "neutral"]:
            model_path = self.smplh_model_dir / f"SMPLH_{gender.upper()}.pkl"
            self.body_models[gender] = SmplhModel.load(model_path).to(self.device)

        # Initialize motion processor
        self.motion_processor = MotionProcessor()
        
        # Joint indices mapping
        self.joint_indices = {
            "left_toe": 10,
            "right_toe": 11,
            "left_ankle": 8,
            "right_ankle": 9
        }

    def _convert_rotations(
        self,
        global_orient: Float[Tensor, "... 3"],
        body_pose: Float[Tensor, "... 63"],
        left_hand_pose: Float[Tensor, "... 45"],
        right_hand_pose: Float[Tensor, "... 45"],
        transl: Float[Tensor, "... 3"]
    ) -> tuple[Float[Tensor, "... 7"], Float[Tensor, "... 21 4"], Float[Tensor, "... 15 4"], Float[Tensor, "... 15 4"]]:
        """Convert rotation representations."""
        # Convert global orientation and translation to SE(3)
        T_world_root = SE3.from_rotation_and_translation(
            rotation=SO3.exp(global_orient),
            translation=transl
        ).parameters()  # (..., 7)

        # Convert body pose to quaternions (21 joints)
        body_rots = body_pose.reshape(*body_pose.shape[:-1], 21, 3)
        body_quats = SO3.exp(body_rots).wxyz  # (..., 21, 4)

        # Convert hand poses to quaternions (15 joints each)
        left_hand_rots = left_hand_pose.reshape(*left_hand_pose.shape[:-1], 15, 3)
        right_hand_rots = right_hand_pose.reshape(*right_hand_pose.shape[:-1], 15, 3)
        left_hand_quats = SO3.exp(left_hand_rots).wxyz  # (..., 15, 4)
        right_hand_quats = SO3.exp(right_hand_rots).wxyz  # (..., 15, 4)

        return T_world_root, body_quats, left_hand_quats, right_hand_quats

    def process_frame_data(
        self, split: str, seq_name: str, frame_id: int
    ) -> Tuple[Dict[str, Union[np.ndarray, torch.Tensor]], Optional[np.ndarray], Dict[str, Any]]:
        """Process a single frame of RICH data."""
        scene_name, sub_id, _ = seq_name.split("_")
        
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

        # Load HSC parameters
        hsc_path = (
            self.rich_data_dir / "data/human_scene_contact" / split / seq_name /
            f"{frame_id:05d}" / f"{sub_id}.pkl"
        )
        with open(hsc_path, "rb") as f:
            hsc_params = pickle.load(f)
        
        # Extract contact information
        contact_data = {
            'vertex_contacts': hsc_params['contact'],  # (6890,) vertex contact labels
            'displacement_vectors': hsc_params['s2m_dist_id'],  # (10475,3) displacement vectors
            'closest_faces': hsc_params['closest_triangles_id']  # (10475,3,3) closest scene triangles
        }

        return body_params_tensor, None, contact_data

    def process_sequence(self, split: str, seq_name: str, output_path: Path) -> Optional[Dict[str, Any]]:
        """Process a complete sequence."""
        if output_path.exists():
            logger.info(f"Skipping {seq_name} - already processed")
            return None
            
        scene_name, sub_id, _ = seq_name.split('_')
        gender = self.gender_mapping[f'{int(sub_id)}']
        body_model: SmplhModel = self.body_models[gender]
        
        # Get frame IDs
        seq_dir = self.rich_data_dir / "data/bodies" / split / seq_name
        frame_ids = sorted([
            int(d.name) for d in seq_dir.iterdir() 
            if d.is_dir() and d.name.isdigit()
        ])
        
        # Process each frame
        joints_sequence = []
        processed_frames = []
        contact_data_sequence = []
        
        for frame_id in frame_ids:
            # Load and process frame data
            body_params, _, contact_data = self.process_frame_data(split, seq_name, frame_id)
            
            # Convert rotations to required format
            
            # Convert PCA coefficients to axis-angle format
            # import ipdb; ipdb.set_trace()
            left_hand_pose, right_hand_pose = body_model.convert_hand_poses(
                left_hand_pca=body_params['left_hand_pose'],  # (..., num_pca)
                right_hand_pca=body_params['right_hand_pose']  # (..., num_pca)
            )

            T_world_root, body_quats, left_hand_quats, right_hand_quats = self._convert_rotations(
                body_params['global_orient'],  # (..., 3)
                body_params['body_pose'],  # (..., 63)
                left_hand_pose,  # (..., 15, 3)
                right_hand_pose,  # (..., 15, 3)
                body_params['transl']  # (..., 3)
            )  # Returns (..., 7), (..., 21, 4), (..., 15, 4), (..., 15, 4)

            # Process through SMPL-H pipeline
            shaped: SmplhShaped = body_model.with_shape(body_params['betas'])  # (..., num_betas)
            posed: SmplhShapedAndPosed = shaped.with_pose_decomposed(
                T_world_root=T_world_root,  # (..., 7)
                body_quats=body_quats,  # (..., 21, 4)
                left_hand_quats=left_hand_quats,  # (..., 15, 4)
                right_hand_quats=right_hand_quats  # (..., 15, 4)
            )
            # mesh: SmplMesh = posed.lbs()
            
            # Store joints for floor height detection
            joints = torch.cat([posed.T_world_root[..., None, :], posed.Ts_world_joint], dim=-2)
            joints = joints.detach().cpu().numpy()  # (..., num_verts, 3)
            joints_sequence.append(joints)
            
            # Convert per-vertex contacts to per-joint contacts using posed mesh
            vertex_contacts = torch.from_numpy(contact_data['vertex_contacts']).float().to(self.device)  # (6890,)
            # Use weighted average of nearby vertex contacts for joint contacts
            joint_contacts = posed.compute_joint_contacts(vertex_contacts.unsqueeze(0)).squeeze(0)  # (num_joints,)
            
            # Store processed data
            processed_frames.append(body_params)
            contact_data_sequence.append({
                'vertex_contacts': contact_data['vertex_contacts'],
                'displacement_vectors': contact_data['displacement_vectors'],
                'closest_faces': contact_data['closest_faces'],
                'joint_contacts': joint_contacts.cpu().numpy()
            })
        
        # Process floor height only (removed contact detection)
        # import ipdb; ipdb.set_trace()
        joints_sequence = np.concatenate([j[None] if j.ndim == 2 else j for j in joints_sequence], axis=0)
        floor_height = self.motion_processor.detect_floor_height(joints_sequence, [
            self.joint_indices["left_toe"], self.joint_indices["right_toe"]
        ])
        
        # Adjust heights
        for frame_data in processed_frames:
            if 'transl' in frame_data:
                frame_data['transl'][:, 2] -= floor_height
        
        # Prepare sequence data
        sequence_data = {
            'body_params': processed_frames,
            'frame_ids': frame_ids,
            'gender': gender,
            'fps': self.fps,
            'floor_height': floor_height,
            'contact_data': contact_data_sequence
        }
            
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
            # Stack all contact data from sequence
            save_data.update({
                'contacts': np.stack([
                    frame_data['joint_contacts'] 
                    for frame_data in sequence_data['contact_data']
                ]),
                'vertex_contacts': np.stack([
                    frame_data['vertex_contacts']
                    for frame_data in sequence_data['contact_data']
                ]),
                'displacement_vectors': np.stack([
                    frame_data['displacement_vectors']
                    for frame_data in sequence_data['contact_data']
                ]),
                'closest_faces': np.stack([
                    frame_data['closest_faces']
                    for frame_data in sequence_data['contact_data']
                ])
            })
        # Save compressed NPZ file
        np.savez_compressed(output_path, **save_data)
        logger.info(f"Saved processed sequence to {output_path}") 