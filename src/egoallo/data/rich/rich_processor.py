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
        all_poses = []
        all_trans = []
        all_joints = []
        all_contacts = []
        
        for frame_id in frame_ids:
            # import ipdb; ipdb.set_trace()
            body_params, _, contact_data = self.process_frame_data(split, seq_name, frame_id)
            
            # Convert hand PCA to axis-angle
            left_hand_pose, right_hand_pose = body_model.convert_hand_poses(
                left_hand_pca=body_params['left_hand_pose'],
                right_hand_pca=body_params['right_hand_pose']
            )

            # Combine poses in AMASS format (156 = 3 + 63 + 90)
            poses = torch.cat([
                body_params['global_orient'].reshape(-1),  # (3,)
                body_params['body_pose'].reshape(-1),      # (63,)
                left_hand_pose.reshape(-1),    # (45,)
                right_hand_pose.reshape(-1)    # (45,)
            ], dim=-1)
            
            all_poses.append(poses)
            all_trans.append(body_params['transl'])

            # Process joints and contacts similar to AMASS
            T_world_root, body_quats, left_hand_quats, right_hand_quats = self._convert_rotations(
                body_params['global_orient'],
                body_params['body_pose'],
                left_hand_pose,
                right_hand_pose,
                body_params['transl']
            )

            shaped = body_model.with_shape(body_params['betas'])
            posed = shaped.with_pose_decomposed(
                T_world_root=T_world_root,
                body_quats=body_quats,
                left_hand_quats=left_hand_quats,
                right_hand_quats=right_hand_quats
            )

            # Extract joint positions
            joints = torch.cat([
                posed.T_world_root[..., None, 4:7],
                posed.Ts_world_joint[..., 4:7]
            ], dim=-2).detach().cpu().numpy()
            all_joints.append(joints)

            # Process contacts
            joint_contacts = posed.compute_joint_contacts(
                torch.from_numpy(contact_data['vertex_contacts']).float().to(self.device).unsqueeze(0)
            ).squeeze(0).cpu().numpy()
            all_contacts.append(joint_contacts)

        # Stack sequences first
        poses = torch.stack(all_poses).cpu().numpy()  # (N, 156)
        trans = torch.cat(all_trans, dim=0).cpu().numpy()  # (N, 3)
        joints = np.concatenate(all_joints, axis=0)  # (N, 22, 3)
        contacts = np.stack(all_contacts)  # (N, num_joints)

        # Detect floor height using foot joints
        floor_height = self.motion_processor.detect_floor_height(
            joints,
            [self.joint_indices["left_toe"], self.joint_indices["right_toe"]]
        )
        # import ipdb; ipdb.set_trace()
        

        # Adjust heights
        trans[:, 2] -= floor_height
        joints[..., 2] -= floor_height

        # Rest of processing (velocities, alignment, etc.)
        dt = 1.0 / self.fps
        velocities = {
            'joints': np.stack([
                self.motion_processor.compute_joint_velocity(joints[:, i])
                for i in range(joints.shape[1])
            ], axis=1),
            'trans': self.motion_processor.compute_joint_velocity(trans),
            'root_orient': self.motion_processor.compute_angular_velocity(
                SO3.exp(torch.from_numpy(poses[:, :3])).as_matrix().numpy(),
                dt
            )
        }

        # Compute alignment rotation
        forward_dir = joints[:, 1] - joints[:, 0]  # Pelvis to spine
        forward_dir = forward_dir / np.linalg.norm(forward_dir, axis=-1, keepdims=True)
        align_rot = self.motion_processor.compute_alignment_rotation(forward_dir)

        # Prepare sequence data in AMASS format
        sequence_data = {
            'poses': poses,  # (N, 156)
            'trans': trans,  # (N, 3)
            'betas': body_params['betas'].cpu().numpy(),  # (16,)
            'gender': gender,
            'fps': self.fps,
            'joints': joints,  # (N, 22, 3)
            'contacts': contacts,
            'pose_hand': poses[:, 66:],  # (N, 90)
            'root_orient': poses[:, :3],  # (N, 3)
            'pose_body': poses[:, 3:66],  # (N, 63)
            'velocities': velocities,
            'align_rot': align_rot
        }

        return sequence_data

    def save_sequence(self, sequence_data: Dict[str, Any], output_path: Path) -> None:
        """Save processed sequence data to NPZ file.
        
        Args:
            sequence_data: Dictionary containing processed sequence data
            output_path: Path to save the NPZ file
        """
        np.savez_compressed(output_path, **sequence_data)
        logger.info(f"Saved processed sequence to {output_path}")