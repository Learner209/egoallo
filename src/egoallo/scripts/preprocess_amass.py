"""Script to preprocess AMASS dataset and save to HDF5 format."""
from __future__ import annotations

import dataclasses
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import h5py
import numpy as np
import torch
import trimesh
import tyro
from smplx import SMPLX
from tqdm import tqdm
from sklearn.cluster import DBSCAN

from egoallo.setup_logger import setup_logger

logger = setup_logger(output="logs/amass_preprocess", name=__name__)

@dataclasses.dataclass
class AMASSSplits:
    """AMASS dataset splits."""
    TRAIN = ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 
             'BioMotionLab_NTroje', 'BMLmovi', 'EKUT', 'ACCAD']
    TEST = ['Transitions_mocap', 'HumanEva']
    VAL = ['MPI_HDM05', 'SFU', 'MPI_mosh']
    ALL = TRAIN + TEST + VAL

@dataclasses.dataclass 
class AMASSPreprocessConfig:
    """Configuration for preprocessing AMASS dataset."""
    # Dataset paths
    amass_root: Path = Path("./data/amass_raw")
    smplh_root: Path = Path("./data/smplh")
    output_dir: Path = Path("./data/amass_processed") 
    output_hdf5: Path = Path("./data/amass/amass_dataset.hdf5")
    output_list_file: Path = Path("./data/amass/amass_dataset_files.txt")
    
    # Processing options
    target_fps: int = 30
    min_sequence_length: float = 1.0  # seconds
    split_frame_limit: int = 2000  # Split long sequences to avoid memory issues
    include_contact: bool = True
    include_velocities: bool = True
    include_hand_pose: bool = True
    include_align_rot: bool = True
    use_pca: bool = True
    num_processes: int = 4
    
    # Dataset selection
    datasets: List[str] = dataclasses.field(default_factory=lambda: AMASSSplits.ALL)
    
    # Debug options
    debug: bool = False
    viz_seq: bool = False
    viz_plots: bool = False

class AMASSProcessor:
    """Processes AMASS dataset sequences."""
    
    def __init__(self, config: AMASSPreprocessConfig):
        """Initialize AMASS processor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Constants for processing
        self.floor_vel_thresh = 0.005
        self.floor_height_offset = 0.01
        self.contact_vel_thresh = 0.005
        self.contact_toe_height_thresh = 0.04
        self.contact_ankle_height_thresh = 0.08
        self.terrain_height_thresh = 0.04
        self.root_height_thresh = 0.04
        self.cluster_size_thresh = 0.25
        
        # Initialize body models
        self.body_models = self._init_body_models()
        
    def _init_body_models(self) -> Dict[str, SMPLX]:
        """Initialize SMPL+H models for each gender."""
        body_models = {}
        for gender in ['male', 'female', 'neutral']:
            body_models[gender] = SMPLX(
                str(self.config.smplh_root),  # Convert Path to str
                gender=gender,
                num_pca_comps=12 if self.config.use_pca else 45,
                flat_hand_mean=False,
                create_expression=True,
                create_jaw_pose=True,
                use_pca=self.config.use_pca
            ).to(self.device)
        return body_models

    def process_sequence(self, seq_path: Path) -> Optional[Dict[str, Any]]:
        """Process a single AMASS sequence.
        
        Args:
            seq_path: Path to sequence file
            
        Returns:
            Dictionary of processed data if successful, None otherwise
        """
        try:
            # Load sequence data
            data = np.load(seq_path)
            fps = float(data['mocap_framerate'])
            num_frames = data['poses'].shape[0]
            
            # Skip if sequence is too short
            if num_frames < self.config.min_sequence_length * fps:
                return None
                
            # Extract poses and parameters
            poses = {
                'trans': data['trans'][:],
                'root_orient': data['poses'][:, :3],
                'body_pose': data['poses'][:, 3:66],
                'hand_pose': data['poses'][:, 66:] if self.config.include_hand_pose else None,
                'betas': data['betas'][:]
            }
            
            # Trim sequence to middle 80%
            trim_start = int(0.1 * num_frames)
            trim_end = int(0.9 * num_frames)
            for k, v in poses.items():
                if v is not None:
                    poses[k] = v[trim_start:trim_end]
            num_frames = trim_end - trim_start
            
            # Split into chunks if needed
            chunks = []
            for start_idx in range(0, num_frames, self.config.split_frame_limit):
                end_idx = min(start_idx + self.config.split_frame_limit, num_frames)
                chunk_poses = {k: v[start_idx:end_idx] if v is not None else None 
                             for k, v in poses.items()}
                chunks.append(chunk_poses)
            
            # Process each chunk
            processed_chunks = []
            for chunk_poses in chunks:
                # Forward pass through SMPL+H model
                body = self._get_body_model_output(chunk_poses)
                
                # Process floor height and contacts
                floor_height, contacts = self._process_floor_and_contacts(body.joints.cpu().numpy())
                
                # Adjust heights
                chunk_poses['trans'][:, 2] -= floor_height
                
                # Compute velocities if requested
                velocities = self._compute_velocities(chunk_poses, fps) if self.config.include_velocities else None
                
                # Compute alignment rotations if requested  
                align_rot = self._compute_alignment_rotations(chunk_poses['root_orient']) if self.config.include_align_rot else None
                
                processed_chunks.append({
                    'poses': chunk_poses,
                    'contacts': contacts,
                    'velocities': velocities,
                    'align_rot': align_rot
                })
            
            # Combine chunks
            combined_data = self._combine_chunks(processed_chunks)
            
            # Downsample if needed
            if self.config.target_fps != fps:
                combined_data = self._downsample_sequence(
                    combined_data, fps, self.config.target_fps)
                
            combined_data['fps'] = self.config.target_fps
            return combined_data
            
        except Exception as e:
            logger.error(f"Error processing sequence {seq_path}: {str(e)}")
            return None
            
    def _get_body_model_output(self, poses: Dict[str, np.ndarray]) -> torch.Tensor:
        """Run forward pass through SMPL+H model."""
        # Convert numpy arrays to tensors
        poses_tensor = {k: torch.from_numpy(v).float().to(self.device) 
                       if v is not None else None
                       for k, v in poses.items()}
        
        # Get gender from betas
        gender = 'neutral'  # Could be determined from data
        body_model = self.body_models[gender]
        
        # Forward pass
        output = body_model(
            betas=poses_tensor['betas'],
            global_orient=poses_tensor['root_orient'],
            body_pose=poses_tensor['body_pose'],
            left_hand_pose=poses_tensor['hand_pose'][:, :45] if poses_tensor['hand_pose'] is not None else None,
            right_hand_pose=poses_tensor['hand_pose'][:, 45:] if poses_tensor['hand_pose'] is not None else None,
            transl=poses_tensor['trans'],
            return_verts=True
        )
        
        return output
        
    def _process_floor_and_contacts(self, joints: np.ndarray) -> tuple[float, np.ndarray]:
        """Determine floor height and compute contact labels."""
        # Get toe velocities and heights
        left_toe_vel = np.linalg.norm(np.diff(joints[:, 10], axis=0), axis=1)  # Example joint index
        right_toe_vel = np.linalg.norm(np.diff(joints[:, 11], axis=0), axis=1)  # Example joint index
        
        # Pad velocities to match original length
        left_toe_vel = np.append(left_toe_vel, left_toe_vel[-1])
        right_toe_vel = np.append(right_toe_vel, right_toe_vel[-1])
        
        # Get static frames
        static_mask = (left_toe_vel < self.floor_vel_thresh) | (right_toe_vel < self.floor_vel_thresh)
        static_heights = joints[static_mask, :, 2].flatten()
        
        # Cluster heights to find floor
        clustering = DBSCAN(eps=0.005, min_samples=3).fit(static_heights.reshape(-1, 1))
        floor_height = np.min([np.median(static_heights[clustering.labels_ == i])
                             for i in np.unique(clustering.labels_) if i != -1])
        
        # Compute contacts
        contacts = np.zeros((joints.shape[0], joints.shape[1]), dtype=bool)
        contacts[:, 10] = (left_toe_vel < self.contact_vel_thresh) & (joints[:, 10, 2] < floor_height + self.contact_toe_height_thresh)
        contacts[:, 11] = (right_toe_vel < self.contact_vel_thresh) & (joints[:, 11, 2] < floor_height + self.contact_toe_height_thresh)
        
        return floor_height, contacts
        
    def _compute_velocities(self, poses: Dict[str, np.ndarray], fps: float) -> Dict[str, np.ndarray]:
        """Compute velocities for poses and joints."""
        dt = 1.0 / fps
        velocities = {}
        
        # Linear velocities
        for k in ['trans']:
            if poses[k] is not None:
                velocities[k] = np.gradient(poses[k], dt, axis=0)
        
        # Angular velocities
        for k in ['root_orient', 'body_pose', 'hand_pose']:
            if poses[k] is not None:
                # Convert to rotation matrices
                rot_mats = self._aa_to_rotmat(poses[k].reshape(-1, 3)).reshape(poses[k].shape[0], -1, 3, 3)
                velocities[k] = self._compute_angular_velocity(rot_mats, dt)
        
        return velocities
        
    def _compute_alignment_rotations(self, root_orient: np.ndarray) -> np.ndarray:
        """Compute rotations to align poses with canonical frame."""
        # Convert to rotation matrices
        rot_mats = self._aa_to_rotmat(root_orient)
        
        # Extract forward direction (z-axis)
        forward = rot_mats[..., :3, 2]
        
        # Project to horizontal plane
        forward_flat = forward.copy()
        forward_flat[..., 1] = 0
        forward_flat /= np.linalg.norm(forward_flat, axis=-1, keepdims=True)
        
        # Compute rotation to align with global forward
        global_forward = np.array([0, 0, 1])
        align_rot = np.zeros_like(rot_mats)
        align_rot[..., 2] = forward_flat
        align_rot[..., 1] = np.cross(forward_flat, global_forward)
        align_rot[..., 0] = np.cross(align_rot[..., 1], forward_flat)
        
        return align_rot
        
    def _combine_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine processed chunks into single sequence."""
        combined = {
            'poses': {},
            'contacts': None,
            'velocities': {},
            'align_rot': None
        }
        
        # Combine poses
        for k in chunks[0]['poses'].keys():
            if chunks[0]['poses'][k] is not None:
                combined['poses'][k] = np.concatenate([c['poses'][k] for c in chunks])
        
        # Combine contacts
        if chunks[0]['contacts'] is not None:
            combined['contacts'] = np.concatenate([c['contacts'] for c in chunks])
            
        # Combine velocities
        if chunks[0]['velocities'] is not None:
            for k in chunks[0]['velocities'].keys():
                combined['velocities'][k] = np.concatenate([c['velocities'][k] for c in chunks])
                
        # Combine alignment rotations
        if chunks[0]['align_rot'] is not None:
            combined['align_rot'] = np.concatenate([c['align_rot'] for c in chunks])
            
        return combined
        
    def _downsample_sequence(self, data: Dict[str, Any], src_fps: float, 
                            target_fps: float) -> Dict[str, Any]:
        """Downsample sequence data to target framerate."""
        if target_fps >= src_fps:
            return data
            
        # Compute downsample indices
        fps_ratio = target_fps / src_fps
        num_frames = data['poses']['trans'].shape[0]
        new_num_frames = int(fps_ratio * num_frames)
        downsamp_inds = np.linspace(0, num_frames-1, num=new_num_frames, dtype=int)
        
        # Downsample all arrays
        downsampled = {
            'poses': {},
            'contacts': None,
            'velocities': {},
            'align_rot': None
        }
        
        # Downsample poses
        for k, v in data['poses'].items():
            if v is not None:
                downsampled['poses'][k] = v[downsamp_inds]
                
        # Downsample contacts
        if data['contacts'] is not None:
            downsampled['contacts'] = data['contacts'][downsamp_inds]
            
        # Downsample velocities
        if data['velocities'] is not None:
            for k, v in data['velocities'].items():
                downsampled['velocities'][k] = v[downsamp_inds]
                
        # Downsample alignment rotations
        if data['align_rot'] is not None:
            downsampled['align_rot'] = data['align_rot'][downsamp_inds]
            
        return downsampled
        
    @staticmethod
    def _aa_to_rotmat(aa: np.ndarray) -> np.ndarray:
        """Convert axis-angle representation to rotation matrices."""
        # Implementation using Rodrigues formula
        theta = np.linalg.norm(aa, axis=-1, keepdims=True)
        mask = theta > 0
        k = aa / (theta + 1e-8)
        
        K = np.zeros(aa.shape[:-1] + (3, 3), dtype=aa.dtype)
        K[..., 0, 1] = -k[..., 2]
        K[..., 0, 2] = k[..., 1]
        K[..., 1, 0] = k[..., 2]
        K[..., 1, 2] = -k[..., 0]
        K[..., 2, 0] = -k[..., 1]
        K[..., 2, 1] = k[..., 0]
        
        R = np.eye(3) + np.sin(theta)[..., None] * K + (1 - np.cos(theta))[..., None] * np.matmul(K, K)
        R = np.where(mask[..., None, None], R, np.eye(3))
        
        return R
        
    @staticmethod
    def _compute_angular_velocity(rot_mats: np.ndarray, dt: float) -> np.ndarray:
        """Compute angular velocities from sequence of rotation matrices."""
        # Implementation using finite differences
        dR = np.gradient(rot_mats, dt, axis=0)
        R = rot_mats[1:-1]  # Use middle frames
        
        # Convert to skew-symmetric matrices
        w_mat = np.matmul(dR, np.transpose(R, (0, 1, 3, 2)))
        
        # Extract angular velocity vector
        ang_vel = np.stack([
            -w_mat[..., 1, 2] + w_mat[..., 2, 1],
            w_mat[..., 0, 2] - w_mat[..., 2, 0],
            -w_mat[..., 0, 1] + w_mat[..., 1, 0]
        ], axis=-1) / 2.0
        
        return ang_vel

def main(config: AMASSPreprocessConfig) -> None:
    """Main preprocessing function."""
    start_time = time.time()
    
    # Create output directories
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = AMASSProcessor(config)
    
    # Set up task queue for parallel processing
    task_queue = queue.Queue[Path]()
    
    # Collect all sequences to process
    for dataset in config.datasets:
        dataset_dir = config.amass_root / dataset
        if not dataset_dir.exists():
            logger.warning(f"Dataset {dataset} not found in {config.amass_root}")
            continue
            
        # Get all sequence files
        for seq_path in dataset_dir.rglob("*_poses.npz"):
            task_queue.put_nowait(seq_path)
    
    total_count = task_queue.qsize()
    logger.info(f"Found {total_count} sequences to process")
    
    # Process sequences
    file_list: List[str] = []
    
    def worker(device_idx: int) -> None:
        """Worker function for parallel processing."""
        while True:
            try:
                seq_path = task_queue.get_nowait()
            except queue.Empty:
                break
                
            # Process sequence
            processed_data = processor.process_sequence(seq_path)
            if processed_data is None:
                continue
                
            # Save to HDF5
            rel_path = seq_path.relative_to(config.amass_root)
            group_name = str(rel_path.parent / rel_path.stem)
            
            with h5py.File(config.output_hdf5, 'a') as f:
                group = f.create_group(group_name)
                for k, v in processed_data.items():
                    if isinstance(v, dict):
                        subgroup = group.create_group(k)
                        for sk, sv in v.items():
                            if sv is not None:
                                chunks = (min(32, sv.shape[0]),) + sv.shape[1:] if sv.ndim > 1 else None
                                subgroup.create_dataset(sk, data=sv, chunks=chunks)
                    elif isinstance(v, np.ndarray):
                        chunks = (min(32, v.shape[0]),) + v.shape[1:] if v.ndim > 1 else None
                        group.create_dataset(k, data=v, chunks=chunks)
                    else:
                        group.attrs[k] = v
                        
            file_list.append(group_name)
            logger.info(f"Processed {group_name}")
    
    # Start worker threads
    workers = [
        threading.Thread(target=worker, args=(i,))
        for i in range(config.num_processes)
    ]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
        
    # Save file list
    config.output_list_file.write_text("\n".join(sorted(file_list)))
    
    total_time = time.time() - start_time
    logger.info(f"Total processing time: {total_time/60:.2f} minutes")

if __name__ == "__main__":
    # Parse config
    config = tyro.cli(AMASSPreprocessConfig)
    
    # Set up debugging
    if config.debug:
        import ipdb
        ipdb.set_trace()
    
    main(config) 