"""Common utilities for motion capture data processing."""
from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.cluster import DBSCAN
from jaxtyping import Float

class MotionProcessor:
    """Common utilities for processing motion capture sequences.
    
    This class provides dataset-independent methods for processing motion data,
    including floor detection, contact processing, and motion analysis.
    """
    
    def __init__(
        self,
        floor_vel_thresh: float = 0.005,
        floor_height_offset: float = 0.01,
        contact_vel_thresh: float = 0.005,
        contact_toe_height_thresh: float = 0.04,
        contact_ankle_height_thresh: float = 0.08,
    ):
        """Initialize motion processor with detection thresholds.
        
        Args:
            floor_vel_thresh: Velocity threshold for static foot detection
            floor_height_offset: Offset from detected floor height
            contact_vel_thresh: Velocity threshold for contact detection
            contact_toe_height_thresh: Height threshold for toe contacts
            contact_ankle_height_thresh: Height threshold for ankle contacts
        """
        self.floor_vel_thresh = floor_vel_thresh
        self.floor_height_offset = floor_height_offset
        self.contact_vel_thresh = contact_vel_thresh
        self.contact_toe_height_thresh = contact_toe_height_thresh
        self.contact_ankle_height_thresh = contact_ankle_height_thresh
    
    def process_floor_and_contacts(
        self, joints: Float[np.ndarray, "*batch 3"], joint_indices: Dict[str, int]
    ) -> Tuple[float, Float[np.ndarray, "*batch 3"]]:
        """Process floor height and contact labels from joint positions.
        
        Args:
            joints: Joint positions of shape (num_frames, num_joints, 3)
            joint_indices: Dictionary mapping joint names to indices
            
        Returns:
            Tuple containing:
                - Estimated floor height
                - Boolean contact labels of shape (num_frames, num_joints)
        """
        floor_height = self.detect_floor_height(joints, list(joint_indices.values()))
        
        # Initialize contact array
        contacts = np.zeros((joints.shape[0], joints.shape[1]), dtype=bool)
        
        # Process toe and ankle contacts
        for side in ["left", "right"]:
            # Toe contacts
            toe_idx = joint_indices[f"{side}_foot"]
            toe_vel = self.compute_joint_velocity(joints[:, toe_idx])
            contacts[:, toe_idx] = (
                (toe_vel < self.contact_vel_thresh) & 
                (joints[:, toe_idx, 2] < floor_height + self.contact_toe_height_thresh)
            )
            
            # Ankle contacts
            ankle_idx = joint_indices[f"{side}_ankle"]
            ankle_vel = self.compute_joint_velocity(joints[:, ankle_idx])
            contacts[:, ankle_idx] = (
                (ankle_vel < self.contact_vel_thresh) & 
                (joints[:, ankle_idx, 2] < floor_height + self.contact_ankle_height_thresh)
            )
            
            # Knee contacts
            knee_idx = joint_indices[f"{side}_knee"]
            knee_vel = self.compute_joint_velocity(joints[:, knee_idx])
            contacts[:, knee_idx] = (
                (knee_vel < self.contact_vel_thresh) &
                (joints[:, knee_idx, 2] < floor_height + self.contact_ankle_height_thresh)
            )
            
            # Elbow contacts
            elbow_idx = joint_indices[f"{side}_elbow"]
            elbow_vel = self.compute_joint_velocity(joints[:, elbow_idx])
            contacts[:, elbow_idx] = (
                (elbow_vel < self.contact_vel_thresh) &
                (joints[:, elbow_idx, 2] < floor_height + self.contact_ankle_height_thresh)
            )
            
            # Wrist/hand contacts
            wrist_idx = joint_indices[f"{side}_wrist"]
            wrist_vel = self.compute_joint_velocity(joints[:, wrist_idx])
            contacts[:, wrist_idx] = (
                (wrist_vel < self.contact_vel_thresh) &
                (joints[:, wrist_idx, 2] < floor_height + self.contact_ankle_height_thresh)
            )
            
            # Foot contacts
            foot_idx = joint_indices[f"{side}_foot"]
            foot_vel = self.compute_joint_velocity(joints[:, foot_idx])
            contacts[:, foot_idx] = (
                (foot_vel < self.contact_vel_thresh) &
                (joints[:, foot_idx, 2] < floor_height + self.contact_toe_height_thresh)
            )
            
        return floor_height, contacts
    
    def detect_floor_height(
        self, joints: Float[np.ndarray, "*batch 3"], foot_joints: list[int]
    ) -> float:
        """Detect floor height using DBSCAN clustering on foot joint positions.
        
        Args:
            joints: Joint positions of shape (num_frames, num_joints, 3)
            foot_joints: List of indices for foot joints to use
            
        Returns:
            Estimated floor height
        """
        MAX_SAMPLES = int(4e5)  # Maximum number of frames to process
        # import ipdb; ipdb.set_trace()
        
        # Get foot joint velocities
        foot_vels = np.stack([
            self.compute_joint_velocity(joints[:, joint_idx])
            for joint_idx in foot_joints
        ])
        
        # Find static frames
        static_mask = np.any(foot_vels < self.floor_vel_thresh, axis=0)
        static_heights = joints[static_mask][..., 2].flatten()
        
        if len(static_heights) == 0:
            return np.min(joints[..., 2])
            
        # Intelligent downsampling if sequence is too long
        if len(static_heights) > MAX_SAMPLES:
            # Compute histogram to understand height distribution
            hist, bin_edges = np.histogram(static_heights, bins='auto')
            weights = 1.0 / (hist[np.digitize(static_heights, bin_edges[1:], right=True)] + 1)
            weights /= np.sum(weights)
            
            # Stratified sampling to preserve distribution
            indices = np.random.choice(
                len(static_heights), 
                size=MAX_SAMPLES, 
                p=weights,
                replace=False
            )
            static_heights = static_heights[indices]
        
        # Cluster heights using DBSCAN
        # TODO: Make eps a parameter tailored to the dataset: AMASS, RICH, HPS.
        clustering = DBSCAN(eps=0.00005, min_samples=3).fit(
            static_heights.reshape(-1, 1)
        )
        valid_clusters = np.unique(clustering.labels_[clustering.labels_ != -1])
        
        if len(valid_clusters) == 0:
            floor_height = np.min(static_heights)
        else:
            # Use lowest significant cluster
            cluster_heights = [
                np.median(static_heights[clustering.labels_ == i])
                for i in valid_clusters
            ]
            floor_height = np.min(cluster_heights)
            
        return float(floor_height)
    
    @staticmethod
    def compute_joint_velocity(positions: Float[np.ndarray, "*batch 3"]) -> Float[np.ndarray, "*batch 3"]:
        """Compute joint velocity from positions.
        
        Args:
            positions: Joint positions of shape (num_frames, 3)
            
        Returns:
            Joint velocities of shape (num_frames,)
        """
        velocities = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        # Pad to match original length
        return np.pad(velocities, (0, 1), mode='edge')
    
    @staticmethod
    def compute_angular_velocity(
        rot_mats: Float[np.ndarray, "*batch 3 3"], 
        dt: float = 1.0/30
    ) -> Float[np.ndarray, "*batch 3"]:
        """Compute angular velocities from rotation matrices.
        
        Args:
            rot_mats: Rotation matrices of shape (..., 3, 3)
            dt: Time step between frames
            
        Returns:
            Angular velocities of shape (..., 3)
        """
        # Compute rotation matrix derivatives along first batch dimension
        dR = np.gradient(rot_mats, dt, axis=0)  # (N, ..., 3, 3)
        
        # Slice rotation matrices to match derivative shape
        R = rot_mats  # (N-2, ..., 3, 3)
        
        # Convert to skew-symmetric matrices
        # Transpose R to align with dR for matrix multiplication
        R_T = np.swapaxes(R, -2, -1)  # (N-2, ..., 3, 3)
        w_mat = np.matmul(dR, R_T)  # (N-2, ..., 3, 3)
        
        # Extract angular velocity vector from skew-symmetric matrix
        ang_vel = np.stack([
            -w_mat[..., 1, 2] + w_mat[..., 2, 1],  # x component
            w_mat[..., 0, 2] - w_mat[..., 2, 0],   # y component
            -w_mat[..., 0, 1] + w_mat[..., 1, 0]   # z component
        ], axis=-1) / 2.0  # (N-2, ..., 3)
        
        # Pad to match original sequence length
        ang_vel = np.pad(ang_vel, ((1,1), *[(0,0) for _ in range(ang_vel.ndim-2)], (0,0)), 
                        mode='edge')  # (N, ..., 3)
        
        return ang_vel
    @staticmethod
    def compute_alignment_rotation(
        forward_dir: np.ndarray, 
        up_dir: np.ndarray = np.array([0, 1, 0])
    ) -> np.ndarray:
        """Compute rotation to align motion with canonical frame.
        
        Args:
            forward_dir: Forward direction vector
            up_dir: Up direction vector (default: y-up)
            
        Returns:
            3x3 rotation matrix
        """
        # Project forward direction to horizontal plane
        forward_flat = forward_dir.copy()
        forward_flat[1] = 0  # Zero out vertical component
        forward_flat /= np.linalg.norm(forward_flat)
        
        # Compute rotation matrix columns
        right = np.cross(up_dir, forward_flat)
        right /= np.linalg.norm(right)
        
        new_up = np.cross(forward_flat, right)
        new_up /= np.linalg.norm(new_up)
        
        # Assemble rotation matrix
        rot_mat = np.stack([right, new_up, forward_flat], axis=1)
        
        return rot_mat 