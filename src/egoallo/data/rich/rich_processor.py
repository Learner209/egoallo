from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from smplx import SMPLX
import trimesh

from egoallo.setup_logger import setup_logger

logger = setup_logger(output="logs/rich_processor", name=__name__)

class RICHDataProcessor:
    """Process RICH dataset sequences for training and evaluation.
    
    This class handles loading and preprocessing of RICH dataset sequences,
    including SMPL-X parameters and human-scene contact annotations.
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

    def load_frame_data(
        self, split: str, seq_name: str, frame_id: int
    ) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Load SMPL-X parameters and contact data for a single frame.
        
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
                
        return body_params, contact_verts, contact_vecs

    def load_sequence(self, split: str, seq_name: str) -> Dict[str, Any]:
        """Load a complete sequence including all frames.
        
        Args:
            split: Dataset split (train/val/test)
            seq_name: Name of the sequence
            
        Returns:
            Dictionary containing:
                - body_params: List of SMPL-X parameters per frame
                - contact_verts: List of contact vertex indices per frame
                - contact_vecs: List of contact displacement vectors per frame
                - frame_ids: List of frame IDs
                - gender: Subject gender
                - fps: Sequence frame rate
        """
        scene_name, sub_id, _ = seq_name.split("_")
        seq_dir = self.rich_data_dir / "data/bodies" / split / seq_name
        
        # Get sorted frame IDs
        frame_ids = sorted([
            int(d.name) for d in seq_dir.iterdir() 
            if d.is_dir() and d.name.isdigit()
        ])
        
        # Initialize sequence data
        sequence_data = {
            "body_params": [],
            "contact_verts": [] if self.include_contact else None,
            "contact_vecs": [] if self.include_contact else None,
            "frame_ids": frame_ids,
            "gender": self.gender_mapping[f"{int(sub_id)}"],
            "fps": self.fps
        }
        
        # Load data for each frame
        for frame_id in frame_ids:
            body_params, contact_verts, contact_vecs = self.load_frame_data(
                split, seq_name, frame_id
            )
            sequence_data["body_params"].append(body_params)
            if self.include_contact:
                sequence_data["contact_verts"].append(contact_verts)
                sequence_data["contact_vecs"].append(contact_vecs)
                
        return sequence_data

    def save_sequence(self, sequence_data: Dict[str, Any], output_path: Path) -> None:
        """Save processed sequence data to NPZ file.
        
        Args:
            sequence_data: Dictionary containing processed sequence data
            output_path: Path to save the NPZ file
        """
        # Convert lists to numpy arrays
        save_data = {
            "body_params": np.array(sequence_data["body_params"]),
            "frame_ids": np.array(sequence_data["frame_ids"]),
            "gender": sequence_data["gender"],
            "fps": sequence_data["fps"]
        }
        
        # Add contact data if included
        if self.include_contact:
            save_data["contact_verts"] = np.array(
                sequence_data["contact_verts"], dtype=object
            )
            save_data["contact_vecs"] = np.array(
                sequence_data["contact_vecs"], dtype=object
            )
            
        # Save compressed NPZ file
        np.savez_compressed(output_path, **save_data)
        logger.info(f"Saved processed sequence to {output_path}") 