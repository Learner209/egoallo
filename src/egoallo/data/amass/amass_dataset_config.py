"""Configuration for AMASS dataset processing."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import List, Any

import torch


@dataclasses.dataclass
class AMASSDatasetConfig:
    """Configuration for AMASS dataset preprocessing."""
    
    # Dataset paths
    amass_dir: Path = Path("datasets/AMASS/SMPLH_G")
    """Path to AMASS dataset root"""
    
    smplh_dir: Path = Path("./assets/smpl_based_model/smplh")
    """Path to SMPL model directory"""
    
    output_dir: Path = Path("./data/amass/processed")
    """Directory for saving processed sequences"""
    
    output_list_file: Path = Path("./data/amass/processed/amass_dataset_files.txt")
    """File to save list of processed sequences"""
    
    # Processing options
    target_fps: int = 30
    """Target frames per second"""
    
    min_sequence_length: int = 30
    """Minimum number of frames per sequence"""
    
    include_velocities: bool = True
    """Whether to compute velocities"""
    
    include_align_rot: bool = True
    """Whether to compute alignment rotations"""
    
    num_processes: int = 1
    """Number of parallel processes"""
    
    # Dataset splits
    train_datasets: List[str] = dataclasses.field(
        default_factory=lambda: [
            'CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset',
            'KIT', 'BioMotionLab_NTroje', 'BMLmovi', 'EKUT', 'ACCAD'
        ]
    )
    
    val_datasets: List[str] = dataclasses.field(
        default_factory=lambda: ['MPI_HDM05', 'SFU', 'MPI_mosh']
    )
    
    test_datasets: List[str] = dataclasses.field(
        default_factory=lambda: ['Transitions_mocap', 'HumanEva']
    )
    
    # Device options
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    """Device to use for processing"""
    
    # Debug options
    debug: bool = False
    """Whether to enable debug mode"""
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        # Convert paths
        self.amass_dir = Path(self.amass_dir)
        self.smplh_dir = Path(self.smplh_dir)
        self.output_dir = Path(self.output_dir)
        self.output_list_file = Path(self.output_list_file)
        
        # Validate paths
        if not self.amass_dir.exists():
            raise ValueError(f"AMASS directory not found: {self.amass_dir}")
        if not self.smplh_dir.exists():
            raise ValueError(f"SMPL directory not found: {self.smplh_dir}")
            
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_list_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Validate parameters
        if self.target_fps <= 0:
            raise ValueError(f"Target FPS must be positive, got {self.target_fps}")
        if self.min_sequence_length <= 0:
            raise ValueError(
                f"Min sequence length must be positive, got {self.min_sequence_length}"
            )
        if self.num_processes <= 0:
            raise ValueError(f"Number of processes must be positive, got {self.num_processes}")
    
    def get_processor_kwargs(self) -> dict[str, Any]:
        """Get kwargs for AMASSProcessor initialization."""
        return {
            "amass_dir": str(self.amass_dir),
            "smplh_dir": str(self.smplh_dir),
            "output_dir": str(self.output_dir),
            "fps": self.target_fps,
            "include_velocities": self.include_velocities,
            "include_align_rot": self.include_align_rot,
            "device": self.device
        } 