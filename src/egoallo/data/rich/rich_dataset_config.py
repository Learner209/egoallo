"""Configuration for RICH dataset processing."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import List

import torch


@dataclasses.dataclass
class RICHDatasetConfig:
    """Configuration for RICH dataset preprocessing.
    
    This class provides a simple interface for configuring the RICHDataProcessor
    with type checking and validation.
    """
    # Dataset paths
    rich_data_dir: Path = Path("./third_party/rich_toolkit")
    """Path to RICH dataset root directory"""
    
    smplx_model_dir: Path = Path("./third_party/rich_toolkit/body_models/smplx")
    """Path to SMPL-X model directory"""
    
    output_dir: Path = Path("./data/rich/processed_data")
    """Directory for saving processed sequences"""
    
    output_list_file: Path = Path("./data/rich/processed_data/rich_dataset_files.txt")
    """File to save list of processed sequences"""
    
    # Processing options
    target_fps: int = 30
    """Target frames per second for sequences"""
    
    min_sequence_length: int = 30
    """Minimum number of frames per sequence"""
    
    max_sequence_length: int = 300
    """Maximum number of frames per sequence"""
    
    include_contact: bool = True
    """Whether to include contact annotations"""
    
    use_pca: bool = True
    """Whether to use PCA for hand poses"""
    
    num_processes: int = 4
    """Number of parallel processes for preprocessing"""
    
    # Data split options
    splits: List[str] = dataclasses.field(
        default_factory=lambda: ["train", "val", "test"]
    )
    """Dataset splits to process"""
    
    # Device options
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    """Device to use for processing"""
    
    # Debug options
    debug: bool = False
    """Whether to enable debug mode"""
    
    def __post_init__(self) -> None:
        """Convert paths to Path objects and validate configuration."""
        # Convert string paths to Path objects
        self.rich_data_dir = Path(self.rich_data_dir)
        self.smplx_model_dir = Path(self.smplx_model_dir)
        self.output_dir = Path(self.output_dir)
        self.output_list_file = Path(self.output_list_file)
        
        # Validate paths
        if not self.rich_data_dir.exists():
            raise ValueError(f"RICH dataset directory not found: {self.rich_data_dir}")
        if not self.smplx_model_dir.exists():
            raise ValueError(f"SMPL-X model directory not found: {self.smplx_model_dir}")
            
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_list_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Validate numeric parameters
        if self.target_fps <= 0:
            raise ValueError(f"Target FPS must be positive, got {self.target_fps}")
        if self.min_sequence_length <= 0:
            raise ValueError(
                f"Min sequence length must be positive, got {self.min_sequence_length}"
            )
        if self.max_sequence_length <= self.min_sequence_length:
            raise ValueError(
                f"Max sequence length ({self.max_sequence_length}) must be greater "
                f"than min sequence length ({self.min_sequence_length})"
            )
        if self.num_processes <= 0:
            raise ValueError(f"Number of processes must be positive, got {self.num_processes}")
            
        # Validate splits
        valid_splits = {"train", "val", "test"}
        invalid_splits = set(self.splits) - valid_splits
        if invalid_splits:
            raise ValueError(f"Invalid splits: {invalid_splits}. Must be one of {valid_splits}")
    
    def get_processor_kwargs(self) -> dict:
        """Get kwargs dictionary for initializing RICHDataProcessor.
        
        Returns:
            Dictionary of parameters for RICHDataProcessor initialization
        """
        return {
            "rich_data_dir": str(self.rich_data_dir),
            "smplx_model_dir": str(self.smplx_model_dir),
            "output_dir": str(self.output_dir),
            "fps": self.target_fps,
            "include_contact": self.include_contact,
            "use_pca": self.use_pca,
            "device": self.device
        } 