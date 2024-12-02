from __future__ import annotations
import dataclasses
from pathlib import Path
import torch

from egoallo.config.train.train_config import EgoAlloTrainConfig

@dataclasses.dataclass
class InferenceConfig(EgoAlloTrainConfig):
    """Configuration for inference."""
    traj_length: int = 128
    """Length of trajectory sequence to process"""
    
    num_samples: int = 1
    """Number of samples to generate during inference"""
    
    batch_size: int = 1
    """Batch size for inference"""
    
    smplh_npz_path: Path = Path("./data/smplh/neutral/model.npz")
    """Path to SMPL+H model file"""
    
    output_dir: Path = Path("./outputs")
    """Directory to save inference outputs"""
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    """Device to run inference on"""
    
    annotation_path: Path = Path("./data/egoexo-default-gt-output")
    """Path to ground truth annotations"""
    
    mask_ratio: float = 0.75
    """Ratio of joints to mask during inference"""
    
    checkpoint_dir: Path = Path("./experiments/nov_29_absrel_jnts_pilot/v2")
    """Directory containing model checkpoints"""

    # Evaluation configs
    compute_metrics: bool = True
    """Whether to compute evaluation metrics after inference"""
    
    use_mean_body_shape: bool = False
    """Whether to use mean body shape for evaluation"""
    
    skip_eval_confirm: bool = True
    """Whether to skip confirmation before evaluation"""
    
    num_workers: int = 4
    """Number of workers for data loading"""
    
    visualize_traj: bool = False
    """Whether to visualize trajectories"""

    # Dataset configs
    split: str = "train"
    """Dataset split to use"""
    
    anno_type: str = "manual"
    """Type of annotations to use"""
    
    guidance_post: bool = False
    """Post guidance weight"""
    
    guidance_inner: bool = False
    """Inner guidance weight"""
