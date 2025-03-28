from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

from egoallo.types import DatasetType, DatasetSliceStrategy, DatasetSplit
from egoallo.guidance_optimizer_jax import GuidanceMode
from typing import Optional


@dataclasses.dataclass
class InferenceConfig:
    """Configuration for inference."""

    start_index: int = 0
    """Start index of trajectory sequence to process"""

    traj_length: int = 128
    """Length of trajectory sequence to process"""

    num_samples: int = 1
    """Number of samples to generate during inference"""

    batch_size: int = 1
    """Batch size for inference"""

    smplh_model_path: Path = Path("assets/smpl_based_model/smplh/SMPLH_MALE.pkl")
    """Path to SMPL+H model file"""

    output_dir: Path = Path("./exp/test-amass")
    """Directory to save inference outputs"""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    """Device to run inference on"""

    annotation_path: Path = Path("./data/egoexo-default-gt-output")
    """Path to ground truth annotations"""

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

    random_visualize: bool = True
    """Whether to randomly visualize trajectories, Default is to visualize all trajectories."""

    online_render: bool = False
    """Whether to online render trajectories in real-time. This would pop-out a pyrender interactive window"""

    guidance_post: bool = False
    """Post guidance weight"""

    guidance_inner: bool = False
    """Inner guidance weight"""

    guidance_mode: GuidanceMode = "aria_hamer"
    """Which guidance mode to use."""

    dataset_type: DatasetType = "AdaptiveAmassHdf5Dataset"
    """Dataset type to use"""

    dataset_slice_strategy: DatasetSliceStrategy = "full_sequence"
    """Dataset slice strategy to use"""

    splits: tuple[DatasetSplit, ...] = ("test",)
    """Dataset splits to use"""

    debug_max_iters: Optional[int] = None
    """Maximum number of iterations for debugging"""
