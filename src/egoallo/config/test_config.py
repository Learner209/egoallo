from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from egoallo.guidance_optimizer_jax import GuidanceMode
from .train_config import EgoAlloTrainConfig


@dataclass(frozen=False)
class TestConfig(EgoAlloTrainConfig):
    """Test-specific configuration."""

    # Model and checkpoint settings
    checkpoint_dir: Path = Path("./egoallo_checkpoint_april13/checkpoints_3000000")
    smplh_npz_path: Path = Path("./data/smplh/neutral/model.npz")

    # Test settings
    num_inference_steps: int = 50
    guidance_scale: float = 3.0
    num_samples: int = 1

    # Guidance settings
    guidance_mode: GuidanceMode = "no_hands"
    guidance_inner: bool = True
    guidance_post: bool = True

    # Device settings
    dtype: Literal["float16", "float32"] = (
        "float16" if torch.cuda.is_available() else "float32"
    )
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Output settings
    output_dir: Path = Path("./test_results")

    # Evaluation settings
    compute_metrics: bool = True
    use_mean_body_shape: bool = True
    skip_eval_confirm: bool = True

    # Dataset settings
    batch_size: int = 1  # Override default of 256

    def __post_init__(self):
        """Initialize after dataclass creation."""
        # Override train splits with test splits
        self.train_splits = ("test",)
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
