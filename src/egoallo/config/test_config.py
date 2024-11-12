from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import torch

from .train_config import EgoAlloTrainConfig

@dataclass(frozen=False)
class TestConfig(EgoAlloTrainConfig):
    """Test-specific configuration."""
    
    # Model and checkpoint settings
    checkpoint_dir: Path = Path("./experiments/nov12_first/v0/checkpoints/checkpoint-15000")
    
    # Test settings
    num_inference_steps: int = 50
    guidance_scale: float = 3.0
    num_samples: int = 1
    
    # Device settings
    dtype: Literal["float16", "float32"] = "float16" if torch.cuda.is_available() else "float32"
    
    # Output settings
    output_dir: Path = Path("./test_results")
    
    # Evaluation settings
    compute_metrics: bool = True
    use_mean_body_shape: bool = True
    skip_eval_confirm: bool = False
    
    def __post_init__(self):
        # Override train splits with test splits
        self.train_splits = ("test",)
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True) 