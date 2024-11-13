from dataclasses import dataclass
from typing import Literal
import torch

from .base_config import EgoAlloBaseConfig

@dataclass(frozen=False)
class EgoAlloTrainConfig(EgoAlloBaseConfig):
    """Training-specific configuration."""
    
    # Experiment settings
    experiment_name: str = "april13"
    
    # Training splits
    train_splits: tuple[Literal["train", "val", "test", "just_humaneva"], ...] = ("train",)
    
    # Conditioning settings
    condition_on_prev_window: bool = False
    
    # Optimizer settings
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Learning rate settings
    base_batch_size: int = 256
    base_learning_rate: float = 1e-4
    learning_rate_scaling: Literal["sqrt", "linear", "none"] = "sqrt"
    
    def __post_init__(self):
        # # Update model config with conditioning setting
        # self.model = self.model.__class__(condition_on_prev_window=self.condition_on_prev_window) 
        pass