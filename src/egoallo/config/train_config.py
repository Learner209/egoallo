from dataclasses import dataclass, field
from typing import Literal, Dict, Any
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
    batch_size: int = 256
    learning_rate: float = 4e-5

    # Max steps
    max_steps: int = 100000
    
    # EMA settings
    use_ema: bool = True
    ema_decay: float = 0.9999
    
    def __post_init__(self):
        super().__post_init__()
        # import ipdb; ipdb.set_trace()
        self.model = self.model.__class__(condition_on_prev_window=self.condition_on_prev_window)
