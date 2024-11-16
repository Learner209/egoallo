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
    base_batch_size: int = 256
    base_learning_rate: float = 1e-4
    learning_rate_scaling: Literal["sqrt", "linear", "none"] = "sqrt"

    # Max steps
    max_steps: int = 100000
    
    # Loss settings
    loss_config: Dict[str, Any] = field(default_factory=lambda: {
        "loss_weights": {
            "body_rot6d": 1.0,
            "betas": 0.1,
            "contacts": 0.1
        },
        "cond_dropout_prob": 0.0,
        "weight_loss_by_t": "emulate_eps_pred"
    })
    
    # EMA settings
    use_ema: bool = True
    ema_decay: float = 0.9999
    
    def __post_init__(self):
        super().__post_init__()
        self.model = self.model.__class__(condition_on_prev_window=self.condition_on_prev_window)
