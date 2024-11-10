"""Training metrics tracking utilities."""
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional
import torch
from torch import Tensor

@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    step: int
    iterations_per_sec: float
    loss_values: Dict[str, float]
    learning_rate: float
    memory_used: float
    memory_total: float

class MetricsTracker:
    """Tracks and computes training metrics."""
    
    def __init__(self, counter_init: int = 0):
        self.counter = counter_init
        self.last_time = time.time()
        self._iterations = 0
        
    def step(self) -> TrainingMetrics:
        """Update counters and compute metrics for current step."""
        current_time = time.time()
        self._iterations += 1
        self.counter += 1
        
        # Compute iterations per second
        it_per_sec = self._iterations / (current_time - self.last_time)
        
        # Get GPU memory stats
        mem_free, mem_total = torch.cuda.mem_get_info()
        mem_used = (mem_total - mem_free) / 1024**3
        mem_total = mem_total / 1024**3
        
        return TrainingMetrics(
            step=self.counter,
            iterations_per_sec=it_per_sec,
            loss_values={},  # To be filled by caller
            learning_rate=0.0,  # To be filled by caller
            memory_used=mem_used,
            memory_total=mem_total
        )

    def reset_timing(self) -> None:
        """Reset timing counters."""
        self.last_time = time.time()
        self._iterations = 0 