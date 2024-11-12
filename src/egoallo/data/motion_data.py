"""Data structures for motion data."""
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

@dataclass
class MotionData:
    """Container for motion data."""
    betas: Tensor
    T_world_root: Tensor  
    local_quats: Tensor
    
    @classmethod
    def from_tensor(cls, tensor: Tensor) -> "MotionData":
        """Create MotionData from packed tensor."""
        # Unpack tensor based on your data format
        return cls(
            betas=tensor[..., :16],
            T_world_root=tensor[..., 16:23],
            local_quats=tensor[..., 23:]
        ) 