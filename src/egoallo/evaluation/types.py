from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from jaxtyping import Float
from numpy.typing import NDArray
from torch import Tensor

# Type aliases
PathLike = Union[str, Path]
Device = Union[str, torch.device]

# Numpy array types
FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int32]
BoolArray = NDArray[np.bool_]

# Model types
ModelType = Literal["smplh", "mano", "smplx"]
HandSide = Literal["left", "right"]
GuidanceMode = Literal[
    "hamer",
    "no_hands",
    "aria_wrist_only", 
    "aria_hamer",
    "hamer_wrist",
    "hamer_reproj2"
]

# Tensor types
JointTransforms = Float[Tensor, "time 21 7"]
RootTransforms = Float[Tensor, "time 7"]
BatchedJointTransforms = Float[Tensor, "num_samples time 21 7"]
BatchedRootTransforms = Float[Tensor, "num_samples time 7"]
Points3D = Float[Tensor, "*batch N 3"]

# Dictionary types
MetricsDict = Dict[str, float]
StatsDict = Dict[str, FloatArray]

# Output types for procrustes alignment
ProcrustesOutput = Union[Tuple[Tensor, Tensor, Tensor], Tensor]
ProcrustesMode = Literal["transforms", "aligned_x"]

# Evaluation types
EvalMode = Union[Literal["hamer"], GuidanceMode]