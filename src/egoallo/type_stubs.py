from pathlib import Path
from typing import Dict
from typing import Literal
from typing import Tuple
from typing import Union
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from egoallo.network import AbsoluteDenoiseTraj
    from egoallo.network import JointsOnlyTraj
    from egoallo.network import VelocityDenoiseTraj
    from egoallo.network_aadecomp import AbsoluteDenoiseTrajAADecomp

    from egoallo.data.dataclass import EgoTrainingData
    from egoallo.data.dataclass_aadecomp import EgoTrainingDataAADecomp

from egoallo.middleware.third_party.HybrIK.hybrik.models.layers.smplh.fncsmplh import (
    SmplhModel,
    SmplhShaped,
    SmplhShapedAndPosed,
    SmplhMesh,
)
from egoallo.middleware.third_party.HybrIK.hybrik.models.layers.smplx.fncsmplx_aadecomp import (
    SmplxModelAADecomp,
    SmplxShapedAADecomp,
    SmplxShapedAndPosedAADecomp,
    SmplxMeshAADecomp,
)
from egoallo.middleware.third_party.HybrIK.hybrik.models.layers.smplh.fncsmplh_aadecomp import (
    SmplhModelAADecomp,
    SmplhShapedAADecomp,
    SmplhShapedAndPosedAADecomp,
    SmplhMeshAADecomp,
)
from egoallo.middleware.third_party.HybrIK.hybrik.models.layers.smpl.fncsmpl_aadecomp import (
    SmplModelAADecomp,
    SmplShapedAADecomp,
    SmplShapedAndPosedAADecomp,
    SmplMeshAADecomp,
)

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
    "hamer_reproj2",
]

# Tensor types
JointTransforms = Float[Tensor, "time 21 7"]
RootTransforms = Float[Tensor, "time 7"]
BatchedJointTransforms = Float[Tensor, "batch time 21 7"]
BatchedRootTransforms = Float[Tensor, "batch time 7"]
Points3D = Float[Tensor, "*batch N 3"]

# Dictionary types
MetricsDict = Dict[str, float]
StatsDict = Dict[str, FloatArray]

# Output types for procrustes alignment
ProcrustesOutput = Union[Tuple[Tensor, Tensor, Tensor], Tensor]
ProcrustesMode = Literal["transforms", "aligned_x"]

# Evaluation types
EvalMode = Union[Literal["hamer"], GuidanceMode]

# Dataset type (for both training and inference)
DatasetType = Literal[
    "AdaptiveAmassHdf5Dataset",
    "VanillaEgoAmassHdf5Dataset",
    "EgoExoDataset",
    "AriaDataset",
    "AriaInferenceDataset",
]

DatasetSliceStrategy = Literal[
    "deterministic",
    "random_uniform_len",
    "random_variable_len",
    "full_sequence",
]

DatasetSplit = Literal["train", "val", "test", "just_humaneva"]

JointCondMode = Literal[
    "absolute",
    "absrel_jnts",
    "absrel",
    "absrel_global_deltas",
    "vel_acc",
    "joints_only",
]

LossWeights = Dict[str, float]

DenoiseTrajType = Union[
    "AbsoluteDenoiseTraj",
    "JointsOnlyTraj",
    "VelocityDenoiseTraj",
    "AbsoluteDenoiseTrajAADecomp",
]
DenoiseTrajTypeLiteral = Literal[
    "AbsoluteDenoiseTraj",
    "JointsOnlyTraj",
    "VelocityDenoiseTraj",
]

EgoTrainingDataType = Union["EgoTrainingData", "EgoTrainingDataAADecomp"]
EgoTrainingDataTypeLiteral = Literal[
    "EgoTrainingData",
    "EgoTrainingDataAADecomp",
]

SmplFamilyModelType = Union[
    "SmplhModel",
    "SmplhShaped",
    "SmplhShapedAndPosed",
    "SmplhMesh",
    "SmplxModelAADecomp",
    "SmplxShapedAADecomp",
    "SmplxShapedAndPosedAADecomp",
    "SmplxMeshAADecomp",
    "SmplhModelAADecomp",
    "SmplhShapedAADecomp",
    "SmplhShapedAndPosedAADecomp",
    "SmplhMeshAADecomp",
    "SmplModelAADecomp",
    "SmplShapedAADecomp",
    "SmplShapedAndPosedAADecomp",
    "SmplMeshAADecomp",
]
SmplFamilyModelTypeLiteral = Literal[
    "SmplhModel",
    "SmplhShaped",
    "SmplhShapedAndPosed",
    "SmplhMesh",
    "SmplxModelAADecomp",
    "SmplxShapedAADecomp",
    "SmplxShapedAndPosedAADecomp",
    "SmplxMeshAADecomp",
    "SmplhModelAADecomp",
    "SmplhShapedAADecomp",
    "SmplhShapedAndPosedAADecomp",
    "SmplhMeshAADecomp",
    "SmplModelAADecomp",
    "SmplShapedAADecomp",
    "SmplShapedAndPosedAADecomp",
    "SmplMeshAADecomp",
]
