from __future__ import annotations

import dataclasses

from egoallo.types import DatasetSliceStrategy
from egoallo.types import DatasetType
from typing import Any
from egoallo.config.inference.defaults import InferenceConfig
from egoallo.config.data.egoexo.defaults import EgoExoConfig


@dataclasses.dataclass
class EgoExoInferenceConfig(InferenceConfig):
    """Configuration for EgoExo dataset inference."""

    dataset_slice_strategy: DatasetSliceStrategy = "full_sequence"
    """Strategy for slicing sequences: 'sliding_window' or 'full_sequence'"""

    subseq_len: int = 128
    """Length of subsequences to process"""

    dataset_type: DatasetType = "EgoExoDataset"
    """Dataset type to use"""

    # Add EgoExo config instance
    egoexo: EgoExoConfig = dataclasses.field(default_factory=EgoExoConfig)
    """EgoExo dataset specific configurations"""

    def __getitem__(self, key: str) -> Any:
        """Enable dictionary-style access to attributes."""
        return getattr(self, key)
