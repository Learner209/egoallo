from __future__ import annotations

from typing import TYPE_CHECKING
import dataclasses
from pathlib import Path
from egoallo.types import DatasetType, DatasetSliceStrategy


if TYPE_CHECKING:
    from egoallo.config.inference.inference_defaults import InferenceConfig


class EgoExoInferenceConfig(InferenceConfig):
    """Configuration for EgoExo dataset inference."""

    # Dataset-specific configs
    bodypose_anno_dir: tuple[Path, ...] = dataclasses.field(default=None)  # type: ignore
    """Paths to body pose annotation directories, only used when dataset_type is EgoExoDataset"""

    dataset_slice_strategy: DatasetSliceStrategy = "full_sequence"
    """Strategy for slicing sequences: 'sliding_window' or 'full_sequence'"""

    subseq_len: int = 128
    """Length of subsequences to process"""

    dataset_type: DatasetType = "EgoExoDataset"
    """Dataset type to use"""
