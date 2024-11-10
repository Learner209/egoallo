from pathlib import Path
from typing import Any, Literal, assert_never, cast, Optional, Union, TYPE_CHECKING
from dataclasses import dataclass

import h5py
import numpy as np
import torch
import torch.utils
import torch.utils.data

from .dataclass import EgoTrainingData
from .amass import AMASS_SPLITS

if TYPE_CHECKING:
    from training_pipeline import EgoAlloTrainConfig

@dataclass
class DatasetState:
    """Helper class to store dataset state for prev_window tracking."""
    last_group_index: int = -1
    last_slice_index: int = -1
    prev_window: Optional[EgoTrainingData] = None

class EgoAmassHdf5DatasetDynamic(torch.utils.data.Dataset[EgoTrainingData]):
    """Dataset which loads from our preprocessed hdf5 file with dynamic window support."""

    def __init__(self, config: "EgoAlloTrainConfig") -> None:
        """Initialize dataset with configuration."""
        self.config = config
        self._dataset_state = DatasetState()
        self._hdf5_path = config.dataset_hdf5_path
        self._subseq_len = config.subseq_len
        self._slice_strategy = config.dataset_slice_strategy
        self._random_variable_len_proportion = config.dataset_slice_random_variable_len_proportion
        self._random_variable_len_min = 16

        # Get datasets from splits
        datasets = []
        for split in set(config.train_splits):
            datasets.extend(AMASS_SPLITS[split])

        # Initialize groups and cache
        with h5py.File(self._hdf5_path, "r") as hdf5_file:
            self._groups = [
                p
                for p in config.dataset_files_path.read_text().splitlines()
                if p.partition("/")[0] in datasets
                and cast(
                    h5py.Dataset,
                    cast(h5py.Group, hdf5_file[p])["T_world_root"],
                ).shape[0]
                >= self._subseq_len * 2  # Ensure enough frames for both current and prev window
            ]
            assert len(self._groups) > 0
            assert len(cast(h5py.Group, hdf5_file[self._groups[0]]).keys()) > 0

            # Store group lengths and compute cumulative lengths, accounting for prev window requirement
            self._group_lengths = [
                cast(
                    h5py.Dataset, cast(h5py.Group, hdf5_file[g])["T_world_root"]
                ).shape[0] - self._subseq_len  # Subtract subseq_len to ensure space for prev window
                for g in self._groups
            ]
            self._cum_len = np.cumsum(self._group_lengths)

            # Cache for better performance
            self._cache: dict[str, dict[str, np.ndarray]] = {}

    def _load_sequence_data(self, group: str, start_t: int, end_t: int, total_t: int) -> dict[str, Any]:
        """Load sequence data from HDF5 file or cache."""
        npz_group = self._get_npz_group(group)
        kwargs: dict[str, Any] = {}
        
        for k in npz_group.keys():
            if k == "joints_wrt_world":
                continue

            v = npz_group[k]
            assert isinstance(k, str)
            assert isinstance(v, (h5py.Dataset, np.ndarray))
            
            if k == "betas":
                assert v.shape == (1, 16)
                array = v[:]
            else:
                assert v.shape[0] == total_t
                array = v[start_t:end_t]

            # Pad if necessary
            if array.shape[0] != self._subseq_len:
                array = np.concatenate([
                    array,
                    np.repeat(array[-1:,], self._subseq_len - array.shape[0], axis=0)
                ], axis=0)
                
            kwargs[k] = torch.from_numpy(array)
            
        return kwargs

    def _get_npz_group(self, group: str) -> Union[h5py.Group, dict[str, np.ndarray]]:
        """Get NPZ group from cache or HDF5 file."""
        if group not in self._cache:
            with h5py.File(self._hdf5_path, "r") as hdf5_file:
                self._cache[group] = {
                    k: np.array(v)
                    for k, v in cast(h5py.Group, hdf5_file[group]).items()
                }
        return self._cache[group]

    def __getitem__(self, index: int) -> EgoTrainingData:
        """Get item with prev_window always available."""
        global_index = index * self._subseq_len + self._subseq_len  # Offset by subseq_len to ensure prev window
        group_index, slice_index = self._find_group_and_slice_index(global_index)
        
        # Get group and load data
        group = self._groups[group_index]
        npz_group = self._get_npz_group(group)
        total_t = cast(h5py.Dataset, npz_group["T_world_root"]).shape[0]
        assert total_t >= self._subseq_len * 2

        # Calculate slice indices
        start_t = slice_index + self._subseq_len  # Ensure start_t is at least subseq_len
        end_t = min(start_t + self._subseq_len, total_t)

        # Load current window data
        kwargs = self._load_sequence_data(group, start_t, end_t, total_t)
        kwargs["mask"] = torch.ones(self._subseq_len, dtype=torch.bool)
        
        current_window = EgoTrainingData(**kwargs)
        return current_window

    def _find_group_and_slice_index(self, global_index: int) -> tuple[int, int]:
        """Find group and slice index from global index."""
        group_index = np.searchsorted(self._cum_len, global_index, side='right')
        slice_index = global_index - (self._cum_len[group_index - 1] if group_index > 0 else 0)
        return int(group_index), int(slice_index)

    def __len__(self) -> int:
        return self._cum_len[-1] // self._subseq_len
