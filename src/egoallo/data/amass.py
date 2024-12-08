from pathlib import Path
from typing import Any, Literal, assert_never, cast, Union

import h5py
import numpy as np
import torch
import torch.utils
import torch.utils.data

from egoallo.config.train.train_config import EgoAlloTrainConfig

from .dataclass import EgoTrainingData
from ..network import EgoDenoiserConfig


from egoallo.config import make_cfg, CONFIG_FILE
local_config_file = CONFIG_FILE
CFG = make_cfg(config_name="defaults", config_file=local_config_file, cli_args=[])

from egoallo.utils.setup_logger import setup_logger

logger = setup_logger(output=None, name=__name__)

AMASS_SPLITS = {
    "train": [
        "ACCAD",
        "BMLhandball",
        "BMLmovi",
        "BioMotionLab_NTroje",
        "CMU",
        "DFaust_67",
        "DanceDB",
        "EKUT",
        "Eyes_Japan_Dataset",
        "KIT",
        "MPI_Limits",
        "TCD_handMocap",
        "TotalCapture",
    ],
    "val": [
        "HumanEva",
        "MPI_HDM05",
        "MPI_mosh",
        "SFU",
    ],
    "test": [
        "Transitions_mocap",
        "SSM_synced",
    ],
    "test_humor": [
        # HuMoR splits are different...
        "Transitions_mocap",
        "HumanEva",
    ],
    # This is just used for debugging / overfitting...
    "just_humaneva": [
        "HumanEva",
    ],
}


class EgoAmassHdf5Dataset(torch.utils.data.Dataset[EgoTrainingData]):
    """Dataset which loads from our preprocessed hdf5 file.

    Args:
        hdf5_path: Path to the HDF5 file containing the dataset.
        file_list_path: Path to the file containing the list of NPZ files in the dataset.
        splits: List of splits to include in the dataset.
        subseq_len: Length of subsequences to sample from the dataset.
        cache_files: Whether to cache the entire dataset in memory.
        deterministic_slices: Set to True to always use the same slices. This
            is useful for reproducible eval.
    """

    def __init__(
        self,
        config: EgoAlloTrainConfig,
        cache_files: bool,
        random_variable_len_proportion: float = 0.3,
        random_variable_len_min: int = 16,
    ) -> None:

        min_subseq_len = None

        self.config = config
        hdf5_path = config.dataset_hdf5_path
        file_list_path = config.dataset_files_path
        splits = config.train_splits
        subseq_len = config.subseq_len
        slice_strategy = config.dataset_slice_strategy

        datasets = []
        for split in set(splits):
            datasets.extend(AMASS_SPLITS[split])

        self._slice_strategy: Literal[
            "deterministic", "random_uniform_len", "random_variable_len"
        ] = slice_strategy

        self._random_variable_len_proportion = random_variable_len_proportion
        self._random_variable_len_min = random_variable_len_min
        self._hdf5_path = hdf5_path
        self.config = config

        with h5py.File(self._hdf5_path, "r") as hdf5_file:
            self._groups = [
                p
                for p in file_list_path.read_text().splitlines()
                if p.partition("/")[0] in datasets
                and cast(
                    h5py.Dataset,
                    cast(h5py.Group, hdf5_file[p])["T_world_root"],
                ).shape[0]
                >= (subseq_len if min_subseq_len is None else min_subseq_len)
                # These datasets had weird joint positions in the original
                # version of the processed data. They should be fine now.
                # and not p.endswith("KIT/317/run05_poses.npz")
                # and not p.endswith("KIT/424/run02_poses.npz")
            ]
            self._subseq_len = subseq_len
            assert len(self._groups) > 0
            assert len(cast(h5py.Group, hdf5_file[self._groups[0]]).keys()) > 0

            # Number of subsequences we would have to sample in order to see each timestep once.
            # This is an underestimate, since sampled subsequences can overlap.
            self._approximated_length = (
                sum(
                    cast(
                        h5py.Dataset, cast(h5py.Group, hdf5_file[g])["T_world_root"]
                    ).shape[0]
                    for g in self._groups
                )
                // subseq_len
            )

        self._cache: dict[str, dict[str, Any]] | None = {} if cache_files else None

    def __getitem__(self, index: int) -> EgoTrainingData:
        group_index = index % len(self._groups)
        slice_index = index // len(self._groups)
        del index

        # Get group corresponding to a single NPZ file.
        group = self._groups[group_index]

        # We open the file only if we're not loading from the cache.
        hdf5_file = None

        if self._cache is not None:
            if group not in self._cache:
                hdf5_file = h5py.File(self._hdf5_path, "r")
                assert hdf5_file is not None
                self._cache[group] = {
                    k: np.array(v)
                    for k, v in cast(h5py.Group, hdf5_file[group]).items()
                }
            npz_group = self._cache[group]
        else:
            hdf5_file = h5py.File(self._hdf5_path, "r")
            npz_group = hdf5_file[group]
            assert isinstance(npz_group, h5py.Group)

        total_t = cast(h5py.Dataset, npz_group["T_world_root"]).shape[0]
        assert total_t >= self._subseq_len

        # Determine slice indexing.
        mask = torch.ones(self._subseq_len, dtype=torch.bool)

        if self._slice_strategy == "full_sequence":
            start_t, end_t = 0, total_t

        elif self._slice_strategy == "deterministic":
            # A deterministic, non-overlapping slice.
            valid_start_indices = total_t - self._subseq_len
            start_t = (
                (slice_index * self._subseq_len) % valid_start_indices
                if valid_start_indices > 0
                else 0
            )
            end_t = start_t + self._subseq_len
        elif self._slice_strategy == "random_uniform_len":
            # Sample a random slice. Ideally we could make this more reproducible...
            start_t = np.random.randint(0, total_t - self._subseq_len + 1)
            end_t = start_t + self._subseq_len
        elif self._slice_strategy == "random_variable_len":
            # Sample a random slice. Ideally we could make this more reproducible...
            random_subseq_len = min(
                # With 30% likelihood, sample a shorter subsequence.
                (
                    np.random.randint(self._random_variable_len_min, self._subseq_len)
                    if np.random.random() < self._random_variable_len_proportion
                    # Otherwise, use the full subsequence.
                    else self._subseq_len
                ),
                total_t,
            )
            start_t = np.random.randint(0, total_t - random_subseq_len + 1)
            end_t = start_t + random_subseq_len
            mask[random_subseq_len:] = False
        else:
            assert_never(self._slice_strategy)

        # Read slices of the dataset.
        kwargs: dict[str, Any] = {}
        for k in npz_group.keys():
            # Possibly saved in the hdf5 file, but we don't need/want to read them.
            # if k == "joints_wrt_world":
            #     continue

            v = npz_group[k]
            assert isinstance(k, str)
            assert isinstance(v, (h5py.Dataset, np.ndarray))
            if k == "betas":
                assert v.shape == (1, 16)
                array = v[:]
            else:
                assert v.shape[0] == total_t
                array = v[start_t:end_t]

            # Only pad if not using full_sequence
            if self._slice_strategy != "full_sequence" and array.shape[0] != self._subseq_len:
                array = np.concatenate(
                    [
                        array,
                        np.repeat(
                            array[-1:,], self._subseq_len - array.shape[0], axis=0
                        ),
                    ],
                    axis=0,
                )
            kwargs[k] = torch.from_numpy(array)
        kwargs["mask"] = mask

        # Older versions of the processed dataset don't have hands.
        if "hand_quats" not in kwargs:
            kwargs["hand_quats"] = None

        subseq_len = self._subseq_len if self._slice_strategy != "full_sequence" else total_t
        # Generate MAE-style masking
        num_joints = CFG.smplh.num_joints
        device = kwargs["joints_wrt_world"].device

        # Generate random mask for sequence
        num_masked = int(num_joints * self.config.mask_ratio)
        visible_joints_mask = torch.ones((subseq_len, num_joints), dtype=torch.bool, device=device)
        
        # * Randomly select joints to mask, all data within a timestep is masked together, across batch is different.
        rand_indices = torch.randperm(num_joints)
        masked_indices = rand_indices[:num_masked]
        visible_joints_mask[:, masked_indices] = False
        # breakpoint()

        # Get original joints_wrt_world
        joints_wrt_world = kwargs["joints_wrt_world"]  # shape: [time, 21, 3]
        if self._slice_strategy != "full_sequence":
            assert joints_wrt_world.shape == (self._subseq_len, num_joints, 3)
        else:
            assert joints_wrt_world.shape == (total_t, num_joints, 3)
        
        # Create visible_joints tensor containing only unmasked joints
        visible_joints = joints_wrt_world[visible_joints_mask].reshape(subseq_len, num_joints - num_masked, 3)

        # Update kwargs with new MAE-style masking tensors
        kwargs["visible_joints"] = visible_joints
        kwargs["visible_joints_mask"] = visible_joints_mask
        kwargs["joints_wrt_world"] = joints_wrt_world  # Keep original joints for computing loss

        # Close the file if we opened it.
        if hdf5_file is not None:
            hdf5_file.close()

        return EgoTrainingData(**kwargs)

    def __len__(self) -> int:
        return self._approximated_length


class AdaptiveAmassHdf5Dataset(torch.utils.data.Dataset[EgoTrainingData]):
    """Dataset that loads from a preprocessed HDF5 file with dynamic window support."""

    def __init__(self, config: "EgoAlloTrainConfig") -> None:
        """Initialize dataset with configuration parameters.

        Args:
            config (EgoAlloTrainConfig): Configuration object containing dataset parameters.
        """
        self.config = config
        self._hdf5_path = config.dataset_hdf5_path
        self._subseq_len = config.subseq_len
        self._slice_strategy = config.dataset_slice_strategy
        self._random_variable_len_proportion = config.dataset_slice_random_variable_len_proportion
        self._random_variable_len_min = 16
        self._mask_ratio = config.mask_ratio

        # Initialize groups and cache
        with h5py.File(self._hdf5_path, "r") as hdf5_file:
            self.min_seq_len = self._subseq_len  # Removed conditioning on previous window
            self._groups = self._initialize_groups(hdf5_file)
            self._group_lengths = self._calculate_group_lengths(hdf5_file)
            self._cum_len = np.cumsum(self._group_lengths)

            # Cache for better performance
            self._cache: dict[str, dict[str, np.ndarray[Any, Any]]] = {}

    def _initialize_groups(self, hdf5_file: h5py.File) -> list[str]:
        """Initialize groups based on the HDF5 file content.

        Args:
            datasets (list[str]): List of dataset names.
            hdf5_file (h5py.File): Open HDF5 file object.

        Returns:
            list[str]: List of groups that meet the criteria.
        """
        # Get all paths from the text file
        all_paths = self.config.dataset_files_path.read_text().splitlines()
        
        # Filter paths that start with any of the split names
        split_prefixes = [split + '/' for split in self.config.train_splits]
        groups = [
            p for p in all_paths
            if any(p.startswith(prefix) for prefix in split_prefixes) and
            cast(h5py.Dataset, cast(h5py.Group, hdf5_file[p])["T_world_root"]).shape[0] >= self.min_seq_len
        ]
        
        assert len(groups) > 0, f"No valid groups found for splits: {self.config.train_splits}"
        assert len(cast(h5py.Group, hdf5_file[groups[0]]).keys()) > 0, f"First group {groups[0]} has no keys"
        
        return groups

    def _calculate_group_lengths(self, hdf5_file: h5py.File) -> list[int]:
        """Calculate the lengths of each group in the dataset.

        Args:
            hdf5_file (h5py.File): Open HDF5 file object.

        Returns:
            list[int]: List of lengths for each group.
        """
        return [
            cast(h5py.Dataset, cast(h5py.Group, hdf5_file[g])["T_world_root"]).shape[0] - self._subseq_len
            for g in self._groups
        ]

    def __getitem__(self, index: int) -> EgoTrainingData:
        """Retrieve an item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            EgoTrainingData: Data object containing the requested item.
        """
        global_index = index * self._subseq_len  # Directly using global index without previous window conditioning
        group_index, slice_index = self._find_group_and_slice_index(global_index)

        group = self._groups[group_index]
        npz_group = self._get_npz_group(group)
        total_t = cast(h5py.Dataset, npz_group["T_world_root"]).shape[0]
        assert total_t >= self.min_seq_len

        # Calculate slice indices for the current window
        start_t = slice_index
        end_t = min(start_t + self._subseq_len, total_t)

        # Load current window data
        kwargs = self._load_sequence_data(group, start_t, end_t, total_t)
        kwargs["mask"] = torch.ones(self._subseq_len, dtype=torch.bool)

        # Add MAE-style masking
        # breakpoint()
        num_joints = CFG.smplh.num_joints
        device = kwargs["joints_wrt_world"].device

        # Generate random mask for sequence
        num_masked = int(num_joints * self._mask_ratio)
        visible_joints_mask = torch.ones((self._subseq_len, num_joints), dtype=torch.bool, device=device)
        
        # * Randomly select joints to mask, all data within a timestep is masked together, across batch is different.
        rand_indices = torch.randperm(num_joints)
        masked_indices = rand_indices[:num_masked]
        visible_joints_mask[:, masked_indices] = False

        # Get original joints_wrt_world
        # Combine root position from T_world_root with other joints to get full 22 joints
        assert kwargs["joints_wrt_world"].shape == (self._subseq_len, num_joints-1, 3)
        root_pos = kwargs["T_world_root"][..., 4:7]  # Get translation part [time, 3]
        joints_wrt_world = torch.cat([
            root_pos.unsqueeze(1),  # Add joint dimension: [time, 1, 3] 
            kwargs["joints_wrt_world"]  # [time, 21, 3]
        ], dim=1)  # Final shape: [time, 22, 3]
        assert joints_wrt_world.shape == (self._subseq_len, num_joints, 3), f"Expected shape: {(self._subseq_len, num_joints, 3)}, got: {joints_wrt_world.shape}"
        
        # Create visible_joints tensor containing only unmasked joints
        visible_joints = joints_wrt_world[visible_joints_mask].reshape(self._subseq_len, num_joints - num_masked, 3)

        # Update kwargs with new MAE-style masking tensors
        kwargs["visible_joints"] = visible_joints
        kwargs["visible_joints_mask"] = visible_joints_mask
        kwargs["joints_wrt_world"] = joints_wrt_world  # Keep original joints for computing loss

        return EgoTrainingData(**kwargs)

    def _load_sequence_data(self, group: str, start_t: int, end_t: int, total_t: int) -> dict[str, Any]:
        """Load sequence data from HDF5 file or cache.

        Args:
            group (str): Group name to load data from.
            start_t (int): Start time index for loading data.
            end_t (int): End time index for loading data.
            total_t (int): Total time steps available.

        Returns:
            dict[str, Any]: Dictionary containing loaded data.
        """
        npz_group = self._get_npz_group(group)
        kwargs: dict[str, Any] = {}

        for k in npz_group.keys():
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

    def _get_npz_group(self, group: str) -> Union[h5py.Group, dict[str, np.ndarray[Any, Any]]]:
        """Get NPZ group from cache or HDF5 file.

        Args:
            group (str): Group name to retrieve.

        Returns:
            Union[h5py.Group, dict[str, np.ndarray[Any, Any]]]: NPZ group data.
        """
        if group not in self._cache:
            with h5py.File(self._hdf5_path, "r") as hdf5_file:
                self._cache[group] = {
                    k: np.array(v)
                    for k, v in cast(h5py.Group, hdf5_file[group]).items()
                }
        return self._cache[group]

    def _find_group_and_slice_index(self, global_index: int) -> tuple[int, int]:
        """Find the group and slice index from the global index.

        Args:
            global_index (int): Global index to map.

        Returns:
            tuple[int, int]: Tuple containing group index and slice index.
        """
        group_index = np.searchsorted(self._cum_len, global_index, side='right')
        slice_index = global_index - (self._cum_len[group_index - 1] if group_index > 0 else 0)
        return int(group_index), int(slice_index)

    def __len__(self) -> int:
        """Return the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return self._cum_len[-1] // self._subseq_len
