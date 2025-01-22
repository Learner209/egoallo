from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union, assert_never, cast
from jaxtyping import jaxtyped, Float, Bool
import typeguard

from accelerate.commands.menu.helpers import forceWrite
import h5py
import math
import random
from einops import rearrange
import numpy as np
import torch
import torch.utils
import torch.utils.data
import scipy

if TYPE_CHECKING:
    from egoallo.config.train.train_config import EgoAlloTrainConfig

from egoallo.config import CONFIG_FILE, make_cfg

from ..dataclass import EgoTrainingData

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


# range
class VanillaEgoAmassHdf5Dataset(torch.utils.data.Dataset[EgoTrainingData]):
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
        config: "EgoAlloTrainConfig",
        cache_files: bool = True,
        random_variable_len_proportion: float = 0.3,
        random_variable_len_min: int = 16,
    ) -> None:
        import warnings
        warnings.warn("VanillaEgoAmassHdf5Dataset is deprecated. Use AdaptiveAmassHdf5Dataset instead.", DeprecationWarning, stacklevel=2)

        min_subseq_len = None

        self.config = config
        hdf5_path = config.dataset_hdf5_path
        file_list_path = config.dataset_files_path
        splits = config.splits
        subseq_len = config.subseq_len
        slice_strategy = config.dataset_slice_strategy

        datasets = []
        for split in set(splits):
            datasets.extend(AMASS_SPLITS[split])

        self._slice_strategy: Literal[
            "deterministic",
            "random_uniform_len",
            "random_variable_len",
            "full_sequence",
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
        mask = torch.ones(
            (self._subseq_len if self._slice_strategy != "full_sequence" else total_t),
            dtype=torch.bool,
        )

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
            if (
                self._slice_strategy != "full_sequence"
                and array.shape[0] != self._subseq_len
                and k != "betas"
            ):
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

        subseq_len = (
            self._subseq_len if self._slice_strategy != "full_sequence" else total_t
        )
        # Generate MAE-style masking
        num_joints = CFG.smplh.num_joints
        device = kwargs["joints_wrt_world"].device

        # Generate random mask for sequence
        spatial_mask_ratio = self._get_mask_ratio(mask_type="spatial")
        num_masked = int(num_joints * spatial_mask_ratio)
        visible_joints_mask = torch.ones(
            (subseq_len, num_joints), dtype=torch.bool, device=device
        )

        # * Randomly select joints to mask, all data within a timestep is masked together, across batch is different.
        rand_indices = torch.randperm(num_joints)
        masked_indices = rand_indices[:num_masked]
        visible_joints_mask[:, masked_indices] = False

        # Get original joints_wrt_world
        joints_wrt_world = kwargs["joints_wrt_world"]  # shape: [time, 22, 3]
        if self._slice_strategy != "full_sequence":
            assert (
                joints_wrt_world.shape == (self._subseq_len, num_joints, 3)
            ), f"Expected shape: {(self._subseq_len, num_joints, 3)}, got: {joints_wrt_world.shape}"
        else:
            assert (
                joints_wrt_world.shape == (total_t, num_joints, 3)
            ), f"Expected shape: {(total_t, num_joints, 3)}, got: {joints_wrt_world.shape}"

        # Create visible_joints tensor containing only unmasked joints
        # visible_joints = joints_wrt_world[visible_joints_mask].reshape(subseq_len, num_joints - num_masked, 3)

        # Update kwargs with new MAE-style masking tensors
        # kwargs["visible_joints"] = visible_joints
        kwargs["visible_joints_mask"] = visible_joints_mask
        kwargs["joints_wrt_world"] = (
            joints_wrt_world  # Keep original joints for computing loss
        )

        # Create metadata object first
        metadata = EgoTrainingData.MetaData()
        metadata.stage = "raw"  # Set initial stage
        # uid servers as a null value just for compatibility with EgoExoDataset
        metadata.take_name = (f"name_{group}_uid_{group}_t{start_t}_{end_t}",)
        metadata.scope = "train"
        metadata.dataset_type = "VanillaAmassHdf5Dataset"
        
        # Add metadata to kwargs before creating EgoTrainingData
        kwargs["metadata"] = metadata

        # Close the file if we opened it.
        if hdf5_file is not None:
            hdf5_file.close()

        return EgoTrainingData(**kwargs)

    def __len__(self) -> int:
        return self._approximated_length

    def _get_mask_ratio(self, mask_type: Literal["spatial", "temporal"]) -> float:
        """Get mask ratio - either fixed or randomly sampled"""
        if self.config.random_sample_mask_ratio:
            # Randomly sample between 0~mask_ratio
            if mask_type == "spatial":
                return np.random.uniform(self.config.spatial_mask_ratio / 4, self.config.spatial_mask_ratio)
            elif mask_type == "temporal":
                return np.random.uniform(self.config.temporal_mask_ratio / 4, self.config.temporal_mask_ratio)
        return self.config.spatial_mask_ratio if mask_type == "spatial" else self.config.temporal_mask_ratio

# endrange

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
        self._random_variable_len_proportion = (
            config.dataset_slice_random_variable_len_proportion
        )
        self._random_variable_len_min = 16
        self._spatial_mask_ratio = config.spatial_mask_ratio

        self._fps_aug = config.fps_aug
        if self._fps_aug:
            self._fps_aug_rates = config.fps_aug_rates
            self._base_fps_rate = config.base_fps_rate
            self._fps_aug_multiplier = list(set(rate / self._base_fps_rate for rate in self._fps_aug_rates))
            del self._fps_aug_rates
            del self._base_fps_rate

            self._min_fps_multiplier = min(self._fps_aug_multiplier)
            self._max_seq_len = int(math.ceil(self._subseq_len / self._min_fps_multiplier))
        else:
            self._max_seq_len = self._subseq_len

        self._traj_aug = config.traj_aug
        # assert not self._traj_aug, "Trajectory augmentation is not supported for AdaptiveAmassHdf5Dataset yet."

        # Initialize groups and cache
        with h5py.File(self._hdf5_path, "r") as hdf5_file:
            self.min_seq_len = (
                self._subseq_len
            )  # Removed conditioning on previous window
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
        split_prefixes = [split + "/" for split in self.config.splits]
        groups = [
            p
            for p in all_paths
            if any(p.startswith(prefix) for prefix in split_prefixes)
            and cast(
                h5py.Dataset, cast(h5py.Group, hdf5_file[p])["T_world_root"]
            ).shape[0]
            >= self._max_seq_len
        ]

        assert (
            len(groups) > 0
        ), f"No valid groups found for splits: {self.config.splits}"
        assert (
            len(cast(h5py.Group, hdf5_file[groups[0]]).keys()) > 0
        ), f"First group {groups[0]} has no keys"

        return groups

    def _calculate_group_lengths(self, hdf5_file: h5py.File) -> list[int]:
        """Calculate the lengths of each group in the dataset.

        Args:
            hdf5_file (h5py.File): Open HDF5 file object.

        Returns:
            list[int]: List of lengths for each group.
        """
        assert self._max_seq_len >= self._subseq_len, f"max_seq_len {self._max_seq_len} should be greater than subseq_len {self._subseq_len}"
        return [
            cast(h5py.Dataset, cast(h5py.Group, hdf5_file[g])["T_world_root"]).shape[0]
            - self._max_seq_len
            for g in self._groups
        ]

    def __getitem__(self, index: int) -> EgoTrainingData:
        """Retrieve an item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            EgoTrainingData: Data object containing the requested item.
        """
        if self._slice_strategy == "full_sequence":
            group = self._groups[index]
            npz_group = self._get_npz_group(group)
            total_t = cast(h5py.Dataset, npz_group["T_world_root"]).shape[0]
            start_t, end_t = 0, total_t
            multiplier = None
        else:
            global_index = (
                index * self._subseq_len
            )  # Directly using global index without previous window conditioning
            group_index, slice_index = self._find_group_and_slice_index(global_index)

            group = self._groups[group_index]
            npz_group = self._get_npz_group(group)
            total_t = cast(h5py.Dataset, npz_group["T_world_root"]).shape[0]
            assert total_t >= self.min_seq_len

            # Calculate slice indices for the current window
            start_t = slice_index
            end_t = min(start_t + self._subseq_len, total_t)

            multiplier = None
            if self._fps_aug:
                multiplier = random.choice(self._fps_aug_multiplier)
                req_seq_len = int(round(self._subseq_len * multiplier))

                if req_seq_len < 1:
                    req_seq_len = 1
                if req_seq_len > total_t - start_t:
                    req_seq_len = total_t - start_t

                end_t = start_t + req_seq_len

        # Load current window data
        seq_len = (
            total_t if self._slice_strategy == "full_sequence" else end_t - start_t
        )
        dtype=torch.float32
        kwargs = self._load_sequence_data(group, start_t, end_t, total_t, seq_len, dtype=dtype)

        # Add MAE-style masking
        num_joints = CFG.smplh.num_joints
        # assert num_joints == 22, f"Expected 22 joints, got {num_joints}"
        device = kwargs["joints_wrt_world"].device

        if self._fps_aug and seq_len != self._subseq_len and self._slice_strategy != "full_sequence":
            assert multiplier is not None and multiplier != 1, f"multiplier should not be 1, got {multiplier}"
            for key in self.config.ts_keys:
                if key in kwargs:
                    assert isinstance(kwargs[key], torch.Tensor) and kwargs[key].shape[0] == seq_len, f"Expected shape: {(seq_len, *kwargs[key].shape[1:])}, got: {kwargs[key].shape}"
                    data = kwargs[key].numpy(force=True)
                    resampled_data = self.resample_data(data, self._subseq_len, multiplier < 1)
                    kwargs[key] = torch.from_numpy(resampled_data).to(device=device, dtype=dtype)

		# After performing possible fps aug down/up sampling, set seq_len to the default window length.
        seq_len = total_t if self._slice_strategy == "full_sequence" else self._subseq_len

        # NOTE: the height from floor shoud be set to zeros as the preprocessing process alredy subtracted the floor height. 
        kwargs['height_from_floor'] = torch.zeros((seq_len, 1), dtype=dtype, device=device)

        kwargs["mask"] = torch.ones(seq_len, dtype=torch.bool, device=device)

        # Get spatial mask ratio and temporal mask ratio
        spatial_mask_ratio = self._get_mask_ratio(mask_type="spatial")
        num_masked = int(num_joints * spatial_mask_ratio)

        # Create initial visible joints mask
        visible_joints_mask = torch.ones(
            (seq_len, num_joints), dtype=torch.bool, device=device
        )

        # Create temporal patch mask
        patch_size = self.config.temporal_patch_size
        # Pad sequence length to be divisible by patch size
        pad_len = (patch_size - seq_len % patch_size) % patch_size
        padded_seq_len = seq_len + pad_len
        
        # Create padded mask and rearrange into patches
        temporal_mask = torch.ones((padded_seq_len,), dtype=torch.bool, device=device)
        temporal_patches = rearrange(temporal_mask, '(n p) -> n p', p=patch_size)
        
        # Randomly mask temporal patches
        temporal_mask_ratio = self._get_mask_ratio(mask_type="temporal")
        num_patches = temporal_patches.shape[0]
        num_masked_patches = int(num_patches * temporal_mask_ratio)

        # prevent corner cases.
        if num_patches == num_masked_patches:
            logger.warning(f"num_patches == num_masked_patches: {num_patches}")
        if num_masked_patches == 0:
            logger.warning(f"num_masked_patches == 0: {num_masked_patches}")

        patch_indices = torch.randperm(num_patches)[:num_masked_patches]
        temporal_patches[patch_indices] = False
        
        # Rearrange back and trim padding
        temporal_mask = rearrange(temporal_patches, 'n p -> (n p)')[:seq_len]
        
        # Apply both spatial and temporal masks
        # First apply spatial masking
        rand_indices = torch.randperm(num_joints)
        masked_indices = rand_indices[:num_masked]
        visible_joints_mask[:, masked_indices] = False
        
        # Then apply temporal patch mask
        visible_joints_mask = visible_joints_mask & temporal_mask.unsqueeze(-1)

        # Get original joints_wrt_world
        # Combine root position from T_world_root with other joints to get full 22 joints
        joints_wrt_world = kwargs["joints_wrt_world"]  # shape: [time, 22, 3]
        assert joints_wrt_world.shape == (
            seq_len,
            num_joints,
            3,
        ), f"Expected shape: {(seq_len, num_joints, 3)}, got: {joints_wrt_world.shape}"

        # Update kwargs with new MAE-style masking tensors
        kwargs["visible_joints_mask"] = visible_joints_mask

        # Zero out invisible joints while keeping original joints for loss computation
        masked_joints = joints_wrt_world
        kwargs["joints_wrt_world"] = masked_joints

        # Create metadata object first
        metadata = EgoTrainingData.MetaData()
        metadata.stage = "raw"  # Set initial stage
        # uid servers as a null value just for compatibility with EgoExoDataset
        metadata.take_name = (f"name_{group}_uid_{group}_t{start_t}_{end_t}",)
        metadata.scope = "train"
        metadata.dataset_type = self.__class__.__name__
        
        # Add metadata to kwargs before creating EgoTrainingData
        kwargs["metadata"] = metadata
        
        ret = EgoTrainingData(**kwargs)  # Create with metadata
        ret = ret.preprocess()  # Preprocess data (will update metadata.stage)

        # Apply SE2 trajectory augmentation if enabled
        if self._traj_aug:
            rand_radian = torch.rand(1) * 2 * np.pi
            ret = ret._rotate(rand_radian)

        return ret

    @jaxtyped(typechecker=typeguard.typechecked)
    def resample_data(self, data: Float[np.ndarray, "time *dim"] | Bool[np.ndarray, "time *dim"], target_len: int, upsample: bool) -> Float[np.ndarray, "{target_len} *dim"] |  Bool[np.ndarray, "{target_len} *dim"]:
        current_len = data.shape[0]
        if current_len == target_len:
            return data
        if upsample:
            assert target_len > current_len, f"target_len {target_len} should be greater than current_len {current_len}"
            # Use spline interpolation for upsampling
            t_current = np.linspace(0, 1, current_len)
            t_target = np.linspace(0, 1, target_len)
            return scipy.interpolate.interp1d(t_current, data, kind='cubic', axis=0, fill_value='extrapolate')(t_target)
        else:
            assert target_len < current_len, f"target_len {target_len} should be less than current_len {current_len}"
            # Use signal resample for downsampling
            return scipy.signal.resample(data, target_len, axis=0)

    def _load_sequence_data(
        self, group: str, start_t: int, end_t: int, total_t: int, seq_len: int, dtype: torch.dtype = torch.float32
    ) -> dict[str, Any]:
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
            if array.shape[0] != seq_len and k != "betas":
                array = np.concatenate(
                    [
                        array,
                        np.repeat(array[-1:,], seq_len - array.shape[0], axis=0),
                    ],
                    axis=0,
                )

            kwargs[k] = torch.from_numpy(array).to(dtype=dtype)

        return kwargs

    def _get_npz_group(
        self, group: str
    ) -> Union[h5py.Group, dict[str, np.ndarray[Any, Any]]]:
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
        group_index = np.searchsorted(self._cum_len, global_index, side="right")
        slice_index = global_index - (
            self._cum_len[group_index - 1] if group_index > 0 else 0
        )
        return int(group_index), int(slice_index)

    def __len__(self) -> int:
        """Return the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        if self._slice_strategy == "full_sequence":
            return len(self._groups)
        else:
            _ = self._cum_len[-1] // self._subseq_len
            return _.item()

    def _get_mask_ratio(self, mask_type: Literal["spatial", "temporal"]) -> float:
        """Get mask ratio - either fixed or randomly sampled"""
        if self.config.random_sample_mask_ratio:
            # Randomly sample between 0~mask_ratio
            if mask_type == "spatial":
                return np.random.uniform(self.config.spatial_mask_ratio / 3, self.config.spatial_mask_ratio)
            elif mask_type == "temporal":
                return np.random.uniform(self.config.temporal_mask_ratio / 3, self.config.temporal_mask_ratio)
        return self.config.spatial_mask_ratio if mask_type == "spatial" else self.config.temporal_mask_ratio