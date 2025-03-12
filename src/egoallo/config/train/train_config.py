import dataclasses
from pathlib import Path
from typing import Literal

from egoallo import network
from egoallo import training_loss
from egoallo.types import DatasetSliceStrategy
from egoallo.types import DatasetSplit
from egoallo.types import DatasetType
from egoallo.types import JointCondMode


@dataclasses.dataclass
class EgoAlloTrainConfig:
    """Configuration for EgoAllo training."""

    from egoallo.data.dataclass import EgoTrainingData

    # experiment config.
    experiment_name: str = "motion_prior"
    experiment_dir: Path = Path("")
    dataset_hdf5_path: Path = Path("data/egoalgo_no_skating_dataset.hdf5")
    dataset_files_path: Path = Path("data/egoalgo_no_skating_dataset_files.txt")
    smplh_npz_path: Path = Path("./data/smplh/neutral/model.npz")

    # MAE params.
    spatial_mask_ratio: float = 0.75
    """Mask ratio for spatial dim, typically joint dimension."""

    temporal_mask_ratio: float = 0.3
    """Mask ratio for temporal dim"""

    temporal_patch_size: int = 12
    """Patch size for temporal masking"""

    random_sample_mask_ratio: bool = True
    """If True, randomly sample mask ratio between mask_ratio / 3 ~ mask_ratio for each batch"""

    joint_cond_mode: JointCondMode = "absrel"

    # Model and denoising configuration
    model: network.EgoDenoiserConfig = dataclasses.field(init=False)
    denoising: network.DenoisingConfig = dataclasses.field(init=False)
    loss: training_loss.TrainingLossConfig = dataclasses.field(
        default_factory=training_loss.TrainingLossConfig,
    )

    # Dataset arguments.
    batch_size: int = 256
    """Effective batch size."""
    num_workers: int = 0
    subseq_len: int = 128
    dataset_slice_strategy: DatasetSliceStrategy = "random_uniform_len"
    dataset_slice_random_variable_len_proportion: float = 0.3
    """Only used if dataset_slice_strategy == 'random_variable_len'."""
    splits: tuple[DatasetSplit, ...] = ("train", "val")
    data_collate_fn: Literal[
        "DefaultBatchCollator",
        "ExtendedBatchCollator",
        "EgoTrainingDataBatchCollator",
        "TensorOnlyDataclassBatchCollator",
    ] = "EgoTrainingDataBatchCollator"
    dataset_type: DatasetType = "AdaptiveAmassHdf5Dataset"
    bodypose_anno_dir: Path | None = None
    """Path to body pose annotation directory, only used when dataset_type is EgoExoDataset"""

    # Optimizer options.
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0

    # Network arch.
    use_fourier_in_masked_joints: bool = True
    use_joint_embeddings: bool = True

    # debug
    debug: bool = False
    max_steps: int = (
        1000000000  # never reached , since max_steps is a debug-only handle.
    )

    # Data aug.
    fps_aug: bool = False
    """Whether to augment data with different FPS. Implmented with upsamping with five-order spline interpolation or downsampling at different rates."""

    fps_aug_rates: tuple[float, ...] = (7.5, 15, 30, 60, 120)
    """FPS rates to augment data with."""

    base_fps_rate: float = 30
    """Base FPS rate for data augmentation, also is the default rate in training data."""

    traj_aug: bool = False
    """Whether to augment data with different traj rotations. Implemented with random rotations in the 2D plane."""

    # traj_aug_num_samples: int = 10
    # """Number of samples the aug rotation is sampled uniformly along the unit circle."""

    # Misc
    ts_keys: tuple[str, ...] = tuple(
        [
            field.name
            for field in dataclasses.fields(EgoTrainingData)
            if field.name not in ("betas", "metadata")
        ],
    )
    """Keys that contain time-series data in the `EgoTrainingData` dataclass."""

    def __post_init__(self):
        # Create model config with denoising settings
        self.model = network.EgoDenoiserConfig(
            mask_ratio=self.spatial_mask_ratio,
            joint_cond_mode=self.joint_cond_mode,
            smplh_npz_path=self.smplh_npz_path,
            use_fourier_in_masked_joints=self.use_fourier_in_masked_joints,
            use_joint_embeddings=self.use_joint_embeddings,
        )
        # Create denoising config first since model config depends on it
        self.denoising = network.DenoisingConfig.from_joint_cond_mode(
            joint_cond_mode=self.joint_cond_mode,
            include_hands=self.model.include_hands,
        )
