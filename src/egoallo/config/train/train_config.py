import dataclasses
from pathlib import Path
from typing import Literal

from egoallo import network
from egoallo import training_loss
from egoallo.type_stubs import DatasetSliceStrategy, EgoTrainingDataTypeLiteral
from egoallo.type_stubs import DatasetSplit
from egoallo.type_stubs import DatasetType
from egoallo.type_stubs import JointCondMode
from egoallo.type_stubs import SmplFamilyModelTypeLiteral
from egoallo.type_stubs import EgoTrainingDataType
from egoallo.constants import EgoTrainingDataZoo

from egoallo.utilities import get_class_from_path


@dataclasses.dataclass
class EgoAlloTrainConfig:
    """Configuration for EgoAllo training."""

    from egoallo.data.dataclass import EgoTrainingData

    # Model and denoising configuration
    model: network.EgoDenoiserConfig = dataclasses.field(init=True)
    denoising: network.DenoisingConfig = dataclasses.field(init=True)
    loss: training_loss.TrainingLossConfig = dataclasses.field(init=True)

    # experiment config.
    experiment_name: str = "motion_prior"
    experiment_dir: Path = Path("")
    dataset_hdf5_path: Path = Path("data/egoalgo_no_skating_dataset.hdf5")
    dataset_files_path: Path = Path("data/egoalgo_no_skating_dataset_files.txt")
    smpl_family_model_basedir: Path = Path(
        "assets/smpl_based_model",
    )
    smpl_family_meta_model_name: SmplFamilyModelTypeLiteral = "SmplhModel"

    spatial_mask_ratio: float = 0.75

    temporal_mask_ratio: float = 0.3

    temporal_patch_size: int = 12

    random_sample_mask_ratio: bool = True

    joint_cond_mode: JointCondMode = "absolute"

    # Dataset arguments.
    batch_size: int = 256
    num_workers: int = 0
    subseq_len: int = 128
    dataset_slice_strategy: DatasetSliceStrategy = "random_uniform_len"
    dataset_slice_random_variable_len_proportion: float = 0.3
    splits: tuple[DatasetSplit, ...] = ("train", "val")
    data_collate_fn: Literal[
        "DefaultBatchCollator",
        "ExtendedBatchCollator",
        "EgoTrainingDataBatchCollator",
        "TensorOnlyDataclassBatchCollator",
    ] = "TensorOnlyDataclassBatchCollator"

    ego_training_data_name: EgoTrainingDataTypeLiteral = "EgoTrainingData"
    dataset_type: DatasetType = "AdaptiveAmassHdf5Dataset"
    bodypose_anno_dir: Path | None = None

    # Optimizer options.
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    max_grad_norm: float = 5.0

    # use CosineAnnealingLR as default scheduler.
    lr_scheduler: Literal[
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
        "piecewise_constant",
    ] = "cosine"
    num_epochs: int = 1800

    # whether to use ema.
    use_ema: bool = True
    ema_update_after_step: int = 0
    ema_inv_gamma: float = 1.0
    ema_power: float = 0.75
    ema_min_value: float = 0.0
    ema_max_value: float = 0.9999

    # Network arch.
    use_fourier_in_masked_joints: bool = True
    use_joint_embeddings: bool = True

    # debug
    debug: bool = False
    max_steps: int = (
        1000000000  # never reached , since max_steps is a debug-only handle.
    )

    # detect loss spikes after this step.
    detect_loss_spike_start_step: int = 2000
    # discard loss spikes backward after this step.
    discard_loss_spike_start_step: int = 8000

    # eval every this step.
    eval_every_step: int = 2000
    early_stopping_patience: int = 10
    early_stopping_delta: float = 0

    # test every this step.
    test_every_step: int = 10000

    # restore training from previous ckpt.
    restore_checkpoint_dir: Path | None = None

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

    ts_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Misc
        DataClass: EgoTrainingDataType = get_class_from_path(
            EgoTrainingDataZoo[self.ego_training_data_name],
        )

        self.ts_keys = tuple(
            [
                field.name
                for field in dataclasses.fields(DataClass)
                if field.name not in ("betas", "metadata")
            ],
        )
