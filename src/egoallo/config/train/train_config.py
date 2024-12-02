from pathlib import Path
from typing import Literal
import dataclasses
from egoallo import network, training_loss

@dataclasses.dataclass
class EgoAlloTrainConfig:
    experiment_name: str = "motion_prior"
    dataset_hdf5_path: Path = Path("data/egoalgo_no_skating_dataset.hdf5")
    dataset_files_path: Path = Path("data/egoalgo_no_skating_dataset_files.txt")

    mask_ratio: float = 0.75
    joint_cond_mode: Literal["absolute", "absrel_jnts", "absrel", "absrel_global_deltas"] = "absrel"

    model: network.EgoDenoiserConfig = dataclasses.field(init=False)
    loss: training_loss.TrainingLossConfig = training_loss.TrainingLossConfig()

    # Dataset arguments.
    batch_size: int = 256
    """Effective batch size."""
    num_workers: int = 2
    subseq_len: int = 128
    dataset_slice_strategy: Literal[
        "deterministic", "random_uniform_len", "random_variable_len", "full_sequence"
    ] = "random_uniform_len"
    dataset_slice_random_variable_len_proportion: float = 0.3
    """Only used if dataset_slice_strategy == 'random_variable_len'."""
    train_splits: tuple[Literal["train", "val", "test", "just_humaneva"], ...] = (
        "train", 
        "val"
    )

    # Optimizer options.
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0

    # debug
    debug: bool = False

    def __post_init__(self):
        self.model = network.EgoDenoiserConfig(
            mask_ratio=self.mask_ratio,
            joint_cond_mode=self.joint_cond_mode
        )