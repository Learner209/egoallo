from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import torch

from egoallo import training_loss, network


@dataclass(frozen=False)
class EgoAlloBaseConfig:
    """Base configuration class with common settings."""

    # Data paths
    dataset_hdf5_path: Path = Path("./data/egoalgo_no_skating_dataset.hdf5")
    dataset_files_path: Path = Path("./data/egoalgo_no_skating_dataset_files.txt")
    smplh_model_path: Path = Path("assets/smpl_based_model/smplh/SMPLH_NEUTRAL.pkl")

    # Model settings
    model: network.EgoDenoiserConfig = network.EgoDenoiserConfig()
    loss: training_loss.TrainingLossConfig = training_loss.TrainingLossConfig()

    # Dataset settings
    batch_size: int = 256
    num_workers: int = 8
    subseq_len: int = 128
    dataset_slice_strategy: Literal[
        "deterministic",
        "random_uniform_len",
        "random_variable_len",
    ] = "deterministic"
    dataset_slice_random_variable_len_proportion: float = 0.3

    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Debug settings
    use_ipdb: bool = False
