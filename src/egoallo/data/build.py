from typing import Any, assert_never

import torch
import torch.utils.data

from egoallo.config.train.train_config import EgoAlloTrainConfig

from .datasets.amass_dataset import AdaptiveAmassHdf5Dataset, VanillaEgoAmassHdf5Dataset
from .datasets.egoexo_dataset import EgoExoDataset
from egoallo.utils.setup_logger import setup_logger
from egoallo.data.dataclass import EgoTrainingData

logger = setup_logger(output=None, name=__name__)


def build_dataset(
    cfg: EgoAlloTrainConfig,
) -> type[torch.utils.data.Dataset[EgoTrainingData]]:
    """Build dataset(s) from config."""
    if cfg.dataset_type == "AdaptiveAmassHdf5Dataset":
        return AdaptiveAmassHdf5Dataset
    elif cfg.dataset_type == "VanillaEgoAmassHdf5Dataset":
        return VanillaEgoAmassHdf5Dataset
    elif cfg.dataset_type == "EgoExoDataset":
        return EgoExoDataset
    else:
        assert_never(cfg.dataset_type)
