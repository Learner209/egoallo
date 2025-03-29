from typing import assert_never
from typing import TYPE_CHECKING

import torch.utils.data

if TYPE_CHECKING:
    from egoallo.config.train.train_config import EgoAlloTrainConfig
    from egoallo.type_stubs import EgoTrainingDataType

from .datasets.amass_dataset import AdaptiveAmassHdf5Dataset, VanillaEgoAmassHdf5Dataset
from .datasets.egoexo_dataset import EgoExoDataset
from egoallo.egopose.bodypose.data.dataset_egoexo import Dataset_EgoExo as AriaDataset
from egoallo.egopose.bodypose.data.dataset_egoexo import (
    Dataset_EgoExo_inference as AriaInferenceDataset,
)
from egoallo.utils.setup_logger import setup_logger

logger = setup_logger(output=None, name=__name__)


def build_dataset(
    cfg: "EgoAlloTrainConfig",
) -> type[torch.utils.data.Dataset["EgoTrainingDataType"]]:
    """Build dataset(s) from config."""
    if cfg.dataset_type == "AdaptiveAmassHdf5Dataset":
        return AdaptiveAmassHdf5Dataset
    elif cfg.dataset_type == "VanillaEgoAmassHdf5Dataset":
        return VanillaEgoAmassHdf5Dataset
    elif cfg.dataset_type == "EgoExoDataset":
        return EgoExoDataset
    elif cfg.dataset_type == "AriaDataset":
        return AriaDataset
    elif cfg.dataset_type == "AriaInferenceDataset":
        return AriaInferenceDataset
    else:
        assert_never(cfg.dataset_type)
