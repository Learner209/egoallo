from typing import Any, Callable, TYPE_CHECKING
from .default_batch_collator import default_collate
from .extended_batch_collator import extended_collate
from ..dataclass import collate_dataclass

if TYPE_CHECKING:
    from egoallo.config.train.train_config import EgoAlloTrainConfig


def make_batch_collator[T](cfg: "EgoAlloTrainConfig") -> Callable[[list[T]], T]:
    """Get appropriate collate function based on dataset type."""

    if cfg.data_collate_fn == "DefaultBatchCollator":
        return default_collate
    elif cfg.data_collate_fn == "ExtendedBatchCollator":
        return extended_collate
    elif cfg.data_collate_fn == "EgoTrainingDataBatchCollator":
        return collate_dataclass

    return collate_dataclass
