from egoallo.registry import DATASET_MANIPULATORS
from 


@DATASET_MANIPULATORS.register('DefaultBatchCollator')
def build_default_bach_collator(cfg):
    return DefaultBatchCollator()