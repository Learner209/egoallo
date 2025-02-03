from egoallo.registry import DATASET_MANIPULATORS
from .egoexo_utils import EgoExoUtils
from egoallo.config import make_cfg, CONFIG_FILE

local_config_file = CONFIG_FILE
CFG = make_cfg(config_name="defaults", config_file=local_config_file, cli_args=[])


@DATASET_MANIPULATORS.register("DefaultDatasetManipulator")
def build_default_dataset_manipulator(cfg):
    return EgoExoUtils(
        cfg,
        lazy_loading=cfg.instantiate.egoexo_utils.lazy_loading,
        run_demo=cfg.instantiate.egoexo_utils.run_demo,
    )


def make_dataset_manipulator(cfg):
    return DATASET_MANIPULATORS[cfg.datasets.manipulator.name](cfg)


EGOEXO_UTILS_INST: EgoExoUtils = make_dataset_manipulator(CFG)
