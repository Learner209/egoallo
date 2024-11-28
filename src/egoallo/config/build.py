from egoallo.registry import CONFIGS
from .defaults import get_cfg_defaults

import os.path as osp
from egoallo.utils.setup_logger import setup_logger
logger = setup_logger(output=None, name=__name__)

CONFIG_FILE = "config/experiment.yaml"
local_config_file = CONFIG_FILE

@CONFIGS.register('defaults')
def build_default_cfg():
	return get_cfg_defaults()

def make_cfg(config_name="defaults", config_file="", cli_args=[], print_arg=False):
	cfg = CONFIGS[config_name]()
	if cli_args is not None:
		cfg.merge_from_list(cli_args)
	if osp.exists(config_file):
		cfg.merge_from_file(config_file)

	cfg.freeze()
	if print_arg:
		logger.debug(cfg)

	return cfg

CFG = make_cfg(config_name="defaults", config_file=local_config_file, cli_args=[])