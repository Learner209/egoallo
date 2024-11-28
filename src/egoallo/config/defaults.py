import os
from yacs.config import CfgNode as CN
import os.path as osp
from collections import defaultdict
import numpy as np
import torch
from datetime import datetime

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  res = _C.clone()
  res.freeze()
  return res



from egoallo.mapping import EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS, EGOEXO4D_EGOPOSE_HANDPOSE_MAPPINGS

BODY_JOINTS = EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS
HAND_JOINTS = EGOEXO4D_EGOPOSE_HANDPOSE_MAPPINGS
NUM_OF_HAND_JOINTS = len(HAND_JOINTS) // 2
NUM_OF_BODY_JOINTS = len(BODY_JOINTS)  
NUM_OF_JOINTS = NUM_OF_BODY_JOINTS + NUM_OF_HAND_JOINTS * 2


_C = CN()
_C.project_name = "egoallo"
_C.project_root = ""
_C.dataset_root_dir = osp.join(_C.project_root, "datasets")
_C.assets_path = osp.join(_C.project_root, "assets")
_C.save_id = f"time_{datetime.now().strftime('%m_%d_%H_%M_%S')}"
_C.config_file = None

_C.phases = ["train", "test", "inference"]

_C.dbg = True
_C.evaltime = False
_C.deterministic = True
_C.backup_src = True

_C.smplh = CN()
_C.smplh.smplh_root_path = osp.join(_C.project_root, "assets/smpl_based_model/smplh")
_C.smplh.smplh_model = "male" # male, female, neutral
_C.smplh.num_expressions = 16
_C.smplh.num_betas = 16
_C.smplh.num_joints = 21