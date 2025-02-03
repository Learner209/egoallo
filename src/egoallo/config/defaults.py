from yacs.config import CfgNode as CN
import os.path as osp
import numpy as np
from datetime import datetime
from egoallo.mapping import (
    EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS,
    EGOEXO4D_EGOPOSE_HANDPOSE_MAPPINGS,
)


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    res = _C.clone()
    res.freeze()
    return res


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
_C.smplh.smplh_model = "male"  # male, female, neutral
_C.smplh.num_expressions = 16
_C.smplh.num_betas = 16
_C.smplh.num_joints = 22

_C.solver = CN()
_C.solver.num_epochs = 11
# optimizer
_C.solver.max_lr = 5e-4
_C.solver.end_lr = 1e-5
_C.solver.bias_lr_factor = 1
_C.solver.momentum = 0.9

# EMA
_C.solver.use_ema = True
_C.solver.ema_decay = 0.995
_C.solver.step_start_ema = 2000
_C.solver.ema_update_every = 10

# diffusion
_C.solver.learning_rate = 5e-5
_C.solver.amp = False
_C.solver.save_and_sample_every = 10000

# lr_scheduler
_C.solver.weight_decay = 0.0005
_C.solver.weight_decay_bias = 0.0
_C.solver.gamma = 0.1
_C.solver.lrate_decay = 250
_C.solver.steps = (30000,)
_C.solver.warmup_factor = 1.0 / 3
_C.solver.warmup_iters = 500
_C.solver.warmup_method = "linear"
_C.solver.num_iters = 1000
_C.solver.min_factor = 0.1
_C.solver.log_interval = 1

_C.solver.optimizer = "Adam"
_C.solver.scheduler = "WarmupMultiStepLR"
_C.solver.scheduler_decay_thresh = 0.00005

# grad_clip
_C.solver.do_grad_clip = False
_C.solver.grad_clip_type = "norm"  # norm or value
_C.solver.grad_clip = 1.0

# loss_fn
_C.solver.loss_func = "l2"

# early stopping
_C.solver.early_stopping = CN()
_C.solver.early_stopping.patience = 6
_C.solver.early_stopping.verbose = False
_C.solver.early_stopping.delta = 0


# model specific configs
_C.solver.use_additional_mask_layer_for_nan_val = False
_C.solver.enable_temporal_smoothness_loss = False
_C.solver.include_hand_pose = False
_C.solver.include_head_vel = False
_C.solver.include_head_qpose = False
_C.solver.pred_last = False
_C.solver.cut_off = None
_C.solver.jnt_idx_for_move_trans = 1
_C.solver.use_min_max = True
_C.solver.use_mean_std = False
_C.solver.align_bodypose_with_head_qpose = True

_C.solver.val_freq = 1
_C.solver.save_last_only = False
_C.solver.empty_cache = True
_C.solver.trainer = "diffusion"
_C.solver.run_demo = False
_C.solver.load_model = ""
_C.solver.load = ""
_C.solver.broadcast_buffers = False
_C.solver.find_unused_parameters = False
_C.solver.resume = False
_C.solver.dist = CN()
_C.solver.save_optimizer = True
_C.solver.save_scheduler = True

# region
_C.coordinate = CN()
_C.coordinate.transform = CN()
_C.coordinate.transform.opengl2smpl = np.array(
    [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
).tolist()
_C.coordinate.transform.smpl2opengl = np.array(
    [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
).tolist()
_C.coordinate.transform.aria2opengl = np.array(
    [[0, -1, 0], [-1, 0, 0], [0, 0, -1]]
).tolist()
_C.coordinate.transform.opengl2aria = np.linalg.inv(
    np.array(_C.coordinate.transform.aria2opengl)
).tolist()
_C.coordinate.transform.smpl2ros = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).tolist()
_C.coordinate.transform.ros2smpl = np.linalg.inv(
    np.array(_C.coordinate.transform.smpl2ros)
).tolist()
# endregion


# region
_C.mujoco = CN()
_C.mujoco.mujoco_executable_path = ""
_C.mujoco.vis = CN()
_C.mujoco.vis.enable = True
_C.mujoco.io = CN()
_C.mujoco.io.root_path = osp.join(_C.project_root, "egoego/env/exp/")
_C.mujoco.mujoco_assets = CN()
_C.mujoco.mujoco_assets.root_path = osp.join(
    _C.project_root, "egoego/env/mujoco/mujoco_assets_2.1.0"
)
_C.mujoco.mujoco_assets.model_id = "humanoid_smpl_neutral_mesh"
# endregion

# region
_C.blender = CN()
_C.blender.blender_executable_path = "/home/minghao/src/robotflow/new_egoego/assets/blender/blender-3.6.13-linux-x64/blender"
_C.blender.scene_blender_root_path = osp.join(
    _C.project_root, "egoego/utils/blender_utils"
)
_C.blender.scene_blender_demo_path = osp.join(
    _C.blender.scene_blender_root_path, "for_demo.blend"
)
_C.blender.scene_blender_colorful_mat_path = osp.join(
    _C.blender.scene_blender_root_path,
    "floor_colorful_mat_human_w_head_pose_hres.blend",
)

_C.blender.scripts = CN()
_C.blender.scripts.root_path = osp.join(
    _C.project_root, "third_party/egoego/egoego/vis"
)
_C.blender.scripts.blender_vis_human_utils = osp.join(
    _C.blender.scripts.root_path, "blender_vis_human_utils.py"
)
_C.blender.scripts.blender_vis_cmp_human_utils = osp.join(
    _C.blender.scripts.root_path, "blender_vis_cmp_human_utils.py"
)
_C.blender.scripts.blender_vis_human_and_headpose_utils = osp.join(
    _C.blender.scripts.root_path, "blender_vis_human_and_headpose_utils.py"
)
# endregion

_C.io = CN()
_C.io.main_exp_path = osp.join(_C.project_root, "exp")
_C.io.exp_path = osp.join(_C.io.main_exp_path, "egoexo-default-exp")

_C.io.egoexo = CN()
_C.io.egoexo.root_path = osp.join(_C.dataset_root_dir, "egoexo-default")
_C.io.egoexo.exp_path = osp.join(_C.io.exp_path, "egoexo")
_C.io.egoexo.save_mesh = CN()
_C.io.egoexo.save_mesh.vis_folder = osp.join(_C.io.egoexo.exp_path, "mesh")

_C.io.egoexo.preprocessing = CN()
_C.io.egoexo.preprocessing.save_root = osp.join(_C.io.egoexo.exp_path, "preprocess")
_C.io.egoexo.preprocessing.egoexo_root_path = osp.join(
    _C.dataset_root_dir, "egoexo-default"
)
_C.io.egoexo.preprocessing.config_file = None

_C.io.egoexo.preprocessing.test_public_file_path = osp.join(
    _C.project_root,
    "egoego/egopose/handpose/data_preparation/ego_pose_gt_anno_test_public.json",
)
_C.io.egoexo.preprocessing.gt_output_dir = osp.join(
    _C.io.egoexo.exp_path, "egoexo-default-gt-output"
)

_C.io.egoexo.preprocessing.gt_handpose = CN()
_C.io.egoexo.preprocessing.gt_handpose.output_dir = osp.join(
    _C.io.egoexo.preprocessing.gt_output_dir, "handpose"
)

_C.io.egoexo.preprocessing.gt_bodypose = CN()
_C.io.egoexo.preprocessing.gt_bodypose.run_demo = False
_C.io.egoexo.preprocessing.gt_bodypose.discard_seq_than = 30


_C.io.egoexo.preprocessing.gt_bodypose.output = CN()
_C.io.egoexo.preprocessing.gt_bodypose.output.root = osp.join(
    _C.io.egoexo.preprocessing.gt_output_dir, "bodypose", "canonical"
)
_C.io.egoexo.preprocessing.gt_bodypose.output.save_dir = osp.join(
    _C.io.egoexo.preprocessing.gt_bodypose.output.root, _C.save_id
)
_C.io.egoexo.preprocessing.gt_bodypose.output.config_save_dir = osp.join(
    _C.io.egoexo.preprocessing.gt_bodypose.output.save_dir, "config"
)
_C.io.egoexo.preprocessing.gt_bodypose.output.log_save_dir = osp.join(
    _C.io.egoexo.preprocessing.gt_bodypose.output.save_dir, "logs"
)

_C.io.egoexo.preprocessing.gt_bodypose.sample_output = CN()
_C.io.egoexo.preprocessing.gt_bodypose.sample_output.root = osp.join(
    _C.io.egoexo.preprocessing.gt_output_dir, "bodypose", "test_sample"
)
_C.io.egoexo.preprocessing.gt_bodypose.sample_output.save_dir = osp.join(
    _C.io.egoexo.preprocessing.gt_bodypose.sample_output.root, _C.save_id
)
_C.io.egoexo.preprocessing.gt_bodypose.sample_output.config_save_dir = osp.join(
    _C.io.egoexo.preprocessing.gt_bodypose.sample_output.save_dir, "config"
)
_C.io.egoexo.preprocessing.gt_bodypose.sample_output.log_save_dir = osp.join(
    _C.io.egoexo.preprocessing.gt_bodypose.sample_output.save_dir, "logs"
)

# Sampling num for run_demo control.
_C.io.egoexo.preprocessing.gt_bodypose.num_sample_takes = 3
_C.io.egoexo.preprocessing.gt_bodypose.require_valid_kpts = [
    "left-wrist",
    "right-wrist",
]

_C.io.egoexo.preprocessing.aria_img_output_dir = osp.join(
    _C.io.egoexo.preprocessing.gt_output_dir, "aria_img"
)
_C.io.egoexo.preprocessing.sample_aria_img_output_dir = osp.join(
    _C.io.egoexo.preprocessing.gt_output_dir, "sample_aria_img"
)
_C.io.egoexo.preprocessing.aria_img_output_run_demo = False
_C.io.egoexo.preprocessing.extract_aria_img_multiprocessing_thread_num = 5

_C.io.egoexo.preprocessing.aria_calib_output_dir = osp.join(
    _C.io.egoexo.preprocessing.gt_output_dir, "aria_calib"
)

# _C.io.egoexo.preprocessing.smplh_anno_output_dir = osp.join(_C.io.egoexo.preprocessing.gt_output_dir, "smplh_anno")
# _C.io.egoexo.preprocessing.smplh_anno_multiprocessing_thread_num = 15
# _C.io.egoexo.preprocessing.aligned_anno_output_dir = osp.join(_C.io.egoexo.preprocessing.gt_output_dir, "aligned_anno")

# _C.io.egoexo.preprocessing.align_slam_multiprocessing_thread_num = 15

_C.io.egoexo.preprocessing.egoexo_train_data_output_dir = osp.join(
    _C.io.egoexo.preprocessing.gt_output_dir, "egoexo_train_data", "canonical"
)
_C.io.egoexo.preprocessing.sample_egoexo_train_data_output_dir = osp.join(
    _C.io.egoexo.preprocessing.gt_output_dir, "egoexo_train_data", "test_sample"
)
_C.io.egoexo.preprocessing.egoexo_train_data_multiprocessing_thread_num = 15

_C.io.egoexo.preprocessing.exported_mp4_path = osp.join(
    _C.io.egoexo.preprocessing.gt_output_dir, "exported_mp4"
)
_C.io.egoexo.preprocessing.export_mp4_multiprocessing_thread_num = 12

_C.io.egoexo.preprocessing.splits = ["train", "val", "test"]
_C.io.egoexo.preprocessing.all_splits = _C.io.egoexo.preprocessing.splits

_C.io.egoexo.preprocessing.anno_types = ["manual", "auto"]
_C.io.egoexo.preprocessing.all_anno_types = _C.io.egoexo.preprocessing.anno_types

_C.io.egoexo.preprocessing.portrait_view = False
_C.io.egoexo.preprocessing.valid_kpts_num_thresh = 10
_C.io.egoexo.preprocessing.bbox_padding = 20
_C.io.egoexo.preprocessing.reproj_error_threshold = 30

_C.io.egoexo.preprocessing.smplh_model = _C.smplh.smplh_model

_C.io.egoexo.preprocessing.fps = 30
_C.io.egoexo.preprocessing.steps = [
    "aria_calib",
    "hand_gt_anno",
    "body_gt_anno",
    "raw_image",
    "undistorted_image",
]

# _C.io.egoexo.preprocessing.smpl_origin_running_mean_window = 20


_C.instantiate = CN()
_C.instantiate.egoexo_utils = CN()
_C.instantiate.egoexo_utils.lazy_loading = True
_C.instantiate.egoexo_utils.run_demo = False
_C.instantiate.generate_smpl_anno = CN()
_C.instantiate.generate_smpl_anno.dry_run = True
_C.instantiate.datasets = CN()
_C.instantiate.datasets.anno_types = ["manual", "auto"]
_C.instantiate.test = CN()
_C.instantiate.test.dry_run = True


_C.io.diffusion = CN()
_C.io.diffusion.exp_path = osp.join(_C.io.exp_path, "diffusion")
_C.io.diffusion.project = osp.join(_C.io.diffusion.exp_path, _C.save_id)
_C.io.diffusion.project_exp_name = osp.join(_C.io.diffusion.project, "")
_C.io.diffusion.ckpt_save_path = osp.join(_C.io.diffusion.project_exp_name, "weights")
_C.io.diffusion.mesh_save_path = osp.join(_C.io.diffusion.project_exp_name, "mesh_vis")
_C.io.diffusion.val_save_path = osp.join(_C.io.diffusion.project_exp_name, "val_exp")
_C.io.diffusion.log_save_path = osp.join(_C.io.diffusion.project_exp_name, "logs")

_C.io.diffusion.eval = CN()
_C.io.diffusion.eval.save_gt_enable = True
_C.io.diffusion.eval.save_model_pred_enable = False

_C.io.config = CN()
_C.io.config.save_path = osp.join(_C.io.diffusion.project_exp_name, "config")

# region
_C.empirical_val = CN()
# for determining floor height
_C.empirical_val.ankle_floor_height_offset = 0.01
_C.empirical_val.floor_vel_thresh = 0.005
# for determining contacts
_C.empirical_val.contact_vel_thresh = 0.01
_C.empirical_val.contact_toe_height_thresh = 0.04
_C.empirical_val.contact_ankle_height_thresh = 0.08
# for determining terrain interaction
_C.empirical_val.terrain_height_thresh = 0.04  # if this_cluster_toe_height is more thatn min_toe_median + TERRAIN_HEIGHT_THRESH, then discard seq.
_C.empirical_val.root_height_thresh = 0.04  # if this_cluster_root_height is more thatn min_pelvis_median + ROOT_HEIGHT_THRESH, then discard seq.
_C.empirical_val.cluster_size_thresh = 0.25  # if cluster has more than this faction of fps (30 for 120 fps) , then discard seq.
# for discarding seqs with duration shorter than this
_C.empirical_val.discard_shorter_than_sec = 1.0
# for discarding seqs with frames shorter than this, in EgoExoDiffusionDataset
_C.empirical_val.discard_shorter_than_frames = 30
# for splitting long sequence to avoid OOM.
_C.empirical_val.split_frame_limit = 2000
# trimming ratio for long sequences
_C.empirical_val.trim_ratio_begin = 0.1
_C.empirical_val.trim_ratio_end = 0.9
_C.empirical_val.smplh_head_vert_idx = 444

_C.empirical_val.metric = CN()
_C.empirical_val.metric.foot_sliding = CN()
_C.empirical_val.metric.foot_sliding.ankle_height_threshold = 0.08  # meter
_C.empirical_val.metric.foot_sliding.toe_height_threshold = 0.04  # meter

_C.empirical_val.smpl = CN()
_C.empirical_val.smpl.smpl_root_offset_z = 0.91437225
# endregion

_C.egoexo_cfg = CN()
_C.egoexo_cfg.aria = CN()
_C.egoexo_cfg.aria.frame_rate = 30

_C.cuda = CN()
_C.cuda.processing_device = 0  # int number
_C.cuda.train_device = 0  # int number
_C.cuda.inference_device = 0  # int number
_C.cuda.test_device = 0  # int number
_C.cuda.val_device = _C.cuda.train_device

_C.logging = CN()
_C.logging.checkpoint_print = 200
_C.logging.level = "INFO"

_C.logging.wandb = CN()
_C.logging.wandb.use_wandb = True
_C.logging.wandb.wandb_pj_name = "egoego"
_C.logging.wandb.entity = "train_cond_motion_diffusion"
_C.logging.wandb.name = "minghao"
_C.logging.wandb.exp_name = "egopose_transformer_diffusion"
if _C.solver.run_demo:
    _C.logging.wandb.mode = "offline"
else:
    _C.logging.wandb.mode = "online"
_C.logging.wandb.save_dir = osp.join(_C.io.diffusion.project_exp_name, "wandb")

# _C.logging.wandb.config.

_C.datasets = CN()
_C.datasets.shuffle = True
_C.datasets.dataset_type = "EgoExoDiffusionDataset"
_C.datasets.concat = True
_C.datasets.manipulator = CN()
_C.datasets.manipulator.name = "DefaultDatasetManipulator"
_C.datasets.dataloaders = CN()
_C.datasets.dataloaders.names = [
    "EgoExoDiffusionDataset",
]
_C.datasets.dataloaders.configs = list()

_C.datasets.canonicalize_init_head = False
_C.datasets.use_min_max = _C.solver.use_min_max
_C.datasets.use_mean_std = _C.solver.use_mean_std
_C.datasets.window_size = 90

_C.datasets.train = CN()
_C.datasets.train.phase = "train"
_C.datasets.train.split = "train"
_C.datasets.train.has_gt = True
_C.datasets.train.run_demo = False
_C.datasets.train.dataloader_shuffle = True
_C.datasets.train.dataloader_num_workers = 0
_C.datasets.train.dataloader_batch_size = 512
_C.datasets.train.window_size = _C.datasets.window_size
_C.datasets.train.anno_types = _C.instantiate.datasets.anno_types
_C.datasets.train.root = _C.io.egoexo.root_path
_C.datasets.train.gt_bodypose_output_dir = (
    _C.io.egoexo.preprocessing.gt_bodypose.output.save_dir
)
_C.datasets.train.gt_bodypose_sample_output_dir = (
    _C.io.egoexo.preprocessing.gt_bodypose.sample_output.save_dir
)
_C.datasets.train.recording_fps = _C.io.egoexo.preprocessing.fps

# Test dataset specifics
_C.datasets.test = CN()
_C.datasets.test.dataloader_batch_size = 1
_C.datasets.test.dataloader_num_workers = 0
_C.datasets.test.window_size = _C.datasets.window_size
_C.datasets.test.phase = "val"
_C.datasets.test.split = "val"
_C.datasets.test.has_gt = True
_C.datasets.test.run_demo = False
_C.datasets.test.anno_types = _C.instantiate.datasets.anno_types
_C.datasets.test.root = _C.io.egoexo.root_path
_C.datasets.test.gt_bodypose_output_dir = (
    _C.io.egoexo.preprocessing.gt_bodypose.output.save_dir
)
_C.datasets.test.gt_bodypose_sample_output_dir = (
    _C.io.egoexo.preprocessing.gt_bodypose.sample_output.save_dir
)
_C.datasets.test.test_milestone = 7
_C.datasets.test.ckpt_weight_path = osp.join(
    _C.io.diffusion.ckpt_save_path, f"model_{_C.datasets.test.test_milestone}.pth"
)
_C.datasets.test.save_path_root = osp.join(
    _C.io.diffusion.project_exp_name, "test_results"
)
_C.datasets.test.recording_fps = _C.io.egoexo.preprocessing.fps


# inference dataset specifics
_C.datasets.inference = CN()
_C.datasets.inference.name = "filtered_egoexo_inference"
_C.datasets.inference.phase = "inference"
_C.datasets.inference.split = "test"
_C.datasets.inference.dataloader_batch_size = 1
_C.datasets.inference.dataloader_num_workers = 0
_C.datasets.inference.window_size = _C.datasets.window_size
_C.datasets.inference.has_gt = False
_C.datasets.inference.run_demo = False
_C.datasets.inference.anno_types = _C.instantiate.datasets.anno_types
_C.datasets.inference.root = _C.io.egoexo.root_path
_C.datasets.inference.gt_bodypose_output_dir = (
    _C.io.egoexo.preprocessing.gt_bodypose.output.save_dir
)
_C.datasets.inference.gt_bodypose_sample_output_dir = (
    _C.io.egoexo.preprocessing.gt_bodypose.sample_output.save_dir
)
_C.datasets.inference.inference_milestone = _C.datasets.test.test_milestone
_C.datasets.inference.ckpt_weight_path = osp.join(
    _C.io.diffusion.ckpt_save_path,
    f"model_{_C.datasets.inference.inference_milestone}.pth",
)
_C.datasets.inference.recording_fps = _C.io.egoexo.preprocessing.fps

# vis dataset specifics
_C.datasets.vis = CN()
_C.datasets.vis.phase = "vis"
_C.datasets.vis.split = "val"
_C.datasets.vis.dataloader_batch_size = 1
_C.datasets.vis.dataloader_num_workers = 0
_C.datasets.vis.window_size = _C.datasets.window_size
_C.datasets.vis.has_gt = True
_C.datasets.vis.run_demo = True
_C.datasets.vis.save_sample_num = 5
_C.datasets.vis.gt_bodypose_output_dir = (
    _C.io.egoexo.preprocessing.gt_bodypose.output.save_dir
)
_C.datasets.vis.gt_bodypose_sample_output_dir = (
    _C.io.egoexo.preprocessing.gt_bodypose.sample_output.save_dir
)
_C.datasets.vis.show_epoch_interval = 2
_C.datasets.vis.recording_fps = _C.io.egoexo.preprocessing.fps

_C.datasets.vis.vis_from_path = CN()
_C.datasets.vis.vis_from_path.enable = False
_C.datasets.vis.vis_from_path.gt_path = ""
_C.datasets.vis.vis_from_path.model_pred_path = ""

_C.dataloader = CN()
_C.dataloader.num_workers = 0
_C.dataloader.collator = "DefaultBatchCollator"
_C.dataloader.pin_memory = True


_C.input = CN()
_C.input.transforms = []

_C.test = CN()
_C.test.batch_size = 1
_C.test.evaluators = []
_C.test.visualizer = ""
_C.test.force_recompute = True
_C.test.do_evaluation = False
_C.test.do_visualization = False
_C.test.save_predictions = False
