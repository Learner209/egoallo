import json
import os.path as osp
import random

import numpy as np
import torch
from egoallo.config import CONFIG_FILE
from egoallo.config import make_cfg
from egoallo.egoexo import EGOEXO_UTILS_INST
from egoallo.utils.setup_logger import setup_logger
from egoallo.utils.smpl_mapping.mapping import EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS
from egoallo.utils.smpl_mapping.mapping import EGOEXO4D_EGOPOSE_HANDPOSE_MAPPINGS
from egoallo.utils.transformation import T_to_qpose
from egoallo.utils.utils import find_numerical_key_in_dict
from torch.utils.data import Dataset
from tqdm import tqdm
# from egoallo.egopose.main import load_raw_anno

local_config_file = CONFIG_FILE
CFG = make_cfg(config_name="defaults", config_file=local_config_file, cli_args=[])
EGOPOSE_CFG = CFG.egopose


logger = setup_logger(output=None, name=__name__)
torch.multiprocessing.set_start_method("spawn", force=True)


BODY_JOINTS = EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS
HAND_JOINTS = EGOEXO4D_EGOPOSE_HANDPOSE_MAPPINGS
NUM_OF_HAND_JOINTS = len(HAND_JOINTS) // 2
NUM_OF_BODY_JOINTS = len(BODY_JOINTS)
NUM_OF_JOINTS = NUM_OF_BODY_JOINTS + NUM_OF_HAND_JOINTS * 2


def load_raw_anno(opt, anno_type, split, run_demo=False):
    if run_demo:
        gt_bodypose_anno_output_dir = opt["gt_bodypose"]["sample_output_dir"]
    else:
        gt_bodypose_anno_output_dir = opt["gt_bodypose"]["output_dir"]

    gt_anno_output_dir = osp.join(gt_bodypose_anno_output_dir, "annotation", anno_type)
    body_gt_annot_path = osp.join(
        gt_anno_output_dir,
        f"ego_pose_gt_anno_{split}_public.json",
    )
    anno = json.load(open(body_gt_annot_path))
    return anno, anno_type, split


class Filtered_Dataset_EgoExo(Dataset):
    def __init__(self, opt):
        super(Filtered_Dataset_EgoExo, self).__init__()

        self.split = opt["split"]
        self.slice_window = opt["window_size"]
        self.anno, self.anno_type, self.split = load_raw_anno(
            opt,
            opt["anno_type"],
            opt["split"],
        )

        self.take_uids = list(self.anno.keys())
        self.valid_take_uids = []
        self.skip_uids = []
        self.Ts_world_cam = {}
        self.ego_cam_traj = {}
        self.body_3d_anno = {}
        self.body_3d_anno_valid_flags = {}
        self.valid_frames = {}
        self.take_names = {}
        self.egoexo_utils = EGOEXO_UTILS_INST

        for take_idx, take_uid in tqdm(
            enumerate(self.take_uids),
            total=len(self.take_uids),
            desc="Enumerating take uids",
            ascii=" >=",
        ):
            assert take_uid == self.anno[take_uid]["metadata"]["take_uid"]
            take_name = self.anno[take_uid]["metadata"]["take_name"]

            valid_frames = np.asarray(
                sorted(find_numerical_key_in_dict(self.anno[take_uid])),
            )
            if len(valid_frames) < self.slice_window:
                self.skip_uids.append(take_uid)
                continue

            Ts_world_cam = np.stack(
                [
                    self.anno[take_uid][str(frame_idx)]["ego_camera_extrinsics"]
                    for frame_idx in valid_frames
                ],
                axis=0,
            )

            body_3d = np.stack(
                [
                    self.anno[take_uid][str(frame_idx)]["body_3d"]
                    for frame_idx in valid_frames
                ],
                axis=0,
            )
            body_valid_3d_flags = np.stack(
                [
                    self.anno[take_uid][str(frame_idx)]["body_valid_3d"]
                    for frame_idx in valid_frames
                ],
                axis=0,
            )
            this_take_ego_cam_traj = T_to_qpose(Ts_world_cam, take_inv=True)  # N x 7

            self.take_names[take_uid] = take_name
            self.Ts_world_cam[take_uid] = Ts_world_cam
            self.ego_cam_traj[take_uid] = this_take_ego_cam_traj
            self.body_3d_anno[take_uid] = body_3d
            self.body_3d_anno_valid_flags[take_uid] = body_valid_3d_flags
            self.valid_frames[take_uid] = valid_frames

            self.valid_take_uids.append(take_uid)

    def __getitem__(self, index):
        take_uid = self.valid_take_uids[index]
        valid_frames = self.valid_frames[take_uid]

        if self.split == "train":
            frames_idx = random.randint(self.slice_window, len(valid_frames))
            frames_window = np.arange(frames_idx - self.slice_window, frames_idx)
        else:
            frames_window = np.arange(0, len(valid_frames))

        t_window = valid_frames[frames_window]  # T
        body3d_anno_win = self.body_3d_anno[take_uid][frames_window]  # T x 17 x 3
        body3d_valid_flags_win = self.body_3d_anno_valid_flags[take_uid][
            frames_window
        ]  # T x 17
        aria_traj_win = self.ego_cam_traj[take_uid][frames_window]  # T x 7

        body3d_anno_win = torch.FloatTensor(body3d_anno_win)
        body3d_valid_flags_win = torch.FloatTensor(body3d_valid_flags_win)
        aria_traj_win = torch.FloatTensor(aria_traj_win)
        head_offset = aria_traj_win.unsqueeze(1).repeat(1, NUM_OF_BODY_JOINTS, 1)
        condition = aria_traj_win
        task = self.egoexo_utils.find_take_from_take_uid(take_uid)["task_id"]
        task = torch.tensor(task).int()

        return {
            "cond": condition,
            "gt": body3d_anno_win,
            "visible": body3d_valid_flags_win,
            "t": t_window,
            "aria": aria_traj_win,
            "offset": head_offset,
            "task": task,
            "take_name": self.take_names[take_uid],
            "take_uid": take_uid,
        }

    def __len__(self):
        return len(self.valid_take_uids)


class Filtered_Dataset_EgoExo_inference(Dataset):
    def __init__(self, opt):
        super(Filtered_Dataset_EgoExo_inference, self).__init__()

        self.split = opt["split"]
        self.use_pseudo = opt["use_pseudo"]
        self.slice_window = opt["window_size"]
        self.anno, self.anno_type, self.split = load_raw_anno(opt, opt["run_demo"])

        self.take_uids = list(self.anno.keys())
        self.valid_take_uids = []
        self.skip_uids = []
        self.Ts_world_cam = {}
        self.ego_cam_traj = {}
        self.valid_frames = {}
        self.take_names = {}
        self.egoexo_utils = EGOEXO_UTILS_INST

        for take_idx, take_uid in tqdm(
            enumerate(self.take_uids),
            total=len(self.take_uids),
            desc="Enumerating take uids",
            ascii=" >=",
        ):
            assert take_uid == self.anno[take_uid]["metadata"]["take_uid"]
            take_name = self.anno[take_uid]["metadata"]["take_name"]

            valid_frames = np.asarray(
                sorted(find_numerical_key_in_dict(self.anno[take_uid])),
            )

            Ts_world_cam = np.stack(
                [
                    self.anno[take_uid][str(frame_idx)]["ego_camera_extrinsics"]
                    for frame_idx in valid_frames
                ],
                axis=0,
            )  # T x 3 x 4

            this_take_ego_cam_traj = T_to_qpose(Ts_world_cam, take_inv=True)  # N x 7

            self.take_names[take_uid] = take_name
            self.Ts_world_cam[take_uid] = Ts_world_cam
            self.ego_cam_traj[take_uid] = this_take_ego_cam_traj
            self.valid_frames[take_uid] = valid_frames

            self.valid_take_uids.append(take_uid)

    def __getitem__(self, index):
        take_uid = self.valid_take_uids[index]
        valid_frames = self.valid_frames[take_uid]

        frames_window = np.arange(0, len(valid_frames))

        t_window = valid_frames[frames_window]  # T
        aria_traj_win = self.ego_cam_traj[take_uid][frames_window]  # T x 7

        aria_traj_win = torch.FloatTensor(aria_traj_win)
        head_offset = aria_traj_win.unsqueeze(1).repeat(1, NUM_OF_BODY_JOINTS, 1)
        condition = aria_traj_win
        task = self.egoexo_utils.find_take_from_take_uid(take_uid)["task_id"]
        task = torch.tensor(task).int()

        return {
            "cond": condition,
            "t": t_window,
            "aria": aria_traj_win,
            "offset": head_offset,
            "task": task,
            "take_name": self.take_names[take_uid],
            "take_uid": take_uid,
        }

    def __len__(self):
        return len(self.valid_take_uids)
