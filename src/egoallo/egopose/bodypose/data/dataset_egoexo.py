import torch
from torch.utils.data import Dataset, DataLoader
import os.path as osp
import joblib
import pandas as pd
import numpy as np
import json
import os
import shutil
import random
from PIL import Image
from tqdm import tqdm
import multiprocessing
import random
import threading
import glob
import skimage.io as io
import torchvision.transforms as T
from tqdm import tqdm
import matplotlib.pyplot as plt

# from egoallo.data.build import EgoTrainingData
from typing import List, Dict, Any, Tuple
from jaxtyping import Float, Bool, jaxtyped
import typeguard
from torch import Tensor
from egoallo.mapping import (
    EGOEXO4D_BODYPOSE_TO_SMPLH_INDICES,
    SMPLH_KINTREE,
    EGOEXO4D_BODYPOSE_KINTREE_PARENTS,
)
from egoallo.utils.setup_logger import setup_logger
from egoallo.utilities import find_numerical_key_in_dict
from pathlib import Path
import torch
import numpy as np
from scipy.signal import savgol_filter
from typing import Tuple

logger = setup_logger(output=None, name=__name__)
import joblib


random.seed(1)


class Dataset_EgoExo(Dataset):
    def __init__(self, config: Dict[str, Any]):
        super(Dataset_EgoExo, self).__init__()

        self.root = config["dataset_path"]
        self.root_takes = os.path.join(self.root, "takes")
        self.split = config["split"]
        self.root_poses = os.path.join(
            self.root, "annotations", "ego_pose", self.split, "body"
        )
        self.use_pseudo = config["use_pseudo"]
        self.coord = config["coord"]
        gt_ground_height_anno_dir = config["gt_ground_height_anno_dir"]
        self.gt_ground_height = json.load(
            open(
                Path(gt_ground_height_anno_dir)
                / f"ego_pose_gt_anno_{self.split}_public_height.json"
            )
        )
        # self.slice_window =  config["window_size"]
        self.slice_window = 128

        manually_annotated_takes = os.listdir(
            os.path.join(self.root_poses, "annotation")
        )
        self.manually_annotated_takes = [
            take.split(".")[0] for take in manually_annotated_takes
        ]
        if self.use_pseudo:
            pseudo_annotated_takes = os.listdir(
                os.path.join(self.root_poses, "automatic")
            )
            self.pseudo_annotated_takes = [
                take.split(".")[0] for take in pseudo_annotated_takes
            ]

        self.cameras = os.listdir(self.root_poses.replace("body", "camera_pose"))
        self.metadata = json.load(open(os.path.join(self.root, "takes.json")))

        self.takes_uids = (
            self.pseudo_annotated_takes
            if self.use_pseudo
            else self.manually_annotated_takes
        )
        self.takes_metadata = {}

        self.valid_take_uid_save_label = (
            "valid_takes_{}_use_manual.pkl".format(self.split)
            if not self.use_pseudo
            else "valid_takes_{}_use_pseudo.pkl".format(self.split)
        )

        for take_uid in self.takes_uids:
            take_temp = self.get_metadata_take(take_uid)
            if take_temp and "bouldering" not in take_temp["take_name"]:
                self.takes_metadata[take_uid] = take_temp

        if not osp.exists(self.valid_take_uid_save_label):
            self.valid_take_uids = []
            manually = 0
            no_man = 0
            no_cam = 0
            no_cam_list = []

            cnt = 0

            for take_uid in tqdm(
                self.takes_metadata,
                total=len(self.takes_metadata),
                desc="takes_metadata",
                ascii=" >=",
            ):
                # if cnt > 50:
                #     break
                cnt += 1
                if take_uid + ".json" in self.cameras:
                    camera_json = json.load(
                        open(
                            os.path.join(
                                self.root_poses.replace("body", "camera_pose"),
                                take_uid + ".json",
                            )
                        )
                    )
                    take_name = camera_json["metadata"]["take_name"]
                    if not take_uid in self.manually_annotated_takes:
                        no_man += 1
                        if self.use_pseudo and take_uid in self.pseudo_annotated_takes:
                            pose_json = json.load(
                                open(
                                    os.path.join(
                                        self.root_poses, "automatic", take_uid + ".json"
                                    )
                                )
                            )
                            if (
                                len(pose_json) > (self.slice_window + 2)
                            ) and self.split == "train":
                                ann, traj = self.translate_poses(
                                    pose_json, camera_json, self.coord
                                )
                                if len(traj) > (self.slice_window + 2):
                                    self.valid_take_uids.append(take_uid)
                            elif self.split != "train":
                                ann, traj = self.translate_poses(
                                    pose_json, camera_json, self.coord
                                )
                                self.valid_take_uids.append(take_uid)
                    elif take_uid in self.manually_annotated_takes:
                        pose_json = json.load(
                            open(
                                os.path.join(
                                    self.root_poses, "annotation", take_uid + ".json"
                                )
                            )
                        )
                        if (
                            len(pose_json) > (self.slice_window + 2)
                        ) and self.split == "train":
                            ann, traj = self.translate_poses(
                                pose_json, camera_json, self.coord
                            )
                            if len(traj) > (self.slice_window + 2):
                                self.valid_take_uids.append(take_uid)
                        elif self.split != "train":
                            ann, traj = self.translate_poses(
                                pose_json, camera_json, self.coord
                            )
                            self.valid_take_uids.append(take_uid)

                else:
                    # print("No take uid {} in camera poses".format(take_uid))
                    no_cam += 1
                    no_cam_list.append(take_uid)

            # self.joint_names = ['left-wrist', 'left-eye', 'nose', 'right-elbow', 'left-ear', 'left-shoulder', 'right-hip', 'right-ear', 'left-knee', 'left-hip', 'right-wrist', 'right-ankle', 'right-eye', 'left-elbow', 'left-ankle', 'right-shoulder', 'right-knee']
            if len(self.valid_take_uids) > 0:
                joblib.dump(self.valid_take_uids, self.valid_take_uid_save_label)
            else:
                raise UserWarning("No valid takes found")
        else:
            self.valid_take_uids = joblib.load(self.valid_take_uid_save_label)
            logger.info(f"Loaded valid take uids from {len(self.valid_take_uids)}")
            # self.valid_take_uids = self.valid_take_uids

        self.joint_idxs = [i for i in range(17)]  # 17 keypoints in total

        self.joint_names = [
            "nose",
            "left-eye",
            "right-eye",
            "left-ear",
            "right-ear",
            "left-shoulder",
            "right-shoulder",
            "left-elbow",
            "right-elbow",
            "left-wrist",
            "right-wrist",
            "left-hip",
            "right-hip",
            "left-knee",
            "right-knee",
            "left-ankle",
            "right-ankle",
        ]
        # self.single_joint = opt['single_joint']
        logger.info(f"Dataset lenght: {len(self.valid_take_uids)}")
        logger.info(f"Split: {self.split}")
        # logger.info('No Manually: {}'.format(no_man))
        # logger.info('No camera: {}'.format(no_cam))
        # logger.info('No camera list: {}'.format(no_cam_list))

    def translate_poses(self, anno, cams, coord):
        """
        Translate poses from EgoExo4D to global coordinates.
        NOTE: the raw ['camera_extrinsics'] are in global coordinates, which transforms world coordinates to camera coordinates.
        """
        trajectory = {}
        to_remove = []
        for key in cams.keys():
            if "aria" in key:
                aria_key = key
                break
        first = next(iter(anno))
        first_cam = cams[aria_key]["camera_extrinsics"][first]
        T_first_camera = np.eye(4)
        T_first_camera[:3, :] = np.array(first_cam)
        for frame in anno:
            try:
                current_anno = anno[frame]
                current_cam = cams[aria_key]["camera_extrinsics"][frame]
                T_world_camera_ = np.eye(4)
                T_world_camera_[:3, :] = np.array(current_cam)

                if coord == "global":
                    T_world_camera = np.linalg.inv(T_world_camera_)
                elif coord == "aria":
                    T_world_camera = np.dot(
                        T_first_camera, np.linalg.inv(T_world_camera_)
                    )
                else:
                    T_world_camera = T_world_camera_
                assert len(current_anno) != 0
                for idx in range(len(current_anno)):
                    joints = current_anno[idx]["annotation3D"]
                    for joint_name in joints:
                        joint4d = np.ones(4)
                        joint4d[:3] = np.array(
                            [
                                joints[joint_name]["x"],
                                joints[joint_name]["y"],
                                joints[joint_name]["z"],
                            ]
                        )
                        if coord == "global":
                            new_joint4d = joint4d
                        elif coord == "aria":
                            new_joint4d = T_first_camera.dot(joint4d)
                        else:
                            new_joint4d = T_world_camera_.dot(
                                joint4d
                            )  # The skels always stay in 0,0,0 wrt their camera frame
                        joints[joint_name]["x"] = new_joint4d[0]
                        joints[joint_name]["y"] = new_joint4d[1]
                        joints[joint_name]["z"] = new_joint4d[2]
                    current_anno[idx]["annotation3D"] = joints
                traj = T_world_camera[:3, 3]
                trajectory[frame] = traj
            except:
                to_remove.append(frame)
            anno[frame] = current_anno
        keys_old = list(anno.keys())
        for frame in keys_old:
            if frame in to_remove:
                del anno[frame]
        return anno, trajectory

    def get_metadata_take(self, uid):
        for take in self.metadata:
            if take["take_uid"] == uid:
                return take

    def parse_skeleton(self, skeleton):
        poses = []
        flags = []
        keypoints = skeleton.keys()
        for keyp in self.joint_names:
            if keyp in keypoints:
                flags.append(1)  # visible
                poses.append(
                    [skeleton[keyp]["x"], skeleton[keyp]["y"], skeleton[keyp]["z"]]
                )  # visible
            else:
                flags.append(0)  # not visible
                poses.append([float("nan")] * 3)  # not visible
        return poses, flags

    @jaxtyped(typechecker=typeguard.typechecked)
    def _process_joints(
        self,
        data: Float[Tensor, "timesteps 17 3"],
        vis: Float[Tensor, "timesteps 17"],
        ground_height: float = 0.0,
        return_smplh_joints: bool = True,
        num_joints: int = 22,
        debug_vis: bool = False,
    ) -> Tuple[
        Float[Tensor, "timesteps {num_joints} 3"],
        Bool[Tensor, "timesteps {num_joints}"],
    ]:
        """Process joint data from annotations.

        Args:
            data: List of frame dictionaries containing body pose data
            return_smplh_joints: If True, converts joints from EgoExo4D (17 joints) to SMPLH format (22 body joints).
                Invalid mappings will be filled with zeros.
            debug_vis: If True, visualize joints using polyscope (for debugging)

        Returns:
            Tuple of:
            - joints_world: World coordinate joint positions (timesteps x J x 3) where J is 17 for EgoExo4D or 22 for SMPLH
            - visible: Joint visibility mask (timesteps x J) where J is 17 for EgoExo4D or 22 for SMPLH
        """
        # Initialize SMPLH tensors with NaN for positions and False for visibility
        if return_smplh_joints:
            T = data.shape[0]
            smplh_world = torch.full((T, 22, 3), float("nan"), dtype=torch.float32)
            smplh_visible = torch.zeros((T, 22), dtype=torch.bool)

            # Map joints using EGOEXO4D_BODYPOSE_TO_SMPLH_INDICES
            for smplh_idx, ego_idx in enumerate(EGOEXO4D_BODYPOSE_TO_SMPLH_INDICES):
                if ego_idx != -1:
                    # Valid mapping - copy data
                    smplh_world[:, smplh_idx] = data[:, ego_idx]
                    smplh_visible[:, smplh_idx] = vis[:, ego_idx]
            return smplh_world, smplh_visible
        else:
            return data, vis.bool()

    def apply_kinematic_constraints_v2(
        self,
        joints_world: torch.Tensor,
        joints_world_coco: torch.Tensor,
        threshold: float = 3.0,
        window_size: int = 11,  # Odd number for centered window
        temporal_sigma: float = 2.0,  # For Gaussian weighting
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Enhanced version with temporal awareness and smoothing:
        1. Uses weighted temporal window for distance statistics
        2. Applies Savitzky-Golay temporal smoothing
        3. Fallback to global statistics when local window is insufficient
        """

        def process_joints(joints: torch.Tensor, kintree: list) -> torch.Tensor:
            T, J, _ = joints.shape
            device = joints.device

            # Create Gaussian weights for temporal window
            half_window = window_size // 2
            x = np.linspace(-temporal_sigma, temporal_sigma, window_size)
            weights = torch.tensor(
                np.exp(-(x**2) / 2), dtype=torch.float32, device=device
            )
            weights /= weights.sum()

            for j in range(J):
                parent_idx = kintree[j]
                if parent_idx == -1:
                    continue

                # Pre-calculate valid frames for efficiency
                valid_mask = ~torch.isnan(joints[:, j]).any(dim=1) & ~torch.isnan(
                    joints[:, parent_idx]
                ).any(dim=1)
                valid_ts = torch.where(valid_mask)[0].cpu().numpy()

                # Calculate global statistics as fallback
                if len(valid_ts) > 1:
                    global_dists = torch.norm(
                        joints[valid_ts, j] - joints[valid_ts, parent_idx], dim=1
                    )
                    global_mean = global_dists.mean()
                    global_std = global_dists.std()
                else:
                    continue  # Insufficient data for this joint pair

                for t in range(T):
                    if not valid_mask[t]:
                        continue

                    # Get temporal window bounds
                    start = max(0, t - half_window)
                    end = min(T, t + half_window + 1)
                    window_ts = torch.arange(start, end, device=device)

                    # Find valid frames in window
                    window_valid = valid_mask[window_ts]
                    if window_valid.sum() < 3:  # Use global stats if insufficient
                        mean_dist = global_mean
                        std_dist = global_std
                    else:
                        # Calculate weighted statistics in window
                        window_weights = weights[window_ts - t + half_window][
                            window_valid
                        ]
                        window_dists = torch.norm(
                            joints[window_ts[window_valid], j]
                            - joints[window_ts[window_valid], parent_idx],
                            dim=1,
                        )
                        mean_dist = (
                            window_dists * window_weights
                        ).sum() / window_weights.sum()
                        std_dist = torch.sqrt(
                            (window_weights * (window_dists - mean_dist) ** 2).sum()
                            / window_weights.sum()
                        )

                    # Adjust outliers
                    current_dist = torch.norm(joints[t, j] - joints[t, parent_idx])
                    if not (
                        mean_dist - threshold * std_dist
                        <= current_dist
                        <= mean_dist + threshold * std_dist
                    ):
                        direction = joints[t, j] - joints[t, parent_idx]
                        direction_normalized = direction / (current_dist + 1e-7)
                        adjusted_joint = (
                            joints[t, parent_idx] + direction_normalized * mean_dist
                        )
                        joints[t, j] = adjusted_joint

            # Temporal smoothing after adjustments
            for j in range(J):
                valid_ts = torch.where(~torch.isnan(joints[:, j]).any(dim=1))[0]
                if len(valid_ts) > window_size:
                    # Savitzky-Golay smoothing for joint trajectories
                    try:
                        smoothed = savgol_filter(
                            joints[valid_ts, j].cpu().numpy(),
                            window_length=window_size,
                            polyorder=2,
                            axis=0,
                        )
                        joints[valid_ts, j] = torch.tensor(smoothed, device=device)
                    except:
                        logger.warning(
                            "Applying temporal smoothing using Savitzky-Golay smoothing failed due to insufficient valid ts samples."
                        )
                        pass

            return joints

        # SMPLH kinematic tree (parent indices)
        smplh_kintree = SMPLH_KINTREE

        # COCO kinematic tree (parent indices)
        coco_kintree = EGOEXO4D_BODYPOSE_KINTREE_PARENTS

        # Process both joint sets
        joints_world = process_joints(joints_world, smplh_kintree)
        joints_world_coco = process_joints(joints_world_coco, coco_kintree)

        return joints_world, joints_world_coco

    @jaxtyped(typechecker=typeguard.typechecked)
    def apply_kinematic_constraints(
        self,
        joints_world: Float[Tensor, "timesteps 22 3"],  # SMPLH joints: (T, 22, 3)
        joints_world_coco: Float[Tensor, "timesteps 17 3"],  # COCO joints: (T, 17, 3)
        threshold: float = 3.0,  # Number of standard deviations for outlier detection
    ) -> Tuple[Float[Tensor, "timesteps 22 3"], Float[Tensor, "timesteps 17 3"]]:
        """
        Applies kinematic constraints to joint positions to filter outliers and adjust positions based on expected distances.

        Args:
            joints_world (torch.Tensor): SMPLH joint positions in world coordinates, shape (T, 22, 3)
            joints_world_coco (torch.Tensor): COCO joint positions in world coordinates, shape (T, 17, 3)
            threshold (float): Number of standard deviations for defining outlier thresholds

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Adjusted SMPLH and COCO joints
        """
        # Define kinematic trees for SMPLH and COCO
        smplh_kintree = SMPLH_KINTREE
        coco_kintree = EGOEXO4D_BODYPOSE_KINTREE_PARENTS

        # Process SMPLH joints
        for joint_idx in range(len(smplh_kintree)):
            parent_idx = smplh_kintree[joint_idx]
            if parent_idx == -1:
                continue  # Skip root node

            valid_frames = []
            for t in range(joints_world.shape[0]):
                # Check if parent and child are not NaN
                parent_valid = not torch.isnan(joints_world[t, parent_idx]).any()
                child_valid = not torch.isnan(joints_world[t, joint_idx]).any()
                if parent_valid and child_valid:
                    valid_frames.append(t)

            if not valid_frames:
                continue

            # Calculate distances between parent and child in valid frames
            parent_pos = joints_world[valid_frames, parent_idx]  # [valid_frames, 3]
            child_pos = joints_world[valid_frames, joint_idx]  # [valid_frames, 3]
            distances = torch.norm(child_pos - parent_pos, dim=1)  # [valid_frames]
            mean_dist = torch.mean(distances)
            std_dist = torch.std(distances)

            upper_bound = mean_dist + threshold * std_dist
            lower_bound = mean_dist - threshold * std_dist

            # Adjust outliers in each valid frame
            for t in valid_frames:
                current_parent = joints_world[t, parent_idx]  # [3]
                current_child = joints_world[t, joint_idx]  # [3]j
                dist = torch.norm(current_child - current_parent)  # [1]

                if dist < lower_bound or dist > upper_bound:
                    direction = current_child - current_parent  # [3]
                    direction_normalized = direction / (dist + 1e-7)  # [3]
                    adjusted_child = (
                        current_parent + direction_normalized * mean_dist
                    )  # [3]
                    joints_world[t, joint_idx] = adjusted_child

        # Process COCO joints
        for joint_idx in range(len(coco_kintree)):
            parent_idx = coco_kintree[joint_idx]
            if parent_idx == -1:
                continue  # Skip root node

            valid_frames = []
            for t in range(joints_world_coco.shape[0]):
                parent_valid = not torch.isnan(joints_world_coco[t, parent_idx]).any()
                child_valid = not torch.isnan(joints_world_coco[t, joint_idx]).any()
                if parent_valid and child_valid:
                    valid_frames.append(t)

            if not valid_frames:
                continue

            # Calculate distances between parent and child in valid frames
            parent_pos = joints_world_coco[valid_frames, parent_idx]
            child_pos = joints_world_coco[valid_frames, joint_idx]
            distances = torch.norm(child_pos - parent_pos, dim=1)
            mean_dist = torch.mean(distances)
            std_dist = torch.std(distances)

            upper_bound = mean_dist + threshold * std_dist
            lower_bound = mean_dist - threshold * std_dist

            # Adjust outliers in each valid frame
            for t in valid_frames:
                current_parent = joints_world_coco[t, parent_idx]
                current_child = joints_world_coco[t, joint_idx]
                dist = torch.norm(current_child - current_parent)

                if dist < lower_bound or dist > upper_bound:
                    direction = current_child - current_parent
                    direction_normalized = direction / (dist + 1e-7)
                    adjusted_child = current_parent + direction_normalized * mean_dist
                    joints_world_coco[t, joint_idx] = adjusted_child

        return joints_world, joints_world_coco

    def __getitem__(self, index):
        take_uid = self.valid_take_uids[index]

        camera_json = json.load(
            open(
                os.path.join(
                    self.root_poses.replace("body", "camera_pose"), take_uid + ".json"
                )
            )
        )
        take_name = camera_json["metadata"]["take_name"]
        gt_ground_height = (
            self.gt_ground_height[take_uid]
            if take_uid in self.gt_ground_height
            else 0.0
        )
        if self.use_pseudo and take_uid in self.pseudo_annotated_takes:
            pose_json = json.load(
                open(os.path.join(self.root_poses, "automatic", take_uid + ".json"))
            )
            if (len(pose_json) > (self.slice_window + 2)) and self.split == "train":
                ann, traj = self.translate_poses(pose_json, camera_json, self.coord)
            elif self.split != "train":
                ann, traj = self.translate_poses(pose_json, camera_json, self.coord)
        elif take_uid in self.manually_annotated_takes:
            pose_json = json.load(
                open(os.path.join(self.root_poses, "annotation", take_uid + ".json"))
            )
            if (len(pose_json) > (self.slice_window + 2)) and self.split == "train":
                ann, traj = self.translate_poses(pose_json, camera_json, self.coord)
            elif self.split != "train":
                ann, traj = self.translate_poses(pose_json, camera_json, self.coord)
        else:
            raise UserWarning(
                "Take uid {} not found in any annotation folder".format(take_uid)
            )

        pose = ann
        aria_trajectory = traj

        capture_frames = find_numerical_key_in_dict(pose)
        # capture_frames =  list(pose.keys())
        # Create continuous frame sequence from min to max frame keys
        min_frame = min(capture_frames)
        max_frame = max(capture_frames)
        continuous_frames = list(range(min_frame, max_frame + 1))

        seq_len = len(continuous_frames)

        # Prepare data for interpolation
        frame_keys_list = list(capture_frames)
        skeletons_window = []
        flags_window = []
        aria_window = []

        for frame in frame_keys_list:
            skeleton = pose[str(frame)][0]["annotation3D"]
            skeleton, flags = self.parse_skeleton(skeleton)
            skeletons_window.append(skeleton)
            flags_window.append(flags)
            aria_window.append(aria_trajectory[str(frame)])

        skeletons_window = torch.Tensor(np.array(skeletons_window))  # T, 17, 3
        flags_window = torch.Tensor(np.array(flags_window))  # T, 17
        aria_window = torch.Tensor(np.array(aria_window))  # T, 3

        # Process original keyframes
        joints_world_orig, visible_mask_orig = self._process_joints(
            skeletons_window,
            flags_window.float(),
            ground_height=float(gt_ground_height),
            return_smplh_joints=True,
            num_joints=22,
            debug_vis=False,
        )
        joints_world_orig_coco, _ = self._process_joints(
            skeletons_window,
            flags_window.float(),
            ground_height=float(gt_ground_height),
            return_smplh_joints=False,
            num_joints=17,
            debug_vis=False,
        )

        # Import scipy interpolation
        from scipy.interpolate import make_interp_spline

        # Create interpolation functions for each joint dimension
        num_joints = joints_world_orig.shape[1]

        # Interpolate world coordinates
        joints_world = torch.full((seq_len, num_joints, 3), float("nan"))
        joints_world_coco = torch.full((seq_len, 17, 3), float("nan"))

        for j in range(num_joints):
            for d in range(3):
                # Get joint data and corresponding frames
                joint_data = joints_world_orig[:, j, d].numpy(force=True)
                valid_mask = ~np.isnan(joint_data)
                valid_frames = np.array(frame_keys_list)[valid_mask]
                valid_data = joint_data[valid_mask]
                if len(valid_frames) == 0:
                    # If no valid frames, fill with NaN
                    continue  # do nothing as the initial value is NaN.
                elif len(valid_frames) >= 4:  # Need at least 4 points for cubic spline
                    # Create B-spline interpolation
                    bspl = make_interp_spline(valid_frames, valid_data, k=3)
                    # Evaluate spline at all frames
                    interpolated = bspl(continuous_frames)
                else:
                    # Fall back to linear interpolation for too few points
                    interpolated = np.interp(
                        continuous_frames, valid_frames, valid_data
                    )

                joints_world[:, j, d] = torch.from_numpy(interpolated)

        for j in range(17):
            for d in range(3):
                # Get joint data and corresponding frames
                joint_data = joints_world_orig_coco[:, j, d].numpy(force=True)
                valid_mask = ~np.isnan(joint_data)
                valid_frames = np.array(frame_keys_list)[valid_mask]
                valid_data = joint_data[valid_mask]
                if len(valid_frames) == 0:
                    # If no valid frames, fill with NaN
                    continue
                elif len(valid_frames) >= 4:  # Need at least 4 points for cubic spline
                    # Create B-spline interpolation
                    bspl = make_interp_spline(valid_frames, valid_data, k=3)
                    # Evaluate spline at all frames
                    interpolated = bspl(continuous_frames)
                else:
                    # Fall back to linear interpolation for too few points
                    interpolated = np.interp(
                        continuous_frames, valid_frames, valid_data
                    )

                joints_world_coco[:, j, d] = torch.from_numpy(interpolated)

        # joints_world, joints_world_coco = self.apply_kinematic_constraints(
        joints_world, joints_world_coco = self.apply_kinematic_constraints_v2(
            joints_world=joints_world, joints_world_coco=joints_world_coco
        )
        # Create visibility mask based on non-nan values in world coordinates
        visible_mask = ~torch.isnan(joints_world).any(
            dim=-1
        )  # shape: (seq_len, num_joints)
        visible_mask_orig_coco = ~torch.isnan(joints_world_coco).any(
            dim=-1
        )  # shape: (seq_len, num_joints)
        take_name = f"name_{take_name}_uid_{take_uid}_t{continuous_frames[0]}_{continuous_frames[-1]}"

        from egoallo.data.dataclass import EgoTrainingData

        ret = EgoTrainingData(
            joints_wrt_world=joints_world,  # Already computed above
            joints_wrt_cpf=torch.zeros_like(joints_world),  # Same shape as joints_world
            T_world_root=torch.zeros(
                (seq_len, 7)
            ),  # T x 7 for translation + quaternion
            T_world_cpf=torch.zeros((seq_len, 7)),  # T x 7 for translation + quaternion
            visible_joints_mask=visible_mask,  # Already computed above
            mask=torch.ones(seq_len, dtype=torch.bool),  # T
            betas=torch.zeros((1, 16)),  # 1 x 16 for SMPL betas
            body_quats=torch.zeros(
                (seq_len, 21, 4)
            ),  # T x 21 x 4 for body joint rotations
            hand_quats=torch.zeros(
                (seq_len, 30, 4)
            ),  # T x 30 x 4 for hand joint rotations
            contacts=torch.zeros((seq_len, 22)),  # T x 22 for contact states
            height_from_floor=torch.full((seq_len, 1), gt_ground_height),  # T x 1
            metadata=EgoTrainingData.MetaData(  # raw data.
                take_name=(take_name,),
                frame_keys=tuple(continuous_frames),  # Convert to tuple of ints
                stage="raw",
                scope="test",
                dataset_type="AriaDataset",
                aux_joints_wrt_world_placeholder=joints_world_coco,  # Placeholder for COCO joints
                aux_visible_joints_mask_placeholder=visible_mask_orig_coco,  # Placeholder for COCO visibility
            ),
        )
        ret = ret.preprocess()

        return ret

    def __len__(self):
        return len(self.valid_take_uids)


class Dataset_EgoExo_inference(Dataset):
    def __init__(self, config: Dict[str, Any]):
        super(Dataset_EgoExo_inference, self).__init__()

        self.root = config["dataset_path"]
        self.root_takes = os.path.join(self.root, "takes")
        self.split = config["split"]  # val or test
        self.camera_poses = os.path.join(
            self.root, "annotations", "ego_pose", self.split, "camera_pose"
        )
        self.use_pseudo = config["use_pseudo"]
        self.coord = config["coord"]

        self.metadata = json.load(open(os.path.join(self.root, "takes.json")))

        self.dummy_json = json.load(open(config["dummy_json_path"]))
        self.takes_uids = [*self.dummy_json]
        self.takes_metadata = {}

        for take_uid in self.takes_uids:
            take_temp = self.get_metadata_take(take_uid)
            if take_temp and "bouldering" not in take_temp["take_name"]:
                self.takes_metadata[take_uid] = take_temp

        self.trajectories = {}
        self.cameras = {}

        for take_uid in tqdm(self.takes_metadata):
            trajectory = {}
            camera_json = json.load(
                open(os.path.join(self.camera_poses, take_uid + ".json"))
            )
            take_name = camera_json["metadata"]["take_name"]
            self.cameras[take_uid] = camera_json
            traj = self.translate_camera(
                [*self.dummy_json[take_uid]["body"]], camera_json, self.coord
            )
            self.trajectories[take_uid] = traj

        print("Dataset lenght: {}".format(len(self.trajectories)))
        print("Split: {}".format(self.split))

    def translate_camera(self, frames, cams, coord):
        trajectory = {}
        for key in cams.keys():
            if "aria" in key:
                aria_key = key
                break
        first = frames[0]
        first_cam = cams[aria_key]["camera_extrinsics"][first]
        T_first_camera = np.eye(4)
        T_first_camera[:3, :] = np.array(first_cam)
        for frame in frames:
            current_cam = cams[aria_key]["camera_extrinsics"][frame]
            T_world_camera_ = np.eye(4)
            T_world_camera_[:3, :] = np.array(current_cam)

            if coord == "global":
                T_world_camera = np.linalg.inv(T_world_camera_)
            elif coord == "aria":
                T_world_camera = np.dot(T_first_camera, np.linalg.inv(T_world_camera_))
            else:
                T_world_camera = T_world_camera_

            traj = T_world_camera[:3, 3]
            trajectory[frame] = traj

        return trajectory

    def get_metadata_take(self, uid):
        for take in self.metadata:
            if take["take_uid"] == uid:
                return take

    def __getitem__(self, index):
        take_uid = self.takes_uids[index]
        aria_trajectory = self.trajectories[take_uid]
        aria_window = []
        frames_window = list(aria_trajectory.keys())
        for frame in frames_window:
            aria_window.append(aria_trajectory[frame])

        aria_window = torch.Tensor(np.array(aria_window))
        head_offset = aria_window.unsqueeze(1).repeat(1, 17, 1)
        condition = aria_window
        task = torch.tensor(self.takes_metadata[take_uid]["task_id"])
        take_name = self.takes_metadata[take_uid]["root_dir"]

        return {
            "cond": condition,
            "t": frames_window,
            "aria": aria_window,
            "offset": head_offset,
            "task": task,
            "take_name": take_name,
            "take_uid": take_uid,
        }

    def __len__(self):
        return len(self.trajectories)

