import json
import os
import os.path as osp

import cv2
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import pandas as pd
from projectaria_tools.core import calibration
from egoallo.egopose.handpose.data_preparation.utils.utils import (
    aria_landscape_to_portrait,
    cam_to_img,
    get_ego_pose_takes_from_splits,
    pad_bbox_from_kpts,
    rand_bbox_from_kpts,
    body_jnts_dist_angle_check,
    reproj_error_check,
    world_to_cam,
)
from egoallo.utils.smpl_mapping.mapping import EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS
from egoallo.utils.aria_utils.aria_calib import CalibrationUtilities
from egoallo.utils.setup_logger import setup_logger
from egoallo.egoexo import EGOEXO_UTILS_INST
from egoallo.egoexo.egoexo_utils import EgoExoUtils
from egoallo.config import make_cfg, CONFIG_FILE
from egoallo.egopose.handpose.data_preparation.handpose_dataloader import hand_pose_anno_loader
from egoallo.egopose.stats_collector import PreprocessingStatsCollector, KeypointFilterStats
from dataclasses import dataclass
from egoallo.data.aria_mps import load_point_cloud_and_find_ground
from pathlib import Path
from typing import Dict, Optional, Any

local_config_file = CONFIG_FILE
CFG = make_cfg(config_name="defaults", config_file=local_config_file, cli_args=[])

logger = setup_logger(output=CFG.io.egoexo.preprocessing.gt_bodypose.output.log_save_dir, name=__name__)

# Loads dataframe at target path to csv
def load_csv_to_df(filepath: str) -> pd.DataFrame:
    if not osp.exists(filepath):
        return None
    with open(filepath, "r") as csv_file:
        return pd.read_csv(csv_file)

class body_pose_anno_loader(hand_pose_anno_loader):
    """
    Load Ego4D data and create ground truth annotation JSON file for egoexo4d body-pose baseline model
    """

    def __init__(self, args, split, anno_type):
        # Initialize stats collector
        self.stats_collector = PreprocessingStatsCollector()
        # Set dataloader parameters
        self.dataset_root = args.egoexo_root_path
        self.require_valid_kpts = args.gt_bodypose.require_valid_kpts
        self.discard_seq_than = args.gt_bodypose.discard_seq_than
        self.anno_type = anno_type
        self.split = split
        self.num_joints = len(EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS)  # Number of joints for single body
        self.undist_img_dim = (2160, 3840)  # Dimension of undistorted aria image [H, W]
        self.valid_kpts_threshold = (
            args.valid_kpts_num_thresh
        )  # Threshold of minimum number of valid kpts in single body
        self.bbox_padding = (
            args.bbox_padding
        )  # Amount of pixels to pad around kpts to find bbox
        self.reproj_error_threshold = args.reproj_error_threshold
        self.portrait_view = (
            args.portrait_view
        )  # Whether use portrait view (Default is landscape view)
        self.aria_calib_dir = args.aria_calib_output_dir
        self.takes = EGOEXO_UTILS_INST.takes
        self.splits = EGOEXO_UTILS_INST.splits
        self.test_run = getattr(args.gt_bodypose, 'test_run', False)
        self.test_max_takes = getattr(args.gt_bodypose, 'test_max_takes', 2)
        self.test_max_frames_per_take = getattr(args.gt_bodypose, 'test_max_frames_per_take', 10)
        self.test_frame_interval = getattr(args.gt_bodypose, 'test_frame_interval', 30)  # Sample every 30 frames

        # Determine annotation and camera pose directory
        anno_type_dir_dict = {"manual": "annotation", "auto": "automatic"}
        self.body_anno_dir = (
            os.path.join(
                self.dataset_root, "annotations/ego_pose/test/body/annotation"
            )
            if self.split == "test"
            else os.path.join(
                self.dataset_root,
                f"annotations/ego_pose/{split}/body",
                anno_type_dir_dict[self.anno_type],
            )
        )
        self.cam_pose_dir = os.path.join(
            self.dataset_root, f"annotations/ego_pose/{split}/camera_pose"
        )

        # Add ground height cache
        self.ground_heights = {}

        # Load dataset
        self.db = self.load_raw_data()

    def load_raw_data(self):
        gt_db = {}

        # Find all annotation takes from local direcctory by splits
        # Check test anno availability. No gt-anno will be generated for public.
        if not os.path.exists(self.body_anno_dir):
            assert (
                self.split == "test"
            ), f"No annotation found for {self.split} split at {self.body_anno_dir}.\
                Make sure you follow step 0 to download data first."
            # return gt_db
        if not self.split == "test":
            # Get all local annotation takes for train/val split
            split_all_local_takes = [
                k.split(".")[0] for k in os.listdir(self.body_anno_dir)
            ]
            # take to uid dict
            take_to_uid = {
                t["take_name"]: t["take_uid"]
                for t in self.takes
                if t["take_uid"] in split_all_local_takes
            }
            uid_to_take = {uid: take for take, uid in take_to_uid.items()}
            # (0). Filter common takes
            comm_local_take_uid = list(
                set(split_all_local_takes)
            )


        # Get all valid local take uids that are used in current split
        # 1. Filter takes based on split (train/val/test)
        curr_split_uid = self.splits["split_to_take_uids"][self.split]
        # 2. Filter takes based on benchmark (ego_pose)
        ego_pose_uid = get_ego_pose_takes_from_splits(self.splits)
        curr_split_ego_pose_uid = list(set(curr_split_uid) & set(ego_pose_uid))
        
        # 3. Filter takes with available camera pose file
        available_cam_pose_uid = [
            k.split(".")[0] for k in os.listdir(self.cam_pose_dir)
        ]
        comm_take = list(set(curr_split_ego_pose_uid) & set(available_cam_pose_uid))

        if self.split == "test":
            comm_take_w_cam_pose = sorted(list(set(comm_take)))
        else:
            comm_take_w_cam_pose = sorted(list(set(comm_take) & set(comm_local_take_uid)))
        
        # Sample takes if in test run mode
        if self.test_run:
            import random
            num_takes = min(self.test_max_takes, len(comm_take_w_cam_pose))
            comm_take_w_cam_pose = random.sample(comm_take_w_cam_pose, num_takes)
            logger.info(f"Test run: Sampling {num_takes} takes")

        logger.info(f"Find {len(comm_take_w_cam_pose)} takes in {self.split} ({self.anno_type}) dataset. Start data processing...")

        overall_frames_num = 0
        overall_valid_frames_num = 0

        # Iterate through all takes from annotation directory and check
        for curr_take_uid in tqdm(comm_take_w_cam_pose):

            curr_take_name = EGOEXO_UTILS_INST.find_take_name_from_take_uid(curr_take_uid)
            assert curr_take_name is not None, f"Take name not found for {curr_take_uid}."
            # Load annotation, camera pose JSON and image directory
            curr_take_anno_path = os.path.join(
                self.body_anno_dir, f"{curr_take_uid}.json"
            )
            curr_take_cam_pose_path = os.path.join(
                self.cam_pose_dir, f"{curr_take_uid}.json"
            )
            curr_take = [t for t in self.takes if t["take_name"] == curr_take_name][0]
            traj_dir = os.path.join(self.dataset_root, curr_take["root_dir"], "trajectory")
            exo_traj_path = os.path.join(traj_dir, "gopro_calibs.csv")

            # assert os.path.exists(exo_traj_path), f"Exo trajectory file not found at {exo_traj_path}."
            assert os.path.exists(curr_take_cam_pose_path), f"Camera pose file not found at {curr_take_cam_pose_path}."

            exo_traj_df = load_csv_to_df(exo_traj_path)

            # Load in annotation JSON and image directory
            curr_take_cam_pose = json.load(open(curr_take_cam_pose_path))
            # aria_mask, aria_cam_name = self.load_aria_calib(curr_take_name)
            aria_cam_name = EgoExoUtils.get_ego_aria_cam_name(curr_take)
            if self.split == "test":
                exo_cam_masks, exo_cam_names = CalibrationUtilities.get_exo_cam_masks(curr_take, exo_traj_df, portrait_view=self.portrait_view, dimension=self.undist_img_dim[::-1])
                if len(exo_cam_names) == 0 or len(exo_cam_masks) == 0:
                    continue
                curr_take_data = self.load_take_raw_data_for_test(
                    curr_take_name, curr_take_uid, curr_take_cam_pose, aria_cam_name, exo_cam_names
                )
                if len(curr_take_data) > 0:
                    gt_db[curr_take_uid] = curr_take_data
                    # for split part data, all the frames are valid counts since there are only camera pose annotation.
                    overall_frames_num += len(curr_take_data)
                    overall_valid_frames_num += len(curr_take_data)
            else:
                assert os.path.exists(curr_take_anno_path), f"Annotation file not found at {curr_take_anno_path}."

                curr_take_anno = json.load(open(curr_take_anno_path))

                # Get valid takes info for all frames
                if len(curr_take_anno) > 0:
                    exo_cam_masks, exo_cam_names = CalibrationUtilities.get_exo_cam_masks(curr_take, exo_traj_df, portrait_view=self.portrait_view, dimension=self.undist_img_dim[::-1])
                    if exo_cam_names is not None:
                        curr_take_data = self.load_take_raw_data(
                            curr_take_name,
                            curr_take_uid,
                            curr_take_anno,
                            curr_take_cam_pose,
                            aria_cam_name,
                            exo_cam_names,
                            exo_cam_masks,
                        )
                        # Append into dataset if has at least valid annotation
                        if len(curr_take_data) > 0:
                            gt_db[curr_take_uid] = curr_take_data
                            overall_frames_num += len(curr_take_anno.values())
                            overall_valid_frames_num += len(curr_take_data)
                        else:
                            pass
                            # logger.warning(f"Take {curr_take_name} has no valid annotation. Skipped this take.")
        logger.warning(f"Egoexo dataset achieves {overall_valid_frames_num}/{overall_frames_num} valid frames for [{self.split}/{self.anno_type}].")
        return gt_db
    
    def _get_ground_height(self, take_uid: str) -> float:
        """Get ground height for a take, computing it if not cached."""
        if take_uid not in self.ground_heights:
            # Get point cloud path
            point_cloud_path = EGOEXO_UTILS_INST.load_semidense_pts(take_uid)
            
            if point_cloud_path is None:
                logger.warning(f"No point cloud found for take {take_uid}")
                self.ground_heights[take_uid] = None
                return None
                
            try:
                # Load point cloud and find ground
                _, ground_height = load_point_cloud_and_find_ground(
                    Path(point_cloud_path),
                    return_points="filtered",
                    cache_files=False
                )
                self.ground_heights[take_uid] = float(ground_height)
            except Exception as e:
                logger.error(f"Error computing ground height for take {take_uid}: {e}")
                self.ground_heights[take_uid] = None
                
        return self.ground_heights[take_uid]

    def load_take_raw_data(
        self,
        take_name,
        take_uid,
        anno,
        cam_pose,
        aria_cam_name,
        exo_cam_names,
        exo_cam_masks,
    ):
        """Load and validate data for a single take"""
        curr_take_db = {}
        
        # Initialize take stats
        self.stats_collector.mark_take_processed(take_uid, valid=False)
        
        curr_exo_intrs, curr_exo_extrs = self.load_static_ego_exo_cam_poses(
            cam_pose, exo_cam_names
        )

        for frame_idx, curr_frame_anno in anno.items():
            # Initialize frame stats with keypoint tracking
            keypoint_stats = KeypointFilterStats()
            keypoint_stats.total = self.num_joints

            # Load frame data
            curr_body_2d_kpts, curr_body_3d_kpts, joints_view_stat = self.load_frame_body_2d_3d_kpts(
                curr_frame_anno, exo_cam_names
            )
            
            curr_ego_intr, curr_ego_extr = self.load_frame_cam_pose(
                frame_idx, cam_pose, aria_cam_name
            )

            # Basic validation of data presence
            if curr_body_3d_kpts is None or curr_body_2d_kpts is None or curr_exo_extrs is None or curr_exo_intrs is None:
                keypoint_stats.missing_annotation = self.num_joints
                self.stats_collector.update_frame_stats(take_uid, frame_idx, keypoint_stats, valid=False)
                continue

            # Track initially missing annotations
            missing_anno_mask = np.any(np.isnan(curr_body_3d_kpts), axis=1)
            keypoint_stats.missing_annotation = np.sum(missing_anno_mask)
            
            # Only validate biomechanics for keypoints that have annotations
            valid_anno_mask = ~missing_anno_mask
            curr_body_3d_kpts, biomech_valid_mask = body_jnts_dist_angle_check(curr_body_3d_kpts)
            # Only count biomechanical failures for keypoints that had valid annotations
            biomech_invalid = ~biomech_valid_mask & valid_anno_mask
            keypoint_stats.biomechanical_invalid = np.sum(biomech_invalid)

            # Track overall valid keypoints considering both annotation and biomechanics
            valid_3d_kpts_flag = valid_anno_mask & biomech_valid_mask

            # Per-camera validation
            camera_validations = {}
            for curr_ind, (curr_exo_intr, curr_exo_extr, curr_exo_cam_mask, curr_exo_cam_name) in enumerate(
                zip(curr_exo_intrs.values(), curr_exo_extrs.values(), exo_cam_masks.values(), exo_cam_names)
            ):
                # Process 2D keypoints for this camera
                this_cam_body_2d_kpts = curr_body_2d_kpts[curr_exo_cam_name][:self.num_joints].copy()
                
                # Project 3D to 2D for validation
                this_body_3d_kpts_cam = world_to_cam(curr_body_3d_kpts, curr_exo_extr)
                this_cam_body_proj_2d_kpts = cam_to_img(this_body_3d_kpts_cam, curr_exo_intr)

                # Validate visibility and bounds only for keypoints that were valid after previous checks
                _, visibility_mask = self.body_kpts_valid_check(
                    this_cam_body_2d_kpts, curr_exo_cam_mask
                )
                visibility_invalid = ~visibility_mask & valid_3d_kpts_flag
                keypoint_stats.visibility_invalid += np.sum(visibility_invalid)

                # Check reprojection error only for visible keypoints
                reproj_valid_mask = reproj_error_check(
                    this_cam_body_2d_kpts,
                    this_cam_body_proj_2d_kpts,
                    self.reproj_error_threshold
                )
                reproj_invalid = ~reproj_valid_mask & valid_3d_kpts_flag & visibility_mask
                keypoint_stats.projection_error += np.sum(reproj_invalid)

                # Update per-camera validation results
                camera_validations[curr_exo_cam_name] = {
                    "visibility_valid": bool(np.any(visibility_mask & valid_3d_kpts_flag)),
                    "reprojection_valid": bool(np.any(reproj_valid_mask & valid_3d_kpts_flag & visibility_mask))
                }

                # Update global keypoint validity considering all checks
                valid_3d_kpts_flag &= visibility_mask & reproj_valid_mask

            # Final validation
            keypoint_stats.final_valid = np.sum(valid_3d_kpts_flag)
            keypoint_stats.valid_keypoint_indices = np.where(valid_3d_kpts_flag)[0].tolist()

            frame_valid = keypoint_stats.final_valid >= self.valid_kpts_threshold

            # Update frame stats
            self.stats_collector.update_frame_stats(
                take_uid,
                frame_idx, 
                keypoint_stats,
                valid=frame_valid,
                camera_validations=camera_validations
            )

            if frame_valid:
                curr_frame_anno = self._prepare_valid_frame_data(
                    curr_body_3d_kpts,
                    valid_3d_kpts_flag,
                    curr_ego_intr,
                    curr_ego_extr,
                    aria_cam_name
                )
                curr_take_db[frame_idx] = curr_frame_anno
                # breakpoint()

        # Mark take as valid if it has enough valid frames
        take_valid = len(curr_take_db) > self.discard_seq_than
        self.stats_collector.mark_take_processed(take_uid, valid=take_valid)

        if take_valid:
            # breakpoint()
            ground_height = self._get_ground_height(take_uid)

            curr_take_db["metadata"] = {
                "take_uid": take_uid,
                "take_name": take_name,
                "exo_cam_names": exo_cam_names.tolist(),
                "exo_camera_intrinsics": {k: v.tolist() for k, v in curr_exo_intrs.items()},
                "exo_camera_extrinsics": {k: v.tolist() for k, v in curr_exo_extrs.items()},
                "ground_height": ground_height
            }
            logger.info(f"Take {take_name} has {len(curr_take_db)-1}/{len(anno.items())} valid frames.")

        return curr_take_db

    def load_take_raw_data_for_test(
        self,
        take_name,
        take_uid,
        cam_pose,
        aria_cam_name,
        exo_cam_names,
    ):
        """
        Load raw data for test split with statistics collection.
        Only camera intrinsics/extrinsics and metadata are loaded for test data.
        """
        curr_take_db = {}

        # Initialize take stats
        self.stats_collector.mark_take_processed(take_uid, valid=False)

        # Basic validation of camera data
        if (
            aria_cam_name not in cam_pose.keys()
            or "camera_intrinsics" not in cam_pose[aria_cam_name].keys()
            or "camera_extrinsics" not in cam_pose[aria_cam_name].keys()
        ):
            return curr_take_db

        # Build camera projection matrix
        curr_ego_intr = np.array(
            cam_pose[aria_cam_name]["camera_intrinsics"]
        ).astype(np.float32)

        # Process each frame
        for frame_idx in cam_pose[aria_cam_name]["camera_extrinsics"].keys():
            # Initialize frame statistics
            keypoint_stats = KeypointFilterStats()
            keypoint_stats.total = self.num_joints  # Set total number of joints
            
            # For test data, we mark all keypoints as missing since we don't have annotations
            keypoint_stats.missing_annotation = self.num_joints
            
            curr_frame_anno = {}
            curr_ego_extr = np.array(
                cam_pose[aria_cam_name]["camera_extrinsics"][frame_idx]
            ).astype(np.float32)

            # Prepare camera validations dictionary
            camera_validations = {
                cam_name: {
                    "visibility_valid": True,  # Default to True for test data
                    "reprojection_valid": True  # Default to True for test data
                }
                for cam_name in exo_cam_names
            }

            # Update frame stats
            self.stats_collector.update_frame_stats(
                take_uid,
                frame_idx,
                keypoint_stats,
                valid=True,  # All frames are considered valid for test data
                camera_validations=camera_validations
            )

            # Prepare frame data
            frame_datum = {
                "ego_camera_intrinsics": curr_ego_intr.tolist(),
                "ego_camera_extrinsics": curr_ego_extr.tolist(),
                "aria_camera_name": aria_cam_name,
            }
            curr_frame_anno = {**frame_datum, **curr_frame_anno}
            curr_take_db[frame_idx] = curr_frame_anno

        # Process exo camera information
        curr_intrs = {}
        curr_extrs = {}
        for exo_cam_name in exo_cam_names:
            if (
                exo_cam_name not in cam_pose.keys()
                or "camera_intrinsics" not in cam_pose[exo_cam_name].keys()
                or "camera_extrinsics" not in cam_pose[exo_cam_name].keys()
            ):
                curr_cam_intrinsics, curr_cam_extrinsics = None, None
            else:
                curr_cam_intrinsics = np.array(
                    cam_pose[exo_cam_name]["camera_intrinsics"]
                ).astype(np.float32)
                curr_cam_extrinsics = np.array(
                    cam_pose[exo_cam_name]["camera_extrinsics"]
                ).astype(np.float32)
            curr_intrs[exo_cam_name] = curr_cam_intrinsics
            curr_extrs[exo_cam_name] = curr_cam_extrinsics

        # Mark take as valid if it has frames
        take_valid = len(curr_take_db) > self.discard_seq_than
        self.stats_collector.mark_take_processed(take_uid, valid=take_valid)

        if take_valid:
            ground_height = self._get_ground_height(take_uid)
            metadata = {
                "take_uid": take_uid,
                "take_name": take_name,
                "exo_cam_names": exo_cam_names.tolist(),
                "exo_camera_intrinsics": {k: v.tolist() for k, v in curr_intrs.items()},
                "exo_camera_extrinsics": {k: v.tolist() for k, v in curr_extrs.items()},
                "ground_height": ground_height,
            }
            curr_take_db["metadata"] = metadata

        return curr_take_db

    def load_frame_body_2d_3d_kpts(self, frame_anno, egoexo_cam_names):
        """
        load frame body 2d and 3d kpts for this frame.

        Parameters
        ----------
        frame_anno : dict, annotation for current frame
        egoexo_cam_names : list,  egoexo camera names

        Returns
        -------
        curr_frame_2d_kpts : dict of numpy array of shape (17,2) 
            each key being the egoexo cam name, with corresponding value being the 2D body keypoints in original frame
        curr_frame_3d_kpts : (17,3) 3D body keypoints in world coordinate system
        joints_view_stat : (17,) Number of triangulation views for each 3D body keypoints

        """
        ### Load 2D GT body kpts ###
        # Return NaN if no annotation exists
        all_cam_curr_frame_2d_kpts = {}
        for egoexo_cam_name in egoexo_cam_names:
            if (
                len(frame_anno) == 0
                or "annotation2D" not in frame_anno[0].keys()
                or egoexo_cam_name not in frame_anno[0]["annotation2D"].keys()
                or len(frame_anno[0]["annotation2D"][egoexo_cam_name]) == 0
            ):
                curr_frame_2d_kpts = [[None, None] for _ in range(self.num_joints)]
            else:
                curr_frame_2d_anno = frame_anno[0]["annotation2D"][egoexo_cam_name]
                curr_frame_2d_kpts = []
                # Load 3D annotation for both bodys
                for body_jnt in EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS:
                    if body_jnt in curr_frame_2d_anno.keys():
                        curr_frame_2d_kpts.append(
                            [
                                curr_frame_2d_anno[body_jnt]["x"],
                                curr_frame_2d_anno[body_jnt]["y"],
                            ]
                        )
                    else:
                        curr_frame_2d_kpts.append([None, None])
            all_cam_curr_frame_2d_kpts[egoexo_cam_name] = np.array(curr_frame_2d_kpts).astype(np.float32)

        ### Load 3D GT body kpts ###
        # Return NaN if no annotation exists
        if (
            len(frame_anno) == 0
            or "annotation3D" not in frame_anno[0].keys()
            or len(frame_anno[0]["annotation3D"]) == 0
        ):
            return None, None, None
        else:
            curr_frame_3d_anno = frame_anno[0]["annotation3D"]
            curr_frame_3d_kpts = []
            joints_view_stat = []
            # Load 3D annotation for both bodys
            for body_jnt in EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS:
                if body_jnt in curr_frame_3d_anno.keys() and (
                    curr_frame_3d_anno[body_jnt]["num_views_for_3d"]
                    >= 2
                    or self.anno_type == "auto"
                ):
                    curr_frame_3d_kpts.append(
                        [
                            curr_frame_3d_anno[body_jnt]["x"],
                            curr_frame_3d_anno[body_jnt]["y"],
                            curr_frame_3d_anno[body_jnt]["z"],
                        ]
                    )
                    joints_view_stat.append(
                        curr_frame_3d_anno[body_jnt][
                            "num_views_for_3d"
                        ]
                    )
                else:
                    curr_frame_3d_kpts.append([None, None, None])
                    joints_view_stat.append(None)
                    
        return (
            all_cam_curr_frame_2d_kpts,
            np.array(curr_frame_3d_kpts).astype(np.float32),
            np.array(joints_view_stat).astype(np.float32),
        )

    def load_static_ego_exo_cam_poses(self, cam_pose, egoexo_cam_names):
        """
        Load static camera poses for ego and exo cameras for current frame
        Retrun a dict for each key being the egoexo_cam_name
        NOTE: intr is 3x3 matrix, extr is 3x4 matrix"""
        # Check if current frame has corresponding camera pose
        curr_intrs = {}
        curr_extrs = {}
        for egoexo_cam_name in egoexo_cam_names:
            if (
                egoexo_cam_name not in cam_pose.keys()
                or "camera_intrinsics" not in cam_pose[egoexo_cam_name].keys()
                or "camera_extrinsics" not in cam_pose[egoexo_cam_name].keys()
            ):
                curr_cam_intrinsics, curr_cam_extrinsics = None, None
            else:
                # Build camera projection matrix
                curr_cam_intrinsics = np.array(
                    cam_pose[egoexo_cam_name]["camera_intrinsics"]
                ).astype(np.float32)
                curr_cam_extrinsics = np.array(
                    cam_pose[egoexo_cam_name]["camera_extrinsics"]
                ).astype(np.float32)
            curr_intrs[egoexo_cam_name] = curr_cam_intrinsics
            curr_extrs[egoexo_cam_name] = curr_cam_extrinsics
        return curr_intrs, curr_extrs

    def body_kpts_valid_check(self, kpts, egoexo_cam_mask):
        """
        Return valid kpts with three checks:
            - Has valid kpts
            - Within image bound
            - Visible within aria mask
        Input:
            kpts: (17,2) raw single 2D body kpts
            egoexo_cam_masks: (H,W) binary mask that has same shape as undistorted aria image
        Output:
            new_kpts: (17,2)
            flag: (17,)
        """
        new_kpts = kpts.copy()
        # 1. Check missing annotation kpts
        miss_anno_flag = np.any(np.isnan(kpts), axis=1)
        new_kpts[miss_anno_flag] = 0
        # 2. Check out-bound annotation kpts
        # Width
        x_out_bound = np.logical_or(
            new_kpts[:, 0] < 0, new_kpts[:, 0] >= self.undist_img_dim[1]
        )
        # Height
        y_out_bound = np.logical_or(
            new_kpts[:, 1] < 0, new_kpts[:, 1] >= self.undist_img_dim[0]
        )
        out_bound_flag = np.logical_or(x_out_bound, y_out_bound)
        new_kpts[out_bound_flag] = 0
        # 3. Check in-bound but invisible kpts
        invis_flag = (
            egoexo_cam_mask[new_kpts[:, 1].astype(np.int64), new_kpts[:, 0].astype(np.int64)]
            == 0
        )
        # 4. Get valid flag
        invalid_flag = miss_anno_flag + out_bound_flag + invis_flag
        valid_flag = ~invalid_flag
        # 5. Assign invalid kpts as None
        new_kpts[invalid_flag] = None

        return new_kpts, valid_flag

    def _prepare_valid_frame_data(
        self,
        body_3d_kpts: np.ndarray,
        valid_kpts_mask: np.ndarray,
        ego_intrinsics: np.ndarray,
        ego_extrinsics: np.ndarray,
        aria_cam_name: str
    ) -> Dict[str, Any]:
        """
        Prepare frame data for valid frames by processing 3D keypoints and camera parameters.
        
        Args:
            body_3d_kpts: Array of shape (num_joints, 3) containing 3D keypoints in world coordinates
            valid_kpts_mask: Boolean array of shape (num_joints,) indicating valid keypoints
            ego_intrinsics: 3x3 camera intrinsic matrix
            ego_extrinsics: 3x4 camera extrinsic matrix
            aria_cam_name: Name of the aria camera
            
        Returns:
            Dictionary containing processed frame data including:
            - Filtered 3D keypoints (world and camera coordinates)
            - Camera parameters
            - Validation masks
        """
        # Filter 3D keypoints based on validation mask
        filtered_3d_kpts_world = body_3d_kpts.copy()
        filtered_3d_kpts_world[~valid_kpts_mask] = None
        
        # Transform to camera coordinates
        body_3d_kpts_cam = world_to_cam(body_3d_kpts, ego_extrinsics)
        filtered_3d_kpts_cam = body_3d_kpts_cam.copy()
        filtered_3d_kpts_cam[~valid_kpts_mask] = None
        
        # Project to 2D for verification
        body_2d_kpts = cam_to_img(body_3d_kpts_cam, ego_intrinsics)
        if self.portrait_view:
            body_2d_kpts = aria_landscape_to_portrait(
                body_2d_kpts, 
                self.undist_img_dim
            )
        
        # Generate body bbox
        if self.split == "test":
            body_bbox = rand_bbox_from_kpts(
                body_2d_kpts[valid_kpts_mask],
                self.undist_img_dim
            )
        else:
            body_bbox = pad_bbox_from_kpts(
                body_2d_kpts[valid_kpts_mask],
                self.undist_img_dim,
                self.bbox_padding
            )
        
        # Verify keypoint filtering consistency
        assert np.sum(np.isnan(np.mean(filtered_3d_kpts_cam, axis=1))) == np.sum(~valid_kpts_mask), \
            "Mismatch in number of filtered keypoints between mask and processed data"
        
        # Prepare frame data dictionary
        frame_data = {
            "body_3d_world": filtered_3d_kpts_world.tolist(),
            "body_3d_cam": filtered_3d_kpts_cam.tolist(),
            "body_2d": body_2d_kpts.tolist(),
            "body_bbox": body_bbox.tolist(),
            "body_valid_3d": valid_kpts_mask.tolist(),
            "ego_camera_intrinsics": ego_intrinsics.tolist(),
            "ego_camera_extrinsics": ego_extrinsics.tolist(),
            "aria_camera_name": aria_cam_name
        }
        
        return frame_data

