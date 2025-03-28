import json
import os

import cv2
import numpy as np
from egoallo.egoexo import EGOEXO_UTILS_INST
from egoallo.egoexo.egoexo_utils import EgoExoUtils
from egoallo.egopose.handpose.data_preparation.utils.utils import (
    aria_landscape_to_portrait,
)
from egoallo.egopose.handpose.data_preparation.utils.utils import cam_to_img
from egoallo.egopose.handpose.data_preparation.utils.utils import (
    get_ego_pose_takes_from_splits,
)
from egoallo.egopose.handpose.data_preparation.utils.utils import (
    hand_jnts_dist_angle_check,
)
from egoallo.egopose.handpose.data_preparation.utils.utils import HAND_ORDER
from egoallo.egopose.handpose.data_preparation.utils.utils import pad_bbox_from_kpts
from egoallo.egopose.handpose.data_preparation.utils.utils import rand_bbox_from_kpts
from egoallo.egopose.handpose.data_preparation.utils.utils import reproj_error_check
from egoallo.egopose.handpose.data_preparation.utils.utils import world_to_cam
from egoallo.utils.setup_logger import setup_logger
from egoallo.utils.smpl_mapping.mapping import EGOEXO4D_EGOPOSE_HANDPOSE_MAPPINGS
from projectaria_tools.core import calibration
from tqdm import tqdm

HAND_JOINTS = EGOEXO4D_EGOPOSE_HANDPOSE_MAPPINGS
NUM_OF_HAND_JOINTS = len(HAND_JOINTS) // 2


logger = setup_logger(output=None, name=__name__)


class hand_pose_anno_loader:
    """
    Load Ego4D data and create ground truth annotation JSON file for Ego-pose baseline model
    """

    def __init__(self, args, split, anno_type):
        # Set dataloader parameters
        self.dataset_root = args.egoexo_root_path
        self.anno_type = anno_type
        self.split = split
        self.num_joints = NUM_OF_HAND_JOINTS  # Number of joints for single hand
        self.undist_img_dim = (512, 512)  # Dimension of undistorted aria image [H, W]
        self.valid_kpts_threshold = (
            args.valid_kpts_num_thresh
        )  # Threshold of minimum number of valid kpts in single hand
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

        # Determine annotation and camera pose directory
        anno_type_dir_dict = {"manual": "annotation", "auto": "automatic"}
        self.hand_anno_dir = (
            os.path.join(self.dataset_root, "annotations/ego_pose/test/hand/annotation")
            if self.split == "test"
            else os.path.join(
                self.dataset_root,
                f"annotations/ego_pose/{split}/hand",
                anno_type_dir_dict[self.anno_type],
            )
        )
        self.cam_pose_dir = os.path.join(
            self.dataset_root,
            f"annotations/ego_pose/{split}/camera_pose",
        )

        # Load dataset
        self.db = self.load_raw_data()

    def load_raw_data(self):
        gt_db = {}

        # Find all annotation takes from local direcctory by splits
        # Check test anno availability. No gt-anno will be generated for public.
        if not os.path.exists(self.hand_anno_dir):
            assert self.split == "test", (
                f"No annotation found for {self.split} split at {self.hand_anno_dir}.\
                Make sure you follow step 0 to download data first."
            )
            # return gt_db
        # Get all local annotation takes for train/val split
        if self.split == "test":
            pass
        else:
            split_all_local_takes = [
                k.split(".")[0] for k in os.listdir(self.hand_anno_dir)
            ]
            # take to uid dict
            {
                t["take_name"]: t["take_uid"]
                for t in self.takes
                if t["take_uid"] in split_all_local_takes
            }
            # uid_to_take = {uid: take for take, uid in take_to_uid.items()}
            # (0). Filter common takes
            comm_local_take_uid = list(set(split_all_local_takes))

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
            comm_take_w_cam_pose = sorted(
                list(set(comm_take) & set(comm_local_take_uid)),
            )

        logger.info(
            f"Find {len(comm_take_w_cam_pose)} takes in {self.split} ({self.anno_type}) dataset. Start data processing...",
        )

        # Iterate through all takes from annotation directory and check
        for curr_take_uid in tqdm(comm_take_w_cam_pose):
            curr_take_name = EGOEXO_UTILS_INST.find_take_name_from_take_uid(
                curr_take_uid,
            )
            assert curr_take_name is not None, (
                f"Take name not found for {curr_take_uid}."
            )
            # Load annotation, camera pose JSON and image directory
            curr_take_anno_path = os.path.join(
                self.hand_anno_dir,
                f"{curr_take_uid}.json",
            )
            curr_take_cam_pose_path = os.path.join(
                self.cam_pose_dir,
                f"{curr_take_uid}.json",
            )
            # Load in annotation JSON and image directory
            curr_take_cam_pose = json.load(open(curr_take_cam_pose_path))

            if self.split == "test":
                aria_mask, aria_cam_name = self.load_aria_calib(curr_take_name)
                curr_take_data = self.load_take_raw_data_for_test(
                    curr_take_name,
                    curr_take_uid,
                    curr_take_cam_pose,
                    aria_cam_name,
                )
            else:
                curr_take_anno = json.load(open(curr_take_anno_path))

                # Get valid takes info for all frames
                if len(curr_take_anno) > 0:
                    aria_mask, aria_cam_name = self.load_aria_calib(curr_take_name)
                    if aria_mask is not None:
                        curr_take_data = self.load_take_raw_data(
                            curr_take_name,
                            curr_take_uid,
                            curr_take_anno,
                            curr_take_cam_pose,
                            aria_cam_name,
                            aria_mask,
                        )

            # Append into dataset if has at least valid annotation
            if len(curr_take_data) > 0:
                gt_db[curr_take_uid] = curr_take_data
        return gt_db

    def load_take_raw_data_for_test(
        self,
        take_name,
        take_uid,
        cam_pose,
        aria_cam_name,
    ):
        """
        Load raw data for test split. only the cam intr/extr,(take_uid,take_name,frame_number) is loaded into metadata saved into every timestamp. Others data are None."""
        curr_take_db = {}
        if (
            aria_cam_name not in cam_pose.keys()
            or "camera_intrinsics" not in cam_pose[aria_cam_name].keys()
            or "camera_extrinsics" not in cam_pose[aria_cam_name].keys()
        ):
            return curr_take_db
        # Build camera projection matrix
        curr_intri = np.array(cam_pose[aria_cam_name]["camera_intrinsics"]).astype(
            np.float32,
        )
        for frame_idx in cam_pose[aria_cam_name]["camera_extrinsics"].keys():
            curr_frame_anno = {}
            curr_extri = np.array(
                cam_pose[aria_cam_name]["camera_extrinsics"][frame_idx],
            ).astype(np.float32)
            # Compose current hand GT info in current frame
            for hand_idx, hand_name in enumerate(HAND_ORDER):
                curr_frame_anno[f"{hand_name}_hand_3d_cam"] = None
                curr_frame_anno[f"{hand_name}_hand_3d_world"] = None
                curr_frame_anno[f"{hand_name}_hand_2d"] = None
                curr_frame_anno[f"{hand_name}_hand_bbox"] = None
                curr_frame_anno[f"{hand_name}_hand_valid_3d"] = None

            # Append current frame into GT JSON if at least one valid hand exists
            metadata = {
                "take_uid": take_uid,
                "take_name": take_name,
                "frame_number": int(frame_idx),
                "camera_intrinsics": curr_intri.tolist(),
                "camera_extrinsics": curr_extri.tolist(),
                "camera_name": aria_cam_name,
            }
            curr_frame_anno["metadata"] = metadata
            curr_take_db[frame_idx] = curr_frame_anno
        return curr_take_db

    def load_take_raw_data(
        self,
        take_name,
        take_uid,
        anno,
        cam_pose,
        aria_cam_name,
        aria_mask,
    ):
        curr_take_db = {}

        for frame_idx, curr_frame_anno in anno.items():
            # Load in current frame's 2D & 3D annotation and camera parameter
            curr_hand_2d_kpts, curr_hand_3d_kpts, _ = self.load_frame_hand_2d_3d_kpts(
                curr_frame_anno,
                aria_cam_name,
            )
            curr_ego_intr, curr_ego_extr = self.load_frame_cam_pose(
                frame_idx,
                cam_pose,
                aria_cam_name,
            )
            # Skip this frame if missing valid data
            if (
                curr_hand_3d_kpts is None
                or curr_ego_intr is None
                or curr_ego_extr is None
            ):
                continue
            # Look at each hand in current frame
            curr_frame_anno = {}
            at_least_one_hands_valid = False
            for hand_idx, hand_name in enumerate(HAND_ORDER):
                # Get current hand's 2D kpts and 3D world kpts
                start_idx, end_idx = (
                    self.num_joints * hand_idx,
                    self.num_joints * (hand_idx + 1),
                )
                one_hand_2d_kpts = curr_hand_2d_kpts[start_idx:end_idx]
                # Transform annotation 2d kpts if in portrait view
                if self.portrait_view:
                    one_hand_2d_kpts = aria_landscape_to_portrait(
                        one_hand_2d_kpts,
                        self.undist_img_dim,
                    )
                one_hand_3d_kpts_world = curr_hand_3d_kpts[start_idx:end_idx]
                # Skip this hand if the hand wrist (root) is None
                if np.any(np.isnan(one_hand_3d_kpts_world[0])):
                    one_hand_3d_kpts_world[:, :] = None

                # Hand biomechanical structure check
                one_hand_3d_kpts_world = hand_jnts_dist_angle_check(
                    one_hand_3d_kpts_world,
                )
                # 3D world to camera
                one_hand_3d_kpts_cam = world_to_cam(
                    one_hand_3d_kpts_world,
                    curr_ego_extr,
                )
                # Camera to image plane
                one_hand_proj_2d_kpts = cam_to_img(one_hand_3d_kpts_cam, curr_ego_intr)
                # Transform projected 2d kpts if in portrait view
                if self.portrait_view:
                    one_hand_proj_2d_kpts = aria_landscape_to_portrait(
                        one_hand_proj_2d_kpts,
                        self.undist_img_dim,
                    )

                # Filter projected 2D kpts
                (
                    one_hand_filtered_proj_2d_kpts,
                    valid_proj_2d_flag,
                ) = self.one_hand_kpts_valid_check(one_hand_proj_2d_kpts, aria_mask)

                # Filter 2D annotation kpts
                one_hand_filtered_anno_2d_kpts, _ = self.one_hand_kpts_valid_check(
                    one_hand_2d_kpts,
                    aria_mask,
                )

                # Filter 3D anno by checking reprojection error with 2D anno (which is usually better)
                valid_reproj_flag = reproj_error_check(
                    one_hand_filtered_proj_2d_kpts,
                    one_hand_filtered_anno_2d_kpts,
                    self.reproj_error_threshold,
                )
                valid_3d_kpts_flag = valid_proj_2d_flag * valid_reproj_flag

                # Prepare 2d kpts, 3d kpts, bbox and flag data based on number of valid 3D kpts
                if sum(valid_3d_kpts_flag) >= self.valid_kpts_threshold:
                    at_least_one_hands_valid = True
                    # Assign original hand wrist 3d kpts back (needed for offset hand wrist)
                    one_hand_filtered_3d_kpts_cam = one_hand_3d_kpts_cam.copy()
                    one_hand_filtered_3d_kpts_cam[~valid_3d_kpts_flag] = None
                    one_hand_filtered_3d_kpts_cam[0] = one_hand_3d_kpts_cam[0]
                    one_hand_filtered_3d_kpts_world = one_hand_3d_kpts_world.copy()
                    one_hand_filtered_3d_kpts_world[~valid_3d_kpts_flag] = None
                    one_hand_filtered_3d_kpts_world[0] = one_hand_3d_kpts_world[0]
                    valid_3d_kpts_flag = ~np.isnan(
                        np.mean(one_hand_filtered_3d_kpts_cam, axis=1),
                    )
                    # Generate hand bbox based on 2D GT kpts
                    if self.split == "test":
                        one_hand_bbox = rand_bbox_from_kpts(
                            one_hand_filtered_proj_2d_kpts[valid_3d_kpts_flag],
                            self.undist_img_dim,
                        )
                    else:
                        # For train and val, generate hand bbox with padding
                        one_hand_bbox = pad_bbox_from_kpts(
                            one_hand_filtered_proj_2d_kpts[valid_3d_kpts_flag],
                            self.undist_img_dim,
                            self.bbox_padding,
                        )
                    assert sum(
                        np.isnan(np.mean(one_hand_filtered_3d_kpts_cam, axis=1)),
                    ) == sum(~valid_3d_kpts_flag), (
                        f"Missing 3d kpts count mismatch: {sum(np.isnan(np.mean(one_hand_filtered_3d_kpts_cam, axis=1)))} vs {sum(~valid_3d_kpts_flag)}"
                    )
                    assert sum(
                        np.isnan(np.mean(one_hand_filtered_3d_kpts_world, axis=1)),
                    ) == sum(~valid_3d_kpts_flag), (
                        f"Missing 3d kpts count mismatch: {sum(np.isnan(np.mean(one_hand_filtered_3d_kpts_world, axis=1)))} vs {sum(~valid_3d_kpts_flag)}"
                    )
                # If no valid annotation for current hand, assign empty bbox, anno and valid flag
                else:
                    one_hand_bbox = np.array([])
                    one_hand_filtered_3d_kpts_cam = np.array([])
                    one_hand_filtered_3d_kpts_world = np.array([])
                    one_hand_filtered_anno_2d_kpts = np.array([])
                    valid_3d_kpts_flag = np.array([])

                # Compose current hand GT info in current frame
                curr_frame_anno[f"{hand_name}_hand_3d_cam"] = (
                    one_hand_filtered_3d_kpts_cam.tolist()
                )
                curr_frame_anno[f"{hand_name}_hand_3d_world"] = (
                    one_hand_filtered_3d_kpts_world.tolist()
                )
                curr_frame_anno[f"{hand_name}_hand_2d"] = (
                    one_hand_filtered_anno_2d_kpts.tolist()
                )
                curr_frame_anno[f"{hand_name}_hand_bbox"] = one_hand_bbox.tolist()
                curr_frame_anno[f"{hand_name}_hand_valid_3d"] = (
                    valid_3d_kpts_flag.tolist()
                )

            # Append current frame into GT JSON if at least one valid hand exists
            if at_least_one_hands_valid:
                metadata = {
                    "take_uid": take_uid,
                    "take_name": take_name,
                    "frame_number": int(frame_idx),
                    "camera_intrinsics": curr_ego_intr.tolist(),
                    "camera_extrinsics": curr_ego_extr.tolist(),
                    "camera_name": aria_cam_name,
                }
                curr_frame_anno["metadata"] = metadata
                curr_take_db[frame_idx] = curr_frame_anno

        return curr_take_db

    def load_aria_calib(self, curr_take_name):
        # Find aria names
        take = [t for t in self.takes if t["take_name"] == curr_take_name]
        take = take[0]
        aria_cam_name = EgoExoUtils.get_ego_aria_cam_name(take)
        # Load aria calibration model
        curr_aria_calib_json_path = os.path.join(
            self.aria_calib_dir,
            f"{curr_take_name}.json",
        )
        if not os.path.exists(curr_aria_calib_json_path):
            logger.warning(
                f"No Aria calibration JSON file found at {curr_aria_calib_json_path}. Skipped this take.",
            )
            return None, None
        aria_rgb_calib = calibration.device_calibration_from_json(
            curr_aria_calib_json_path,
        ).get_camera_calib("camera-rgb")
        dst_cam_calib = calibration.get_linear_camera_calibration(512, 512, 150)
        # Generate mask in undistorted aria view
        mask = np.full((1408, 1408), 255, dtype=np.uint8)
        undistorted_mask = calibration.distort_by_calibration(
            mask,
            dst_cam_calib,
            aria_rgb_calib,
        )
        undistorted_mask = (
            cv2.rotate(undistorted_mask, cv2.ROTATE_90_CLOCKWISE)
            if self.portrait_view
            else undistorted_mask
        )
        undistorted_mask = undistorted_mask / 255
        return undistorted_mask, aria_cam_name

    def load_frame_hand_2d_3d_kpts(self, frame_anno, aria_cam_name):
        """
        Input:
            frame_anno: annotation for current frame
            aria_cam_name: aria camera name
        Output:
            curr_frame_2d_kpts: (42,2) 2D hand keypoints in original frame
            curr_frame_3d_kpts: (42,3) 3D hand keypoints in world coordinate system
            joints_view_stat: (42,) Number of triangulation views for each 3D hand keypoints
        """
        # Finger dict to load annotation
        finger_dict = {
            "wrist": None,
            "thumb": [1, 2, 3, 4],
            "index": [1, 2, 3, 4],
            "middle": [1, 2, 3, 4],
            "ring": [1, 2, 3, 4],
            "pinky": [1, 2, 3, 4],
        }

        ### Load 2D GT hand kpts ###
        # Return NaN if no annotation exists
        if (
            len(frame_anno) == 0
            or "annotation2D" not in frame_anno[0].keys()
            or aria_cam_name not in frame_anno[0]["annotation2D"].keys()
            or len(frame_anno[0]["annotation2D"][aria_cam_name]) == 0
        ):
            curr_frame_2d_kpts = [[None, None] for _ in range(42)]
        else:
            curr_frame_2d_anno = frame_anno[0]["annotation2D"][aria_cam_name]
            curr_frame_2d_kpts = []
            # Load 3D annotation for both hands
            for hand in HAND_ORDER:
                for finger, finger_joint_order in finger_dict.items():
                    if finger_joint_order:
                        for finger_joint_idx in finger_joint_order:
                            finger_k_json = f"{hand}_{finger}_{finger_joint_idx}"
                            # Load 3D if exist annotation, and check for minimum number of visible views
                            if finger_k_json in curr_frame_2d_anno.keys():
                                curr_frame_2d_kpts.append(
                                    [
                                        curr_frame_2d_anno[finger_k_json]["x"],
                                        curr_frame_2d_anno[finger_k_json]["y"],
                                    ],
                                )
                            else:
                                curr_frame_2d_kpts.append([None, None])
                    else:
                        finger_k_json = f"{hand}_{finger}"
                        # Load 3D if exist annotation, and check for minimum number of visible views
                        if finger_k_json in curr_frame_2d_anno.keys():
                            curr_frame_2d_kpts.append(
                                [
                                    curr_frame_2d_anno[finger_k_json]["x"],
                                    curr_frame_2d_anno[finger_k_json]["y"],
                                ],
                            )
                        else:
                            curr_frame_2d_kpts.append([None, None])

        ### Load 3D GT hand kpts ###
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
            # Load 3D annotation for both hands
            for hand in HAND_ORDER:
                for finger, finger_joint_order in finger_dict.items():
                    if finger_joint_order:
                        for finger_joint_idx in finger_joint_order:
                            finger_k_json = f"{hand}_{finger}_{finger_joint_idx}"
                            # Load 3D if exist annotation, and check for minimum number of visible views for manual annotations
                            if finger_k_json in curr_frame_3d_anno.keys() and (
                                curr_frame_3d_anno[finger_k_json]["num_views_for_3d"]
                                >= 3
                                or self.anno_type == "auto"
                            ):
                                curr_frame_3d_kpts.append(
                                    [
                                        curr_frame_3d_anno[finger_k_json]["x"],
                                        curr_frame_3d_anno[finger_k_json]["y"],
                                        curr_frame_3d_anno[finger_k_json]["z"],
                                    ],
                                )
                                joints_view_stat.append(
                                    curr_frame_3d_anno[finger_k_json][
                                        "num_views_for_3d"
                                    ],
                                )
                            else:
                                curr_frame_3d_kpts.append([None, None, None])
                                joints_view_stat.append(None)
                    else:
                        finger_k_json = f"{hand}_{finger}"
                        # Load 3D if exist annotation, and check for minimum number of visible views
                        if finger_k_json in curr_frame_3d_anno.keys() and (
                            curr_frame_3d_anno[finger_k_json]["num_views_for_3d"] >= 3
                            or self.anno_type == "auto"
                        ):
                            curr_frame_3d_kpts.append(
                                [
                                    curr_frame_3d_anno[finger_k_json]["x"],
                                    curr_frame_3d_anno[finger_k_json]["y"],
                                    curr_frame_3d_anno[finger_k_json]["z"],
                                ],
                            )
                            joints_view_stat.append(
                                curr_frame_3d_anno[finger_k_json]["num_views_for_3d"],
                            )
                        else:
                            curr_frame_3d_kpts.append([None, None, None])
                            joints_view_stat.append(None)
        return (
            np.array(curr_frame_2d_kpts).astype(np.float32),
            np.array(curr_frame_3d_kpts).astype(np.float32),
            np.array(joints_view_stat).astype(np.float32),
        )

    def load_frame_cam_pose(self, frame_idx, cam_pose, aria_cam_name):
        # Check if current frame has corresponding camera pose
        # returns: curr_cam_intrinsic: (3,3), curr_cam_extrinsics: (3,4)
        if (
            aria_cam_name not in cam_pose.keys()
            or "camera_intrinsics" not in cam_pose[aria_cam_name].keys()
            or "camera_extrinsics" not in cam_pose[aria_cam_name].keys()
            or frame_idx not in cam_pose[aria_cam_name]["camera_extrinsics"].keys()
        ):
            return None, None
        # Build camera projection matrix
        curr_cam_intrinsic = np.array(
            cam_pose[aria_cam_name]["camera_intrinsics"],
        ).astype(np.float32)
        curr_cam_extrinsics = np.array(
            cam_pose[aria_cam_name]["camera_extrinsics"][frame_idx],
        ).astype(np.float32)
        return curr_cam_intrinsic, curr_cam_extrinsics

    def one_hand_kpts_valid_check(self, kpts, aria_mask):
        """
        Return valid kpts with three checks:
            - Has valid kpts
            - Within image bound
            - Visible within aria mask
        Input:
            kpts: (21,2) raw single 2D hand kpts
            aria_mask: (H,W) binary mask that has same shape as undistorted aria image
        Output:
            new_kpts: (21,2)
            flag: (21,)
        """
        new_kpts = kpts.copy()
        # 1. Check missing annotation kpts
        miss_anno_flag = np.any(np.isnan(kpts), axis=1)
        new_kpts[miss_anno_flag] = 0
        # 2. Check out-bound annotation kpts
        x_out_bound = np.logical_or(
            new_kpts[:, 0] < 0,
            new_kpts[:, 0] >= self.undist_img_dim[1],
        )
        y_out_bound = np.logical_or(
            new_kpts[:, 1] < 0,
            new_kpts[:, 1] >= self.undist_img_dim[0],
        )
        out_bound_flag = np.logical_or(x_out_bound, y_out_bound)
        new_kpts[out_bound_flag] = 0
        # 3. Check in-bound but invisible kpts
        invis_flag = (
            aria_mask[new_kpts[:, 1].astype(np.int64), new_kpts[:, 0].astype(np.int64)]
            == 0
        )
        # 4. Get valid flag
        invalid_flag = miss_anno_flag + out_bound_flag + invis_flag
        valid_flag = ~invalid_flag
        # 5. Assign invalid kpts as None
        new_kpts[invalid_flag] = None

        return new_kpts, valid_flag
