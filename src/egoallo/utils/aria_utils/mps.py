import json
from egoallo.utils.utils import images_to_video, images_to_video_w_imageio
import os.path as osp
import open3d as o3d
from typing import Optional
from tqdm import tqdm

import polyscope as ps
import numpy as np
from collections import defaultdict
import plotly.graph_objs as go
from pytorch3d import transforms
import torch
from typing import Tuple, Dict, List
from projectaria_tools.core import mps  # IO for Aria MPS assets
from projectaria_tools.core.mps.utils import (
    get_nearest_pose,
    filter_points_from_count,
    filter_points_from_confidence,
)
from projectaria_tools.utils.vrs_to_mp4_utils import get_timestamp_from_mp4

from egoallo.egoexo.egoexo_utils import EgoExoUtils
from egoallo.smpl.smplh_utils import (
    NUM_EGOEXO4D_EGOPOSE_JNTS,
)
from egoallo.utils.ego_geom import (
    aria_camera_device2opengl_pose,
)
from egoallo.utils.setup_logger import setup_logger
from egoallo.utils.utils import find_numerical_key_in_dict
from egoallo.utils.transformation import qpose_to_T, T_to_qpose
from egoallo.config import make_cfg, CONFIG_FILE
from egoallo.utils.utils import NDArray
from egoallo.utils.aria_utils.open3d_vis import (
    build_cam_frustum_w_extr as build_cam_frustum_w_extr_open3d,
)
from egoallo.utils.aria_utils.open3d_vis import (
    draw_coco_kinematic_tree as draw_coco_kinematic_tree_open3d,
)
from egoallo.utils.aria_utils.ps_vis import (
    build_cam_frustum_w_extr as build_cam_frustum_w_extr_ps,
)
from egoallo.utils.aria_utils.ps_vis import (
    draw_coco_kinematic_tree as draw_coco_kinematic_tree_ps,
)
from egoallo.utils.aria_utils.ps_vis import draw_camera_pose as draw_camera_pose_ps
from egoallo.utils.aria_utils.plotly_vis import (
    build_cam_frustum_w_extr as build_cam_frustum_w_extr_plotly,
)
from egoallo.utils.aria_utils.plotly_vis import (
    draw_coco_kinematic_tree as draw_coco_kinematic_tree_plotly,
)
from egoallo.utils.aria_utils.plotly_vis import (
    draw_camera_pose as draw_camera_pose_plotly,
)


local_config_file = CONFIG_FILE
CFG = make_cfg(config_name="defaults", config_file=local_config_file, cli_args=[])

logger = setup_logger(output=None, name=__name__)


class AriaMPSService:
    # ## Visualize the trajectory and point cloud in a 3D interactive plot
    #
    # -   Load trajectory
    # -   Load global point cloud
    # -   Render dense trajectory (1Khz) as points.
    # -   Render subsampled 6DOF poses via camera frustum. Use calibration to transform RGB camera pose to world frame
    # -   Render subsampled point cloud
    #
    # _Please wait a minute for all the data to load. Zoom in to the point cloud and adjust your view. Then use the time slider to move the camera_
    #

    # Load MPS output
    #
    # The loaders for MPS outputs (projectaria_tools/main/core/mps) make it easer to use the data downstream. As part of this, the loaders put the outputs into data structures that are easier for other tools to consume.
    #
    # MPS Data Formats provides details about output schemas and the specifics of each MPS output. This page focuses loading APIs in Python and C++, where there isn't a standalone code samples page:
    #
    #     Eye Gaze Code Samples
    #
    # Open loop/Closed loop trajectory
    #

    # To extract the vrs timestamps, use the get_timestamp_from_mp4 function. The resulting timestamp is in TimeDomain.DEVICE_TIME
    #

    QUAT_XYZW_TO_WXYZ = np.asarray([3, 0, 1, 2])
    QUAT_WXYZ_TO_XYZW = np.asarray([1, 2, 3, 0])

    def __init__(
        self,
        vrs_file_path,
        vrs_exported_mp4_path,
        take,
        open_loop_traj_path,
        close_loop_traj_path,
        gopro_calibs_path,
        cam_pose_anno_path,
        generalized_eye_gaze_path,
        calibrated_eye_gaze_path,
        online_calib_json_path,
        wrist_and_palm_poses_path,
        point_observation_path,
        lazy_loading=True,
    ):
        self.vrs_path = vrs_file_path
        self.vrs_exported_mp4_path = vrs_exported_mp4_path
        self.take = take
        self.open_loop_traj_path = open_loop_traj_path
        self.closed_loop_traj_path = close_loop_traj_path
        self.goprocalibs_path = gopro_calibs_path
        self.cam_pose_anno_path = cam_pose_anno_path
        self.generalized_eye_gaze_path = generalized_eye_gaze_path
        self.calibrated_eye_gaze_path = calibrated_eye_gaze_path
        self.online_calib_json_path = online_calib_json_path
        self.wrist_and_palm_poses_path = wrist_and_palm_poses_path
        self.point_observation_path = point_observation_path
        self.take_root_path = osp.dirname(vrs_file_path)
        if not lazy_loading:
            self.mps_data_paths_provider = mps.MpsDataPathsProvider(self.take_root_path)
            self.mps_data_paths = self.mps_data_paths_provider.get_data_paths()
            self.mps_data_provider = mps.MpsDataProvider(self.mps_data_paths)
        else:
            self.mps_data_provider = None

        # self.provider = data_provider.create_vrs_data_provider(vrs_file_path)
        # if not self.provider:
        #     raise ValueError("Invalid VRS data provider")

    def get_timestamps_from_exported_mp4(self):
        """Sample timestamps from exported mp4 file."""
        try:
            vrs_timestamps = get_timestamp_from_mp4(self.vrs_exported_mp4_path)
            return vrs_timestamps
        except Exception as e:
            logger.info(f"Error encoutered during get timestamps from mp4 files: {e}")

    def sample_close_loop_traj_from_timestamps(
        self, timestamps: NDArray, subsample_rate: int = 1
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """Read close loop traj.

        Parameters
        ------
            timestamps : np.array of shape (N,)
            subsample_rate (int, optional): Defaults to 100.

        Returns
        -----
            sampled_close_loop_traj_pose : numpy.ndarray of shape (N, 7)
                representing the 7-dim pose
            T_world_imus : numpy.ndarray of shape (N, 4, 4)
                representing T_world_imu
            close_timestamp_tracking_s : numpy.ndarray of shape (N,)
                the DEVICE_TIME timestamp in seconds
            close_utc_timestamp : numpy.ndarray of shape (N,)
                the UTC timestamp in seconds
        """
        if not isinstance(self.closed_loop_traj_path, str) or not osp.exists(
            self.closed_loop_traj_path
        ):
            logger.warning(
                f"close loop trajectory file not found at {self.closed_loop_traj_path}"
            )
            return None, None, None, None

        closed_loop_traj = mps.read_closed_loop_trajectory(self.closed_loop_traj_path)
        sampled_loop_traj = []
        for sample_idx, sample_timestamp in enumerate(timestamps):
            sampled_loop_traj.append(
                get_nearest_pose(closed_loop_traj, sample_timestamp)
            )

        sampled_loop_traj = sampled_loop_traj[::subsample_rate]

        sampled_close_loop_traj_pose = np.empty([len(sampled_loop_traj), 7])
        T_world_imus = np.tile(
            np.eye(4)[np.newaxis, :, :], (len(sampled_loop_traj), 1, 1)
        )
        close_timestamp_tracking_s = np.empty([len(sampled_loop_traj)])
        close_utc_timestamp = np.empty([len(sampled_loop_traj)])

        for ind, close_loop_pose in enumerate(sampled_loop_traj):
            transform_world_cam = close_loop_pose.transform_world_device.translation()
            quat_world_device = (
                close_loop_pose.transform_world_device.rotation().to_quat()
            )
            T_world_imus[ind] = close_loop_pose.transform_world_device.to_matrix()
            sampled_close_loop_traj_pose[ind, :3] = transform_world_cam
            sampled_close_loop_traj_pose[ind, 3:] = quat_world_device

            timestamp_tracking_s = close_loop_pose.tracking_timestamp.total_seconds()
            utc_timestamp = close_loop_pose.utc_timestamp.total_seconds()
            close_timestamp_tracking_s[ind] = timestamp_tracking_s
            close_utc_timestamp[ind] = utc_timestamp

            # logger.info(f"Timestamp tracking: {timestamp_tracking_s}, UTC timestamp: {utc_timestamp}")

        return (
            sampled_close_loop_traj_pose,
            T_world_imus,
            close_timestamp_tracking_s,
            close_utc_timestamp,
        )

    def sample_open_loop_traj_from_timestamps(
        self, timestamps: NDArray, subsample_rate: int = 1
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """Read open loop traj.

        Parameters
        ------
            timestamps : np.array of shape (N,)
            subsample_rate (int, optional): Defaults to 100.

        Returns
        -----
            sampled_open_loop_traj_pose : numpy.ndarray of shape (N, 7)
                representing the 7-dim pose
            T_world_imus : numpy.ndarray of shape (N, 4, 4)
                representing T_world_imu
            open_timestamp_tracking_s : numpy.ndarray of shape (N,)
                the DEVICE_TIME timestamp in seconds
            open_utc_timestamp : numpy.ndarray of shape (N,)
                the UTC timestamp in seconds
        """
        if not isinstance(self.open_loop_traj_path, str) or not osp.exists(
            self.open_loop_traj_path
        ):
            logger.warning(
                f"open loop trajectory file not found at {self.open_loop_traj_path}"
            )
            return None, None, None, None

        opend_loop_traj = mps.read_open_loop_trajectory(self.open_loop_traj_path)
        sampled_loop_traj = []
        for sample_idx, sample_timestamp in enumerate(timestamps):
            sampled_loop_traj.append(
                get_nearest_pose(opend_loop_traj, sample_timestamp)
            )

        sampled_loop_traj = sampled_loop_traj[::subsample_rate]

        sampled_open_loop_traj_pose = np.empty([len(sampled_loop_traj), 7])
        T_world_imus = np.tile(
            np.eye(4)[np.newaxis, :, :], (len(sampled_loop_traj), 1, 1)
        )
        open_timestamp_tracking_s = np.empty([len(sampled_loop_traj)])
        open_utc_timestamp = np.empty([len(sampled_loop_traj)])

        for ind, open_loop_pose in enumerate(sampled_loop_traj):
            transform_world_cam = open_loop_pose.transform_odometry_device.translation()
            quat_world_device = (
                open_loop_pose.transform_odometry_device.rotation().to_quat()
            )
            T_world_imus[ind] = open_loop_pose.transform_odometry_device.to_matrix()
            sampled_open_loop_traj_pose[ind, :3] = transform_world_cam
            sampled_open_loop_traj_pose[ind, 3:] = quat_world_device

            timestamp_tracking_s = open_loop_pose.tracking_timestamp.total_seconds()
            utc_timestamp = open_loop_pose.utc_timestamp.total_seconds()
            open_timestamp_tracking_s[ind] = timestamp_tracking_s
            open_utc_timestamp[ind] = utc_timestamp

            # logger.info(f"Timestamp tracking: {timestamp_tracking_s}, UTC timestamp: {utc_timestamp}")

        return (
            sampled_open_loop_traj_pose,
            T_world_imus,
            open_timestamp_tracking_s,
            open_utc_timestamp,
        )

    def rotate_upright_image_and_calib(self, take):
        """_summary_

        Args:
            take (_type_): _description_

        Returns:
            _type_: _description_
        """
        # upright_image, upright_calibration = rotate_upright_image_and_calibratio(original_rgb_image, rgb_camera_calibration)

        # Calibration for upright RGB
        # The output RGB frames in the MP4 file are rotated upright (clockwise 90 degrees). The camera calibration will be changed after such rotation and can be obtained using the following interface.
        #
        return 1, 2, 3
        pass

    def read_open_loop_traj(
        self, skip: int = 100
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """Read open loop traj.

        Open loop trajectory is the high frequency (IMU rate, which is 1kHz) odometry estimation output by the visual-inertial odometry (VIO), in an arbitrary odometry coordinate frame. The estimation includes pose and dynamics (translational and angular velocities).
        The open loop trajectory has good “relative” and “local” accuracy: the relative transformation between two poses is accurate when the time span between two frames is short (within a few minutes). However, the open loop trajectory has increased drift error accumulated over time spent and travel distance. Consider using closed loop trajectory if you are looking for trajectory without drift error.

        Parameters
        ------
            skip (int, optional): Defaults to 100.

        Returns
        -----
            mps_open_traj_subset : numpy.ndarray of shape (N, 7)
                representing the 7-dim pose
            T_world_imus : numpy.ndarray of shape (N, 4, 4)
                representing T_world_imu
            open_timestamp_tracking_s : numpy.ndarray of shape (N,)
                the DEVICE_TIME timestamp in seconds
            open_utc_timestamp : numpy.ndarray of shape (N,)
                the UTC timestamp in seconds
        """
        if not isinstance(self.open_loop_traj_path, str) or not osp.exists(
            self.open_loop_traj_path
        ):
            logger.warning(
                f"open loop trajectory file not found at {self.open_loop_traj_path}"
            )
            return None, None, None, None

        open_loop_traj = mps.read_open_loop_trajectory(self.open_loop_traj_path)

        open_traj = np.empty([len(open_loop_traj), 7])
        T_world_imus = np.tile(np.eye(4)[np.newaxis, :, :], (len(open_loop_traj), 1, 1))
        open_timestamp_tracking_s = np.empty([len(open_loop_traj)])
        open_utc_timestamp = np.empty([len(open_loop_traj)])

        for ind, open_loop_pose in enumerate(open_loop_traj):
            transform_world_cam = open_loop_pose.transform_odometry_device.translation()
            quat_world_device = (
                open_loop_pose.transform_odometry_device.rotation().to_quat()
            )
            T_world_imus[ind] = open_loop_pose.transform_odometry_device.to_matrix()
            open_traj[ind, :3] = transform_world_cam
            open_traj[ind, 3:] = quat_world_device

            timestamp_tracking_s = open_loop_pose.tracking_timestamp.total_seconds()
            utc_timestamp = open_loop_pose.utc_timestamp.total_seconds()
            open_timestamp_tracking_s[ind] = timestamp_tracking_s
            open_utc_timestamp[ind] = utc_timestamp

            # logger.info(f"Timestamp tracking: {timestamp_tracking_s}, UTC timestamp: {utc_timestamp}")

        mps_open_traj_subset = open_traj[::skip]
        open_timestamp_tracking_s = open_timestamp_tracking_s[::skip]
        open_utc_timestamp = open_utc_timestamp[::skip]
        T_world_imus = T_world_imus[::skip]

        return (
            mps_open_traj_subset,
            T_world_imus,
            open_timestamp_tracking_s,
            open_utc_timestamp,
        )

    def read_close_loop_traj(
        self, skip: int = 100
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """Read close loop traj from file.
        Closed loop trajectory is the high frequency (IMU rate, which is 1kHz) pose estimation output by our mapping process, in an arbitrary gravity aligned world coordinate frame. The estimation includes pose and dynamics (translational and angular velocities).

        Closed loop trajectories are fully bundle adjusted with detected loop closures, reducing the VIO drift which is present in the open loop trajectories. However, due to the loop closure correction, the “relative” and “local” trajectory accuracy within a short time span (i.e. seconds) might be worse compared to open loop trajectories.

        In some open datasets we also share and use this format for trajectory pose ground truth from simulation or Optitrack, and the files will be called in a different file name aria_gt_trajectory.csv. as `closed_loop_trajectory.csv`

        Parameters
        ------
            skip : int, default 100.

        Returns
        -----
            mps_close_traj_subset : numpy.ndarray of shape (N, 7)
                representing the 7-dim pose
            T_world_imus : numpy.ndarray of shape (N, 4, 4)
                representing T_world_imu
            close_timestamp_tracking_s : numpy.ndarray of shape (N,)
                the DEVICE_TIME timestamp in seconds
            close_utc_timestamp : numpy.ndarray of shape (N,)
                the UTC timestamp in seconds
        """
        if not isinstance(self.closed_loop_traj_path, str) or not osp.exists(
            self.closed_loop_traj_path
        ):
            logger.warning(
                f"Closed loop trajectory file not found at {self.closed_loop_traj_path}"
            )
            return None, None, None, None

        closed_loop_traj = mps.read_closed_loop_trajectory(self.closed_loop_traj_path)

        # example: get transformation from this device to a world coordinate frame
        close_traj = np.empty([len(closed_loop_traj), 7])
        T_world_imus = np.tile(np.eye(4)[np.newaxis, :, :], (len(close_traj), 1, 1))
        close_timestamp_tracking_s = np.empty([len(close_traj)])
        close_utc_timestamp = np.empty([len(close_traj)])

        for ind, closed_loop_pose in enumerate(closed_loop_traj):
            transform_world_cam = closed_loop_pose.transform_world_device.translation()
            quat_world_device = (
                closed_loop_pose.transform_world_device.rotation().to_quat()
            )
            T_world_imus[ind] = closed_loop_pose.transform_world_device.to_matrix()
            close_traj[ind, :3] = transform_world_cam
            close_traj[ind, 3:] = quat_world_device

            timestamp_tracking_s = closed_loop_pose.tracking_timestamp.total_seconds()
            utc_timestamp = closed_loop_pose.utc_timestamp.total_seconds()
            close_timestamp_tracking_s[ind] = timestamp_tracking_s
            close_utc_timestamp[ind] = utc_timestamp
            # logger.info(f"Timestamp tracking: {timestamp_tracking_s}, UTC timestamp: {utc_timestamp}")

        mps_close_traj_subset = close_traj[::skip]
        close_timestamp_tracking_s = close_timestamp_tracking_s[::skip]
        close_utc_timestamp = close_utc_timestamp[::skip]
        T_world_imus = T_world_imus[::skip]

        return (
            mps_close_traj_subset,
            T_world_imus,
            close_timestamp_tracking_s,
            close_utc_timestamp,
        )

    def get_exo_cams_calibs_w_cam_json(
        self, take: dict
    ) -> Tuple[List[str], Dict[str, NDArray], Dict[str, NDArray]]:
        """Get exo_cam_calibs from `camera.json`.

        Parameters
        ------
            take: loaded json object from `take.json`.

        Returns
        -----
            exo_cam_names: list(str)
                list of exo camera names.
            exo_cam_calibs: dict[numpy.ndarray of shape (4, 4)]
                each key contains the **string repr** of exo-cam-name, with the corresponding value being T_world_device
            exo_cam_calibs_pose: dict[numpy.ndarray of shape (1, 7)]
                same with exo_cam_calibs, but the value is 7-dim pose
        """
        if not isinstance(self.cam_pose_anno_path, str) or not osp.exists(
            self.cam_pose_anno_path
        ):
            logger.warning(
                f"Camera pose annotation file not found at {self.cam_pose_anno_path}"
            )
            return None, None, None

        exo_cam_names = EgoExoUtils.get_exo_cam_names(take)
        cam_pose_anno = json.load(open(self.cam_pose_anno_path))
        exo_cams_calibs = {}
        exo_cams_calibs_pose = {}

        for ind, exo_cam_name in enumerate(exo_cam_names):
            if exo_cam_name not in cam_pose_anno.keys():
                continue
            T_exo_cam_world = np.eye(4)
            T_exo_cam_world[:3] = cam_pose_anno[exo_cam_name]["camera_extrinsics"]
            T_exo_world_cam = np.linalg.inv(T_exo_cam_world)
            exo_cams_calibs[exo_cam_name] = T_exo_world_cam
            tmp_pose = np.empty((1, 7))
            tmp_pose[:, :3] = T_exo_world_cam[:3, 3]
            tmp_pose[:, 3:] = (
                transforms.matrix_to_quaternion(
                    torch.from_numpy(T_exo_world_cam[:3, :3])
                )
                .cpu()
                .numpy()
            )
            exo_cams_calibs_pose[exo_cam_name] = tmp_pose
        return exo_cam_names, exo_cams_calibs, exo_cams_calibs_pose

    def get_ego_cam_calibs(
        self, take: dict
    ) -> Tuple[str, Dict[str, NDArray], Dict[str, NDArray]]:
        """Get calibration results of ego cameras from `camera.json`.

        Parameters
        ----------
            take: loaded json object from `take.json`.

        Returns
        -------
        ego_cam_name : str
            The name of the ego camera.
        T_ego_world_cams: dict[numpy.ndarray of shape (4, 4)]
            Each key is a **string** repr of frame number, with the corresponding value being `T_ego_world_cam` (4x4 matrix).
        T_ego_world_cam_poses: dictt[numpy.ndarray of shape (1, 7)]
            Each key is a **string** repr of frame number, with the corresponding value being `T_ego_world_cam_pose` (7-dim pose).
        """

        if not isinstance(self.cam_pose_anno_path, str) or not osp.exists(
            self.cam_pose_anno_path
        ):
            logger.warning(
                f"Camera pose annotation file not found at {self.cam_pose_anno_path}"
            )
            return None, None, None
        ego_cam_name = EgoExoUtils.get_ego_aria_cam_name(take)
        cam_pose_anno = json.load(open(self.cam_pose_anno_path))
        T_ego_world_cams = {}
        T_ego_world_cam_poses = {}
        frames = find_numerical_key_in_dict(
            cam_pose_anno[ego_cam_name]["camera_extrinsics"]
        )

        for frame_ind, frame_num in enumerate(frames):
            T_ego_cam_world = np.eye(4)
            T_ego_cam_world[:3] = cam_pose_anno[ego_cam_name]["camera_extrinsics"][
                str(frame_num)
            ]
            T_ego_world_cam = np.linalg.inv(T_ego_cam_world)
            T_ego_world_cams[str(frame_num)] = T_ego_world_cam
            tmp_pose = np.empty((1, 7))
            tmp_pose[:, :3] = T_ego_world_cam[:3, 3]
            tmp_pose[:, 3:] = (
                transforms.matrix_to_quaternion(
                    torch.from_numpy(T_ego_world_cam[:3, :3])
                )
                .cpu()
                .numpy()
            )
            T_ego_world_cam_poses[str(frame_num)] = tmp_pose

        return ego_cam_name, T_ego_world_cams, T_ego_world_cam_poses

    def get_exo_cams_calibs_w_goprocalibs(self, take):
        """Loading Exo static camera calibration data. From `goprocalibs.csv`.

        Parameters
        ------
            take: loaded json object from `take.json`.

        Returns
        ------
            exo_cam_names: list
                list of exocentric camera names.
            exo_cam_calibs: dict[np.array of shape (4, 4)]
                each key contains the exo-cam-name, with the corresponding value being T_world_device.
            exo_cam_calibs_pose: dict[np.array of shape (1, 7)]
                same with exo_cam_calibs, but the value is 7-dim pose
        """
        exo_cam_names = EgoExoUtils.get_exo_cam_names(take)
        exo_cams_calibs = {}
        exo_cams_calibs_pose = {}
        if not isinstance(self.goprocalibs_path, str) or not osp.exists(
            self.goprocalibs_path
        ):
            logger.warning(
                f"GoPro calibration file not found at {self.goprocalibs_path}"
            )
            return None, None, None
        static_calibrations = mps.read_static_camera_calibrations(self.goprocalibs_path)

        for ind, (static_calibration, exo_cam_name) in enumerate(
            zip(static_calibrations, exo_cam_names)
        ):
            T_world_device = static_calibration.transform_world_cam.to_matrix()
            trans_world_device = static_calibration.transform_world_cam.translation()
            quat_world_device = (
                static_calibration.transform_world_cam.rotation().to_quat()
            )
            exo_cams_calibs[exo_cam_name] = T_world_device
            tmp_pose = np.empty((1, 7))
            tmp_pose[:, :3] = trans_world_device
            tmp_pose[:, 3:] = quat_world_device
            exo_cams_calibs_pose[exo_cam_name] = tmp_pose

        return exo_cam_names, exo_cams_calibs, exo_cams_calibs_pose

    def plot_3d(
        self,
        T_world_imus_w_open=None,
        T_world_imus_w_close=None,
        tracking_timestamps=None,
        valid_timestamps: NDArray = None,
        offset_z: NDArray = None,
        point_cloud=None,
        load_exo_cams=False,
        load_ego_cam_traj=False,
        **kwargs,
    ):
        """Plot elements.

        Parameters
        ----------
        Ts_world_cams : list of np.ndarray (T, 3, 4)
        """
        if load_exo_cams:
            (
                exo_cam_names,
                T_exo_world_cam_poses_w_cam_anno,
                T_exo_world_cam_qpose,
            ) = self.get_exo_cams_calibs_w_cam_json(self.take)
            T_exo_world_cam_poses_w_cam_anno = {
                k: v[:3, :] for k, v in T_exo_world_cam_poses_w_cam_anno.items()
            }
        else:
            T_exo_world_cam_poses_w_cam_anno = None

        if load_exo_cams:
            (
                exo_cam_names_,
                T_exo_world_cam_w_gopro_calib,
                T_exo_world_cam_qpose,
            ) = self.get_exo_cams_calibs_w_goprocalibs(self.take)
            assert exo_cam_names_ == exo_cam_names, "Exo cam names do not match."
            T_exo_world_cam_w_gopro_calib = {
                k: v[:3, :] for k, v in T_exo_world_cam_w_gopro_calib.items()
            }
        else:
            T_exo_world_cam_w_gopro_calib = None

        if load_ego_cam_traj:
            (
                ego_cam_name,
                T_ego_world_cams,
                T_ego_world_cam_poses,
            ) = self.get_ego_cam_calibs(self.take)
            T_ego_world_cams = {
                k: v[:3, :]
                for k, v in T_ego_world_cams.items()
                if int(k) in valid_timestamps.tolist()
            }
            T_ego = len(T_ego_world_cams)
            if T_world_imus_w_close is not None and T_world_imus_w_open is not None:
                T_open, *_ = T_world_imus_w_open.shape
                T_close, *_ = T_world_imus_w_close.shape
                assert T_ego == T_open == T_close, (
                    f"Open and close trajectory lengths do not match. Open: {T_open}, Close: {T_close}, Ego: {T_ego}"
                )
        else:
            T_ego_world_cams = None

        self._plot_polyscope(
            T_exo_world_cam_poses_w_cam_anno=T_exo_world_cam_poses_w_cam_anno,
            T_exo_world_cam_w_gopro_calib=T_exo_world_cam_w_gopro_calib,
            T_ego_world_cams=T_ego_world_cams,
            T_world_imus_w_open=T_world_imus_w_open,
            T_world_imus_w_close=T_world_imus_w_close,
            tracking_timestamps=tracking_timestamps,
            offset_z=offset_z,
            point_clouds=point_cloud,
            downsample_point_cloud=False,  # downsample_point_cloud
            **kwargs,
        )

    def read_online_calibration_json(self):
        """
        read online calibration json file.
        """
        # region
        if not osp.exists(self.online_calib_json_path):
            raise ValueError(
                f"Online calibration json file not found at {self.online_calib_json_path}"
            )
        online_calibs = mps.read_online_calibration(self.online_calib_json_path)

        for calib in online_calibs:
            # example: get left IMU's online calibration
            for imuCalib in calib.imu_calibs:
                if imuCalib.get_label() == "imu-left":
                    pass
            # example: get left SLAM camera's online calibration
            for camCalib in calib.camera_calibs:
                if camCalib.get_label() == "camera-slam-left":
                    pass

    # endregion

    def read_semidense_point_cloud(self):
        if not osp.exists(self.point_observation_path):
            raise ValueError(
                f"Point observation file not found at {self.point_observation_path}"
            )
        if self.mps_data_provider is None:
            self.mps_data_paths_provider = mps.MpsDataPathsProvider(self.take_root_path)
            self.mps_data_paths = self.mps_data_paths_provider.get_data_paths()
            self.mps_data_provider = mps.MpsDataProvider(self.mps_data_paths)
        semidense_observations = self.mps_data_provider.get_semidense_point_cloud()
        return semidense_observations

    def _plot_polyscope(
        self,
        T_exo_world_cam_poses_w_anno=None,
        T_exo_world_cam_w_gopro_calib=None,
        T_ego_world_cams=None,
        T_world_imus_w_open=None,
        T_world_imus_w_close=None,
        tracking_timestamps=None,
        offset_z=None,
        point_clouds=None,
        downsample_point_cloud=False,
        **kwargs,
    ):
        assert tracking_timestamps is not None, "Tracking timestamps must be provided."

        T = len(tracking_timestamps)

        mean_offset_z = np.mean(offset_z)
        gen_vis_save_root_dir = kwargs.get("gen_vis_save_root_dir", None)
        out_vid_dir = kwargs.get("out_vid_dir", None)
        use_ffmpeg = kwargs.get("use_ffmpeg", True)
        render_to_screen = kwargs.get("render_to_screen", True)
        save_max_frames = kwargs.get("save_max_frames", 1000)

        ps.look_at_dir(
            camera_location=(0.0, -5.0, 2.5),
            target=(0.0, -1.5, -0.5),
            up_dir=(0.0, 0.0, 1.0),
            fly_to=False,
        )

        # Create a camera view from parameters
        intrinsics = ps.CameraIntrinsics(fov_vertical_deg=60, aspect=2)
        extrinsics = ps.CameraExtrinsics(
            root=(0.0, -2.0, 1.0), look_dir=(0.0, 0.0, 0.0), up_dir=(0.0, 0.0, 1.0)
        )
        params = ps.CameraParameters(intrinsics, extrinsics)
        cam = ps.register_camera_view("cam", params)
        # Set some options for the camera view
        # these can also be set as keyword args in register_camera_view()
        cam.set_widget_focal_length(0.07)  # size of displayed widget (relative value)
        cam.set_widget_thickness(0.005)  # thickness of widget lines
        cam.set_widget_color((0.25, 0.25, 0.25))  # color of widget lines

        # Add an image to be displayed in the camera frame
        H, W = 1080, 1920
        cam.add_scalar_image_quantity(
            "scalar_img", np.zeros((H, W)), enabled=True, show_in_camera_billboard=True
        )

        zero_trans = np.eye(4)
        draw_camera_pose_ps(T=zero_trans[:3])

        # Loading static scenes
        if (
            T_exo_world_cam_poses_w_anno is not None
            and CFG.plotly.camera_frustum.enable
        ):
            for exo_cam_name in T_exo_world_cam_poses_w_anno.keys():
                T_exo_world_cam_pose_w_anno = T_exo_world_cam_poses_w_anno[
                    exo_cam_name
                ].copy()
                T_exo_world_cam_pose_w_anno[2, 3] += mean_offset_z
                build_cam_frustum_w_extr_ps(
                    T_exo_world_cam_pose_w_anno, surface_mesh_id="exo_world_cam"
                )

        if (
            T_exo_world_cam_w_gopro_calib is not None
            and CFG.plotly.camera_frustum.enable
        ):
            for exo_cam_name in T_exo_world_cam_w_gopro_calib.keys():
                T_tmp = T_exo_world_cam_w_gopro_calib[exo_cam_name].copy()
                T_tmp[2, 3] += mean_offset_z
                build_cam_frustum_w_extr_ps(
                    T_tmp, surface_mesh_id="exo_world_cam_gopro"
                )

        if point_clouds is not None and CFG.plotly.semidense_observations.enable:
            threshold_invdep = 5e-4
            threshold_dep = 5e-4
            point_clouds = filter_points_from_confidence(
                point_clouds, threshold_invdep, threshold_dep
            )
            if downsample_point_cloud:
                point_clouds = filter_points_from_count(point_clouds, 500_000)
            point_clouds = np.stack([it.position_world for it in point_clouds])
            tmp_pts = point_clouds.copy()
            tmp_pts[:, 2] += mean_offset_z
            cloud = ps.register_point_cloud("ptc", tmp_pts)
            cloud.set_radius(
                0.0009
            )  # radius is relative to a scene length scale by default
            cloud.set_material("candy")
            cloud.set_transparency(0.95)

        if (
            T_world_imus_w_open is not None
            and T_world_imus_w_close is not None
            and CFG.plotly.closed_slam_traj.enable
            and CFG.plotly.open_slam_traj.enable
        ):
            T_open, *_ = T_world_imus_w_open.shape
            T_close, *_ = T_world_imus_w_close.shape
            assert T_open == T_close == T, (
                f"Open and close trajectory lengths do not match. Open: {T_open}, Close: {T_close}, T: {T}"
            )

        coco_world_3d_kpts_key = "coco_world_3d_pts"
        coco_world_3d_pts = kwargs.get(coco_world_3d_kpts_key, None)
        if coco_world_3d_pts is not None and CFG.plotly.coco_kinematic_tree.enable:
            assert T == len(coco_world_3d_pts), (
                f"Tracking timestamps and coco world 3d points do not match. Tracking: {T}, Coco: {len(coco_world_3d_pts)}"
            )
        model_pred_coco_kpts = "model_pred_coco_kpts"
        model_pred_coco_kpts = kwargs.get(model_pred_coco_kpts, None)
        if model_pred_coco_kpts is not None and CFG.plotly.model_pred_coco_kpts.enable:
            assert T == len(model_pred_coco_kpts), (
                f"Tracking timestamps and coco world 3d points do not match. Tracking: {T}, Coco: {len(model_pred_coco_kpts)}"
            )
        model_gt_coco_kpts = "model_gt_coco_kpts"
        model_gt_coco_kpts = kwargs.get(model_gt_coco_kpts, None)
        if model_gt_coco_kpts is not None and CFG.plotly.model_gt_coco_kpts.enable:
            assert T == len(model_gt_coco_kpts), (
                f"Tracking timestamps and coco world 3d points do not match. Tracking: {T}, Coco: {len(model_gt_coco_kpts)}"
            )

        # Loading dynamic scenes
        for render_ind, tracking_timestamp in tqdm(
            enumerate(tracking_timestamps),
            total=min(T, save_max_frames),
            desc="enumerating polyscope routine rendering",
            ascii=" >=",
        ):
            if render_to_screen and ps.window_requests_close():
                break

            if render_ind >= save_max_frames:
                break

            # Build ego camera trajectories
            if (
                T_ego_world_cams is not None
                and CFG.plotly.camera_3d_coordinate.enable
                and CFG.plotly.ego_cam_traj.enable
            ):
                frame_num = list(T_ego_world_cams.keys())[render_ind]
                T_ego_world_cam = T_ego_world_cams[frame_num]
                ego_world_cam_qpose = T_to_qpose(T_ego_world_cam[None])
                opengl_cam_pose = aria_camera_device2opengl_pose(ego_world_cam_qpose)
                T_ego_world_cam = qpose_to_T(opengl_cam_pose)[0].copy()
                T_ego_world_cam[2, 3] += offset_z[render_ind]
                build_cam_frustum_w_extr_ps(
                    T_ego_world_cam, surface_mesh_id="ego_world_cam"
                )

            # Build IMU trajectories
            if (
                T_world_imus_w_open is not None
                and T_world_imus_w_close is not None
                and CFG.plotly.closed_slam_traj.enable
                and CFG.plotly.open_slam_traj.enable
            ):
                T_world_imu_w_close_traj = T_world_imus_w_close[render_ind].copy()
                T_world_imu_w_close_traj[2, 3] += offset_z[render_ind]
                build_cam_frustum_w_extr_ps(
                    T_world_imu_w_close_traj, surface_mesh_id="imu_close_traj"
                )
                T_world_imu_w_open_traj = T_world_imus_w_open[render_ind].copy()
                T_world_imu_w_open_traj[2, 3] += offset_z[render_ind]
                build_cam_frustum_w_extr_ps(
                    T_world_imu_w_open_traj, surface_mesh_id="imu_close_traj"
                )

            # Handle additional kwargs for plotting
            Ts_world_cams = kwargs.get("Ts_world_cams", None)
            if Ts_world_cams is not None and CFG.plotly.camera_frustum.enable:
                for render_ind, Ts_world_cam in enumerate(Ts_world_cams):
                    T, *_ = Ts_world_cam.shape
                    assert T == T_open, (
                        f"IMU and cam trajectory lengths do not match. IMU: {T_open}, Cam: {T}"
                    )

                    T_tmp = Ts_world_cam[render_ind].copy()
                    build_cam_frustum_w_extr_ps(T_tmp, surface_mesh_id="Ts_woldr_cam")

            if coco_world_3d_pts is not None and CFG.plotly.coco_kinematic_tree.enable:
                tmp_3d_pt = coco_world_3d_pts[render_ind][
                    :NUM_EGOEXO4D_EGOPOSE_JNTS
                ].copy()
                # tmp_3d_pt[:,2] += offset_z[ind]
                draw_coco_kinematic_tree_ps(
                    tmp_3d_pt,
                    coco_cfg=CFG.plotly.coco_kinematic_tree,
                    curve_network_id="coco_world_3d_pts",
                )

            if (
                model_pred_coco_kpts is not None
                and CFG.plotly.model_pred_coco_kpts.enable
            ):
                tmp_3d_pt = model_pred_coco_kpts[render_ind][
                    :NUM_EGOEXO4D_EGOPOSE_JNTS
                ].copy()
                # tmp_3d_pt[:,2] += offset_z[ind]
                draw_coco_kinematic_tree_ps(
                    tmp_3d_pt,
                    coco_cfg=CFG.plotly.model_pred_coco_kpts,
                    curve_network_id="model_pred_coco_world_3d_pts",
                )

            if model_gt_coco_kpts is not None and CFG.plotly.model_gt_coco_kpts.enable:
                tmp_3d_pt = model_gt_coco_kpts[render_ind][
                    :NUM_EGOEXO4D_EGOPOSE_JNTS
                ].copy()
                # tmp_3d_pt[:,2] += offset_z[ind]
                draw_coco_kinematic_tree_ps(
                    tmp_3d_pt,
                    coco_cfg=CFG.plotly.model_gt_coco_kpts,
                    curve_network_id="model_gt_coco_world_3d_pts",
                )

            # import ipdb; ipdb.set_trace()
            if render_to_screen:
                ps.frame_tick()

            if gen_vis_save_root_dir is not None:
                gen_vis_save_frame_path = osp.join(
                    gen_vis_save_root_dir, "{:06d}.png".format(render_ind)
                )
                ps.screenshot(gen_vis_save_frame_path)

        # ps.shutdown()
        ps.remove_all_groups()
        ps.remove_all_structures()

        if out_vid_dir is not None and gen_vis_save_root_dir is not None:
            if use_ffmpeg:
                images_to_video(
                    img_folder=gen_vis_save_root_dir,
                    output_vid_file=osp.join(out_vid_dir, "output.mp4"),
                )
            else:
                images_to_video_w_imageio(
                    img_folder=gen_vis_save_root_dir,
                    output_vid_file=osp.join(out_vid_dir, "output.mp4"),
                )

    # region
    def _plot_open3d(
        self,
        T_exo_world_cam_poses_w_anno: Optional[Dict[str, NDArray]] = None,
        T_exo_world_cam_w_gopro_calib: Optional[Dict[str, NDArray]] = None,
        T_ego_world_cams: Optional[Dict[str, NDArray]] = None,
        T_world_imus_w_open: Optional[Dict[str, NDArray]] = None,
        T_world_imus_w_close: Optional[Dict[str, NDArray]] = None,
        tracking_timestamps: Optional[Dict[str, NDArray]] = None,
        offset_z: NDArray = None,
        point_clouds: Optional[Dict[str, NDArray]] = None,
        downsample_point_cloud: bool = False,
        **kwargs,
    ):
        """
        Plot elements.

        Parameters
        ----------
        T_exo_world_cam_poses_w_anno  : dict of np.ndarray (3, 4)
        T_exo_world_cam_poses : dict of np.ndarray (3, 4)
        T_ego_world_cams : dict of np.ndarray (3, 4)
        T_world_imus_w_open_traj : np.array of shape (T, 3, 4)
        T_world_imus_w_close_traj : np.array of shape (T, 3, 4)
        tracking_timestamps : np.array of shape (T,)
        cam_trajs : list of np.ndarray (T, 3, 4)

        Notes
        -----
        The current timestamp steps is set to come from close_loop_traj.
        """
        assert tracking_timestamps is not None, "Tracking timestamps must be provided."

        static_traces = []
        T = len(tracking_timestamps)

        def _():
            return [None] * T

        dynamic_traces = defaultdict(_)
        mean_offset_z = np.mean(offset_z)

        if (
            T_exo_world_cam_poses_w_anno is not None
            and CFG.plotly.camera_frustum.enable
        ):
            exo_cam_frustums = [None] * len(T_exo_world_cam_poses_w_anno)
            for i, exo_cam_name in enumerate(T_exo_world_cam_poses_w_anno.keys()):
                T_exo_world_cam_pose_w_anno = T_exo_world_cam_poses_w_anno[
                    exo_cam_name
                ].copy()
                T_exo_world_cam_pose_w_anno[2, 3] += mean_offset_z
                exo_cam_frustums[i] = build_cam_frustum_w_extr_open3d(
                    T_exo_world_cam_pose_w_anno
                )
            static_traces.extend(exo_cam_frustums)

        if (
            T_exo_world_cam_w_gopro_calib is not None
            and CFG.plotly.camera_frustum.enable
        ):
            gopro_cams_frustums = [None] * len(T_exo_world_cam_w_gopro_calib)
            for i, exo_cam_name in enumerate(T_exo_world_cam_w_gopro_calib.keys()):
                T_tmp = T_exo_world_cam_w_gopro_calib[exo_cam_name].copy()
                T_tmp[2, 3] += mean_offset_z
                gopro_cams_frustums[i] = build_cam_frustum_w_extr_open3d(T_tmp)
            static_traces.extend(gopro_cams_frustums)

        if (
            T_ego_world_cams is not None
            and CFG.plotly.camera_3d_coordinate.enable
            and CFG.plotly.ego_cam_traj.enable
        ):
            T_ego = len(T_ego_world_cams)
            assert T_ego == T, (
                f"Tracking timestamps and T_ego_world_cams do not match. Tracking: {T}, Ego: {T_ego}"
            )

            def _():
                return [None] * T_ego

            dynamic_traces = defaultdict(_)
            for i, frame_num in enumerate(T_ego_world_cams.keys()):
                T_ego_world_cam = T_ego_world_cams[frame_num]
                ego_world_cam_qpose = T_to_qpose(T_ego_world_cam[None])
                opengl_cam_pose = aria_camera_device2opengl_pose(ego_world_cam_qpose)
                T_ego_world_cam = qpose_to_T(opengl_cam_pose)[0].copy()
                T_ego_world_cam[2, 3] += offset_z[i]

                (
                    dynamic_traces["ego_world_cam_x"][i],
                    dynamic_traces["ego_world_cam_y"][i],
                    dynamic_traces["ego_world_cam_z"][i],
                ) = draw_camera_pose(T_ego_world_cam)

            T_ego_world_cams = np.stack(list(T_ego_world_cams.values()), axis=0).copy()
            T_ego_world_cams[:, 2, 3] += offset_z
            logger.info(
                f"ego_cam_traj | x: {np.mean(T_ego_world_cams[:, 0, 3])}, y: {np.mean(T_ego_world_cams[:, 1, 3])}, z: {np.mean(T_ego_world_cams[:, 2, 3])}"
            )
            ego_cam_traj = go.Scatter3d(
                x=T_ego_world_cams[:, 0, 3],
                y=T_ego_world_cams[:, 1, 3],
                z=T_ego_world_cams[:, 2, 3],
                mode="markers",
                marker={
                    "size": CFG.plotly.ego_cam_traj.marker_size,
                    "opacity": 0.8,
                    "color": CFG.plotly.ego_cam_traj.color,
                },
                name="T_world_cam_traj",
                hoverinfo="none",
                visible=False,
            )

            static_traces.append(ego_cam_traj)

        if (
            T_world_imus_w_close is not None
            and T_world_imus_w_open is not None
            and CFG.plotly.closed_slam_traj.enable
            and CFG.plotly.open_slam_traj.enable
        ):
            T_open, *_ = T_world_imus_w_open.shape
            T_close, *_ = T_world_imus_w_close.shape
            assert T_open == T_close == T, (
                f"Open and close trajectory lengths do not match. Open: {T_open}, Close: {T_close}, T: {T}"
            )

            close_traj_cam_frustums = [None] * T_close

            for i in range(len(T_world_imus_w_close)):
                T_world_imu_w_close_traj = T_world_imus_w_close[i].copy()
                T_world_imu_w_close_traj[2, 3] += offset_z[i]
                close_traj_cam_frustums[i] = build_cam_frustum_w_extr_open3d(
                    T_world_imu_w_close_traj
                )
            dynamic_traces["close_traj_cam_frustums"] = close_traj_cam_frustums

            open_traj_cam_frustums = [None] * T_open
            for i in range(len(T_world_imus_w_open)):
                T_world_imu_w_open_traj = T_world_imus_w_open[i].copy()
                T_world_imu_w_open_traj[2, 3] += offset_z[i]
                open_traj_cam_frustums[i] = build_cam_frustum_w_extr_open3d(
                    T_world_imu_w_open_traj
                )
            dynamic_traces["open_traj_cam_frustums"] = open_traj_cam_frustums

        # plot Ts_world_cams
        Ts_world_cams = kwargs["Ts_world_cams"] if "Ts_world_cams" in kwargs else None

        world_cam_trajs = []

        def _():
            return [None] * T_ego

        if Ts_world_cams is not None and CFG.plotly.camera_frustum.enable:
            for ind, Ts_world_cam in enumerate(Ts_world_cams):
                T, *_ = Ts_world_cam.shape
                assert T == T_open, (
                    f"IMU and cam trajectory lengths do not match. IMU: {T_open}, Cam: {T}"
                )

                for i in range(T):
                    T_tmp = Ts_world_cam[i].copy()
                    # T_tmp[2, 3] += offset_z[i]
                    dynamic_traces[f"T_world_cam_{ind}_frustums"][i] = (
                        build_cam_frustum_w_extr_open3d(T_tmp)
                    )
                    (
                        dynamic_traces[f"T_world_cam_{ind}_x_frame"][i],
                        dynamic_traces[f"T_world_cam_{ind}_y_frame"][i],
                        dynamic_traces[f"T_world_cam_{ind}_z_frame"][i],
                    ) = draw_camera_pose(T_tmp)

                Ts_world_cam_ = Ts_world_cam.copy()
                # Ts_world_cam_[:, 2, 3] += offset_z
                ego_cam_traj = go.Scatter3d(
                    x=Ts_world_cam_[:, 0, 3],
                    y=Ts_world_cam_[:, 1, 3],
                    z=Ts_world_cam_[:, 2, 3],
                    mode="markers",
                    marker={"size": 2, "opacity": 0.8, "color": "blue"},
                    name="T_world_cam_traj",
                    hoverinfo="none",
                    visible=False,
                )
                logger.info(
                    f"Ts_world_cams/{ind} | x: {np.mean(Ts_world_cam_[:, 0, 3])}, y: {np.mean(Ts_world_cam_[:, 1, 3])}, z: {np.mean(Ts_world_cam_[:, 2, 3])}"
                )
                world_cam_trajs.append(ego_cam_traj)
            static_traces.extend(world_cam_trajs)

        # plot additional coco format world_3d_pts
        coco_world_3d_kpts_key = "coco_world_3d_pts"
        coco_world_3d_pts = (
            kwargs[coco_world_3d_kpts_key] if coco_world_3d_kpts_key in kwargs else None
        )
        if coco_world_3d_pts is not None and CFG.plotly.coco_kinematic_tree.enable:
            assert T == len(coco_world_3d_pts), (
                f"Tracking timestamps and coco world 3d points do not match. Tracking: {T}, Coco: {len(coco_world_3d_pts)}"
            )
            for ind, coco_world_3d_pt in enumerate(coco_world_3d_pts):
                tmp_3d_pt = coco_world_3d_pt[:NUM_EGOEXO4D_EGOPOSE_JNTS].copy()
                # tmp_3d_pt[:,2] += offset_z[ind]
                coco_lines_vis = draw_coco_kinematic_tree_open3d(
                    tmp_3d_pt, coco_cfg=CFG.plotly.coco_kinematic_tree
                )
                coco_lines_vis = {
                    f"coco_world_3d_pts_{i}": coco_lines_vis[i]
                    for i in range(len(coco_lines_vis))
                }
                for key, val in coco_lines_vis.items():
                    dynamic_traces[key][ind] = val

        # plot additional coco format world_3d_pts
        model_pred_coco_kpts = "model_pred_coco_kpts"
        model_pred_coco_kpts = (
            kwargs[model_pred_coco_kpts] if model_pred_coco_kpts in kwargs else None
        )
        if model_pred_coco_kpts is not None and CFG.plotly.model_pred_coco_kpts.enable:
            assert T == len(model_pred_coco_kpts), (
                f"Tracking timestamps and coco world 3d points do not match. Tracking: {T}, Coco: {len(model_pred_coco_kpts)}"
            )
            for ind, coco_world_3d_pt in enumerate(model_pred_coco_kpts):
                tmp_3d_pt = coco_world_3d_pt[:NUM_EGOEXO4D_EGOPOSE_JNTS].copy()
                # tmp_3d_pt[:,2] += offset_z[ind]
                coco_lines_vis = draw_coco_kinematic_tree_open3d(
                    tmp_3d_pt, coco_cfg=CFG.plotly.model_pred_coco_kpts
                )
                coco_lines_vis = {
                    f"model_pred_coco_world_3d_pts_{i}": coco_lines_vis[i]
                    for i in range(len(coco_lines_vis))
                }
                for key, val in coco_lines_vis.items():
                    dynamic_traces[key][ind] = val

        # plot additional coco format world_3d_pts
        model_gt_coco_kpts = "model_gt_coco_kpts"
        model_gt_coco_kpts = (
            kwargs[model_gt_coco_kpts] if model_gt_coco_kpts in kwargs else None
        )
        if model_gt_coco_kpts is not None and CFG.plotly.model_gt_coco_kpts.enable:
            assert T == len(model_gt_coco_kpts), (
                f"Tracking timestamps and coco world 3d points do not match. Tracking: {T}, Coco: {len(model_gt_coco_kpts)}"
            )
            for ind, coco_world_3d_pt in enumerate(model_gt_coco_kpts):
                tmp_3d_pt = coco_world_3d_pt[:NUM_EGOEXO4D_EGOPOSE_JNTS].copy()
                # tmp_3d_pt[:,2] += offset_z[ind]
                coco_lines_vis = draw_coco_kinematic_tree_open3d(
                    tmp_3d_pt, coco_cfg=CFG.plotly.model_gt_coco_kpts
                )
                coco_lines_vis = {
                    f"model_gt_coco_world_3d_pts_{i}": coco_lines_vis[i]
                    for i in range(len(coco_lines_vis))
                }
                for key, val in coco_lines_vis.items():
                    dynamic_traces[key][ind] = val

        # Plot trajectory and point cloud
        # We color the points by their z coordinate
        if T_world_imus_w_close is not None and CFG.plotly.closed_slam_traj.enable:
            tmp_T = T_world_imus_w_close.copy()
            tmp_T[:, 2, 3] += offset_z
            imu_close_traj_points = np.array(
                [tmp_T[:, 0, 3], tmp_T[:, 1, 3], tmp_T[:, 2, 3]]
            ).T
            imu_close_traj_line_set = o3d.geometry.LineSet()
            imu_close_traj_line_set.points = o3d.utility.Vector3dVector(
                imu_close_traj_points
            )
            imu_close_traj_line_set.lines = o3d.utility.Vector2iVector(
                [[i, i + 1] for i in range(len(imu_close_traj_points) - 1)]
            )
            imu_close_traj_line_set.paint_uniform_color(
                CFG.plotly.closed_slam_traj.color
            )
            static_traces.append(imu_close_traj_line_set)
            logger.info(
                f"close_trajectory | x: {np.mean(tmp_T[:, 0, 3])}, y: {np.mean(tmp_T[:, 1, 3])}, z: {np.mean(tmp_T[:, 2, 3])}"
            )

        if T_world_imus_w_open is not None and CFG.plotly.open_slam_traj.enable:
            tmp_T = T_world_imus_w_open.copy()
            tmp_T[:, 2, 3] += offset_z
            imu_open_traj_points = np.array(
                [tmp_T[:, 0, 3], tmp_T[:, 1, 3], tmp_T[:, 2, 3]]
            ).T
            imu_open_traj_line_set = o3d.geometry.LineSet()
            imu_open_traj_line_set.points = o3d.utility.Vector3dVector(
                imu_open_traj_points
            )
            imu_open_traj_line_set.lines = o3d.utility.Vector2iVector(
                [[i, i + 1] for i in range(len(imu_open_traj_points) - 1)]
            )
            imu_open_traj_line_set.paint_uniform_color(CFG.plotly.open_slam_traj.color)
            static_traces.append(imu_open_traj_line_set)
            logger.info(
                f"open_trajectory | x: {np.mean(tmp_T[:, 0, 3])}, y: {np.mean(tmp_T[:, 1, 3])}, z: {np.mean(tmp_T[:, 2, 3])}"
            )

        if point_clouds is not None and CFG.plotly.semidense_observations.enable:
            # Filter the point cloud by inv depth and depth and load
            threshold_invdep = 5e-4
            threshold_dep = 5e-4
            point_clouds = filter_points_from_confidence(
                point_clouds, threshold_invdep, threshold_dep
            )
            if downsample_point_cloud:
                # Downsampling the data for web viewing
                point_clouds = filter_points_from_count(point_clouds, 500_000)
            # Retrieve point position
            point_clouds = np.stack([it.position_world for it in point_clouds])
            tmp_pts = point_clouds.copy()
            tmp_pts[:, 2] += mean_offset_z

            # Create point cloud visualization
            global_points = o3d.geometry.PointCloud()
            global_points.points = o3d.utility.Vector3dVector(tmp_pts)
            colors = (tmp_pts[:, 2] - np.min(tmp_pts[:, 2])) / (
                np.max(tmp_pts[:, 2]) - np.min(tmp_pts[:, 2])
            )  # Normalize z-coordinates for color mapping
            global_points.colors = o3d.utility.Vector3dVector(
                np.column_stack([colors, colors, colors])
            )  # Grayscale based on z
            static_traces.append(global_points)

        # for static_trace in static_traces:
        #     static_trace.visible = True

        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Add static traces
        if static_traces is not None and len(static_traces) > 0:
            for static_trace in static_traces:
                vis.add_geometry(static_trace)

        # Assuming dynamic_traces is a list of lists containing Open3D geometries
        # Create a dictionary to manage visibility based on time
        dynamic_visibility = {i: [] for i in range(T)}

        # Add dynamic traces but keep them invisible initially
        for ele_key, traces in dynamic_traces.items():
            for i, trace in enumerate(traces):
                dynamic_visibility[i].append(trace)
                vis.add_geometry(
                    trace
                )  # Add all geometries to the visualizer initially

        # Function to update visibility based on the current time step
        def update_visibility(current_step):
            for i, traces in dynamic_visibility.items():
                for trace in traces:
                    vis.remove_geometry(trace)  # Remove all geometries first
                    if i == current_step:
                        vis.add_geometry(trace)  # Add only the current step traces

        # Initialize the visualizer
        vis.poll_events()
        vis.update_renderer()

        # Simulate the slider effect
        for i in range(T):
            # Update visibility based on the current time step
            update_visibility(i)

            # Update the visualizer and wait for a bit
            vis.poll_events()
            vis.update_renderer()
            # You can use time.sleep(seconds) here if you want a pause between steps

        # Close the visualizer
        vis.destroy_window()

    # endregion

    # region
    def _plot_plotly(
        self,
        T_exo_world_cam_poses_w_anno: Optional[Dict[str, NDArray]] = None,
        T_exo_world_cam_w_gopro_calib: Optional[Dict[str, NDArray]] = None,
        T_ego_world_cams: Optional[Dict[str, NDArray]] = None,
        T_world_imus_w_open: Optional[Dict[str, NDArray]] = None,
        T_world_imus_w_close: Optional[Dict[str, NDArray]] = None,
        tracking_timestamps: Optional[Dict[str, NDArray]] = None,
        offset_z: NDArray = None,
        point_clouds: Optional[Dict[str, NDArray]] = None,
        downsample_point_cloud: bool = False,
        **kwargs,
    ):
        """
        Plot elements.

        Parameters
        ----------
        T_exo_world_cam_poses_w_anno  : dict of np.ndarray (3, 4)
        T_exo_world_cam_poses : dict of np.ndarray (3, 4)
        T_ego_world_cams : dict of np.ndarray (3, 4)
        T_world_imus_w_open_traj : np.array of shape (T, 3, 4)
        T_world_imus_w_close_traj : np.array of shape (T, 3, 4)
        tracking_timestamps : np.array of shape (T,)
        cam_trajs : list of np.ndarray (T, 3, 4)

        Notes
        -----
        The current timestamp steps is set to come from close_loop_traj.
        """
        assert tracking_timestamps is not None, "Tracking timestamps must be provided."

        static_traces = []
        T = len(tracking_timestamps)

        def _():
            return [None] * T

        dynamic_traces = defaultdict(_)
        mean_offset_z = np.mean(offset_z)

        if (
            T_exo_world_cam_poses_w_anno is not None
            and CFG.plotly.camera_frustum.enable
        ):
            exo_cam_frustums = [None] * len(T_exo_world_cam_poses_w_anno)
            for i, exo_cam_name in enumerate(T_exo_world_cam_poses_w_anno.keys()):
                T_exo_world_cam_pose_w_anno = T_exo_world_cam_poses_w_anno[
                    exo_cam_name
                ].copy()
                T_exo_world_cam_pose_w_anno[2, 3] += mean_offset_z
                exo_cam_frustums[i] = build_cam_frustum_w_extr_plotly(
                    T_exo_world_cam_pose_w_anno
                )
            static_traces.extend(exo_cam_frustums)

        if (
            T_exo_world_cam_w_gopro_calib is not None
            and CFG.plotly.camera_frustum.enable
        ):
            gopro_cams_frustums = [None] * len(T_exo_world_cam_w_gopro_calib)
            for i, exo_cam_name in enumerate(T_exo_world_cam_w_gopro_calib.keys()):
                T_tmp = T_exo_world_cam_w_gopro_calib[exo_cam_name].copy()
                T_tmp[2, 3] += mean_offset_z
                gopro_cams_frustums[i] = build_cam_frustum_w_extr_plotly(T_tmp)
            static_traces.extend(gopro_cams_frustums)

        if (
            T_ego_world_cams is not None
            and CFG.plotly.camera_3d_coordinate.enable
            and CFG.plotly.ego_cam_traj.enable
        ):
            T_ego = len(T_ego_world_cams)
            assert T_ego == T, (
                f"Tracking timestamps and T_ego_world_cams do not match. Tracking: {T}, Ego: {T_ego}"
            )

            def _():
                return [None] * T_ego

            dynamic_traces = defaultdict(_)
            for i, frame_num in enumerate(T_ego_world_cams.keys()):
                T_ego_world_cam = T_ego_world_cams[frame_num]
                ego_world_cam_qpose = T_to_qpose(T_ego_world_cam[None])
                opengl_cam_pose = aria_camera_device2opengl_pose(ego_world_cam_qpose)
                T_ego_world_cam = qpose_to_T(opengl_cam_pose)[0].copy()
                T_ego_world_cam[2, 3] += offset_z[i]

                (
                    dynamic_traces["ego_world_cam_x"][i],
                    dynamic_traces["ego_world_cam_y"][i],
                    dynamic_traces["ego_world_cam_z"][i],
                ) = draw_camera_pose_plotly(T_ego_world_cam)

            T_ego_world_cams = np.stack(list(T_ego_world_cams.values()), axis=0).copy()
            T_ego_world_cams[:, 2, 3] += offset_z
            logger.info(
                f"ego_cam_traj | x: {np.mean(T_ego_world_cams[:, 0, 3])}, y: {np.mean(T_ego_world_cams[:, 1, 3])}, z: {np.mean(T_ego_world_cams[:, 2, 3])}"
            )
            ego_cam_traj = go.Scatter3d(
                x=T_ego_world_cams[:, 0, 3],
                y=T_ego_world_cams[:, 1, 3],
                z=T_ego_world_cams[:, 2, 3],
                mode="markers",
                marker={
                    "size": CFG.plotly.ego_cam_traj.marker_size,
                    "opacity": 0.8,
                    "color": CFG.plotly.ego_cam_traj.color,
                },
                name="T_world_cam_traj",
                hoverinfo="none",
                visible=False,
            )

            static_traces.append(ego_cam_traj)

        if (
            T_world_imus_w_close is not None
            and T_world_imus_w_open is not None
            and CFG.plotly.closed_slam_traj.enable
            and CFG.plotly.open_slam_traj.enable
        ):
            T_open, *_ = T_world_imus_w_open.shape
            T_close, *_ = T_world_imus_w_close.shape
            assert T_open == T_close == T, (
                f"Open and close trajectory lengths do not match. Open: {T_open}, Close: {T_close}, T: {T}"
            )

            close_traj_cam_frustums = [None] * T_close

            for i in range(len(T_world_imus_w_close)):
                T_world_imu_w_close_traj = T_world_imus_w_close[i].copy()
                T_world_imu_w_close_traj[2, 3] += offset_z[i]
                close_traj_cam_frustums[i] = build_cam_frustum_w_extr_plotly(
                    T_world_imu_w_close_traj
                )
            dynamic_traces["close_traj_cam_frustums"] = close_traj_cam_frustums

            open_traj_cam_frustums = [None] * T_open
            for i in range(len(T_world_imus_w_open)):
                T_world_imu_w_open_traj = T_world_imus_w_open[i].copy()
                T_world_imu_w_open_traj[2, 3] += offset_z[i]
                open_traj_cam_frustums[i] = build_cam_frustum_w_extr_plotly(
                    T_world_imu_w_open_traj
                )
            dynamic_traces["open_traj_cam_frustums"] = open_traj_cam_frustums

        # plot Ts_world_cams
        Ts_world_cams = kwargs["Ts_world_cams"] if "Ts_world_cams" in kwargs else None

        world_cam_trajs = []

        def _():
            return [None] * T_ego

        if Ts_world_cams is not None and CFG.plotly.camera_frustum.enable:
            for ind, Ts_world_cam in enumerate(Ts_world_cams):
                T, *_ = Ts_world_cam.shape
                assert T == T_open, (
                    f"IMU and cam trajectory lengths do not match. IMU: {T_open}, Cam: {T}"
                )

                for i in range(T):
                    T_tmp = Ts_world_cam[i].copy()
                    # T_tmp[2, 3] += offset_z[i]
                    dynamic_traces[f"T_world_cam_{ind}_frustums"][i] = (
                        build_cam_frustum_w_extr_plotly(T_tmp)
                    )
                    (
                        dynamic_traces[f"T_world_cam_{ind}_x_frame"][i],
                        dynamic_traces[f"T_world_cam_{ind}_y_frame"][i],
                        dynamic_traces[f"T_world_cam_{ind}_z_frame"][i],
                    ) = draw_camera_pose(T_tmp)

                Ts_world_cam_ = Ts_world_cam.copy()
                # Ts_world_cam_[:, 2, 3] += offset_z
                ego_cam_traj = go.Scatter3d(
                    x=Ts_world_cam_[:, 0, 3],
                    y=Ts_world_cam_[:, 1, 3],
                    z=Ts_world_cam_[:, 2, 3],
                    mode="markers",
                    marker={"size": 2, "opacity": 0.8, "color": "blue"},
                    name="T_world_cam_traj",
                    hoverinfo="none",
                    visible=False,
                )
                logger.info(
                    f"Ts_world_cams/{ind} | x: {np.mean(Ts_world_cam_[:, 0, 3])}, y: {np.mean(Ts_world_cam_[:, 1, 3])}, z: {np.mean(Ts_world_cam_[:, 2, 3])}"
                )
                world_cam_trajs.append(ego_cam_traj)
            static_traces.extend(world_cam_trajs)

        # plot additional coco format world_3d_pts
        coco_world_3d_kpts_key = "coco_world_3d_pts"
        coco_world_3d_pts = (
            kwargs[coco_world_3d_kpts_key] if coco_world_3d_kpts_key in kwargs else None
        )
        if coco_world_3d_pts is not None and CFG.plotly.coco_kinematic_tree.enable:
            assert T == len(coco_world_3d_pts), (
                f"Tracking timestamps and coco world 3d points do not match. Tracking: {T}, Coco: {len(coco_world_3d_pts)}"
            )
            for ind, coco_world_3d_pt in enumerate(coco_world_3d_pts):
                tmp_3d_pt = coco_world_3d_pt[:NUM_EGOEXO4D_EGOPOSE_JNTS].copy()
                # tmp_3d_pt[:,2] += offset_z[ind]
                coco_lines_vis = draw_coco_kinematic_tree_plotly(
                    tmp_3d_pt, coco_cfg=CFG.plotly.coco_kinematic_tree
                )
                coco_lines_vis = {
                    f"coco_world_3d_pts_{i}": coco_lines_vis[i]
                    for i in range(len(coco_lines_vis))
                }
                for key, val in coco_lines_vis.items():
                    dynamic_traces[key][ind] = val

        # plot additional coco format world_3d_pts
        model_pred_coco_kpts = "model_pred_coco_kpts"
        model_pred_coco_kpts = (
            kwargs[model_pred_coco_kpts] if model_pred_coco_kpts in kwargs else None
        )
        if model_pred_coco_kpts is not None and CFG.plotly.model_pred_coco_kpts.enable:
            assert T == len(model_pred_coco_kpts), (
                f"Tracking timestamps and coco world 3d points do not match. Tracking: {T}, Coco: {len(model_pred_coco_kpts)}"
            )
            for ind, coco_world_3d_pt in enumerate(model_pred_coco_kpts):
                tmp_3d_pt = coco_world_3d_pt[:NUM_EGOEXO4D_EGOPOSE_JNTS].copy()
                # tmp_3d_pt[:,2] += offset_z[ind]
                coco_lines_vis = draw_coco_kinematic_tree_plotly(
                    tmp_3d_pt, coco_cfg=CFG.plotly.model_pred_coco_kpts
                )
                coco_lines_vis = {
                    f"model_pred_coco_world_3d_pts_{i}": coco_lines_vis[i]
                    for i in range(len(coco_lines_vis))
                }
                for key, val in coco_lines_vis.items():
                    dynamic_traces[key][ind] = val

        # plot additional coco format world_3d_pts
        model_gt_coco_kpts = "model_gt_coco_kpts"
        model_gt_coco_kpts = (
            kwargs[model_gt_coco_kpts] if model_gt_coco_kpts in kwargs else None
        )
        if model_gt_coco_kpts is not None and CFG.plotly.model_gt_coco_kpts.enable:
            assert T == len(model_gt_coco_kpts), (
                f"Tracking timestamps and coco world 3d points do not match. Tracking: {T}, Coco: {len(model_gt_coco_kpts)}"
            )
            for ind, coco_world_3d_pt in enumerate(model_gt_coco_kpts):
                tmp_3d_pt = coco_world_3d_pt[:NUM_EGOEXO4D_EGOPOSE_JNTS].copy()
                # tmp_3d_pt[:,2] += offset_z[ind]
                coco_lines_vis = draw_coco_kinematic_tree_plotly(
                    tmp_3d_pt, coco_cfg=CFG.plotly.model_gt_coco_kpts
                )
                coco_lines_vis = {
                    f"model_gt_coco_world_3d_pts_{i}": coco_lines_vis[i]
                    for i in range(len(coco_lines_vis))
                }
                for key, val in coco_lines_vis.items():
                    dynamic_traces[key][ind] = val

        # Plot trajectory and point cloud
        # We color the points by their z coordinate
        if T_world_imus_w_close is not None and CFG.plotly.closed_slam_traj.enable:
            tmp_T = T_world_imus_w_close.copy()
            tmp_T[:, 2, 3] += offset_z
            imu_close_traj_go_item = go.Scatter3d(
                x=tmp_T[:, 0, 3],
                y=tmp_T[:, 1, 3],
                z=tmp_T[:, 2, 3],
                mode="markers",
                marker={
                    "size": CFG.plotly.closed_slam_traj.marker_size,
                    "opacity": CFG.plotly.closed_slam_traj.opacity,
                    "color": CFG.plotly.closed_slam_traj.color,
                },
                name="close_loop_trajecotory",
                hoverinfo="none",
                visible=False,
            )
            static_traces.append(imu_close_traj_go_item)
            logger.info(
                f"close_trajectory | x: {np.mean(tmp_T[:, 0, 3])}, y: {np.mean(tmp_T[:, 1, 3])}, z: {np.mean(tmp_T[:, 2, 3])}"
            )

        if T_world_imus_w_open is not None and CFG.plotly.open_slam_traj.enable:
            tmp_T = T_world_imus_w_open.copy()
            tmp_T[:, 2, 3] += offset_z
            imu_open_traj_go_item = go.Scatter3d(
                x=tmp_T[:, 0, 3],
                y=tmp_T[:, 1, 3],
                z=tmp_T[:, 2, 3],
                mode="markers",
                marker={
                    "size": CFG.plotly.open_slam_traj.marker_size,
                    "opacity": CFG.plotly.open_slam_traj.opacity,
                    "color": CFG.plotly.open_slam_traj.color,
                },
                name="open_loop_trajectory",
                hoverinfo="none",
                visible=False,
            )
            static_traces.append(imu_open_traj_go_item)
            logger.info(
                f"open_trajectory | x: {np.mean(tmp_T[:, 0, 3])}, y: {np.mean(tmp_T[:, 1, 3])}, z: {np.mean(tmp_T[:, 2, 3])}"
            )

        if point_clouds is not None and CFG.plotly.semidense_observations.enable:
            # Filter the point cloud by inv depth and depth and load
            threshold_invdep = 5e-4
            threshold_dep = 5e-4
            point_clouds = filter_points_from_confidence(
                point_clouds, threshold_invdep, threshold_dep
            )
            # point_clouds = filter_points_from_confidence(point_clouds)
            if downsample_point_cloud:
                # Downsampling the data for web viewing
                point_clouds = filter_points_from_count(point_clouds, 500_000)
            # Retrieve point position
            point_clouds = np.stack([it.position_world for it in point_clouds])
            tmp_pts = point_clouds.copy()
            tmp_pts[:, 2] += mean_offset_z
            global_points = go.Scatter3d(
                x=tmp_pts[:, 0],
                y=tmp_pts[:, 1],
                z=tmp_pts[:, 2],
                mode="markers",
                marker={
                    "size": CFG.plotly.semidense_observations.size,
                    "color": tmp_pts[:, 2],
                    "cmin": -1.5,
                    "cmax": 2,
                    "colorscale": CFG.plotly.semidense_observations.colorscale,
                },
                name="Global Points",
                hoverinfo="none",
                visible=False,
            )
            static_traces.append(global_points)

        for static_trace in static_traces:
            static_trace.visible = True

        if static_traces is not None and len(static_traces) > 0:
            fig = go.Figure(data=static_traces)
        else:
            fig = go.Figure()

        for ele_key, traces in dynamic_traces.items():
            for trace in traces:
                fig.add_trace(trace)

        for ele_key, traces in dynamic_traces.items():
            traces[0].visible = True

        # Create slider steps
        steps = []
        for i in range(T):
            step = dict(
                method="update",
                args=[
                    {
                        "visible": [True] * len(static_traces)
                        + [
                            time_step == i
                            for traces in dynamic_traces.values()
                            for time_step in range(T)
                        ]
                    }
                ],
                label=tracking_timestamps[i],
            )
            steps.append(step)

        # Add sliders to the layout
        fig.update_layout(
            sliders=[
                dict(
                    currentvalue={"suffix": " s", "prefix": "Time :"},
                    pad={"t": 5},
                    steps=steps,
                )
            ],
            scene=dict(
                bgcolor="lightgray",
                dragmode="orbit",
                aspectmode="data",
                xaxis_visible=CFG.plotly.layout.x_visible,
                yaxis_visible=CFG.plotly.layout.y_visible,
                zaxis_visible=CFG.plotly.layout.z_visible,
            ),
        )

        fig.show()
        # endregion


if __name__ == "__main__":
    pass
