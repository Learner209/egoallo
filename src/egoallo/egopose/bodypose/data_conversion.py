import os
import os.path as osp

from tqdm import tqdm
from egoallo.egopose.handpose.data_preparation.utils.utils import HAND_ORDER
import numpy as np
import torch

from egoallo.utils.smpl_mapping.mapping import EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS, EGOEXO4D_EGOPOSE_HANDPOSE_MAPPINGS, USED_SMPLH_JOINT_NAMES
from egoallo.utils.smpl_mapping.mapping import EGOEXO4D_BODYPOSE_TO_SMPLH_INDICES, EGOEXO4D_HANDPOSE_TO_SMPLH_w_HAND_INDICES, SMPLH_BODY_JOINTS, SMPLH_HAND_JOINTS
from egoallo.smpl.smplh_utils import SMPL_JOINT_NAMES, NUM_SMPL_JNTS, EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS, NUM_EGOEXO4D_EGOPOSE_JNTS 
from egoallo.utils.time_utils import linear_interpolate_missing_vals
from egoallo.utils.ego_geom import batch_ZupLookAT2smplorigin ,openglpose2smplorigin, ros_pts_2_smpl_pts,rospose2smplorigin
from sklearn.cluster import DBSCAN
from egoallo.utils.setup_logger import setup_logger
from egoallo.smpl.simple_ik import simple_ik_solver_w_smplh
from egoallo.smpl.smplh_utils import SMPLHUtils
from egoallo.utils.transformation import T_to_qpose, qpose_to_T
from egoallo.utils.ego_geom import batch_align_to_reference_pose
from egoallo.config import make_cfg, CONFIG_FILE
from egoallo.smpl.smplh_utils import NUM_OF_BODY_JOINTS as SMPLH_BODY_JNT_NUM
from egoallo.smpl.smplh_utils import NUM_OF_HAND_JOINTS as SMPLH_HAND_JNT_NUM
from egoallo.smpl.smplh_utils import  SMPLH_BODY_JOINTS, SMPLH_HAND_JOINTS, SMPLH_NUM_BETAS
from egoallo.smpl import SMPLH_UTILS_INST
from egoallo.utils.utils import NDArray, Tensor, find_numerical_key_in_dict

from typing import Optional, Tuple, List, Dict, Any

local_config_file = CONFIG_FILE
CFG = make_cfg(config_name="defaults", config_file=local_config_file, cli_args=[])



from egoallo.utils.aria_utils.mps import AriaMPSService
from egoallo.egoexo.egoexo_utils import EgoExoUtils
from egoallo.env.mujoco.smpl_mujoco import SMPL_M_Viewer, SMPL_M_Renderer

import matplotlib.pyplot as plt

VIZ_PLOTS = False
VIS_W_MUJOCO = CFG.mujoco.vis.enable
VIS_W_GO3D = CFG.plotly.enable

logger = setup_logger(output=None, name=__name__)

BODY_JOINTS = EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS
HAND_JOINTS = EGOEXO4D_EGOPOSE_HANDPOSE_MAPPINGS
SMPLH_JOINT_NAMES = USED_SMPLH_JOINT_NAMES


class EgoPoseDataPreparation:
    def __init__(self, args):
        self.args = args
        self.egoexo_root_path = args.egoexo_root_path
        self.export_mp4_root_dir = args.exported_mp4_path
        self.portrait_view = args.portrait_view
        self.valid_kpts_num_thresh = args.valid_kpts_num_thresh
        self.bbox_padding = args.bbox_padding
        self.reproj_error_threshold = args.reproj_error_threshold
        self.hand_order = HAND_ORDER
        self.bm, self.bm_path = SMPLH_UTILS_INST.get_bm(smplh_key=CFG.smplh.smplh_model)

        # Load calibration
        # self.calib = calibration.Calibration(self.egoexo_root_path)

    def align_exported_anno_to_slam_traj(self, take_uid, egoexo_util_inst: EgoExoUtils, this_take_valid_timestamps: NDArray, this_take_ego_cam_traj: NDArray, this_take_ego_cam_intr: NDArray, this_take_anno_3d: Optional[NDArray], this_take_anno_3d_valid_flag):
        """
        Aligns exported annotations to SLAM traj data for a given take.

        Parameters
        ----------
        take_uid : str
            Unique identifier for the take.
        egoexo_util_inst : EgoExoUtils
            An instance of the EgoExoUtils class, which provides utility functions and data related to the take.
        this_take_valid_timestamps : ndarray, shape (N',)
        this_take_ego_cam_traj : ndarray, shape (N, 7)
        this_take_ego_cam_intr : ndarray of shape (N, 3, 3)
        this_take_anno_3d : ndarray, shape (T, J, 3) or None if there are no valid anno 3d.
            The 3D annotations for the take, where T is the number of frames, J is the number of joints.
        this_take_anno_3d_valid_flag : ndarray, shape (T, J) or None if there are not valid anno 3d.
            A boolean array indicating the validity of each 3D annotation.

        Returns
        -------
        this_take_aligned_3d : ndarray, shape (T, J, 3) or None if there are no valid anno 3d.
        this_take_aligned_ego_cam_traj : ndarray, shape (T, 7)
        exported_vrs_timestamps : ndarray, shape (T,)
        
        """

        take_name = egoexo_util_inst.find_take_name_from_take_uid(take_uid)
        exported_mp4_file_path = egoexo_util_inst.get_exported_mp4_path_from_take_name(take_name=take_name, exported_mp4_output_dir=self.export_mp4_root_dir)

        _, take_name, take, open_loop_traj_path, close_loop_traj_path, gopro_calibs_path, \
            cam_pose_anno_path, online_calib_json_path, vrs_path, vrs_noimagestreams_path, semidense_observations_path \
            = egoexo_util_inst.get_take_metadata_from_take_uid(take_uid)
        aria_mps_serv = AriaMPSService(vrs_file_path=vrs_path,
                                vrs_exported_mp4_path = exported_mp4_file_path,
                                take=take,
                                open_loop_traj_path=open_loop_traj_path,
                                close_loop_traj_path=close_loop_traj_path,
                                gopro_calibs_path=gopro_calibs_path,
                                cam_pose_anno_path=cam_pose_anno_path,
                                generalized_eye_gaze_path=None,
                                calibrated_eye_gaze_path=None,
                                online_calib_json_path=online_calib_json_path,
                                wrist_and_palm_poses_path=None,
                                point_observation_path=semidense_observations_path)

    
        exported_vrs_timestamps = aria_mps_serv.get_timestamps_from_exported_mp4()
        exported_vrs_timestamps = exported_vrs_timestamps[this_take_valid_timestamps]
        T_mp4, *_ = exported_vrs_timestamps.shape
        
        sampled_mps_close_traj_pose, T_world_imus_w_close_traj, tracking_timestamp_close_traj, utc_close_traj = aria_mps_serv.sample_close_loop_traj_from_timestamps(timestamps=exported_vrs_timestamps)
        sampled_mps_open_traj_pose, T_world_imus_w_open_traj, tracking_timestamp_open_traj, utc_open_traj = aria_mps_serv.sample_open_loop_traj_from_timestamps(timestamps=exported_vrs_timestamps)
 
        try:
            assert len(this_take_ego_cam_traj) == T_mp4 
        except Exception as e:
            logger.critical(f"Ego cam anno pose json: {len(this_take_ego_cam_traj)} | exported_mp4_timestamps: {T_mp4}")
            return

        # aligned_ego_traj_trans, aligned_ego_traj_rot_mat, aligned_traj_quat_wxyz, to_align2ref_rot_seq, move_to_ref_trans = batch_align_to_reference_pose(this_take_ego_cam_traj[None,], sampled_mps_close_traj_pose)
        # aligned_ego_traj_trans, aligned_ego_traj_rot_mat, aligned_traj_quat_wxyz = aligned_ego_traj_trans[0], aligned_ego_traj_rot_mat[0], aligned_traj_quat_wxyz[0]
        # this_take_aligned_ego_cam_traj = np.concatenate([aligned_ego_traj_trans, aligned_traj_quat_wxyz], axis=1) # T x 7
        # T_ego_world_cams = qpose_to_T(this_take_aligned_ego_cam_traj) # T x 3 x 4
        T_ego_world_cams = qpose_to_T(this_take_ego_cam_traj)

        aligned_open_traj, aligned_open_traj_rot_mat, aligned_open_traj_quat_wxyz, _, _ = batch_align_to_reference_pose(sampled_mps_open_traj_pose[None,], sampled_mps_close_traj_pose)
        aligned_open_traj, aligned_open_traj_rot_mat, aligned_open_traj_quat_wxyz = aligned_open_traj[0], aligned_open_traj_rot_mat[0], aligned_open_traj_quat_wxyz[0]
        this_take_aligned_open_traj = np.concatenate([aligned_open_traj, aligned_open_traj_quat_wxyz], axis=1) # T x 7
        T_world_imus_w_open_traj = qpose_to_T(this_take_aligned_open_traj) # T x 3 x 4
        


        if this_take_anno_3d is not None and this_take_anno_3d_valid_flag is not None:
            # ! plotly.graph_objs's observations tell us that align_to_slam_traj and subtracting ankle height is not necessary.
            # old_shape = this_take_anno_3d.shape
            # this_take_anno_3d = this_take_anno_3d.reshape(-1, 3)[...,None] #  (T*J) x 3 x 1
            # this_take_aligned_3d = (to_align2ref_rot_seq[0:1] @ this_take_anno_3d)[...,0] # (T*J) x 3
            # this_take_aligned_3d = this_take_aligned_3d move_to_ref_trans[0:1] # (T*J) x 3
            # this_take_aligned_3d = this_take_aligned_3d.reshape(old_shape) # T x J x 3

            # Make sure the aligned 3D annotations has their ankle heights on the z=0 plane.
            # this_take_root_height_3d, has_left_ankle_flag, has_right_ankle_flag = self.extract_ankle_height(this_take_aligned_3d, this_take_anno_3d_valid_flag)
            # this_take_floor_height_3d = this_take_root_height_3d - CFG.empirical_val.ankle_floor_height_offset # T 
            # this_take_aligned_3d[:,:,2] = this_take_aligned_3d[:,:,2] this_take_floor_height_3d[:, None] 
            this_take_aligned_3d = this_take_anno_3d
            pass
        else:
            this_take_aligned_3d = None

        if VIS_W_GO3D:
            point_clouds = aria_mps_serv.read_semidense_point_cloud()
            aria_mps_serv.plot_3d(T_world_imus_w_close_traj,
                                T_world_imus_w_open_traj,
                                tracking_timestamp_close_traj,
                                this_take_valid_timestamps,
                                point_clouds,
                                Ts_world_cams=T_ego_world_cams[None,],
                                world_3d_pts=this_take_aligned_3d,
                                )
        return this_take_anno_3d, this_take_ego_cam_traj, exported_vrs_timestamps
    
    def predict_pelvis_origin(self, this_take_anno_3d, this_take_anno_3d_valid_flag):
        # region
        """
        Predict the pelvis origin for each frame based on the 3D annotations of body joints(in CoCo25 convention). The method 
        processes annotations to determine the probable pelvis origin using velocity clustering of joint movements 
        and height calculations based on ankle positions.

        Parameters
        ----------
        this_take_anno_3d : ndarray of shape (T, J, D)
            The input 3D annotation array with shape (T, J, D), where T is the number of frames, 
            J is the number of joints, and D is the number of dimensions for each joint's coordinates (typically 3D).

        this_take_anno_3d_valid_flag : ndarray of shape (T, J)
            The validity flags for the input 3D annotations with shape (T, J), where T is the number of frames 
            and J is the number of joints. A true value in this array indicates that the corresponding joint's 
            annotation is valid and can be used for calculations.

        Returns
        -------
        pred_pelvis_origins: ndarray of shape (T, 3)
            An array with the predicted pelvis 3d coordinate directions having shape (T, 3), where T is the number of frames. There should be no NaN values in this output.

        Notes
        -----
        The function assumes that the joint annotations include specific indices for the left and right ankles. It handles frames where ankle data 
        may be missing and uses DBSCAN clustering to analyze joint velocity patterns to infer the most probable 
        pelvis position.

        The method is sensitive to the accuracy and completeness of the input data, particularly the validity of 
        ankle positions and the overall movement dynamics represented in the velocity of joint movements.

        Examples
        --------
        >>> this_take_anno_3d = np.random.rand(100, 42+17, 3)
        >>> this_take_anno_3d_valid_flag = np.random.rand(100, 42+17) > 0.3
        >>> predicted_origins = predict_pelvis_origin(this_take_anno_3d, this_take_anno_3d_valid_flag)
        >>> print(predicted_origins.shape)
        (100, 3)

        """

        T, J, D = this_take_anno_3d.shape

        assert J == len(BODY_JOINTS) + len(HAND_JOINTS) or len(BODY_JOINTS), f"The input 3D annotation should have {len(BODY_JOINTS) + len(HAND_JOINTS)} or {len(BODY_JOINTS)} joints"
        assert D == 3, "The input 3D annotation should have 3D coordinates"

        this_take_body_anno_3d, this_take_hand_anno_3d = this_take_anno_3d[:,:len(BODY_JOINTS)], this_take_anno_3d[:,len(BODY_JOINTS):]
        this_take_body_anno_3d_valid_flag, this_take_hand_anno_3d_valid_flag = this_take_anno_3d_valid_flag[:,:len(BODY_JOINTS)], this_take_anno_3d_valid_flag[:,len(BODY_JOINTS):]
        # T x (body_jnts), T x (hand_jnts)

        this_take_prev_body_anno_3d = this_take_body_anno_3d.copy() # T x (body_jnts+hand_jnts) x 3
        this_take_prev_body_anno_3d = np.concatenate([np.zeros((1,J,3)), this_take_prev_body_anno_3d[:-1]], axis=0) # T x (body_jnts+hand_jnts) x 3
        this_take_anno_3d_vel = this_take_body_anno_3d - this_take_prev_body_anno_3d # T x (body_jnts+hand_jnts) x 3
        # Assuming the first frame is consistent with the second frame.
        this_take_anno_3d_vel[0] = this_take_anno_3d_vel[1] # T x (body_jnts+hand_jnts) x 3 
        body_vel_valid_flag = np.logical_and(~np.isnan(np.mean(this_take_body_anno_3d, axis=-1)), ~np.isnan(np.mean(this_take_prev_body_anno_3d, axis=-1))) # T x (body_jnts+hand_jnts)
        body_vel_valid_flag[0] = body_vel_valid_flag[1] 

        # Skip this body if left-ankle and right-ankle are both None
        left_ankle_idx = BODY_JOINTS.index("left-ankle")
        right_ankle_idx = BODY_JOINTS.index("right-ankle")

        miss_left_ankle_flag = (~this_take_body_anno_3d_valid_flag)[:,left_ankle_idx]
        miss_right_ankle_flag = (~this_take_body_anno_3d_valid_flag)[:,right_ankle_idx]

        assert not np.any(np.logical_and(miss_left_ankle_flag, miss_right_ankle_flag)), "Both left ankle and right ankle are missing, skip this body"
        assert np.any(np.sum(this_take_body_anno_3d_valid_flag, axis=-1) >= 9), "At least 9 joints should be valid, skip this body"

        tmp_body_anno_3d_w_left_ankle = this_take_body_anno_3d.copy() # T x J x 3
        tmp_body_anno_3d_w_right_ankle = this_take_body_anno_3d.copy() # T x J x 3
        # A little hack since both the left and right ankle could not be missing at the same time.
        tmp_body_anno_3d_w_left_ankle[miss_left_ankle_flag,left_ankle_idx] = tmp_body_anno_3d_w_right_ankle[miss_left_ankle_flag,right_ankle_idx]
        tmp_body_anno_3d_w_right_ankle[miss_right_ankle_flag,right_ankle_idx] = tmp_body_anno_3d_w_left_ankle[miss_right_ankle_flag,left_ankle_idx]
        
        this_take_root_height_3d = np.mean(np.stack([tmp_body_anno_3d_w_left_ankle[:,left_ankle_idx],tmp_body_anno_3d_w_right_ankle[:,right_ankle_idx]],axis=1), axis=1) # T x 3
        # this_take_root_height_3d = this_take_root_height_3d[:,2] # T

        pred_pelvis_origin_valid = []

        for frame_ind in tqdm(range(T), total=T, desc="Predicting pelvis origin", ascii=' >='):
            # ! Perform DBSCAN clustering on the num_of_jnts vels between the current frame and the prev frame to `find the most likely global orient`
    
            in_cluster_inds = body_vel_valid_flag[frame_ind] # (body_jnts+hand_jnts)
            in_anno_3d = this_take_body_anno_3d[frame_ind][this_take_anno_3d_valid_flag[frame_ind]] # J' x 3
            in_anno_3d_vel = this_take_anno_3d_vel[frame_ind][in_cluster_inds] # J' x 3
            in_anno_root_heights = this_take_root_height_3d[frame_ind] # J'

            # NOTE: this clustering assumes the `pelvis` facing direction is aligned to human-body-movement to perform balanced and prioceptive actions. 
            cluster_vels = []
            cluster_sizes = []

            # cluster foot heights and find one with smallest median
            clustering = DBSCAN(eps=0.005, min_samples=3).fit(in_anno_3d_vel) # J' x 3
            all_labels = np.unique(clustering.labels_)
            all_static_inds = np.arange(sum(in_cluster_inds))

            min_median = min_root_median = float('inf')
            for cur_label in all_labels:
                cur_clust = in_anno_3d_vel[clustering.labels_ == cur_label]
                cur_clust_inds = np.unique(all_static_inds[clustering.labels_ == cur_label]) # inds in the original sequence that correspond to this cluster
                if VIZ_PLOTS:
                    plt.scatter(cur_clust, np.zeros_like(cur_clust), label='foot %d' % (cur_label))
                # get median foot height and use this as height
                cur_median = np.median(cur_clust)
                # cur_mean = np.mean(cur_clust)
                cluster_vels.append(cur_median)
                cluster_sizes.append(cur_clust.shape[0])

                # update min info
                if cur_median < min_median:
                    min_median = cur_median

            cluster_size_acc = np.sum(cluster_sizes)
            cluster_weight_acc = np.zeros((1,3))
            for cluster_ind, (cluster_size, cluster_vel) in enumerate(zip(cluster_sizes, cluster_vels)):
                cluster_weight_acc += cluster_size * cluster_vel
            weighted_cluster_vel = cluster_weight_acc / cluster_size_acc
            pred_pelvis_origin_valid.append(weighted_cluster_vel)

        pred_pelvis_origin_valid = np.vstack(pred_pelvis_origin_valid) # T x 3

        return pred_pelvis_origin_valid
        # endregion
    
    def extract_ankle_height(self, this_take_anno_3d, this_take_anno_3d_valid_flag):
        """
        Extracts the average height of the left and right ankles for each frame in a dataset.

        Parameters
        ----------
        this_take_anno_3d : np.ndarray
            3D annotations for each joint per timestep. Shape (T, J, D) where T is the number of timesteps, J is the number of joints, and D is the dimension of the coordinates (3 for x, y, z).
        this_take_anno_3d_valid_flag : np.ndarray
            Validity flags for the annotations, indicating whether a joint's data is valid (1) or missing (0). Shape matches `this_take_anno_3d` in the first two dimensions (T, J).

        Returns
        -------
        this_take_root_height_3d : np.ndarray
            The average z-coordinate (height) of the left and right ankles for each timestep. Shape (T,).
        has_left_ankle_flag : np.ndarray
            Boolean array indicating presence of left ankle data per timestep. Shape (T,).
        has_right_ankle_flag : np.ndarray
            Boolean array indicating presence of right ankle data per timestep. Shape (T,).

        Raises
        ------
        AssertionError
            If both left and right ankles are missing in any timestep or if any required joint data is missing entirely.

        """

        T, J, D = this_take_anno_3d.shape

        assert J == len(BODY_JOINTS) + len(HAND_JOINTS) or J == len(BODY_JOINTS), f"The input 3D annotation should have {len(BODY_JOINTS) + len(HAND_JOINTS)} or {len(BODY_JOINTS)} joints, instead have {J}."
        assert D == 3, "The input 3D annotation should have 3D coordinates"

        this_take_body_anno_3d, this_take_hand_anno_3d = this_take_anno_3d[:,:len(BODY_JOINTS)], this_take_anno_3d[:,len(BODY_JOINTS):]
        this_take_body_anno_3d_valid_flag, this_take_hand_anno_3d_valid_flag = this_take_anno_3d_valid_flag[:,:len(BODY_JOINTS)], this_take_anno_3d_valid_flag[:,len(BODY_JOINTS):]
        # T x (body_jnts), T x (hand_jnts)

        left_ankle_idx = BODY_JOINTS.index("left-ankle")
        right_ankle_idx = BODY_JOINTS.index("right-ankle")

        miss_left_ankle_flag = (~this_take_body_anno_3d_valid_flag)[:,left_ankle_idx]
        miss_right_ankle_flag = (~this_take_body_anno_3d_valid_flag)[:,right_ankle_idx]

        assert not np.any(np.logical_and(miss_left_ankle_flag, miss_right_ankle_flag)), "Both left ankle and right ankle are missing, skip this body"
        assert np.any(np.sum(this_take_body_anno_3d_valid_flag, axis=-1) >= 9), "At least 9 joints should be valid, skip this body"

        tmp_body_anno_3d_w_left_ankle = this_take_body_anno_3d.copy() # T x J x 3
        tmp_body_anno_3d_w_right_ankle = this_take_body_anno_3d.copy() # T x J x 3
        # A little hack since both the left and right ankle could not be missing at the same time.
        tmp_body_anno_3d_w_left_ankle[miss_left_ankle_flag,left_ankle_idx] = tmp_body_anno_3d_w_right_ankle[miss_left_ankle_flag,right_ankle_idx]
        tmp_body_anno_3d_w_right_ankle[miss_right_ankle_flag,right_ankle_idx] = tmp_body_anno_3d_w_left_ankle[miss_right_ankle_flag,left_ankle_idx]
        
        this_take_root_height_3d = np.mean(np.stack([tmp_body_anno_3d_w_left_ankle[:,left_ankle_idx],tmp_body_anno_3d_w_right_ankle[:,right_ankle_idx]],axis=1), axis=1) # T
        this_take_root_height_3d = this_take_root_height_3d[:,2] # T
        has_left_ankle_flag = ~miss_left_ankle_flag
        has_right_ankle_flag = ~miss_left_ankle_flag

        return this_take_root_height_3d, has_left_ankle_flag, has_right_ankle_flag

    def predict_hip_trans(self, this_take_anno_3d, this_take_anno_3d_valid_flag):
        # region
        """
        Predict the average translation vector for the hip joints using 3D annotations of left and right hips,
        interpolating missing values where necessary.

        Parameters
        ----------
        this_take_anno_3d : np.ndarray
            A 3D numpy array of shape (T, J, D) containing the 3D joint coordinates,
            where T is the number of frames, J is the number of joints, and D is the dimensionality (always 3).
        this_take_anno_3d_valid_flag : np.ndarray
            A 2D boolean numpy array of shape (T, J) indicating the validity of each joint annotation per frame.

        Returns
        -------
        np.ndarray
            A numpy array of shape (T, 3), representing the interpolated translation vectors for the hip
            across T frames.

        """
        T, J, D = this_take_anno_3d.shape

        assert J == len(BODY_JOINTS) + len(HAND_JOINTS) or J == len(BODY_JOINTS), f"The input 3D annotation should have {len(BODY_JOINTS) + len(HAND_JOINTS)} or {len(BODY_JOINTS)} joints, instead have {J}."
        assert D == 3, "The input 3D annotation should have 3D coordinates"

        this_take_body_anno_3d, this_take_hand_anno_3d = this_take_anno_3d[:,:len(BODY_JOINTS)], this_take_anno_3d[:,len(BODY_JOINTS):]
        this_take_body_anno_3d_valid_flag, this_take_hand_anno_3d_valid_flag = this_take_anno_3d_valid_flag[:,:len(BODY_JOINTS)], this_take_anno_3d_valid_flag[:,len(BODY_JOINTS):]
        # T x (body_jnts), T x (hand_jnts)

        left_hip_ind = BODY_JOINTS.index("left-hip")
        right_hip_ind = BODY_JOINTS.index("right-hip")

        # TODO; consider scenario where left ankle and right ankle doesn't has valid annotations.
        # TODO: the current impl only uses mean of left ankle and right ankle as pesudo transl, and gloal_orient is set to all zeros.
        this_take_trans_left_hip = this_take_body_anno_3d.copy()[:,left_hip_ind] # T x 3
        this_take_trans_right_hip = this_take_body_anno_3d.copy()[:,right_hip_ind] # T x 3

        miss_left_hip_flag = np.isnan(np.mean(this_take_trans_left_hip,axis=1)) # T
        miss_right_hip_flag = np.isnan(np.mean(this_take_trans_right_hip,axis=1)) # T
        miss_both_hips_flag = np.logical_and(miss_left_hip_flag, miss_right_hip_flag) # T

        # assert not np.any(miss_both_hips_flag), "Both left hip and right hip are missing, skip this body"

        this_take_trans_left_hip[miss_left_hip_flag] = this_take_trans_right_hip[miss_left_hip_flag]
        this_take_trans_right_hip[miss_right_hip_flag] = this_take_trans_left_hip[miss_right_hip_flag]

        this_take_trans_hip = np.mean(np.stack([this_take_trans_left_hip, this_take_trans_right_hip]),axis=0) # T x 3
        this_take_trans_hip[miss_both_hips_flag] = np.nan # T x 3

        this_take_trans_hip = linear_interpolate_missing_vals(this_take_trans_hip)

        # TODO: the pesudo transl may have conflicts with the optimization process, so detract a small value.
        this_take_trans_hip[:,2] += 0.05 # T x 3
        
        this_take_prev_hip_trans = this_take_trans_hip.copy() # T x 3
        this_take_prev_hip_trans = np.concatenate([np.zeros((1,3)), this_take_prev_hip_trans[:-1]], axis=0) # T x 3
        this_take_prev_3d_vel = this_take_trans_hip - this_take_prev_hip_trans # T x 3
        this_take_prev_3d_vel[0] = this_take_prev_3d_vel[1] # T x 3 
        
        return this_take_trans_hip, this_take_prev_3d_vel
        # endregion

    def generate_smpl_converted_anno(self, this_take_valid_timestamps, this_take_anno_3d, this_take_anno_3d_valid_flag, this_take_ego_cam_traj: NDArray, egoexo_util_inst: EgoExoUtils,   take_uid: str, take_name: str, dry_run: bool = False, **kwargs):
        """
        Converts 3D annotations to the SMPL-H format, computes transformations, and visualizes the results.

        Parameters
        ----------
        this_take_anno_3d : ndarray
            The 3D annotations for the take with shape (T, J, 3), where T is the number of frames, J is the number of joints.
        this_take_anno_3d_valid_flag : ndarray
            A boolean array indicating the validity of each 3D annotation with shape (T, J).
        seq_name : str
            The sequence name used for visualization and file naming.

        Returns
        -------
        tuple of np.ndarray.
            A tuple of np.ndarray containing:
            - opt_this_take_smplh_anno_3d: Converted SMPL-H 3D annotations (T, 52, 3).
            - opt_this_take_smplh_verts: SMPL-H vertex coordinates (T, 6890, 3).
            - opt_this_take_smplh_faces: SMPL-H face indices (T, 13776, 3).
            - opt_this_take_smplh_local_aa_rep: SMPL-H local axis-angle representation (T, 52, 3).
            - opt_this_take_smplh_root_trans: SMPL-H root translation (T, 3).
            - opt_this_take_smplh_root_rot: SMPL-H root rotation (T, 3).
             

        Raises
        ------
        AssertionError
            If the input joint count does not match expected counts or the annotation is not in 3D format.

        Notes
        -----
        This function uses an IK solver and the SMPL-H model to align and convert the input 3D annotations. Visualization
        of the results is optionally provided based on configuration settings.
        """
        T, J, D = this_take_anno_3d.shape

        assert J == len(BODY_JOINTS) + len(HAND_JOINTS) or J == len(BODY_JOINTS), f"The input 3D annotation should have {len(BODY_JOINTS) + len(HAND_JOINTS)} or {len(BODY_JOINTS)} joints, instead have {J}."
        assert D == 3, "The input 3D annotation should have 3D coordinates"

        this_take_body_anno_3d, this_take_hand_anno_3d = this_take_anno_3d[:,:len(BODY_JOINTS)], this_take_anno_3d[:,len(BODY_JOINTS):]
        this_take_body_anno_3d_valid_flag, this_take_hand_anno_3d_valid_flag = this_take_anno_3d_valid_flag[:,:len(BODY_JOINTS)], this_take_anno_3d_valid_flag[:,len(BODY_JOINTS):]
        # T x (body_jnts), T x (hand_jnts)

        # Extract ankle-height and put the ankle on the z=0 plane.
        this_take_root_height_3d, has_left_ankle_flag, has_right_ankle_flag = self.extract_ankle_height(this_take_anno_3d, this_take_anno_3d_valid_flag)
        this_take_root_height_3d = this_take_root_height_3d - CFG.empirical_val.ankle_floor_height_offset # T 
        # this_take_anno_3d[:,:,2] = this_take_anno_3d[:,:,2] - this_take_root_height_3d[:, None] 
        # this_take_ego_cam_traj[:,2] = this_take_ego_cam_traj[:,2] - this_take_root_height_3d # T x 7

        this_take_hip_trans, this_take_pelvis_origin = self.predict_hip_trans(this_take_anno_3d, this_take_anno_3d_valid_flag) # T x 3

        # smpl_origin_as_quat, smpl_origin_as_euler = batch_ZupLookAT2smplorigin(this_take_pelvis_origin) # T x 3, T x 4
        # running_mean_win = CFG.io.egoexo.preprocessing.smpl_origin_running_mean_window
        # TODO: it is better to execute running mean on this_take_hip_traj(quat) to obtain more stable results, the impl requires quat_interpolation (quat_slerp), maybe worth investigating later.
        this_take_hip_traj = this_take_ego_cam_traj.copy() # T x 7
        this_take_hip_traj[:,:3] = this_take_hip_trans # T x 7
        T_smpl_world_cam, R_smpl_world_cam_as_euler, R_smpl_world_cam_as_rotvec = openglpose2smplorigin(this_take_hip_traj) # T x 4 x 4, T x 3, T x 3

        this_take_pelvis_origin = R_smpl_world_cam_as_rotvec # T x 3

        this_take_smplh_anno_3d, this_take_smplh_anno_3d_valid_flag = self.convert_to_smplh_convention(this_take_anno_3d,this_take_anno_3d_valid_flag)
        # T x 52 x 3, T x 52
        # From ros to smpl convention, only applied to this_take_body_3d and this_take_hip_trans.
        # this_take_hip_origin is taken care by `opengl2smplorigin`.
        # ! There is no need to execute ros_2_smpl conversion.
        # old_shape = this_take_smplh_anno_3d.shape
        # smpl_coord_anno_3d = ros_pts_2_smpl_pts(this_take_smplh_anno_3d.reshape(-1, 3)) # T x 52 x 3
        # smpl_coord_anno_3d = smpl_coord_anno_3d.reshape(old_shape) # T x 52 x 3
        # smpl_coord_hip_trans = ros_pts_2_smpl_pts(this_take_hip_trans) # T x 3

        if VIS_W_GO3D:
            # TODO: put into the model pred coco_kpts for visualization temporarily.
            coco_val_pred = kwargs['coco_val_pred']
            coco_val_gt = kwargs['coco_val_gt']
            coco_test_pred = kwargs['coco_test_pred']
            model_pred_set = set(coco_val_gt.keys()).union(set(coco_test_pred.keys()))
            coco_model_gt = {**coco_val_gt}
            coco_model_pred = {**coco_val_pred}

            if take_uid not in model_pred_set:
                return None,None,None,None,None,None 

            this_take_coco_model_pred = coco_model_pred[take_uid]['body']

            coco_model_frames = np.asarray(find_numerical_key_in_dict(this_take_coco_model_pred))
            this_take_valid_timestamps = np.asarray(list(set(this_take_valid_timestamps.tolist()).intersection(set(coco_model_frames.tolist()))))

            this_take_coco_model_pred = np.stack(list(this_take_coco_model_pred.values())) # N x 17 x 3

            this_take_coco_model_gt = coco_model_gt[take_uid]['body']
            this_take_coco_model_gt = np.stack(list(this_take_coco_model_gt.values())) # N x 17 x 3
            
            this_take_coco_model_gt = this_take_coco_model_gt[np.isin(coco_model_frames, this_take_valid_timestamps)] # T x 17 x 3
            this_take_coco_model_pred = this_take_coco_model_pred[np.isin(coco_model_frames, this_take_valid_timestamps)] # N x 17 x 3

            # TODO: this is a failed temporary fix, this_take_root_height_3d has different frames numbers with this_take_coco_model_pred. 
            # this_take_coco_model_pred[:,:,2] = this_take_coco_model_pred[:,:,2] - this_take_root_height_3d # T x 7
            this_take_root_height_3d = np.zeros((this_take_coco_model_pred.shape[0],)) # T

            take_name = egoexo_util_inst.find_take_name_from_take_uid(take_uid)
            exported_mp4_file_path = egoexo_util_inst.get_exported_mp4_path_from_take_name(take_name=take_name, exported_mp4_output_dir=self.export_mp4_root_dir)

            _, take_name, take, open_loop_traj_path, close_loop_traj_path, gopro_calibs_path, \
                cam_pose_anno_path, online_calib_json_path, vrs_path, vrs_noimagestreams_path, semidense_observations_path \
                = egoexo_util_inst.get_take_metadata_from_take_uid(take_uid)
            aria_mps_serv = AriaMPSService(vrs_file_path=vrs_path,
                                    vrs_exported_mp4_path = exported_mp4_file_path,
                                    take=take,
                                    open_loop_traj_path=open_loop_traj_path,
                                    close_loop_traj_path=close_loop_traj_path,
                                    gopro_calibs_path=gopro_calibs_path,
                                    cam_pose_anno_path=cam_pose_anno_path,
                                    generalized_eye_gaze_path=None,
                                    calibrated_eye_gaze_path=None,
                                    online_calib_json_path=online_calib_json_path,
                                    wrist_and_palm_poses_path=None,
                                    point_observation_path=semidense_observations_path)
            T_ego_world_cams = qpose_to_T(this_take_ego_cam_traj)
        
            exported_vrs_timestamps = aria_mps_serv.get_timestamps_from_exported_mp4()
            exported_vrs_timestamps = exported_vrs_timestamps[this_take_valid_timestamps]
            T_mp4, *_ = exported_vrs_timestamps.shape
            
            sampled_mps_close_traj_pose, T_world_imus_w_close_traj, tracking_timestamp_close_traj, utc_close_traj = aria_mps_serv.sample_close_loop_traj_from_timestamps(timestamps=exported_vrs_timestamps)
            sampled_mps_open_traj_pose, T_world_imus_w_open_traj, tracking_timestamp_open_traj, utc_open_traj = aria_mps_serv.sample_open_loop_traj_from_timestamps(timestamps=exported_vrs_timestamps)
    
            point_clouds = aria_mps_serv.read_semidense_point_cloud()
            aria_mps_serv.plot_3d(T_world_imus_w_close_traj,
                                T_world_imus_w_open_traj,
                                tracking_timestamp_close_traj,
                                this_take_valid_timestamps,
                                -this_take_root_height_3d, # T 
                                point_clouds,
                                # None,
                                Ts_world_cams=T_smpl_world_cam[None,],
                                # coco_world_3d_pts=this_take_anno_3d,
                                coco_world_3d_pts=None,
                                smpl_world_3d_pts=this_take_smplh_anno_3d[:,:NUM_SMPL_JNTS,:],
                                model_pred_coco_kpts=this_take_coco_model_pred,
                                model_gt_coco_kpts=this_take_coco_model_gt,
                                # model_pred_coco_kpts=None,
                                )
 
        opt_this_take_smplh_anno_3d = []  # jnts
        opt_this_take_smplh_verts = []
        opt_this_take_smplh_faces = []
        opt_this_take_smplh_local_aa_rep = []
        opt_this_take_smplh_root_trans = []
        opt_this_take_smplh_root_rot = []

        if dry_run:
            return None, None, None, None, None, None 

        for frame_ind in tqdm(range(T), total=T, desc="Geenrating smpl converted annotation using simple ik sovler", ascii=' >='):
            
            this_frame_smplh_anno_3d_valid_flag = this_take_smplh_anno_3d_valid_flag[frame_ind] # 52
            this_frame_smplh_anno_3d = this_take_smplh_anno_3d[frame_ind] # 52 x 3
            this_frame_hip_trans = this_take_hip_trans[frame_ind] # 3
            this_frame_pelvis_origin = this_take_pelvis_origin[frame_ind] # 3

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            this_frame_hip_trans = torch.from_numpy(this_frame_hip_trans).to(device)
            this_frame_pelvis_origin = torch.from_numpy(this_frame_pelvis_origin).to(device)

            root_smplh_path = osp.join(CFG.smplh.smplh_root_path, CFG.smplh.smplh_model, "model.npz")
            
            # TODO: currently only suports bodypose, hand pose excluded.
            opt_local_aa_rep = simple_ik_solver_w_smplh(root_smplh_path = root_smplh_path, 
                                    target=torch.from_numpy(this_frame_smplh_anno_3d).to(device),
                                    target_mask=torch.from_numpy(this_frame_smplh_anno_3d_valid_flag).to(device),
                                    transl=this_frame_hip_trans[None],
                                    global_orient=this_frame_pelvis_origin[None],
                                    device=device) # 52 x 3
            opt_local_aa_rep[0,:3] = this_frame_pelvis_origin # 3

            this_frame_hip_trans = this_frame_hip_trans[None, None] # BS(1) x T(1) x 3
            opt_local_aa_rep  = opt_local_aa_rep[None, None] # BS(1) x T(1) x 52 x 3
            smplh_betas = torch.zeros(1, SMPLH_NUM_BETAS) # BS(1) x 16

            opt_pred_smplh_jnts, opt_pred_smplh_verts, opt_pred_smplh_faces = SMPLH_UTILS_INST.run_smpl_model(this_frame_hip_trans, opt_local_aa_rep, smplh_betas, self.bm, device=device)
            # B(1) x T(1) x 52 x 3, B(1) x T(1) x 6890 x 3, 13776 x 3

            dest_mesh_vis_folder = CFG.io.egoexo.save_mesh.vis_folder
            os.makedirs(dest_mesh_vis_folder, exist_ok=True)
            # gen_full_body_vis(opt_pred_smplh_verts[0], opt_pred_smplh_faces, dest_mesh_vis_folder, seq_name, vis_gt=False)

            opt_this_take_smplh_anno_3d.append(opt_pred_smplh_jnts[0,0].detach().cpu().numpy())
            opt_this_take_smplh_verts.append(opt_pred_smplh_jnts[0,0].detach().cpu().numpy())
            opt_this_take_smplh_faces.append(opt_pred_smplh_faces.detach().cpu().numpy())
            opt_this_take_smplh_local_aa_rep.append(opt_local_aa_rep[0,0].detach().cpu().numpy())
            opt_this_take_smplh_root_trans.append(this_frame_hip_trans[0,0].detach().cpu().numpy())
            opt_this_take_smplh_root_rot.append(this_frame_pelvis_origin.detach().cpu().numpy())

            if VIS_W_MUJOCO:
                mujoco_assets_file = osp.join(CFG.mujoco.mujoco_assets.root_path, f"{CFG.mujoco.mujoco_assets.model_id}.xml")
                smpl_m_viewer = SMPL_M_Viewer(model_file=mujoco_assets_file)
                T = 1
                smpl_local_aa_pose = opt_local_aa_rep[0,0:1,:SMPLH_BODY_JNT_NUM, :].reshape(T, -1).detach().cpu().numpy() # T x (22*3)
                smpl_local_aa_pose = np.concatenate([smpl_local_aa_pose.reshape(T, -1), np.zeros((T, 2 * 3))],axis=1) # T x (24*3)
                smpl_global_trans = this_frame_hip_trans[0].reshape(T, -1).detach().cpu().numpy() # T x 3
                
                smpl_m_viewer.set_smpl_pose(pose=smpl_local_aa_pose, trans=smpl_global_trans, offset_z=0)
                smpl_m_viewer.show_pose(return_img=False, loop=False, size=(1920, 1080))

        opt_this_take_smplh_anno_3d = np.stack(opt_this_take_smplh_anno_3d, axis=0) # T x 52 x 3
        opt_this_take_smplh_verts = np.stack(opt_this_take_smplh_verts, axis=0) # T x 6890 x 3
        opt_this_take_smplh_faces = np.stack(opt_this_take_smplh_faces, axis=0) #  T x 13776 x 3
        opt_this_take_smplh_local_aa_rep = np.stack(opt_this_take_smplh_local_aa_rep, axis=0) # T x 52 x 3
        opt_this_take_smplh_root_trans = np.stack(opt_this_take_smplh_root_trans, axis=0) # T x  3
        opt_this_take_smplh_root_rot = np.stack(opt_this_take_smplh_root_rot, axis=0) # T x 3

        if VIS_W_MUJOCO:
            mujoco_assets_file = osp.join(CFG.mujoco.mujoco_assets.root_path, f"{CFG.mujoco.mujoco_assets.model_id}.xml")
            smpl_m_viewer = SMPL_M_Viewer(model_file=mujoco_assets_file)
            smpl_local_aa_pose = opt_this_take_smplh_local_aa_rep[:, :SMPLH_BODY_JNT_NUM-2, :].reshape(T, -1) # T x (22*3)
            smpl_local_aa_pose = np.concatenate([smpl_local_aa_pose, np.zeros((T, 2 * 3))]) # T x (24*3)
            smpl_global_trans = opt_this_take_smplh_root_trans.reshape(T, -1) # T x 3
            
            smpl_m_viewer.set_smpl_pose(pose=smpl_local_aa_pose, trans=smpl_global_trans, offset_z=0)
            smpl_m_viewer.show_pose(return_img=False, loop=False, size=(1920, 1080))

        return opt_this_take_smplh_anno_3d, opt_this_take_smplh_verts, opt_this_take_smplh_faces, opt_this_take_smplh_local_aa_rep, opt_this_take_smplh_root_trans, opt_this_take_smplh_root_rot
            
    def convert_to_smplh_convention(self, this_take_anno_3d, this_take_anno_3d_valid_flag):
        """
        Converts 3D joint annotations into the SMPL-H convention, which is suitable for human body models that
        include hand and body joints. This function adjusts the annotations to align with the joint indices used in
        the SMPL-H model.

        Parameters
        ----------
        this_take_anno_3d : ndarray
            The input 3D annotation array with shape (T, J, D), where T is the number of frames,
            J is the number of joints (body and hand joints combined), and D is the dimensionality (typically 3D).

        this_take_anno_3d_valid_flag : ndarray
            A boolean array with shape (T, J) indicating the validity of each joint's annotation per frame.

        Returns
        -------
        tuple
            A tuple containing two ndarrays:
            - The first ndarray is the converted 3D annotations with shape (T, K, D), where K is the number of SMPL-H joints.
            - The second ndarray is a boolean array with shape (T, K) representing the validity of each converted joint's annotation.

        Examples
        --------
        >>> this_take_anno_3d = np.random.rand(100, 42+17, 3)
        >>> this_take_anno_3d_valid_flag = np.random.rand(100, 42+17) > 0.3
        >>> smplh_anno, smplh_valid_flags = convert_to_smplh_convention(this_take_anno_3d, this_take_anno_3d_valid_flag)
        >>> print(smplh_anno.shape, smplh_valid_flags.shape)
        (100, 52, 3), (100, 52)
        
        """
        T, J, D = this_take_anno_3d.shape

        assert J == len(BODY_JOINTS) + len(HAND_JOINTS) or J == len(BODY_JOINTS), f"The input 3D annotation should have {len(BODY_JOINTS) + len(HAND_JOINTS)} or {len(BODY_JOINTS)} joints, instead have {J}."
        assert D == 3, "The input 3D annotation should have 3D coordinates"

        thsi_take_anno_has_hand_flag = J == len(BODY_JOINTS) + len(HAND_JOINTS)
        this_take_body_anno_3d, this_take_hand_anno_3d = this_take_anno_3d[:,:len(BODY_JOINTS)], this_take_anno_3d[:,len(BODY_JOINTS):]
        this_take_body_anno_3d_valid_flag, this_take_hand_anno_3d_valid_flag = this_take_anno_3d_valid_flag[:,:len(BODY_JOINTS)], this_take_anno_3d_valid_flag[:,len(BODY_JOINTS):]
        # T x (body_jnts), T x (hand_jnts)


        this_take_anno_has_hand_flag = J == len(BODY_JOINTS) + len(HAND_JOINTS)
        this_take_smplh_anno_3d = np.full((T, len(SMPLH_JOINT_NAMES), 3), np.nan)

        this_take_smplh_body_anno_3d, this_take_smplh_hand_anno_3d = this_take_smplh_anno_3d[:,:len(SMPLH_BODY_JOINTS)], this_take_smplh_anno_3d[:,len(SMPLH_BODY_JOINTS):]

        body_pose_valid_flag = np.where(np.asarray(EGOEXO4D_BODYPOSE_TO_SMPLH_INDICES) != -1)[0]
        this_take_smplh_body_anno_3d[:,body_pose_valid_flag,:] = this_take_body_anno_3d[:,np.asarray(EGOEXO4D_BODYPOSE_TO_SMPLH_INDICES)[body_pose_valid_flag],:]
        this_take_smplh_body_anno_3d_valid_flag = ~np.isnan(np.mean(this_take_smplh_body_anno_3d, axis=-1)) # T x (body_jnts)

        if this_take_anno_has_hand_flag:
            hand_pose_valid_flag = np.where(np.asarray(EGOEXO4D_HANDPOSE_TO_SMPLH_w_HAND_INDICES) != -1)[0]
            this_take_smplh_hand_anno_3d[:,hand_pose_valid_flag,:] = this_take_hand_anno_3d[:,np.asarray(EGOEXO4D_HANDPOSE_TO_SMPLH_w_HAND_INDICES)[hand_pose_valid_flag],:]
            this_take_smplh_hand_anno_3d_valid_flag = ~np.isnan(np.mean(this_take_smplh_hand_anno_3d, axis=-1)) # T x (hand_jnts)
        else:
            this_take_smplh_hand_anno_3d_valid_flag = np.full((T, len(SMPLH_HAND_JOINTS)), False)


        this_take_smplh_anno_3d_valid_flag = np.concatenate([this_take_smplh_body_anno_3d_valid_flag, this_take_smplh_hand_anno_3d_valid_flag], axis=1)

        return this_take_smplh_anno_3d, this_take_smplh_anno_3d_valid_flag
