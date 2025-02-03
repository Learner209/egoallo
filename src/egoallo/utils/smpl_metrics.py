import warnings
from collections import defaultdict

import numpy as np
import torch

from egoallo.config import CONFIG_FILE, make_cfg
from egoallo.smpl.smplh_utils import (
    SMPLH_HEAD_IDX,
    SMPLH_LEFT_ANKLE_IDX,
    SMPLH_LEFT_FOOT_IDX,
    SMPLH_RIGHT_ANKLE_IDX,
    SMPLH_RIGHT_FOOT_IDX,
)
from egoallo.utils.setup_logger import setup_logger
from egoallo.utils.transformation import (
    euler_from_quat,
    get_qvel_fd,
    qpose_to_T,
    quat_to_rotMat_t,
)

warnings.simplefilter(action="ignore", category=FutureWarning)
logger = setup_logger(output=None, name=__name__)
local_config_file = CONFIG_FILE
CFG = make_cfg(config_name="defaults", config_file=local_config_file, cli_args=[])


def smpl_mat_to_aa(poses):
    poses_aa = []
    for pose_frame in poses:
        pose_frames = []
        for joint in pose_frame:
            pose_frames.append(cv2.Rodrigues(joint)[0].flatten())
        pose_frames = np.array(pose_frames)
        poses_aa.append(pose_frames)
    poses_aa = np.array(poses_aa)
    return poses_aa


def get_root_angles(poses):
    root_angs = []
    for pose in poses:
        root_euler = np.array(euler_from_quat(pose[3:7]))
        root_angs.append(root_euler)

    return np.array(root_angs)


def get_root_matrix(poses):
    """
    Parameters
    ----------
    poses : np.ndarray of shape (T, 7)

    Returns
    -------
    T : np.ndarray of shape (T, 4, 4)
    """
    matrices = []
    for pose in poses:
        mat = np.identity(4)
        root_pos = pose[:3]
        root_quat = pose[3:7]
        mat = quat_to_rotMat_t(root_quat)
        mat[:3, 3] = root_pos
        matrices.append(mat)
    return matrices


def get_joint_vels(poses, dt):
    vels = []
    for i in range(poses.shape[0] - 1):
        v = get_qvel_fd(poses[i], poses[i + 1], dt, "heading")
        vels.append(v)
    vels = np.vstack(vels)
    return vels


def get_joint_accels(vels, dt):
    accels = np.diff(vels, axis=0) / dt
    accels = np.vstack(accels)
    return accels


def get_root_pos(poses):
    return poses[:, :3]


def get_mean_dist(x, y):
    return np.linalg.norm(x - y, axis=1).mean()


def get_mean_abs(x):
    return np.abs(x).mean()


def get_frobenious_norm(x, y):
    """Compute the frobenious norm between two matrices.

    Parameters
    ----------
    x : np.ndarray of shape (N, 4, 4)
    y : np.ndarray of shape (N, 4, 4)

    Returns
    -------
    res : float, the average frobenious norm between the matrices.
    """
    error = 0.0
    for i in range(len(x)):
        x_mat = x[i]
        y_mat_inv = np.linalg.inv(y[i])
        error_mat = np.matmul(x_mat, y_mat_inv)
        ident_mat = np.identity(4)
        error += np.linalg.norm(ident_mat - error_mat, "fro")
    return error / len(x)


def get_frobenious_norm_rot_only(x, y):
    """Compute the frobenious norm between two rotMats.

    Parameters
    ----------
    x : np.ndarray of shape (N, 4, 4)
    y : np.ndarray of shape (N, 4, 4)

    Returns
    -------
    res : float, the average frobenious norm between the matrices.
    """
    error = 0.0
    for i in range(len(x)):
        x_mat = x[i][:3, :3]
        y_mat_inv = np.linalg.inv(y[i][:3, :3])
        error_mat = np.matmul(x_mat, y_mat_inv)
        ident_mat = np.identity(3)
        error += np.linalg.norm(ident_mat - error_mat, "fro")
    return error / len(x)


def compute_accel(joints):
    """
    Computes acceleration of 3D joints.

    Args:
        joints (Nx25x3).

    Returns:
        Accelerations (N-2).
    """
    velocities = joints[1:] - joints[:-1]
    acceleration = velocities[1:] - velocities[:-1]
    acceleration_normed = np.linalg.norm(acceleration, axis=2)
    return np.mean(acceleration_normed, axis=1)


def compute_error_accel(joints_gt, joints_pred, vis=None):
    """
    Computes acceleration error:
        1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.

    Args:
        joints_gt : np.ndarray of shape N x J x 3.
        joints_pred : np.ndarray of shape N x J x 3.
        vis : np.ndarray of shape (N,), optional.

    Returns:
        error_accel (N-2).
    """
    # (N-2)x14x3
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

    normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

    if vis is None:
        new_vis = np.ones(len(normed), dtype=bool)
    else:
        invis = np.logical_not(vis)
        invis1 = np.roll(invis, -1)
        invis2 = np.roll(invis, -2)
        new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
        new_vis = np.logical_not(new_invis)

    return np.mean(normed[new_vis], axis=1)


def compute_vel(joints):
    velocities = joints[1:] - joints[:-1]
    velocity_normed = np.linalg.norm(velocities, axis=2)
    return np.mean(velocity_normed, axis=1)


def compute_error_vel(joints_gt, joints_pred, vis=None):
    vel_gt = joints_gt[1:] - joints_gt[:-1]
    vel_pred = joints_pred[1:] - joints_pred[:-1]
    normed = np.linalg.norm(vel_pred - vel_gt, axis=2)

    if vis is None:
        new_vis = np.ones(len(normed), dtype=bool)
    return np.mean(normed[new_vis], axis=1)


def compute_head_pose_metrics(head_trans, head_rot, gt_head_trans, gt_head_rot):
    """
    Parameters
    ----------
    head_trans : np.ndarray of shape (T, 3)
    head_rot : np.ndarray of shape (T, 3, 3)
    gt_head_trans : np.ndarray of shape (T, 3)
    gt_head_rot : np.ndarray of shape (T, 3, 3)
    """
    T = head_trans.shape[0]
    pred_head_mat = np.zeros((T, 4, 4))
    gt_head_mat = np.zeros((T, 4, 4))
    pred_head_mat[:, :3, :3] = head_rot
    pred_head_mat[:, 3, 3] = 1.0
    gt_head_mat[:, :3, :3] = gt_head_rot
    gt_head_mat[:, 3, 3] = 1.0
    pred_head_mat[:, :3, 3] = head_trans
    gt_head_mat[:, :3, 3] = gt_head_trans

    head_dist = get_frobenious_norm(pred_head_mat, gt_head_mat)

    head_dist_rot_only = get_frobenious_norm_rot_only(head_rot, gt_head_rot)
    head_trans_err = np.linalg.norm(head_trans - gt_head_trans, axis=1).mean() * 1000

    return head_dist, head_dist_rot_only, head_trans_err


def compute_foot_sliding_for_smpl(pred_global_jpos, floor_height):
    """
    Compute foot sliding error for SMPL.

    Parameters
    ----------
    pred_global_jpos : np.ndarray of shape (T, J, 3)
    floor_height : float

    Returns
    -------
    float

    Notes
    -----
    The foot sliding error is computed as the average of the following four metrics:
    - Left ankle sliding
    - Left foot sliding
    - Right ankle sliding
    - Right foot sliding

    Each sliding is computed as follows: first sample a subset where the foot/ankle is below a certain threshold.
    Then  in the subset: compute the displacement of the foot/ankle in the xy-plane and multiply it by a factor that depends on the height of the foot/ankle.
    The factor is computed as 2 - 2 ** (height / threshold).
    """
    T, J, *_ = pred_global_jpos.shape

    # Put human mesh to floor z = 0 and compute.
    pred_global_jpos[:, :, 2] = pred_global_jpos[:, :, 2] - floor_height

    lankle_pos = pred_global_jpos[:, SMPLH_LEFT_ANKLE_IDX, :]  # T X 3
    lfoot_pos = pred_global_jpos[:, SMPLH_LEFT_FOOT_IDX, :]  # T X 3

    rankle_pos = pred_global_jpos[:, SMPLH_RIGHT_ANKLE_IDX, :]  # T X 3
    rfoot_pos = pred_global_jpos[:, SMPLH_RIGHT_FOOT_IDX, :]  # T X 3

    H_ankle = CFG.empirical_val.metric.foot_sliding.ankle_height_threshold
    H_toe = CFG.empirical_val.metric.foot_sliding.toe_height_threshold

    lankle_disp = np.linalg.norm(
        lankle_pos[1:, :2] - lankle_pos[:-1, :2], axis=1
    )  # T-1
    lfoot_disp = np.linalg.norm(lfoot_pos[1:, :2] - lfoot_pos[:-1, :2], axis=1)  # T-1
    rankle_disp = np.linalg.norm(
        rankle_pos[1:, :2] - rankle_pos[:-1, :2], axis=1
    )  # T-1
    rfoot_disp = np.linalg.norm(rfoot_pos[1:, :2] - rfoot_pos[:-1, :2], axis=1)  # T-1

    lankle_subset = lankle_pos[:-1, -1] < H_ankle
    lfoot_subset = lfoot_pos[:-1, -1] < H_toe
    rankle_subset = rankle_pos[:-1, -1] < H_ankle
    rfoot_subset = rfoot_pos[:-1, -1] < H_toe

    lankle_sliding_stats = np.abs(
        lankle_disp * (2 - 2 ** (lankle_pos[:-1, -1] / H_ankle))
    )[lankle_subset]
    lankle_sliding = np.sum(lankle_sliding_stats) / T * 1000

    lfoot_sliding_stats = np.abs(lfoot_disp * (2 - 2 ** (lfoot_pos[:-1, -1] / H_toe)))[
        lfoot_subset
    ]
    lfoot_sliding = np.sum(lfoot_sliding_stats) / T * 1000

    rankle_sliding_stats = np.abs(
        rankle_disp * (2 - 2 ** (rankle_pos[:-1, -1] / H_ankle))
    )[rankle_subset]
    rankle_sliding = np.sum(rankle_sliding_stats) / T * 1000

    rfoot_sliding_stats = np.abs(rfoot_disp * (2 - 2 ** (rfoot_pos[:-1, -1] / H_toe)))[
        rfoot_subset
    ]
    rfoot_sliding = np.sum(rfoot_sliding_stats) / T * 1000

    sliding = (lankle_sliding + lfoot_sliding + rankle_sliding + rfoot_sliding) / 4.0

    return sliding


def compute_metrics_for_smpl(
    gt_global_quat,
    gt_global_jpos,
    gt_floor_height,
    pred_global_quat,
    pred_global_jpos,
    pred_floor_height,
):
    """Compute metrics for SMPL.

    Parameters
    ----------
    gt_global_quat : tensor of shape (T, J, 4)
    gt_global_jpos : tensor of shape (T, J, 3)
    gt_floor_height : float
    pred_global_quat : tensor of shape (T, J, 4)
    pred_global_jpos : tensor of shape (T, J, 3)
    pred_floor_height : float

    Returns
    -------
    dict : keys are listed as follows:

    - root_dist : the frobenious norm between `T_root_pred` and `T_root_gt`.
    - root_rot_dist : the frobenious norm between `T_root_pred` and `T_root_gt`, only for the rotation part.
    - accel_pred : the average acceleration of the predicted joints.
    - accel_gt : the average acceleration of the ground truth joints.
    - accel_err : the average acceleration error.
    - pred_fs : the average foot sliding error for the predicted joints.
    - gt_fs : the average foot sliding error for the ground truth joints.
    - head_trans_dist : the mean translation error for the head joint.
    - mpjpe : the mean per joint position error.
    - mpjpe_wo_hand : the mean per joint position error without the hand joints.
    - head_dist : the frobenious norm between `T_head_pred` and `T_head_gt`.
    - head_rot_dist : the frobenious norm between `T_head_pred` and `T_head_gt`, only for the rotation part.
    - single_jpe : the mean per joint position error for each joint.
    - jpe_{idx} : the mean per joint position error for joint idx.
    """

    res_dict = defaultdict(list)

    root_idx = 0
    T, J, *_ = gt_global_quat.shape

    root_traj_pred = (
        torch.cat(
            (pred_global_jpos[:, root_idx, :], pred_global_quat[:, root_idx, :]), dim=-1
        )
        .data.cpu()
        .numpy()
    )  # T X 7
    root_traj_gt = (
        torch.cat(
            (gt_global_jpos[:, root_idx, :], gt_global_quat[:, root_idx, :]), dim=-1
        )
        .data.cpu()
        .numpy()
    )  # T X 7

    T_root_pred = qpose_to_T(root_traj_pred)  # T x 3 x 4
    T_root_pred = np.concatenate(
        [T_root_pred, np.zeros((T, 1, 4))], axis=1
    )  # T x 4 x 4
    T_root_gt = qpose_to_T(root_traj_gt)  # T x 3 x 4
    T_root_gt = np.concatenate([T_root_gt, np.zeros((T, 1, 4))], axis=1)  # T x 4 x 4

    # compute frobenious norm for T_root_pred and T_root_gt
    root_dist = get_frobenious_norm(T_root_pred, T_root_gt)
    root_rot_dist = get_frobenious_norm_rot_only(T_root_pred, T_root_gt)

    head_idx = SMPLH_HEAD_IDX
    head_traj_pred = (
        torch.cat(
            (pred_global_jpos[:, head_idx, :], pred_global_quat[:, head_idx, :]), dim=-1
        )
        .data.cpu()
        .numpy()
    )
    head_traj_gt = (
        torch.cat(
            (gt_global_jpos[:, head_idx, :], gt_global_quat[:, head_idx, :]), dim=-1
        )
        .data.cpu()
        .numpy()
    )

    T_head_pred = qpose_to_T(head_traj_pred)  # T x 3 x 4
    T_head_pred = np.concatenate(
        [T_head_pred, np.zeros((T, 1, 4))], axis=1
    )  # T x 4 x 4
    T_head_gt = qpose_to_T(head_traj_gt)  # T x 3 x 4
    T_head_gt = np.concatenate([T_head_gt, np.zeros((T, 1, 4))], axis=1)  # T x 4 x 4

    # compute frobenious norm for T_head_pred and T_head_gt
    head_dist = get_frobenious_norm(T_head_pred, T_head_gt)  # scalar
    head_rot_dist = get_frobenious_norm_rot_only(T_head_pred, T_head_gt)  # scalar

    # Compute accl and accl err.
    accels_pred = (
        np.mean(compute_accel(pred_global_jpos.data.cpu().numpy())) * 1000
    )  # scalar
    accels_gt = (
        np.mean(compute_accel(gt_global_jpos.data.cpu().numpy())) * 1000
    )  # scalar

    accel_dist = (
        np.mean(
            compute_error_accel(
                pred_global_jpos.data.cpu().numpy(), gt_global_jpos.data.cpu().numpy()
            )
        )
        * 1000
    )  # scalar

    # Compute foot sliding error
    pred_fs_metric = compute_foot_sliding_for_smpl(
        pred_global_jpos.data.cpu().numpy().copy(), pred_floor_height
    )
    gt_fs_metric = compute_foot_sliding_for_smpl(
        gt_global_jpos.data.cpu().numpy().copy(), gt_floor_height
    )

    jpos_pred = pred_global_jpos - pred_global_jpos[:, 0:1]  # T x J x 3 zero out root
    jpos_gt = gt_global_jpos - gt_global_jpos[:, 0:1]  # T x J x 3
    jpos_pred = jpos_pred.data.cpu().numpy()
    jpos_gt = jpos_gt.data.cpu().numpy()
    mpjpe = np.linalg.norm(jpos_pred - jpos_gt, axis=2).mean() * 1000

    # Add jpe for each joint
    single_jpe = np.linalg.norm(jpos_pred - jpos_gt, axis=2).mean(axis=0) * 1000  # J

    # Remove joints 18, 19, 20, 21
    mpjpe_wo_hand = single_jpe[:18].mean()

    # Jiaman: add root translation error
    pred_root_trans = root_traj_pred[:, :3]  # T X 3
    gt_root_trans = root_traj_gt[:, :3]  # T X 3
    root_trans_err = (
        np.linalg.norm(pred_root_trans - gt_root_trans, axis=1).mean() * 1000
    )
    res_dict["root_trans_dist"].append(root_trans_err)

    # Add accl and accer
    res_dict["accel_pred"] = accels_pred
    res_dict["accel_gt"] = accels_gt
    res_dict["accel_err"] = accel_dist

    # Add foot sliding metric
    res_dict["pred_fs"] = pred_fs_metric
    res_dict["gt_fs"] = gt_fs_metric

    pred_head_trans = head_traj_pred[:, :3]
    gt_head_trans = head_traj_gt[:, :3]
    head_trans_err = (
        np.linalg.norm(pred_head_trans - gt_head_trans, axis=1).mean() * 1000
    )
    res_dict["head_trans_dist"].append(head_trans_err)

    res_dict["root_dist"].append(root_dist)
    res_dict["root_rot_dist"].append(root_rot_dist)
    res_dict["mpjpe"].append(mpjpe)
    res_dict["mpjpe_wo_hand"].append(mpjpe_wo_hand)
    res_dict["head_dist"].append(head_dist)
    res_dict["head_rot_dist"].append(head_rot_dist)

    res_dict["single_jpe"].append(single_jpe)
    for tmp_idx in range(single_jpe.shape[0]):
        res_dict["jpe_" + str(tmp_idx)].append(single_jpe[tmp_idx])

    res_dict = {k: np.mean(v) for k, v in res_dict.items()}

    return res_dict


def compute_foot_sliding(foot_data, traj_qpos):
    seq_len = len(traj_qpos)
    H = 0.033
    z_threshold = 0.65
    z = traj_qpos[1:, 2]
    foot = np.array(foot_data).copy()
    foot[:, -1] = foot[:, -1] - np.mean(foot[:3, -1])  # Grounding it
    foot_disp = np.linalg.norm(foot[1:, :2] - foot[:-1, :2], axis=1)

    foot_avg = (foot[:-1, -1] + foot[1:, -1]) / 2
    subset = np.logical_and(foot_avg < H, z > z_threshold)
    # import pdb; pdb.set_trace()

    sliding_stats = np.abs(foot_disp * (2 - 2 ** (foot_avg / H)))[subset]
    sliding = np.sum(sliding_stats) / seq_len * 1000
    return sliding, sliding_stats


def norm_qpos(qpos):
    qpos_norm = qpos.copy()
    qpos_norm[:, 3:7] /= np.linalg.norm(qpos_norm[:, 3:7], axis=1)[:, None]

    return qpos_norm


def trans2velocity(root_trans):
    # root_trans: T X 3
    root_velocity = root_trans[1:] - root_trans[:-1]
    return root_velocity  # (T-1) X 3


def velocity2trans(init_root_trans, root_velocity):
    # init_root_trans: 3
    # root_velocity: (T-1) X 3

    timesteps = root_velocity.shape[0] + 1
    absolute_pose_data = np.zeros((timesteps, 3))  # T X 3
    absolute_pose_data[0, :] = init_root_trans.copy()

    root_trans = init_root_trans[np.newaxis].copy()  # 1 X 3
    for t_idx in range(1, timesteps):
        root_trans += root_velocity[t_idx - 1 : t_idx, :]  # 1 X 3
        absolute_pose_data[t_idx, :] = root_trans  # 1 X 3

    return absolute_pose_data  # T X 3
