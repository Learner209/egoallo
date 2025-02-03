"""
=====================
Coordiante transforms
=====================

Aria rgb/slam-left/slam-right cam: X down, Z in front, Y Left, thus image portrait view is horizontal.
SMPL follows the convention of X left, Y up, Z front, thus egoego follows the same convention.
OpenGL Camera follows the convention of X right, Y up, Z back.
OpenCV Camera follows the convention of X right, Y down, Z front.


"""

from pytorch3d import transforms
from scipy.spatial.transform import Rotation as sRot

from egoallo.utils.transformation import (
    aa_from_quat,
    normalize_t,
    quat_normalize_t,
    quat_inv_t,
    quat_mul_t,
    quat_mul_vec_t,
    quat_mul_vec_t_batch,
    quat_between_t,
    get_heading_q,
    quat_from_expmap_t,
    transform_vec_t,
    T_to_qpose,
    qpose_to_T,
)
import torch
import numpy as np
from egoallo.utils.setup_logger import setup_logger
from egoallo.utils.utils import NDArray
from egoallo.config import make_cfg, CONFIG_FILE

logger = setup_logger(output=None, name=__name__)


local_config_file = CONFIG_FILE
CFG = make_cfg(config_name="defaults", config_file=local_config_file, cli_args=[])


def aria_camera_device2opengl_pose(aria_pose):
    # region
    """
    Converts a pose from the Aria slam-left/slam-right/camera-rgb system to the OpenGL coordinate system.

    Parameters
    ----------
    aria_pose : numpy.ndarray
        An array of shape (T, 7) where each row represents a pose in the Aria coordinate
        system, consisting of a translation (x, y, z) and a quaternion (w, x, y, z).

    Returns
    -------
    opengl_pose : numpy.ndarray
        An array of shape (T, 7) where each row represents the pose in the OpenGL
        coordinate system, consisting of a translation (x, y, z) and a quaternion (w, x, y, z).

    Notes
    -----
    This function assumes that the input pose is given in the Aria coordinate system and
    applies a rotation to convert it into the OpenGL coordinate system.
    """
    T, *_ = aria_pose.shape
    aria2opengl = CFG.coordinate.transform.aria2opengl
    # aria_trans, aria_quat = aria_pose[:, :3], aria_pose[:, 3:]
    # aria2opengl_rot_mat = transforms.euler_angles_to_matrix(torch.tensor([0, 0, 1], dtype=torch.float32)[None],convention="XYZ")
    # aria2opengl_quat = transforms.matrix_to_quaternion(aria2opengl_rot_mat)
    # aria2opengl_quat = aria2opengl_quat.repeat(aria_quat.shape[0], 1)
    # aria_quat = quat_mul(aria_quat, aria2opengl_quat)
    T_aria_world_device = np.tile(np.eye(4), (T, 1, 1))
    T_aria_world_device[:, :3] = qpose_to_T(aria_pose)
    R_aria_world_device = T_aria_world_device[:, :3, :3]
    R_aria_world_device = R_aria_world_device @ aria2opengl
    T_opengl_world_device = T_aria_world_device.copy()
    T_opengl_world_device[:, :3, :3] = R_aria_world_device
    opengl_pose = T_to_qpose(T_opengl_world_device[:, :3])
    return opengl_pose
    # endregion


def get_head_vel(head_pose, dt=1 / 30):
    # region
    """
    Compute the head velocity (**in the head frame**) from sequential head poses.

    Parameters
    ----------
    head_pose : numpy.ndarray
        Sequence of head poses with position and orientation. Shape: (num_frames, 7)
    dt : float, optional
        Time interval between frames. Default is 1/30.

    Returns
    -------
    numpy.ndarray
        The velocities of the head for each frame. Shape: (num_frames, 6)
    """
    # get head velocity
    head_vels = []
    head_pose[0]

    for i in range(head_pose.shape[0] - 1):
        curr_qpos = head_pose[i, :]
        next_qpos = head_pose[i + 1, :]
        v = (next_qpos[:3] - curr_qpos[:3]) / dt
        # multiply world-frame diff by current head quat to get the velocity in the head frame.
        #! 'heading' option is applied, so x,y components are erased out, only consider rotation around z-axis.
        v = transform_vec_t(v.data.cpu().numpy(), curr_qpos[3:7], "heading")

        qrel = transforms.quaternion_multiply(
            next_qpos[3:7], transforms.quaternion_invert(curr_qpos[3:7])
        )
        axis, angle = rotation_from_quaternion_np(qrel, True)

        if angle > np.pi:  # -180 < angle < 180
            angle = angle - 2 * np.pi
        elif angle < -np.pi:
            angle = angle + 2 * np.pi

        rv = (axis * angle) / dt  # axis-angle repr to rotation vector repr
        rv = transform_vec_t(rv, curr_qpos[3:7], "root")

        head_vels.append(np.concatenate((v, rv)))

    head_vels.append(
        head_vels[-1].copy()
    )  # copy last one since there will be one less through finite difference
    head_vels = np.vstack(head_vels)
    return head_vels
    # endregion


def get_root_relative_head(root_poses, head_poses):
    # region
    """
    Calculate the relative pose of the root with respect to the head's pose for each pair of root and head poses.

    Parameters
    ----------
    root_poses : ndarray
        An array of root poses with shape (N, 7), where N is the number of samples. Each pose consists of 3 position values and 4 quaternion values representing rotation.
    head_poses : ndarray
        An array of head poses with the same shape and format as `root_poses`.

    Returns
    -------
    ndarray
        An array of relative poses of the root with respect to the head, with shape (N, 6), where the first three columns are the positional offsets and the last three columns are the rotational vectors in axis-angle format.
    """
    res_root_poses = []

    for idx in range(head_poses.shape[0]):
        head_qpos = head_poses[idx]
        root_qpos = root_poses[idx]

        head_pos = head_qpos[:3]
        head_rot = head_qpos[3:7]
        get_heading_q(head_rot).copy()

        root_pos = root_qpos[:3].copy()
        diff = root_pos - head_pos
        diff_loc = transform_vec_t(diff, head_rot, "heading")

        root_quat = root_qpos[3:7].copy()
        root_quat_local = transforms.quaternion_multiply(
            transforms.quaternion_invert(head_rot), root_quat
        )  # ???? Should it be flipped?
        axis, angle = aa_from_quat(root_quat_local, separate=True)

        if angle > np.pi:  # -180 < angle < 180
            angle = angle - 2 * np.pi
        elif angle < -np.pi:
            angle = angle + 2 * np.pi

        rv = axis * angle
        rv = transform_vec_t(rv, head_rot, "root")  # root 2 head diff in head's frame

        root_pose = np.concatenate((diff_loc, rv))
        res_root_poses.append(root_pose)

    res_root_poses = np.array(res_root_poses)
    return res_root_poses
    # endregion


def root_from_relative_head(root_relative, head_poses):
    # region
    """
    Compute the absolute root poses given their relative poses to the head and the absolute poses of the head.

    Parameters
    ----------
    root_relative : ndarray
        An array of root poses relative to the head, shape (N, 6), where N is the number of samples. The first three columns are positional offsets and the last three columns are rotational vectors.
    head_poses : ndarray
        An array of head poses, shape (N, 7), corresponding to the relative root poses.

    Returns
    -------
    ndarray
        An array of absolute root poses, shape (N, 7), formatted as [position (3,), quaternion (4,)] for each pose.

    Notes
    -----
    This function applies transformations based on the head's orientation to adjust the relative positions and rotations of the root poses to their absolute values.
    """
    assert root_relative.shape[0] == head_poses.shape[0]
    root_poses = []
    for idx in range(root_relative.shape[0]):
        head_pos = head_poses[idx][:3]
        head_rot = head_poses[idx][3:7]
        q_heading = get_heading_q(head_rot).copy()

        root_pos_delta = root_relative[idx][:3]
        root_rot_delta = root_relative[idx][3:]

        root_pos = quat_mul_vec_t(q_heading, root_pos_delta) + head_pos
        root_rot_delta = quat_mul_vec_t(head_rot, root_rot_delta)
        root_rot = transforms.quaternion_multiply(
            head_rot, quat_from_expmap_t(root_rot_delta)
        )
        root_pose = np.hstack([root_pos, root_rot])
        root_poses.append(root_pose)
    return np.array(root_poses)
    # endregion


def get_obj_relative_pose(obj_poses, ref_poses, num_objs=1):
    # region
    """
    Calculate the relative poses of objects with respect to a reference pose for each pair in the dataset.

    Parameters
    ----------
    obj_poses : ndarray
        An array of object poses, shape (N, 7 * num_objs), where N is the number of samples, and each object pose is represented by position (3,) and quaternion (4,).
    ref_poses : ndarray
        An array of reference poses with shape (N, 7), each consisting of 3 position values and 4 quaternion values representing rotation.
    num_objs : int, optional
        Number of objects considered in each sample, default is 1.

    Returns
    -------
    ndarray
        An array of relative poses of objects with respect to the reference poses, shape (N, num_objs * 7), formatted similarly to `obj_poses`.

    Notes
    -----
    This function computes the relative position by subtracting the reference position and then transforming this difference into the reference frame's local coordinate system.
    """
    # get object pose relative to the head
    res_obj_poses = []

    for idx in range(ref_poses.shape[0]):
        ref_qpos = ref_poses[idx]
        obj_qpos = obj_poses[idx]

        ref_pos = ref_qpos[:3]
        ref_rot = ref_qpos[3:7]
        q_heading = get_heading_q(ref_rot).copy()
        obs = []
        for oidx in range(num_objs):
            obj_pos = obj_qpos[oidx * 7 : oidx * 7 + 3].copy()
            diff = obj_pos - ref_pos
            diff_loc = transform_vec_t(diff, ref_rot, "heading")

            obj_quat = obj_qpos[oidx * 7 + 3 : oidx * 7 + 7].copy()
            obj_quat_local = transforms.quaternion_multiply(
                transforms.quaternion_invert(q_heading), obj_quat
            )
            obj_pose = np.concatenate((diff_loc, obj_quat_local))
            obs.append(obj_pose)

        res_obj_poses.append(np.concatenate(obs))

    res_obj_poses = np.array(res_obj_poses)
    return res_obj_poses
    # endregion


def rotate_at_frame_smplh(root_pose, cano_t_idx=0):
    """Rotating the transformation trajectory and the quat trajectory based on the reference frame dictated by the `cano_t_idx`
    numpy array. More specifically, let the forward direction of the `cano_t_idx` frame to be the x-axis of the SMPL frame (ignoring z axis).

    Parameters
    ------
        root pose : torch.tensor of shape BS X T X 7
        cano_t_idx : int, use which frame for forward direction canonicalization

    Returns
    ------
        new_glob_X : torch.tensor of shape BS X T X 3
        new_glob_Q : torch.tensor of shape BS X T X 4
        yrot : torch.tensor of shape BS X 4
            `yrot` is needed for visualization. `yrot` deirecly applied to canonicalized trans/rotation will recover it to original scene.

    Examples
    ------
    >>> root_pose = torch.rand(2, 10, 7)
    >>> new_glob_X, new_glob_Q, yrot = rotate_at_frame_smplh(root_pose, cano_t_idx=0)
    >>> new_glob_X.cpu().numpy().shape, new_glob_Q.cpu().numpy().shape, yrot.cpu().numpy().shape
    ((2, 10, 3), (2, 10, 4), (2, 4))

    """
    BS, T, *_ = root_pose.shape
    root_trans, root_quat = root_pose[:, :, :3], root_pose[:, :, 3:]

    global_q = root_quat[:, None, :, :]  # BS X 1 X T X 4
    global_x = root_trans[:, None, :, :]  # BS X 1 X T X 3

    key_glob_Q = global_q[:, :, cano_t_idx : cano_t_idx + 1, :]  # BS X 1 X 1 X 4

    # The floor is on z = xxx. Project the forward direction to xy plane.
    project_t = torch.FloatTensor([1, 1, 0])[None, None, None, :]
    world_forward = torch.FloatTensor([1, 0, 0])[None, None, None, :]
    loc_forward = project_t * quat_mul_vec_t_batch(
        key_glob_Q, world_forward.repeat(BS, 1, 1, 1)
    )  # BS x 1 x 1 x 3

    loc_forward = normalize_t(loc_forward)
    yrot = quat_normalize_t(
        quat_between_t(world_forward, loc_forward)
    )  # BS x 1 x 1 X 4
    new_glob_Q = quat_mul_t(quat_inv_t(yrot).expand(-1, -1, T, -1), global_q)
    new_glob_X = quat_mul_vec_t(quat_inv_t(yrot).expand(-1, -1, T, -1), global_x)

    # BS X T X 3, BS X T X 4, BS(1) X 1 X 1 X 4
    return new_glob_X[:, 0, :, :], new_glob_Q[:, 0, :, :], yrot[:, 0, 0, :]


def batch_align_to_reference_pose(to_align_pose, reference_pose):
    """Align the to_align_pose to let the first frame of `to_align_pose` to match the *first frame* of `reference_pose`

    Parameters
    ------
    to_align_pose : numpy array, B x T x 7
    reference_pose : numpy array, T x 7

    Returns
    ------
    aligned_seq_trans : numpy array, B x T x 3
    aligned_seq_rot_mat : numpy array, B x T x 3 x 3
    aligned_seq_rot_quat_wxyz : numpy array, B x T x 4
    to_align2ref_rot_seq : numpy array, B x 3 x 3
    move_to_ref_trans : numpy array, B x 3

    Notes
    ------
    - the convention of quat is `wxyz` since pytorch3d uses this convention.
    - `to_align2ref_rot_seq` is the rotation matrix that rotates the first frame of `to_align_pose` to the first frame of `reference_pose`
    - `move_to_ref_trans` is the translation that moves the first frame of `to_align_pose` to the first frame of `reference_pose`, applied by order:
    .. math:: P_{\text{aligned}} = R_{to\_align2ref}P_{\text{to\_align}} - T_{move\_to\_ref\_}

    Examples
    ------
    >>> to_align_pose = np.random.rand(2, 10, 7)
    >>> reference_pose = np.random.rand(10, 7)
    >>> aligned_seq_trans, aligned_seq_rot_mat, aligned_seq_rot_quat, toalign2ref_rot_seq, move_to_ref_trans = align_to_reference_pose(to_align_pose, reference_pose)
    >>> aligned_seq_trans.shape, aligned_seq_rot_mat.shape, aligned_seq_rot_quat.shape, toalign2ref_rot_seq.shape, move_to_ref_trans.shape
    ((2, 10, 3), (2, 10, 3, 3), (2, 10, 4), (2, 3, 3), (2, 3))

    """

    B, T, _ = to_align_pose.shape
    to_align_trans, to_align_quat_wxyz = (
        to_align_pose[:, :, :3],
        to_align_pose[:, :, 3:],
    )  # B x T x 3, B x T x 4
    to_align_rot_mat = (
        transforms.quaternion_to_matrix(torch.from_numpy(to_align_quat_wxyz).float())
        .data.cpu()
        .numpy()
    )  # B x T x 3 x 3

    ref_trans = reference_pose[:, :3]  # T X 3
    ref_quat = reference_pose[:, 3:]  # T X 4
    ref_rot_mat = (
        transforms.quaternion_to_matrix(torch.from_numpy(ref_quat).float())
        .data.cpu()
        .numpy()
    )

    to_align2ref_rot = np.matmul(
        ref_rot_mat[0:1], to_align_rot_mat[:, 0].transpose(0, 2, 1)
    )  # B x 3 X 3
    seq_to_align_rot_mat = torch.from_numpy(to_align_rot_mat).float()  # B x T X 3 X 3
    to_align2ref_rot_seq = torch.from_numpy(to_align2ref_rot)[
        :, np.newaxis, :, :
    ].float()  # B x 1 X 3 X 3
    aligned_seq_rot_mat = torch.matmul(
        to_align2ref_rot_seq, seq_to_align_rot_mat
    )  # B x T X 3 X 3
    aligned_seq_rot_quat_wxyz = transforms.matrix_to_quaternion(
        aligned_seq_rot_mat
    )  # B x T X 4

    aligned_seq_rot_mat = aligned_seq_rot_mat.data.cpu().numpy()
    aligned_seq_rot_quat_wxyz = aligned_seq_rot_quat_wxyz.data.cpu().numpy()

    seq_to_align_trans = torch.from_numpy(to_align_trans).float()[
        ..., None
    ]  # B x T X 3 X 1
    aligned_seq_trans = torch.matmul(to_align2ref_rot_seq, seq_to_align_trans)[
        ..., 0
    ]  # B x T X 3
    aligned_seq_trans = aligned_seq_trans.data.cpu().numpy()

    # Make initial x,y,z aligned
    move_to_ref_trans = ref_trans[0:1, :] - aligned_seq_trans[:, 0:1, :]  # B x 1 x 3
    aligned_seq_trans = aligned_seq_trans + move_to_ref_trans  # B x T x 3

    to_align2ref_rot_seq = to_align2ref_rot_seq.data.cpu().numpy()

    return (
        aligned_seq_trans,
        aligned_seq_rot_mat,
        aligned_seq_rot_quat_wxyz,
        to_align2ref_rot_seq[:, 0],
        move_to_ref_trans[:, 0],
    )


def align_to_reference_pose(to_align_pose, reference_pose):
    """
     Align the to_align_pose to let the first frame of to_align_pose to match the first frame of reference_pose
    NOTE: the convention of quat is `wxyz` since pytorch3d uses this convention.
    Parameters:
    to_align_pose: numpy array, T x 7, the convention of quat is wxyz
    reference_pose: numpy array, T x 7, the convention of quat is wxyz
    """
    to_align_trans, to_align_quat_wxyz = to_align_pose[:, :3], to_align_pose[:, 3:]
    to_align_rot_mat = (
        transforms.quaternion_to_matrix(torch.from_numpy(to_align_quat_wxyz).float())
        .data.cpu()
        .numpy()
    )

    ref_trans = reference_pose[:, :3]  # T X 3
    ref_quat = reference_pose[:, 3:]  # T X 4
    ref_rot_mat = (
        transforms.quaternion_to_matrix(torch.from_numpy(ref_quat).float())
        .data.cpu()
        .numpy()
    )

    to_align2ref_rot = np.matmul(ref_rot_mat[0], to_align_rot_mat[0].T)  # 3 X 3
    # print("pred2gt_rot:{0}".format(pred2gt_rot))
    seq_to_align_rot_mat = torch.from_numpy(to_align_rot_mat).float()  # T X 3 X 3
    to_align2ref_rot_seq = torch.from_numpy(to_align2ref_rot).float()[
        None, :, :
    ]  # 1 X 3 X 3
    aligned_seq_rot_mat = torch.matmul(
        to_align2ref_rot_seq, seq_to_align_rot_mat
    )  # T X 3 X 3
    aligned_seq_rot_quat_wxyz = transforms.matrix_to_quaternion(aligned_seq_rot_mat)

    aligned_seq_rot_mat = aligned_seq_rot_mat.data.cpu().numpy()
    aligned_seq_rot_quat_wxyz = aligned_seq_rot_quat_wxyz.data.cpu().numpy()

    seq_to_align_trans = torch.from_numpy(to_align_trans).float()[
        :, :, None
    ]  # T X 3 X 1
    aligned_seq_trans = torch.matmul(to_align2ref_rot_seq, seq_to_align_trans)[
        :, :, 0
    ]  # T X 3
    aligned_seq_trans = aligned_seq_trans.data.cpu().numpy()

    # Make initial x,y,z aligned
    move_to_gt_trans = ref_trans[0:1, :] - aligned_seq_trans[0:1, :]
    aligned_seq_trans = aligned_seq_trans + move_to_gt_trans

    return aligned_seq_trans, aligned_seq_rot_mat, aligned_seq_rot_quat_wxyz


def lookAt(eye, target, up):
    """
    Compute a view matrix looking from `eye` towards `target` with the specified `up` vector.

    Parameters
    ----------
    eye : ndarray
        The position of the camera in world space. Must be a 1D array of shape (3,).
    target : ndarray
        The point in world space where the camera is looking. Must be a 1D array of shape (3,).
    up : ndarray
        The up direction for the camera in world space. Must be a 1D array of shape (3,).

    Returns
    -------
    view_matrix : ndarray
        A 4x4 view matrix as a NumPy array.

    Notes
    -----
    - Positive Z-axis points out of the screen (towards the viewer in default orientation).
    - Negative Z-axis points into the screen (where the camera is looking towards).
    - Positive X-axis is to the right.
    - Positive Y-axis is upwards.

    Examples
    --------
    >>> eye = np.array([1.0, 1.0, 1.0])
    >>> target = np.array([0.0, 0.0, 0.0])
    >>> up = np.array([0.0, 1.0, 0.0])
    >>> view_matrix = lookAt(eye, target, up)
    >>> np.linalg.inv(view_matrix).shape
    (4, 4)
    """
    direction = eye - target
    if direction.any():
        direction /= np.linalg.norm(direction)

    right = np.cross(up, direction)
    if right.any():
        right /= np.linalg.norm(right)

    up = np.cross(direction, right)
    if up.any():
        up /= np.linalg.norm(up)

    R = np.array([right, up, -direction])

    T = np.eye(4)
    T[0:3, 3] = -eye

    M = np.eye(4)
    M[:3, :3] = R.T
    view_matrix = M @ T

    return view_matrix


def batchOpenGLlookAt(eye, target, up):
    """
    Parameters
    ----------
    eye : ndarray of shape (BS, 3)
        The position of the camera in world space.
    target : ndarray of shape (BS, 3)
        The point in world space where the camera is looking.
    up : ndarray of shape (BS, 3)
        The up direction for the camera in world space.

    Returns
    -------
    view_matrix : ndarray of shape (BS, 4, 4)

    Examples
    --------
    >>> eye = np.random.rand(45, 3)
    >>> target = np.random.rand(45, 3)
    >>> up = np.random.rand(45, 3)
    >>> view_matrix = batchOpenGLlookAt(eye, target, up)
    >>> view_matrix.shape
    (45, 4, 4)
    """
    BS, *_ = eye.shape
    direction = eye - target  # BS x 3
    if direction.any():
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True)

    right = np.cross(up, direction)  # BS x 3
    if right.any():
        right /= np.linalg.norm(right, axis=-1, keepdims=True)

    up = np.cross(direction, right)  # BS x 3
    if up.any():
        up /= np.linalg.norm(up, axis=-1, keepdims=True)

    R = np.stack([right, up, direction], axis=1)  # BS x 3 x 3

    T = np.tile(np.eye(4), (BS, 1, 1))  # BS x 4 x 4
    T[:, 0:3, 3] = -eye

    M = np.tile(np.eye(4), (BS, 1, 1))  # BS x 4 x 4
    M[:, :3, :3] = R
    view_matrix = M @ T

    return view_matrix


def OpenGLlookAt(eye, target, up):
    """
    Compute a view matrix looking from `eye` towards `target` with the specified `up` vector.

    Parameters
    ----------
    eye : ndarray of dtype float64
        The position of the camera in world space. Must be a 1D array of shape (3,).
    target : ndarray of dtype float64
        The point in world space where the camera is looking. Must be a 1D array of shape (3,).
    up : ndarray
        The up direction for the camera in world space. Must be a 1D array of shape (3,).

    Returns
    -------
    view_matrix (T_cam_world) : ndarray
        A 4x4 view matrix as a NumPy array.

    Notes
    -----
    This function creates a view matrix for a right-handed coordinate system with a direct
    analogy to the traditional OpenGL `lookAt` function.

    Examples
    --------
    >>> eye = np.array([1.0, 1.0, 1.0])
    >>> target = np.array([0.0, 0.0, 0.0])
    >>> up = np.array([0.0, 1.0, 0.0])
    >>> view_matrix = lookAt(eye, target, up)
    >>> np.linalg.inv(view_matrix).shape
    (4, 4)

    """
    direction = eye - target
    if direction.any():
        direction /= np.linalg.norm(direction)

    right = np.cross(up, direction)
    if right.any():
        right /= np.linalg.norm(right)

    up = np.cross(direction, right)
    if up.any():
        up /= np.linalg.norm(up)

    R = np.array([right, up, direction])

    T = np.eye(4)
    T[0:3, 3] = -eye

    M = np.eye(4)
    M[:3, :3] = R
    view_matrix = M @ T

    return view_matrix


def openglpose2smplorigin(openglpose):
    """
    Parameters
    ----------
    openglpose : numpy.ndarray of shape (T, 7)
        The pose in OpenGL coordinate system.

    Returns
    -------
    T_smpl_world_cam : numpy.ndarray of shape (T, 4, 4)
        The pose in SMPL coordinate system.
    R_smpl_world_cam_as_euler : numpy.ndarray of shape (T, 3)
        The pose in SMPL coordinate system as Euler angles.
    """
    T, *_ = openglpose.shape
    T_opengl_world_cam = qpose_to_T(openglpose)
    R_opengl_world_cam = T_opengl_world_cam[:, :3, :3]
    opengl2smpl = np.asarray(CFG.coordinate.transform.opengl2smpl)
    R_smpl_world_cam = R_opengl_world_cam @ opengl2smpl
    R_smpl_world_cam_ = sRot.from_matrix(R_smpl_world_cam)
    R_smpl_world_cam_as_quat = R_smpl_world_cam_.as_quat()  # xyzw
    R_smpl_world_cam_as_quat = R_smpl_world_cam_as_quat[:, [3, 0, 1, 2]]  # wxyz
    R_smpl_world_cam_as_euler = R_smpl_world_cam_.as_euler(seq="xyz", degrees=False)
    R_smpl_world_cam_as_rotvec = R_smpl_world_cam_.as_rotvec()
    T_smpl_world_cam = T_opengl_world_cam.copy()
    T_smpl_world_cam[:, :3, :3] = R_smpl_world_cam
    return T_smpl_world_cam, R_smpl_world_cam_as_euler, R_smpl_world_cam_as_rotvec


def opengl_pts_2_smpl_pts(opengl_pts: NDArray):
    """
    Parameters
    ----------
    opengl_pts : numpy.ndarray of shape (T, 3)

    Returns
    -------
    smpl_pts : ndarray of shape (T, 3)
    """

    T, *_ = opengl_pts.shape
    opengl2smpl = np.asarray(CFG.coordinate.transform.opengl2smpl)
    smpl_pts = opengl2smpl @ opengl_pts.T
    smpl_pts = smpl_pts.T
    return smpl_pts


def rospose2smplorigin(rospose: NDArray):
    """
    Parameters
    ----------
    rospose : numpy.ndarray of shape (T, 7)
        The pose in ROS coordinate system.

    Returns
    -------
    T_smpl_world_cam : numpy.ndarray of shape (T, 4, 4)
        The pose in SMPL coordinate system.
    R_smpl_world_cam_as_euler : numpy.ndarray of shape (T, 3)
        The pose in SMPL coordinate system as Euler angles.
    """
    T, *_ = rospose.shape
    T_ros_world_cam = qpose_to_T(rospose)
    R_ros_world_cam = T_ros_world_cam[:, :3, :3]
    ros2smpl = np.asarray(CFG.coordinate.transform.ros2smpl)
    R_smpl_world_cam = R_ros_world_cam @ ros2smpl
    R_smpl_world_cam_ = sRot.from_matrix(R_smpl_world_cam)
    R_smpl_world_cam_as_quat = R_smpl_world_cam_.as_quat()  # xyzw
    R_smpl_world_cam_as_quat = R_smpl_world_cam_as_quat[:, [3, 0, 1, 2]]  # wxyz
    R_smpl_world_cam_as_euler = R_smpl_world_cam_.as_euler(seq="xyz", degrees=False)
    R_smpl_world_cam_as_rotvec = R_smpl_world_cam_.as_rotvec()
    T_smpl_world_cam = T_ros_world_cam.copy()
    T_smpl_world_cam[:, :3, :3] = R_smpl_world_cam
    return T_smpl_world_cam, R_smpl_world_cam_as_euler, R_smpl_world_cam_as_rotvec


def ros_pts_2_smpl_pts(ros_pts: NDArray):
    """
    convert the lookAt direction in any ros cooridnate system to the origin of SMPL system (x->left,y->up,z->forward).

    Parameters
    ----------
    batch_lookAt_direction : numpy.ndarray of shape (BS, 3)
        The lookAt direction in Any ros coordinate system.

    Returns
    -------
    batch_R_world_cam_as_quat : numpy.ndarray of shape (BS, 4), `wxyz`
    batch_R_world_cam_as_euler : numpy.ndarray of shape (BS, 3)

    Notes
    -----
    Assume the Zup coordinate system aligns with ROS system (x->forward, y->left, z->up).
    """
    T, *_ = ros_pts.shape
    ros2smpl = np.asarray(CFG.coordinate.transform.ros2smpl)
    smpl_pts = ros2smpl @ ros_pts.T
    smpl_pts = smpl_pts.T
    return smpl_pts


def batch_ZupLookAT2smplorigin(batch_lookAt_direction, euler_order="xyz"):
    """
    convert the lookAt direction in any Z-up cooridnate system to the origin of SMPL system (x->left,y->up,z->forward).

    Parameters
    ----------
    batch_lookAt_direction : numpy.ndarray of shape (BS, 3)
        The lookAt direction in Any Z-up coordinate system.

    Returns
    -------
    batch_R_world_cam_as_quat : numpy.ndarray of shape (BS, 4), `wxyz`
    batch_R_world_cam_as_euler : numpy.ndarray of shape (BS, 3)

    Examples
    --------
    >>> batch_lookAt_direction = np.random.rand(45, 3)
    >>> batch_R_world_cam_as_quat, batch_R_world_cam_as_euler = batch_ZupLookAT2smplorigin(batch_lookAt_direction)
    >>> batch_R_world_cam_as_quat.shape, batch_R_world_cam_as_euler.shape
    ((45, 4), (45, 3))

    Notes
    -----
    No **assumption** is made on the Zup coordinate system.
    """
    T, *_ = batch_lookAt_direction.shape
    camera_eye = np.zeros((T, 3))
    camera_target = camera_eye + batch_lookAt_direction
    camera_up = np.tile(np.array([0, 0, 1]), (T, 1))
    batch_T_cam_world = batchOpenGLlookAt(camera_eye, camera_target, camera_up)
    batch_T_world_cam = np.linalg.inv(batch_T_cam_world)

    batch_R_world_cam = batch_T_world_cam[:, :3, :3]
    opengl2smpl = np.asarray(CFG.coordinate.transform.opengl2smpl)
    batch_R_world_cam = batch_R_world_cam @ opengl2smpl
    batch_R_world_cam = sRot.from_matrix(batch_R_world_cam)
    batch_R_world_cam_as_quat = batch_R_world_cam.as_quat()  # xyzw
    batch_R_world_cam_as_quat = batch_R_world_cam_as_quat[:, [3, 0, 1, 2]]  # wxyz
    batch_R_world_cam_as_euler = batch_R_world_cam.as_euler(
        seq=euler_order, degrees=False
    )
    return batch_R_world_cam_as_quat, batch_R_world_cam_as_euler


def local2global_pose(local_pose, kintree):
    """
    Convert local joint rotation matrices to global rotation matrices given a kinematic tree.

    Parameters
    ----------
    local_pose : torch.Tensor
        The local rotation matrices for joints in a batch of sequences.
        Shape: (BS, J, 3, 3)
    kintree : list
        The kinematic tree represented as a list where each element is the parent joint index.
        The root joint should have a parent index of -1.

    Returns
    -------
    torch.Tensor
        The global rotation matrices for joints.
        Shape: (BS, J, 3, 3)

    Examples
    --------
    >>> local_pose = torch.rand(2, 24, 3, 3)
    >>> kintree = [-1, 0, 1, 2, 0, 4, 5, 6, 0, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18, 0, 20, 21, 22]
    >>> local2global_pose(local_pose, kintree).shape
    torch.Size([2, 24, 3, 3])

    """
    # local_pose: T X J X 3 X 3
    bs = local_pose.shape[0]

    local_pose = local_pose.view(bs, -1, 3, 3)

    global_pose = local_pose.clone()

    for jId in range(len(kintree)):
        parent_id = kintree[jId]
        if parent_id >= 0:
            global_pose[:, jId] = torch.matmul(
                global_pose[:, parent_id], global_pose[:, jId]
            )

    return global_pose


def quat_ik_torch(grot_mat, kintree):
    """
    Compute local joint rotations from global joint rotations using inverse kinematics.

    Parameters
    ----------
    grot_mat : torch.Tensor of shape (N, J, 3, 3)
        Global rotations of the joints represented as rotation matrices. Shape: (N, J, 3, 3)
        where N is the batch size and J is the number of joints.

    Returns
    -------
    torch.Tensor
        Local joint rotations as rotation matrices. Shape: (N, J, 3, 3)

    Examples
    --------
    >>> grot_mat = torch.rand(2, 24, 3, 3)
    >>> kintree = [-1, 0, 1, 2, 0, 4, 5, 6, 0, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18, 0, 20, 21, 22]
    >>> quat_ik_torch(grot_mat, kintree).shape
    torch.Size([2, 24, 3, 3])


    """
    grot = transforms.matrix_to_quaternion(grot_mat)  # T X J X 4

    res = torch.cat(
        [
            grot[..., :1, :],
            transforms.quaternion_multiply(
                transforms.quaternion_invert(grot[..., kintree[1:], :]),
                grot[..., 1:, :],
            ),
        ],
        dim=-2,
    )  # N X J X 4

    res_mat = transforms.quaternion_to_matrix(res)  # N X J X 3 X 3

    return res_mat


def quat_fk_torch(lrot_mat, lpos, kintree):
    """
    Perform forward kinematics to compute global joint rotations and translations from local joint rotations.

    Parameters
    ----------
    lrot_mat : torch.Tensor of shape (N, J, 3, 3)
        **Local** rotations of the joints relative to their parent joints, represented as rotation matrices.
        Shape: (N, J, 3, 3) where N is the batch size and J is the number of joints.
    lpos : torch.Tensor of shape (N, J, 3)
        **Local** positions of the joints relative to their parent joints. The root joint is **in global space**.
        Shape: (N, J, 3)

    Returns
    -------
    tuple of torch.Tensor
        - Global rotations as a tensor of quaternions. Shape: (N, J, 4)
        - Global translations as a tensor. Shape: (N, J, 3)


    Examples
    --------
    >>> lrot = torch.rand(2, 24, 3, 3)
    >>> lpos = torch.rand(2, 24, 3)
    >>> kintree = [-1, 0, 1, 2, 0, 4, 5, 6, 0, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18, 0, 20, 21, 22]
    >>> res = quat_fk_torch(lrot, lpos, kintree)
    >>> res[0].shape, res[1].shape
    (torch.Size([2, 24, 4]), torch.Size([2, 24, 3]))

    """
    lrot = transforms.matrix_to_quaternion(lrot_mat)

    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
    for i in range(1, len(kintree)):
        gp.append(
            transforms.quaternion_apply(gr[kintree[i]], lpos[..., i : i + 1, :])
            + gp[kintree[i]]
        )
        gr.append(
            transforms.quaternion_multiply(gr[kintree[i]], lrot[..., i : i + 1, :])
        )

    res = torch.cat(gr, dim=-2), torch.cat(gp, dim=-2)

    return res
