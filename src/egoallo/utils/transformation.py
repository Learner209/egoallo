"""
====================================================
Homogeneous Transformation Matrices and Quaternions.
====================================================

This module provides a library for calculating 4x4 transformation matrices for operations such as translating, rotating, reflecting, scaling, shearing, projecting, orthogonalizing, and superimposing arrays of 3D homogeneous coordinates. It also includes conversion functions between rotation matrices, Euler angles, and quaternions, alongside an implementation of the Arcball control object and functions for decomposing transformation matrices.

Routine Listings:
- identity_matrix(): Returns the identity matrix.
- translation_matrix(direction): Returns a translation matrix along the direction vector.
- reflection_matrix(point, normal): Returns a matrix to mirror at a plane defined by point and normal vector.
- rotation_matrix(angle, direction, point=None): Returns a rotation matrix around the axis defined by point and direction.
- scale_matrix(factor, origin=None, direction=None): Returns a matrix to scale by factor around origin in the specified direction.
- projection_matrix(point, normal, direction=None, perspective=None, pseudo=False): Returns a projection matrix onto a plane defined by point and normal.
- clip_matrix(left, right, bottom, top, near, far, perspective=False): Returns a matrix to map the specified frustum to the unit cube.
- shear_matrix(angle, direction, point, normal): Returns a matrix to shear by angle along direction vector on a plane defined by point and normal.
- decompose_matrix(matrix): Decomposes a matrix into scale, shear, angles, translate, and perspective components.
- compose_matrix(scale=None, shear=None, angles=None, translate=None, perspective=None): Composes a transformation matrix from scale, shear, angles, translate, and perspective components.
- orthogonalization_matrix(lengths, angles): Returns the orthogonalization matrix for the crystallographic cell coordinates based on lengths and angles.
- affine_matrix_from_points(v0, v1, shear=True, scale=True, usesvd=True): Returns an affine transform matrix to register two sets of points.
- superimposition_matrix(v0, v1, scale=False, usesvd=True): Returns a matrix to superimpose two sets of points.

Examples and additional details for each function can be generated using help() built-in function or in the interactive Python environment.

This library is designed for ease of use in graphics, robotics, and simulation applications where transformation matrices are a fundamental tool.

Version: 2017.02.17
Author: Christoph Gohlke
Organization: Laboratory for Fluorescence Dynamics, University of California, Irvine
"""

from __future__ import division, print_function

import math

import numpy
import torch
from torch import nn

import numpy as np
from pytorch3d import transforms as transforms


__all__ = [
    # functional api
    "PI",
    "rad2deg",
    "deg2rad",
    "convert_points_from_homogeneous",
    "convert_points_to_homogeneous",
    "aa_to_rotMat",
    "rotMat_to_aa",
    "rotMat_to_quat",
    "quat_to_aa_t",
    "rtvec_to_pose",
    # layer api
    "RadToDeg",
    "DegToRad",
    "ConvertPointsFromHomogeneous",
    "ConvertPointsToHomogeneous",
]


"""Constant with number pi
"""
PI = torch.Tensor([3.14159265358979323846])


# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of (inner axis, parity, repetition, frame)
_AXES2TUPLE = {
    "sxyz": (0, 0, 0, 0),
    "sxyx": (0, 0, 1, 0),
    "sxzy": (0, 1, 0, 0),
    "sxzx": (0, 1, 1, 0),
    "syzx": (1, 0, 0, 0),
    "syzy": (1, 0, 1, 0),
    "syxz": (1, 1, 0, 0),
    "syxy": (1, 1, 1, 0),
    "szxy": (2, 0, 0, 0),
    "szxz": (2, 0, 1, 0),
    "szyx": (2, 1, 0, 0),
    "szyz": (2, 1, 1, 0),
    "rzyx": (0, 0, 0, 1),
    "rxyx": (0, 0, 1, 1),
    "ryzx": (0, 1, 0, 1),
    "rxzx": (0, 1, 1, 1),
    "rxzy": (1, 0, 0, 1),
    "ryzy": (1, 0, 1, 1),
    "rzxy": (1, 1, 0, 1),
    "ryxy": (1, 1, 1, 1),
    "ryxz": (2, 0, 0, 1),
    "rzxz": (2, 0, 1, 1),
    "rxyz": (2, 1, 0, 1),
    "rzyz": (2, 1, 1, 1),
}
_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

numpy.set_printoptions(suppress=True, precision=5)


def length(x, axis=-1, keepdims=True):
    """
    Computes vector norm along a numpy array (axis/axes)

    :param x: np.ndarray
    :param axis: axis(axes) along which to compute the norm
    :param keepdims: indicates if the dimension(s) on axis should be kept
    :return: The length or vector of lengths.
    """
    lgth = np.sqrt(np.sum(x * x, axis=axis, keepdims=keepdims))
    return lgth


def length_t(x, dim=-1, keepdim=True):
    """
    Computes vector norm along a tensor axis(axes)

    :param x: tensor
    :param axis: dim(dims) along which to compute the norm
    :param keepdim: indicates if the dimension(s) on dim should be kept
    :return: The length or vector of lengths.
    """
    lgth = torch.sqrt(torch.sum(x * x, dim=dim, keepdim=keepdim))
    return lgth


def normalize(x, axis=-1, eps=1e-8):
    """
    Normalizes a np.ndarray over some axis (axes)

    :param x: data np.ndarray
    :param axis: axis(axes) along which to compute the norm
    :param eps: epsilon to prevent numerical instabilities
    :return: The normalized tensor
    """
    res = x / (length(x, axis=axis) + eps)
    return res


def normalize_t(x, dim=-1, eps=1e-8):
    """
    Normalizes a tensor over some dim (dims)

    :param x: data tensor
    :param dim: dim(dims) along which to compute the norm
    :param eps: epsilon to prevent numerical instabilities
    :return: The normalized tensor
    """
    res = x / (length_t(x, dim=dim) + eps)
    return res


def safe_acos(q):
    """
    Parameters
    ----------
    q : tensor of shape (..., 4)

    Returns
    -------
    out : tensor of shape (..., 4)

    Notes
    -----
    pytorch acos nan: https://github.com/pytorch/pytorch/issues/8069

    Examples
    --------
    >>> res = safe_acos(torch.rand(5,3,4))
    """
    return torch.acos(torch.clamp(q, -1.0 + 1e-7, 1.0 - 1e-7))


def rad2deg(t):
    r"""Function that converts angles from radians to degrees.

    Args:
        tensor (Tensor): Tensor of arbitrary shape.

    Returns:
        Tensor: Tensor with same shape as input.

    Example:
        >>> input = PI * torch.rand(1, 3, 3)
        >>> output = rad2deg(input)
    """
    if not torch.is_tensor(t):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(t)))

    return 180.0 * t / PI.to(t.device).type(t.dtype)


def deg2rad(tensor):
    r"""Function that converts angles from degrees to radians.

    Args:
        tensor (Tensor): Tensor of arbitrary shape.

    Returns:
        Tensor: Tensor with same shape as input.

    Examples::

        >>> input = 360. * torch.rand(1, 3, 3)
        >>> output = deg2rad(input)
    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(tensor)))

    return tensor * PI.to(tensor.device).type(tensor.dtype) / 180.0


def convert_points_from_homogeneous(points):
    r"""Function that converts points from homogeneous to Euclidean space.

    Notes
    -----
    Normalize accoding to the last dimension.

    Examples::

        >>> input = torch.rand(2, 4, 3)  # BxNx3
        >>> output = convert_points_from_homogeneous(input)  # BxNx2
    """
    if not torch.is_tensor(points):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(points)))
    if len(points.shape) < 2:
        raise ValueError(
            "Input must be at least a 2D tensor. Got {}".format(points.shape)
        )
    return points[..., :-1] / points[..., -1:]


def convert_points_to_homogeneous(points):
    r"""Function that converts points from Euclidean to homogeneous space.

    See :class:`~torchgeometry.ConvertPointsToHomogeneous` for details.

    Examples::

        >>> input = torch.rand(2, 4, 3)  # BxNx3
        >>> output = convert_points_to_homogeneous(input)  # BxNx4
    """
    if not torch.is_tensor(points):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(points)))
    if len(points.shape) < 2:
        raise ValueError(
            "Input must be at least a 2D tensor. Got {}".format(points.shape)
        )
    return nn.functional.pad(points, (0, 1), "constant", 1.0)


def aa_to_rotMat(aa):
    """Convert 3d vector of axis-angle rotation to 4x4 rotation matrix

    Args:
        aa (Tensor): tensor of 3d vector of axis-angle rotations.

    Returns:
        Tensor: tensor of 4x4 rotation matrices.

    Shape:
        - Input: :math:`(N, 3)`
        - Output: :math:`(N, 4, 4)`

    Example:
    >>> input = torch.rand(10, 3)  # Nx3
    >>> output = aa_to_rotMat(input)  # Nx4x4
    """

    def _compute_rotMat(aa, theta2, eps=1e-6):
        # We want to be careful to only evaluate the square root if the
        # norm of the aa vector is greater than zero. Otherwise
        # we get a division by zero.
        k_one = 1.0
        theta = torch.sqrt(theta2)
        wxyz = aa / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = torch.cat(
            [r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1
        )
        return rotation_matrix.view(-1, 3, 3)

    def _compute_rotMat_taylor(aa):
        rx, ry, rz = torch.chunk(aa, 3, dim=1)
        k_one = torch.ones_like(rx)
        rotation_matrix = torch.cat(
            [k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1
        )
        return rotation_matrix.view(-1, 3, 3)

    # stolen from ceres/rotation.h

    _aa = torch.unsqueeze(aa, dim=1)
    theta2 = torch.matmul(_aa, _aa.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotMat(aa, theta2)
    rotation_matrix_taylor = _compute_rotMat_taylor(aa)

    # create mask to handle both cases
    eps = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask).type_as(theta2)
    mask_neg = (mask == False).type_as(theta2)  # noqa

    # create output pose matrix
    batch_size = aa.shape[0]
    rotMat = torch.eye(4).to(aa.device).type_as(aa)
    rotMat = rotMat.view(1, 4, 4).repeat(batch_size, 1, 1)
    # fill output matrix with masked values
    rotMat[..., :3, :3] = (
        mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    )
    return rotMat  # Nx4x4


def rtvec_to_pose(rtvec):
    """
    Convert axis-angle rotation and translation vector (combined a.k.a **Rodrigues** vector) to 4x4 pose matrix

    Args:
        rtvec (Tensor): **Rodrigues** vector transformations

    Returns:
        Tensor: transformation matrices

    Shape:
        - Input: :math:`(N, 6)`
        - et numberutput: :math:`(N, 4, 4)`

    Example:
        >>> input = torch.rand(3, 6)  # Nx6
        >>> output = rtvec_to_pose(input)  # Nx4x4
    """
    assert rtvec.shape[-1] == 6, "rtvec=[rx, ry, rz, tx, ty, tz]"
    pose = aa_to_rotMat(rtvec[..., :3])  # N x 4 x 4
    pose[..., :3, 3] = rtvec[..., 3:]  # N x 4 x 4
    return pose


def orth6d_to_rotMat(ortho6d):
    """
    Parameters
    ----------
    ortho6d : tensor of shape (N, 6)

    Returns
    -------
    rotMats : tensor of shape (N, 3, 4)

    Examples
    --------
    >>> ortho6d = torch.rand(10,6)
    >>> rotMats = orth6d_to_rotMat(ortho6d)
    """

    def normalize_vector(v, return_mag=False):
        batch = v.shape[0]
        v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
        v_mag = torch.max(
            v_mag,
            torch.autograd.Variable(
                torch.tensor([1e-8], dtype=v_mag.dtype).to(v.device)
            ),
        )
        v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
        v = v / v_mag
        if return_mag is True:
            return v, v_mag[:, 0]
        else:
            return v

    def cross_product(u, v):
        batch = u.shape[0]
        i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
        j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
        k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

        out = torch.cat(
            (i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1
        )  # BS x 3

        return out

    x_raw = ortho6d[:, 0:3]  # BS x 3
    y_raw = ortho6d[:, 3:6]  # BS x 3

    x = normalize_vector(x_raw)  # BS x 3
    z = cross_product(x, y_raw)  # BS x 3
    z = normalize_vector(z)  # BS x 3
    y = cross_product(z, x)  # BS x 3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    zeros = torch.zeros(z.shape, dtype=z.dtype).to(ortho6d.device)
    rotMats = torch.cat((x, y, z, zeros), dim=2)  # BS x 3 x 4
    return rotMats


def rotMat_to_orth6d(rotMats):
    """
    Parameters
    ----------
    rotMats : tensor of shape (N, 3, D) (D>=2)

    Returns
    -------
    ortho6d : tensor of shape (N, 6)

    Examples
    --------
    >>> rotMats = torch.rand(2,3,4)
    >>> ortho6d = rotMat_to_orth6d(rotMats)
    """
    orth6ds = rotMats[:, :, :2].transpose(1, 2).reshape(-1, 6)
    return orth6ds


def rotMat_to_aa(rotation_matrix):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
    >>> input = torch.rand(2, 3, 4)  # Nx4x4
    >>> output = rotMat_to_aa(input)  # Nx3
    """
    if rotation_matrix.shape[1:] == (3, 3):
        rot_mat = rotation_matrix.reshape(-1, 3, 3)
        hom = (
            torch.tensor([0, 0, 1], dtype=torch.float32, device=rotation_matrix.device)
            .reshape(1, 3, 1)
            .expand(rot_mat.shape[0], -1, -1)
        )
        rotation_matrix = torch.cat([rot_mat, hom], dim=-1)

    quat = rotMat_to_quat(rotation_matrix)
    aa = quat_to_aa_t(quat)
    aa[torch.isnan(aa)] = 0.0
    return aa


def aa_from_quat(_q, separate=False):
    """
    Parameters
    ----------
    _q : tensor of shape (4,)

    Returns
    -------
    axis : tensor of shape (3,)
    angle : float

    Examples
    --------
    >>> q = torch.rand(4)
    >>> _ = aa_from_quat(q)
    """
    q = _q.clone()
    device = q.device
    dtype = q.dtype
    if 1 - q[0] < 1e-6:
        axis = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
        angle = 0.0
    else:
        axis = q[1:4] / torch.sqrt(torch.abs(1 - q[0] * q[0]))
        angle = 2 * safe_acos(q[0])
    return (axis, angle) if separate else axis * angle


def aa_from_quat_batch(_q, separate=False):
    """
    q: size(Bx4)
    Output: size(Bx3)
    """

    assert _q.shape[-1] == 4

    q = _q.clone()
    B = q.size()[0]
    device = q.device
    dtype = q.dtype
    zero_axis = torch.tensor([1.0, 0.0, 0.0] * B, dtype=dtype, device=device).view(B, 3)
    zero_angle = torch.tensor([0.0] * B, dtype=dtype, device=device)

    # q = F.normalize(q,p=2, dim=1)

    cond = torch.abs(torch.sin(safe_acos(q[:, 0]))) < 1e-5
    axis = torch.where(
        cond.unsqueeze(1).repeat(1, 3),
        zero_axis,
        q[:, 1:4] / (torch.sin(safe_acos(q[:, 0]))).view(B, 1),
    )
    angle = torch.where(cond, zero_angle, 2 * safe_acos(q[:, 0]))
    assert angle.size()[0] == axis.size()[0]
    return (axis, angle) if separate else axis * angle.view(B, 1)


def aa_to_orth6d(aa_poses):
    """
    Parameters
    ----------
    poses : tensor of shape (..., 3)

    Returns
    -------
    ortho6d : tensor of shape (..., 6)

    Examples
    --------
    >>> poses = torch.rand(10,3)
    >>> ortho6d = aa_to_orth6d(poses)
    """
    curr_pose = aa_poses.to(aa_poses.device).float().reshape(-1, 3)
    rot_mats = aa_to_rotMat(curr_pose)  # N x 4 x 4
    rot_mats = rot_mats[:, :3, :]  # N x 3 x 4
    orth6d = rotMat_to_orth6d(rot_mats)  # N x 6
    orth6d = orth6d.view(aa_poses.shape[0], -1, 6)  # B x N x 6
    return orth6d


def orth6d_to_aa(orth6d):
    """
    Parameters
    ----------
    orth6d : tensor of shape (..., 6)

    Returns
    -------
    poseaa : tensor of shape (..., 3)

    Examples
    --------
    >>> orth6d = torch.rand(10,6)
    >>> poseaa = orth6d_to_aa(orth6d)
    """
    orth6d_flat = orth6d.reshape(-1, 6)
    rot_mat6d = orth6d_to_rotMat(orth6d_flat)  # N x 3 x 4
    pose_aa = rotMat_to_aa(rot_mat6d)  # N x 3

    shape_curr = list(orth6d.shape)
    shape_curr[-1] /= 2
    shape_curr = tuple([int(i) for i in shape_curr])
    pose_aa = pose_aa.reshape(shape_curr)  # ... x 3
    return pose_aa


def quat_to_orth6d(quats):
    """
    Parameters
    ----------
    quats : tensor of shape (..., 4)

    Returns
    -------
    quat_6d : tensor of shape (..., 6)

    Examples
    --------
    >>> quats = torch.rand(10,4)
    >>> quat_6d = quat_to_orth6d(quats)
    """
    quat_flat = quats.reshape(-1, 4)
    all_mats = quat_matrix_batch_t(quat_flat)  # N x 3 x 3
    quat_6d = rotMat_to_orth6d(all_mats)  # N x 6

    shape_curr = list(quats.shape)
    shape_curr[-1] = 6
    shape_curr = tuple([int(i) for i in shape_curr])
    quat_6d = quat_6d.reshape(shape_curr)
    return quat_6d


def orth6d_to_quat(orth6d):
    """
    Parameters
    ----------
    orth6d : tensor of shape (..., 6)

    Returns
    -------
    quats : tensor of shape (..., 4)

    Examples
    --------
    >>> orth6d = torch.rand(10,6)
    >>> quats = orth6d_to_quat(orth6d)
    """
    orth6d_flat = orth6d.reshape(-1, 6)

    rot_mat6d = orth6d_to_rotMat(orth6d_flat)  # N x 3 x 3
    quats = rotMat_to_quat(rot_mat6d)  # N x 4

    shape_curr = list(orth6d.shape)
    shape_curr[-1] = 4
    shape_curr = tuple([int(i) for i in shape_curr])
    quats = quats.reshape(shape_curr)

    return quats


def canonicalize_smpl_root(aa_poses, root_vec=[PI, 0, 0]):
    """
    Canonicalize the rotation denoted by `aa_poses` to let its frist frame have the same rotation as `root_vec`.

    Parameters
    ----------
    aa_poses : tensor of shape (N, D) (D>=3, the first 3 elements must be in axis-angle formats.)
    root_vec : list of shape (3,) in axis-angle format.

    Returns
    -------
    aa_poses : tensor of shape (N, 3)

    Examples
    --------
    >>> aa_poses = torch.rand(10, 3)
    >>> root_vec = [PI, 0, 0]
    >>> aa_poses = canonicalize_smpl_root(aa_poses, root_vec)
    """
    device = aa_poses.device

    target_mat = aa_to_rotMat(
        torch.tensor([root_vec], dtype=aa_poses.dtype).to(device)
    )[:, :3, :3].to(device)  # 1 x 3 x 3
    org_mats = aa_to_rotMat(aa_poses[:, :3])[:, :3, :3].to(device)  # N x 3 x 3
    org_mat_inv = torch.inverse(org_mats[0]).to(device)  # 3 x 3
    apply_mat = torch.matmul(target_mat, org_mat_inv)  # 1 x 3 x 3
    res_root_mat = torch.matmul(apply_mat, org_mats)  # N x 3 x 3
    zeros = torch.zeros(
        (res_root_mat.shape[0], res_root_mat.shape[1], 1), dtype=res_root_mat.dtype
    ).to(device)  # N x 3 x 1
    res_root_mats_4 = torch.cat((res_root_mat, zeros), dim=2)  # N x 3 x 4
    res_root_aa = rotMat_to_aa(res_root_mats_4)  # N x 3

    aa_poses[:, :3] = res_root_aa
    return aa_poses


def batch_rodrigues(aa):
    """
    Parameters
    ----------
    aa : tensor of shape (N, 3)

    Returns
    -------
    R : tensor of shape (N, 3, 3)

    See also
    --------
    This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L37

    Examples
    --------
    >>> aa = torch.rand(10,3)
    >>> R = batch_rodrigues(aa)
    """
    aa_norm = torch.norm(aa + 1e-8, p=2, dim=1)  # N
    angle = torch.unsqueeze(aa_norm, -1)  # N x 1
    aa_normed = torch.div(aa, angle)  # N x 3
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * aa_normed], dim=1)  # N x 4
    rot_mat = quat_to_rotMat_t(quat)  # N x 3 x 3
    rot_mat = rot_mat.view(rot_mat.shape[0], 9)  # N x 9
    return rot_mat


def perspective_projection_cam(pred_joints, pred_camera):
    """
    *Use with care* as there are some empirical constants in the function.

    Parameters
    ----------
    pred_joints : tensor of shape (N, 24, 3)
    pred_camera : tensor of shape (N, 3)

    Returns
    -------
    pred_keypoints_2d : tensor of shape (N, 24, 2)

    Examples
    --------
    >>> pred_joints = torch.rand(10,24,3)
    >>> pred_camera = torch.rand(10,3)
    >>> pred_keypoints_2d = perspective_projection_cam(pred_joints, pred_camera)
    """
    pred_cam_t = torch.stack(
        [
            pred_camera[:, 1],
            pred_camera[:, 2],
            2 * 5000.0 / (224.0 * pred_camera[:, 0] + 1e-9),
        ],
        dim=-1,
    )
    BS = pred_joints.shape[0]
    camera_center = torch.zeros(BS, 2)
    pred_keypoints_2d = perspective_projection(
        pred_joints,
        rotation=torch.eye(3).unsqueeze(0).expand(BS, -1, -1).to(pred_joints.device),
        translation=pred_cam_t,
        focal_length=5000.0,
        camera_center=camera_center,
    )  # N x J x 2
    # Normalize keypoints to [-1,1]
    pred_keypoints_2d = pred_keypoints_2d / (224.0 / 2.0)
    return pred_keypoints_2d


def perspective_projection(points, rotation, translation, focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points.

    Parameters
    ----------
    points : tensor of shape (BS, N, 3) indicating 3D points
    rotation : tensor of shape (BS, 3, 3) indicating camera rotation
    translation : tensor of shape (BS, 3) indicating Camera translation
    focal_length : tensor of shape (BS,) or scalar indicating focal length
    camera_center : tensor of shape (BS, 2) indicating camera center

    Returns
    -------
    projected_points : tensor of shape (BS, N, 2) indicating 2D points

    Examples
    --------
    >>> points = torch.rand(10,24,3)
    >>> rotation = torch.rand(10,3,3)
    >>> translation = torch.rand(10,3)
    >>> focal_length = torch.rand(10)
    >>> camera_center = torch.rand(10,2)
    >>> projected_points = perspective_projection(points, rotation, translation, focal_length, camera_center)
    """

    BS = points.shape[0]
    K = torch.zeros([BS, 3, 3], device=points.device)  # B x 3 x 3
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1.0
    K[:, :-1, -1] = camera_center

    # Transform points
    points = torch.einsum("bij,bkj->bki", rotation, points)  #  B x N x 3
    points = points + translation.unsqueeze(1)  # B x N x 3

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)  # B x N x 3

    # Apply camera intrinsics
    projected_points = torch.einsum("bij,bkj->bki", K, projected_points)  # B x N x 3

    return projected_points[:, :, :-1]  # B x N x 2


def quat_smooth(quat, ratio=0.3):
    """Converts quat to minimize Euclidean distance from previous quat (wxyz order)

    Parameters
    ----------
    quat : np.ndarray of shape (N, 4)

    Returns
    -------
    quat : np.ndarray of shape (N, 4)

    Examples
    --------
    >>> quat = np.random.rand(10,4)
    >>> quat = quat_smooth(quat)
    """
    for _ in range(1, quat.shape[0]):
        quat[_] = quat_slerp(quat[_ - 1], quat[_], ratio)
    return quat


def quat_slerp(q0, q1, fraction, spin=0, shortestpath=True):
    """Return spherical linear interpolation between two quats.

    Parameters
    ----------
    q0 : np.ndarray of shape (4,)
    q1 : np.ndarray of shape (4,)

    Returns
    -------
    q : np.ndarray of shape (4,)

    Examples
    --------
    >>> q0 = rand_quat()
    >>> q1 = rand_quat()
    >>> q = quat_slerp(q0, q1, 0)
    >>> np.allclose(q, q0)
    True
    >>> q = quat_slerp(q0, q1, 1, 1)
    >>> np.allclose(q, q1)
    True
    >>> q = quat_slerp(q0, q1, 0.5)
    >>> angle = math.acos(np.dot(q0, q))
    >>> np.allclose(2, math.acos(np.dot(q0, q1)) / angle) or \
        np.allclose(2, math.acos(-np.dot(q0, q1)) / angle)
    True

    """
    q0 = unit_vector(q0[:4])
    q1 = unit_vector(q1[:4])
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < _EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        np.negative(q1, q1)
    angle = math.acos(d) + spin * math.pi
    if abs(angle) < _EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0


def rotMat_to_quat(rotation_matrix, eps=1e-6):
    """Convert 3x4 rotation matrix to 4d quat vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quat

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = rotMat_to_quat(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError(
            "Input type is not a torch.Tensor. Got {}".format(type(rotation_matrix))
        )

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape
            )
        )
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape
            )
        )

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack(
        [
            rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
            t0,
            rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
            rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
        ],
        -1,
    )
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack(
        [
            rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
            rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
            t1,
            rmat_t[:, 1, 2] + rmat_t[:, 2, 1],
        ],
        -1,
    )
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack(
        [
            rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
            rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
            rmat_t[:, 1, 2] + rmat_t[:, 2, 1],
            t2,
        ],
        -1,
    )
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack(
        [
            t3,
            rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
            rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
            rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
        ],
        -1,
    )
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * (~mask_d0_d1)
    mask_c2 = (~mask_d2) * mask_d0_nd1
    mask_c3 = (~mask_d2) * (~mask_d0_nd1)
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(
        t0_rep * mask_c0
        + t1_rep * mask_c1
        + t2_rep * mask_c2  # noqa
        + t3_rep * mask_c3
    )  # noqa
    q *= 0.5
    return q


def quat_from_euler_t(ai, aj, ak, axes="sxyz"):
    """ "
    Parameters
    ----------
    ai, aj, ak: tensor of shape B x 1

    Returns
    -------
    quat: tensor of shape B x 4

    Examples
    --------
    >>> ai = torch.rand(10,1)
    >>> aj = torch.rand(10,1)
    >>> ak = torch.rand(10,1)
    >>> quat = quat_from_euler_t(ai, aj, ak)
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    B = ai.shape[0]
    ai, aj, ak = ai.clone(), aj.clone(), ak.clone()

    device = ai.device
    dtype = ai.dtype

    i = firstaxis + 1
    j = _NEXT_AXIS[i + parity - 1] + 1
    k = _NEXT_AXIS[i - parity] + 1

    if frame:
        ai, ak = ak.clone(), ai.clone()
    if parity:
        aj = -aj.clone()

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = torch.cos(ai)
    si = torch.sin(ai)
    cj = torch.cos(aj)
    sj = torch.sin(aj)
    ck = torch.cos(ak)
    sk = torch.sin(ak)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk

    q = torch.tensor([1.0, 0.0, 0.0, 0.0] * B, dtype=dtype, device=device).view(B, 4)
    if repetition:
        q[:, 0:1] = cj * (cc - ss)
        q[:, i : i + 1] = cj * (cs + sc)
        q[:, j : j + 1] = sj * (cc + ss)
        q[:, k : k + 1] = sj * (cs - sc)
    else:
        q[:, 0:1] = cj * cc + sj * ss
        q[:, i : i + 1] = cj * sc - sj * cs
        q[:, j : j + 1] = cj * ss + sj * cc
        q[:, k : k + 1] = cj * cs - sj * sc
    if parity:
        q[:, j] *= -1.0

    return q


def quat_from_euler(ai, aj, ak, axes="sxyz"):
    """ "
    Parameters
    ----------
    ai, aj, ak: tensor of shape B x 1

    Returns
    -------
    quat: tensor of shape B x 4

    Examples
    --------
    >>> ai = np.random.rand(10,1)
    >>> aj = np.random.rand(10,1)
    >>> ak = np.random.rand(10,1)
    >>> quat = quat_from_euler(ai, aj, ak)
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    B = ai.shape[0]
    ai, aj, ak = ai.copy(), aj.copy(), ak.copy()

    i = firstaxis + 1
    j = _NEXT_AXIS[i + parity - 1] + 1
    k = _NEXT_AXIS[i - parity] + 1

    if frame:
        ai, ak = ak.copy(), ai.copy()
    if parity:
        aj = -aj.copy()

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = np.cos(ai)
    si = np.sin(ai)
    cj = np.cos(aj)
    sj = np.sin(aj)
    ck = np.cos(ak)
    sk = np.sin(ak)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk

    q = np.array([1.0, 0.0, 0.0, 0.0] * B).reshape(B, 4)
    if repetition:
        q[:, 0:1] = cj * (cc - ss)
        q[:, i : i + 1] = cj * (cs + sc)
        q[:, j : j + 1] = sj * (cc + ss)
        q[:, k : k + 1] = sj * (cs - sc)
    else:
        q[:, 0:1] = cj * cc + sj * ss
        q[:, i : i + 1] = cj * sc - sj * cs
        q[:, j : j + 1] = cj * ss + sj * cc
        q[:, k : k + 1] = cj * cs - sj * sc
    if parity:
        q[:, j] *= -1.0

    return q


def quat_normalize(x, eps=1e-8):
    """
    Normalizes a quat np.ndarray

    :param x: data np.ndarray of shape (... , 4)
    :param eps: epsilon to prevent numerical instabilities
    :return: The normalized quats np.ndarray of shape (... , 4)
    """
    res = normalize(x, eps=eps)
    return res


def quat_normalize_t(x, eps=1e-8):
    """
    Normalizes a quat tensor

    :param x: data tensor of shape (... , 4)
    :param eps: epsilon to prevent numerical instabilities
    :return: The normalized quats tensor of shape (... , 4)
    """
    res = normalize_t(x, eps=eps)
    return res


def quat_inv(q):
    """
    Parameters
    ----------
    q : ndarray of shape (..., 4)

    Returns
    -------
    res : ndarray of shape (..., 4)

    Examples
    --------
    >>> q = np.random.rand(2,3,4)
    >>> res = quat_inv(q)
    """
    res = np.asarray([1, -1, -1, -1], dtype=np.float32) * q
    return res


def quat_inv_t(q):
    """
    Parameters
    ----------
    q : tensor of shape (..., 4)

    Returns
    -------
    res : tensor of shape (..., 4)

    Examples
    --------
    >>> q = torch.rand(2,3,4)
    >>> res = quat_inv_t(q)
    """
    res = torch.FloatTensor([1, -1, -1, -1]) * q
    return res


def quat_mul_t(q, q_):
    """
    Performs quat multiplication on arrays of quats

    :param x: tensor of quats of shape (..., 4)
    :param y: tensor of quats of shape (..., 4)
    :return: tensor of shape (..., 4)
    """
    x0, x1, x2, x3 = q[..., 0:1], q[..., 1:2], q[..., 2:3], q[..., 3:4]
    y0, y1, y2, y3 = q_[..., 0:1], q_[..., 1:2], q_[..., 2:3], q_[..., 3:4]

    res = torch.cat(
        [
            y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
            y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
            y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
            y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0,
        ],
        dim=-1,
    )

    return res


def quat_mul(q, q_):
    """
    Performs quat multiplication on arrays of quats

    :param x: ndarray of quats of shape (..., 4)
    :param y: ndarray of quats of shape (..., 4)
    :return: ndarray of shape (..., 4)
    """
    x0, x1, x2, x3 = q[..., 0:1], q[..., 1:2], q[..., 2:3], q[..., 3:4]
    y0, y1, y2, y3 = q_[..., 0:1], q_[..., 1:2], q_[..., 2:3], q_[..., 3:4]

    res = np.concatenate(
        [
            y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
            y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
            y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
            y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0,
        ],
        axis=-1,
    )

    return res


def quat_between(x, y):
    """
    quat rotations between two 3D-vector arrays

    :param x: numpy.ndarray of shape (..., 3)
    :param y: numpy.ndarray of shape (..., 3)
    :return: numpy.ndarray of quats of convention `wxyz`.
    """
    res = np.concatenate(
        [
            np.sqrt(np.sum(x * x, axis=-1) * np.sum(y * y, axis=-1))[..., np.newaxis]
            + np.sum(x * y, axis=-1)[..., np.newaxis],
            np.cross(x, y),
        ],
        axis=-1,
    )
    return res


def quat_between_t(x, y):
    """
    quat rotations between two 3D-vector tensors

    :param x: tensor of shape (..., 3)
    :param y: tensor of shape (..., 3)
    :return: tensor of shape (..., 4) of convention `wxyz`.
    """
    x_norm = torch.sqrt(torch.sum(x * x, dim=-1))
    y_norm = torch.sqrt(torch.sum(y * y, dim=-1))
    dot_product = torch.sum(x * y, dim=-1)
    w = x_norm * y_norm + dot_product
    xyz = torch.cross(x, y)
    res = torch.cat([w.unsqueeze(-1), xyz], dim=-1)
    return res


def quat_to_rotMat_t(q):
    """
    Convert quat(`wxyz`) to rotMat.

    Parameters
    ----------
        quat: tensor of shape (BS, 4)

    Returns
    -------
        rotMat: tensor of shape (BS, 3, 3)

    See also
    --------
    This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L50

    Examples
    --------
    >>> q = torch.rand(10,4)
    >>> rotMat = quat_to_rotMat_t(q)
    """
    norm_quat = q
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    batch_size = q.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=1,
    ).view(batch_size, 3, 3)
    return rotMat


def quat_to_rotMat(q):
    """Convert q(`wxyz`) to rotMat.

    Parameters
    ----------
    q : np.ndarray or list of shape (4,)

    Returns
    -------
    rotMat : np.ndarray of shape (3, 3)

    Examples
    --------
    >>> M = quat_to_rotMat([0.99810947, 0.06146124, 0, 0])
    >>> np.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quat_to_rotMat([1, 0, 0, 0])
    >>> np.allclose(M, np.identity(4))
    True
    >>> M = quat_to_rotMat([0, 1, 0, 0])
    >>> np.allclose(M, np.diag([1, -1, -1, 1]))
    True

    """
    q_ = np.array(q, dtype=np.float64, copy=True)
    n = np.dot(q_, q_)
    if n < _EPS:
        return np.identity(4)
    q_ *= math.sqrt(2.0 / n)
    q_ = np.outer(q_, q_)
    return np.array(
        [
            [1.0 - q_[2, 2] - q_[3, 3], q_[1, 2] - q_[3, 0], q_[1, 3] + q_[2, 0], 0.0],
            [q_[1, 2] + q_[3, 0], 1.0 - q_[1, 1] - q_[3, 3], q_[2, 3] - q_[1, 0], 0.0],
            [q_[1, 3] - q_[2, 0], q_[2, 3] + q_[1, 0], 1.0 - q_[1, 1] - q_[2, 2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def quat_conjugate(q):
    """Return conjugate of quat.

    >>> q0 = rand_quat()
    >>> q1 = quat_conjugate(q0)
    >>> q1[0] == q0[0] and all(q1[1:] == -q0[1:])
    True

    """
    q_ = np.array(q, dtype=np.float64, copy=True)
    np.negative(q_[1:], q_[1:])
    return q_


def quat_real(q):
    """Return real part of quats.

    >>> quat_real([3, 0, 1, 2])
    3.0

    """
    return float(q[0])


def quat_imag(q):
    """Return imaginary part of quats."""
    return numpy.array(q[1:4], dtype=numpy.float64, copy=True)


def quat_to_bullet(q):
    return np.array([q[1], q[2], q[3], q[0]])


def quat_from_bullet(q):
    return np.array([q[3], q[0], q[1], q[2]])


def quat_to_aa_t(quaternion: torch.Tensor) -> torch.Tensor:
    """
    This function is borrowed from https://github.com/kornia/kornia
    Convert quaternion vector to angle axis of rotation.
    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h
    Args:
        quaternion (torch.Tensor): tensor with quaternions.
    Return:
        torch.Tensor: tensor with angle axis of rotation.
    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`
    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError(
            "Input type is not a torch.Tensor. Got {}".format(type(quaternion))
        )

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape Nx4 or 4. Got {}".format(quaternion.shape)
        )
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta),
    )

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def quat_mul_vec(q, v):
    """
    Performs multiplication of an array of 3D vectors by an array of quats (rotation).

    :param q: array of shape (..., 4)
    :param x: array of  shape (..., 3)
    :return: the resulting array of rotated vectors
    """
    t = 2.0 * np.cross(q[..., 1:], v)
    res = v + q[..., 0][..., np.newaxis] * t + np.cross(q[..., 1:], t)

    return res


def quat_mul_vec_t(q, v):
    """
    Parameters
    ----------
    q : tensor of shape (4,)
    v : tensor of shape (3,)
    """
    q_ = q.unsqueeze(0)
    v_ = v.unsqueeze(0)
    return quat_mul_vec_t_batch(q_, v_).squeeze(0)


def quat_mul_vec_t_batch(q, v):
    """
    Parameters
    ----------
    q : tensor of shape (..., 4)
    v : tensor of shape (..., 3)
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.reshape(-1, 4)
    v = v.reshape(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).reshape(original_shape)


def quat_multiply(_q1, _q0):
    """
    Parameters
    ----------
    _q0 : tensor of shape (4,)
    _q1 : tensor of shape (4,)

    Returns
    -------
    res : tensor of shape (4,)

    Examples
    --------
    >>> q0 = torch.rand(4)
    >>> q1 = torch.rand(4)
    >>> res = quat_multiply(q0, q1)
    """
    q0 = _q0.clone()
    q1 = _q1.clone()
    device = q0.device
    dtype = q0.dtype
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    return torch.tensor(
        [
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
        ],
        dtype=dtype,
        device=device,
    )


def quat_multiply_batch(q0, q1):
    """
    Multiply quat(s) q0 with quat(s) q1.

    Parameters
    ----------
    _q0 : tensor of shape (... , 4)
    _q1 : tensor of shape (... , 4)

    Returns
    -------
    res : tensor of shape (... , 4)

    Examples
    --------
    >>> q0 = torch.rand(10,4)
    >>> q1 = torch.rand(10,4)
    >>> res = quat_multiply_batch(q0, q1)

    See also
    --------
    https://github.com/facebookresearch/QuaterNet/blob/master/common/quaternion.py
    """
    assert q1.shape[-1] == 4
    assert q0.shape[-1] == 4

    original_shape = q0.shape

    # Compute outer product
    q1_view = q1.reshape(-1, 4, 1).clone()
    q0_view = q0.reshape(-1, 1, 4).clone()
    terms = torch.bmm(q1_view, q0_view)

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).reshape(original_shape)


def quat_matrix_batch_t(_q):
    """
    Parameters
    ----------
    _q : tensor of shape (N, 4)

    Returns
    -------
    res : tensor of shape (N, 3, 3)

    Examples
    --------
    >>> q = torch.rand(10,4)
    >>> rotMat = quat_matrix_batch_t(q)
    """
    # ZL: from YE, needs to be changed
    q = _q.clone()
    q_norm = torch.norm(q, dim=1).view(-1, 1)
    q = q / q_norm
    tx = q[..., 1] * 2.0
    ty = q[..., 2] * 2.0
    tz = q[..., 3] * 2.0
    twx = tx * q[..., 0]
    twy = ty * q[..., 0]
    twz = tz * q[..., 0]
    txx = tx * q[..., 1]
    txy = ty * q[..., 1]
    txz = tz * q[..., 1]
    tyy = ty * q[..., 2]
    tyz = tz * q[..., 2]
    tzz = tz * q[..., 3]
    res = torch.stack(
        (
            torch.stack((1.0 - (tyy + tzz), txy + twz, txz - twy), dim=1),
            torch.stack((txy - twz, 1.0 - (txx + tzz), tyz + twx), dim=1),
            torch.stack((txz + twy, tyz - twx, 1.0 - (txx + tyy)), dim=1),
        ),
        dim=2,
    )
    #     res = torch.zeros(res.shape).to(_q.device)
    return res


def quat_about_axis_t(angle, axis):
    """
    Parameters
    ----------
    angle : tensor of shape (1,)
    axis : tensor of shape (3,)

    Returns
    -------
    q : tensor of shape (4,)
    """
    device = angle.device
    dtype = angle.dtype
    q = torch.tensor([0.0, axis[0], axis[1], axis[2]], dtype=dtype, device=device)
    qlen = torch.norm(q, p=2)
    if qlen > _EPS:
        q = q * torch.sin(angle / 2.0) / qlen
    q[0] = torch.cos(angle / 2.0)

    return q


def quat_from_expmap_t(e):
    """
    Parameters
    ----------
    e : tensor of shape (3,)

    Returns
    -------
    q : tensor of shape (4,)
    """
    device = e.device
    dtype = e.dtype
    angle = torch.norm(e, p=2)

    if angle < 1e-8:
        axis = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
    else:
        axis = e / angle

    return quat_about_axis_t(angle, axis)


def quat_from_expmap(e):
    """
    Parameters
    ----------
    e : ndarray of shape (3,)

    Returns
    -------
    q : ndarray of shape (4,)
    """
    angle = np.linalg.norm(e)
    if angle < 1e-12:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = e / angle
    return quat_about_axis_t(angle, axis)


def quat_about_axis_batch(angle, axis):
    """
    Parameters
    ----------
    angle : tensor of shape (N, 1)
    axis : tensor of shape (N, 3)

    Returns
    -------
    q : tensor of shape (N, 4)

    """
    device = angle.device
    dtype = angle.dtype
    batch_size, _ = angle.shape
    q = torch.zeros((batch_size, 4)).to(device).type(dtype)
    q[:, 1] = axis[:, 0]
    q[:, 2] = axis[:, 1]
    q[:, 3] = axis[:, 2]

    qlen = torch.norm(q, dim=1, p=2)
    q_change = (
        q[qlen > _EPS, :]
        * torch.sin(angle[qlen > _EPS, :] / 2.0)
        / qlen[qlen > _EPS].view(-1, 1)
    )
    q_res = q.clone()
    q_res[qlen > _EPS, :] = q_change
    q_res[:, 0:1] = torch.cos(angle / 2.0)
    return q_res


def quat_from_expmap_batch(e):
    """
    Parameters
    ----------
    e : ndarray of shape (... , 3)

    Returns
    -------
    q : ndarray of shape (... , 4)
    """
    device = e.device
    dtype = e.dtype
    angle = torch.norm(e, dim=1, p=2)
    axis = torch.zeros(e.shape).to(device).type(dtype)

    axis[angle < 1e-8] = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
    axis[angle >= 1e-8] = e[angle >= 1e-8] / angle[angle >= 1e-8].view(-1, 1)
    return quat_about_axis_batch(angle.view(-1, 1), axis)


def euler_matrix(ai, aj, ak, axes="sxyz"):
    """Return homogeneous rotation matrix from Euler angles and axis sequence.

    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple

    >>> R = euler_matrix(1, 2, 3, 'syxz')
    >>> numpy.allclose(numpy.sum(R[0]), -1.34786452)
    True
    >>> R = euler_matrix(1, 2, 3, (0, 1, 0, 1))
    >>> numpy.allclose(numpy.sum(R[0]), -0.383436184)
    True
    >>> ai, aj, ak = (4*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R = euler_matrix(ai, aj, ak, axes)
    >>> for axes in _TUPLE2AXES.keys():
    ...    R = euler_matrix(ai, aj, ak, axes)

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    M = numpy.identity(4)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj * si
        M[i, k] = sj * ci
        M[j, i] = sj * sk
        M[j, j] = -cj * ss + cc
        M[j, k] = -cj * cs - sc
        M[k, i] = -sj * ck
        M[k, j] = cj * sc + cs
        M[k, k] = cj * cc - ss
    else:
        M[i, i] = cj * ck
        M[i, j] = sj * sc - cs
        M[i, k] = sj * cc + ss
        M[j, i] = cj * sk
        M[j, j] = sj * ss + cc
        M[j, k] = sj * cs - sc
        M[k, i] = -sj
        M[k, j] = cj * si
        M[k, k] = cj * ci
    return M


def euler_from_matrix(matrix, axes="sxyz"):
    """Return Euler angles from rotation matrix for specified axis sequence.

    axes : One of 24 axis sequences as string or encoded tuple

    Note that many Euler angle triplets can describe one matrix.

    >>> R0 = euler_matrix(1, 2, 3, 'syxz')
    >>> al, be, ga = euler_from_matrix(R0, 'syxz')
    >>> R1 = euler_matrix(al, be, ga, 'syxz')
    >>> numpy.allclose(R0, R1)
    True
    >>> angles = (4*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R0 = euler_matrix(axes=axes, *angles)
    ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
    ...    if not numpy.allclose(R0, R1): print(axes, "failed")

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > _EPS:
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > _EPS:
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


def euler_from_quat(quat, axes="sxyz"):
    """Return Euler angles from quat for specified axis sequence.

    >>> angles = euler_from_quat([0.99810947, 0.06146124, 0, 0])
    >>> numpy.allclose(angles, [0.123, 0, 0])
    True

    """
    return euler_from_matrix(quat_to_rotMat(quat), axes)


def rand_quat(rand=None):
    """Return uniform random unit quat.

    rand: array like or None
        Three independent random variables that are uniformly distributed
        between 0 and 1.

    >>> q = rand_quat()
    >>> numpy.allclose(1, vector_norm(q))
    True
    >>> q = rand_quat(numpy.random.random(3))
    >>> len(q.shape), q.shape[0]==4
    (1, True)

    """
    if rand is None:
        rand = np.random.rand(3)
    else:
        assert len(rand) == 3
    r1 = np.sqrt(1.0 - rand[0])
    r2 = np.sqrt(rand[0])
    pi2 = math.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    return np.array(
        [np.cos(t2) * r2, np.sin(t1) * r1, np.cos(t1) * r1, np.sin(t2) * r2]
    )


def transform_vec_t(v, q, trans="root"):
    """
    Parameters
    ----------
    v : tensor of shape (3,)
    q : tensor
    trans : str, optional, default='root'

    Returns
    -------
    v : tensor of shape (3,)

    Examples
    --------
    >>> v = torch.rand(3)
    >>> q = torch.rand(4)
    >>> v = transform_vec_t(v, q)
    """
    if trans == "root":
        rot = quat_to_rotMat_t(q[None])[0, :3, :3]
    elif trans == "heading":
        hq = q.clone()
        hq[1] = 0.0
        hq[2] = 0.0
        hq = hq / torch.norm(hq, p=2)
        rot = quat_to_rotMat_t(hq[None])[0, :3, :3]
    else:
        assert False

    v = torch.matmul(torch.transpose(rot, 0, 1), v)
    return v


def transform_vec_batch_t(v, q, trans="root"):
    """
    Parameters
    ----------
    v : tensor of shape (... , 3)
    q : tensor
    trans : str, optional, default='root'

    Returns
    -------
    v : tensor of shape (... , 3)

    Examples
    --------
    >>> v = torch.rand(10, 3)
    >>> q = torch.rand(10, 4)
    >>> v = transform_vec_batch_t(v, q)
    """
    if trans == "root":
        rot = quat_matrix_batch_t(q)
    elif trans == "heading":
        hq = get_heading_q_batch(q)
        rot = quat_matrix_batch_t(hq)
    else:
        assert False
    v = torch.matmul(torch.transpose(rot, 1, 2), v.unsqueeze(2))
    return v.squeeze(2)


def get_qvel_fd(cur_qpos, next_qpos, dt, transform=None):
    """
    Parameters
    ----------
    cur_qpos : tensor of shape (7,)
    next_qpos : tensor of shape (7,)
    dt : float
    transform : str, optional, default=None

    Returns
    -------
    qvel : tensor of shape (7,)

    Examples
    --------
    >>> cur_qpos = torch.rand(7)
    >>> next_qpos = torch.rand(7)
    >>> dt = 0.1
    >>> qvel = get_qvel_fd(cur_qpos, next_qpos, dt)
    """
    v = (next_qpos[:3] - cur_qpos[:3]) / dt
    qrel = quat_multiply(next_qpos[3:7], quat_inv_t(cur_qpos[3:7]))
    axis, angle = aa_from_quat(qrel, True)
    if angle > PI:  # -180 < angle < 180
        angle = angle - 2 * PI
    elif angle < -PI:
        angle = angle + 2 * PI
    rv = (axis * angle) / dt

    rv = transform_vec_t(rv, cur_qpos[3:7], "root")
    qvel = (next_qpos[7:] - cur_qpos[7:]) / dt
    qvel = torch.cat((v, rv, qvel))
    if transform is not None:
        v = transform_vec_t(v, cur_qpos[:, 3:7], transform)
        qvel[:, :3] = v

    return qvel


def get_qvel_fd_batch(cur_qpos, next_qpos, dt, transform=None):
    """
    Parameters
    ----------
    cur_qpos : tensor of shape (N, 7)
    next_qpos : tensor of shape (N, 7)
    dt : float
    transform : str, optional, default=None

    Returns
    -------
    qvel : tensor of shape (N, 7)

    Examples
    --------
    >>> cur_qpos = torch.rand(10, 7)
    >>> next_qpos = torch.rand(10, 7)
    >>> dt = 0.1
    >>> qvel = get_qvel_fd_batch(cur_qpos, next_qpos, dt)
    """
    v = (next_qpos[:, :3] - cur_qpos[:, :3]) / dt
    qrel = quat_multiply_batch(next_qpos[:, 3:7], quat_inv_t(cur_qpos[:, 3:7]))
    axis, angle = aa_from_quat_batch(qrel, True)

    angle[angle > PI] = angle[angle > PI] - 2 * PI
    angle[angle < -PI] = angle[angle < -PI] + 2 * PI

    rv = (axis * angle.view(-1, 1)) / dt

    rv = transform_vec_batch_t(rv, cur_qpos[:, 3:7], "root")
    qvel = (next_qpos[:, 7:] - cur_qpos[:, 7:]) / dt
    qvel = torch.cat((v, rv, qvel), dim=1)
    if transform is not None:
        v = transform_vec_t(v, cur_qpos[3:7], transform)
        qvel[:3] = v
    return qvel


def get_heading_q(_q):
    """
    Parameters
    ----------
    _q : tensor of shape (4,)

    Returns
    -------
    q : tensor of shape (4,)

    """
    q = _q.clone()
    q[1] = 0.0
    q[2] = 0.0
    q_norm = torch.norm(q, p=2)
    return q / q_norm


def get_heading_q_batch(_q):
    """
    Parameters
    ----------
    _q : tensor of shape (BS, 4)

    Returns
    -------
    q : tensor of shape (BS, 4)

    """
    q = _q.clone()
    q[:, 1] = 0.0
    q[:, 2] = 0.0

    q_norm = torch.norm(q, dim=1, p=2).view(-1, 1)
    return q / q_norm


def de_heading(q):
    """
    Parameters
    ----------
    q : tensor of shape (4,)

    Returns
    -------
    q_deheaded : tensor of shape (4,)
    """
    return quat_multiply(quat_inv_t(get_heading_q(q)), q)


def de_heading_batch(q):
    """
    Parameters
    ----------
    q : tensor of shape (BS, 4)

    Returns
    -------
    q_deheaded : tensor of shape (BS, 4)
    """
    q_deheaded = get_heading_q_batch(q)
    q_deheaded_inv = quat_inv_t(q_deheaded)

    return quat_multiply_batch(q_deheaded_inv, q)


def get_heading(q):
    """
    Parameters
    ----------
    q : tensor of shape (4,)

    Returns
    -------
    heading : tensor of shape (1,)
    """
    hq = q.clone()
    hq[1] = 0
    hq[2] = 0
    if hq[3] < 0.0:
        hq = -1.0 * hq
    hq = hq / torch.norm(hq, p=2)
    w = 2 * safe_acos(hq[0])
    heading = torch.tensor([w], dtype=hq.dtype, device=hq.device)
    return heading


def get_angvel_fd(prev_bquat, cur_bquat, dt):
    """
    Parameters
    ----------
    prev_bquat : np.ndarray of shape (4,)
    cur_bquat : np.ndarray of shape (4,)
    dt : float

    Returns
    -------
    body_angvel : np.ndarray of shape (3,)
    """
    q_diff = multi_quat_diff(cur_bquat, prev_bquat)
    n_joint = q_diff.shape[0] // 4
    body_angvel = np.zeros(n_joint * 3)
    for i in range(n_joint):
        body_angvel[3 * i : 3 * i + 3] = aa_from_quat(q_diff[4 * i : 4 * i + 4]) / dt
    return body_angvel


def rand_heading():
    quat = rand_quat()
    return get_heading_q(quat)


def multi_quat_diff(nq1, nq0):
    """return the relative quats q1-q0 of N joints

    Parameters
    ----------
    nq1 : np.ndarray of shape (N, 4)
    nq0 : np.ndarray of shape (N, 4)

    Returns
    -------
    nq_diff : np.ndarray of shape (N, 4)
    """

    nq_diff = np.zeros_like(nq0)
    for i in range(nq1.shape[0] // 4):
        ind = slice(4 * i, 4 * i + 4)
        q1 = nq1[ind]
        q0 = nq0[ind]
        nq_diff[ind] = quat_multiply(q1, quat_inv_t(q0))
    return nq_diff


def multi_quat_norm(nq):
    """return the scalar rotation of a N joints"""

    nq_norm = np.arccos(np.clip(abs(nq[::4]), -1.0, 1.0))
    return nq_norm


def identity_matrix():
    """Return 4x4 identity/unit matrix.

    >>> I = identity_matrix()
    >>> numpy.allclose(I, numpy.dot(I, I))
    True
    >>> numpy.sum(I), numpy.trace(I)
    (4.0, 4.0)
    >>> numpy.allclose(I, numpy.identity(4))
    True

    """
    return numpy.identity(4)


def translation_matrix(direction):
    """Return matrix to translate by direction vector.

    >>> v = numpy.random.random(3) - 0.5
    >>> numpy.allclose(v, translation_matrix(v)[:3, 3])
    True

    """
    M = numpy.identity(4)
    M[:3, 3] = direction[:3]
    return M


def translation_from_matrix(matrix):
    """Return translation vector from translation matrix.

    >>> v0 = numpy.random.random(3) - 0.5
    >>> v1 = translation_from_matrix(translation_matrix(v0))
    >>> numpy.allclose(v0, v1)
    True

    """
    return numpy.array(matrix, copy=False)[:3, 3].copy()


def reflection_matrix(point, normal):
    """Return matrix to mirror at plane defined by point and normal vector.

    >>> v0 = numpy.random.random(4) - 0.5
    >>> v0[3] = 1.
    >>> v1 = numpy.random.random(3) - 0.5
    >>> R = reflection_matrix(v0, v1)
    >>> numpy.allclose(2, numpy.trace(R))
    True
    >>> numpy.allclose(v0, numpy.dot(R, v0))
    True
    >>> v2 = v0.copy()
    >>> v2[:3] += v1
    >>> v3 = v0.copy()
    >>> v2[:3] -= v1
    >>> numpy.allclose(v2, numpy.dot(R, v3))
    True

    """
    normal = unit_vector(normal[:3])
    M = numpy.identity(4)
    M[:3, :3] -= 2.0 * numpy.outer(normal, normal)
    M[:3, 3] = (2.0 * numpy.dot(point[:3], normal)) * normal
    return M


def reflection_from_matrix(matrix):
    """Return mirror plane point and normal vector from reflection matrix.

    >>> v0 = numpy.random.random(3) - 0.5
    >>> v1 = numpy.random.random(3) - 0.5
    >>> M0 = reflection_matrix(v0, v1)
    >>> point, normal = reflection_from_matrix(M0)
    >>> M1 = reflection_matrix(point, normal)
    >>> is_same_transform(M0, M1)
    True

    """
    M = numpy.array(matrix, dtype=numpy.float64, copy=False)
    # normal: unit eigenvector corresponding to eigenvalue -1
    w, V = numpy.linalg.eig(M[:3, :3])
    i = numpy.where(abs(numpy.real(w) + 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue -1")
    normal = numpy.real(V[:, i[0]]).squeeze()
    # point: any unit eigenvector corresponding to eigenvalue 1
    w, V = numpy.linalg.eig(M)
    i = numpy.where(abs(numpy.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    point = numpy.real(V[:, i[-1]]).squeeze()
    point /= point[3]
    return point, normal


def rotation_matrix(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction.

    >>> R = rotation_matrix(math.pi/2, [0, 0, 1], [1, 0, 0])
    >>> numpy.allclose(numpy.dot(R, [0, 0, 0, 1]), [1, -1, 0, 1])
    True
    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = numpy.random.random(3) - 0.5
    >>> point = numpy.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(-angle, -direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> I = numpy.identity(4, numpy.float64)
    >>> numpy.allclose(I, rotation_matrix(math.pi*2, direc))
    True
    >>> numpy.allclose(2, numpy.trace(rotation_matrix(math.pi/2,
    ...                                               direc, point)))
    True

    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = numpy.diag([cosa, cosa, cosa])
    R += numpy.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += numpy.array(
        [
            [0.0, -direction[2], direction[1]],
            [direction[2], 0.0, -direction[0]],
            [-direction[1], direction[0], 0.0],
        ]
    )
    M = numpy.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = numpy.array(point[:3], dtype=numpy.float64, copy=False)
        M[:3, 3] = point - numpy.dot(R, point)
    return M


def scale_matrix(factor, origin=None, direction=None):
    """Return matrix to scale by factor around origin in direction.

    Use factor -1 for point symmetry.

    >>> v = (numpy.random.rand(4, 5) - 0.5) * 20
    >>> v[3] = 1
    >>> S = scale_matrix(-1.234)
    >>> numpy.allclose(numpy.dot(S, v)[:3], -1.234*v[:3])
    True
    >>> factor = random.random() * 10 - 5
    >>> origin = numpy.random.random(3) - 0.5
    >>> direct = numpy.random.random(3) - 0.5
    >>> S = scale_matrix(factor, origin)
    >>> S = scale_matrix(factor, origin, direct)

    """
    if direction is None:
        # uniform scaling
        M = numpy.diag([factor, factor, factor, 1.0])
        if origin is not None:
            M[:3, 3] = origin[:3]
            M[:3, 3] *= 1.0 - factor
    else:
        # nonuniform scaling
        direction = unit_vector(direction[:3])
        factor = 1.0 - factor
        M = numpy.identity(4)
        M[:3, :3] -= factor * numpy.outer(direction, direction)
        if origin is not None:
            M[:3, 3] = (factor * numpy.dot(origin[:3], direction)) * direction
    return M


def scale_from_matrix(matrix):
    """Return scaling factor, origin and direction from scaling matrix.

    >>> factor = random.random() * 10 - 5
    >>> origin = numpy.random.random(3) - 0.5
    >>> direct = numpy.random.random(3) - 0.5
    >>> S0 = scale_matrix(factor, origin)
    >>> factor, origin, direction = scale_from_matrix(S0)
    >>> S1 = scale_matrix(factor, origin, direction)
    >>> is_same_transform(S0, S1)
    True
    >>> S0 = scale_matrix(factor, origin, direct)
    >>> factor, origin, direction = scale_from_matrix(S0)
    >>> S1 = scale_matrix(factor, origin, direction)
    >>> is_same_transform(S0, S1)
    True

    """
    M = numpy.array(matrix, dtype=numpy.float64, copy=False)
    M33 = M[:3, :3]
    factor = numpy.trace(M33) - 2.0
    try:
        # direction: unit eigenvector corresponding to eigenvalue factor
        w, V = numpy.linalg.eig(M33)
        i = numpy.where(abs(numpy.real(w) - factor) < 1e-8)[0][0]
        direction = numpy.real(V[:, i]).squeeze()
        direction /= vector_norm(direction)
    except IndexError:
        # uniform scaling
        factor = (factor + 2.0) / 3.0
        direction = None
    # origin: any eigenvector corresponding to eigenvalue 1
    w, V = numpy.linalg.eig(M)
    i = numpy.where(abs(numpy.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no eigenvector corresponding to eigenvalue 1")
    origin = numpy.real(V[:, i[-1]]).squeeze()
    origin /= origin[3]
    return factor, origin, direction


def projection_matrix(point, normal, direction=None, perspective=None, pseudo=False):
    """Return matrix to project onto plane defined by point and normal.

    Using either perspective point, projection direction, or none of both.

    If pseudo is True, perspective projections will preserve relative depth
    such that Perspective = dot(Orthogonal, PseudoPerspective).

    >>> P = projection_matrix([0, 0, 0], [1, 0, 0])
    >>> numpy.allclose(P[1:, 1:], numpy.identity(4)[1:, 1:])
    True
    >>> point = numpy.random.random(3) - 0.5
    >>> normal = numpy.random.random(3) - 0.5
    >>> direct = numpy.random.random(3) - 0.5
    >>> persp = numpy.random.random(3) - 0.5
    >>> P0 = projection_matrix(point, normal)
    >>> P1 = projection_matrix(point, normal, direction=direct)
    >>> P2 = projection_matrix(point, normal, perspective=persp)
    >>> P3 = projection_matrix(point, normal, perspective=persp, pseudo=True)
    >>> is_same_transform(P2, numpy.dot(P0, P3))
    True
    >>> P = projection_matrix([3, 0, 0], [1, 1, 0], [1, 0, 0])
    >>> v0 = (numpy.random.rand(4, 5) - 0.5) * 20
    >>> v0[3] = 1
    >>> v1 = numpy.dot(P, v0)
    >>> numpy.allclose(v1[1], v0[1])
    True
    >>> numpy.allclose(v1[0], 3-v1[1])
    True

    """
    M = numpy.identity(4)
    point = numpy.array(point[:3], dtype=numpy.float64, copy=False)
    normal = unit_vector(normal[:3])
    if perspective is not None:
        # perspective projection
        perspective = numpy.array(perspective[:3], dtype=numpy.float64, copy=False)
        M[0, 0] = M[1, 1] = M[2, 2] = numpy.dot(perspective - point, normal)
        M[:3, :3] -= numpy.outer(perspective, normal)
        if pseudo:
            # preserve relative depth
            M[:3, :3] -= numpy.outer(normal, normal)
            M[:3, 3] = numpy.dot(point, normal) * (perspective + normal)
        else:
            M[:3, 3] = numpy.dot(point, normal) * perspective
        M[3, :3] = -normal
        M[3, 3] = numpy.dot(perspective, normal)
    elif direction is not None:
        # parallel projection
        direction = numpy.array(direction[:3], dtype=numpy.float64, copy=False)
        scale = numpy.dot(direction, normal)
        M[:3, :3] -= numpy.outer(direction, normal) / scale
        M[:3, 3] = direction * (numpy.dot(point, normal) / scale)
    else:
        # orthogonal projection
        M[:3, :3] -= numpy.outer(normal, normal)
        M[:3, 3] = numpy.dot(point, normal) * normal
    return M


def projection_from_matrix(matrix, pseudo=False):
    """Return projection plane and perspective point from projection matrix.

    Return values are same as arguments for projection_matrix function:
    point, normal, direction, perspective, and pseudo.

    >>> point = numpy.random.random(3) - 0.5
    >>> normal = numpy.random.random(3) - 0.5
    >>> direct = numpy.random.random(3) - 0.5
    >>> persp = numpy.random.random(3) - 0.5
    >>> P0 = projection_matrix(point, normal)
    >>> result = projection_from_matrix(P0)
    >>> P1 = projection_matrix(*result)
    >>> is_same_transform(P0, P1)
    True
    >>> P0 = projection_matrix(point, normal, direct)
    >>> result = projection_from_matrix(P0)
    >>> P1 = projection_matrix(*result)
    >>> is_same_transform(P0, P1)
    True
    >>> P0 = projection_matrix(point, normal, perspective=persp, pseudo=False)
    >>> result = projection_from_matrix(P0, pseudo=False)
    >>> P1 = projection_matrix(*result)
    >>> is_same_transform(P0, P1)
    True
    >>> P0 = projection_matrix(point, normal, perspective=persp, pseudo=True)
    >>> result = projection_from_matrix(P0, pseudo=True)
    >>> P1 = projection_matrix(*result)
    >>> is_same_transform(P0, P1)
    True

    """
    M = numpy.array(matrix, dtype=numpy.float64, copy=False)
    M33 = M[:3, :3]
    w, V = numpy.linalg.eig(M)
    i = numpy.where(abs(numpy.real(w) - 1.0) < 1e-8)[0]
    if not pseudo and len(i):
        # point: any eigenvector corresponding to eigenvalue 1
        point = numpy.real(V[:, i[-1]]).squeeze()
        point /= point[3]
        # direction: unit eigenvector corresponding to eigenvalue 0
        w, V = numpy.linalg.eig(M33)
        i = numpy.where(abs(numpy.real(w)) < 1e-8)[0]
        if not len(i):
            raise ValueError("no eigenvector corresponding to eigenvalue 0")
        direction = numpy.real(V[:, i[0]]).squeeze()
        direction /= vector_norm(direction)
        # normal: unit eigenvector of M33.T corresponding to eigenvalue 0
        w, V = numpy.linalg.eig(M33.T)
        i = numpy.where(abs(numpy.real(w)) < 1e-8)[0]
        if len(i):
            # parallel projection
            normal = numpy.real(V[:, i[0]]).squeeze()
            normal /= vector_norm(normal)
            return point, normal, direction, None, False
        else:
            # orthogonal projection, where normal equals direction vector
            return point, direction, None, None, False
    else:
        # perspective projection
        i = numpy.where(abs(numpy.real(w)) > 1e-8)[0]
        if not len(i):
            raise ValueError("no eigenvector not corresponding to eigenvalue 0")
        point = numpy.real(V[:, i[-1]]).squeeze()
        point /= point[3]
        normal = -M[3, :3]
        perspective = M[:3, 3] / numpy.dot(point[:3], normal)
        if pseudo:
            perspective -= normal
        return point, normal, None, perspective, pseudo


def clip_matrix(left, right, bottom, top, near, far, perspective=False):
    """Return matrix to obtain normalized device coordinates from frustum.

    The frustum bounds are axis-aligned along x (left, right),
    y (bottom, top) and z (near, far).

    Normalized device coordinates are in range [-1, 1] if coordinates are
    inside the frustum.

    If perspective is True the frustum is a truncated pyramid with the
    perspective point at origin and direction along z axis, otherwise an
    orthographic canonical view volume (a box).

    Homogeneous coordinates transformed by the perspective clip matrix
    need to be dehomogenized (divided by w coordinate).

    >>> frustum = numpy.random.rand(6)
    >>> frustum[1] += frustum[0]
    >>> frustum[3] += frustum[2]
    >>> frustum[5] += frustum[4]
    >>> M = clip_matrix(perspective=False, *frustum)
    >>> numpy.dot(M, [frustum[0], frustum[2], frustum[4], 1])
    array([-1., -1., -1.,  1.])
    >>> numpy.dot(M, [frustum[1], frustum[3], frustum[5], 1])
    array([1., 1., 1., 1.])
    >>> M = clip_matrix(perspective=True, *frustum)
    >>> v = numpy.dot(M, [frustum[0], frustum[2], frustum[4], 1])
    >>> v / v[3]
    array([-1., -1., -1.,  1.])
    >>> v = numpy.dot(M, [frustum[1], frustum[3], frustum[4], 1])
    >>> v / v[3]
    array([ 1.,  1., -1.,  1.])

    """
    if left >= right or bottom >= top or near >= far:
        raise ValueError("invalid frustum")
    if perspective:
        if near <= _EPS:
            raise ValueError("invalid frustum: near <= 0")
        t = 2.0 * near
        M = [
            [t / (left - right), 0.0, (right + left) / (right - left), 0.0],
            [0.0, t / (bottom - top), (top + bottom) / (top - bottom), 0.0],
            [0.0, 0.0, (far + near) / (near - far), t * far / (far - near)],
            [0.0, 0.0, -1.0, 0.0],
        ]
    else:
        M = [
            [2.0 / (right - left), 0.0, 0.0, (right + left) / (left - right)],
            [0.0, 2.0 / (top - bottom), 0.0, (top + bottom) / (bottom - top)],
            [0.0, 0.0, 2.0 / (far - near), (far + near) / (near - far)],
            [0.0, 0.0, 0.0, 1.0],
        ]
    return numpy.array(M)


def shear_matrix(angle, direction, point, normal):
    """Return matrix to shear by angle along direction vector on shear plane.

    The shear plane is defined by a point and normal vector. The direction
    vector must be orthogonal to the plane's normal vector.

    A point P is transformed by the shear matrix into P" such that
    the vector P-P" is parallel to the direction vector and its extent is
    given by the angle of P-P'-P", where P' is the orthogonal projection
    of P onto the shear plane.

    >>> angle = (random.random() - 0.5) * 4*math.pi
    >>> direct = numpy.random.random(3) - 0.5
    >>> point = numpy.random.random(3) - 0.5
    >>> normal = numpy.cross(direct, numpy.random.random(3))
    >>> S = shear_matrix(angle, direct, point, normal)
    >>> numpy.allclose(1, numpy.linalg.det(S))
    True

    """
    normal = unit_vector(normal[:3])
    direction = unit_vector(direction[:3])
    if abs(numpy.dot(normal, direction)) > 1e-6:
        raise ValueError("direction and normal vectors are not orthogonal")
    angle = math.tan(angle)
    M = numpy.identity(4)
    M[:3, :3] += angle * numpy.outer(direction, normal)
    M[:3, 3] = -angle * numpy.dot(point[:3], normal) * direction
    return M


def shear_from_matrix(matrix):
    """Return shear angle, direction and plane from shear matrix.

    >>> angle = (random.random() - 0.5) * 4*math.pi
    >>> direct = numpy.random.random(3) - 0.5
    >>> point = numpy.random.random(3) - 0.5
    >>> normal = numpy.cross(direct, numpy.random.random(3))
    >>> S0 = shear_matrix(angle, direct, point, normal)
    >>> angle, direct, point, normal = shear_from_matrix(S0)
    >>> S1 = shear_matrix(angle, direct, point, normal)
    >>> is_same_transform(S0, S1)
    True

    """
    M = numpy.array(matrix, dtype=numpy.float64, copy=False)
    M33 = M[:3, :3]
    # normal: cross independent eigenvectors corresponding to the eigenvalue 1
    w, V = numpy.linalg.eig(M33)
    i = numpy.where(abs(numpy.real(w) - 1.0) < 1e-4)[0]
    if len(i) < 2:
        raise ValueError("no two linear independent eigenvectors found %s" % w)
    V = numpy.real(V[:, i]).squeeze().T
    lenorm = -1.0
    for i0, i1 in ((0, 1), (0, 2), (1, 2)):
        n = numpy.cross(V[i0], V[i1])
        w = vector_norm(n)
        if w > lenorm:
            lenorm = w
            normal = n
    normal /= lenorm
    # direction and angle
    direction = numpy.dot(M33 - numpy.identity(3), normal)
    angle = vector_norm(direction)
    direction /= angle
    angle = math.atan(angle)
    # point: eigenvector corresponding to eigenvalue 1
    w, V = numpy.linalg.eig(M)
    i = numpy.where(abs(numpy.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no eigenvector corresponding to eigenvalue 1")
    point = numpy.real(V[:, i[-1]]).squeeze()
    point /= point[3]
    return angle, direction, point, normal


def decompose_matrix(matrix):
    """Return sequence of transformations from transformation matrix.

    matrix : array_like
        Non-degenerative homogeneous transformation matrix

    Return tuple of:
        scale : vector of 3 scaling factors
        shear : list of shear factors for x-y, x-z, y-z axes
        angles : list of Euler angles about static x, y, z axes
        translate : translation vector along x, y, z axes
        perspective : perspective partition of matrix

    Raise ValueError if matrix is of wrong type or degenerative.

    >>> T0 = translation_matrix([1, 2, 3])
    >>> scale, shear, angles, trans, persp = decompose_matrix(T0)
    >>> T1 = translation_matrix(trans)
    >>> numpy.allclose(T0, T1)
    True
    >>> S = scale_matrix(0.123)
    >>> scale, shear, angles, trans, persp = decompose_matrix(S)
    >>> scale[0]
    0.123
    >>> R0 = euler_matrix(1, 2, 3)
    >>> scale, shear, angles, trans, persp = decompose_matrix(R0)
    >>> R1 = euler_matrix(*angles)
    >>> numpy.allclose(R0, R1)
    True

    """
    M = numpy.array(matrix, dtype=numpy.float64, copy=True).T
    if abs(M[3, 3]) < _EPS:
        raise ValueError("M[3, 3] is zero")
    M /= M[3, 3]
    P = M.copy()
    P[:, 3] = 0.0, 0.0, 0.0, 1.0
    if not numpy.linalg.det(P):
        raise ValueError("matrix is singular")

    scale = numpy.zeros((3,))
    shear = [0.0, 0.0, 0.0]
    angles = [0.0, 0.0, 0.0]

    if any(abs(M[:3, 3]) > _EPS):
        perspective = numpy.dot(M[:, 3], numpy.linalg.inv(P.T))
        M[:, 3] = 0.0, 0.0, 0.0, 1.0
    else:
        perspective = numpy.array([0.0, 0.0, 0.0, 1.0])

    translate = M[3, :3].copy()
    M[3, :3] = 0.0

    row = M[:3, :3].copy()
    scale[0] = vector_norm(row[0])
    row[0] /= scale[0]
    shear[0] = numpy.dot(row[0], row[1])
    row[1] -= row[0] * shear[0]
    scale[1] = vector_norm(row[1])
    row[1] /= scale[1]
    shear[0] /= scale[1]
    shear[1] = numpy.dot(row[0], row[2])
    row[2] -= row[0] * shear[1]
    shear[2] = numpy.dot(row[1], row[2])
    row[2] -= row[1] * shear[2]
    scale[2] = vector_norm(row[2])
    row[2] /= scale[2]
    shear[1:] /= scale[2]

    if numpy.dot(row[0], numpy.cross(row[1], row[2])) < 0:
        numpy.negative(scale, scale)
        numpy.negative(row, row)

    angles[1] = math.asin(-row[0, 2])
    if math.cos(angles[1]):
        angles[0] = math.atan2(row[1, 2], row[2, 2])
        angles[2] = math.atan2(row[0, 1], row[0, 0])
    else:
        # angles[0] = math.atan2(row[1, 0], row[1, 1])
        angles[0] = math.atan2(-row[2, 1], row[1, 1])
        angles[2] = 0.0

    return scale, shear, angles, translate, perspective


def compose_matrix(
    scale=None, shear=None, angles=None, translate=None, perspective=None
):
    """Return transformation matrix from sequence of transformations.

    This is the inverse of the decompose_matrix function.

    Sequence of transformations:
        scale : vector of 3 scaling factors
        shear : list of shear factors for x-y, x-z, y-z axes
        angles : list of Euler angles about static x, y, z axes
        translate : translation vector along x, y, z axes
        perspective : perspective partition of matrix

    >>> scale = numpy.random.random(3) - 0.5
    >>> shear = numpy.random.random(3) - 0.5
    >>> angles = (numpy.random.random(3) - 0.5) * (2*math.pi)
    >>> trans = numpy.random.random(3) - 0.5
    >>> persp = numpy.random.random(4) - 0.5
    >>> M0 = compose_matrix(scale, shear, angles, trans, persp)
    >>> result = decompose_matrix(M0)
    >>> M1 = compose_matrix(*result)
    >>> is_same_transform(M0, M1)
    True

    """
    M = numpy.identity(4)
    if perspective is not None:
        P = numpy.identity(4)
        P[3, :] = perspective[:4]
        M = numpy.dot(M, P)
    if translate is not None:
        T = numpy.identity(4)
        T[:3, 3] = translate[:3]
        M = numpy.dot(M, T)
    if angles is not None:
        R = euler_matrix(angles[0], angles[1], angles[2], "sxyz")
        M = numpy.dot(M, R)
    if shear is not None:
        Z = numpy.identity(4)
        Z[1, 2] = shear[2]
        Z[0, 2] = shear[1]
        Z[0, 1] = shear[0]
        M = numpy.dot(M, Z)
    if scale is not None:
        S = numpy.identity(4)
        S[0, 0] = scale[0]
        S[1, 1] = scale[1]
        S[2, 2] = scale[2]
        M = numpy.dot(M, S)
    M /= M[3, 3]
    return M


def orthogonalization_matrix(lengths, angles):
    """Return orthogonalization matrix for crystallographic cell coordinates.

    Angles are expected in degrees.

    The de-orthogonalization matrix is the inverse.

    >>> O = orthogonalization_matrix([10, 10, 10], [90, 90, 90])
    >>> numpy.allclose(O[:3, :3], numpy.identity(3, float) * 10)
    True
    >>> O = orthogonalization_matrix([9.8, 12.0, 15.5], [87.2, 80.7, 69.7])
    >>> numpy.allclose(numpy.sum(O), 43.063229)
    True

    """
    a, b, c = lengths
    angles = numpy.radians(angles)
    sina, sinb, _ = numpy.sin(angles)
    cosa, cosb, cosg = numpy.cos(angles)
    co = (cosa * cosb - cosg) / (sina * sinb)
    return numpy.array(
        [
            [a * sinb * math.sqrt(1.0 - co * co), 0.0, 0.0, 0.0],
            [-a * sinb * co, b * sina, 0.0, 0.0],
            [a * cosb, b * cosa, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def affine_matrix_from_points(v0, v1, shear=True, scale=True, usesvd=True):
    """Return affine transform matrix to register two point sets.

    v0 and v1 are shape (ndims, \*) arrays of at least ndims non-homogeneous
    coordinates, where ndims is the dimensionality of the coordinate space.

    If shear is False, a similarity transformation matrix is returned.
    If also scale is False, a rigid/Euclidean transformation matrix
    is returned.

    By default the algorithm by Hartley and Zissermann [15] is used.
    If usesvd is True, similarity and Euclidean transformation matrices
    are calculated by minimizing the weighted sum of squared deviations
    (RMSD) according to the algorithm by Kabsch [8].
    Otherwise, and if ndims is 3, the quat based algorithm by Horn [9]
    is used, which is slower when using this Python implementation.

    The returned matrix performs rotation, translation and uniform scaling
    (if specified).

    >>> v0 = [[0, 1031, 1031, 0], [0, 0, 1600, 1600]]
    >>> v1 = [[675, 826, 826, 677], [55, 52, 281, 277]]
    >>> affine_matrix_from_points(v0, v1)
    array([[  0.14549,   0.00062, 675.50008],
           [  0.00048,   0.14094,  53.24971],
           [  0.     ,   0.     ,   1.     ]])
    >>> T = translation_matrix(numpy.random.random(3)-0.5)
    >>> R = random_rotation_matrix(numpy.random.random(3))
    >>> S = scale_matrix(random.random())
    >>> M = concatenate_matrices(T, R, S)
    >>> v0 = (numpy.random.rand(4, 100) - 0.5) * 20
    >>> v0[3] = 1
    >>> v1 = numpy.dot(M, v0)
    >>> v0[:3] += numpy.random.normal(0, 1e-8, 300).reshape(3, -1)
    >>> M = affine_matrix_from_points(v0[:3], v1[:3])
    >>> numpy.allclose(v1, numpy.dot(M, v0))
    True

    More examples in superimposition_matrix()

    """
    v0 = numpy.array(v0, dtype=numpy.float64, copy=True)
    v1 = numpy.array(v1, dtype=numpy.float64, copy=True)

    ndims = v0.shape[0]
    if ndims < 2 or v0.shape[1] < ndims or v0.shape != v1.shape:
        raise ValueError("input arrays are of wrong shape or type")

    # move centroids to origin
    t0 = -numpy.mean(v0, axis=1)
    M0 = numpy.identity(ndims + 1)
    M0[:ndims, ndims] = t0
    v0 += t0.reshape(ndims, 1)
    t1 = -numpy.mean(v1, axis=1)
    M1 = numpy.identity(ndims + 1)
    M1[:ndims, ndims] = t1
    v1 += t1.reshape(ndims, 1)

    if shear:
        # Affine transformation
        A = numpy.concatenate((v0, v1), axis=0)
        u, s, vh = numpy.linalg.svd(A.T)
        vh = vh[:ndims].T
        B = vh[:ndims]
        C = vh[ndims : 2 * ndims]
        t = numpy.dot(C, numpy.linalg.pinv(B))
        t = numpy.concatenate((t, numpy.zeros((ndims, 1))), axis=1)
        M = numpy.vstack((t, ((0.0,) * ndims) + (1.0,)))
    elif usesvd or ndims != 3:
        # Rigid transformation via SVD of covariance matrix
        u, s, vh = numpy.linalg.svd(numpy.dot(v1, v0.T))
        # rotation matrix from SVD orthonormal bases
        R = numpy.dot(u, vh)
        if numpy.linalg.det(R) < 0.0:
            # R does not constitute right handed system
            R -= numpy.outer(u[:, ndims - 1], vh[ndims - 1, :] * 2.0)
            s[-1] *= -1.0
        # homogeneous transformation matrix
        M = numpy.identity(ndims + 1)
        M[:ndims, :ndims] = R
    else:
        # Rigid transformation matrix via quat
        # compute symmetric matrix N
        xx, yy, zz = numpy.sum(v0 * v1, axis=1)
        xy, yz, zx = numpy.sum(v0 * numpy.roll(v1, -1, axis=0), axis=1)
        xz, yx, zy = numpy.sum(v0 * numpy.roll(v1, -2, axis=0), axis=1)
        N = [
            [xx + yy + zz, 0.0, 0.0, 0.0],
            [yz - zy, xx - yy - zz, 0.0, 0.0],
            [zx - xz, xy + yx, yy - xx - zz, 0.0],
            [xy - yx, zx + xz, yz + zy, zz - xx - yy],
        ]
        # quat: eigenvector corresponding to most positive eigenvalue
        w, V = numpy.linalg.eigh(N)
        q = V[:, numpy.argmax(w)]
        q /= vector_norm(q)  # unit quat
        # homogeneous transformation matrix
        M = quat_to_rotMat(q)

    if scale and not shear:
        # Affine transformation; scale is ratio of RMS deviations from centroid
        v0 *= v0
        v1 *= v1
        M[:ndims, :ndims] *= math.sqrt(numpy.sum(v1) / numpy.sum(v0))

    # move centroids back
    M = numpy.dot(numpy.linalg.inv(M1), numpy.dot(M, M0))
    M /= M[ndims, ndims]
    return M


def superimposition_matrix(v0, v1, scale=False, usesvd=True):
    """Return matrix to transform given 3D point set into second point set.

    v0 and v1 are shape (3, \*) or (4, \*) arrays of at least 3 points.

    The parameters scale and usesvd are explained in the more general
    affine_matrix_from_points function.

    The returned matrix is a similarity or Euclidean transformation matrix.
    This function has a fast C implementation in transformations.c.

    >>> v0 = numpy.random.rand(3, 10)
    >>> M = superimposition_matrix(v0, v0)
    >>> numpy.allclose(M, numpy.identity(4))
    True
    >>> R = random_rotation_matrix(numpy.random.random(3))
    >>> v0 = [[1,0,0], [0,1,0], [0,0,1], [1,1,1]]
    >>> v1 = numpy.dot(R, v0)
    >>> M = superimposition_matrix(v0, v1)
    >>> numpy.allclose(v1, numpy.dot(M, v0))
    True
    >>> v0 = (numpy.random.rand(4, 100) - 0.5) * 20
    >>> v0[3] = 1
    >>> v1 = numpy.dot(R, v0)
    >>> M = superimposition_matrix(v0, v1)
    >>> numpy.allclose(v1, numpy.dot(M, v0))
    True
    >>> S = scale_matrix(random.random())
    >>> T = translation_matrix(numpy.random.random(3)-0.5)
    >>> M = concatenate_matrices(T, R, S)
    >>> v1 = numpy.dot(M, v0)
    >>> v0[:3] += numpy.random.normal(0, 1e-9, 300).reshape(3, -1)
    >>> M = superimposition_matrix(v0, v1, scale=True)
    >>> numpy.allclose(v1, numpy.dot(M, v0))
    True
    >>> M = superimposition_matrix(v0, v1, scale=True, usesvd=False)
    >>> numpy.allclose(v1, numpy.dot(M, v0))
    True
    >>> v = numpy.empty((4, 100, 3))
    >>> v[:, :, 0] = v0
    >>> M = superimposition_matrix(v0, v1, scale=True, usesvd=False)
    >>> numpy.allclose(v1, numpy.dot(M, v[:, :, 0]))
    True

    """
    v0 = numpy.array(v0, dtype=numpy.float64, copy=False)[:3]
    v1 = numpy.array(v1, dtype=numpy.float64, copy=False)[:3]
    return affine_matrix_from_points(v0, v1, shear=False, scale=scale, usesvd=usesvd)


def random_rotation_matrix(rand=None):
    """Return uniform random rotation matrix.

    rand: array like
        Three independent random variables that are uniformly distributed
        between 0 and 1 for each returned quat.

    >>> R = random_rotation_matrix()
    >>> numpy.allclose(numpy.dot(R.T, R), numpy.identity(4))
    True

    """
    return quat_to_rotMat(rand_quat(rand))


def vector_norm(data, axis=None, out=None):
    """Return length, i.e. Euclidean norm, of ndarray along axis.

    >>> v = numpy.random.random(3)
    >>> n = vector_norm(v)
    >>> numpy.allclose(n, numpy.linalg.norm(v))
    True
    >>> v = numpy.random.rand(6, 5, 3)
    >>> n = vector_norm(v, axis=-1)
    >>> numpy.allclose(n, numpy.sqrt(numpy.sum(v*v, axis=2)))
    True
    >>> n = vector_norm(v, axis=1)
    >>> numpy.allclose(n, numpy.sqrt(numpy.sum(v*v, axis=1)))
    True
    >>> v = numpy.random.rand(5, 4, 3)
    >>> n = numpy.empty((5, 3))
    >>> vector_norm(v, axis=1, out=n)
    >>> numpy.allclose(n, numpy.sqrt(numpy.sum(v*v, axis=1)))
    True
    >>> vector_norm([])
    0.0
    >>> vector_norm([1])
    1.0

    """
    data = numpy.array(data, dtype=numpy.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(numpy.dot(data, data))
        data *= data
        out = numpy.atleast_1d(numpy.sum(data, axis=axis))
        numpy.sqrt(out, out)
        return out
    else:
        data *= data
        numpy.sum(data, axis=axis, out=out)
        numpy.sqrt(out, out)


def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.

    >>> v0 = np.random.random(3)
    >>> v1 = unit_vector(v0)
    >>> np.allclose(v1, v0 / np.linalg.norm(v0))
    True
    >>> v0 = np.random.rand(5, 4, 3)
    >>> v1 = unit_vector(v0, axis=-1)
    >>> v2 = v0 / np.expand_dims(np.sqrt(np.sum(v0*v0, axis=2)), 2)
    >>> np.allclose(v1, v2)
    True
    >>> v1 = unit_vector(v0, axis=1)
    >>> v2 = v0 / np.expand_dims(np.sqrt(np.sum(v0*v0, axis=1)), 1)
    >>> np.allclose(v1, v2)
    True
    >>> v1 = np.empty((5, 4, 3))
    >>> unit_vector(v0, axis=1, out=v1)
    >>> np.allclose(v1, v2)
    True
    >>> list(unit_vector([]))
    []
    >>> list(unit_vector([1]))
    [1.0]

    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def random_vector(size):
    """Return array of random doubles in the half-open interval [0.0, 1.0).

    >>> v = random_vector(10000)
    >>> numpy.all(v >= 0) and numpy.all(v < 1)
    True
    >>> v0 = random_vector(10)
    >>> v1 = random_vector(10)
    >>> numpy.any(v0 == v1)
    False

    """
    return numpy.random.random(size)


def vector_product(v0, v1, axis=0):
    """Return vector perpendicular to vectors.

    >>> v = vector_product([2, 0, 0], [0, 3, 0])
    >>> numpy.allclose(v, [0, 0, 6])
    True
    >>> v0 = [[2, 0, 0, 2], [0, 2, 0, 2], [0, 0, 2, 2]]
    >>> v1 = [[3], [0], [0]]
    >>> v = vector_product(v0, v1)
    >>> numpy.allclose(v, [[0, 0, 0, 0], [0, 0, 6, 6], [0, -6, 0, -6]])
    True
    >>> v0 = [[2, 0, 0], [2, 0, 0], [0, 2, 0], [2, 0, 0]]
    >>> v1 = [[0, 3, 0], [0, 0, 3], [0, 0, 3], [3, 3, 3]]
    >>> v = vector_product(v0, v1, axis=1)
    >>> numpy.allclose(v, [[0, 0, 6], [0, -6, 0], [6, 0, 0], [0, -6, 6]])
    True

    """
    return numpy.cross(v0, v1, axis=axis)


def angle_between_vectors(v0, v1, directed=True, axis=0):
    """Return angle between vectors.

    If directed is False, the input vectors are interpreted as undirected axes,
    i.e. the maximum angle is pi/2.

    >>> a = angle_between_vectors([1, -2, 3], [-1, 2, -3])
    >>> numpy.allclose(a, math.pi)
    True
    >>> a = angle_between_vectors([1, -2, 3], [-1, 2, -3], directed=False)
    >>> numpy.allclose(a, 0)
    True
    >>> v0 = [[2, 0, 0, 2], [0, 2, 0, 2], [0, 0, 2, 2]]
    >>> v1 = [[3], [0], [0]]
    >>> a = angle_between_vectors(v0, v1)
    >>> numpy.allclose(a, [0, 1.5708, 1.5708, 0.95532])
    True
    >>> v0 = [[2, 0, 0], [2, 0, 0], [0, 2, 0], [2, 0, 0]]
    >>> v1 = [[0, 3, 0], [0, 0, 3], [0, 0, 3], [3, 3, 3]]
    >>> a = angle_between_vectors(v0, v1, axis=1)
    >>> numpy.allclose(a, [1.5708, 1.5708, 1.5708, 0.95532])
    True

    """
    v0 = numpy.array(v0, dtype=numpy.float64, copy=False)
    v1 = numpy.array(v1, dtype=numpy.float64, copy=False)
    dot = numpy.sum(v0 * v1, axis=axis)
    dot /= vector_norm(v0, axis=axis) * vector_norm(v1, axis=axis)
    return numpy.arccos(dot if directed else numpy.fabs(dot))


def inverse_matrix(matrix):
    """Return inverse of square transformation matrix.

    >>> M0 = random_rotation_matrix()
    >>> M1 = inverse_matrix(M0.T)
    >>> numpy.allclose(M1, numpy.linalg.inv(M0.T))
    True
    >>> for size in range(1, 7):
    ...     M0 = numpy.random.rand(size, size)
    ...     M1 = inverse_matrix(M0)
    ...     if not numpy.allclose(M1, numpy.linalg.inv(M0)): print(size)

    """
    return numpy.linalg.inv(matrix)


def concatenate_matrices(*matrices):
    """Return concatenation of series of transformation matrices.

    >>> M = numpy.random.rand(16).reshape((4, 4)) - 0.5
    >>> numpy.allclose(M, concatenate_matrices(M))
    True
    >>> numpy.allclose(numpy.dot(M, M.T), concatenate_matrices(M, M.T))
    True

    """
    M = numpy.identity(4)
    for i in matrices:
        M = numpy.dot(M, i)
    return M


def is_same_transform(matrix0, matrix1):
    """Return True if two matrices perform same transformation.

    >>> is_same_transform(numpy.identity(4), numpy.identity(4))
    True
    >>> is_same_transform(numpy.identity(4), random_rotation_matrix())
    False

    """
    matrix0 = numpy.array(matrix0, dtype=numpy.float64, copy=True)
    matrix0 /= matrix0[3, 3]
    matrix1 = numpy.array(matrix1, dtype=numpy.float64, copy=True)
    matrix1 /= matrix1[3, 3]
    return numpy.allclose(matrix0, matrix1)


def is_same_quat(q0, q1):
    """Return True if two quats are equal."""
    q0 = numpy.array(q0)
    q1 = numpy.array(q1)
    return numpy.allclose(q0, q1) or numpy.allclose(q0, -q1)


def T_to_qpose(T, take_inv=False):
    """Convert the transformation matrix to pose (translation and rotation)

    Parameters
    ------
    T : numpy array, T x 3 x 4

    Returns
    ------
    pose : numpy array, T x 7

    Notes
    -----
    the convention of quat is `wxyz` since pytorch3d uses this convention.

    Examples
    --------
    >>> T = np.random.rand(10, 3, 4)
    >>> pose = T_to_qpose(T)
    >>> pose.shape
    (10, 7)

    """

    T_ = T
    T_ = np.tile(np.eye(4), (T.shape[0], 1, 1))
    T_[:, :3, :4] = T
    if take_inv:
        T_ = np.linalg.inv(T_)
    trans = T_[:, :3, 3]  # T x 3
    rot_mat = T_[:, :3, :3]  # T x 3 x 3
    quat_wxyz = (
        transforms.matrix_to_quaternion(torch.from_numpy(rot_mat).float())
        .data.cpu()
        .numpy()
    )  # T x 4
    pose = np.concatenate([trans, quat_wxyz], axis=1)  # T x 7
    return pose


def qpose_to_T(pose):
    """Convert the pose to transformation matrix (translation and rotation)

    Parameters
    ------
    pose : numpy array, T x 7

    Returns
    ------
    T : numpy array, T x 3 x 4

    Notes
    -----
    the convention of quat is `wxyz` since pytorch3d uses this convention.

    Examples
    --------
    >>> pose = np.random.rand(10, 7)
     >>> T = qpose_to_T(pose)
    >>> T.shape
    (10, 3, 4)

    """
    tt, *_ = pose.shape
    trans, quat_wxyz = pose[:, :3], pose[:, 3:]  # T x 3, T x 4
    rot_mat = (
        transforms.quaternion_to_matrix(torch.from_numpy(quat_wxyz).float())
        .data.cpu()
        .numpy()
    )  # T x 3 x 3
    T = np.tile(np.eye(4), (tt, 1, 1))
    T[:, :3, :3] = rot_mat
    T[:, :3, 3] = trans
    return T[:, :3, :4]


def _import_module(name, package=None, warn=True, prefix="_py_", ignore="_"):
    """Try import all public attributes from module into global namespace.

    Existing attributes with name clashes are renamed with prefix.
    Attributes starting with underscore are ignored by default.

    Return True on successful import.

    """
    import warnings
    from importlib import import_module

    try:
        if not package:
            module = import_module(name)
        else:
            module = import_module("." + name, package=package)
    except ImportError:
        if warn:
            warnings.warn("failed to import module %s" % name)
    else:
        for attr in dir(module):
            if ignore and attr.startswith(ignore):
                continue
            if prefix:
                if attr in globals():
                    globals()[prefix + attr] = globals()[attr]
                elif warn:
                    warnings.warn("no Python implementation of " + attr)
            globals()[attr] = getattr(module, attr)
        return True


class RadToDeg(nn.Module):
    r"""Creates an object that converts angles from radians to degrees.

    Args:
        tensor (Tensor): Tensor of arbitrary shape.

    Returns:
        Tensor: Tensor with same shape as input.

    Examples::

        >>> input = PI * torch.rand(1, 3, 3)
        >>> output = RadToDeg()(input)
    """

    def __init__(self):
        super(RadToDeg, self).__init__()

    def forward(self, input):
        return rad2deg(input)


class DegToRad(nn.Module):
    r"""Function that converts angles from degrees to radians.

    Args:
        tensor (Tensor): Tensor of arbitrary shape.

    Returns:
        Tensor: Tensor with same shape as input.

    Examples::

        >>> input = 360. * torch.rand(1, 3, 3)
        >>> output = DegToRad()(input)
    """

    def __init__(self):
        super(DegToRad, self).__init__()

    def forward(self, input):
        return deg2rad(input)


class ConvertPointsFromHomogeneous(nn.Module):
    r"""Creates a transformation that converts points from homogeneous to
    Euclidean space.

    Args:
        points (Tensor): tensor of N-dimensional points.

    Returns:
        Tensor: tensor of N-1-dimensional points.

    Shape:
        - Input: :math:`(B, D, N)` or :math:`(D, N)`
        - Output: :math:`(B, D, N + 1)` or :math:`(D, N + 1)`

    Examples::

        >>> input = torch.rand(2, 4, 3)  # BxNx3
        >>> transform =  ConvertPointsFromHomogeneous()
        >>> output = transform(input)  # BxNx2
    """

    def __init__(self):
        super(ConvertPointsFromHomogeneous, self).__init__()

    def forward(self, input):
        return convert_points_from_homogeneous(input)


class ConvertPointsToHomogeneous(nn.Module):
    r"""Creates a transformation to convert points from Euclidean to
    homogeneous space.

    Args:
        points (Tensor): tensor of N-dimensional points.

    Returns:
        Tensor: tensor of N+1-dimensional points.

    Shape:
        - Input: :math:`(B, D, N)` or :math:`(D, N)`
        - Output: :math:`(B, D, N + 1)` or :math:`(D, N + 1)`

    Examples::

        >>> input = torch.rand(2, 4, 3)  # BxNx3
        >>> transform = ConvertPointsToHomogeneous()
        >>> output = transform(input)  # BxNx4
    """

    def __init__(self):
        super(ConvertPointsToHomogeneous, self).__init__()

    def forward(self, input):
        return convert_points_to_homogeneous(input)


_import_module("_transformations", warn=False)

if __name__ == "__main__":
    import doctest

    numpy.set_printoptions(suppress=True, precision=5)
    doctest.testmod()
