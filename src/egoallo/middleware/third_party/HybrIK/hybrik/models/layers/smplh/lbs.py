"""
SMPLH model layer for HybrIK.
Generated by Gemini, modified by me. 2025/3/27
The test script for it is `test_smplh_twist.py`.
The current problem for this testing is :
1. The hand pose is misaligned.
2. When running this script, the error message in `get_twist` func in smplx's 'lbs.py' keeps showing up, indicating something wrong with axis-angle decomposition maybe.
"""

from __future__ import absolute_import, division, print_function

import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from ..smplx.lbs import rot_mat_to_euler

Tensor = Union[torch.Tensor]

from ..smplx.lbs import batch_rodrigues, transform_mat, batch_rigid_transform, vertices2joints, blend_shapes
from ..smplx.lbs import batch_get_pelvis_orient, batch_get_pelvis_orient_svd, batch_get_children_orient, batch_get_children_orient_svd, vectors2rotmat

# --- LBS function adapted for SMPLH ---

def lbs(
    betas: Tensor,
    pose: Tensor,
    v_template: Tensor,
    shapedirs: Tensor,
    posedirs: Tensor, # NOTE: Assumed to be for BODY joints ONLY (e.g., 6890*3 x 21*9)
    J_regressor: Tensor,
    parents: Tensor, # Shape (52,) for SMPLH
    lbs_weights: Tensor,
    pose2rot: bool = True,
    num_body_joints: int = 21, # Number of pose parameters in posedirs (excluding root)
) -> Tuple[Tensor, Tensor]:
    ''' Performs Linear Blend Skinning with the given shape and pose parameters for SMPLH
        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters.
        pose : torch.tensor Bx(J*3) or BxJx3x3 (J=52 for SMPLH)
            The pose parameters in axis-angle or rotation matrix format.
        v_template torch.tensor Vx3
            The template mesh.
        shapedirs : torch.tensor Vx3xNB
            The tensor of shape displacements.
        posedirs : torch.tensor (V*3)x(K*9) (K=num_body_joints=21 for SMPLH)
            The pose PCA coefficients (for body joints only).
        J_regressor : torch.tensor JxV (J=52 for SMPLH)
            The regressor array that is used to calculate the joints from vertices.
        parents: torch.tensor J (J=52 for SMPLH)
            The array that describes the kinematic tree.
        lbs_weights: torch.tensor VxJ (J=52 for SMPLH)
            The linear blend skinning weights.
        pose2rot: bool, optional
            Flag on whether to convert axis-angle to rotation matrices.
        num_body_joints: int
             Number of joints included in `posedirs` (excluding root). Default 21 for SMPLH.
        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying shape and pose displacements.
        joints: torch.tensor BxJx3
            The joints of the model (J=52 for SMPLH).
    '''
    batch_size = max(betas.shape[0], pose.shape[0])
    device, dtype = betas.device, betas.dtype
    num_total_joints = parents.shape[0] # Should be 52 for SMPLH

    # 1. Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # 2. Get the joints
    # BxJx3 array (J = 52)
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Add pose blend shapes (for body joints only)
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
            batch_size, num_total_joints, 3, 3,
        )
    else:
        # Ensure pose is in BxJx3x3 format
        if pose.dim() == 2: # Bx(J*9)
             pose = pose.view(batch_size, num_total_joints, 3, 3)
        rot_mats = pose

    # Select body joint rotations for pose blend shapes (joints 1 to 1+num_body_joints)
    # num_body_joints is typically 21 for SMPLH (joints 1-21)
    pose_feature_body = (rot_mats[:, 1:1+num_body_joints, :, :] - ident).view(batch_size, -1)

    # (B x (K*9)) x ((K*9) x (V*3)) -> B x (V*3) -> B x V x 3
    # Note the transpose on posedirs compared to original SMPL implementation
    pose_offsets = torch.matmul(pose_feature_body, posedirs).view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped

    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is B x V x J
    W = lbs_weights.unsqueeze(dim=0).expand(batch_size, -1, -1)
    # T = B x V x 4 x 4
    T = torch.matmul(W, A.view(batch_size, num_total_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones(
        [batch_size, v_posed.shape[1], 1],
        dtype=dtype, device=device,
    )
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed


# --- IK (HybrIK) functions adapted for SMPLH ---
# Based on SMPLX hybrik, removing face parts, adjusting for 52 joints + leaves


# --- Main IK function for SMPLH ---
def hybrik(
    betas: Tensor,
    pose_skeleton: Tensor, # B x (J + num_leaves) x 3
    phis: Tensor, # B x (J-1) x 2 (twist params for poseable joints except root)
    v_template: Tensor,
    shapedirs: Tensor,
    posedirs: Tensor, # Body only
    J_regressor: Tensor, # J=52
    parents: Tensor, # Extended parents (J + num_leaves)
    children: Dict, # Extended children map
    lbs_weights: Tensor, # V x J (J=52)
    leaf_indices: List[int], # Indices of vertices used as leaf joints
    num_body_joints_pose: int = 21, # Num joints in posedirs (excluding root)
    num_total_joints: int = 52, # SMPLH base joints
    train: bool = False,
    naive: bool = False, # Use naive IK version?
    use_svd: bool = True, # Use SVD for multi-child joints?
) -> Tuple[Tensor, Tensor, Tensor]:
    ''' Performs Inverse Kinematics and LBS for SMPLH.
        Based on SMPLX hybrik implementation, adapted for SMPLH structure.

        Parameters:
        ----------
         (Similar to SMPLX hybrik, but dimensions adjusted for SMPLH)
        pose_skeleton: B x K x 3, where K = num_total_joints + num_leaves
        phis: B x (num_total_joints - 1) x 2 # Twist for 51 poseable joints
        parents: Extended kinematic tree including leaves
        children: Extended children map corresponding to parents
        leaf_indices: Vertex indices for leaf joints
        num_body_joints_pose: Number of joints (excl root) covered by posedirs (21 for SMPLH)
        num_total_joints: Base number of SMPLH joints (52)

        Returns:
        -------
        verts: torch.tensor BxVx3
        joints: torch.tensor BxJx3 (J=52)
        rot_mats: torch.tensor BxJx3x3 (J=52)
    '''
    batch_size = pose_skeleton.shape[0]
    device, dtype = betas.device, betas.dtype
    num_extended_joints = parents.shape[0] # 52 + num_leaves

    # 1. Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # 2. Get the rest joints (including leaves)
    rest_J_base = vertices2joints(J_regressor, v_shaped) # B x 52 x 3
    if leaf_indices:
        leaf_vertices = v_shaped[:, leaf_indices].clone() # B x num_leaves x 3
        rest_J = torch.cat([rest_J_base, leaf_vertices], dim=1) # B x K x 3
    else:
        rest_J = rest_J_base # Should not happen if leaves are used

    # 3. Compute Rotations using IK
    # Prepare inputs for IK function
    rel_rest_pose = rest_J.clone()
    rel_rest_pose[:, 1:] -= rest_J[:, parents[1:]].clone()
    rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1) # B x K x 3 x 1

    rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1) # B x K x 3 x 1
    rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, parents[1:]].clone()
    rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0] # Use rest root position

    # Normalize phis
    phis = phis / (torch.norm(phis, dim=2, keepdim=True) + 1e-8)

    # --- IK Core Logic (Adapted from SMPLX naive version) ---
    rot_mat_chain = torch.zeros(batch_size, num_extended_joints, 3, 3, dtype=dtype, device=device)
    rot_mat_local = torch.zeros_like(rot_mat_chain)

    # Global Orientation (Pelvis)
    if use_svd:
        global_orient_mat = batch_get_pelvis_orient_svd(
            rel_pose_skeleton, rel_rest_pose, parents, children, dtype,
        )
    else:
         global_orient_mat = batch_get_pelvis_orient(
            rel_pose_skeleton, rel_rest_pose, parents, children, dtype,
         )

    rot_mat_chain[:, 0] = global_orient_mat
    rot_mat_local[:, 0] = global_orient_mat

    # Representative children for IK , needs to be determined by parents, children variables.
    representative_children = {
        0: 3,   # pelvis -> spine1
        9: 12,  # spine3 -> neck
        20: 25, # left wrist -> left index finger base
        21: 40,  # right wrist -> right index finger base
    }
    # Iterate through joints (level by level or sequentially)
    # Using sequential iteration for simplicity, similar to SMPLX naive
    for i in range(1, num_extended_joints):
        parent_idx = parents[i]
        child_info = children.get(i, -1) # Use .get for safety

        # Calculate current joint's world position in rotated rest pose
        # This is needed if child's target depends on parent's *posed* location
        # For naive version, we use relative vectors directly
        # rotate_rest_pose_i = rotate_rest_pose[:, parent_idx] + torch.matmul(
        #     rot_mat_chain[:, parent_idx], rel_rest_pose[:, i]
        # )

        if child_info == -1: # Leaf node in the extended tree
            # Leaves don't have rotations in this IK formulation
            # Assign identity locally, chain remains parent's chain
            rot_mat_local[:, i] = torch.eye(3, dtype=dtype, device=device)
            rot_mat_chain[:, i] = rot_mat_chain[:, parent_idx] # Chain carries parent rotation
            continue

        # Calculate local rotation for joint i
        current_rot_chain_parent = rot_mat_chain[:, parent_idx]

        if child_info < -1: # Multiple children case (e.g., shoulders, hips)
            child_indices = [idx for idx, p in enumerate(parents) if p == i]
            if not child_indices:
                rot_mat = torch.eye(3, dtype=dtype, device=device)
            else:
                # Start with representative child if defined
                if i in representative_children:
                    child_indices = [representative_children[i]] + [c for c in child_indices if c != representative_children[i]]

                rel_pose_children = []
                rel_rest_children = []
                for child_idx in child_indices:
                    target_rel_vec = rel_pose_skeleton[:, child_idx]
                    rel_pose_children.append(target_rel_vec)
                    rel_rest_children.append(rel_rest_pose[:, child_idx])

                if use_svd:
                    rot_mat = batch_get_children_orient_svd(
                        rel_pose_children, rel_rest_children, current_rot_chain_parent, children_list=child_indices, dtype=dtype,
                    )
                else:
                    # Fallback or alternative: use first child for alignment + twist
                    # (Similar to single child case below, but potentially less stable)
                    # Using SVD is generally better for multi-child joints
                    rot_mat = batch_get_children_orient_svd(
                        rel_pose_children, rel_rest_children, current_rot_chain_parent, children_list=child_indices, dtype=dtype,
                    )

        else: # Single child case
            child_idx = child_info

            # Target vector: child's relative position in parent's local frame
            child_final_loc_local = torch.matmul(
                current_rot_chain_parent.transpose(1, 2),
                rel_pose_skeleton[:, child_idx],
            )
            child_rest_loc = rel_rest_pose[:, child_idx]

            # --- Swing Calculation (vectors2rotmat) ---
            rot_mat_swing = vectors2rotmat(child_rest_loc, child_final_loc_local, dtype)

            # --- Twist Calculation ---
            # Axis for twist is the rest bone direction
            child_rest_norm = torch.norm(child_rest_loc, dim=1, keepdim=True)
            spin_axis = child_rest_loc / (child_rest_norm + 1e-8)

            # Get twist parameters (cos, sin)
            # phis index is i-1 because phis is for joints 1 to J-1
            if i < num_total_joints: # Only apply twist to base SMPLH joints
                cos_phi, sin_phi = torch.split(phis[:, i - 1], 1, dim=1) # Bx1
                cos_phi = cos_phi.unsqueeze(2) # Bx1x1
                sin_phi = sin_phi.unsqueeze(2) # Bx1x1

                # Rodrigues for twist
                rx, ry, rz = torch.split(spin_axis, 1, dim=1)
                zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)
                K_spin = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
                    .view((batch_size, 3, 3))
                ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
                rot_mat_twist = ident + sin_phi * K_spin + (1 - cos_phi).clamp(min=0) * torch.bmm(K_spin, K_spin)

                # Combine swing and twist
                rot_mat = torch.matmul(rot_mat_swing, rot_mat_twist)
            else:
                # No twist for leaf joints in this formulation
                rot_mat = rot_mat_swing


        # Update local and chain rotations
        rot_mat_local[:, i] = rot_mat
        rot_mat_chain[:, i] = torch.matmul(current_rot_chain_parent, rot_mat)
        # --- End IK Core Logic ---

    # Extract rotations for the base SMPLH joints (first num_total_joints)
    final_rot_mats = rot_mat_local[:, :num_total_joints] # B x 52 x 3 x 3

    # 4. Add pose blend shapes (body only)
    ident = torch.eye(3, dtype=dtype, device=device)
    # Use body joints (1 to 1+num_body_joints_pose) for posedirs
    pose_feature_body = (final_rot_mats[:, 1:1+num_body_joints_pose] - ident).view(batch_size, -1)
    pose_offsets = torch.matmul(pose_feature_body, posedirs).view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped # Apply pose blend shapes

    # 5. Skinning
    # Get posed joints and transformation matrices using the *base* rest pose and *base* rotations
    posed_J_base, A_base = batch_rigid_transform(
        final_rot_mats, rest_J_base, parents[:num_total_joints], dtype=dtype,
    )

    # W is B x V x J (J=52)
    W = lbs_weights.unsqueeze(dim=0).expand(batch_size, -1, -1)
    # T = B x V x 4 x 4
    T = torch.matmul(W, A_base.view(batch_size, num_total_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones(
        [batch_size, v_posed.shape[1], 1],
        dtype=dtype, device=device,
    )
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, posed_J_base, final_rot_mats
