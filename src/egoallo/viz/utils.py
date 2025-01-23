import numpy as np

from jaxtyping import Float, Bool, Array, jaxtyped
import typeguard
import numpy as np
from typing import Optional
from egoallo.mapping import EGOEXO4D_BODYPOSE_KINTREE_PARENTS, EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS, SMPLH_TO_EGOEXO4D_BODYPOSE_INDICES, SMPLH_KINTREE

def blend_with_background(image: np.ndarray, background_color: tuple) -> np.ndarray:
    """Blend RGBA image with solid background color."""
    if image.shape[2] != 4:
        return image
        
    alpha = image[:, :, 3:4] / 255.0
    background = np.ones_like(image[:, :, :3]) * np.array(background_color) * 255
    blended = image[:, :, :3] * alpha + background * (1 - alpha)
    return blended.astype(np.uint8) 

@jaxtyped(typechecker=typeguard.typechecked)
def create_skeleton_point_cloud(
    joints_wrt_world: Float[np.ndarray, "num_joints 3"],
    visible_joints_mask: Bool[np.ndarray, "num_joints"], 
    input_smplh: bool = False,
    num_samples_per_bone: int = 100,
    return_colors: Optional[bool] = False
) -> tuple[Float[np.ndarray, "num_points 3"], Float[np.ndarray, "num_points 3"]] | Float[np.ndarray, "num_points 3"]:
    """Create a point cloud representing the skeleton by densely sampling along bones.
    
    Args:
        joints_wrt_world: Joint positions in world coordinates, shape [num_joints, 3]
        visible_joints_mask: Boolean mask indicating valid joints, shape [num_joints]
        input_smplh: If True, input uses SMPLH convention (52 joints), otherwise COCO (17 joints)
        num_samples_per_bone: Number of points to sample along each bone
        return_colors: If True, returns colors for points (green=valid, red=invalid)
    
    Returns:
        points: Sampled 3D points forming the skeleton, shape [num_points, 3]
        colors: Optional point colors, shape [num_points, 3]
    """
    # Verify input dimensions based on convention
    if input_smplh:
        assert joints_wrt_world.shape[0] == 22, f"SMPLH joints should have 22 joints, got {joints_wrt_world.shape[0]}"
        # SMPLH kinematic tree (parent indices for each joint)
        # Only includes body joints, not hand joints
        kintree = SMPLH_KINTREE
    else:
        assert joints_wrt_world.shape[0] == 17, f"COCO joints should have 17 joints, got {joints_wrt_world.shape[0]}"
        # COCO kinematic tree (parent indices for each joint)
        kintree = EGOEXO4D_BODYPOSE_KINTREE_PARENTS 

    # Initialize list to store sampled points and colors
    vis_pts = []
    invis_pts = []
    visc = [] if return_colors else None
    invisc = [] if return_colors else None
    
    # Define colors
    green = np.array([0, 255, 0]).astype(np.uint8)
    red = np.array([0, 255, 0]).astype(np.uint8)
    
    # Iterate through kinematic tree
    for joint_idx, parent_idx in enumerate(kintree):
        if parent_idx == -1:
            continue
            
        # Get joint positions
        start = joints_wrt_world[parent_idx]
        end = joints_wrt_world[joint_idx]
        
        # Skip if either contains NaN
        if np.isnan(start).any() or np.isnan(end).any():
            continue
            
        # Get validity of both joints
        start_valid = visible_joints_mask[parent_idx]
        end_valid = visible_joints_mask[joint_idx]

        # Sample points along bone
        for t in np.linspace(0, 1, num_samples_per_bone):
            point = start + t * (end - start)
            if start_valid and end_valid:
                vis_pts.append(point)
                if return_colors:
                    visc.append(green)
            elif not start_valid and not end_valid:
                invis_pts.append(point)
                if return_colors:
                    invisc.append(red)
            else:
                if (start_valid and t > 0.5) or (end_valid and t <= 0.5):
                    invis_pts.append(point)
                    if return_colors:
                        invisc.append(red)
                else:
                    vis_pts.append(point)
                    if return_colors:
                        visc.append(green)
            
    if len(vis_pts) == 0:
        vis_ret = (np.zeros((0,3)), np.zeros((0,3))) if return_colors else np.zeros((0,3))
    else:
        vis_ret = (np.stack(vis_pts), np.stack(visc)) if return_colors else np.stack(vis_pts)
    
    if len(invis_pts) == 0:
        invis_ret = (np.zeros((0,3)), np.zeros((0,3))) if return_colors else np.zeros((0,3))
    else:
        invis_ret = (np.stack(invis_pts), np.stack(invisc)) if return_colors else np.stack(invis_pts)

    return vis_ret, invis_ret