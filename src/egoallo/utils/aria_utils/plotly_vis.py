# # MPS Tutorial
#
# This sample will show you how to use the Aria MPS data via the MPS apis.
# Please refer to the MPS wiki for more information about data formats and schemas
#
# ### Notebook stuck?
#
# Note that because of Jupyter and Plotly issues, sometimes the code may stuck at visualization. We recommend **restart the kernels** and try again to see if the issue is resolved.
#

# from projectaria_tools.utils.vrs_to_mp4_utils import get_timestamp_from_mp4
import numpy as np
import plotly.graph_objs as go

from egoallo.utils.setup_logger import setup_logger

from egoallo.config import make_cfg, CONFIG_FILE
from typing import List

from egoallo.utils.utils import NDArray
from egoallo.smpl.smplh_utils import (
    SMPL_JOINT_NAMES,
    EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS,
)

local_config_file = CONFIG_FILE
CFG = make_cfg(config_name="defaults", config_file=local_config_file, cli_args=[])

logger = setup_logger(output=None, name=__name__)


def build_cam_frustum_w_extr(T_world_cam):
    """build_cam_frustum_w_extr

    Parameters
    ----------
    T_world_cam : np.array of shape (3, 4)

    Returns
    -------
    go.Mesh3d
        an **invisible** go.Mesh3d object representing the camera frustum.

    Examples:
    -------
    >>> T_world_cam = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]])
    >>> build_cam_frustum_w_extr(T_world_cam)
    """

    points = (
        np.array(
            [
                [0, 0, 0, 1],
                [0.5, 0.5, 1, 1],
                [-0.5, 0.5, 1, 1],
                [-0.5, -0.5, 1, 1],
                [0.5, -0.5, 1, 1],
            ]
        )
        * CFG.plotly.camera_frustum.scale
    )
    points_transformed = T_world_cam @ points.transpose()

    return go.Mesh3d(
        x=points_transformed[0, :],
        y=points_transformed[1, :],
        z=points_transformed[2, :],
        i=[0, 0, 0, 0, 1, 1],
        j=[1, 2, 3, 4, 2, 3],
        k=[2, 3, 4, 1, 3, 4],
        showscale=False,
        visible=False,
        colorscale=CFG.plotly.camera_frustum.colorscale,
        intensity=points[:, 2],
        opacity=1.0,
        hoverinfo="none",
    )


def draw_coco_kinematic_tree(coco_kpts: NDArray, coco_cfg: None) -> List[go.Scatter3d]:
    """
    Parameters
    ----------
    coco_kpts : np.array of shape (17, 3)
        3D coordinates for the 17 keypoints in the COCO dataset.

    Returns
    -------
    list of go.Scatter3d
        Each Scatter3d object represents a segment of the kinematic tree.
        ['nose','left-eye','right-eye','left-ear','right-ear','left-shoulder','right-shoulder','left-elbow','right-elbow','left-wrist','right-wrist','left-hip','right-hip','left-knee','right-knee','left-ankle','right-ankle']
    """
    # Define connections between keypoints based on updated list
    edges = [
        (0, 1),  # Nose to Left Eye
        (0, 2),  # Nose to Right Eye
        (1, 3),  # Left Eye to Left Ear
        (2, 4),  # Right Eye to Right Ear
        (5, 6),  # Left Shoulder to Right Shoulder
        (5, 7),  # Left Shoulder to Left Elbow
        (7, 9),  # Left Elbow to Left Wrist
        (6, 8),  # Right Shoulder to Right Elbow
        (8, 10),  # Right Elbow to Right Wrist
        (11, 12),  # Left Hip to Right Hip
        (5, 11),  # Left Shoulder to Left Hip
        (6, 12),  # Right Shoulder to Right Hip
        (11, 13),  # Left Hip to Left Knee
        (13, 15),  # Left Knee to Left Ankle
        (12, 14),  # Right Hip to Right Knee
        (14, 16),  # Right Knee to Right Ankle
    ]

    traces = []
    for ind, (p1, p2) in enumerate(edges):
        x = [coco_kpts[p1][0], coco_kpts[p2][0]]
        y = [coco_kpts[p1][1], coco_kpts[p2][1]]
        z = [coco_kpts[p1][2], coco_kpts[p2][2]]

        if ind < 4:
            marker_color = coco_cfg.marker_color_for_head
            line_color = coco_cfg.line_color_for_head
        else:
            marker_color = coco_cfg.marker_color_for_body
            line_color = coco_cfg.line_color_for_body
        trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines+markers",
            marker=dict(
                size=coco_cfg.marker_size,
                color=marker_color,
            ),
            line=dict(color=line_color, width=coco_cfg.line_size),
            visible=False,
            name=f"COCO: {EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS[p1]} to {EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS[p2]}",
        )
        traces.append(trace)

    return traces


# endregion


def draw_smpl_kinematic_tree(smpl_kpts):
    """
    Draw a kinematic tree for the SMPL model using Plotly.

    Parameters
    ----------
    smpl_kpts : np.array of shape (24, 3)
        3D coordinates for the 24 joints in the SMPL model.

    Returns
    -------
    list of go.Scatter3d
        Each Scatter3d object represents a segment of the kinematic tree.
    """
    # Define connections between joints
    edges = [
        (0, 1),  # Pelvis to Left Hip
        (0, 2),  # Pelvis to Right Hip
        (1, 4),  # Left Hip to Left Knee
        (2, 5),  # Right Hip to Right Knee
        (4, 7),  # Left Knee to Left Ankle
        (5, 8),  # Right Knee to Right Ankle
        (7, 10),  # Left Ankle to Left Foot
        (8, 11),  # Right Ankle to Right Foot
        (0, 3),  # Pelvis to Spine1
        (3, 6),  # Spine1 to Spine2
        (6, 9),  # Spine2 to Spine3
        (9, 12),  # Spine3 to Neck
        (12, 15),  # Neck to Head
        (12, 16),  # Neck to Left Collar
        (12, 17),  # Neck to Right Collar
        (16, 18),  # Left Collar to Left Shoulder
        (17, 19),  # Right Collar to Right Shoulder
        (18, 20),  # Left Shoulder to Left Elbow
        (19, 21),  # Right Shoulder to Right Elbow
        (20, 22),  # Left Elbow to Left Wrist
        (21, 23),  # Right Elbow to Right Wrist
    ]

    traces = []
    for ind, (p1, p2) in enumerate(edges):
        x = [smpl_kpts[p1][0], smpl_kpts[p2][0]]
        y = [smpl_kpts[p1][1], smpl_kpts[p2][1]]
        z = [smpl_kpts[p1][2], smpl_kpts[p2][2]]

        if ind < 4:
            marker_color = CFG.plotly.smpl_kinematic_tree.marker_color_for_head
            line_color = CFG.plotly.smpl_kinematic_tree.line_color_for_head
        else:
            marker_color = CFG.plotly.smpl_kinematic_tree.marker_color_for_body
            line_color = CFG.plotly.smpl_kinematic_tree.line_color_for_body

        trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines+markers",
            marker=dict(
                size=CFG.plotly.smpl_kinematic_tree.line_size,
                color=marker_color,
            ),
            line=dict(color=line_color, width=CFG.plotly.smpl_kinematic_tree.line_size),
            visible=False,
            name=f"SMPL: {SMPL_JOINT_NAMES[p1]} to {SMPL_JOINT_NAMES[p2]}",
        )
        traces.append(trace)

    return traces


def draw_camera_pose(
    T: NDArray, colors: List[str] = ["red", "green", "blue"]
) -> List[go.Scatter3d]:
    """
    Draw an RGB 3D coordinate system based on a 3x4 camera pose matrix.

    Parameters
    ----------
    T : np.array of shape (3, 4)
        Camera pose matrix with rotation and translation.

    Returns
    -------
    Plotly figure
    """
    R = T[:, :3]
    t = T[:, 3]

    # Define unit vectors for x, y, z axes
    scale = 0.2
    x_axis = R @ (np.array([1, 0, 0]) * scale)
    y_axis = R @ (np.array([0, 1, 0]) * scale)
    z_axis = R @ (np.array([0, 0, 1]) * scale)

    # Create traces for each axis
    axes = []
    labels = ["x", "y", "z"]
    for axis, color, label in zip([x_axis, y_axis, z_axis], colors, labels):
        axes.append(
            go.Scatter3d(
                x=[t[0], t[0] + axis[0]],
                y=[t[1], t[1] + axis[1]],
                z=[t[2], t[2] + axis[2]],
                mode="lines",
                line=dict(
                    color=color,
                    width=CFG.plotly.camera_3d_coordinate.width,
                ),
                name=label,
                visible=False,
            )
        )
    return axes
