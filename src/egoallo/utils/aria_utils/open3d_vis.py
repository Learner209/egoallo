##
# Visualization utils
# Utility function to log to rerun:
#  - AriaGlasses outline
#  - Camera Calibration
#  - Camera pose
#  - Images
#  - Point cloud
##

import numpy as np
import open3d as o3d
from typing import List

from egoallo.config import make_cfg, CONFIG_FILE
from egoallo.utils.setup_logger import setup_logger

local_config_file = CONFIG_FILE
CFG = make_cfg(config_name="defaults", config_file=local_config_file, cli_args=[])

logger = setup_logger(output=None, name=__name__)


def build_cam_frustum_w_extr(T_world_cam):
    """Create an invisible mesh representing the camera frustum."""

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
    points_transformed = points_transformed[:3, :].T

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points_transformed)
    mesh.triangles = o3d.utility.Vector3iVector(
        [[0, 1, 2], [0, 2, 3], [0, 3, 4], [1, 2, 3], [1, 3, 4]]
    )
    mesh.paint_uniform_color([0, 0, 0])  # Invisible
    return mesh


def draw_coco_kinematic_tree(coco_kpts, coco_cfg) -> List[o3d.geometry.LineSet]:
    """Draw the COCO kinematic tree."""
    edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (11, 12),
        (5, 11),
        (6, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ]

    line_sets = []
    for p1, p2 in edges:
        points = np.array([coco_kpts[p1], coco_kpts[p2]])
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
        line_set.paint_uniform_color(
            np.asarray(coco_cfg.color_rgb_line_body).astype(np.float64)
            if p1 > 4
            else np.asarray(coco_cfg.color_rgb_line_head).astype(np.float64)
        )
        line_sets.append(line_set)

    return line_sets


def draw_camera_pose(
    T: np.ndarray, colors: List[str] = ["red", "green", "blue"]
) -> List[o3d.geometry.LineSet]:
    """
    Draw an RGB 3D coordinate system based on a 3x4 camera pose matrix using Open3D.

    Parameters
    ----------
    T : np.array of shape (3, 4)
        Camera pose matrix with rotation and translation.
    colors : list of str
        List of colors for the x, y, z axes.

    Returns
    -------
    List of Open3D LineSet objects representing the axes.
    """
    R = T[:, :3]
    t = T[:, 3]

    # Define unit vectors for x, y, z axes
    scale = 0.2
    x_axis = R @ (np.array([1, 0, 0]) * scale)
    y_axis = R @ (np.array([0, 1, 0]) * scale)
    z_axis = R @ (np.array([0, 0, 1]) * scale)

    # Create a list to hold the LineSet objects
    axes = []
    axis_vectors = [x_axis, y_axis, z_axis]

    for axis, color in zip(axis_vectors, colors):
        # Create points for the line
        points = np.array([t, t + axis])

        # Create LineSet object
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector([[0, 1]])

        # Set color for the line
        line_set.paint_uniform_color(
            np.array(o3d.utility.Vector3dVector(np.array([color])))
        )

        # Add the LineSet to the list
        axes.append(line_set)

    return axes
