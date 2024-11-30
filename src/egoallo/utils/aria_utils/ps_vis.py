import polyscope as ps
import numpy as np
from typing import Tuple, Dict, List, Any, Union, Optional, Type

from egoallo.config import make_cfg, CONFIG_FILE
from egoallo.utils.setup_logger import setup_logger
from datetime import datetime

local_config_file = CONFIG_FILE
CFG = make_cfg(config_name="defaults", config_file=local_config_file, cli_args=[])

logger = setup_logger(output=None, name=__name__)


def build_cam_frustum_w_extr(T_world_cam, surface_mesh_id=None):
    """Create an invisible mesh representing the camera frustum.
    the T_world_cam should be in OpenGL camera coordinate system.
    """

    points = (
            np.array(
                [[0, 0, 0, 1], [0.5, 0.5, -1, 1], [-0.5, 0.5, -1, 1], [-0.5, -0.5, -1, 1], [0.5, -0.5, -1, 1]]
            )
    )
    points[:, :-1] = points[:, :-1] * CFG.plotly.camera_frustum.scale  # Scale the frustum

    if surface_mesh_id is None:
        cur_time = datetime.now().strftime("%m_%d_%H_%M_%S")
        surface_mesh_id = f"coco_kin_tree_{cur_time}"

    points_transformed = T_world_cam @ points.transpose()
    points_transformed = points_transformed[:3, :].T

    ps.register_surface_mesh(f"camera_frustum_{surface_mesh_id}", points_transformed, 
                              [[0, 1, 2], [0, 2, 3], [0, 3, 4], [1, 2, 3], [1, 3, 4]],
                              color=(0.4, 0.7, 0.1))  # Invisible color (RGBA)

def draw_coco_kinematic_tree(coco_kpts, coco_cfg, curve_network_id=None) -> None:
    """Draw the COCO kinematic tree."""
    edges = [
        (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8),
        (8, 10), (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]

    if curve_network_id is None:
        cur_time = datetime.now().strftime("%m_%d_%H_%M_%S")
        curve_network_id = f"coco_kin_tree_{cur_time}"
    # import ipdb; ipdb.set_trace() 
    for p1, p2 in edges:
        points = np.array([coco_kpts[p1], coco_kpts[p2]])
        # polyscope doesn't cope well with inf values
        if np.isinf(points).any() or np.isnan(points).any():
            points = np.array([[0, 0, 0], [0, 0, 0]])
        
        # Create a line between the two points
        network = ps.register_curve_network("coco_{}_{}_{}".format(p1, p2, curve_network_id), points, 
                                   np.array([[0, 1]]),
                                   color=np.asarray(coco_cfg.color_rgb_line_body).astype(np.float64) if p1 > 4 else 
                                         np.asarray(coco_cfg.color_rgb_line_head).astype(np.float64))
        # network.set_radius(0.0009)
        network.set_radius(0.005, relative=False) # radius in absolute world units


def draw_camera_pose(T: np.ndarray, colors: List[str] = ['red', 'green', 'blue'], curve_network_id=None):
    """
    Draw an RGB 3D coordinate system based on a 3x4 camera pose matrix using Polyscope.

    Parameters
    ----------
    T : np.array of shape (3, 4)
        Camera pose matrix with rotation and translation.
    colors : list of str
        List of colors for the x, y, z axes.
    """
    R = T[:, :3] # 3 x 3
    t = T[:, 3] # 3 x 1

    # Define unit vectors for x, y, z axes
    scale = 0.2
    x_axis = R @ (np.array([1, 0, 0]) * scale) # 3 x 1
    y_axis = R @ (np.array([0, 1, 0]) * scale) # 3 x 1
    z_axis = R @ (np.array([0, 0, 1]) * scale) # 3 x 1

    if curve_network_id is None:
        cur_time = datetime.now().strftime("%m_%d_%H_%M_%S")
        curve_network_id = f"camera_pose_{cur_time}"
    # Create axes with Polyscope
    ps.register_curve_network(f"{curve_network_id}_x", np.array([t, t + x_axis]), np.array([[0, 1]]), color=(1,0,0), radius=0.004)
    ps.register_curve_network(f"{curve_network_id}_y", np.array([t, t + y_axis]), np.array([[0, 1]]), color=(0,1,0), radius=0.004)
    ps.register_curve_network(f"{curve_network_id}_z", np.array([t, t + z_axis]), np.array([[0, 1]]), color=(0,0,1), radius=0.004)

if __name__ == "__main__":
    # Initialize Polyscope
    ps.init()

    # Create some example data: a simple point cloud
    # For demonstration purposes, let's create a cloud with random points
    num_points = 100
    points = np.random.rand(num_points, 3)

    # Register the point cloud with Polyscope
    ps.register_point_cloud("Sample Point Cloud", points)

    # Show the Polyscope window
    ps.show()
