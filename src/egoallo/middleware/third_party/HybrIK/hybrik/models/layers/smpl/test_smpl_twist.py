"""
test smpl twist.py Date: 2025.03.28
Problems:
1 (Fixed in Mar29 in lbs.py). Whether it is for 29 joints or 24 joints, the linear blend skinning mesh always has misalignments around head, hands, and feet, since they are 5 leaf indices generated.
Small things:
1. The current vis process consists of two parts, the first of which is a polyscope user interactive interface, and second of which is a consistent temporal visualization using pyrender.
2. There is one option can be set SMPL model: `num_joints`, it can be either 24 or 29. the code includes them all, and comments out one of them.
"""
import os
import sys
import torch
import numpy as np
import pickle as pk
import open3d as o3d
import time
from pathlib import Path
import argparse
import pytest
from egoallo import training_utils

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from egoallo.middleware.third_party.HybrIK.hybrik.models.layers.smpl.fncsmpl_aadecomp import SmplModelAADecomp
from hybrik.models.layers.smpl.SMPL import SMPL_layer
from egoallo.transforms import SO3
from egoallo.viz.hybrik_twist_angle_visualizer import InteractiveSMPLViewer

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Test SMPL hybrik function')

    parser.add_argument(
        '--pickle_path', type=str, default='',
        help='Path to the pickle file with saved results',
    )
    parser.add_argument(
        '--smpl_family_model_basedir', type=str,
        default=os.path.join(project_root, './model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'),
        help='Path to the SMPL model file',
    )
    parser.add_argument(
        '--depth_factor', type=float, default=2.2,
        help='Depth factor for scaling joints (default: 2.2)',
    )
    parser.add_argument(
        '--delay', type=float, default=0.1,
        help='Delay between frames (default: 0.1)',
    )
    parser.add_argument(
        '--no_vis', action='store_true',
        help='Disable visualization',
    )
    parser.add_argument(
        '--max_frames', type=int, default=None,
        help='Maximum number of frames to process',
    )
    parser.add_argument(
        '--num_joints', type=int, default=24, choices=[24, 29],
        help='Number of joints to use (24 or 29)',
    )

    args = parser.parse_args()

    if args.pickle_path:
        assert Path(args.pickle_path).exists(), f"Pickle file {args.pickle_path} does not exist"
    return args

def load_data_from_pickle(file_path):
    """Load data from pickle file."""
    try:
        with open(file_path, 'rb') as f:
            data = pk.load(f)
        return data
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        sys.exit(1)

def init_smpl_model(model_path, num_joints=24):
    """Initialize SMPL model."""
    smpl = SmplModelAADecomp.load(
        model_path,
        num_joints=num_joints,
    )
    return smpl

def create_mesh(vertices, faces, color=[0.7, 0.7, 0.9]):
    """Create open3d mesh from vertices and faces."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh

def process_frame(smpl, data, frame_idx, depth_factor=2.2, num_joints=24, visualize=False, delay=0.1, vis=None, dummy_mesh=None):
    """Process a single frame of data."""
    # Extract required data for this frame
    pred_xyz_jts_29 = torch.tensor(data['pred_xyz_29'][frame_idx]).unsqueeze(0)  # Add batch dimension
    pred_shape = torch.tensor(data['pred_betas'][frame_idx]).unsqueeze(0)  # Add batch dimension
    pred_phi = torch.tensor(data['pred_phi'][frame_idx]).unsqueeze(0)  # Add batch dimension

    # Select appropriate joints based on num_joints
    pred_xyz_jts = pred_xyz_jts_29
    if num_joints == 24:
        pred_xyz_jts_24 = pred_xyz_jts_29[..., :24, :]
        pred_xyz_jts = pred_xyz_jts_24

    leaf_thetas = SO3.identity(device=pred_phi.device, dtype=pred_phi.dtype).wxyz.to(pred_phi.device, pred_phi.dtype)[None, None, ...].repeat(1, 5, 1)

    # Run hybrik function
    with torch.no_grad():  # No need for gradients
        output = smpl.model.hybrik(
            pose_skeleton=pred_xyz_jts * depth_factor,  # unit: meter
            betas=pred_shape,
            phis=pred_phi,
            global_orient=None,
            leaf_thetas=None if num_joints == 24 else leaf_thetas,
            return_verts=True,
        )

    # Create viewer if requested
    ps_vis = False  # Set to False by default for testing
    if ps_vis:
        ind = 0
        viewer = InteractiveSMPLViewer(
            smpl_aadecomp_model=smpl,
            pose_skeleton=pred_xyz_jts[ind] * depth_factor,
            betas=pred_shape[0, :10],
            transl=pred_xyz_jts[ind, 0],
            initial_phis=pred_phi[ind],
            global_orient=None,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            num_hybrik_joints=num_joints,
            leaf_thetas=None if num_joints == 24 else leaf_thetas[ind],
            coordinate_transform=True,
        )
        viewer.show()

    # Extract vertices for current frame
    pred_vertices = output.vertices.detach().cpu().numpy()[0]  # Remove batch dimension
    pred_joints = output.joints.detach().cpu().numpy()[0]  # Remove batch dimension

    # Visualize if enabled
    if visualize and vis is not None:
        # Create mesh for current frame
        faces = smpl.model.faces
        mesh = create_mesh(pred_vertices, faces)

        # Create joint markers
        joint_markers = []
        for joint in pred_joints:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere.translate(joint)
            sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Red for joints
            joint_markers.append(sphere)

        # Update the visualizer
        vis.remove_geometry(dummy_mesh)
        vis.add_geometry(mesh)

        # Update joints (for simplicity, only update the mesh, not individual joints)
        new_dummy_mesh = mesh

        # Update camera view and render
        vis.update_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()

        # Add some delay to better see the animation
        time.sleep(delay)

        return new_dummy_mesh, pred_vertices, pred_joints

    return dummy_mesh, pred_vertices, pred_joints

def main():
    # Parse arguments
    args = parse_args()

    # Load data from pickle file
    print(f"Loading data from {args.pickle_path}")
    data = load_data_from_pickle(args.pickle_path)

    # Initialize SMPL model
    print(f"Initializing SMPL model with {args.num_joints} joints")
    smpl = init_smpl_model(Path(args.smpl_family_model_basedir), num_joints=args.num_joints)

    # Extract faces for visualization
    faces = smpl.model.faces

    # Get frame count
    n_frames = data['pred_xyz_29'].shape[0]
    if args.max_frames and args.max_frames < n_frames:
        n_frames = args.max_frames

    # Prepare visualizer if needed
    vis = None
    dummy_mesh = None

    if not args.no_vis:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="SMPL hybrik visualizer", width=1280, height=720)

        # Add a coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        vis.add_geometry(coordinate_frame)

        # Create material for better visualization
        mesh_material = o3d.visualization.rendering.MaterialRecord()
        mesh_material.shader = "defaultLit"

        # Add initial mesh (will be updated)
        dummy_mesh = create_mesh(np.zeros((1, 3)), np.zeros((1, 3)))
        vis.add_geometry(dummy_mesh)

        # Setup camera view
        opt = vis.get_render_option()
        opt.background_color = np.array([0.1, 0.1, 0.1])
        opt.mesh_show_back_face = True

    # Process each frame
    for frame_idx in range(n_frames):
        print(f"Processing frame {frame_idx+1}/{n_frames}")

        dummy_mesh, pred_vertices, pred_joints = process_frame(
            smpl=smpl,
            data=data,
            frame_idx=frame_idx,
            depth_factor=args.depth_factor,
            num_joints=args.num_joints,
            visualize=not args.no_vis,
            delay=args.delay,
            vis=vis,
            dummy_mesh=dummy_mesh,
        )

        # Print some statistics
        print(f"  Vertices shape: {pred_vertices.shape}")
        print(f"  Joints shape: {pred_joints.shape}")

    # Close visualizer
    if vis is not None and not args.no_vis:
        vis.destroy_window()

@pytest.mark.parametrize("num_joints", [24, 29])
def test_smpl_hybrik(num_joints):
    """Test SMPL hybrik function with different joint counts."""
    # Use a test pickle file path here - adjust as needed
    pickle_path = os.environ.get('TEST_PICKLE_PATH', '')
    if not pickle_path:
        pytest.skip("TEST_PICKLE_PATH environment variable not set")

    smpl_model_path = os.path.join(project_root, './model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')

    # Initialize SMPL model
    print(f"Testing SMPL hybrik with {num_joints} joints")
    smpl = init_smpl_model(Path(smpl_model_path), num_joints=num_joints)

    # Check if the model was initialized successfully
    assert smpl is not None, f"Failed to initialize SMPL model with {num_joints} joints"

    # If pickle file exists, test processing a frame
    if os.path.exists(pickle_path):
        data = load_data_from_pickle(pickle_path)

        # Process a single frame
        dummy_mesh, pred_vertices, pred_joints = process_frame(
            smpl=smpl,
            data=data,
            frame_idx=0,
            depth_factor=2.2,
            num_joints=num_joints,
            visualize=False,
        )

        # Check the output shapes
        if num_joints == 24:
            assert pred_joints.shape[0] == 24, f"Expected 24 joints, got {pred_joints.shape[0]}"
        else:
            assert pred_joints.shape[0] == 29, f"Expected 29 joints, got {pred_joints.shape[0]}"

        # Check that vertices were generated (6890 is the standard SMPL vertex count)
        assert pred_vertices.shape[0] == 6890, f"Expected 6890 vertices, got {pred_vertices.shape[0]}"

        print(f"Test passed for {num_joints} joints")

if __name__ == "__main__":
    training_utils.ipdb_safety_net()
    main()
