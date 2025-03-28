"""
2025.3.28: test_smplx_twist.py on testing hybrik's inference result on dance.mp4.
Several problems exists:
1. The neck, jaw / left(right) hands / (left)right foot are deformed. Need to inspect why. Similar to problem in `SmplModelAADecomp`.
2. When running this script, the error message in `get_twist` func in smplx's 'lbs.py' keeps showing up, indicating something wrong with axis-angle decomposition maybe.
Several potential modifications exists:
1. Write a polyscope interactive interface for this model to visualize the deformed crux and testing twist angels's effects on HybrIK func.
"""
import os
import sys
import torch
import numpy as np
import pickle as pk
import time
import argparse
from pathlib import Path
import open3d as o3d
from egoallo import training_utils
from egoallo.middleware.third_party.HybrIK.hybrik.models.layers.smplh.test_smplh_twist import SmplFamilyModelTypeLiteral

training_utils.ipdb_safety_net()
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# Import SMPLX model
from egoallo.constants import SmplFamilyMetaModelZoo, SmplFamilyMetaModelName
from egoallo.type_stubs import SmplFamilyModelType
from src.egoallo.middleware.third_party.HybrIK.hybrik.models.layers.smplx.body_models import SMPLXLayer

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Test SMPLX twist function')

    parser.add_argument(
        '--pickle_path', type=str, default='',
        help='Path to the pickle file with saved results',
    )
    parser.add_argument(
        '--smpl_family_model_basedir', type=str,
        default='',
        help='Path to the SMPLH model file',
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
        '--max_frames', type=int, default=None,
        help='Maximum number of frames to process',
    )
    parser.add_argument(
        '--save_output', action='store_true',
        help='Save twist output to a pickle file',
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Visualize the SMPLX model using Open3D',
    )
    parser.add_argument(
        '--vis_delay', type=float, default=0.05,
        help='Delay between visualization frames (default: 0.05)',
    )
    parser.add_argument(
        '--joint_radius', type=float, default=0.01,
        help='Radius of joint spheres in visualization (default: 0.01)',
    )
    parser.add_argument(
        '--mesh_color', type=str, default='0.8,0.8,0.8',
        help='Mesh color in RGB format (default: 0.8,0.8,0.8 for light gray)',
    )

    args = parser.parse_args()

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

def init_smplx_model(model_path: Path):
    """Initialize SMPLX model."""
    try:
        smplx_meta_name: SmplFamilyModelTypeLiteral = "SmplxModelAADecomp"
        smplx = SmplFamilyMetaModelZoo[smplx_meta_name].load(
                    model_path,
                    use_pca=False,
        )
        return smplx
    except Exception as e:
        print(f"Error initializing SMPLX model: {e}")
        sys.exit(1)

def process_frame(smplx: SmplFamilyModelType, data, frame_idx, depth_factor):
    """Process a single frame and compute twist angles."""
    # Extract required data for this frame
    pred_betas = torch.tensor(data['pred_beta'][frame_idx]).unsqueeze(0)  # Add batch dimension
    pred_thetas = torch.tensor(data['pred_theta_mat'][frame_idx])  # Rotation matrices
    pred_xyz_full = torch.tensor(data['pred_xyz_full'][frame_idx]).unsqueeze(0)  # Add batch dimension

    pred_theta_mats = pred_thetas.reshape(55, 9)
    global_orient = pred_theta_mats[0:1].reshape(1, 1,3,3)  # Global orientation (1, 1, 3, 3)
    body_pose = pred_theta_mats[1:22].reshape(1, 21,3,3)  # Body pose (1, 21, 3, 3)
    left_hand_pose = pred_theta_mats[22:37].reshape(1, 15,3,3)  # Left hand pose (1, 15, 3, 3)
    right_hand_pose = pred_theta_mats[37:52].reshape(1, 15,3,3)  # Right hand pose (1, 15, 3, 3)
    jaw_pose = pred_theta_mats[52:53].reshape(1, 1,3,3)  # Jaw pose (1, 1, 3, 3)
    leye_pose = pred_theta_mats[53:54].reshape(1, 1,3,3)  # Left eye pose (1, 1, 3, 3)
    reye_pose = pred_theta_mats[54:55].reshape(1, 1,3,3)  # Right eye pose (1, 1, 3, 3)


    # Run forward_get_twist to get twist angles
    with torch.no_grad():
        twist = smplx.model.forward_get_twist(
            betas=pred_betas,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            expression=None,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            full_pose=None,
        )

        print(f"  Computed twist shape: {twist.shape}")

    # Run hybrik function with computed twist angles
    num_joints = 71
    pred_xyz_jts = pred_xyz_full.reshape(-1, 3)[..., :num_joints, :3].unsqueeze(0)

    # rotate pred_xyz_jts around x-axis by 90 degrees.
    pred_xyz_jts = torch.matmul(pred_xyz_jts, torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float32))

    cos_sin_phis = torch.cat([torch.cos(twist), torch.sin(twist)], dim=-1)
    with torch.no_grad():
        output = smplx.model.hybrik(
            betas=pred_betas,
            expression=None,
            pose_skeleton=pred_xyz_jts * depth_factor,  # scale to meters
            phis=cos_sin_phis,  # use computed twist angles
            transl=None,
            return_verts=True,
            root_align=True,
        )

    # Extract results
    pred_vertices = output['vertices'].detach().cpu().numpy()[0]  # Remove batch dimension
    pred_joints = output['joints'].detach().cpu().numpy()[0]  # Remove batch dimension

    return {
        'twist': twist.detach().cpu().numpy(),
        'vertices': pred_vertices,
        'joints': pred_joints,
    }

def create_mesh_with_vertices(vertices, faces, color=None):
    """Create an Open3D mesh from vertices and faces."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    if color is not None:
        mesh.paint_uniform_color(color)

    return mesh

def create_joint_markers(joints, radius=0.01, color=None):
    """Create spheres to represent joints."""
    markers = []
    for joint_pos in joints:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(joint_pos)
        if color is not None:
            sphere.paint_uniform_color(color)
        markers.append(sphere)
    return markers

def create_visualizer():
    """Create and initialize an Open3D visualizer."""
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Set rendering options
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
    opt.point_size = 5.0

    # Set camera parameters for a better view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)

    return vis

def update_camera_position(vis, root_joint_pos, frame_idx):
    """
    Update camera position to rotate around the root joint.

    Args:
        vis: Open3D visualizer
        root_joint_pos: Position of the root joint (0th joint)
        frame_idx: Current frame index for animation
    """
    # Get view control
    ctr = vis.get_view_control()
    cam_params = ctr.convert_to_pinhole_camera_parameters()

    # Calculate camera position parameters
    angle = frame_idx * 0.05  # Controls rotation speed
    distance = 2.0  # Distance from subject
    height = 0.5  # Height offset

    # Calculate new camera position
    cam_x = root_joint_pos[0] + distance * np.sin(angle)
    cam_y = root_joint_pos[1] + distance * np.cos(angle)
    cam_z = root_joint_pos[2] + height

    cam_pos = np.array([cam_x, cam_y, cam_z])

    # Calculate camera orientation
    forward = root_joint_pos - cam_pos
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, [0, 0, 1])
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    # Create camera extrinsic matrix
    T_world_cam = np.eye(4)
    T_world_cam[:3, 0] = right
    T_world_cam[:3, 1] = -up
    T_world_cam[:3, 2] = forward
    T_world_cam[:3, 3] = cam_pos

    # Update camera parameters
    cam_params.extrinsic = np.linalg.inv(T_world_cam)
    ctr.convert_from_pinhole_camera_parameters(cam_params)

def update_visualizer(vis, vertices, joints, faces=None, mesh_color=None, joint_radius=0.01, frame_idx=0):
    """Update the visualizer with new geometries."""
    # Clear previous geometries
    vis.clear_geometries()

    # Parse mesh color
    if mesh_color is not None and isinstance(mesh_color, str):
        try:
            mesh_color = np.array([float(c) for c in mesh_color.split(',')])
        except:
            mesh_color = np.array([0.8, 0.8, 0.8])  # Default light gray
    else:
        mesh_color = np.array([0.8, 0.8, 0.8])  # Default light gray

    # Create mesh or point cloud from vertices
    geometry = create_mesh_with_vertices(vertices, faces, color=mesh_color)
    vis.add_geometry(geometry)

    # Create joint markers
    joint_color = np.array([1.0, 0.0, 0.0])  # Red for joints
    joint_markers = create_joint_markers(joints, radius=joint_radius, color=joint_color)
    for marker in joint_markers:
        vis.add_geometry(marker)

    # Update camera to look at root joint (0th joint)
    root_joint_pos = joints[0]  # Get the position of the root joint
    update_camera_position(vis, root_joint_pos, frame_idx)

    # Update visualization
    vis.poll_events()
    vis.update_renderer()

    return vis

def main():
    # Parse arguments
    args = parse_args()

    # Load data from pickle file
    print(f"Loading data from {args.pickle_path}")
    data = load_data_from_pickle(args.pickle_path)

    # Initialize SMPLX model
    print(f"Initializing SMPLX model from {args.smpl_family_model_basedir}")
    smplx = init_smplx_model(Path(args.smpl_family_model_basedir))

    # Get frame count
    # the mean of betas: ([0.4095085 , 0.7482123 , 0.17419332, 0.65669   , 0.33137372,
    #    0.37701008, 0.26376358, 0.06721091, 0.15774761, 0.20842415,
    #    0.01375938]

    n_frames = len(data['pred_beta'])
    if args.max_frames and args.max_frames < n_frames:
        n_frames = args.max_frames

    # Process frames and collect results
    results = []

    # Initialize visualization variables
    vis = None
    if args.visualize:
        try:
            vis = create_visualizer()
            print("Initialized Open3D visualizer")
        except Exception as e:
            print(f"Error initializing visualizer: {e}")
            args.visualize = False

    # Process all frames
    for frame_idx in range(n_frames):
        print(f"Processing frame {frame_idx+1}/{n_frames}")
        frame_result = process_frame(smplx, data, frame_idx, args.depth_factor)
        results.append(frame_result)

        # Print statistics
        vertices = frame_result['vertices']
        joints = frame_result['joints']
        faces = smplx.model.faces_tensor.numpy(force=True)
        print(f"  Vertices shape: {vertices.shape}")
        print(f"  Joints shape: {joints.shape}")

        # Visualize if requested
        if args.visualize and vis is not None:
            try:
                update_visualizer(
                    vis,
                    vertices,
                    joints,
                    faces=faces,
                    mesh_color=args.mesh_color,
                    joint_radius=args.joint_radius,
                    frame_idx=frame_idx,  # Pass frame index for camera animation
                )
                time.sleep(args.vis_delay)  # Add delay for visualization
            except Exception as e:
                print(f"Visualization error: {e}")
                args.visualize = False  # Disable visualization for remaining frames
                if vis is not None:
                    try:
                        vis.destroy_window()
                    except:
                        pass
                    vis = None

        # Add delay between frames for viewing logs
        if frame_idx < n_frames - 1 and not args.visualize:
            time.sleep(args.delay)

    # Save results if requested
    if args.save_output:
        output_dir = os.path.join(project_root, 'output', 'twist_test')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'twist_results.pk')
        with open(output_file, 'wb') as f:
            pk.dump(results, f)
        print(f"Saved results to {output_file}")

    # Clean up visualization
    if vis is not None:
        try:
            vis.destroy_window()
        except:
            pass

    print(f"Successfully processed {n_frames} frames")
    if not args.visualize:
        print("To visualize results, run with --visualize flag")

if __name__ == "__main__":
    main()
