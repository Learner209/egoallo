#!/usr/bin/env python
"""
Test script for fncsmplx_aadecomp.py

This script tests all components of the SMPLX body model wrapper and visualizes
output using Open3D.
"""

import os
import sys
import numpy as np
import torch
import pytest
import open3d as o3d
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

from egoallo.middleware.third_party.HybrIK.hybrik.models.layers.smplx.fncsmplx_aadecomp import (
    SmplxModelAADecomp,
    SmplxShapedAADecomp,
    SmplxShapedAndPosedAADecomp,
    SmplxMeshAADecomp,
)
from egoallo.transforms import SE3, SO3


# Global fixtures for the test module
@pytest.fixture(scope="module")
def model_path():
    """Get the SMPL-X model path."""
    path = Path("assets/smpl_based_model")
    print(path)

    # Check if model_path exists
    if not path.exists():
        print(f"Warning: Model path {path} does not exist. Tests may fail.")
        print("Please set SMPLX_MODEL_DIR environment variable to the directory containing SMPLX models.")

    return path

@pytest.fixture(scope="module")
def device():
    """Get the device for testing."""
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {dev}")
    return dev

@pytest.fixture(scope="module")
def model(model_path, device):
    """Load the model for testing if available."""
    try:
        model = SmplxModelAADecomp.load(
            model_path,
        ).to(device)
        print("Successfully loaded SMPLX model")
        return model
    except Exception as e:
        print(f"Warning: Could not load SMPLX model: {e}")
        pytest.skip("SMPLX model not available")


class TestSmplxModelAADecomp:
    """Test cases for the SmplxModelAADecomp class and related functions."""

    def test_load_model(self, model):
        """Test loading the SMPLX model."""
        assert model is not None
        assert isinstance(model, SmplxModelAADecomp)

        # Check model exists
        assert model.model is not None
        assert model.model.faces is not None

    def test_with_shape(self, model, device):
        """Test with_shape method."""
        batch_size = 2
        num_betas = 11

        # Create random betas
        betas = torch.randn(batch_size, num_betas, device=device, dtype=torch.float32)

        # Test with_shape
        shaped = model.with_shape(betas)
        assert isinstance(shaped, SmplxShapedAADecomp)

        # Check attributes
        assert shaped.betas.shape == (batch_size, num_betas)

    def test_with_pose_decomposed(self, model, device):
        """Test with_pose_decomposed method."""
        batch_size = 2
        num_betas = 11

        # Create random betas
        betas = torch.randn(batch_size, num_betas, device=device, dtype=torch.float32)

        # Create a shaped model
        shaped = model.with_shape(betas)

        # Create random pose parameters
        # Root pose in (w, x, y, z, tx, ty, tz) format
        T_world_root = SE3.identity(device=device, dtype=torch.float32).parameters().repeat(batch_size, 1)

        # Body joint rotations
        body_quats = torch.zeros(batch_size, 21, 4, device=device)
        body_quats[..., 0] = 1.0  # w component of quaternion

        # Hand joint rotations
        left_hand_quats = torch.zeros(batch_size, 15, 4, device=device)
        left_hand_quats[..., 0] = 1.0

        right_hand_quats = torch.zeros(batch_size, 15, 4, device=device)
        right_hand_quats[..., 0] = 1.0

        # Expression parameters
        expression = torch.zeros(batch_size, 10, device=device)

        # Eye poses
        leye_pose = torch.zeros(batch_size, 1, 3, device=device)
        reye_pose = torch.zeros(batch_size, 1, 3, device=device)

        # Test with_pose_decomposed
        posed = shaped.with_pose_decomposed(
            T_world_root, body_quats, left_hand_quats, right_hand_quats,
            expression, leye_pose, reye_pose,
        )
        # Check attributes
        assert posed.body_quats.shape == (batch_size, 21, 4)
        assert posed.left_hand_quats.shape == (batch_size, 15, 4)
        assert posed.right_hand_quats.shape == (batch_size, 15, 4)

    @pytest.mark.skip
    def test_lbs(self, model, device, pickle_path):
        """Test linear blend skinning (lbs) method."""
        print(f"Mesh created with vertices and faces")


    def visualize_mesh(self, mesh, title="SMPLX Mesh"):
        """Visualize a mesh using Open3D."""
        if mesh.vertices.shape[0] > 1:
            print("Visualizing first mesh in batch")
            vertices = mesh.vertices[0].detach().cpu().numpy()
        else:
            vertices = mesh.vertices.detach().cpu().numpy()

        faces = mesh.faces.detach().cpu().numpy()

        # Create Open3D mesh
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

        # Compute normals for better visualization
        o3d_mesh.compute_vertex_normals()

        # Add color
        o3d_mesh.paint_uniform_color([0.7, 0.7, 0.9])

        # Create coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        # Visualization using Open3D version 0.16.0
        print(f"Visualizing {title}")

        # Create a visualizer instance
        visualizer = o3d.visualization.Visualizer()  # type: ignore  # Open3D visualization is available at runtime
        visualizer.create_window(window_name=title)
        visualizer.add_geometry(o3d_mesh)
        visualizer.add_geometry(coord_frame)

        # Configure rendering
        render_option = visualizer.get_render_option()
        render_option.mesh_show_wireframe = True
        render_option.point_size = 5.0

        # Set view
        view_control = visualizer.get_view_control()
        view_control.set_zoom(0.8)

        # Run the visualizer
        visualizer.run()
        visualizer.destroy_window()

        return o3d_mesh

    def visualize_skeleton(self, posed_model, title="SMPLX Skeleton"):
        """Visualize the skeleton using Open3D."""
        if posed_model.Ts_world_joint.shape[0] > 1:
            print("Visualizing first skeleton in batch")
            joints = posed_model.Ts_world_joint[0].detach().cpu().numpy()
        else:
            joints = posed_model.Ts_world_joint.detach().cpu().numpy()

        # Create Open3D point cloud for joints
        joints_pcd = o3d.geometry.PointCloud()
        joints_pcd.points = o3d.utility.Vector3dVector(joints)

        # Paint joints with a red color
        joints_pcd.paint_uniform_color([1.0, 0.0, 0.0])

        # Create coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        # Visualization using Open3D version 0.16.0
        print(f"Visualizing {title}")

        # Create a visualizer instance
        visualizer = o3d.visualization.Visualizer()  # type: ignore  # Open3D visualization is available at runtime
        visualizer.create_window(window_name=title)
        visualizer.add_geometry(joints_pcd)
        visualizer.add_geometry(coord_frame)

        # Configure rendering
        render_option = visualizer.get_render_option()
        render_option.point_size = 8.0

        # Set view
        view_control = visualizer.get_view_control()
        view_control.set_zoom(0.8)

        # Run the visualizer
        visualizer.run()
        visualizer.destroy_window()

    @pytest.mark.visualize
    def test_visualize(self, model, device):
        """Create and visualize a SMPLX model."""
        print("\nRunning visualization test...")

        # Create a model with non-zero parameters for visualization
        batch_size = 1
        num_betas = 10

        # Create some shape parameters (taller, thinner)
        betas = torch.zeros(batch_size, num_betas, device=device)
        betas[0, 0] = 2.0  # Height
        betas[0, 1] = -1.0  # Weight

        # Create a shaped model
        shaped = model.with_shape(betas)

        # Create a simple pose
        T_world_root = torch.zeros(batch_size, 7, device=device)
        T_world_root[..., 0] = 1.0  # w component of quaternion

        # Create identity pose body joints
        body_quats = torch.zeros(batch_size, 21, 4, device=device)
        body_quats[..., 0] = 1.0  # w component of quaternion

        # Hand joint rotations
        left_hand_quats = torch.zeros(batch_size, 15, 4, device=device)
        left_hand_quats[..., 0] = 1.0

        right_hand_quats = torch.zeros(batch_size, 15, 4, device=device)
        right_hand_quats[..., 0] = 1.0

        # Expression parameters
        expression = torch.zeros(batch_size, 10, device=device)

        # Eye poses
        leye_pose = torch.zeros(batch_size, 1, 3, device=device)
        reye_pose = torch.zeros(batch_size, 1, 3, device=device)

        # Create neutral pose model
        posed_neutral = shaped.with_pose_decomposed(
            T_world_root, body_quats, left_hand_quats, right_hand_quats,
            expression, leye_pose, reye_pose,
        )
        mesh_neutral = posed_neutral.lbs()

        # Visualize neutral pose
        self.visualize_mesh(mesh_neutral, "SMPLX Neutral Pose")
        self.visualize_skeleton(posed_neutral, "SMPLX Neutral Skeleton")

        # Create a more interesting pose for arms
        # Left shoulder joint rotation around z-axis (approx. joint 16)
        left_shoulder_idx = 16
        angle = np.pi / 3  # 60 degrees
        c, s = np.cos(angle / 2), np.sin(angle / 2)
        body_quats[0, left_shoulder_idx] = torch.tensor([c, 0, 0, s], device=device)

        # Right shoulder joint rotation around z-axis (approx. joint 17)
        right_shoulder_idx = 17
        angle = -np.pi / 3  # -60 degrees
        c, s = np.cos(angle / 2), np.sin(angle / 2)
        body_quats[0, right_shoulder_idx] = torch.tensor([c, 0, 0, s], device=device)

        # Elbow bends (approx. joints 18 and 19)
        left_elbow_idx = 18
        angle = np.pi / 4  # 45 degrees
        c, s = np.cos(angle / 2), np.sin(angle / 2)
        body_quats[0, left_elbow_idx] = torch.tensor([c, s, 0, 0], device=device)

        right_elbow_idx = 19
        angle = np.pi / 4  # 45 degrees
        c, s = np.cos(angle / 2), np.sin(angle / 2)
        body_quats[0, right_elbow_idx] = torch.tensor([c, s, 0, 0], device=device)

        # Create posed model
        posed = shaped.with_pose_decomposed(
            T_world_root, body_quats, left_hand_quats, right_hand_quats,
            expression, leye_pose, reye_pose,
        )
        mesh = posed.lbs()

        # Visualize posed model
        self.visualize_mesh(mesh, "SMPLX Posed Mesh")
        self.visualize_skeleton(posed, "SMPLX Posed Skeleton")

        print("Visualization test completed")


def main():
    """Run the tests with visualization."""
    parser = argparse.ArgumentParser(description='Test SMPLX model wrapper with visualization')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization tests')
    parser.add_argument('--model_dir', type=str, help='Directory containing SMPLX models')

    args = parser.parse_args()

    # Set model directory if provided
    if args.model_dir:
        os.environ['SMPLX_MODEL_DIR'] = args.model_dir

    # Build pytest arguments
    pytest_args = [__file__]

    # Add visualization marker if requested
    if args.visualize:
        pytest_args.append('-m')
        pytest_args.append('visualize')
    else:
        pytest_args.append('-m')
        pytest_args.append('not visualize')

    # Run pytest
    pytest.main(pytest_args)


if __name__ == "__main__":
    main()
