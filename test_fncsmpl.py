import torch
import numpy as np
from pathlib import Path
from smplx import SMPLH
from egoallo.training_utils import ipdb_safety_net
import pyvista as pv

from egoallo.fncsmpl_library import SmplhModel as SmplhModel


def aa_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle to wxyz quaternion."""
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / (angle + 1e-8)
    half_angle = angle / 2
    w = torch.cos(half_angle)
    xyz = axis * torch.sin(half_angle)
    return torch.cat([w, xyz], dim=-1)


def test_smplh_vs_smplx():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Path to SMPL-H model file (adjust as needed)
    model_path = Path("assets/smpl_based_model/smplh/SMPLH_NEUTRAL.pkl")
    num_betas = 16  # Number of shape coefficients to test

    # Load custom model
    custom_model = SmplhModel.load(
        model_path,
        use_pca=False,
        batch_size=50,
        num_betas=num_betas,
    ).to(device=device)
    # custom_model = SmplhModel.load(model_path)

    # Load SMPLX model
    smplx_model = SMPLH(
        model_path=model_path,
        use_pca=False,
        num_betas=num_betas,
        num_pca_comps=45,  # Not used since use_pca=False
        ext="pkl",
    ).to(device=device, dtype=dtype)

    # Generate random input parameters
    batch_size = 50
    np.random.seed(42)
    torch.manual_seed(42)

    # Shape parameters
    betas = torch.randn(batch_size, num_betas, dtype=dtype, device=device)

    # Root orientation (axis-angle) and translation
    global_orient = torch.randn(batch_size, 3, dtype=dtype, device=device) * 0.1
    trans = torch.randn(batch_size, 3, dtype=dtype, device=device) * 0.1

    # Body pose (21 joints, axis-angle)
    body_pose = torch.randn(batch_size, 21 * 3, dtype=dtype, device=device) * 0.1

    # Hand poses (15 joints each, axis-angle)
    left_hand_pose = torch.randn(batch_size, 15 * 3, dtype=dtype, device=device) * 0.1
    right_hand_pose = torch.randn(batch_size, 15 * 3, dtype=dtype, device=device) * 0.1

    # Run SMPLX
    smplx_output = smplx_model(
        betas=betas,
        global_orient=global_orient,
        body_pose=body_pose,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        transl=trans,
    )
    smplx_verts = smplx_output.vertices

    # Convert parameters for custom model
    # Convert global_orient to SE3 (quaternion + translation)
    global_orient_quat = aa_to_quaternion(global_orient.reshape(-1, 3)).view(
        batch_size,
        4,
    )
    T_world_root = torch.cat([global_orient_quat, trans], dim=-1)  # (7,)

    # Convert body and hand poses to quaternions
    body_quats = aa_to_quaternion(body_pose.view(-1, 3)).view(batch_size, 21, 4)
    left_quats = aa_to_quaternion(left_hand_pose.view(-1, 3)).view(batch_size, 15, 4)
    right_quats = aa_to_quaternion(right_hand_pose.view(-1, 3)).view(batch_size, 15, 4)

    # Run custom model
    shaped = custom_model.with_shape(betas)
    posed = shaped.with_pose_decomposed(
        T_world_root=T_world_root,
        body_quats=body_quats,
        left_hand_quats=left_quats,
        right_hand_quats=right_quats,
    )
    custom_mesh = posed.lbs()
    custom_verts = custom_mesh.vertices

    custom_verts_np = custom_verts.squeeze().cpu().numpy(force=True)
    smplx_verts_np = smplx_verts.squeeze().cpu().numpy(force=True)

    # Create PyVista meshes
    custom_mesh = pv.PolyData(custom_verts_np)
    smplx_mesh = pv.PolyData(smplx_verts_np)

    # Plot both meshes
    plotter = pv.Plotter()
    plotter.add_mesh(custom_mesh, color="red", label="Custom Model")
    plotter.add_mesh(smplx_mesh, color="blue", label="SMPL-X Model")
    plotter.add_legend()
    plotter.show()

    # Compare vertices
    max_diff = torch.max(torch.abs(custom_verts - smplx_verts)).item()
    mse = torch.mean((custom_verts - smplx_verts) ** 2).item()

    print(f"Max vertex difference: {max_diff:.6f}")
    print(f"MSE: {mse:.6f}")

    # Assertions (adjust tolerance based on your requirements)
    assert max_diff < 1e-4, "Vertex positions differ significantly"
    assert mse < 1e-8, "MSE between vertices is too high"


if __name__ == "__main__":
    ipdb_safety_net()
    test_smplh_vs_smplx()
