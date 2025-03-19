"""Test the optimized pyrender-based rendering function."""

import os
import torch
from pathlib import Path
from egoallo.data.dataclass import EgoTrainingData

project_root = Path(__file__).resolve().parent.parent.parent.parent


def main():
    """Test the optimized rendering function."""
    # Configure the test
    output_path = "pyrender_test_output.mp4"

    test_traj_path = "exp/test-amass-Mar-18/test/Gym_010_cooking1_0_775.npz_t0_775/gt_test/Gym_010_cooking1_0_775.npz_t0_775.pt"

    # Path to SMPL model
    smplh_model_path = (
        project_root / "assets" / "smpl_based_model" / "smplh" / "SMPLH_NEUTRAL.pkl"
    )
    if not os.path.exists(smplh_model_path):
        print(f"SMPL model file not found at {smplh_model_path}")
        print("Please specify a valid SMPL model path")
        return

    # Load trajectory
    try:
        print(f"Loading trajectory from {test_traj_path}")
        traj = torch.load(test_traj_path, map_location="cpu")
        print(f"Loaded trajectory: {type(traj)}")

        # Visualize using the optimized pyrender-based function
        print(f"Rendering trajectory to {output_path}")
        print("This may take a while...")

        # Set PYOPENGL_PLATFORM to 'egl' to suppress logging spam from pyrender
        os.environ["PYOPENGL_PLATFORM"] = "egl"

        EgoTrainingData.visualize_ego_training_data(
            traj,
            smplh_model_path,
            output_path,
        )
        print(f"Rendering complete. Output saved to {output_path}")

    except Exception as e:
        print(f"Error loading or rendering trajectory: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
