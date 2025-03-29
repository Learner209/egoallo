"""
Test script for SMPLViewer.render_sequence
This script fabricates the necessary data to test the render_sequence function
without requiring a real DenoiseTrajType object.
"""

import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from egoallo.transforms import SE3, SO3
from egoallo.middleware.third_party.HybrIK.hybrik.models.layers.smplh.fncsmplh import (
    SmplhModel,
)
from egoallo import training_utils


@dataclass
class MockMetadata:
    """Mock metadata class to simulate DenoiseTrajType.metadata"""

    stage: str = "postprocessed"
    dataset_type: str = "AdaptiveAmassHdf5Dataset"
    aux_joints_wrt_world_placeholder: Optional[torch.Tensor] = None
    aux_visible_joints_mask_placeholder: Optional[torch.Tensor] = None


@dataclass
class MockDenoiseTraj:
    """Mock class to simulate DenoiseTrajType"""

    R_world_root: torch.Tensor  # [seq_len, 3, 3]
    t_world_root: torch.Tensor  # [seq_len, 3]
    body_rotmats: torch.Tensor  # [seq_len, num_joints, 3, 3]
    betas: torch.Tensor  # [num_betas]
    joints_wrt_world: torch.Tensor  # [seq_len, num_joints, 3]
    visible_joints_mask: torch.Tensor  # [seq_len, num_joints]
    metadata: MockMetadata = field(default_factory=MockMetadata)

    def to(self, device):
        """Move tensors to device"""
        self.R_world_root = self.R_world_root.to(device)
        self.t_world_root = self.t_world_root.to(device)
        self.body_rotmats = self.body_rotmats.to(device)
        self.betas = self.betas.to(device)
        self.joints_wrt_world = self.joints_wrt_world.to(device)
        self.visible_joints_mask = self.visible_joints_mask.to(device)
        return self

    def map(self, fn):
        """Apply function to tensors"""
        result = MockDenoiseTraj(
            R_world_root=fn(self.R_world_root),
            t_world_root=fn(self.t_world_root),
            body_rotmats=fn(self.body_rotmats),
            betas=fn(self.betas),
            joints_wrt_world=fn(self.joints_wrt_world),
            visible_joints_mask=fn(self.visible_joints_mask),
            metadata=self.metadata,
        )
        return result


def create_mock_trajectory(
    smplh_model_path: Path,
    seq_len=60,
    num_joints=22,
    num_betas=10,
):
    """Create a mock trajectory for testing"""
    # Create rotation matrices for root (identity with small random rotation)
    R_world_root = torch.eye(3).unsqueeze(0).repeat(seq_len, 1, 1)

    # Add a circular motion to make the visualization interesting
    t = torch.linspace(0, 2 * np.pi, seq_len)
    radius = 1.0

    # Create translation for root (circular path)
    t_world_root = torch.zeros(seq_len, 3)
    t_world_root[:, 0] = radius * torch.cos(t)  # x coordinate
    t_world_root[:, 1] = radius * torch.sin(t)  # y coordinate
    t_world_root[:, 2] = 1.0  # constant height

    # Create body rotation matrices (identity with small random rotation for each joint)
    body_rotmats = (
        torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(seq_len, num_joints - 1, 1, 1)
    )

    # Create body shape parameters (zeros for neutral shape)
    betas = torch.zeros(seq_len, num_betas)

    body_model = SmplhModel.load(
        smplh_model_path,
        use_pca=False,
        batch_size=seq_len,
    )
    T_world_root = SE3.from_rotation_and_translation(
        SO3.from_matrix(R_world_root),
        t_world_root,
    ).parameters()

    shaped = body_model.with_shape(betas)
    posed = shaped.with_pose_decomposed(
        T_world_root=T_world_root,
        body_quats=SO3.from_matrix(body_rotmats).wxyz,
        left_hand_quats=None,
        right_hand_quats=None,
    )
    joints_wrt_world = torch.cat(
        [posed.T_world_root[..., None, 4:7], posed.Ts_world_joint[..., 4:7]],
        dim=-2,
    )[..., :num_joints, :]
    # joints_wrt_world = posed.T_world_root[..., 4:7].unsqueeze(-2).repeat(1, num_joints, 1)

    visible_joints_mask = torch.ones(seq_len, num_joints, dtype=torch.bool)

    # Create mock trajectory
    mock_traj = MockDenoiseTraj(
        R_world_root=R_world_root,
        t_world_root=t_world_root,
        body_rotmats=body_rotmats,
        betas=betas,
        joints_wrt_world=joints_wrt_world,
        visible_joints_mask=visible_joints_mask,
    )

    return mock_traj


def main():
    # Path to SMPL-H model - replace with actual path
    smpl_family_model_basedir = Path("assets/smpl_based_model")

    # Check if the model path exists
    if not smpl_family_model_basedir.exists():
        print(f"Warning: SMPL-H model not found at {smpl_family_model_basedir}")
        print("Please provide a valid path to the SMPL-H model")
        print("You can download it from https://mano.is.tue.mpg.de/")
        return

    # Create mock trajectory
    _mock_traj = create_mock_trajectory(
        seq_len=600,
        num_betas=16,
        smplh_model_path=smpl_family_model_basedir,
    )


if __name__ == "__main__":
    training_utils.ipdb_safety_net()
    main()
