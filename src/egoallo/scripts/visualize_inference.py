from __future__ import annotations
import builtins

import cv2
import numpy as np
import torch
import tyro
from pathlib import Path

from egoallo.config.inference.defaults import InferenceConfig
from egoallo.data.dataclass import EgoTrainingData
from egoallo.scripts.aria_inference import AriaInference
from egoallo.types import DenoiseTrajType, DenoiseTrajTypeLiteral, DatasetType
from egoallo.training_utils import ipdb_safety_net
from egoallo.mapping import SMPLH_BODY_JOINTS


def visualize_saved_trajectory(
    config: InferenceConfig,
    trajectory_path: tuple[Path, ...],
    trajectory_type: DenoiseTrajTypeLiteral,
    dataset_type: DatasetType,
    smplh_model_path: Path = Path("assets/smpl_based_model/smplh/SMPLH_MALE.pkl"),
    output_dir: Path = Path("./visualization_output"),
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> None:
    """Visualize saved trajectory data.

    Args:
        trajectory_path: Path to saved trajectory file (.pt or .npz).
                       For COMPARISON type, provide (gt_path, pred_path).
                       For EGOEXO type, provide single trajectory path.
        dataset_type: Type of visualization - "egoexo" or "comparison"
        smplh_model_path: Path to SMPL-H model file
        output_dir: Directory to save visualization outputs
    """
    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load SMPL-H body model

    # Generate output paths
    gt_path = output_dir / "gt_trajectory.mp4"
    pred_path = output_dir / "pred_trajectory.mp4"
    combined_path = output_dir / "combined_trajectory.mp4"

    assert len(trajectory_path) == 2, "trajectory_path must be a tuple of length 2"

    if dataset_type in ("AdaptiveAmassHdf5Dataset", "VanillaEgoAmassHdf5Dataset"):
        # Load both ground truth and predicted trajectories
        gt_traj: DenoiseTrajType = torch.load(trajectory_path[0], map_location=device)
        pred_traj: DenoiseTrajType = torch.load(trajectory_path[1], map_location=device)

        EgoTrainingData.visualize_ego_training_data(
            gt_traj,
            smplh_model_path,
            str(gt_path),
            online_render=config.online_render,
        )

        EgoTrainingData.visualize_ego_training_data(
            pred_traj,
            smplh_model_path,
            str(pred_path),
            online_render=config.online_render,
        )
    elif dataset_type in ("AriaDataset", "AriaInferenceDataset", "EgoExoDataset"):
        traj_root = config.egoexo.traj_root
        from egoallo.network import AbsoluteDenoiseTraj

        pred_traj: DenoiseTrajType = torch.load(trajectory_path[1], map_location=device)

        assert isinstance(pred_traj, AbsoluteDenoiseTraj), (
            "Prediction trajectory should be AbsoluteDenoiseTraj for visualization."
        )
        assert pred_traj.metadata.stage == "postprocessed", (
            "Prediction trajectory should be postprocessed before visualization."
        )

        frame_keys = (
            pred_traj.metadata.frame_keys
            if pred_traj.metadata.frame_keys and len(pred_traj.metadata.frame_keys) > 0
            else None
        )
        aria_inference_toolkit = AriaInference(
            config,
            traj_root,
            output_path=output_dir,
            glasses_x_angle_offset=0.0,
        )
        rgb_frames = aria_inference_toolkit.extract_rgb_frames(
            list(frame_keys),
            cache_files=True,
        )
        pc_container, points_data, floor_z = (
            aria_inference_toolkit.load_pc_and_find_ground()
        )

        EgoTrainingData.visualize_ego_training_data(
            pred_traj,
            smplh_model_path,
            str(pred_path),
            scene_obj=pc_container,
            online_render=config.online_render,
        )
        # Save frames as video
        if len(rgb_frames) > 0:
            first_frame = rgb_frames[0]
            height, width = first_frame.shape[:2]

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(gt_path), fourcc, 30.0, (width, height))

            for frame in rgb_frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            out.release()
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")

    # Get masked joints from visibility mask
    masked_joints = []
    assert pred_traj.visible_joints_mask is not None, (
        "visible_joints_mask is not present in the trajectory"
    )  # gt_traj can be unbound
    # Get indices of joints that are masked in at least one frame
    masked_indices = torch.where(~pred_traj.visible_joints_mask.any(dim=0))[0]
    # Map indices to joint names using SMPLH joint names
    masked_joints = [SMPLH_BODY_JOINTS[idx] for idx in masked_indices]

    gt_video = cv2.VideoCapture(str(gt_path))
    pred_video = cv2.VideoCapture(str(pred_path))

    # Get video properties
    gt_width = int(gt_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    gt_height = int(gt_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pred_width = int(pred_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    pred_height = int(pred_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = gt_video.get(cv2.CAP_PROP_FPS)

    # Calculate dimensions for combined video
    max_width = max(gt_width, pred_width)
    total_height = gt_height + pred_height

    # Create video writer for combined video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(combined_path), fourcc, fps, (max_width, total_height))
    while True:
        ret1, frame1 = gt_video.read()
        ret2, frame2 = pred_video.read()

        if not ret1 or not ret2:
            break

        # Pad frames to match max width if needed
        if gt_width < max_width:
            pad_width = max_width - gt_width
            frame1 = cv2.copyMakeBorder(
                frame1,
                0,
                0,
                0,
                pad_width,
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
            )
        if pred_width < max_width:
            pad_width = max_width - pred_width
            frame2 = cv2.copyMakeBorder(
                frame2,
                0,
                0,
                0,
                pad_width,
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
            )
        # Add text labels with colorful text
        cv2.putText(
            frame1,
            "Ground Truth",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (50, 205, 50),
            2,
        )  # Lime Green
        cv2.putText(
            frame2,
            "Prediction",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 140, 0),
            2,
        )  # Dark Orange

        # Calculate mask ratio
        spatial_mask_ratio = len(masked_joints) / len(SMPLH_BODY_JOINTS)

        # Add masked joints text with mask ratio
        y_offset = 60
        cv2.putText(
            frame1,
            f"Masked Joints ({spatial_mask_ratio:.1%}):",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (147, 112, 219),
            1,
        )  # Medium Purple
        for i, joint in enumerate(masked_joints):
            y_pos = y_offset + (i + 1) * 20
            cv2.putText(
                frame1,
                joint,
                (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 20, 147),
                1,
            )  # Deep Pink

        # Combine frames vertically
        combined_frame = np.vstack((frame1, frame2))
        out.write(combined_frame)

    # Release everything
    gt_video.release()
    pred_video.release()
    out.release()


def main(
    config: InferenceConfig,
    trajectory_path: tuple[Path, ...],
    trajectory_type: DenoiseTrajTypeLiteral,
    dataset_type: DatasetType,
    smplh_model_path: Path = Path("assets/smpl_based_model/smplh/SMPLH_MALE.pkl"),
    output_dir: Path = Path("./visualization_output"),
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    debug: bool = False,
) -> None:
    """Main entry point for trajectory visualization.

    Args:
        trajectory_path: Path to saved trajectory file
        dataset_type: Type of visualization - "egoexo" or "comparison"
        smplh_model_path: Path to SMPL-H model file
        output_dir: Directory to save visualization outputs
    """
    if debug:
        builtins.breakpoint()

    ipdb_safety_net()

    visualize_saved_trajectory(
        config=config,
        trajectory_path=trajectory_path,
        trajectory_type=trajectory_type,
        dataset_type=dataset_type,
        smplh_model_path=smplh_model_path,
        output_dir=output_dir,
        device=device,
    )


if __name__ == "__main__":
    tyro.cli(main)
