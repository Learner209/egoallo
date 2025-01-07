from __future__ import annotations

import cv2
import numpy as np
import torch
import tyro
from pathlib import Path

from egoallo import fncsmpl
from egoallo.data.dataclass import EgoTrainingData
from egoallo.types import DenoiseTrajType, DenoiseTrajTypeLiteral
from egoallo.training_utils import ipdb_safety_net

def visualize_saved_trajectory(
    trajectory_path: tuple[Path, ...],
    trajectory_type: DenoiseTrajTypeLiteral,
    smplh_model_path: Path = Path("./data/smplh/neutral/model.npz"),
    output_dir: Path = Path("./visualization_output"),
    combine_videos: bool = True, # if `combine_videos` is True, then the ground truth and predicted trajectories will be combined into a single video, and trajectory_paths should be of length 2.
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> None:
    """Visualize saved trajectory data.
    
    Args:
        trajectory_path: Path to saved trajectory file (.pt or .npz)
        smplh_model_path: Path to SMPL-H model file
        output_dir: Directory to save visualization outputs
        combine_videos: Whether to combine ground truth and predicted videos side by side
    """
    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)

    assert len(trajectory_path) == 2 and combine_videos or len(trajectory_path) == 1 and not combine_videos, "trajectory_path must be a tuple of length 2"
    
    # Load SMPL-H body model
    body_model = fncsmpl.SmplhModel.load(smplh_model_path).to(device)
    
    # Generate output paths
    gt_path = output_dir / "gt_trajectory.mp4"
    pred_path = output_dir / "pred_trajectory.mp4" 
    combined_path = output_dir / "combined_trajectory.mp4"

    if combine_videos:
        # Load both ground truth and predicted trajectories
        gt_traj = torch.load(trajectory_path[0], map_location=device)
        pred_traj = torch.load(trajectory_path[1], map_location=device)
        
        # Visualize ground truth
        EgoTrainingData.visualize_ego_training_data(
            gt_traj,
            body_model,
            str(gt_path)
        )
        
        # Visualize prediction
        EgoTrainingData.visualize_ego_training_data(
            pred_traj,
            body_model, 
            str(pred_path)
        )
    else:
        # Just load and visualize prediction
        pred_traj = torch.load(trajectory_path[0], map_location=device)
        EgoTrainingData.visualize_ego_training_data(
            pred_traj,
            body_model,
            str(pred_path)
        )

    # Combine videos if requested and both exist
    if combine_videos and gt_path.exists() and pred_path.exists():
        # Open both videos
        gt_video = cv2.VideoCapture(str(gt_path))
        pred_video = cv2.VideoCapture(str(pred_path))

        # Get video properties
        width = int(gt_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(gt_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = gt_video.get(cv2.CAP_PROP_FPS)

        # Create video writer for combined video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(combined_path), fourcc, fps, (width*2, height))

        while True:
            ret1, frame1 = gt_video.read()
            ret2, frame2 = pred_video.read()
            
            if not ret1 or not ret2:
                break

            # Add text labels
            cv2.putText(frame1, 'Ground Truth', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(frame2, 'Prediction', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            # Combine frames side by side
            combined_frame = np.hstack((frame1, frame2))
            out.write(combined_frame)

        # Release everything
        gt_video.release()
        pred_video.release()
        out.release()

def main(
    trajectory_path: tuple[Path, ...],
    trajectory_type: DenoiseTrajTypeLiteral,
    smplh_model_path: Path = Path("./data/smplh/neutral/model.npz"),
    output_dir: Path = Path("./visualization_output"),
    combine_videos: bool = True,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> None:
    """Main entry point for trajectory visualization.
    
    Args:
        trajectory_path: Path to saved trajectory file
        smplh_model_path: Path to SMPL-H model file
        output_dir: Directory to save visualization outputs
        combine_videos: Whether to combine ground truth and predicted videos
    """
    ipdb_safety_net()

    visualize_saved_trajectory(
        trajectory_path=trajectory_path,
        trajectory_type=trajectory_type,
        smplh_model_path=smplh_model_path,
        output_dir=output_dir,
        combine_videos=combine_videos,
        device=device
    )

if __name__ == "__main__":
    tyro.cli(main)
