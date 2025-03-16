from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

import os
import cv2

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import torch

from egoallo.data.aria_mps import load_point_cloud_and_find_ground
from egoallo.hand_detection_structs import (
    CorrespondedAriaHandWristPoseDetections,
    CorrespondedHamerDetections,
)
from egoallo.inference_utils import (
    InferenceInputTransforms,
    InferenceTrajectoryPaths,
)
from egoallo.transforms import SE3, SO3
from egoallo.training_utils import ipdb_safety_net
from egoallo.config.inference.inference_defaults import InferenceConfig
from egoallo.utils.setup_logger import setup_logger

if TYPE_CHECKING:
    from third_party.cloudrender.cloudrender.render.pointcloud import Pointcloud


import hashlib
import tempfile

logger = setup_logger(output=None, name=__name__)


class AriaInference:
    def __init__(
        self,
        config: InferenceConfig,
        traj_root: Path,
        output_path: Path,
        glasses_x_angle_offset: float = 0.0,
    ):
        self.config = config
        self.device = torch.device("cuda")
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.traj_paths = InferenceTrajectoryPaths.find(traj_root)
        if self.traj_paths.splat_path is not None:
            print("Found splat at", self.traj_paths.splat_path)
        else:
            print("No scene splat found.")

        # Read transforms from VRS / MPS, downsampled.
        self.transforms = InferenceInputTransforms.load(
            self.traj_paths.vrs_file,
            self.traj_paths.slam_root_dir,
            fps=30,
        ).to(device=self.device)

        # Note the off-by-one for Ts_world_cpf, which we need for relative transform computation.
        self.config.traj_length = (
            len(self.transforms.Ts_world_cpf) - self.config.start_index - 1
        )
        self.Ts_world_cpf = (
            SE3(
                self.transforms.Ts_world_cpf[
                    self.config.start_index : self.config.start_index
                    + self.config.traj_length
                    + 1
                ],
            )
            @ SE3.from_rotation(
                SO3.from_x_radians(
                    self.transforms.Ts_world_cpf.new_tensor(glasses_x_angle_offset),
                ),
            )
        ).parameters()
        self.pose_timestamps_sec = self.transforms.pose_timesteps[
            self.config.start_index + 1 : self.config.start_index
            + self.config.traj_length
            + 1
        ]
        self.Ts_world_device = self.transforms.Ts_world_device[
            self.config.start_index + 1 : self.config.start_index
            + self.config.traj_length
            + 1
        ]
        del self.transforms

        self.setup_detections()
        print(f"{self.Ts_world_cpf.shape=}")

    def setup_detections(self):
        # Get temporally corresponded HaMeR detections.
        if self.traj_paths.hamer_outputs is not None:
            self.hamer_detections = CorrespondedHamerDetections.load(
                self.traj_paths.hamer_outputs,
                self.pose_timestamps_sec,
            ).to(self.device)
        else:
            print("No hand detections found.")
            self.hamer_detections = None

        # Get temporally corresponded Aria wrist and palm estimates.
        if self.traj_paths.wrist_and_palm_poses_csv is not None:
            self.aria_detections = CorrespondedAriaHandWristPoseDetections.load(
                self.traj_paths.wrist_and_palm_poses_csv,
                self.pose_timestamps_sec,
                Ts_world_device=self.Ts_world_device.numpy(force=True),
            ).to(self.device)
        else:
            print("No Aria hand detections found.")
            self.aria_detections = None

    def load_pc_and_find_ground(
        self,
    ) -> tuple[Pointcloud.PointcloudContainer, np.ndarray, float]:
        """Load point cloud and find ground plane."""
        pc_container, points_data, floor_z = load_point_cloud_and_find_ground(
            points_path=self.traj_paths.points_path,
            cache_files=True,
            return_points="filtered",
        )
        return pc_container, points_data, floor_z

    def extract_rgb_frames(
        self,
        times: list[float] | list[int] | None = None,
        cache_files: bool = True,
    ) -> list[np.ndarray]:
        """Extract RGB frames from VRS file and save as video.

        Args:
            times: Optional list of timestamps or frame indices to extract frames at.
                  If float values are provided, they are treated as timestamps in seconds.
                  If int values are provided, they are treated as frame indices.
                  If None, uses sequential frame indices.
            cache_files: Whether to cache extracted frames to disk for faster future loading
        """
        # Create hash of video path and times to use as cache filename
        cache_key = f"{str(self.traj_paths.ego_preview_path)}_{str(times)}"
        frames_path_hash = hashlib.md5(cache_key.encode()).hexdigest()

        # Create persistent temp directory if it doesn't exist
        temp_cache_dir = Path(tempfile.gettempdir()) / "aria_frames_cache"
        temp_cache_dir.mkdir(exist_ok=True, parents=True)

        frames_cache_path = temp_cache_dir / f"{frames_path_hash}_frames.npz"

        # Check if we should use cached files
        if cache_files and frames_cache_path.exists():
            logger.debug("Loading cached frames from %s", frames_cache_path)
            rgb_frames = list(np.load(frames_cache_path)["frames"])
        else:
            # Extract frames from video
            video = cv2.VideoCapture(str(self.traj_paths.ego_preview_path))
            rgb_frames = []

            if times is not None:
                for frame_idx in times:
                    video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = video.read()
                    if ret:
                        rgb_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                while True:
                    ret, frame = video.read()
                    if not ret:
                        break
                    rgb_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            video.release()
            logger.debug(f"Extracted {len(rgb_frames)} RGB frames")

            # Cache frames if enabled
            if cache_files and len(rgb_frames) > 0:
                np.savez_compressed(frames_cache_path, frames=rgb_frames)
                logger.debug("Cached frames to %s", frames_cache_path)

        # Save frames as video
        if len(rgb_frames) > 0:
            first_frame = rgb_frames[0]
            height, width = first_frame.shape[:2]

            output_path = Path(self.config.output_dir) / "rgb_frames.mp4"
            output_path.parent.mkdir(exist_ok=True, parents=True)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(output_path), fourcc, 30.0, (width, height))

            for frame in rgb_frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            out.release()
            print(f"Saved RGB video to {output_path}")

        return rgb_frames


def main(
    config: InferenceConfig,
    traj_root: Path,
    output_path: Path,
    glasses_x_angle_offset: float = 0.0,
) -> None:
    inference = AriaInference(config, traj_root, output_path, glasses_x_angle_offset)
    inference.extract_rgb_frames()


if __name__ == "__main__":
    import tyro

    ipdb_safety_net()
    tyro.cli(main)
