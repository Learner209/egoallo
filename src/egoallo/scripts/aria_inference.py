from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
import time
from pathlib import Path

import os
import cv2

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import torch
import viser
import yaml

from egoallo import fncsmpl, fncsmpl_extensions
from egoallo.data.aria_mps import load_point_cloud_and_find_ground
from egoallo.guidance_optimizer_jax import GuidanceMode
from egoallo.hand_detection_structs import (
    CorrespondedAriaHandWristPoseDetections,
    CorrespondedHamerDetections,
)
from egoallo.inference_utils import (
    InferenceInputTransforms,
    InferenceTrajectoryPaths,
    load_denoiser,
)
from egoallo.sampling import (
    run_sampling_with_masked_data,
)
from egoallo.transforms import SE3, SO3
from egoallo.vis_helpers import visualize_traj_and_hand_detections
from egoallo.training_utils import ipdb_safety_net
from egoallo.config.inference.inference_defaults import InferenceConfig

if TYPE_CHECKING:
    from third_party.cloudrender.cloudrender.render.pointcloud import Pointcloud

from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import RecordableTypeId, StreamId


class AriaInference:
    def __init__(self, config: InferenceConfig, traj_root: Path, output_path: Path, glasses_x_angle_offset: float = 0.0):
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
            self.traj_paths.vrs_file, self.traj_paths.slam_root_dir, fps=30
        ).to(device=self.device)

        # Note the off-by-one for Ts_world_cpf, which we need for relative transform computation.
        self.config.traj_length = len(self.transforms.Ts_world_cpf) - self.config.start_index - 1
        self.Ts_world_cpf = (
            SE3(
                self.transforms.Ts_world_cpf[
                    self.config.start_index : self.config.start_index + self.config.traj_length + 1
                ]
            )
            @ SE3.from_rotation(
                SO3.from_x_radians(
                    self.transforms.Ts_world_cpf.new_tensor(glasses_x_angle_offset)
                )
            )
        ).parameters()
        self.pose_timestamps_sec = self.transforms.pose_timesteps[
            self.config.start_index + 1 : self.config.start_index + self.config.traj_length + 1
        ]
        self.Ts_world_device = self.transforms.Ts_world_device[
            self.config.start_index + 1 : self.config.start_index + self.config.traj_length + 1
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

    def load_pc_and_find_ground(self) -> tuple[Pointcloud.PointcloudContainer, np.ndarray, float]:
        """Load point cloud and find ground plane."""
        pc_container, points_data, floor_z = load_point_cloud_and_find_ground(
            points_path=self.traj_paths.points_path,
            cache_files=False,
            return_points="filtered",
        )
        return pc_container, points_data, floor_z

    def extract_rgb_frames(self, times: list[float] | list[int] | None = None) -> list[np.ndarray]:
        """Extract RGB frames from VRS file and save as video.
        
        Args:
            times: Optional list of timestamps or frame indices to extract frames at. 
                  If float values are provided, they are treated as timestamps in seconds.
                  If int values are provided, they are treated as frame indices.
                  If None, uses sequential frame indices.
        """
        vrs_data_provider = data_provider.create_vrs_data_provider(str(self.traj_paths.vrs_file))
        rgb_stream_id = vrs_data_provider.get_stream_id_from_label("camera-rgb")
        
        rgb_frames = []
        if times is not None:
            # Check if times contains floats or ints
            is_float = any(isinstance(t, float) for t in times)
            
            for time in times:
                if is_float:
                    # Use timestamp-based extraction for float values
                    rgb_record = vrs_data_provider.get_image_data_by_time_ns(rgb_stream_id, int(time * 1e9))
                else:
                    # Use index-based extraction for int values
                    rgb_record = vrs_data_provider.get_image_data_by_index(rgb_stream_id, time)
                
                if rgb_record is not None and rgb_record[0].pixel_frame is not None:
                    rgb_frames.append(rgb_record[0].to_numpy_array())
        else:
            # Extract frames by sequential indices
            for frame_idx in range(self.config.start_index, self.config.start_index + self.config.traj_length + 1):
                rgb_record = vrs_data_provider.get_image_data_by_index(rgb_stream_id, frame_idx)
                if rgb_record is not None and rgb_record[0].pixel_frame is not None:
                    rgb_frames.append(rgb_record[0].to_numpy_array())

        print(f"Extracted {len(rgb_frames)} RGB frames")
        
        # Save frames as video
        if len(rgb_frames) > 0:
            first_frame = rgb_frames[0]
            height, width = first_frame.shape[:2]
            
            output_path = Path(self.config.output_dir) / "rgb_frames.mp4"
            output_path.parent.mkdir(exist_ok=True, parents=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, 30.0, (width, height))
            
            for frame in rgb_frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                
            out.release()
            print(f"Saved RGB video to {output_path}")
            
        return rgb_frames

    def _extract_rgb_frames(self, times: list[float] | list[int] | None = None) -> list[np.ndarray]:
        """Extract RGB frames from VRS file and save as video.
        
        Args:
            times: Optional list of timestamps or frame indices to extract frames at. 
                  If float values are provided, they are treated as timestamps in seconds.
                  If int values are provided, they are treated as frame indices.
                  If None, uses sequential frame indices.
        """
        vrs_data_provider = data_provider.create_vrs_data_provider(str(self.traj_paths.vrs_file))
        rgb_stream_id = vrs_data_provider.get_stream_id_from_label("camera-rgb")
        time_domain = TimeDomain.DEVICE_TIME
        option = TimeQueryOptions.CLOSEST
        
        rgb_frames = []
        seq = vrs_data_provider.deliver_queued_sensor_data()
        breakpoint()
        obj = next(seq)
        while True:
            ns = obj.get_time_ns(TimeDomain.DEVICE_TIME)
            pixel = vrs_data_provider.get_image_data_by_time_ns(rgb_stream_id, ns, time_domain, option)
            assert pixel is not None
            rgb_frames.append(pixel[0].to_numpy_array())

            try:
                obj = next(seq)

            except StopIteration:
                break

        print(f"Extracted {len(rgb_frames)} RGB frames")
        
        # Save frames as video
        if len(rgb_frames) > 0:
            first_frame = rgb_frames[0]
            height, width = first_frame.shape[:2]
            
            output_path = Path(self.config.output_dir) / "rgb_frames.mp4"
            output_path.parent.mkdir(exist_ok=True, parents=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, 30.0, (width, height))
            
            for frame in rgb_frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                
            out.release()
            print(f"Saved RGB video to {output_path}")
            
        return rgb_frames




def main(config: InferenceConfig, traj_root: Path, output_path: Path, glasses_x_angle_offset: float = 0.0) -> None:
    inference = AriaInference(config, traj_root, output_path, glasses_x_angle_offset)
    # inference.extract_rgb_frames()
    inference._extract_rgb_frames()

if __name__ == "__main__":
    import tyro

    ipdb_safety_net()
    tyro.cli(main)

