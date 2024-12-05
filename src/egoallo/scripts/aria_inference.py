from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

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
from egoallo.sampling import run_sampling_with_stitching
from egoallo.transforms import SE3, SO3
from egoallo.vis_helpers import visualize_traj_and_hand_detections
from egoallo.training_utils import ipdb_safety_net
from egoallo.config.inference_config import InferenceConfig

def main(config: InferenceConfig) -> None:

    if config.use_ipdb:
        import ipdb; ipdb.set_trace()

    device = torch.device("cuda")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    traj_paths = InferenceTrajectoryPaths.find(config.traj_root, config.output_dir, soft_link=False)
    if traj_paths.splat_path is not None:
        print("Found splat at", traj_paths.splat_path)
    else:
        print("No scene splat found.")
    # Get point cloud + floor.
    points_data, floor_z = load_point_cloud_and_find_ground(points_path=traj_paths.points_path)

    # Read transforms from VRS / MPS, downsampled.
    transforms = InferenceInputTransforms.load(
        traj_paths.vrs_file, traj_paths.slam_root_dir, fps=30
    ).to(device=device)

    # Note the off-by-one for Ts_world_cpf, which we need for relative transform computation.
    config.traj_length = len(transforms.Ts_world_cpf) - config.start_index - 1
    Ts_world_cpf = (
        SE3(
            transforms.Ts_world_cpf[
                config.start_index : config.start_index + config.traj_length + 1
            ]
        )
        @ SE3.from_rotation(
            SO3.from_x_radians(
                transforms.Ts_world_cpf.new_tensor(config.glasses_x_angle_offset)
            )
        )
    ).parameters()
    pose_timestamps_sec = transforms.pose_timesteps[
        config.start_index + 1 : config.start_index + config.traj_length + 1
    ]
    Ts_world_device = transforms.Ts_world_device[
        config.start_index + 1 : config.start_index + config.traj_length + 1
    ]
    del transforms

    # Get temporally corresponded HaMeR detections.
    if traj_paths.hamer_outputs is not None:
        hamer_detections = CorrespondedHamerDetections.load(
            traj_paths.hamer_outputs,
            pose_timestamps_sec,
        ).to(device)
    else:
        print("No hand detections found.")
        hamer_detections = None

    # Get temporally corresponded Aria wrist and palm estimates.
    if traj_paths.wrist_and_palm_poses_csv is not None:
        aria_detections = CorrespondedAriaHandWristPoseDetections.load(
            traj_paths.wrist_and_palm_poses_csv,
            pose_timestamps_sec,
            Ts_world_device=Ts_world_device.numpy(force=True),
        ).to(device)
    else:
        print("No Aria hand detections found.")
        aria_detections = None

    print(f"{Ts_world_cpf.shape=}")

    server = None
    if config.visualize_traj:
        server = viser.ViserServer()
        server.gui.configure_theme(dark_mode=True)

    denoiser_network, train_config = load_denoiser(config.checkpoint_dir).to(device)
    body_model = fncsmpl.SmplhModel.load(config.smplh_npz_path).to(device)

    # traj = run_sampling_with_stitching(
    traj = real_time_sampling_with_stitching(
        denoiser_network,
        body_model=body_model,
        guidance_mode=config.guidance_mode,
        guidance_inner=config.guidance_inner,
        guidance_post=config.guidance_post,
        Ts_world_cpf=Ts_world_cpf,
        hamer_detections=hamer_detections,
        aria_detections=aria_detections,
        num_samples=config.num_samples,
        device=device,
        floor_z=floor_z,
    )

    # Save outputs in case we want to visualize later.
    if config.save_traj:
        save_name = (
            time.strftime("%Y%m%d-%H%M%S")
            + f"_start_{config.start_index}_end_{config.start_index + config.traj_length}_guidance_mode_{config.guidance_mode}_guidance_post_{config.guidance_post}"
        )
        out_path = config.output_dir / "egoallo_outputs" / (save_name + ".npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        assert not out_path.exists()
        (config.output_dir / "egoallo_outputs" / (save_name + "_args.yaml")).write_text(
            yaml.dump(dataclasses.asdict(config))
        )

        posed = traj.apply_to_body(body_model)
        Ts_world_root = fncsmpl_extensions.get_T_world_root_from_cpf_pose(
            posed, Ts_world_cpf[..., 1:, :]
        )
        print(f"Saving to {out_path}...", end="")
        np.savez(
            out_path,
            Ts_world_cpf=Ts_world_cpf[1:, :].numpy(force=True),
            Ts_world_root=Ts_world_root.numpy(force=True),
            body_quats=posed.local_quats[..., :21, :].numpy(force=True),
            left_hand_quats=posed.local_quats[..., 21:36, :].numpy(force=True),
            right_hand_quats=posed.local_quats[..., 36:51, :].numpy(force=True),
            contacts=traj.contacts.numpy(force=True),  # Sometimes we forgot this...
            betas=traj.betas.numpy(force=True),
            frame_nums=np.arange(config.start_index, config.start_index + config.traj_length),
            timestamps_ns=(np.array(pose_timestamps_sec) * 1e9).astype(np.int64),
        )
        print("saved!")

    # Visualize.
    if config.visualize_traj:
        assert server is not None
        loop_cb = visualize_traj_and_hand_detections(
            server,
            Ts_world_cpf[1:],
            traj,
            body_model,
            hamer_detections,
            aria_detections,
            points_data=points_data,
            splat_path=traj_paths.splat_path,
            floor_z=floor_z,
        )
        while True:
            loop_cb()


if __name__ == "__main__":
    import tyro
    ipdb_safety_net()

    main(tyro.cli(InferenceConfig))
