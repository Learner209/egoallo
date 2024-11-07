from __future__ import annotations

from typing import Optional
import time

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from tqdm.auto import tqdm

from . import fncsmpl, network
from .guidance_optimizer_jax import (
    GuidanceMode,
    do_guidance_optimization,
)
from .hand_detection_structs import (
    CorrespondedAriaHandWristPoseDetections,
    CorrespondedHamerDetections,
)
from .tensor_dataclass import TensorDataclass
from .transforms import SE3
from src.egoallo.setup_logger import setup_logger
from .motion_diffusion_pipeline import MotionDiffusionPipeline
from .data.dataclass import EgoTrainingData

logger = setup_logger(output=None, name=__name__)

def run_sampling_with_stitching(
    pipeline: MotionDiffusionPipeline,
    body_model: fncsmpl.SmplhModel,
    guidance_mode: GuidanceMode,
    guidance_post: bool,
    guidance_inner: bool,
    Ts_world_cpf: Float[Tensor, "time 7"],
    floor_z: float,
    hamer_detections: None | CorrespondedHamerDetections,
    aria_detections: None | CorrespondedAriaHandWristPoseDetections,
    num_samples: int,
    device: torch.device,
) -> network.EgoDenoiseTraj:
    """Run sampling with stitching using the diffusion pipeline."""
    # Offset the T_world_cpf transform to place the floor at z=0
    Ts_world_cpf_shifted = Ts_world_cpf.clone()
    Ts_world_cpf_shifted[..., 6] -= floor_z

    T_cpf_tm1_cpf_t = (
        SE3(Ts_world_cpf[..., :-1, :]).inverse() @ SE3(Ts_world_cpf[..., 1:, :])
    ).wxyz_xyz.to(device)  # Ensure on correct device

    seq_len = Ts_world_cpf.shape[0] - 1
    window_size = 128
    overlap_size = 32

    start_time = time.time()
    x_0_packed_pred = torch.zeros(
        (num_samples, seq_len, pipeline.unet.config.d_state),
        device=device
    )
    overlap_weights = torch.zeros((1, seq_len, 1), device=device)

    # Process windows
    for start_t in tqdm(range(0, seq_len, window_size - overlap_size)):
        end_t = min(start_t + window_size, seq_len)
        window_len = end_t - start_t
        # Prepare training data for this window
        train_batch = EgoTrainingData(
            T_world_root=torch.zeros((num_samples, window_len, 7), device=device),  # Placeholder
            contacts=torch.zeros((num_samples, window_len, 21), device=device),  # Placeholder 
            betas=torch.zeros((num_samples, 1, 16), device=device),  # Placeholder
            body_quats=torch.zeros((num_samples, window_len, 21, 4), device=device),  # Placeholder
            T_cpf_tm1_cpf_t=T_cpf_tm1_cpf_t[None, start_t:end_t, :].repeat(num_samples, 1, 1),
            T_world_cpf=Ts_world_cpf_shifted[None, start_t + 1:end_t + 1, :].repeat(num_samples, 1, 1),
            height_from_floor=Ts_world_cpf_shifted[None, start_t + 1:end_t + 1, 6:7].repeat(num_samples, 1, 1),
            joints_wrt_cpf=torch.zeros((num_samples, window_len, 21, 3), device=device),  # Placeholder
            mask=torch.ones((num_samples, window_len), dtype=torch.bool, device=device),
            hand_quats=None if not pipeline.unet.config.include_hands else torch.zeros((num_samples, window_len, 30, 4), device=device),
            prev_window=None
        )

        # Run pipeline for this window
        output = pipeline(
            batch_size=num_samples,
            num_inference_steps=50,  # Or configure as needed
            train_batch=train_batch,
            return_intermediates=guidance_inner,
        )

        # Handle guidance if enabled
        if guidance_mode != "off" and guidance_inner and output.intermediate_states is not None:
            for intermediate in output.intermediate_states:
                intermediate, _ = do_guidance_optimization(
                    Ts_world_cpf=Ts_world_cpf[start_t + 1:end_t + 1, :],
                    traj=intermediate,
                    body_model=body_model,
                    guidance_mode=guidance_mode,
                    phase="inner",
                    hamer_detections=hamer_detections,
                    aria_detections=aria_detections,
                )

        # Calculate overlap weights
        overlap_weights_slice = torch.minimum(
            torch.tensor(overlap_size, device=device),
            torch.minimum(
                torch.arange(1, window_len + 1, device=device),
                torch.arange(window_len, 0, -1, device=device),
            ),
        )[None, :, None] / overlap_size

        # Accumulate results
        overlap_weights[:, start_t:end_t, :] += overlap_weights_slice
        x_0_packed_pred[:, start_t:end_t, :] += output.motion.pack() * overlap_weights_slice

    # Average overlapping regions
    x_0_packed_pred = x_0_packed_pred / (overlap_weights + 1e-8)

    duration = time.time() - start_time
    logger.info(
        f"RUNTIME: {duration:.6f}, SEQ_LEN: {seq_len}, FPS: {seq_len / duration:.2f}"
    )

    # Final trajectory
    final_traj = network.EgoDenoiseTraj.unpack(
        x_0_packed_pred,
        include_hands=pipeline.unet.config.include_hands,
        should_project_rot6d=False,
    )

    # Post-guidance if enabled
    if guidance_mode != "off" and guidance_post:
        final_traj, _ = do_guidance_optimization(
            Ts_world_cpf=Ts_world_cpf[1:, :],
            traj=final_traj,
            body_model=body_model,
            guidance_mode=guidance_mode,
            phase="post",
            hamer_detections=hamer_detections,
            aria_detections=aria_detections,
        )

    return final_traj

def real_time_sampling_with_stitching(
    pipeline: MotionDiffusionPipeline,
    body_model: fncsmpl.SmplhModel,
    guidance_mode: GuidanceMode,
    guidance_post: bool,
    guidance_inner: bool,
    Ts_world_cpf: Float[Tensor, "time 7"],
    floor_z: float,
    hamer_detections: None | CorrespondedHamerDetections,
    aria_detections: None | CorrespondedAriaHandWristPoseDetections,
    num_samples: int,
    device: torch.device,
) -> network.EgoDenoiseTraj:
    """Real-time sampling with stitching using the diffusion pipeline."""
    # Offset the T_world_cpf transform
    Ts_world_cpf_shifted = Ts_world_cpf.clone()
    Ts_world_cpf_shifted[..., 6] -= floor_z

    T_cpf_tm1_cpf_t = (
        SE3(Ts_world_cpf[..., :-1, :]).inverse() @ SE3(Ts_world_cpf[..., 1:, :])
    ).wxyz_xyz.to(device)  # Ensure on correct device

    seq_len = Ts_world_cpf.shape[0] - 1
    window_size = 128
    overlap_size = 64

    start_time = time.time()
    x_0_packed_pred = torch.zeros(
        (num_samples, seq_len, pipeline.unet.config.d_state),
        device=device
    )

    prev_window_motion: Optional[EgoTrainingData] = None

    # Process windows sequentially
    for start_t in tqdm(range(0, seq_len, window_size - overlap_size)):
        end_t = min(start_t + window_size, seq_len)
        window_len = end_t - start_t

        # Prepare training batch for this window
        train_batch = EgoTrainingData(
            T_world_root=torch.zeros((num_samples, window_len, 7), device=device),  # Placeholder
            contacts=torch.zeros((num_samples, window_len, 21), device=device),  # Placeholder
            betas=torch.zeros((num_samples, 1, 16), device=device),  # Placeholder
            body_quats=torch.zeros((num_samples, window_len, 21, 4), device=device),  # Placeholder
            T_cpf_tm1_cpf_t=T_cpf_tm1_cpf_t[None, start_t:end_t, :].repeat(
                (num_samples, 1, 1)
            ),  # Ensure on correct device
            T_world_cpf=Ts_world_cpf_shifted[None, start_t + 1:end_t + 1, :].repeat(
                (num_samples, 1, 1)
            ),  # Ensure on correct device
            height_from_floor=Ts_world_cpf_shifted[None, start_t + 1:end_t + 1, 6:7].repeat(
                (num_samples, 1, 1)
            ),  # Ensure on correct device
            joints_wrt_cpf=torch.zeros((num_samples, window_len, 21, 3), device=device),  # Placeholder
            mask=torch.ones((num_samples, window_len), dtype=torch.bool, device=device),
            hand_quats=None,  # Optional field
            prev_window=prev_window_motion if prev_window_motion is not None else None
        ).to(device)

        # Run pipeline for this window
        output = pipeline(
            batch_size=num_samples,
            num_inference_steps=50,  # Or configure as needed
            train_batch=train_batch,
            return_intermediates=False,
        )

        window_motion = output.motion

        # Store for next window's conditioning
        prev_window_motion = train_batch

        # Accumulate results with overlap handling
        if start_t > 0:
            # Blend the overlapping region
            blend_start = start_t
            blend_end = start_t + overlap_size
            alpha = torch.linspace(0, 1, overlap_size, device=device)
            alpha = alpha.view(1, -1, 1)
            
            x_0_packed_pred[:, blend_start:blend_end, :] = (
                (1 - alpha) * x_0_packed_pred[:, blend_start:blend_end, :]
                + alpha * window_motion.pack()[:, :overlap_size, :]
            )
            # Copy the non-overlapping part
            x_0_packed_pred[:, blend_end:end_t, :] = window_motion.pack()[:, overlap_size:, :]
        else:
            x_0_packed_pred[:, start_t:end_t, :] = window_motion.pack()

    duration = time.time() - start_time
    logger.info(
        f"RUNTIME: {duration:.6f}, SEQ_LEN: {seq_len}, FPS: {seq_len / duration:.2f}"
    )

    # Final trajectory
    final_traj = network.EgoDenoiseTraj.unpack(
        x_0_packed_pred,
        include_hands=pipeline.unet.config.include_hands,
        should_project_rot6d=False,
    )

    # Post-guidance if enabled
    if guidance_mode != "off" and guidance_post:
        final_traj, _ = do_guidance_optimization(
            Ts_world_cpf=Ts_world_cpf[1:, :].to(device),  # Ensure on correct device
            traj=final_traj,
            body_model=body_model,
            guidance_mode=guidance_mode,
            phase="post",
            hamer_detections=hamer_detections,
            aria_detections=aria_detections,
        )
    return final_traj
