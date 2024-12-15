from __future__ import annotations

import time

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

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
from .transforms import SE3, SO3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data.amass import EgoTrainingData


def quadratic_ts(timesteps: int) -> np.ndarray:
    """DDIM sampling schedule."""
    end_step = 0
    start_step = timesteps
    x = np.arange(end_step, int(np.sqrt(start_step))) ** 2
    x[-1] = start_step
    return x[::-1]


def linear_ts(timesteps: int) -> np.ndarray:
    """
    DDPM sampling schedule using linear timesteps.
    Returns evenly spaced timesteps from `timesteps` to 0 in descending order.
    """
    start_step = timesteps
    end_step = 0
    return np.arange(start_step, end_step - 1, -1)


class CosineNoiseScheduleConstants(TensorDataclass):
    alpha_t: Float[Tensor, "T"]
    r"""$1 - \beta_t$"""

    alpha_bar_t: Float[Tensor, "T+1"]
    r"""$\Prod_{j=1}^t (1 - \beta_j)$"""

    @staticmethod
    def compute(timesteps: int, s: float = 0.008) -> CosineNoiseScheduleConstants:
        steps = timesteps + 1
        x = torch.linspace(0, 1, steps, dtype=torch.float64)

        def get_betas():
            alphas_cumprod = torch.cos((x + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0, 0.999)

        alpha_t = 1.0 - get_betas()
        assert len(alpha_t.shape) == 1
        alpha_cumprod_t = torch.cat(
            [torch.ones((1,)), torch.cumprod(alpha_t, dim=0)],
            dim=0,
        )
        return CosineNoiseScheduleConstants(
            alpha_t=alpha_t, alpha_bar_t=alpha_cumprod_t
        )


def run_sampling_with_stitching(
    denoiser_network: network.EgoDenoiser,
    body_model: fncsmpl.SmplhModel,
    guidance_mode: GuidanceMode,
    guidance_post: bool,
    guidance_inner: bool,
    Ts_world_cpf: Float[Tensor, "time 7"],
    hamer_detections: None | CorrespondedHamerDetections,
    aria_detections: None | CorrespondedAriaHandWristPoseDetections,
    num_samples: int,
    device: torch.device,
) -> network.EgoDenoiseTraj:
    noise_constants = CosineNoiseScheduleConstants.compute(timesteps=1000).to(
        device=device
    )
    alpha_bar_t = noise_constants.alpha_bar_t
    alpha_t = noise_constants.alpha_t

    x_t_packed = torch.randn(
        (num_samples, Ts_world_cpf.shape[0] - 1, denoiser_network.get_d_state()),
        device=device,
    )
    x_t = network.EgoDenoiseTraj.unpack(
        x_t_packed, include_hands=denoiser_network.config.include_hands
    )
    x_t.T_world_root = Ts_world_cpf[1:]  # Use the transformed poses
    x_t_list = [x_t]
    ts = quadratic_ts(timesteps=1000)

    seq_len = x_t_packed.shape[1]

    start_time = None

    window_size = 128
    overlap_size = 32
    canonical_overlap_weights = (
        torch.from_numpy(
            np.minimum(
                # Make this shape /```\
                overlap_size,
                np.minimum(
                    # Make this shape: /
                    np.arange(1, seq_len + 1),
                    # Make this shape: \
                    np.arange(1, seq_len + 1)[::-1],
                ),
            )
            / overlap_size,
        )
        .to(device)
        .to(torch.float32)
    )
    for i in range(len(ts) - 1):
        print(f"Sampling {i}/{len(ts) - 1}")
        t = ts[i]
        t_next = ts[i + 1]

        with torch.inference_mode():
            # Chop everything into windows.
            x_0_packed_pred = torch.zeros_like(x_t_packed)
            overlap_weights = torch.zeros((1, seq_len, 1), device=x_t_packed.device)

            # Denoise each window.
            for start_t in range(0, seq_len, window_size - overlap_size):
                end_t = min(start_t + window_size, seq_len)
                assert end_t - start_t > 0
                overlap_weights_slice = canonical_overlap_weights[
                    None, : end_t - start_t, None
                ]
                overlap_weights[:, start_t:end_t, :] += overlap_weights_slice
                x_0_packed_pred[:, start_t:end_t, :] += (
                    denoiser_network.forward(
                        x_t_packed[:, start_t:end_t, :],
                        torch.tensor([t], device=device).expand((num_samples,)),
                        T_world_cpf=Ts_world_cpf[
                            None, start_t + 1 : end_t + 1, :
                        ].repeat((num_samples, 1, 1)),
                        project_output_rotmats=False,
                        hand_positions_wrt_cpf=None,  # TODO: this should be filled in!!
                        mask=None,
                    )
                    * overlap_weights_slice
                )

            # Take the mean for overlapping regions.
            x_0_packed_pred /= overlap_weights

            x_0_packed_pred = network.EgoDenoiseTraj.unpack(
                x_0_packed_pred,
                include_hands=denoiser_network.config.include_hands,
                project_rotmats=True,
            ).pack()

        if torch.any(torch.isnan(x_0_packed_pred)):
            print("found nan", i)
        sigma_t = torch.cat(
            [
                torch.zeros((1,), device=device),
                torch.sqrt(
                    (1.0 - alpha_bar_t[:-1]) / (1 - alpha_bar_t[1:]) * (1 - alpha_t)
                )
                * 0.8,
            ]
        )

        if guidance_mode != "off" and guidance_inner:
            x_0_pred, _ = do_guidance_optimization(
                # It's important that we _don't_ use the shifted transforms here.
                Ts_world_cpf=Ts_world_cpf[1:, :],
                traj=network.EgoDenoiseTraj.unpack(
                    x_0_packed_pred, include_hands=denoiser_network.config.include_hands
                ),
                body_model=body_model,
                guidance_mode=guidance_mode,
                phase="inner",
                hamer_detections=hamer_detections,
                aria_detections=aria_detections,
            )
            x_0_packed_pred = x_0_pred.pack()
            del x_0_pred

        if start_time is None:
            start_time = time.time()

        # print(sigma_t)
        x_t_packed = (
            torch.sqrt(alpha_bar_t[t_next]) * x_0_packed_pred
            + (
                torch.sqrt(1 - alpha_bar_t[t_next] - sigma_t[t] ** 2)
                * (x_t_packed - torch.sqrt(alpha_bar_t[t]) * x_0_packed_pred)
                / torch.sqrt(1 - alpha_bar_t[t] + 1e-1)
            )
            + sigma_t[t] * torch.randn(x_0_packed_pred.shape, device=device)
        )
        x_t_list.append(
            network.EgoDenoiseTraj.unpack(
                x_t_packed, include_hands=denoiser_network.config.include_hands
            )
        )

    if guidance_mode != "off" and guidance_post:
        constrained_traj = x_t_list[-1]
        constrained_traj, _ = do_guidance_optimization(
            # It's important that we _don't_ use the shifted transforms here.
            Ts_world_cpf=Ts_world_cpf[1:, :],
            traj=constrained_traj,
            body_model=body_model,
            guidance_mode=guidance_mode,
            phase="post",
            hamer_detections=hamer_detections,
            aria_detections=aria_detections,
        )
        assert start_time is not None
        print("RUNTIME (exclude first optimization)", time.time() - start_time)
        return constrained_traj
    else:
        assert start_time is not None
        print("RUNTIME (exclude first optimization)", time.time() - start_time)
        return x_t_list[-1]


def run_sampling_with_masked_data(
    denoiser_network: network.EgoDenoiser,
    body_model: fncsmpl.SmplhModel,
    masked_data: EgoTrainingData,
    guidance_mode: GuidanceMode,
    guidance_post: bool,
    guidance_inner: bool,
    floor_z: float,
    hamer_detections: None | CorrespondedHamerDetections,
    aria_detections: None | CorrespondedAriaHandWristPoseDetections,
    num_samples: int,
    device: torch.device,
) -> network.EgoDenoiseTraj:
    # FIXME: currently the batch-size dimension of `masked_data` is not supported, as the num_samples `param` would conflict with batch_size dim of `masked_data`.
    noise_constants = CosineNoiseScheduleConstants.compute(timesteps=1000).to(
        device=device
    )
    alpha_bar_t = noise_constants.alpha_bar_t
    alpha_t = noise_constants.alpha_t

    x_t_packed = torch.randn(
        (
            num_samples,
            masked_data.joints_wrt_world.shape[1],
            denoiser_network.get_d_state(),
        ),
        device=device,
    )
    x_t_list = [
        network.EgoDenoiseTraj.unpack(
            x_t_packed, include_hands=denoiser_network.config.include_hands
        )
    ]
    ts = quadratic_ts(timesteps=1000)

    seq_len = x_t_packed.shape[1]
    window_size = 128
    overlap_size = 32

    canonical_overlap_weights = (
        torch.from_numpy(
            np.minimum(
                overlap_size,
                np.minimum(
                    np.arange(1, seq_len + 1),
                    np.arange(1, seq_len + 1)[::-1],
                ),
            )
            / overlap_size,
        )
        .to(device)
        .to(torch.float32)
    )

    # breakpoint()
    for i in range(len(ts) - 1):
        t = ts[i]
        t_next = ts[i + 1]

        with torch.inference_mode():
            x_0_packed_pred = torch.zeros_like(x_t_packed)
            overlap_weights = torch.zeros((1, seq_len, 1), device=x_t_packed.device)

            for start_t in range(0, seq_len, window_size - overlap_size):
                end_t = min(start_t + window_size, seq_len)
                overlap_weights_slice = canonical_overlap_weights[
                    None, : end_t - start_t, None
                ]
                overlap_weights[:, start_t:end_t, :] += overlap_weights_slice

                x_0_packed_pred[:, start_t:end_t, :] += (
                    denoiser_network.forward(
                        x_t_packed=x_t_packed[:, start_t:end_t, :],
                        t=torch.tensor([t], device=device).expand((num_samples,)),
                        joints=masked_data.joints_wrt_world[:, start_t:end_t, :],
                        visible_joints_mask=masked_data.visible_joints_mask[
                            :, start_t:end_t, :
                        ],
                        project_output_rotmats=True,
                        mask=masked_data.mask[:, start_t:end_t],
                    )
                    * overlap_weights_slice
                )

            x_0_packed_pred /= overlap_weights

            x_0_pred = network.EgoDenoiseTraj.unpack(
                x_0_packed_pred,
                include_hands=denoiser_network.config.include_hands,
                project_rotmats=True,
            )

        if guidance_mode != "off" and guidance_inner:
            x_0_pred, _ = do_guidance_optimization(
                T_world_root=SE3.from_rotation_and_translation(
                    SO3.from_matrix(x_0_pred.R_world_root), x_0_pred.t_world_root
                )
                .parameters()
                .squeeze(0),
                traj=x_0_pred,
                body_model=body_model,
                guidance_mode=guidance_mode,
                phase="inner",
                hamer_detections=hamer_detections,
                aria_detections=aria_detections,
            )
        x_0_packed_pred = x_0_pred.pack()

        if torch.any(torch.isnan(x_0_packed_pred)):
            print("found nan", i)
        sigma_t = torch.cat(
            [
                torch.zeros((1,), device=device),
                torch.sqrt(
                    (1.0 - alpha_bar_t[:-1]) / (1 - alpha_bar_t[1:]) * (1 - alpha_t)
                )
                * 0.8,
            ]
        )
        x_t_packed = (
            torch.sqrt(alpha_bar_t[t_next]) * x_0_packed_pred
            + (
                torch.sqrt(1 - alpha_bar_t[t_next] - sigma_t[t] ** 2)
                * (x_t_packed - torch.sqrt(alpha_bar_t[t]) * x_0_packed_pred)
                / torch.sqrt(1 - alpha_bar_t[t] + 1e-1)
            )
            + sigma_t[t] * torch.randn_like(x_0_packed_pred)
        )
        x_t_list.append(
            network.EgoDenoiseTraj.unpack(
                x_t_packed,
                include_hands=denoiser_network.config.include_hands,
                project_rotmats=True,
            )
        )

    if guidance_mode != "off" and guidance_post:
        constrained_traj = x_t_list[-1]
        constrained_traj, _ = do_guidance_optimization(
            T_world_root=SE3.from_rotation_and_translation(
                SO3.from_matrix(constrained_traj.R_world_root),
                constrained_traj.t_world_root,
            )
            .parameters()
            .squeeze(0),
            traj=constrained_traj,
            body_model=body_model,
            guidance_mode=guidance_mode,
            phase="post",
            hamer_detections=hamer_detections,
            aria_detections=aria_detections,
        )
        return constrained_traj
    else:
        return x_t_list[-1]


def run_sampling_with_masked_data_hard_coded_sliding_window(
    denoiser_network: network.EgoDenoiser,
    body_model: fncsmpl.SmplhModel,
    masked_data: EgoTrainingData,
    guidance_mode: GuidanceMode,
    guidance_post: bool,
    guidance_inner: bool,
    floor_z: float,
    hamer_detections: None | CorrespondedHamerDetections,
    aria_detections: None | CorrespondedAriaHandWristPoseDetections,
    num_samples: int,
    device: torch.device,
) -> network.EgoDenoiseTraj:
    # Initialize noise schedule
    noise_constants = CosineNoiseScheduleConstants.compute(timesteps=1000).to(
        device=device
    )
    alpha_bar_t = noise_constants.alpha_bar_t
    alpha_t = noise_constants.alpha_t

    # Initialize random noise
    x_t_packed = torch.randn(
        (
            num_samples,
            masked_data.joints_wrt_world.shape[1],
            denoiser_network.get_d_state(),
        ),
        device=device,
    )
    x_t_list = [
        network.EgoDenoiseTraj.unpack(
            x_t_packed, include_hands=denoiser_network.config.include_hands
        )
    ]
    ts = quadratic_ts(timesteps=1000)

    # Set fixed window size without overlap
    window_size = 128
    seq_len = x_t_packed.shape[1]

    for i in range(len(ts) - 1):
        t = ts[i]
        t_next = ts[i + 1]

        with torch.inference_mode():
            x_0_packed_pred = torch.zeros_like(x_t_packed)

            # Process each window without overlap
            for start_t in range(0, seq_len, window_size):
                end_t = min(start_t + window_size, seq_len)

                # Simple window processing with weight=1
                x_0_packed_pred[:, start_t:end_t, :] = denoiser_network.forward(
                    x_t_packed=x_t_packed[:, start_t:end_t, :],
                    t=torch.tensor([t], device=device).expand((num_samples,)),
                    joints=masked_data.joints_wrt_world[:, start_t:end_t, :],
                    visible_joints_mask=masked_data.visible_joints_mask[
                        :, start_t:end_t, :
                    ],
                    project_output_rotmats=False,
                    mask=masked_data.mask[:, start_t:end_t],
                )

            x_0_pred = network.EgoDenoiseTraj.unpack(
                x_0_packed_pred,
                include_hands=denoiser_network.config.include_hands,
                project_rotmats=True,
            )

        # Apply guidance optimization if enabled
        if guidance_mode != "off" and guidance_inner:
            x_0_pred, _ = do_guidance_optimization(
                T_world_root=SE3.from_rotation_and_translation(
                    SO3.from_matrix(x_0_pred.R_world_root), x_0_pred.t_world_root
                )
                .parameters()
                .squeeze(0),
                traj=x_0_pred,
                body_model=body_model,
                guidance_mode=guidance_mode,
                phase="inner",
                hamer_detections=hamer_detections,
                aria_detections=aria_detections,
            )
        x_0_packed_pred = x_0_pred.pack()

        # Check for NaN values
        if torch.any(torch.isnan(x_0_packed_pred)):
            print("found nan", i)

        # Compute sigma for noise schedule
        sigma_t = torch.cat(
            [
                torch.zeros((1,), device=device),
                torch.sqrt(
                    (1.0 - alpha_bar_t[:-1]) / (1 - alpha_bar_t[1:]) * (1 - alpha_t)
                )
                * 0.8,
            ]
        )

        # Update x_t
        x_t_packed = (
            torch.sqrt(alpha_bar_t[t_next]) * x_0_packed_pred
            + (
                torch.sqrt(1 - alpha_bar_t[t_next] - sigma_t[t] ** 2)
                * (x_t_packed - torch.sqrt(alpha_bar_t[t]) * x_0_packed_pred)
                / torch.sqrt(1 - alpha_bar_t[t] + 1e-1)
            )
            + sigma_t[t] * torch.randn_like(x_0_packed_pred)
        )
        x_t_list.append(
            network.EgoDenoiseTraj.unpack(
                x_t_packed,
                include_hands=denoiser_network.config.include_hands,
                project_rotmats=True,
            )
        )

    # Apply post-guidance optimization if enabled
    if guidance_mode != "off" and guidance_post:
        constrained_traj = x_t_list[-1]
        constrained_traj, _ = do_guidance_optimization(
            T_world_root=SE3.from_rotation_and_translation(
                SO3.from_matrix(constrained_traj.R_world_root),
                constrained_traj.t_world_root,
            )
            .parameters()
            .squeeze(0),
            traj=constrained_traj,
            body_model=body_model,
            guidance_mode=guidance_mode,
            phase="post",
            hamer_detections=hamer_detections,
            aria_detections=aria_detections,
        )
        return constrained_traj
    else:
        return x_t_list[-1]


# Implementation of DDPM sampling
def run_sampling_with_masked_data_ddpm(
    denoiser_network: network.EgoDenoiser,
    body_model: fncsmpl.SmplhModel,
    masked_data: EgoTrainingData,
    guidance_mode: GuidanceMode,
    guidance_post: bool,
    guidance_inner: bool,
    floor_z: float,
    hamer_detections: None | CorrespondedHamerDetections,
    aria_detections: None | CorrespondedAriaHandWristPoseDetections,
    num_samples: int,
    device: torch.device,
) -> network.EgoDenoiseTraj:
    """
    DDPM sampling version of run_sampling_with_masked_data.
    Uses full stochastic sampling instead of DDIM's deterministic sampling.
    """
    # Initialize noise schedule - same as DDIM
    noise_constants = CosineNoiseScheduleConstants.compute(timesteps=1000).to(
        device=device
    )
    alpha_bar_t = noise_constants.alpha_bar_t
    alpha_t = noise_constants.alpha_t

    # Initialize random noise
    x_t_packed = torch.randn(
        (
            num_samples,
            masked_data.joints_wrt_world.shape[1],
            denoiser_network.get_d_state(),
        ),
        device=device,
    )
    x_t_list = [
        network.EgoDenoiseTraj.unpack(
            x_t_packed, include_hands=denoiser_network.config.include_hands
        )
    ]
    ts = linear_ts(timesteps=1000)

    seq_len = x_t_packed.shape[1]
    window_size = 128
    overlap_size = 32

    canonical_overlap_weights = (
        torch.from_numpy(
            np.minimum(
                overlap_size,
                np.minimum(
                    np.arange(1, seq_len + 1),
                    np.arange(1, seq_len + 1)[::-1],
                ),
            )
            / overlap_size,
        )
        .to(device)
        .to(torch.float32)
    )

    # breakpoint()
    for i in range(len(ts) - 1):
        t = ts[i]
        t_next = ts[i + 1]

        with torch.inference_mode():
            x_0_packed_pred = torch.zeros_like(x_t_packed)
            overlap_weights = torch.zeros((1, seq_len, 1), device=x_t_packed.device)

            # Process windows with overlap
            for start_t in range(0, seq_len, window_size - overlap_size):
                end_t = min(start_t + window_size, seq_len)
                overlap_weights_slice = canonical_overlap_weights[
                    None, : end_t - start_t, None
                ]
                overlap_weights[:, start_t:end_t, :] += overlap_weights_slice

                x_0_packed_pred[:, start_t:end_t, :] += (
                    denoiser_network.forward(
                        x_t_packed=x_t_packed[:, start_t:end_t, :],
                        t=torch.tensor([t], device=device).expand((num_samples,)),
                        joints=masked_data.joints_wrt_world[:, start_t:end_t, :],
                        visible_joints_mask=masked_data.visible_joints_mask[
                            :, start_t:end_t, :
                        ],
                        project_output_rotmats=False,
                        mask=masked_data.mask[:, start_t:end_t],
                    )
                    * overlap_weights_slice
                )

            x_0_packed_pred /= overlap_weights

            x_0_pred = network.EgoDenoiseTraj.unpack(
                x_0_packed_pred,
                include_hands=denoiser_network.config.include_hands,
                project_rotmats=True,
            )

        if guidance_mode != "off" and guidance_inner:
            x_0_pred, _ = do_guidance_optimization(
                T_world_root=SE3.from_rotation_and_translation(
                    SO3.from_matrix(x_0_pred.R_world_root), x_0_pred.t_world_root
                )
                .parameters()
                .squeeze(0),
                traj=x_0_pred,
                body_model=body_model,
                guidance_mode=guidance_mode,
                phase="inner",
                hamer_detections=hamer_detections,
                aria_detections=aria_detections,
            )
        x_0_packed_pred = x_0_pred.pack()

        if torch.any(torch.isnan(x_0_packed_pred)):
            print("found nan", i)

        # DDPM update equation
        # x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise
        noise = torch.randn_like(x_0_packed_pred)
        x_t_packed = (
            torch.sqrt(alpha_t[t_next]) * x_0_packed_pred
            + torch.sqrt(1 - alpha_t[t_next]) * noise
        )

        x_t_list.append(
            network.EgoDenoiseTraj.unpack(
                x_t_packed,
                include_hands=denoiser_network.config.include_hands,
                project_rotmats=True,
            )
        )

    if guidance_mode != "off" and guidance_post:
        constrained_traj = x_t_list[-1]
        constrained_traj, _ = do_guidance_optimization(
            T_world_root=SE3.from_rotation_and_translation(
                SO3.from_matrix(constrained_traj.R_world_root),
                constrained_traj.t_world_root,
            )
            .parameters()
            .squeeze(0),
            traj=constrained_traj,
            body_model=body_model,
            guidance_mode=guidance_mode,
            phase="post",
            hamer_detections=hamer_detections,
            aria_detections=aria_detections,
        )
        return constrained_traj
    else:
        return x_t_list[-1]
