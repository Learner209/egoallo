from __future__ import annotations

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

logger = setup_logger(output=None, name=__name__)

def quadratic_ts() -> np.ndarray:
    """DDIM sampling schedule."""
    end_step = 0
    start_step = 1000
    x = np.arange(end_step, int(np.sqrt(start_step))) ** 2
    x[-1] = start_step
    return x[::-1]


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
    floor_z: float,
    hamer_detections: None | CorrespondedHamerDetections,
    aria_detections: None | CorrespondedAriaHandWristPoseDetections,
    num_samples: int,
    device: torch.device,
) -> network.EgoDenoiseTraj:
    """
    Run the sampling process with stitching, updated to use rot6d representations.
    """
    # Offset the T_world_cpf transform to place the floor at z=0 for the denoiser network.
    # All of the network outputs are local, so we don't need to un-offset when returning.
    Ts_world_cpf_shifted = Ts_world_cpf.clone()
    Ts_world_cpf_shifted[..., 6] -= floor_z

    noise_constants = CosineNoiseScheduleConstants.compute(timesteps=1000).to(
        device=device
    )
    alpha_bar_t = noise_constants.alpha_bar_t
    alpha_t = noise_constants.alpha_t

    T_cpf_tm1_cpf_t = (
        SE3(Ts_world_cpf[..., :-1, :]).inverse() @ SE3(Ts_world_cpf[..., 1:, :])
    ).wxyz_xyz

    x_t_packed = torch.randn(
        (num_samples, Ts_world_cpf.shape[0] - 1, denoiser_network.get_d_state()),
        device=device,
    )
    x_t_list = [
        network.EgoDenoiseTraj.unpack(
            x_t_packed, include_hands=denoiser_network.config.include_hands
        )
    ]
    ts = quadratic_ts()

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

    start_time = time.time()
    for i in tqdm(range(len(ts) - 1)):
        logger.info(f"Sampling {i}/{len(ts) - 1}, Overall size: {seq_len}")
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
                    start_t:end_t
                ][None, :, None]
                overlap_weights[:, start_t:end_t, :] += overlap_weights_slice

                x_t_packed_slice = x_t_packed[:, start_t:end_t, :]
                T_cpf_tm1_cpf_t_slice = T_cpf_tm1_cpf_t[None, start_t:end_t, :].repeat(
                    (num_samples, 1, 1)
                )
                T_world_cpf_slice = Ts_world_cpf_shifted[
                    None, start_t + 1 : end_t + 1, :
                ].repeat((num_samples, 1, 1))

                x_0_packed_pred_slice = denoiser_network.forward(
                    x_t_packed_slice,
                    torch.full((num_samples,), t, device=device),
                    T_cpf_tm1_cpf_t=T_cpf_tm1_cpf_t_slice,
                    T_world_cpf=T_world_cpf_slice,
                    project_output_rot6d=False,
                    hand_positions_wrt_cpf=None,
                    mask=None,
                ) * overlap_weights_slice

                x_0_packed_pred[:, start_t:end_t, :] += x_0_packed_pred_slice

            # Take the mean for overlapping regions.
            x_0_packed_pred /= overlap_weights

            x_0_packed_pred = network.EgoDenoiseTraj.unpack(
                x_0_packed_pred,
                include_hands=denoiser_network.config.include_hands,
                project_rot6d=True,
            ).pack()

        if torch.any(torch.isnan(x_0_packed_pred)):
            logger.warning(f"Found NaN at iteration {i}")

        sigma_t = torch.cat(
            [
                torch.zeros((1,), device=device),
                torch.sqrt(
                    (1.0 - alpha_bar_t[:-1]) / (1 - alpha_bar_t[1:]) * (1 - alpha_t)
                )
                * 0.8,
            ]
        ).to(device)

        if guidance_mode != "off" and guidance_inner:
            x_0_pred = network.EgoDenoiseTraj.unpack(
                x_0_packed_pred, include_hands=denoiser_network.config.include_hands
            )
            x_0_pred, _ = do_guidance_optimization(
                Ts_world_cpf=Ts_world_cpf[1:, :],
                traj=x_0_pred,
                body_model=body_model,
                guidance_mode=guidance_mode,
                phase="inner",
                hamer_detections=hamer_detections,
                aria_detections=aria_detections,
            )
            x_0_packed_pred = x_0_pred.pack()
            del x_0_pred

        # Update x_t_packed for next iteration
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
    duration = time.time() - start_time
    logger.info(
        f"RUNTIME: {duration:.6f}, SEQ_LEN: {seq_len}, FPS: {seq_len / duration:.2f}"
    )

    if guidance_mode != "off" and guidance_post:
        constrained_traj = x_t_list[-1]
        constrained_traj, _ = do_guidance_optimization(
            Ts_world_cpf=Ts_world_cpf[1:, :],
            traj=constrained_traj,
            body_model=body_model,
            guidance_mode=guidance_mode,
            phase="post",
            hamer_detections=hamer_detections,
            aria_detections=aria_detections,
        )
        logger.info(
            f"RUNTIME (exclude first optimization): {time.time() - start_time:.6f}"
        )
        return constrained_traj
    else:
        logger.info(
            f"RUNTIME (exclude first optimization): {time.time() - start_time:.6f}"
        )
        return x_t_list[-1]


def real_time_sampling_with_stitching(
    denoiser_network: network.EgoDenoiser,
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
    """
    Perform real-time sampling with stitching, updated to use rot6d representations.
    """
    # Offset the T_world_cpf transform to place the floor at z=0 for the denoiser network.
    Ts_world_cpf_shifted = Ts_world_cpf.clone()
    Ts_world_cpf_shifted[..., 6] -= floor_z

    noise_constants = CosineNoiseScheduleConstants.compute(timesteps=1000).to(
        device=device
    )
    alpha_bar_t = noise_constants.alpha_bar_t
    alpha_t = noise_constants.alpha_t

    T_cpf_tm1_cpf_t = (
        SE3(Ts_world_cpf[..., :-1, :]).inverse() @ SE3(Ts_world_cpf[..., 1:, :])
    ).wxyz_xyz

    seq_len = Ts_world_cpf.shape[0] - 1

    window_size = 128
    overlap_size = 64
    canonical_overlap_weights = (
        torch.from_numpy(
            np.minimum(
                overlap_size,
                np.minimum(np.arange(1, seq_len + 1), np.arange(1, seq_len + 1)[::-1]),
            )
            / overlap_size
        )
        .to(device)
        .float()
    )

    start_time = time.time()
    x_0_packed_pred = torch.zeros(
        (num_samples, seq_len, denoiser_network.get_d_state()), device=device
    )
    sigma_t = torch.cat(
        [
            torch.zeros((1,), device=device),
            torch.sqrt(
                (1.0 - alpha_bar_t[:-1]) / (1 - alpha_bar_t[1:]) * (1 - alpha_t)
            )
            * 0.8,
        ]
    ).to(device)

    with torch.inference_mode():
        prev_window_x_t = None  # Store the latent representations from the previous window

        for start_t in tqdm(range(0, seq_len, window_size - overlap_size)):
            end_t = min(start_t + window_size, seq_len)
            window_len = end_t - start_t
            assert window_len > 0

            # Initialize x_t_packed_slice for the current window
            x_t_packed_slice = torch.randn(
                (num_samples, window_len, denoiser_network.get_d_state()), device=device
            )

            # Blend the latent representations in the overlapping region
            if prev_window_x_t is not None:
                num_overlap = min(overlap_size, window_len)
                # Use the previous window's output for the overlapping region
                x_t_packed_slice[:, :num_overlap, :] = prev_window_x_t[
                    :, -num_overlap:, :
                ]

            # Denoising timesteps
            ts = quadratic_ts()
            for i in range(len(ts) - 1):
                t = ts[i]
                t_next = ts[i + 1]

                T_cpf_tm1_cpf_t_slice = T_cpf_tm1_cpf_t[
                    None, start_t:end_t, :
                ].repeat(num_samples, 1, 1)
                T_world_cpf_slice = Ts_world_cpf_shifted[
                    None, start_t + 1 : end_t + 1, :
                ].repeat(num_samples, 1, 1)

                x_0_packed_pred_slice = denoiser_network.forward(
                    x_t_packed_slice,
                    torch.full((num_samples,), t, device=device),
                    T_cpf_tm1_cpf_t=T_cpf_tm1_cpf_t_slice,
                    T_world_cpf=T_world_cpf_slice,
                    project_output_rot6d=False,
                    hand_positions_wrt_cpf=None,
                    mask=None,
                )

                x_t_packed_slice = (
                    torch.sqrt(alpha_bar_t[t_next]) * x_0_packed_pred_slice
                    + (
                        torch.sqrt(1 - alpha_bar_t[t_next] - sigma_t[t] ** 2)
                        * (x_t_packed_slice - torch.sqrt(alpha_bar_t[t]) * x_0_packed_pred_slice)
                        / torch.sqrt(1 - alpha_bar_t[t] + 1e-1)
                    )
                    + sigma_t[t] * torch.randn_like(x_0_packed_pred_slice)
                )

            if guidance_mode != "off" and guidance_post:
                x_0_packed_pred_slice_traj = network.EgoDenoiseTraj.unpack(
                    x_0_packed_pred_slice,
                    include_hands=denoiser_network.config.include_hands,
                )
                x_0_packed_pred_slice_traj, _ = do_guidance_optimization(
                    Ts_world_cpf=Ts_world_cpf[1 + start_t : end_t + 1, :],
                    traj=x_0_packed_pred_slice_traj,
                    body_model=body_model,
                    guidance_mode=guidance_mode,
                    phase="post",
                    hamer_detections=hamer_detections,
                    aria_detections=aria_detections,
                )
                x_0_packed_pred_slice = x_0_packed_pred_slice_traj.pack()

            # Store the latent representations for the next window
            prev_window_x_t = x_0_packed_pred_slice.clone()

            # Accumulate the denoised outputs
            if start_t > 0:
                x_0_packed_pred[:, start_t + overlap_size : end_t, :] = x_0_packed_pred_slice[
                    :, overlap_size:, :
                ]
            else:
                x_0_packed_pred[:, start_t:end_t, :] = x_0_packed_pred_slice

    x_0_packed_pred = network.EgoDenoiseTraj.unpack(
        x_0_packed_pred,
        include_hands=denoiser_network.config.include_hands,
        project_rot6d=True,
    )

    if torch.any(torch.isnan(x_0_packed_pred.pack())):
        raise RuntimeError("Found NaN in the denoised trajectory.")

    duration = time.time() - start_time
    logger.info(
        f"RUNTIME: {duration:.6f}, SEQ_LEN: {seq_len}, FPS: {seq_len / duration:.2f}"
    )

    return x_0_packed_pred
