from __future__ import annotations

import copy
import dataclasses
import time
from typing import TYPE_CHECKING

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from egoallo.type_stubs import EgoTrainingDataType
from egoallo.setup_logger import setup_logger

from .middleware.third_party.HybrIK.hybrik.models.layers.smplh import (
    fncsmplh as fncsmpl,
)

from . import network
from .guidance_optimizer_jax import do_guidance_optimization
from .guidance_optimizer_jax import GuidanceMode
from .hand_detection_structs import CorrespondedAriaHandWristPoseDetections
from .hand_detection_structs import CorrespondedHamerDetections
from egoallo.tensor_dataclass import TensorDataclass
from egoallo.transforms import SE3
from egoallo.transforms import SO3
import typeguard
from jaxtyping import jaxtyped

if TYPE_CHECKING:
    from .config.train.train_config import EgoAlloTrainConfig

logger = setup_logger(output=None, name=__name__)


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


@jaxtyped(typechecker=typeguard.typechecked)
class CosineNoiseScheduleConstants(TensorDataclass):
    """Constants used for cosine noise scheduling."""

    alpha_t: Float[Tensor, "T"]
    r"""$1 - \beta_t$"""

    alpha_bar_t: Float[Tensor, "T+1"]
    r"""$\Prod_{j=1}^t (1 - \beta_j)$"""

    @staticmethod
    def compute(timesteps: int, s: float = 0.008) -> CosineNoiseScheduleConstants:
        """Compute cosine noise schedule constants.

        Args:
            timesteps: Number of timesteps
            s: Offset parameter

        Returns:
            CosineNoiseScheduleConstants: Computed schedule constants
        """
        steps = timesteps + 1
        x = torch.linspace(0, 1, steps, dtype=torch.float32)

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
            alpha_t=alpha_t,
            alpha_bar_t=alpha_cumprod_t,
        )


@jaxtyped(typechecker=typeguard.typechecked)
def _run_sampling_with_masked_data(
    denoiser_network: network.EgoDenoiser,
    body_model: fncsmpl.SmplhModel,
    masked_data: EgoTrainingDataType,
    runtime_config: EgoAlloTrainConfig,
    guidance_mode: GuidanceMode,
    guidance_post: bool,
    guidance_inner: bool,
    floor_z: float,
    hamer_detections: None | CorrespondedHamerDetections,
    aria_detections: None | CorrespondedAriaHandWristPoseDetections,
    num_samples: int,
    window_size: int,
    overlap_size: int,
    device: torch.device,
) -> network.AbsoluteDenoiseTraj:
    assert masked_data.metadata.stage == "preprocessed", (
        "EgoTrainingData should be preprocessed before being used to create trajectories. \
            , The logic is traj should be sent to network so that the ego_data should be between pre and post."
    )
    assert not torch.any(
        torch.isnan(
            masked_data.joints_wrt_world[masked_data.visible_joints_mask.bool()],
        ),
    ), "Found nan in joints_wrt_world"

    # FIXME: currently the batch-size dimension of `masked_data` is not supported, as the num_samples `param` would conflict with batch_size dim of `masked_data`.
    noise_constants = CosineNoiseScheduleConstants.compute(timesteps=1000).to(
        device=device,
    )
    alpha_bar_t = noise_constants.alpha_bar_t
    alpha_t = noise_constants.alpha_t

    x_t_packed = torch.randn(
        (
            num_samples,
            masked_data.joints_wrt_world.shape[1],
            runtime_config.denoising.d_state(),
        ),
        device=device,
    )

    x_t_list = [
        runtime_config.denoising.unpack_traj(
            x_t_packed,
            include_hands=runtime_config.model.include_hands,
            metadata=dataclasses.replace(masked_data.metadata),
        ),
    ]

    post_processed_batch = masked_data.postprocess()

    ts = quadratic_ts(timesteps=1000)

    seq_len = x_t_packed.shape[1]

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

    for i in range(len(ts) - 1):
        t = ts[i]
        t_next = ts[i + 1]

        with torch.inference_mode():
            x_0_packed_pred = torch.zeros_like(x_t_packed)
            overlap_weights = torch.zeros((1, seq_len, 1), device=x_t_packed.device)

            for start_t in range(0, seq_len, window_size - overlap_size):
                end_t = min(start_t + window_size, seq_len)
                overlap_weights_slice = canonical_overlap_weights[
                    None,
                    : end_t - start_t,
                    None,
                ]
                overlap_weights[:, start_t:end_t, :] += overlap_weights_slice

                this_window_data = copy.deepcopy(post_processed_batch[:, start_t:end_t])
                # FIXME: this is a hack to follow the state machine of EgoTrainingData dataclass.
                this_window_data.metadata.stage = "raw"
                this_window_data = this_window_data.preprocess()

                ret = denoiser_network.forward(
                    x_t_unpacked=runtime_config.denoising.unpack_traj(
                        x_t_packed[:, start_t:end_t, :],
                        include_hands=runtime_config.model.include_hands,
                        metadata=dataclasses.replace(this_window_data.metadata),
                    ),
                    t=torch.tensor([t], device=device).expand((num_samples,)),
                    joints=this_window_data.joints_wrt_world,
                    visible_joints_mask=this_window_data.visible_joints_mask,
                    project_output_rotmats=False,
                    mask=this_window_data.mask,
                )

                ret = runtime_config.denoising.unpack_traj(
                    ret,
                    include_hands=runtime_config.model.include_hands,
                    project_rotmats=False,
                    metadata=dataclasses.replace(this_window_data.metadata),
                )

                this_window_data = this_window_data.postprocess()
                ret = this_window_data.postprocess_denoise_traj(ret)
                ret = this_window_data._set_traj(ret)

                ret = ret.pack()
                x_0_packed_pred[:, start_t:end_t, :] += ret * overlap_weights_slice

            x_0_packed_pred /= overlap_weights
            assert not torch.any(torch.isnan(x_0_packed_pred)), (
                "Found nan in x_0_packed_pred"
            )

            x_0_pred = runtime_config.denoising.unpack_traj(  # raw traj
                x_0_packed_pred,
                include_hands=runtime_config.model.include_hands,
                project_rotmats=True,
                metadata=dataclasses.replace(this_window_data.metadata),
            )

        if guidance_mode != "off" and guidance_inner:
            x_0_pred, _ = do_guidance_optimization(
                T_world_root=SE3.from_rotation_and_translation(
                    SO3.from_matrix(x_0_pred.R_world_root),
                    x_0_pred.t_world_root,
                ).parameters(),
                # .squeeze(0),
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
                    (1.0 - alpha_bar_t[:-1]) / (1 - alpha_bar_t[1:]) * (1 - alpha_t),
                )
                * 0.8,
            ],
        )
        x_t_packed = (
            torch.sqrt(alpha_bar_t[t_next]) * x_0_packed_pred
            + (
                torch.sqrt(1 - alpha_bar_t[t_next] - sigma_t[t] ** 2)
                * (x_t_packed - torch.sqrt(alpha_bar_t[t]) * x_0_packed_pred)
                / torch.sqrt(1 - alpha_bar_t[t] + 1e-8)
            )
            + sigma_t[t] * torch.randn_like(x_0_packed_pred)
        )

        metadata = post_processed_batch.metadata
        metadata.stage = "preprocessed"

        x_t_list.append(
            runtime_config.denoising.unpack_traj(
                x_t_packed,
                include_hands=runtime_config.model.include_hands,
                metadata=dataclasses.replace(metadata),
                project_rotmats=True,
            ),
        )
    duration = time.time() - start_time
    logger.info(
        f"RUNTIME: {duration:.6f}, SEQ_LEN: {seq_len:2d}, FPS: {seq_len / duration:.2f}",
    )

    if guidance_mode != "off" and guidance_post:
        constrained_traj = x_t_list[-1]
        constrained_traj, _ = do_guidance_optimization(
            T_world_root=SE3.from_rotation_and_translation(
                SO3.from_matrix(constrained_traj.R_world_root),
                constrained_traj.t_world_root,
            ).parameters(),
            # .squeeze(0),
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


# @jaxtyped(typechecker=typeguard.typechecked)
def run_sampling_with_masked_data(
    denoiser_network: network.EgoDenoiser,
    body_model: fncsmpl.SmplhModel,
    masked_data: EgoTrainingDataType,
    runtime_config: EgoAlloTrainConfig,
    guidance_mode: GuidanceMode,
    guidance_post: bool,
    guidance_inner: bool,
    floor_z: float,
    hamer_detections: None | CorrespondedHamerDetections,
    aria_detections: None | CorrespondedAriaHandWristPoseDetections,
    num_samples: int,
    window_size: int,
    overlap_size: int,
    device: torch.device,
) -> network.AbsoluteDenoiseTraj:
    assert masked_data.metadata.stage == "preprocessed", (
        "EgoTrainingData should be preprocessed before being used to create trajectories. \
            , The logic is traj should be sent to network so that the ego_data should be between pre and post."
    )
    assert not torch.any(
        torch.isnan(
            masked_data.joints_wrt_world[masked_data.visible_joints_mask.bool()],
        ),
    ), "Found nan in joints_wrt_world"

    # FIXME: currently the batch(mask-size dimension of `masked_data` is not supported, as the num_samples `param` would conflict with batch_size dim of `masked_data`.
    noise_constants = CosineNoiseScheduleConstants.compute(timesteps=1000).to(
        device=device,
    )
    alpha_bar_t = noise_constants.alpha_bar_t
    alpha_t = noise_constants.alpha_t

    x_t_packed = torch.randn(
        (
            num_samples,
            masked_data.joints_wrt_world.shape[1],
            runtime_config.denoising.d_state,
        ),
        device=device,
    )
    x_t_list = [
        runtime_config.denoising.unpack_traj(
            x_t_packed,
            include_hands=runtime_config.model.include_hands,
            metadata=masked_data.metadata,
            project_rotmats=False,
        ),
    ]

    ts = quadratic_ts(timesteps=1000)
    seq_len = x_t_packed.shape[1]

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

    # Prepare window data in advance
    window_data = []
    overlap_weights = torch.zeros((1, seq_len, 1), device=x_t_packed.device)

    # Save intermediate batch for metadata assignment later.
    preprocessed_batch = copy.deepcopy(masked_data)
    post_processed_batch = masked_data.postprocess()
    del masked_data

    for start_t in range(0, seq_len, window_size - overlap_size):
        end_t = min(start_t + window_size, seq_len)
        overlap_weights_slice = canonical_overlap_weights[
            None,
            : end_t - start_t,
            None,
        ]
        overlap_weights[:, start_t:end_t, :] += overlap_weights_slice

        win_data = copy.deepcopy(post_processed_batch[:, start_t:end_t])
        # FIXME: this is a hack to follow the state machine of EgoTrainingData dataclass.
        win_data.metadata.stage = "raw"
        win_data = win_data.preprocess()

        window_data.append((start_t, end_t, win_data, overlap_weights_slice))

    start_time = time.time()

    # Main sampling loop with reversed order
    for i in range(len(ts) - 1):
        t = ts[i]
        t_next = ts[i + 1]

        with torch.inference_mode():
            x_0_packed_pred = torch.zeros_like(x_t_packed)

            # Process each window
            for start_t, end_t, win_data, overlap_weights_slice in window_data:
                post_pred_x_0 = denoiser_network.forward(
                    x_t_unpacked=runtime_config.denoising.unpack_traj(
                        x_t_packed[:, start_t:end_t, :],
                        metadata=win_data.metadata,
                        include_hands=runtime_config.model.include_hands,
                    ),
                    t=torch.tensor([t], device=device).expand((num_samples,)),
                    joints=win_data.joints_wrt_world,
                    visible_joints_mask=win_data.visible_joints_mask,
                    project_output_rotmats=False,
                    mask=win_data.mask,
                )

                post_pred_x_0 = runtime_config.denoising.unpack_traj(
                    post_pred_x_0,
                    include_hands=runtime_config.model.include_hands,
                    project_rotmats=False,
                    metadata=win_data.metadata,
                )

                post_pred_x_0 = post_pred_x_0.pack()
                x_0_packed_pred[:, start_t:end_t, :] += (
                    post_pred_x_0 * overlap_weights_slice
                )

            x_0_packed_pred /= overlap_weights
            assert not torch.any(torch.isnan(x_0_packed_pred)), (
                "Found nan in x_0_packed_pred"
            )

            x_0_pred = runtime_config.denoising.unpack_traj(  # raw traj
                x_0_packed_pred,
                include_hands=runtime_config.model.include_hands,
                project_rotmats=False,
                metadata=preprocessed_batch.metadata,
            )

        if guidance_mode != "off" and guidance_inner:
            x_0_pred, _ = do_guidance_optimization(
                T_world_root=SE3.from_rotation_and_translation(
                    SO3.from_matrix(x_0_pred.R_world_root),
                    x_0_pred.t_world_root,
                ).parameters(),
                # .squeeze(0),
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
                    (1.0 - alpha_bar_t[:-1]) / (1 - alpha_bar_t[1:]) * (1 - alpha_t),
                )
                * 0.8,
            ],
        )
        x_t_packed = (
            torch.sqrt(alpha_bar_t[t_next]) * x_0_packed_pred
            + (
                torch.sqrt(1 - alpha_bar_t[t_next] - sigma_t[t] ** 2)
                * (x_t_packed - torch.sqrt(alpha_bar_t[t]) * x_0_packed_pred)
                / torch.sqrt(1 - alpha_bar_t[t] + 1e-8)
            )
            + sigma_t[t] * torch.randn_like(x_0_packed_pred)
        )
        x_t_list.append(
            runtime_config.denoising.unpack_traj(
                x_t_packed,
                include_hands=runtime_config.model.include_hands,
                project_rotmats=False,
                metadata=preprocessed_batch.metadata,
            ),
        )

    pred_x_0 = x_t_list[-1]
    # Post-processing each window's denoised traj.
    for start_t, end_t, win_data, overlap_weights_slice in window_data:
        pred_x_0_window = copy.deepcopy(pred_x_0[:, start_t:end_t])

        win_data.postprocess()
        # breakpoint()
        post_pred_x_0 = win_data.postprocess_denoise_traj(pred_x_0_window)
        post_pred_x_0 = win_data._set_traj(post_pred_x_0)

        pred_x_0[:, start_t:end_t] = post_pred_x_0

    pred_x_0.metadata = post_processed_batch.metadata

    duration = time.time() - start_time
    logger.info(
        f"RUNTIME: {duration:.6f}, SEQ_LEN: {seq_len:2d}, FPS: {seq_len / duration:.2f}",
    )

    if guidance_mode != "off" and guidance_post:
        constrained_traj = pred_x_0
        constrained_traj, _ = do_guidance_optimization(
            T_world_root=SE3.from_rotation_and_translation(
                SO3.from_matrix(constrained_traj.R_world_root),
                constrained_traj.t_world_root,
            ).parameters(),
            # .squeeze(0),
            traj=constrained_traj,
            body_model=body_model,
            guidance_mode=guidance_mode,
            phase="post",
            hamer_detections=hamer_detections,
            aria_detections=aria_detections,
        )
        return constrained_traj
    else:
        return pred_x_0
