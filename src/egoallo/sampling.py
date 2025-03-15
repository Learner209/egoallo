from __future__ import annotations

import copy
import time
from typing import TYPE_CHECKING

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from egoallo.setup_logger import setup_logger
from tqdm import tqdm

from . import fncsmpl_library as fncsmpl

# from . import fncsmpl
from . import network
from .guidance_optimizer_jax import do_guidance_optimization
from .guidance_optimizer_jax import GuidanceMode
from .hand_detection_structs import CorrespondedAriaHandWristPoseDetections
from .hand_detection_structs import CorrespondedHamerDetections
from .tensor_dataclass import TensorDataclass
from .transforms import SE3
from .transforms import SO3

if TYPE_CHECKING:
    from .data.dataclass import EgoTrainingData
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


# @jaxtyped(typechecker=typeguard.typechecked)
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


# # @jaxtyped(typechecker=typeguard.typechecked)
# def run_sampling_with_stitching(
#     denoiser_network: network.EgoDenoiser,
#     body_model: fncsmpl.SmplhModel,
#     runtime_config: EgoAlloTrainConfig,
#     guidance_mode: GuidanceMode,
#     guidance_post: bool,
#     guidance_inner: bool,
#     Ts_world_cpf: Float[Tensor, "time 7"],
#     hamer_detections: None | CorrespondedHamerDetections,
#     aria_detections: None | CorrespondedAriaHandWristPoseDetections,
#     num_samples: int,
#     device: torch.device,
# ) -> network.AbsoluteDenoiseTraj:
#     noise_constants = CosineNoiseScheduleConstants.compute(timesteps=1000).to(
#         device=device
#     )
#     alpha_bar_t = noise_constants.alpha_bar_t
#     alpha_t = noise_constants.alpha_t

#     x_t_packed = torch.randn(
#         (num_samples, Ts_world_cpf.shape[0] - 1, runtime_config.denoising.get_d_state()),
#         device=device,
#     )
#     if runtime_config.denoising.denoising_mode == "absolute":
#         _cls = network.AbsoluteDenoiseTraj
#     elif runtime_config.denoising.denoising_mode == "velocity":
#         _cls = network.VelocityDenoiseTraj
#     elif runtime_config.denoising.denoising_mode == "joints_only":
#         _cls = network.JointsOnlyTraj
#     else:
#         raise ValueError(f"Invalid denoising mode: {runtime_config.denoising.denoising_mode}")

#     x_t = _cls.unpack(
#         x_t_packed, include_hands=denoiser_network.config.include_hands
#     )
#     x_t.T_world_root = Ts_world_cpf[1:]  # Use the transformed poses
#     x_t_list = [x_t]
#     ts = quadratic_ts(timesteps=1000)

#     seq_len = x_t_packed.shape[1]

#     start_time = None

#     window_size = 128
#     overlap_size = 32
#     canonical_overlap_weights = (
#         torch.from_numpy(
#             np.minimum(
#                 # Make this shape /```\
#                 overlap_size,
#                 np.minimum(
#                     # Make this shape: /
#                     np.arange(1, seq_len + 1),
#                     # Make this shape: \
#                     np.arange(1, seq_len + 1)[::-1],
#                 ),
#             )
#             / overlap_size,
#         )
#         .to(device)
#         .to(torch.float32)
#     )
#     for i in range(len(ts) - 1):
#         print(f"Sampling {i}/{len(ts) - 1}")
#         t = ts[i]
#         t_next = ts[i + 1]

#         with torch.inference_mode():
#             # Chop everything into windows.
#             x_0_packed_pred = torch.zeros_like(x_t_packed)
#             overlap_weights = torch.zeros((1, seq_len, 1), device=x_t_packed.device)

#             # Denoise each window.
#             for start_t in range(0, seq_len, window_size - overlap_size):
#                 end_t = min(start_t + window_size, seq_len)
#                 assert end_t - start_t > 0
#                 overlap_weights_slice = canonical_overlap_weights[
#                     None, : end_t - start_t, None
#                 ]
#                 overlap_weights[:, start_t:end_t, :] += overlap_weights_slice
#                 x_0_packed_pred[:, start_t:end_t, :] += (
#                     # FIXME: the param of x_t_unpacked has been changed in `forward` function. Take notes.
#                     denoiser_network.forward(
#                         runtime_config.denoising.unpack_traj(x_t_packed[:, start_t:end_t, :], include_hands=runtime_config.model.include_hands),
#                         torch.tensor([t], device=device).expand((num_samples,)),
#                         T_world_cpf=Ts_world_cpf[
#                             None, start_t + 1 : end_t + 1, :
#                         ].repeat((num_samples, 1, 1)),
#                         project_output_rotmats=False,
#                         hand_positions_wrt_cpf=None,  # TODO: this should be filled in!!
#                         mask=None,
#                     )
#                     * overlap_weights_slice
#                 )

#             # Take the mean for overlapping regions.
#             x_0_packed_pred /= overlap_weights

#             x_0_packed_pred = _cls.unpack(
#                 x_0_packed_pred,
#                 include_hands=denoiser_network.config.include_hands,
#                 project_rotmats=True,
#             ).pack()

#         if torch.any(torch.isnan(x_0_packed_pred)):
#             print("found nan", i)
#         sigma_t = torch.cat(
#             [
#                 torch.zeros((1,), device=device),
#                 torch.sqrt(
#                     (1.0 - alpha_bar_t[:-1]) / (1 - alpha_bar_t[1:]) * (1 - alpha_t)
#                 )
#                 * 0.8,
#             ]
#         )

#         if guidance_mode != "off" and guidance_inner:
#             x_0_pred, _ = do_guidance_optimization(
#                 # It's important that we _don't_ use the shifted transforms here.
#                 Ts_world_cpf=Ts_world_cpf[1:, :],
#                 traj=_cls.unpack(
#                     x_0_packed_pred, include_hands=denoiser_network.config.include_hands
#                 ),
#                 body_model=body_model,
#                 guidance_mode=guidance_mode,
#                 phase="inner",
#                 hamer_detections=hamer_detections,
#                 aria_detections=aria_detections,
#             )
#             x_0_packed_pred = x_0_pred.pack()
#             del x_0_pred

#         if start_time is None:
#             start_time = time.time()

#         # print(sigma_t)
#         x_t_packed = (
#             torch.sqrt(alpha_bar_t[t_next]) * x_0_packed_pred
#             + (
#                 torch.sqrt(1 - alpha_bar_t[t_next] - sigma_t[t] ** 2)
#                 * (x_t_packed - torch.sqrt(alpha_bar_t[t]) * x_0_packed_pred)
#                 / torch.sqrt(1 - alpha_bar_t[t] + 1e-1)
#             )
#             + sigma_t[t] * torch.randn(x_0_packed_pred.shape, device=device)
#         )
#         x_t_list.append(
#             _cls.unpack(
#                 x_t_packed, include_hands=denoiser_network.config.include_hands
#             )
#         )

#     if guidance_mode != "off" and guidance_post:
#         constrained_traj = x_t_list[-1]
#         constrained_traj, _ = do_guidance_optimization(
#             # It's important that we _don't_ use the shifted transforms here.
#             Ts_world_cpf=Ts_world_cpf[1:, :],
#             traj=constrained_traj,
#             body_model=body_model,
#             guidance_mode=guidance_mode,
#             phase="post",
#             hamer_detections=hamer_detections,
#             aria_detections=aria_detections,
#         )
#         assert start_time is not None
#         print("RUNTIME (exclude first optimization)", time.time() - start_time)
#         return constrained_traj
#     else:
#         assert start_time is not None
#         print("RUNTIME (exclude first optimization)", time.time() - start_time)
#         return x_t_list[-1]


# @jaxtyped(typechecker=typeguard.typechecked)
def run_sampling_with_masked_data(
    denoiser_network: network.EgoDenoiser,
    body_model: fncsmpl.SmplhModel,
    masked_data: EgoTrainingData,
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

    post_processed_batch = masked_data.postprocess()
    del masked_data

    x_t_packed = torch.randn(
        (
            num_samples,
            post_processed_batch.joints_wrt_world.shape[1],
            runtime_config.denoising.get_d_state(),
        ),
        device=device,
    )
    x_t_list = [
        runtime_config.denoising.unpack_traj(
            x_t_packed,
            include_hands=runtime_config.model.include_hands,
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
                )

                this_window_data = this_window_data.postprocess()
                ret = this_window_data._post_process(ret)
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
                / torch.sqrt(1 - alpha_bar_t[t] + 1e-1)
            )
            + sigma_t[t] * torch.randn_like(x_0_packed_pred)
        )
        x_t_list.append(
            runtime_config.denoising.unpack_traj(
                x_t_packed,
                include_hands=runtime_config.model.include_hands,
                project_rotmats=False,
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


# # Implementation of DDPM sampling
# # @jaxtyped(typechecker=typeguard.typechecked)
# def run_sampling_with_masked_data_ddpm(
#     denoiser_network: network.EgoDenoiser,
#     body_model: fncsmpl.SmplhModel,
#     masked_data: EgoTrainingData,
#     runtime_config: EgoAlloTrainConfig,
#     guidance_mode: GuidanceMode,
#     guidance_post: bool,
#     guidance_inner: bool,
#     floor_z: float,
#     hamer_detections: None | CorrespondedHamerDetections,
#     aria_detections: None | CorrespondedAriaHandWristPoseDetections,
#     num_samples: int,
#     device: torch.device,
# ) -> network.AbsoluteDenoiseTraj:
#     """
#     DDPM sampling version of run_sampling_with_masked_data.
#     Uses full stochastic sampling instead of DDIM's deterministic sampling.
#     """
#     # Initialize noise schedule - same as DDIM
#     noise_constants = CosineNoiseScheduleConstants.compute(timesteps=1000).to(
#         device=device
#     )
#     alpha_bar_t = noise_constants.alpha_bar_t
#     alpha_t = noise_constants.alpha_t

#     # Initialize random noise
#     # FIXME: the denoiser_network.get_d_state() is not Implemented for now.
#     x_t_packed = torch.randn(
#         (
#             num_samples,
#             masked_data.joints_wrt_world.shape[1],
#             runtime_config.denoising.get_d_state(),
#         ),
#         device=device,
#     )

#     x_t_list = [
#         runtime_config.denoising.unpack_traj(
#             x_t_packed, include_hands=runtime_config.model.include_hands
#         )
#     ]
#     ts = linear_ts(timesteps=1000)

#     seq_len = x_t_packed.shape[1]
#     window_size = 128
#     overlap_size = 32

#     canonical_overlap_weights = (
#         torch.from_numpy(
#             np.minimum(
#                 overlap_size,
#                 np.minimum(
#                     np.arange(1, seq_len + 1),
#                     np.arange(1, seq_len + 1)[::-1],
#                 ),
#             )
#             / overlap_size,
#         )
#         .to(device)
#         .to(torch.float32)
#     )

#
#     for i in tqdm(
#         range(len(ts) - 1), total=len(ts) - 1, desc="DDPM sampling", ascii=" >="
#     ):
#         t = ts[i]
#         t_next = ts[i + 1]

#         with torch.inference_mode():
#             x_0_packed_pred = torch.zeros_like(x_t_packed)
#             overlap_weights = torch.zeros((1, seq_len, 1), device=x_t_packed.device)

#             # Process windows with overlap
#             for start_t in range(0, seq_len, window_size - overlap_size):
#                 end_t = min(start_t + window_size, seq_len)
#                 overlap_weights_slice = canonical_overlap_weights[
#                     None, : end_t - start_t, None
#                 ]
#                 overlap_weights[:, start_t:end_t, :] += overlap_weights_slice

#                 x_0_packed_pred[:, start_t:end_t, :] += (
#                     denoiser_network.forward(
#                         x_t_unpacked=runtime_config.denoising.unpack_traj(x_t_packed[:, start_t:end_t, :], include_hands=runtime_config.model.include_hands),
#                         t=torch.tensor([t], device=device).expand((num_samples,)),
#                         joints=masked_data.joints_wrt_world[:, start_t:end_t, :],
#                         visible_joints_mask=masked_data.visible_joints_mask[
#                             :, start_t:end_t, :
#                         ],
#                         project_output_rotmats=False,
#                         mask=masked_data.mask[:, start_t:end_t],
#                     )
#                     * overlap_weights_slice
#                 )

#             x_0_packed_pred /= overlap_weights

#             x_0_pred = runtime_config.denoising.unpack_traj(
#                 x_0_packed_pred,
#                 include_hands=runtime_config.model.include_hands,
#                 project_rotmats=False,
#             )

#         if guidance_mode != "off" and guidance_inner:
#             x_0_pred, _ = do_guidance_optimization(
#                 T_world_root=SE3.from_rotation_and_translation(
#                     SO3.from_matrix(x_0_pred.R_world_root), x_0_pred.t_world_root
#                 )
#                 .parameters()
#                 .squeeze(0),
#                 traj=x_0_pred,
#                 body_model=body_model,
#                 guidance_mode=guidance_mode,
#                 phase="inner",
#                 hamer_detections=hamer_detections,
#                 aria_detections=aria_detections,
#             )
#         x_0_packed_pred = x_0_pred.pack()

#         if torch.any(torch.isnan(x_0_packed_pred)):
#             print("found nan", i)

#         # DDPM update equation
#         # x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise
#         noise = torch.randn(
#             x_0_packed_pred.shape, dtype=x_0_packed_pred.dtype, device=device
#         )
#         x_t_packed = (
#             torch.sqrt(alpha_bar_t[t_next]) * x_0_packed_pred
#             + torch.sqrt(1.0 - alpha_bar_t[t_next]) * noise
#         )

#         x_t_list.append(
#             runtime_config.denoising.unpack_traj(
#                 x_t_packed,
#                 include_hands=runtime_config.model.include_hands,
#                 project_rotmats=False,
#             )
#         )

#
#     if guidance_mode != "off" and guidance_post:
#         constrained_traj = x_t_list[-1]
#         constrained_traj, _ = do_guidance_optimization(
#             T_world_root=SE3.from_rotation_and_translation(
#                 SO3.from_matrix(constrained_traj.R_world_root),
#                 constrained_traj.t_world_root,
#             )
#             .parameters()
#             .squeeze(0),
#             traj=constrained_traj,
#             body_model=body_model,
#             guidance_mode=guidance_mode,
#             phase="post",
#             hamer_detections=hamer_detections,
#             aria_detections=aria_detections,
#         )
#         return constrained_traj
#     else:
#         return x_t_list[-1]


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
    # Offset the T_world_cpf transform to place the floor at z=0 for the
    # denoiser network. All of the network outputs are local, so we don't need to
    # unoffset when returning.
    Ts_world_cpf_shifted = Ts_world_cpf.clone()
    Ts_world_cpf_shifted[..., 6] -= floor_z

    noise_constants = CosineNoiseScheduleConstants.compute(timesteps=1000).to(
        device=device,
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

    ts = quadratic_ts()

    seq_len = x_t_packed.shape[1]

    start_time = None

    window_size = 128
    overlap_size = 64
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
    start_time = time.time()
    # Chop everything into windows.
    x_0_packed_pred = torch.zeros_like(x_t_packed)
    overlap_weights = torch.zeros((1, seq_len, 1), device=x_t_packed.device)
    sigma_t = torch.cat(
        [
            torch.zeros((1,), device=device),
            torch.sqrt((1.0 - alpha_bar_t[:-1]) / (1 - alpha_bar_t[1:]) * (1 - alpha_t))
            * 0.8,
        ],
    )

    with torch.inference_mode():
        prev_window_x_t = (
            None  # Store the latent representations from the previous window
        )

        for start_t in tqdm(range(0, seq_len, window_size - overlap_size)):
            end_t = min(start_t + window_size, seq_len)
            window_len = end_t - start_t
            assert window_len > 0

            overlap_weights_slice = canonical_overlap_weights[start_t:end_t][
                None,
                :,
                None,
            ]
            overlap_weights[:, start_t:end_t, :] += overlap_weights_slice

            # Initialize x_t_packed for the current window
            x_t_packed_slice = torch.randn(
                (num_samples, window_len, denoiser_network.get_d_state()),
                device=device,
            )

            # Denoising timesteps
            ts = quadratic_ts()
            x_0_packed_pred_slice = None
            for i in range(len(ts) - 1):
                t = ts[i]
                t_next = ts[i + 1]

                # Blend the latent representations in the overlapping region
                if prev_window_x_t is not None:
                    num_overlap = min(overlap_size, end_t - start_t)
                    # Initialize overlapping region with previous output
                    x_t_packed_slice[:, :num_overlap, :] = prev_window_x_t[
                        :,
                        -num_overlap:,
                        :,
                    ]

                x_0_packed_pred_slice = denoiser_network.forward(
                    x_t_packed_slice,
                    torch.full((num_samples,), t, device=device),
                    T_cpf_tm1_cpf_t=T_cpf_tm1_cpf_t[None, start_t:end_t, :].repeat(
                        num_samples,
                        1,
                        1,
                    ),
                    T_world_cpf=Ts_world_cpf_shifted[
                        None,
                        start_t + 1 : end_t + 1,
                        :,
                    ].repeat(num_samples, 1, 1),
                    project_output_rotmats=False,
                    hand_positions_wrt_cpf=None,
                    mask=None,
                )

                x_t_packed_slice = (
                    torch.sqrt(alpha_bar_t[t_next]) * x_0_packed_pred_slice
                    + (
                        torch.sqrt(1 - alpha_bar_t[t_next] - sigma_t[t] ** 2)
                        * (
                            x_t_packed_slice
                            - torch.sqrt(alpha_bar_t[t]) * x_0_packed_pred_slice
                        )
                        / torch.sqrt(1 - alpha_bar_t[t] + 1e-1)
                    )
                    + sigma_t[t] * torch.randn_like(x_0_packed_pred_slice)
                )

            if guidance_mode != "off" and guidance_post:
                x_0_packed_pred_slice, _ = do_guidance_optimization(
                    # It's important that we _don't_ use the shifted transforms here.
                    Ts_world_cpf=Ts_world_cpf[1 + start_t : end_t + 1, :],
                    traj=network.EgoDenoiseTraj.unpack(
                        x_0_packed_pred_slice,
                        include_hands=denoiser_network.config.include_hands,
                    ),
                    body_model=body_model,
                    guidance_mode=guidance_mode,
                    phase="post",
                    hamer_detections=hamer_detections,
                    aria_detections=aria_detections,
                )
                x_0_packed_pred_slice = x_0_packed_pred_slice.pack()

            # Store the latent representations for the next window
            prev_window_x_t = x_0_packed_pred_slice.clone()

            # Accumulate the denoised outputs
            if start_t > 0:
                x_0_packed_pred[:, start_t + overlap_size : end_t, :] = (
                    x_0_packed_pred_slice[:, overlap_size:, :]
                )
            else:
                x_0_packed_pred[:, start_t:end_t, :] = x_0_packed_pred_slice

    x_0_packed_pred = network.EgoDenoiseTraj.unpack(
        x_0_packed_pred,
        include_hands=denoiser_network.config.include_hands,
        project_rotmats=True,
    )

    if torch.any(torch.isnan(x_0_packed_pred.pack())):
        raise RuntimeWarning("found nan")

    duration = time.time() - start_time
    logger.info(
        f"RUNTIME: {duration:.6f}, SEQ_LEN: {seq_len:2d}, FPS: {seq_len / duration:.2f}",
    )

    return x_0_packed_pred
