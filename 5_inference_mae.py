from __future__ import annotations

import numpy as np
import viser
import torch
from pathlib import Path

from egoallo.data.dataclass import EgoTrainingData
from egoallo.inference_utils import (
    create_masked_training_data,
    load_denoiser,
    EgoDenoiseTraj
)
from egoallo import fncsmpl
from egoallo.vis_helpers import visualize_traj_and_hand_detections
from egoallo.sampling import CosineNoiseScheduleConstants, quadratic_ts
import time

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from tqdm.auto import tqdm

from egoallo import fncsmpl, network
from egoallo.guidance_optimizer_jax import (
    GuidanceMode,
    do_guidance_optimization,
)
from egoallo.hand_detection_structs import (
    CorrespondedAriaHandWristPoseDetections,
    CorrespondedHamerDetections,
)
from egoallo.tensor_dataclass import TensorDataclass
from egoallo.transforms import SE3, SO3
from egoallo.training_utils import ipdb_safety_net
import dataclasses


@dataclasses.dataclass
class Args:
    npz_path: Path = Path("./egoallo_example_trajectories/coffeemachine/egoallo_outputs/20240929-011937_10-522.npz")
    """Path to the input trajectory."""
    checkpoint_dir: Path = Path("/mnt/homes/minghao/src/robotflow/egoallo/experiments/predict_T_world_root/v7/checkpoints_10000")
    """Path to the checkpoint directory."""
    smplh_npz_path: Path = Path("./data/smplh/neutral/model.npz")
    """Path to the SMPLH model."""
    traj_length: int = 128
    """How many timesteps to estimate body motion for."""
    num_samples: int = 1
    """Number of samples to take."""
    guidance_mode: GuidanceMode = "aria_hamer"
    """Which guidance mode to use."""
    guidance_inner: bool = True
    """Whether to apply guidance optimizer between denoising steps. This is
    important if we're doing anything with hands. It can be turned off to speed
    up debugging/experiments, or if we only care about foot skating losses."""
    guidance_post: bool = True
    """Whether to apply guidance optimizer after diffusion sampling."""
    visualize_traj: bool = True
    """Whether to visualize the trajectory after sampling."""
    mask_ratio: float = 0.75
    """Ratio of joints to mask."""


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

    noise_constants = CosineNoiseScheduleConstants.compute(timesteps=1000).to(device=device)
    alpha_bar_t = noise_constants.alpha_bar_t
    alpha_t = noise_constants.alpha_t

    # Initialize noise with proper shape
    # import ipdb; ipdb.set_trace()
    x_t_packed = torch.randn(
        (num_samples, masked_data.visible_joints.shape[1], denoiser_network.get_d_state()),
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
    
    # Create overlap weights for windowed processing
    canonical_overlap_weights = torch.from_numpy(
        np.minimum(
            overlap_size,
            np.minimum(
                np.arange(1, seq_len + 1),
                np.arange(1, seq_len + 1)[::-1],
            ),
        )
        / overlap_size,
    ).to(device).to(torch.float32)

    for i in tqdm(range(len(ts) - 1)):
        t = ts[i]
        t_next = ts[i + 1]

        with torch.inference_mode():
            x_0_packed_pred = torch.zeros_like(x_t_packed)
            overlap_weights = torch.zeros((1, seq_len, 1), device=x_t_packed.device)

            # Process each window
            for start_t in range(0, seq_len, window_size - overlap_size):
                end_t = min(start_t + window_size, seq_len)
                overlap_weights_slice = canonical_overlap_weights[None, :end_t - start_t, None]
                overlap_weights[:, start_t:end_t, :] += overlap_weights_slice

                # Forward pass with conditioning from masked data
                x_0_packed_pred[:, start_t:end_t, :] += denoiser_network.forward(
                    x_t_packed=x_t_packed[:, start_t:end_t, :],
                    t=torch.tensor([t], device=device).expand((num_samples,)),
                    visible_joints=masked_data.visible_joints[:, start_t:end_t, :],
                    visible_joints_mask=masked_data.visible_joints_mask[:, start_t:end_t, :],
                    project_output_rotmats=False,
                    mask=masked_data.mask[:, start_t:end_t],
                ) * overlap_weights_slice

            # Average overlapping regions
            x_0_packed_pred /= overlap_weights

            x_0_packed_pred = network.EgoDenoiseTraj.unpack(
                x_0_packed_pred,
                include_hands=denoiser_network.config.include_hands,
                project_rotmats=True,
            ).pack()

        # Apply guidance if needed
        if guidance_mode != "off" and guidance_inner:
            x_0_pred, _ = do_guidance_optimization(
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
        # Update x_t using noise schedule
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
                x_t_packed, include_hands=denoiser_network.config.include_hands
            )
        )

    # Final guidance optimization if needed
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
        return constrained_traj
    else:
        return x_t_list[-1]

def main(
    args: Args,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> network.EgoDenoiseTraj:

    # Load data and models
    traj_data = np.load(args.npz_path)
    body_model = fncsmpl.SmplhModel.load(args.smplh_npz_path).to(device)
    denoiser_network = load_denoiser(args.checkpoint_dir).to(device)

    # Prepare input tensors
    # import ipdb; ipdb.set_trace()

    Ts_world_cpf = torch.from_numpy(traj_data['Ts_world_cpf']).to(device)
    Ts_world_root = torch.from_numpy(traj_data['Ts_world_root']).to(device)
    body_quats = torch.from_numpy(traj_data['body_quats']).to(device)
    left_hand_quats = torch.from_numpy(traj_data['left_hand_quats']).to(device)
    right_hand_quats = torch.from_numpy(traj_data['right_hand_quats']).to(device)
    contacts = torch.from_numpy(traj_data['contacts']).to(device)
    betas = torch.from_numpy(traj_data['betas']).to(device)

    # Create posed data
    local_quats = torch.cat([body_quats, left_hand_quats, right_hand_quats], dim=-2)
    shaped_model = body_model.with_shape(betas)
    posed = shaped_model.with_pose(Ts_world_root, local_quats)

    # Create masked training data
    masked_data = create_masked_training_data(
        posed=posed,
        Ts_world_cpf=Ts_world_cpf,
        contacts=contacts,
        betas=betas,
        mask_ratio=args.mask_ratio
    )

    # Run sampling with masked data
    denoised_traj = run_sampling_with_masked_data(
        denoiser_network=denoiser_network,
        body_model=body_model,
        masked_data=masked_data,
        guidance_mode="no_hands",
        guidance_post=False,
        guidance_inner=False,
        floor_z=0.0,
        hamer_detections=None,
        aria_detections=None,
        num_samples=1,
        device=device,
    )

    # Save outputs in case we want to visualize later.
    # if args.save_traj:
    #     save_name = (
    #         time.strftime("%Y%m%d-%H%M%S")
    #         + f"_{args.start_index}-{args.start_index + args.traj_length}"
    #     )
    #     out_path = args.npz_path.parent / "egoallo_outputs" / (save_name + ".npz")
    #     out_path.parent.mkdir(parents=True, exist_ok=True)
    #     assert not out_path.exists()
    #     (args.npz_path.parent / "egoallo_outputs" / (save_name + "_args.yaml")).write_text(
    #         yaml.dump(dataclasses.asdict(args))
    #     )

    #     posed = traj.apply_to_body(body_model)
    #     Ts_world_root = fncsmpl_extensions.get_T_world_root_from_cpf_pose(
    #         posed, Ts_world_cpf[..., 1:, :]
    #     )

    #     # Save both original and inferred trajectories
    #     print(f"Saving to {out_path}...", end="")
    #     np.savez(
    #         out_path,
    #         # Original trajectory
    #         Ts_world_cpf=Ts_world_cpf[1:, :].numpy(force=True),
    #         Ts_world_root=Ts_world_root.numpy(force=True),
    #         body_quats=posed.local_quats[..., :21, :].numpy(force=True),
    #         left_hand_quats=posed.local_quats[..., 21:36, :].numpy(force=True),
    #         right_hand_quats=posed.local_quats[..., 36:51, :].numpy(force=True),
    #         contacts=traj.contacts.numpy(force=True),
    #         betas=traj.betas.numpy(force=True),
    #         # Masked and inferred data
    #         visible_joints_mask=masked_data.visible_joints_mask.numpy(force=True),
    #         inferred_body_quats=inferred_traj.body_rotmats.numpy(force=True),
    #         inferred_hand_quats=inferred_traj.hand_rotmats.numpy(force=True),
    #         frame_nums=np.arange(args.start_index, args.start_index + args.traj_length),
    #         timestamps_ns=(np.array(pose_timestamps_sec) * 1e9).astype(np.int64),
    #     )
    #     print("saved!")

    # Visualize.
    if args.visualize_traj:
        server = viser.ViserServer()
        server.gui.configure_theme(dark_mode=True)
        assert server is not None
        loop_cb = visualize_traj_and_hand_detections(
            server,
            denoised_traj.T_world_root.squeeze(),
            denoised_traj,
            body_model,
            hamer_detections=None,
            aria_detections=None,
            points_data=None,
            splat_path=None,
            floor_z=0.0,
        )
        while True:
            loop_cb()


if __name__ == "__main__":
    import tyro

    ipdb_safety_net()
    main(tyro.cli(Args))