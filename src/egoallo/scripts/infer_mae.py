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
from egoallo import fncsmpl_extensions

from egoallo import transforms as tf
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


from egoallo.config import make_cfg, CONFIG_FILE

local_config_file = CONFIG_FILE
CFG = make_cfg(config_name="defaults", config_file=local_config_file, cli_args=[])

from egoallo.utils.setup_logger import setup_logger

logger = setup_logger(output=None, name=__name__)


@dataclasses.dataclass
class Args:
    npz_path: Path = Path("./egoallo_example_trajectories/coffeemachine/egoallo_outputs/20240929-011937_10-522.npz")
    """Path to the input trajectory."""
    checkpoint_dir: Path = Path("/mnt/homes/minghao/src/robotflow/egoallo/experiments/predict_T_world_root/v7/checkpoints_15000")
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
        (num_samples, masked_data.joints_wrt_world.shape[1], denoiser_network.get_d_state()),
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
                    t=torch.tensor([t], device=device).expand((num_samples,)).float(),
                    joints=masked_data.joints_wrt_world[:, start_t:end_t, :],
                    visible_joints_mask=masked_data.visible_joints_mask[:, start_t:end_t, :],
                    project_output_rotmats=False,
                    mask=masked_data.mask[:, start_t:end_t],
                ) * overlap_weights_slice

            # Average overlapping regions
            x_0_packed_pred /= overlap_weights

            x_0_pred = network.EgoDenoiseTraj.unpack(
                x_0_packed_pred,
                include_hands=denoiser_network.config.include_hands,
                project_rotmats=True,
            )

        # Apply guidance if needed
        if guidance_mode != "off" and guidance_inner:
            x_0_pred, _ = do_guidance_optimization(
                T_world_root=SE3.from_rotation_and_translation(
                    SO3.from_matrix(x_0_pred.R_world_root),
                    x_0_pred.t_world_root
                ).parameters().squeeze(0),
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
                x_t_packed, include_hands=denoiser_network.config.include_hands, project_rotmats=True
            )
        )

    # Final guidance optimization if needed
    if guidance_mode != "off" and guidance_post:
        constrained_traj = x_t_list[-1]
        constrained_traj, _ = do_guidance_optimization(
            T_world_root=SE3.from_rotation_and_translation(
                SO3.from_matrix(constrained_traj.R_world_root),
                constrained_traj.t_world_root
            ).parameters().squeeze(0),
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

def calculate_metrics(
    original_posed: fncsmpl.SmplhShapedAndPosed,
    denoised_traj: network.EgoDenoiseTraj,
    masked_data: EgoTrainingData,
    body_model: fncsmpl.SmplhModel,
) -> dict[str, float]:
    """Calculate metrics between original and inferred trajectories.
    
    Args:
        original_posed: Original posed SMPL-H model
        denoised_traj: Inferred trajectory from denoising
        masked_data: Training data with masking information
        body_model: SMPL-H body model
        
    Returns:
        Dictionary containing computed metrics
    """
    # Get inferred posed model
    inferred_posed = denoised_traj.apply_to_body(body_model)
    
    # 1. T_world_root error
    # Translation error
    trans_error = torch.norm(
        original_posed.T_world_root[..., 4:7] - 
        inferred_posed.T_world_root[..., 4:7],
        dim=-1
    ).mean().item()
    
    # Rotation error (geodesic distance)
    R1 = original_posed.T_world_root[..., :4]  # quaternions
    R2 = inferred_posed.T_world_root[..., :4]
    rot_error = torch.arccos(
        torch.abs(torch.sum(R1 * R2, dim=-1)).clamp(-1, 1)
    ).mean().item() * 2.0  # multiply by 2 for full rotation distance
    
    # 2. Joint position errors
    # Get original and inferred joint positions
    orig_joints = torch.cat([original_posed.T_world_root[..., 4:7].unsqueeze(-2), original_posed.Ts_world_joint[..., :CFG.smplh.num_joints-1, 4:7]], dim=-2)
    infer_joints = torch.cat([inferred_posed.T_world_root[..., 4:7].unsqueeze(-2), inferred_posed.Ts_world_joint[..., :CFG.smplh.num_joints-1, 4:7]], dim=-2)
    
    # Calculate per-joint errors
    joint_errors = torch.norm(orig_joints - infer_joints, dim=-1)  # [B, T, J]
    
    # Separate masked and unmasked errors using visible_joints_mask
    visible_mask = masked_data.visible_joints_mask
    
    # Unmasked joints error
    unmasked_mpjpe = joint_errors[visible_mask].mean().item() * 1000  # Convert to mm
    
    # Masked joints error  
    masked_mpjpe = joint_errors[~visible_mask].mean().item() * 1000  # Convert to mm
    
    return {
        "translation_error_meters": trans_error,
        "rotation_error_radians": rot_error,
        "unmasked_mpjpe_mm": unmasked_mpjpe,
        "masked_mpjpe_mm": masked_mpjpe
    }

def main(
    config: Args,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> network.EgoDenoiseTraj:

    # Load data and models
    traj_data = np.load(config.npz_path)
    body_model = fncsmpl.SmplhModel.load(config.smplh_npz_path).to(device)
    denoiser_network, train_config = load_denoiser(config.checkpoint_dir).to(device)

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
            mask_ratio=config.mask_ratio
    )

    # Run sampling with masked data
    denoised_traj: EgoDenoiseTraj = run_sampling_with_masked_data(
        denoiser_network=denoiser_network,
        body_model=body_model,
        masked_data=masked_data,
        guidance_mode="no_hands",
        guidance_post=config.guidance_post,
        guidance_inner=config.guidance_inner,
        floor_z=0.0,
        hamer_detections=None,
        aria_detections=None,
        num_samples=1,
        device=device,
    )

    # Create EgoTrainingData instance from denoised trajectory
    T_world_root = SE3.from_rotation_and_translation(
        SO3.from_matrix(denoised_traj.R_world_root),
        denoised_traj.t_world_root,
    ).parameters()
    betas = denoised_traj.betas
    timesteps = betas.shape[1]
    sample_count = betas.shape[0]
    assert betas.shape == (sample_count, timesteps, 16)
    body_quats = SO3.from_matrix(denoised_traj.body_rotmats).wxyz
    assert body_quats.shape == (sample_count, timesteps, 21, 4)
    device = body_quats.device

    # denoised_traj.hand_rotmats = None
    if denoised_traj.hand_rotmats is not None:
        hand_quats = SO3.from_matrix(denoised_traj.hand_rotmats).wxyz
        left_hand_quats = hand_quats[..., :15, :]
        right_hand_quats = hand_quats[..., 15:30, :]
    else:
        left_hand_quats = None
        right_hand_quats = None

    shaped = body_model.with_shape(torch.mean(betas, dim=1, keepdim=True))
    fk_outputs: fncsmpl.SmplhShapedAndPosed = shaped.with_pose_decomposed(
        T_world_root=T_world_root,
        body_quats=body_quats,
        left_hand_quats=left_hand_quats,
        right_hand_quats=right_hand_quats,
    )
    T_world_cpf = fncsmpl_extensions.get_T_world_cpf_from_root_pose(fk_outputs, T_world_root)

    denoised_ego_data = EgoTrainingData(
        T_world_root=T_world_root.squeeze(0).cpu(),
        contacts=denoised_traj.contacts.squeeze(0).cpu(),
        betas=denoised_traj.betas.squeeze(0).cpu(),
        joints_wrt_world=fk_outputs.Ts_world_joint.squeeze(0).cpu(),
        body_quats=SO3.from_matrix(denoised_traj.body_rotmats).wxyz.cpu()[:, :, :21, :].cpu().squeeze(0), #denoised_traj.body_quats.cpu(),
        T_world_cpf=T_world_cpf.cpu().squeeze(0),
        height_from_floor=T_world_cpf[..., 6:7].cpu().squeeze(0),
        T_cpf_tm1_cpf_t=(
            tf.SE3(T_world_cpf[:-1, :]).inverse() @ tf.SE3(T_world_cpf[1:, :])
        ).parameters().cpu().squeeze(0),
        joints_wrt_cpf=(
            # unsqueeze so both shapes are (timesteps, joints, dim)
            tf.SE3(T_world_cpf[0, 1:, None, :]).inverse()
            @ fk_outputs.Ts_world_joint[0, 1:, :21, 4:7].to(T_world_cpf.device)
        ),
        mask=torch.ones_like(denoised_traj.contacts[0, :], dtype=torch.bool),
        hand_quats=None,
        visible_joints_mask=None,
    )

    # Create ground truth EgoTrainingData
    gt_ego_data = EgoTrainingData(
        T_world_root=T_world_root.squeeze(0).cpu(),
        contacts=contacts.squeeze(0).cpu(),
        betas=betas.squeeze(0).cpu(),
        joints_wrt_world=posed.Ts_world_joint.squeeze(0).cpu(),
        body_quats=body_quats.squeeze(0).cpu(),
        T_world_cpf=Ts_world_cpf.squeeze(0).cpu(),
        height_from_floor=Ts_world_cpf[..., 6:7].squeeze(0).cpu(),
        T_cpf_tm1_cpf_t=(
            tf.SE3(Ts_world_cpf[:-1, :]).inverse() @ tf.SE3(Ts_world_cpf[1:, :])
        ).parameters().cpu().squeeze(0),
        joints_wrt_cpf=(
            # unsqueeze so both shapes are (timesteps, joints, dim)
        tf.SE3(Ts_world_cpf[1:, None, :]).inverse() @ posed.Ts_world_joint[0, 1:, :21, 4:7].to(Ts_world_cpf.device)
        ),
        mask=torch.ones_like(contacts[0, :], dtype=torch.bool),
        hand_quats=torch.cat([left_hand_quats, right_hand_quats], dim=-2).squeeze(0).cpu(),
        visible_joints_mask=None,
        visible_joints=None,
    )


    # Calculate metrics between original and inferred trajectories
    metrics = calculate_metrics(
        original_posed=posed,
        denoised_traj=denoised_traj,
        masked_data=masked_data,
        body_model=body_model
    )
    # Print metrics
    print("\nTrajectory Metrics:")
    print(f"Translation Error: {metrics['translation_error_meters']:.3f} meters")
    print(f"Rotation Error: {metrics['rotation_error_radians']:.3f} radians")
    print(f"Unmasked Joints MPJPE: {metrics['unmasked_mpjpe_mm']:.1f} mm")
    print(f"Masked Joints MPJPE: {metrics['masked_mpjpe_mm']:.1f} mm")



    # import ipdb; ipdb.set_trace()
    
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
    if config.visualize_traj:
        server = viser.ViserServer()
        server.gui.configure_theme(dark_mode=True)
        assert server is not None
        use_gt = False

        # Create timestamp for unique filenames
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Create output directories if they don't exist
        output_dir = Path("output_videos")
        output_dir.mkdir(exist_ok=True)
        
        # Generate unique filenames with timestamp
        gt_output_path = output_dir / f"gt_traj_{timestamp}.mp4"
        inferred_output_path = output_dir / f"inferred_traj_{timestamp}.mp4"
        
        # Save GT and inferred trajectories to separate files
        EgoTrainingData.visualize_ego_training_data(
            gt_ego_data, 
            body_model, 
            output_path=str(gt_output_path)
        )
        EgoTrainingData.visualize_ego_training_data(
            denoised_ego_data, 
            body_model, 
            output_path=str(inferred_output_path)
        )

        print(f"Saved ground truth video to: {gt_output_path}")
        print(f"Saved inferred video to: {inferred_output_path}")





if __name__ == "__main__":
    import tyro

    ipdb_safety_net()
    tyro.cli(main)
