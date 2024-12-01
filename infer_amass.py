from __future__ import annotations
import sys
from typing import Literal

import numpy as np
import viser
import torch
from pathlib import Path
import time
import tyro
from egoallo import transforms as tf

from egoallo.data.dataclass import EgoTrainingData
from egoallo.inference_utils import (
    create_masked_training_data,
    load_denoiser,
    EgoDenoiseTraj
)
from egoallo import fncsmpl
from egoallo import fncsmpl_extensions
from egoallo.vis_helpers import visualize_traj_and_hand_detections
from egoallo.sampling import CosineNoiseScheduleConstants, quadratic_ts
from egoallo.data.amass import EgoAmassHdf5Dataset
from egoallo.data.dataclass import collate_dataclass
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
from egoallo.training_utils import ipdb_safety_net
from infer_mae import run_sampling_with_masked_data, calculate_metrics
from config.train import EgoAlloTrainConfig
from egoallo.transforms import SO3, SE3
import dataclasses


@dataclasses.dataclass
class Args(EgoAlloTrainConfig):
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
    guidance_inner: bool = False
    """Whether to apply guidance optimizer between denoising steps. This is
    important if we're doing anything with hands. It can be turned off to speed
    up debugging/experiments, or if we only care about foot skating losses."""
    guidance_post: bool = False
    """Whether to apply guidance optimizer after diffusion sampling."""
    visualize_traj: bool = True
    """Whether to visualize the trajectory after sampling."""
    mask_ratio: float = 0.75
    """Ratio of joints to mask."""
    dataset_slice_strategy: Literal["deterministic", "random_uniform_len", "random_variable_len", "full_sequence"] = "full_sequence"
    """Strategy for slicing the dataset into subsequences."""

def main(
    config: Args,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> network.EgoDenoiseTraj:

    # Load data and models
    dataset = EgoAmassHdf5Dataset(
        config=config,
        cache_files=True,
        random_variable_len_proportion=config.dataset_slice_random_variable_len_proportion,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_dataclass,
    )
    for test_batch in test_loader:
        body_model = fncsmpl.SmplhModel.load(config.smplh_npz_path).to(device)
        denoiser_network = load_denoiser(config.checkpoint_dir).to(device)
        traj_data: EgoTrainingData = test_batch

        # Prepare input tensors
        # import ipdb; ipdb.set_trace()

        Ts_world_cpf = traj_data.T_world_cpf.to(device)
        Ts_world_root = traj_data.T_world_root.to(device)
        body_quats = traj_data.body_quats.to(device)
        left_hand_quats = traj_data.hand_quats[..., :15, :].to(device)
        right_hand_quats = traj_data.hand_quats[..., 15:30, :].to(device)
        contacts = traj_data.contacts.to(device)
        betas = traj_data.betas.to(device)

        # Create posed data
        local_quats = torch.cat([body_quats, left_hand_quats, right_hand_quats], dim=-2)
        shaped_model = body_model.with_shape(betas)
        posed = shaped_model.with_pose(Ts_world_root, local_quats)

        
        # Create ground truth EgoTrainingData
        gt_ego_data = EgoTrainingData(
            T_world_root=traj_data.T_world_root.squeeze(0).cpu(),
            contacts=traj_data.contacts.squeeze(0).cpu(),
            betas=traj_data.betas.squeeze(0).cpu(),
            joints_wrt_world=posed.Ts_world_joint.squeeze(0).cpu(),
            body_quats=body_quats.squeeze(0).cpu(),
            T_world_cpf=Ts_world_cpf.squeeze(0).cpu(),
            height_from_floor=Ts_world_cpf[..., 6:7].squeeze(0).cpu(),
            T_cpf_tm1_cpf_t=(
                tf.SE3(Ts_world_cpf[:-1, :]).inverse() @ tf.SE3(Ts_world_cpf[1:, :])
            ).parameters().cpu().squeeze(0),
            joints_wrt_cpf=(
                # unsqueeze so both shapes are (timesteps, joints, dim)
            tf.SE3(Ts_world_cpf[0, 1:, None, :]).inverse()
                @ posed.Ts_world_joint[0, 1:, :21, 4:7].to(Ts_world_cpf.device)
            ),
            mask=torch.ones_like(traj_data.contacts[0, :], dtype=torch.bool),
            hand_quats=traj_data.hand_quats.squeeze(0).cpu() if traj_data.hand_quats is not None else None,
            visible_joints_mask=None,
        )

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
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            
            # Create output directories if they don't exist
            output_dir = Path("output_videos_absrel")
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
