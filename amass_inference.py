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
from egoallo.transforms import SE3, SO3
from egoallo.training_utils import ipdb_safety_net
from inference_mae import run_sampling_with_masked_data, calculate_metrics
from train_motion_prior import EgoAlloTrainConfig
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
def main(
    config: Args,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> network.EgoDenoiseTraj:

    # Load data and models
    dataset = EgoAmassHdf5Dataset(
        config.dataset_hdf5_path,
        config.dataset_files_path,
        splits=config.train_splits,
        subseq_len=config.subseq_len,
        cache_files=True,
        slice_strategy=config.dataset_slice_strategy,
        random_variable_len_proportion=config.dataset_slice_random_variable_len_proportion,
        config=config.model,
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

        # Create masked training data
        masked_data = create_masked_training_data(
            posed=posed,
            Ts_world_cpf=Ts_world_cpf,
            contacts=contacts,
            betas=betas,
            mask_ratio=config.mask_ratio
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
            gt_traj = network.EgoDenoiseTraj.unpack(
                x=torch.cat([betas,
                             SO3(body_quats).as_matrix().reshape(*body_quats.shape[:-2], 21 * 9),
                             contacts,
                             Ts_world_root,
                             SO3(left_hand_quats).as_matrix().reshape(*left_hand_quats.shape[:-2], 15 * 9),
                             SO3(right_hand_quats).as_matrix().reshape(*right_hand_quats.shape[:-2], 15 * 9)], dim=-1),
                include_hands=True,
                project_rotmats=False
            )
            gt_T_world_root = traj_data.T_world_root.squeeze(0).to(device)
            # import ipdb; ipdb.set_trace()

            server = viser.ViserServer()
            server.gui.configure_theme(dark_mode=True)
            assert server is not None
            loop_cb = visualize_traj_and_hand_detections(
                server,
                gt_T_world_root,
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

    return


if __name__ == "__main__":
    import tyro

    ipdb_safety_net()
    tyro.cli(main)
