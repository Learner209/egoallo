from __future__ import annotations
import dataclasses
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm

from egoallo import fncsmpl, fncsmpl_extensions
from egoallo.data.amass import AdaptiveAmassHdf5Dataset, EgoAmassHdf5Dataset
from egoallo.data.dataclass import collate_dataclass, EgoTrainingData
from egoallo.network import EgoDenoiser, EgoDenoiseTraj, EgoDenoiserConfig

from egoallo.config.inference.inference_defaults import InferenceConfig
from egoallo.evaluation.body_evaluator import BodyEvaluator
from egoallo.utils.setup_logger import setup_logger
from egoallo.inference_utils import load_denoiser, load_runtime_config, create_masked_training_data
from egoallo.transforms import SE3, SO3
from egoallo.sampling import run_sampling_with_masked_data, run_sampling_with_stitching
from egoallo import transforms as tf
from egoallo.training_utils import ipdb_safety_net

logger = setup_logger(output="logs/test", name=__name__)

def save_visualization(
    gt_ego_data: EgoTrainingData,
    denoised_ego_data: EgoTrainingData,
    body_model: fncsmpl.SmplhModel,
    output_dir: Path,
    timestamp: str
) -> Tuple[Path, Path]:
    """Save visualization of ground truth and inferred trajectories.
    
    Args:
        gt_ego_data: Ground truth ego training data
        denoised_ego_data: Denoised/inferred ego training data
        body_model: SMPL-H body model
        output_dir: Directory to save visualization files
        timestamp: Timestamp string for unique filenames
        
    Returns:
        Tuple containing paths to ground truth and inferred trajectory videos
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    gt_output_path = output_dir / f"gt_traj_{timestamp}.mp4"
    inferred_output_path = output_dir / f"inferred_traj_{timestamp}.mp4"
    
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
    
    return gt_output_path, inferred_output_path

def save_sequence_data(
    batch: EgoTrainingData,
    traj: EgoDenoiseTraj,
    seq_idx: int,
    body_model: fncsmpl.SmplhModel,
    output_path: Path
) -> None:
    """Save sequence data for evaluation.
    
    Args:
        batch: Batch of training data
        traj: Trajectory data
        seq_idx: Sequence index
        body_model: SMPL-H body model
        output_path: Path to save sequence data
    """
    # Get posed body model output
    posed = traj.apply_to_body(body_model)
    Ts_world_root = fncsmpl_extensions.get_T_world_root_from_cpf_pose(
        posed, batch.T_world_cpf[seq_idx:seq_idx+1, :, :]
    )
    
    # Save in format expected by body_evaluator.py
    torch.save({
        # Ground truth data
        "groundtruth_betas": batch.betas[seq_idx, :].cpu(),
        "groundtruth_T_world_root": batch.T_world_root[seq_idx, :].cpu(),
        "groundtruth_body_quats": batch.body_quats[seq_idx, :].cpu(),
        
        # Sampled/predicted data
        "sampled_betas": traj.betas[0].cpu(),
        "sampled_T_world_root": Ts_world_root[0].cpu(),
        "sampled_body_quats": posed.local_quats[0, ..., :21, :].cpu()
    }, output_path)
    
    logger.info(f"Saved sequence to {output_path}")

def process_sequence(
    batch: EgoTrainingData,
    seq_idx: int,
    denoiser_network: EgoDenoiser,
    body_model: fncsmpl.SmplhModel,
    device: torch.device,
    inference_config: InferenceConfig,
    model_config: EgoDenoiserConfig,
) -> Tuple[EgoTrainingData, EgoDenoiseTraj, EgoTrainingData]:
    """Process a single sequence to generate predictions.
    
    Args:
        batch: Input batch containing sequences
        seq_idx: Index of sequence to process
        denoiser_network: Denoising network
        body_model: SMPL-H body model
        device: Device to run computation on
        config: Test configuration
        
    Returns:
        Tuple containing (gt_ego_data, traj, denoised_ego_data)
    """
    # Prepare input tensors
    # import ipdb; ipdb.set_trace()

    Ts_world_cpf = batch.T_world_cpf[seq_idx:seq_idx+1, :].to(device)
    Ts_world_root = batch.T_world_root[seq_idx:seq_idx+1, :].to(device)
    body_quats = batch.body_quats[seq_idx:seq_idx+1, :].to(device)
    left_hand_quats = batch.hand_quats[..., :15, :].to(device)
    right_hand_quats = batch.hand_quats[..., 15:30, :].to(device)
    contacts = batch.contacts[seq_idx:seq_idx+1, :].to(device)
    betas = batch.betas[seq_idx:seq_idx+1, :].to(device)

    # Create posed data
    local_quats = torch.cat([body_quats, left_hand_quats, right_hand_quats], dim=-2)
    shaped_model = body_model.with_shape(betas)
    posed = shaped_model.with_pose(Ts_world_root, local_quats)

    
    # Create ground truth EgoTrainingData
    gt_ego_data = EgoTrainingData(
        T_world_root=batch.T_world_root[seq_idx:seq_idx+1, :].squeeze(0).cpu(),
        contacts=batch.contacts[seq_idx:seq_idx+1, :].squeeze(0).cpu(),
        betas=batch.betas[seq_idx:seq_idx+1, :].squeeze(0).cpu(),
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
        mask=torch.ones_like(batch.contacts[seq_idx:seq_idx+1, :].squeeze(0), dtype=torch.bool),
        hand_quats=batch.hand_quats.squeeze(0).cpu() if batch.hand_quats is not None else None,
        visible_joints_mask=None,
    )

    # Create masked training data
    masked_data = create_masked_training_data(
        posed=posed,
        Ts_world_cpf=Ts_world_cpf,
        contacts=contacts,
        betas=betas,
        mask_ratio=model_config.mask_ratio
    )

    # Run sampling with masked data
    denoised_traj: EgoDenoiseTraj = run_sampling_with_masked_data(
        denoiser_network=denoiser_network,
        body_model=body_model,
        masked_data=masked_data,
        guidance_mode="no_hands",
        guidance_post=inference_config.guidance_post,
        guidance_inner=inference_config.guidance_inner,
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

    return gt_ego_data, denoised_traj, denoised_ego_data
def main(inference_config: InferenceConfig) -> None:
    device = torch.device(inference_config.device)
    
    # Initialize model and dataset
    try:
        denoiser_network, model_config = load_denoiser(inference_config.checkpoint_dir)
        denoiser_network = denoiser_network.to(device)
        runtime_config = load_runtime_config(inference_config.checkpoint_dir)
        body_model = fncsmpl.SmplhModel.load(runtime_config.smplh_npz_path).to(device)
        
        runtime_config.dataset_slice_strategy = "full_sequence"
        test_dataset = EgoAmassHdf5Dataset(runtime_config, cache_files=False)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            collate_fn=collate_dataclass,
            drop_last=False
        )
    except Exception as e:
        logger.error(f"Failed to initialize model/dataset: {str(e)}")
        raise

    # Create output directories
    assert inference_config.output_dir is not None
    output_dir = Path(inference_config.output_dir)
    
    # Clear output directory if it exists
    if output_dir.exists():
        for pt_file in output_dir.glob("*.pt"):
            pt_file.unlink()
    
    output_dir.mkdir(exist_ok=True, parents=True)

    # Test loop 
    for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Generating predictions")):
        batch = batch.to(device)
        if batch_idx == 5:
            break
        
        for seq_idx in range(batch.T_world_cpf.shape[0]):
            # try:
            gt_ego_data, denoised_traj, denoised_ego_data = process_sequence(
                batch, seq_idx, denoiser_network, body_model, device, inference_config, model_config
            )
            
            # Save visualizations if requested
            if inference_config.visualize_traj:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                gt_path, inferred_path = save_visualization(
                    gt_ego_data,
                    denoised_ego_data,
                    body_model,
                    output_dir,
                    timestamp
                )
                logger.info(f"Saved ground truth video to: {gt_path}")
                logger.info(f"Saved inferred video to: {inferred_path}")

            # Save sequence data
            output_path = inference_config.output_dir / f"sequence_{batch_idx}_{seq_idx}.pt"
            save_sequence_data(batch, denoised_traj, seq_idx, body_model, output_path)

            # except Exception as e:
            #     logger.error(f"Error processing sequence {batch_idx}_{seq_idx}: {str(e)}")
            #     continue

    # Compute metrics if requested
    if inference_config.compute_metrics:
        logger.info("\nComputing evaluation metrics...")
        try:
            evaluator = BodyEvaluator(
                body_model_path=runtime_config.smplh_npz_path,
                device=device
            )
            
            evaluator.evaluate_directory(
                dir_with_pt_files=inference_config.output_dir,
                use_mean_body_shape=inference_config.use_mean_body_shape,
                skip_confirm=inference_config.skip_eval_confirm
            )
        except Exception as e:
            logger.error(f"Error computing metrics: {str(e)}")

if __name__ == "__main__":
    import tyro
    ipdb_safety_net()
    
    tyro.cli(main)
