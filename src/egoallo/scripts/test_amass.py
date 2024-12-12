from __future__ import annotations
import dataclasses
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
from jaxtyping import jaxtyped
import typeguard

# with install_import_hook("egoallo", "typeguard.typechecked"):

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
    traj: EgoTrainingData,
    seq_idx: int,
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
    # Save in format expected by body_evaluator.py
    assert batch.check_shapes(traj), f"Shapes of batch and trajectory do not match, shape is {batch.get_batch_size()} against {traj.get_batch_size()}"

    torch.save({
        # Ground truth data
        "groundtruth_betas": batch.betas[seq_idx, :].cpu(),
        "groundtruth_T_world_root": batch.T_world_root[seq_idx, :].cpu(),
        "groundtruth_body_quats": batch.body_quats[seq_idx, ..., :21, :].cpu(),

        # Sampled/predicted data
        "sampled_betas": traj.betas[0].cpu(),
        "sampled_T_world_root": traj.T_world_root[0].cpu(),
        "sampled_body_quats": traj.body_quats[0, ..., :21, :].cpu()
    }, output_path)
    breakpoint()

    logger.info(f"Saved sequence to {output_path}")

def prepare_tensors(batch: EgoTrainingData, seq_idx: int, device: torch.device) -> Dict[str, torch.Tensor]:
    """Prepare input tensors with consistent shapes."""
    # Add batch dimension and move to device
    get_tensor = lambda x: x[seq_idx:seq_idx+1].to(device) if x is not None else None

    return {
        'Ts_world_cpf': get_tensor(batch.T_world_cpf), # shape: (1, timesteps, 7)
        'Ts_world_root': get_tensor(batch.T_world_root), # shape: (1, 7)
        'body_quats': get_tensor(batch.body_quats), # shape: (1, timesteps, 21, 4)
        'hand_quats': get_tensor(batch.hand_quats) if batch.hand_quats is not None else None, # shape: (1, timesteps, 30, 4)
        'contacts': get_tensor(batch.contacts), # shape: (1, timesteps, 21)
        'betas': get_tensor(batch.betas) # shape: (1, 1, 16)
    }

def create_ego_data(tensors: Dict[str, torch.Tensor], posed_model: fncsmpl.SmplhShapedAndPosed, device: torch.device, is_gt: bool = True) -> EgoTrainingData:
    """Factory method to create EgoTrainingData instances."""
    # Get base tensors
    Ts_world_cpf = tensors['Ts_world_cpf']
    T_world_root = tensors['Ts_world_root'] if is_gt else tensors['T_world_root']

    # breakpoint()
    joints_wrt_cpf = (
        tf.SE3(Ts_world_cpf[1:, None, :]).inverse()
        @ posed_model.Ts_world_joint[1:, :21, 4:7].to(device)
    )

    return EgoTrainingData(
        T_world_root=T_world_root.cpu(),
        contacts=tensors['contacts'].cpu(),
        betas=tensors['betas'].cpu(),
        joints_wrt_world=posed_model.Ts_world_joint.cpu(),
        body_quats=tensors['body_quats'].cpu(),
        T_world_cpf=Ts_world_cpf.cpu(),
        height_from_floor=Ts_world_cpf[..., 6:7].cpu(),
        joints_wrt_cpf=joints_wrt_cpf,
        mask=torch.ones_like(tensors['contacts'], dtype=torch.bool),
        hand_quats=tensors.get('hand_quats', None),
        visible_joints_mask=None
    )

def process_sequence(batch: EgoTrainingData, seq_idx: int, denoiser_network: EgoDenoiser, body_model: fncsmpl.SmplhModel, device: torch.device, inference_config: InferenceConfig, model_config: EgoDenoiserConfig) -> Tuple[EgoTrainingData, EgoDenoiseTraj, EgoTrainingData]:
    """Process sequence with simplified tensor handling."""
    # Prepare input tensors
    # tensors = prepare_tensors(batch, seq_idx, device)

    # # Create posed data
    # local_quats = torch.cat([
    #     tensors['body_quats'],
    #     tensors['hand_quats'],
    # ], dim=-2)

    # posed = body_model.with_shape(tensors['betas'].mean(dim=0)).with_pose(
    #     tensors['Ts_world_root'], local_quats
    # )

    # Create ground truth data
    gt_ego_data = batch

    # Create masked training data
    # masked_data = create_masked_training_data(
    #     posed=posed,
    #     Ts_world_cpf=tensors['Ts_world_cpf'],
    #     contacts=tensors['contacts'],
    #     betas=tensors['betas'],
    #     mask_ratio=model_config.mask_ratio
    # )

    # Create denoised trajectory (debug version)
    denoised_traj = EgoDenoiseTraj(
        R_world_root=SO3(gt_ego_data.T_world_root[..., :4]).as_matrix(),
        t_world_root=gt_ego_data.T_world_root[..., 4:],
        betas=gt_ego_data.betas.unsqueeze(0),
        body_rotmats=SO3(gt_ego_data.body_quats).as_matrix(),
        contacts=gt_ego_data.contacts.unsqueeze(0),
        hand_rotmats=SO3(gt_ego_data.hand_quats).as_matrix() if gt_ego_data.hand_quats is not None else None
    ).to(device)

    # Prepare denoised tensors
    # denoised_tensors = {
    #     'T_world_root': SE3.from_rotation_and_translation(
    #         SO3.from_matrix(denoised_traj.R_world_root),
    #         denoised_traj.t_world_root
    #     ).parameters(),
    #     'contacts': denoised_traj.contacts,
    #     'betas': denoised_traj.betas,
    #     'body_quats': SO3.from_matrix(denoised_traj.body_rotmats).wxyz,
    #     'Ts_world_cpf': gt_ego_data.Ts_world_cpf['Ts_world_cpf'],
    #     'left_hand_quats': SO3.from_matrix(denoised_traj.hand_rotmats).wxyz[:, :15, :],
    #     'right_hand_quats': SO3.from_matrix(denoised_traj.hand_rotmats).wxyz[:, 15:30, :]
    # }

    # # Create final posed model
    # shaped = body_model.with_shape(denoised_tensors['betas'].mean(dim=0))
    # fk_outputs = shaped.with_pose_decomposed(
    #     T_world_root=denoised_tensors['T_world_root'],
    #     body_quats=denoised_tensors['body_quats'],
    #     left_hand_quats=denoised_tensors['left_hand_quats'],
    #     right_hand_quats=denoised_tensors['right_hand_quats']
    # )

    # denoised_tensors['Ts_world_cpf'] = fncsmpl_extensions.get_T_world_cpf_from_root_pose(
    #     fk_outputs, denoised_tensors['T_world_root']
    # )
    # # breakpoint()

    # Create denoised ego data
    # denoised_ego_data = create_ego_data(denoised_tensors, fk_outputs, device, is_gt=False)

    # TODO: return gt_ego_data for denoised-traj just for debuggin right now.
    return gt_ego_data, denoised_traj, gt_ego_data

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
            num_workers=0,
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
            save_sequence_data(batch, denoised_ego_data, seq_idx, output_path)

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
