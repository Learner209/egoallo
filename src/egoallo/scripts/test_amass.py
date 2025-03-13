from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# from egoallo import fncsmpl, fncsmpl_extensions
from egoallo import fncsmpl_library as fncsmpl
from egoallo import fncsmpl_extensions_library as fncsmpl_extensions
from egoallo.data.amass_dataset import EgoAmassHdf5Dataset
from egoallo.data.dataclass import collate_dataclass
from egoallo.training_utils import ipdb_safety_net
from egoallo.config.test_config import TestConfig
from egoallo.evaluation.body_evaluator import BodyEvaluator
from egoallo.setup_logger import setup_logger
from egoallo.inference_utils import load_denoiser
from egoallo.sampling import (
    run_sampling_with_stitching,
)

logger = setup_logger(output="logs/test", name=__name__)


def main(config: TestConfig):
    device = torch.device(config.device)

    if config.use_ipdb:
        import builtins

        builtins.breakpoint()

    # Initialize model and body model
    denoiser_network = load_denoiser(config.checkpoint_dir).to(device)
    body_model = fncsmpl.SmplhModel.load(config.smplh_model_path, use_pca=False).to(
        device,
    )

    # Initialize test dataset
    test_dataset = EgoAmassHdf5Dataset(config)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_dataclass,
        drop_last=False,
    )

    # Test loop
    for batch_idx, batch in enumerate(
        tqdm(test_dataloader, desc="Generating predictions"),
    ):
        # Move batch to device
        batch = batch.to(device)

        # Generate motion using sampling for each sequence in batch
        for seq_idx in range(batch.T_world_cpf.shape[0]):
            # Generate motion using sampling
            with torch.no_grad():
                traj = run_sampling_with_stitching(
                    denoiser_network,
                    body_model=body_model,
                    guidance_mode=config.guidance_mode,
                    guidance_inner=config.guidance_inner,
                    guidance_post=config.guidance_post,
                    Ts_world_cpf=batch.T_world_cpf[
                        seq_idx
                    ],  # Process one sequence at a time
                    num_samples=config.num_samples,
                    device=device,
                    # Set floor_z to 0.0 if not needed for AMASS testing
                    hamer_detections=None,
                    aria_detections=None,
                    floor_z=0.0,
                )

            # Create output filename
            output_path = config.output_dir / f"sequence_{batch_idx}_{seq_idx}"

            # Get posed body model output
            posed = traj.apply_to_body(body_model)
            Ts_world_root = fncsmpl_extensions.get_T_world_root_from_cpf_pose(
                posed,
                batch.T_world_cpf[seq_idx : seq_idx + 1, 1:, :],  # Keep batch dim
            )

            # Update the output path to use .pt extension
            output_path = output_path.with_suffix(".pt")

            # Save in format expected by body_evaluator.py
            torch.save(
                {
                    # Ground truth data
                    "groundtruth_betas": batch.betas[seq_idx, 1:].cpu(),
                    "groundtruth_T_world_root": batch.T_world_root[seq_idx, 1:].cpu(),
                    "groundtruth_body_quats": batch.body_quats[seq_idx, 1:].cpu(),
                    # Sampled/predicted data
                    "sampled_betas": traj.betas[
                        0
                    ].cpu(),  # Remove batch dim since traj has batch size 1
                    "sampled_T_world_root": Ts_world_root[0].cpu(),  # Remove batch dim
                    "sampled_body_quats": posed.local_quats[
                        0,
                        ...,
                        :21,
                        :,
                    ].cpu(),  # Remove batch dim
                },
                output_path,
            )
            logger.info(f"Saved sequence to {output_path}")

    # Compute metrics if requested
    if config.compute_metrics:
        logger.info("\nComputing evaluation metrics...")
        evaluator = BodyEvaluator(
            body_model_path=config.smplh_model_path,
            device=device,
        )

        evaluator.evaluate_directory(
            dir_with_pt_files=config.output_dir,
            use_mean_body_shape=config.use_mean_body_shape,
            skip_confirm=config.skip_eval_confirm,
        )


if __name__ == "__main__":
    import tyro

    ipdb_safety_net()

    config = tyro.cli(TestConfig)
    main(config)
