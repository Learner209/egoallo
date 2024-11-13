from __future__ import annotations
import dataclasses
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from egoallo import fncsmpl
from egoallo.data.amass_dataset_dynamic import EgoAmassHdf5DatasetDynamic
from egoallo.data.dataclass import collate_dataclass
from egoallo.motion_diffusion_pipeline import MotionDiffusionPipeline
from egoallo.network import EgoDenoiseTraj
from egoallo.training_utils import ipdb_safety_net
from egoallo.config.test_config import TestConfig
from egoallo.evaluation.body_evaluator import BodyEvaluator
from egoallo.setup_logger import setup_logger

logger = setup_logger(output="logs/test", name=__name__)

def main(config: TestConfig):
    device = torch.device(config.device)
    
    if config.use_ipdb:
        import ipdb; ipdb.set_trace()
        
    # Initialize model and pipeline
    pipeline = MotionDiffusionPipeline.from_pretrained(
        str(config.checkpoint_dir),
        use_safetensors=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    ).to(device)
    
    # Initialize test dataset
    test_dataset = EgoAmassHdf5DatasetDynamic(config)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_dataclass,
        drop_last=False
    )
    
    # Test loop
    for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Generating predictions")):
        # Move batch to device
        batch = batch.to(device)
        
        # Generate motion using pipeline
        with torch.no_grad():
            pred_motion = pipeline(
                batch_size=config.batch_size,
                num_inference_steps=config.num_inference_steps,
                train_batch=batch,
                return_intermediates=False
            )
        
        # Get predicted motion
        pred_motion: EgoDenoiseTraj = pred_motion.motion

        # Save ground truth and predictions for each item in batch
        for i in range(batch.betas.shape[0]):
            # Create output filename
            output_path = config.output_dir / f"sequence_{batch_idx}_{i}"

            body_quats = pred_motion.get_body_quats()
            T_world_root = pred_motion.get_T_world_root(
                body_model=pipeline.unet.smpl_model,
                Ts_world_cpf=batch.T_world_cpf[..., 1:, :]
            )
            
            # Update the output path to use .pt extension
            output_path = output_path.with_suffix('.pt')
            
            # Save in format expected by body_evaluator.py
            torch.save({
                # Ground truth data
                "groundtruth_betas": batch.betas[i, 1:].cpu(),
                "groundtruth_T_world_root": batch.T_world_root[i, 1:].cpu(),
                "groundtruth_body_quats": batch.body_quats[i, 1:].cpu(),
                
                # Sampled/predicted data
                "sampled_betas": pred_motion.betas[i].cpu(),
                "sampled_T_world_root": T_world_root[i].cpu(),
                "sampled_body_quats": body_quats[i].cpu()
            }, output_path)
            logger.info(f"Saved sequence to {output_path}")

    # Compute metrics if requested
    if config.compute_metrics:
        logger.info("\nComputing evaluation metrics...")
        evaluator = BodyEvaluator(
            body_model_path=config.smplh_npz_path,
            device=device
        )
        
        evaluator.evaluate_directory(
            dir_with_pt_files=config.output_dir,
            use_mean_body_shape=config.use_mean_body_shape,
            skip_confirm=config.skip_eval_confirm
        )

if __name__ == "__main__":
    import tyro
    import ipdb; ipdb.set_trace()
    ipdb_safety_net()
    
    config = tyro.cli(TestConfig)
    main(config) 