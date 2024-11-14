from __future__ import annotations
import dataclasses
from pathlib import Path
from typing import Literal, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from egoallo import fncsmpl
from egoallo.data.rich.rich_dataset import RICHDataProcessor
from egoallo.data.rich.rich_dataset_config import RICHDatasetConfig
from egoallo.data.dataclass import collate_dataclass
from egoallo.network import EgoDenoiseTraj
from egoallo.training_utils import ipdb_safety_net
from egoallo.evaluation.body_evaluator import BodyEvaluator
# from egoallo.evaluation.contact_evaluator import ContactEvaluator
from egoallo.setup_logger import setup_logger

logger = setup_logger(output="logs/test_rich", name=__name__)

@dataclasses.dataclass
class RICHTestConfig:
    """Configuration for testing on RICH dataset."""
    # Dataset paths and options
    rich_data_dir: Path = Path("./third_party/rich_toolkit")
    smplx_model_dir: Path = Path("./third_party/rich_toolkit/body_models/smplx")
    output_dir: Path = Path("./data/rich/processed_data")
    # checkpoint_dir: Path = Path("./checkpoints/egoallo")
    # smplh_npz_path: Path = Path("./third_party/smplh/SMPLH_MALE.npz")
    
    # Testing options
    device: str = "cuda"
    batch_size: int = 32
    num_workers: int = 4
    num_inference_steps: int = 1000
    
    # Evaluation options
    compute_metrics: bool = True
    compute_contact_metrics: bool = True
    use_mean_body_shape: bool = False
    skip_eval_confirm: bool = True
    
    # Debug options
    use_ipdb: bool = False
    debug: bool = False

def main(config: RICHTestConfig):
    device = torch.device(config.device)
    
    if config.use_ipdb:
        import ipdb; ipdb.set_trace()
    
    # Process RICH dataset
    rich_config = RICHDatasetConfig(
        rich_data_dir=config.rich_data_dir,
        smplx_model_dir=config.smplx_model_dir,
        output_dir=config.output_dir
    )
    
    processor = RICHDataProcessor(
        rich_data_dir=str(rich_config.rich_data_dir),
        smplx_model_dir=str(rich_config.smplx_model_dir),
        output_dir=str(rich_config.output_dir),
        include_contact=config.compute_contact_metrics,
        device=device
    )
    
    # Process dataset and generate predictions
    processor.process_dataset(split='test')
    
    # Load processed sequences
    # sequences = sorted(config.output_dir.glob("rich_sequence_*.pt"))
    
    # for seq_path in tqdm(sequences, desc="Generating predictions"):
    #     # Load sequence data
    #     seq_data = torch.load(seq_path, map_location=device)
        
    #     # Prepare batch
    #     batch = dataclasses.replace(
    #         EgoDenoiseTraj(
    #             betas=seq_data['groundtruth_betas'].unsqueeze(0),
    #             body_quats=seq_data['groundtruth_body_quats'].unsqueeze(0),
    #             T_world_root=seq_data['groundtruth_T_world_root'].unsqueeze(0)
    #         )
    #     )
        
    #     # Generate motion using pipeline
    #     with torch.no_grad():
    #         pred_motion = pipeline(
    #             batch_size=1,
    #             num_inference_steps=config.num_inference_steps,
    #             train_batch=batch,
    #             return_intermediates=False
    #         )
        
    #     # Get predicted motion
    #     pred_motion: EgoDenoiseTraj = pred_motion.motion
        
    #     # Update predictions in sequence data
    #     seq_data.update({
    #         'sampled_betas': pred_motion.betas[0].cpu(),
    #         'sampled_body_quats': pred_motion.get_body_quats()[0].cpu(),
    #         'sampled_T_world_root': pred_motion.get_T_world_root(
    #             body_model=pipeline.unet.smpl_model,
    #             Ts_world_cpf=batch.T_world_cpf[..., 1:, :]
    #         )[0].cpu()
    #     })
        
    #     # Save updated sequence
    #     torch.save(seq_data, seq_path)
    #     logger.info(f"Processed sequence {seq_path.name}")

    # # Compute metrics
    # if config.compute_metrics:
    #     logger.info("\nComputing evaluation metrics...")
        
    #     # Body motion metrics
    #     body_evaluator = BodyEvaluator(
    #         body_model_path=config.smplh_npz_path,
    #         device=device
    #     )
        
    #     body_metrics = body_evaluator.evaluate_directory(
    #         dir_with_pt_files=config.output_dir,
    #         use_mean_body_shape=config.use_mean_body_shape,
    #         skip_confirm=config.skip_eval_confirm
    #     )
        
    #     # Contact metrics if requested
    #     if config.compute_contact_metrics:
    #         contact_evaluator = ContactEvaluator(device=device)
    #         contact_metrics = contact_evaluator.evaluate_directory(
    #             dir_with_pt_files=config.output_dir,
    #             skip_confirm=config.skip_eval_confirm
    #         )
            
    #         # Combine metrics
    #         all_metrics = {**body_metrics, **contact_metrics}
    #         logger.info(f"Final metrics: {all_metrics}")
    #     else:
    #         logger.info(f"Body motion metrics: {body_metrics}")

if __name__ == "__main__":
    import tyro
    ipdb_safety_net()
    
    config = tyro.cli(RICHTestConfig)
    main(config) 