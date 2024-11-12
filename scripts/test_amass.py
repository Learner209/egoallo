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
from egoallo.training_pipeline import EgoAlloTrainConfig

@dataclasses.dataclass(frozen=False)
class TestConfig(EgoAlloTrainConfig):
    """Test configuration that extends training config."""
    # Model and checkpoint settings
    checkpoint_dir: Path = Path("./experiments/nov12_first/v0/checkpoints/checkpoint-15000")
    smplh_npz_path: Path = Path("./data/smplh/neutral/model.npz")
    
    # Test settings
    num_inference_steps: int = 50
    guidance_scale: float = 3.0
    num_samples: int = 1
    
    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: Literal["float16", "float32"] = "float16" if torch.cuda.is_available() else "float32"
    
    # Output settings
    output_dir: Path = Path("./test_results")
    
    def __post_init__(self):
        # Override train splits with test splits
        self.train_splits = ("test",)
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)


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
        collate_fn=collate_dataclass
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
            output_path = config.output_dir / f"sequence_{batch_idx}_{i}.npz"
            
            # Save in format expected by body_evaluator.py
            torch.save({
                # Ground truth data
                "groundtruth_betas": batch.betas[i].cpu(),
                "groundtruth_T_world_root": batch.T_world_root[i].cpu(),
                "groundtruth_body_quats": batch.body_quats[i].cpu(),
                
                # Sampled/predicted data
                "sampled_betas": pred_motion.betas[i].cpu(),
                "sampled_T_world_root": pred_motion.T_world_root[i].cpu(),
                "sampled_body_quats": pred_motion.body_quats[i].cpu()
            }, output_path)
            print(f"Saved sequence to {output_path}")

if __name__ == "__main__":
    import tyro
    ipdb_safety_net()
    
    config = tyro.cli(TestConfig)
    main(config) 