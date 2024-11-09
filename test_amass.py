from __future__ import annotations
from collections import defaultdict
import dataclasses
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Literal

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from egoallo import fncsmpl
from egoallo.data.amass import AMASS_SPLITS
from egoallo.data.amass_dataset_dynamic import EgoAmassHdf5DatasetDynamic
from egoallo.data.dataclass import EgoTrainingData, collate_dataclass
from egoallo.motion_diffusion_pipeline import MotionDiffusionPipeline
from egoallo.network import EgoDenoiseTraj
from egoallo.training_utils import ipdb_safety_net
from training_pipeline import EgoAlloTrainConfig

@dataclasses.dataclass(frozen=False)
class TestConfig(EgoAlloTrainConfig):
    """Test configuration that extends training config."""
    # Model and checkpoint settings
    checkpoint_dir: Path = Path("experiments/nov8_v1/v1/checkpoint-10000")
    smplh_npz_path: Path = Path("./data/smplh/neutral/model.npz")
    
    # Test settings
    visualize: bool = False
    save_results: bool = True
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

def compute_metrics(
    pred_motion: EgoTrainingData,
    gt_motion: EgoTrainingData,
    body_model: fncsmpl.SmplhModel,
) -> Dict[str, float]:
    """Compute metrics between predicted and ground truth motions."""
    # Forward kinematics to get joint positions and rotations
    # First apply shape and pose to get SmplhShapedAndPosed instances
    pred_shaped = body_model.with_shape(pred_motion.betas)
    gt_shaped = body_model.with_shape(gt_motion.betas)
    
    pred_posed = pred_shaped.with_pose(pred_motion.T_world_root, pred_motion.local_quats)
    gt_posed = gt_shaped.with_pose(gt_motion.T_world_root, gt_motion.local_quats)
    
    # Get meshes with LBS
    pred_mesh = pred_posed.lbs()
    gt_mesh = gt_posed.lbs()

    # Compute metrics
    metrics = {}
    
    # Root metrics
    root_dist = torch.norm(pred_motion.T_world_root[..., 4:] - gt_motion.T_world_root[..., 4:], dim=-1).mean()
    root_rot_dist = torch.norm(pred_motion.T_world_root[..., :4] - gt_motion.T_world_root[..., :4], dim=-1).mean() 
    root_trans_dist = torch.norm(pred_motion.T_world_root[..., 4:] - gt_motion.T_world_root[..., 4:], dim=-1).mean()
    
    metrics['root_dist'] = root_dist.item()
    metrics['root_rot_dist'] = root_rot_dist.item()
    metrics['root_trans_dist'] = root_trans_dist.item()

    # Head metrics (using joint indices from eval_stage2.py)
    head_idx = 15  # Head joint index
    pred_head_pos = pred_posed.Ts_world_joint[..., head_idx, 4:]
    gt_head_pos = gt_posed.Ts_world_joint[..., head_idx, 4:]
    pred_head_rot = pred_posed.Ts_world_joint[..., head_idx, :4]
    gt_head_rot = gt_posed.Ts_world_joint[..., head_idx, :4]
    
    head_dist = torch.norm(pred_head_pos - gt_head_pos, dim=-1).mean()
    head_rot_dist = torch.norm(pred_head_rot - gt_head_rot, dim=-1).mean()
    head_trans_dist = torch.norm(pred_head_pos - gt_head_pos, dim=-1).mean()
    
    metrics['head_dist'] = head_dist.item()
    metrics['head_rot_dist'] = head_rot_dist.item() 
    metrics['head_trans_dist'] = head_trans_dist.item()

    # MPJPE metrics
    mpjpe = torch.norm(pred_posed.Ts_world_joint[..., 4:] - gt_posed.Ts_world_joint[..., 4:], dim=-1).mean()
    metrics['mpjpe'] = mpjpe.item()
    
    # MPJPE without hands (excluding hand joints)
    hand_joint_indices = list(range(20, 25)) + list(range(45, 50))
    non_hand_mask = torch.ones(pred_posed.Ts_world_joint.shape[-2], dtype=torch.bool)
    non_hand_mask[hand_joint_indices] = False
    
    mpjpe_wo_hand = torch.norm(
        pred_posed.Ts_world_joint[..., non_hand_mask, 4:] - 
        gt_posed.Ts_world_joint[..., non_hand_mask, 4:], 
        dim=-1
    ).mean()
    metrics['mpjpe_wo_hand'] = mpjpe_wo_hand.item()

    # Per-joint position error
    single_jpe = torch.norm(
        pred_posed.Ts_world_joint[..., 4:] - gt_posed.Ts_world_joint[..., 4:],
        dim=-1
    ).mean(0)
    metrics['single_jpe'] = single_jpe.tolist()

    # Acceleration metrics
    pred_accel = torch.diff(pred_posed.Ts_world_joint[..., 4:], n=2, dim=0)
    gt_accel = torch.diff(gt_posed.Ts_world_joint[..., 4:], n=2, dim=0)
    
    accel_pred = torch.norm(pred_accel, dim=-1).mean()
    accel_gt = torch.norm(gt_accel, dim=-1).mean()
    accel_err = torch.norm(pred_accel - gt_accel, dim=-1).mean()
    
    metrics['accel_pred'] = accel_pred.item()
    metrics['accel_gt'] = accel_gt.item()
    metrics['accel_err'] = accel_err.item()

    # Foot sliding metrics (if needed)
    pred_fs = torch.norm(torch.diff(pred_posed.Ts_world_joint[..., [7,8,10,11], 4:], dim=0), dim=-1).mean()
    gt_fs = torch.norm(torch.diff(gt_posed.Ts_world_joint[..., [7,8,10,11], 4:], dim=0), dim=-1).mean()
    
    metrics['pred_fs'] = pred_fs.item()
    metrics['gt_fs'] = gt_fs.item()
    
    return metrics

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
    
    body_model = fncsmpl.SmplhModel.load(config.smplh_npz_path).to(device)
    
    # Initialize test dataset
    test_dataset = EgoAmassHdf5DatasetDynamic(config)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_dataclass
    )
    
    # Initialize metric storage
    all_metrics = defaultdict(list)
    
    # Test loop
    for batch_idx, batch in enumerate(tqdm(test_dataloader)):
        # Move batch to device
        batch = batch.to(device)
        
        # Generate motion using pipeline
        with torch.no_grad():
            pred_motion = pipeline(
                batch_size=config.batch_size,  # Use config batch size
                num_inference_steps=config.num_inference_steps,  # Use config steps
                train_batch=batch,  # Pass the current batch
                return_intermediates=False  # We don't need intermediates for testing
            )
        
        # Compute metrics for each sequence in batch
        pred_motion: EgoDenoiseTraj = pred_motion.motion
        metrics = compute_metrics(
            pred_motion,
            batch,  # Original batch is already the ground truth
            body_model
        )
        
        # Store metrics
        for k, v in metrics.items():
            all_metrics[k].append(v)
        
        # Log metrics for current sequence
        if batch_idx % 10 == 0:
            print(f"\nMetrics for sequence {batch_idx}:")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")
    
    # Compute and save final metrics
    final_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    
    print("\nFinal Metrics:")
    for k, v in final_metrics.items():
        print(f"{k}: {v:.4f}")
    
    if config.save_results:
        output_path = config.output_dir / "test_metrics.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        print(f"\nSaved metrics to {output_path}")

if __name__ == "__main__":
    import tyro
    ipdb_safety_net()
    
    config = tyro.cli(TestConfig)
    main(config)