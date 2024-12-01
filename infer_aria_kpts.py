from __future__ import annotations
import tyro

import dataclasses
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple

import numpy as np
import torch
import viser
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from config.train import EgoAlloTrainConfig
from egoallo import training_utils
from egoallo.data.dataclass import EgoTrainingData
from egoallo.inference_utils import (
    create_masked_training_data,
    load_denoiser,
    EgoDenoiseTraj
)
from egoallo import fncsmpl, transforms as tf
from egoallo.transforms import SE3, SO3
from egoallo.sampling import CosineNoiseScheduleConstants, quadratic_ts
from egoallo.vis_helpers import visualize_traj_and_hand_detections
from egoallo.training_utils import ipdb_safety_net
from egoallo import fncsmpl_extensions
from infer_mae import run_sampling_with_masked_data
from egoallo.utils.setup_logger import setup_logger
from egoallo.egopose.stats_collector import PreprocessingStatsCollector, KeypointFilterStats
from egoallo.utils.smpl_mapping.mapping import EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS, EGOEXO4D_BODYPOSE_TO_SMPLH_INDICES
from egoallo.egopose.handpose.data_preparation.utils.utils import (
    world_to_cam,
    cam_to_img,
    body_jnts_dist_angle_check,
    reproj_error_check,
    pad_bbox_from_kpts,
    rand_bbox_from_kpts,
    aria_landscape_to_portrait
)
from egoallo.utils.utils import find_numerical_key_in_dict

from egoallo.config import make_cfg, CONFIG_FILE
local_config_file = CONFIG_FILE
CFG = make_cfg(config_name="defaults", config_file=local_config_file, cli_args=[])

logger = logging.getLogger(__name__)

def convert_egoexo4d_to_smplh(egoexo4d_joints: np.ndarray) -> np.ndarray:
    """Convert EgoExo4D format joints to SMPLH convention.
    
    Args:
        egoexo4d_joints: Array of shape (..., 17, 3) in EgoExo4D format
        
    Returns:
        Array of shape (..., 22, 3) in SMPLH body joint convention
        Non-mappable joints are filled with NaN values
    """
    # Initialize output array with NaN values
    # SMPLH has 22 body joints
    output_shape = list(egoexo4d_joints.shape[:-2]) + [22, 3]
    smplh_joints = np.full(output_shape, np.nan)
    
    # Map joints using EGOEXO4D_BODYPOSE_TO_SMPLH_INDICES
    for smplh_idx, egoexo_idx in enumerate(EGOEXO4D_BODYPOSE_TO_SMPLH_INDICES):
        if egoexo_idx != -1:
            smplh_joints[..., smplh_idx, :] = egoexo4d_joints[..., egoexo_idx, :]
            
    return smplh_joints

@dataclasses.dataclass
class InferenceConfig(EgoAlloTrainConfig):
    """Configuration for inference."""
    traj_length: int = 128
    num_samples: int = 1
    batch_size: int = 1
    smplh_model_path: Path = Path("./data/smplh/neutral/model.npz")
    output_dir: Path = Path("./outputs")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    annotation_path: Path = Path("./data/egoexo-default-gt-output")
    mask_ratio: float = 0.75
    checkpoint_dir: Path = Path("./experiments/nov_29_absrel_jnts_pilot/v2")

	# egoexod dataset.
    split: str = "train"
    anno_type: str = "manual"


class AriaKeypointsDataset(Dataset):
    """Dataset for loading and processing Aria keypoints annotations."""
    
    def __init__(
        self, 
        config: Optional[InferenceConfig] = None
    ):
        """Initialize dataset with configuration parameters.
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            anno_type: Type of annotation ('manual', 'auto')
            config: Dataset configuration parameters
        """
        self.split = config.split
        self.anno_type = config.anno_type
        self.config = config
        self.logger = setup_logger(name=self.__class__.__name__)
        self.stats_collector = PreprocessingStatsCollector()
        
        # Determine annotation path based on split and anno_type
        if config.split == "test":
            annotation_path = config.annotation_path / "annotation" / f"ego_pose_gt_anno_{config.split}_public.json"
        else:
            annotation_path = config.annotation_path / "annotation" / config.anno_type /f"ego_pose_gt_anno_{config.split}_public.json"
            
        # Load and process annotations
        self.annotations = self._load_annotations(annotation_path)
        self.take_ids = list(self.annotations.keys())
        
        # Pre-process visible joints data for efficient retrieval
        self.processed_joints_data = self._preprocess_joints_data()
        
    def _preprocess_joints_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Pre-process joints data for all takes for efficient retrieval."""
        processed_data = {}
        
        for take_id, take_data in self.annotations.items():
            jnts = []
            visible_masks = []
            
            all_frames = find_numerical_key_in_dict(take_data)
            # breakpoint()
            for frame_idx in all_frames:
                frame_data = take_data[str(frame_idx)]
                # Get joints and masks from frame data
                joints_3d = np.array(frame_data["body_3d_world"])
                valid_mask = np.array(frame_data["body_valid_3d"])

                jnts_wrt_egoexo = convert_egoexo4d_to_smplh(joints_3d)
                valid_mask = ~np.isnan(jnts_wrt_egoexo).any(axis=-1)
                
                jnts.append(jnts_wrt_egoexo)
                visible_masks.append(valid_mask)
            
            if jnts:
                processed_data[take_id] = {
                    "joints": np.stack(jnts, axis=0),
                    "masks": np.stack(visible_masks, axis=0)
                }
        return processed_data

    def _load_annotations(self, path: Path) -> Dict[str, Any]:
        """Load and validate annotations from JSON file.
        
        References bodypose_dataloader.py load_raw_data() implementation:
        ```python:T_world_root_addition/src/egoallo/egopose/bodypose/bodypose_dataloader.py
        startLine: 98
        endLine: 220
        ```
        """
        with open(path) as f:
            raw_annotations = json.load(f)
            
        return raw_annotations

    def __len__(self) -> int:
        return len(self.take_ids)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get processed data for a specific take.
        
        References bodypose_dataloader.py implementation:
        ```python:T_world_root_addition/src/egoallo/egopose/bodypose/bodypose_dataloader.py
        startLine: 262
        endLine: 275
        ```
        """
        take_id = self.take_ids[idx]
        take_data = self.processed_joints_data[take_id]
        
        return {
            "take_id": take_id,
            "jnts": take_data["joints"],
            "visible_joints_mask": take_data["masks"]
        }

def run_inference(
    model: EgoDenoiseTraj,
    masked_data: EgoTrainingData,
    body_model: fncsmpl.SmplhModel,
    config: InferenceConfig
) -> Tuple[EgoTrainingData, EgoTrainingData]:
    """Run inference on masked data and return both ground truth and predicted trajectories.
    
    Args:
        model: Trained denoiser model
        masked_data: Input data with masking
        body_model: SMPL-H body model
        config: Inference configuration
        
    Returns:
        Tuple containing (ground truth trajectory, predicted trajectory)
    """
    # Run sampling with masked data
    denoised_traj = run_sampling_with_masked_data(
        denoiser_network=model,
        body_model=body_model,
        masked_data=masked_data,
        guidance_mode="no_hands",
        guidance_post=False,
        guidance_inner=False,
        floor_z=0.0,
        hamer_detections=None,
        aria_detections=None,
        num_samples=config.num_samples,
        device=torch.device(config.device),
    )

    # Create EgoTrainingData instances for ground truth and predictions
    gt_ego_data = create_ego_training_data(
        body_model=body_model,
        T_world_root=masked_data.T_world_root,
        contacts=masked_data.contacts,
        betas=masked_data.betas,
        body_quats=masked_data.body_quats,
        T_world_cpf=masked_data.T_world_cpf,
        hand_quats=masked_data.hand_quats,
        visible_joints_mask=masked_data.visible_joints_mask
    )

    pred_ego_data = create_ego_training_data_from_denoised(
        denoised_traj=denoised_traj,
        body_model=body_model
    )

    return gt_ego_data, pred_ego_data

def create_ego_training_data_from_denoised(
    denoised_traj: EgoDenoiseTraj,
    body_model: fncsmpl.SmplhModel
) -> EgoTrainingData:
    """Create EgoTrainingData instance from denoised trajectory.
    
    Args:
        denoised_traj: Output trajectory from denoiser
        body_model: SMPL-H body model
        
    Returns:
        EgoTrainingData instance containing predicted trajectory
    """
    # Get transformation matrices
    T_world_root = SE3.from_rotation_and_translation(
        SO3.from_matrix(denoised_traj.R_world_root),
        denoised_traj.t_world_root,
    ).parameters()
    
    # Get body quaternions
    body_quats = SO3.from_matrix(denoised_traj.body_rotmats).wxyz
    
    # Get hand quaternions if available
    if denoised_traj.hand_rotmats is not None:
        hand_quats = SO3.from_matrix(denoised_traj.hand_rotmats).wxyz
        left_hand_quats = hand_quats[..., :15, :]
        right_hand_quats = hand_quats[..., 15:30, :]
        hand_quats_combined = torch.cat([left_hand_quats, right_hand_quats], dim=-2)
    else:
        hand_quats_combined = None

    # Forward kinematics to get joint positions
    shaped = body_model.with_shape(torch.mean(denoised_traj.betas, dim=1, keepdim=True))
    fk_outputs = shaped.with_pose_decomposed(
        T_world_root=T_world_root,
        body_quats=body_quats,
        left_hand_quats=left_hand_quats if denoised_traj.hand_rotmats is not None else None,
        right_hand_quats=right_hand_quats if denoised_traj.hand_rotmats is not None else None,
    )

    # Get camera pose transforms
    T_world_cpf = fncsmpl_extensions.get_T_world_cpf_from_root_pose(fk_outputs, T_world_root)

    return EgoTrainingData(
        T_world_root=T_world_root.squeeze(0).cpu(),
        contacts=denoised_traj.contacts.squeeze(0).cpu(),
        betas=denoised_traj.betas.squeeze(0).cpu(),
        joints_wrt_world=fk_outputs.Ts_world_joint.squeeze(0).cpu(),
        body_quats=body_quats.squeeze(0).cpu(),
        T_world_cpf=T_world_cpf.squeeze(0).cpu(),
        height_from_floor=T_world_cpf[..., 6:7].squeeze(0).cpu(),
        T_cpf_tm1_cpf_t=(
            tf.SE3(T_world_cpf[:-1, :]).inverse() @ tf.SE3(T_world_cpf[1:, :])
        ).parameters().cpu().squeeze(0),
        joints_wrt_cpf=(
            tf.SE3(T_world_cpf[1:, None, :]).inverse() @ 
            fk_outputs.Ts_world_joint[0, 1:, :21, 4:7].to(T_world_cpf.device)
        ),
        mask=torch.ones_like(denoised_traj.contacts[0, :], dtype=torch.bool),
        hand_quats=hand_quats_combined.squeeze(0).cpu() if hand_quats_combined is not None else None,
        visible_joints_mask=None,
    )

def save_results(
    gt_data: EgoTrainingData,
    pred_data: EgoTrainingData,
    metrics: Dict[str, float],
    output_path: Path
) -> None:
    """Save ground truth and predicted trajectories along with metrics.
    
    Args:
        gt_data: Ground truth trajectory
        pred_data: Predicted trajectory  
        metrics: Dictionary of evaluation metrics
        output_path: Path to save results
    """
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save trajectories
    torch.save(gt_data, output_path / "gt_trajectory.pt")
    torch.save(pred_data, output_path / "pred_trajectory.pt")
    
    # Save metrics
    with open(output_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

def main(config: InferenceConfig):
    """Main function to run inference on Aria keypoints data."""
    # Initialize dataset and dataloader
    training_utils.ipdb_safety_net()

    dataset = AriaKeypointsDataset(
        config=config
    )
    body_model = fncsmpl.SmplhModel.load(config.smplh_model_path).to(config.device)
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Load model and other components
    denoiser = load_denoiser(config.checkpoint_dir)
    denoiser.to(config.device)
    denoiser.eval()

    # Run inference on batches
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            # Extract batch data
            jnts = batch["jnts"].to(config.device) # shape: (batch, time, 22, 3)
            visible_masks = batch["visible_joints_mask"].to(config.device) # shape: (batch, time, 22)
            take_ids = batch["take_id"]

            # Get batch dimensions
            batch_size, seq_len, num_joints, _ = jnts.shape
            
            # Create placeholder tensors with appropriate shapes
            placeholder_T_world_root = SE3.identity(device=jnts.device, dtype=jnts.dtype).parameters().repeat(batch_size, seq_len, 1)
            placeholder_contacts = torch.zeros(batch_size, seq_len, 2).to(jnts.device)  # 2 feet contacts
            placeholder_betas = torch.zeros(batch_size, 10).to(jnts.device)  # SMPL betas
            placeholder_body_quats = torch.zeros(batch_size, seq_len, 21, 4).to(jnts.device)  # 21 joints, 4D quaternions
            placeholder_T_world_cpf = SE3.identity(device=jnts.device, dtype=jnts.dtype).parameters().repeat(batch_size, seq_len, 1)
            placeholder_hand_quats = torch.zeros(batch_size, seq_len, 30, 4).to(jnts.device)  # 15 joints per hand, 4D quaternions
            placeholder_jnts_wrt_cpf = torch.zeros(batch_size, seq_len, 21, 3).to(jnts.device)
            
            # Create EgoTrainingData instance
            masked_data = EgoTrainingData(
                T_world_root=placeholder_T_world_root,
                contacts=placeholder_contacts,
                betas=placeholder_betas,
                joints_wrt_world=jnts.float(),  # Use the actual joints from batch
                body_quats=placeholder_body_quats,
                T_world_cpf=placeholder_T_world_cpf,
                height_from_floor=placeholder_T_world_cpf[..., 2:3].squeeze(0),  # Z component
                T_cpf_tm1_cpf_t=placeholder_T_world_cpf[:-1],  # Previous to current transform
                joints_wrt_cpf=placeholder_jnts_wrt_cpf,  # Use actual joints as placeholder
                mask=torch.ones(batch_size, seq_len, dtype=torch.bool).to(jnts.device),
                hand_quats=placeholder_hand_quats,
                visible_joints_mask=visible_masks
            )

            # Run sampling with masked data
            samples = run_sampling_with_masked_data(
                denoiser_network=denoiser,
                body_model=body_model,
                masked_data=masked_data,
                guidance_mode="no_hands",
                guidance_post=False,
                guidance_inner=False,
                floor_z=0.0,
                hamer_detections=None,
                aria_detections=None,
                num_samples=config.num_samples,
                device=jnts.device
            )

            # Save results for each take in batch
            # TODO: the visible_joints of EgoTrainingData is not used anymore, consider removing any usage of visible_joints
            # for i, take_id in enumerate(take_ids):
            #     output_path = config.output_dir / f"{take_id}_samples.pt"
            #     torch.save({
            #         "samples": samples[i],
            #         "visible_joints": visible_joints[i],
            #         "visible_masks": visible_masks[i]
            #     }, output_path)

if __name__ == "__main__":
    tyro.cli(main)
