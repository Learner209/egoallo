import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from egoallo import fncsmpl
from egoallo.config.train.train_config import EgoAlloTrainConfig
from egoallo.evaluation.metrics import EgoAlloEvaluationMetrics
from egoallo.transforms import SO3
from egoallo.utilities import procrustes_align
from egoallo.utils.setup_logger import setup_logger
from jaxtyping import Float, jaxtyped
from torch import Tensor

import typeguard

from .base import BaseEvaluator
from egoallo.constants import (
    BODY_METRICS,
    FOOT_HEIGHT_THRESHOLDS,
    FOOT_INDICES,
    HEAD_JOINT_INDEX,
)
from egoallo.types import (
    BatchedJointTransforms,
    BatchedRootTransforms,
    FloatArray,
    JointTransforms,
    MetricsDict,
    PathLike,
    ProcrustesMode,
    ProcrustesOutput,
    RootTransforms,
)

logger = setup_logger(output="logs/evaluation", name=__name__, level=logging.INFO)


class BodyEvaluator(BaseEvaluator):
    """Evaluates body pose metrics between predicted and ground truth data."""

    def _load_body_model(self, model_path: Path) -> fncsmpl.SmplhModel:
        """Load the SMPL body model."""
        return fncsmpl.SmplhModel.load(model_path).to(self.device)

    @classmethod
    @jaxtyped(typechecker=typeguard.typechecked)
    def compute_foot_skate(
        cls,
        pred_Ts_world_joint: BatchedJointTransforms,
        device: torch.device,
    ) -> FloatArray:
        """Compute foot skate metric in millimeters."""
        num_samples, time = pred_Ts_world_joint.shape[
            :2
        ]  # pred_Ts_world_joint: [N, T, J, 7]

        # Adjust position to floor
        pred_Ts_world_joint = pred_Ts_world_joint.clone()
        pred_Ts_world_joint[..., 6] -= torch.min(pred_Ts_world_joint[..., 6])

        foot_indices = torch.tensor(FOOT_INDICES, device=device)  # [4]
        foot_positions = pred_Ts_world_joint[:, :, foot_indices, 4:7]  # [N, T, 4, 3]
        foot_positions_diff = (
            foot_positions[:, 1:, :, :2] - foot_positions[:, :-1, :, :2]
        )  # [N, T-1, 4, 2]
        foot_positions_diff_norm = torch.sum(
            torch.abs(foot_positions_diff), dim=-1
        )  # [N, T-1, 4]

        H_thresh = torch.tensor(
            FOOT_HEIGHT_THRESHOLDS,
            device=device,
            dtype=torch.float32,
        )

        foot_contact = foot_positions[:, 1:, :, 2] < H_thresh  # [N, T-1, 4]
        foot_positions_diff_norm *= foot_contact  # [N, T-1, 4]

        exponent = 2 - 2 ** (foot_positions[:, 1:, :, 2] / H_thresh)  # [N, T-1, 4]
        fs_per_sample = (
            torch.sum(torch.sum(foot_positions_diff_norm * exponent, dim=-1), dim=-1)
            * 1000.0
        )  # Convert to mm

        return fs_per_sample.cpu().numpy()


    @classmethod
    @jaxtyped(typechecker=typeguard.typechecked)
    def compute_foot_contact(
        cls,
        pred_Ts_world_joint: BatchedJointTransforms,
        device: torch.device,
    ) -> FloatArray:
        """Compute foot contact metric as a ratio of frames with proper ground contact."""
        foot_indices = torch.tensor(FOOT_INDICES, device=device)
        H_thresh = torch.tensor(
            FOOT_HEIGHT_THRESHOLDS,
            device=device,
            dtype=torch.float32,
        )

        foot_positions = pred_Ts_world_joint[:, :, foot_indices, 4:7]  # [N, T, 4, 3]

        # Check which feet are in contact with ground per frame
        foot_contacts = foot_positions[..., 2] < H_thresh  # [N, T, 4]

        # Count number of feet in contact per frame
        num_contacts_per_frame = torch.sum(foot_contacts, dim=-1)  # [N, T]

        # Calculate ratio of frames with at least one foot contact
        contact_ratio = torch.mean((num_contacts_per_frame > 0).float(), dim=-1)  # [N]

        return contact_ratio.cpu().numpy()

    @classmethod
    @jaxtyped(typechecker=typeguard.typechecked)
    def compute_head_ori(
        cls,
        label_Ts_world_joint: JointTransforms,
        pred_Ts_world_joint: BatchedJointTransforms,
        device: torch.device,
    ) -> FloatArray:
        """Compute head orientation error."""
        pred_head_rot = pred_Ts_world_joint[:, :, HEAD_JOINT_INDEX, :4]
        label_head_rot = label_Ts_world_joint[:, HEAD_JOINT_INDEX, :4]

        pred_matrix = SO3(pred_head_rot).as_matrix()
        label_matrix = SO3(label_head_rot).as_matrix()

        # Remove batch dimension from pred_matrix since we're comparing one at a time
        pred_matrix = pred_matrix.squeeze(0)  # Now shape: [127, 3, 3]
        # Use transpose() to only transpose last two dimensions
        matrix_errors = pred_matrix @ label_matrix.transpose(-2, -1) - torch.eye(
            3, device=device
        )
        errors = torch.linalg.norm(
            matrix_errors.reshape(pred_Ts_world_joint.shape[0], -1, 9), dim=-1
        )
        mean_errors = torch.mean(errors, dim=-1)

        return mean_errors.cpu().numpy()

    @classmethod
    def compute_head_trans(
        cls,
        label_Ts_world_joint: Float[Tensor, "time num_joints 3"],
        pred_Ts_world_joint: Float[Tensor, "num_samples time num_joints 3"],
        device: torch.device,
    ) -> FloatArray:
        """Compute head translation error in millimeters."""
        errors = (
            pred_Ts_world_joint[:, :, HEAD_JOINT_INDEX]
            - label_Ts_world_joint[:, HEAD_JOINT_INDEX]
        )
        mean_errors = (
            torch.mean(torch.linalg.norm(errors, dim=-1), dim=-1) * 1000.0
        )  # Convert to mm
        return mean_errors.cpu().numpy()

    @classmethod
    @jaxtyped(typechecker=typeguard.typechecked)
    def compute_mpjpe(
        cls,
        label_root_pos: Float[Tensor, "time 3"],
        label_joint_pos: Float[Tensor, "time num_joints 3"],
        pred_root_pos: Float[Tensor, "num_samples time 3"],
        pred_joint_pos: Float[Tensor, "num_samples time num_joints 3"],
        per_frame_procrustes_align: bool,
        device: torch.device,
    ) -> FloatArray:
        """Compute Mean Per Joint Position Error in millimeters."""
        num_samples = pred_joint_pos.shape[0]

        # Concatenate root and joints
        label_positions = torch.cat(
            [label_root_pos.unsqueeze(-2), label_joint_pos], dim=-2
        )  # [T, J+1, 3]
        pred_positions = torch.cat(
            [pred_root_pos.unsqueeze(-2), pred_joint_pos], dim=-2
        )  # [N, T, J+1, 3]

        # Expand label positions to match prediction batch size
        label_positions = label_positions.unsqueeze(0).repeat(num_samples, 1, 1, 1)

        if per_frame_procrustes_align:
            pred_positions = cls.procrustes_align(
                points_y=label_positions,
                points_x=pred_positions,
                output="aligned_x",
                device=device,
            )

        position_differences = pred_positions - label_positions  # [N, T, J+1, 3]
        pjpe = torch.linalg.norm(position_differences, dim=-1) * 1000.0  # [N, T, J+1]
        mpjpe = torch.mean(pjpe.reshape(num_samples, -1), dim=-1)  # [N]

        return mpjpe.cpu().numpy()

    @classmethod
    @jaxtyped(typechecker=typeguard.typechecked)
    def procrustes_align(
        cls,
        points_y: Float[Tensor, "*batch time 3"],
        points_x: Float[Tensor, "*batch time 3"],
        output: ProcrustesMode,
        fix_scale: bool = False,
        device: torch.device = torch.device("cpu"),
    ) -> ProcrustesOutput:
        """Perform Procrustes alignment between point sets."""
        s, R, t = procrustes_align(points_y, points_x, fix_scale)

        if output == "transforms":
            return s, R, t
        elif output == "aligned_x":
            aligned_x = (
                s[..., None, :] * torch.einsum("...ij,...nj->...ni", R, points_x)
                + t[..., None, :]
            )
            return aligned_x

    def evaluate_directory(
        self,
        runtime_config: EgoAlloTrainConfig,
        dir_with_pt_files: PathLike,
        use_mean_body_shape: bool = False,
        suffix: str = "",
        skip_confirm: bool = True,
    ) -> EgoAlloEvaluationMetrics:
        """Evaluate sequences in directory."""
        pt_paths = sorted(Path(dir_with_pt_files).glob("*.pt"))
        num_sequences = len(pt_paths)

        # Get metric names from EgoAlloEvaluationMetrics fields
        metric_fields = [
            field
            for field in EgoAlloEvaluationMetrics.__dataclass_fields__
            if field not in ["metrics_file", "summary_file", "coco_mpjpe"]
        ]

        # Initialize metrics dictionary
        metrics_dict = {metric: np.zeros(num_sequences) for metric in metric_fields}

        # Single-threaded processing
        assert num_sequences > 0
        for i in range(num_sequences):
            result = self.process_file(
                pt_paths[i], None, use_mean_body_shape=use_mean_body_shape
            )
            # Store metrics
            for metric in metric_fields:
                metrics_dict[metric][i] = result[metric]

        # # Multi-threaded processing (original version)
        # with ThreadPoolExecutor() as executor:
        #     futures = [
        #         executor.submit(
        #             self.process_file,
        #             pt_paths[i],
        #             None,
        #             use_mean_body_shape=use_mean_body_shape,
        #         )
        #         for i in range(num_sequences)
        #     ]

        #     for i, future in enumerate(futures):
        #         result = future.result()

        #         # Store metrics
        #         for metric in metric_fields:
        #             metrics_dict[metric][i] = result[metric]

        # Create metrics object
        metrics = EgoAlloEvaluationMetrics(**metrics_dict)
        metrics.save(Path(dir_with_pt_files), suffix)

        return metrics

    def process_file(
        self,
        pt_path: PathLike,
        coco_regressor: Optional[torch.Tensor],
        use_mean_body_shape: bool,
    ) -> MetricsDict:
        """Process a single PT file and compute metrics."""
        # Load file
        path = Path(pt_path)
        if not path.suffix == ".pt":
            raise ValueError(f"Expected .pt file, got: {path.suffix}")

        outputs = torch.load(path)

        # Verify required keys exist
        required_keys = [
            "groundtruth_betas",
            "groundtruth_T_world_root",
            "groundtruth_body_quats",
            "sampled_betas",
            "sampled_T_world_root",
            "sampled_body_quats",
        ]
        missing_keys = [key for key in required_keys if key not in outputs]
        if missing_keys:
            raise KeyError(f"Missing required keys in {path}: {missing_keys}")

        # Move tensors to device
        def to_device(x):
            return x.to(self.device) if isinstance(x, torch.Tensor) else x

        # Load ground truth data
        gt_betas = to_device(outputs["groundtruth_betas"])
        gt_T_world_root = to_device(outputs["groundtruth_T_world_root"])
        gt_body_quats = to_device(outputs["groundtruth_body_quats"])

        # Check for NaN values in ground truth
        if torch.isnan(gt_betas).any():
            raise ValueError("NaN values found in groundtruth_betas")
        if torch.isnan(gt_T_world_root).any():
            raise ValueError("NaN values found in groundtruth_T_world_root")
        if torch.isnan(gt_body_quats).any():
            raise ValueError("NaN values found in groundtruth_body_quats")

        gt_shaped = self.body_model.with_shape(gt_betas)
        gt_posed = gt_shaped.with_pose_decomposed(
            T_world_root=gt_T_world_root,
            body_quats=gt_body_quats,
        )

        # Load predicted data
        sampled_betas = to_device(outputs["sampled_betas"])
        sampled_T_world_root = to_device(outputs["sampled_T_world_root"])
        sampled_body_quats = to_device(outputs["sampled_body_quats"])

        # Check for NaN values in predictions
        if torch.isnan(sampled_betas).any():
            raise ValueError("NaN values found in sampled_betas")
        if torch.isnan(sampled_T_world_root).any():
            raise ValueError("NaN values found in sampled_T_world_root")
        if torch.isnan(sampled_body_quats).any():
            raise ValueError("NaN values found in sampled_body_quats")

        assert not use_mean_body_shape
        if use_mean_body_shape:
            mean_betas = torch.zeros_like(sampled_betas.mean(dim=1, keepdim=True))
            sampled_shaped = self.body_model.with_shape(mean_betas)
        else:
            mean_betas = sampled_betas.mean(dim=1, keepdim=True)
            sampled_shaped = self.body_model.with_shape(mean_betas)

        sampled_posed = sampled_shaped.with_pose_decomposed(
            T_world_root=sampled_T_world_root,
            body_quats=sampled_body_quats,
        )
        # Compute metrics
        metrics = {}

        # Add batch dimension to predicted tensors if needed
        if len(sampled_posed.T_world_root.shape) == 2:  # [T, 7]
            sampled_T_world_root = sampled_posed.T_world_root.unsqueeze(
                0
            )  # Add batch dim [1, T, 7]
            sampled_Ts_world_joint = sampled_posed.Ts_world_joint[
                ..., :21, :
            ].unsqueeze(0)  # Add batch dim [1, T, 21, 7]
        else:
            sampled_T_world_root = sampled_posed.T_world_root  # Keep as [N, T, 7]
            sampled_Ts_world_joint = sampled_posed.Ts_world_joint[
                ..., :21, :
            ]  # Keep as [N, T, 21, 7]

        metrics["mpjpe"] = float(
            self.compute_mpjpe(
                label_T_world_root=gt_posed.T_world_root,  # [T, 7]
                label_Ts_world_joint=gt_posed.Ts_world_joint[..., :21, :],  # [T, 21, 7]
                pred_T_world_root=sampled_T_world_root,  # [N, T, 7]
                pred_Ts_world_joint=sampled_Ts_world_joint,  # [N, T, 21, 7]
                per_frame_procrustes_align=False,
            ).mean()
        )

        metrics["pampjpe"] = float(
            self.compute_mpjpe(
                label_T_world_root=gt_posed.T_world_root,  # [T, 7]
                label_Ts_world_joint=gt_posed.Ts_world_joint[..., :21, :],  # [T, 21, 7]
                pred_T_world_root=sampled_T_world_root,  # [N, T, 7]
                pred_Ts_world_joint=sampled_Ts_world_joint,  # [N, T, 21, 7]
                per_frame_procrustes_align=True,
            ).mean()
        )

        metrics["head_ori"] = float(
            self.compute_head_ori(
                label_Ts_world_joint=gt_posed.Ts_world_joint[..., :21, :],  # [T, 21, 7]
                pred_Ts_world_joint=sampled_Ts_world_joint,  # [N, T, 21, 7]
            ).mean()
        )

        metrics["head_trans"] = float(
            self.compute_head_trans(
                label_Ts_world_joint=gt_posed.Ts_world_joint[..., :21, :],  # [T, 21, 7]
                pred_Ts_world_joint=sampled_Ts_world_joint,  # [N, T, 21, 7]
            ).mean()
        )

        metrics["foot_skate"] = float(
            self.compute_foot_skate(
                pred_Ts_world_joint=sampled_Ts_world_joint,  # [N, T, 21, 7]
            ).mean()
        )

        metrics["foot_contact"] = float(
            self.compute_foot_contact(
                pred_Ts_world_joint=sampled_Ts_world_joint,  # [N, T, 21, 7]
            ).mean()
        )

        # Betas error
        metrics["betas_error"] = self.compute_masked_error(
            gt=gt_betas,
            pred=mean_betas.squeeze(1),
            unsqueeze_gt=True,
            unsqueeze_pred=True,
        )

        # Body rotation error
        # breakpoint()
        metrics["body_rotmats_error"] = self.compute_masked_error(
            gt=SO3(gt_body_quats).as_matrix().reshape(*gt_body_quats.shape[:-1], -1),
            pred=SO3(sampled_body_quats)
            .as_matrix()
            .mean(dim=0)
            .reshape(*gt_body_quats.shape[:-1], -1),
            unsqueeze_gt=True,
            unsqueeze_pred=True,
        )

        # Root transform errors
        metrics["R_world_root_error"] = self.compute_masked_error(
            gt=gt_T_world_root[..., :4],
            pred=sampled_T_world_root.mean(dim=0)[..., :4],
            unsqueeze_gt=True,
            unsqueeze_pred=True,
        )

        metrics["t_world_root_error"] = self.compute_masked_error(
            gt=gt_T_world_root[..., 4:],
            pred=sampled_T_world_root.mean(dim=0)[..., 4:],
            unsqueeze_gt=True,
            unsqueeze_pred=True,
        )

		# MPJPE under COCO kpts 
        if coco_regressor is not None:
            gt_mesh = gt_posed.lbs()
            gt_coco_joints = torch.einsum(
                "ij,...jk->...ik", coco_regressor, gt_mesh.verts
            )

            num_samples = sampled_T_world_root.shape[0]
            sampled_coco_joints = []
            for j in range(num_samples):
                sample_posed = sampled_posed.map(
                    lambda t: t[j] if t.shape[0] == num_samples else t
                )
                sample_mesh = sample_posed.lbs()
                sample_coco_joints = torch.einsum(
                    "ij,...jk->...ik", coco_regressor, sample_mesh.verts
                )
                sampled_coco_joints.append(sample_coco_joints)

            sampled_coco_joints = torch.stack(sampled_coco_joints, dim=0)
            coco_errors = (
                torch.linalg.norm(gt_coco_joints - sampled_coco_joints, dim=-1) * 1000.0
            )
            metrics["coco_mpjpe"] = float(coco_errors.mean().item())

        return metrics

    @classmethod
    def compute_masked_error(
        cls,
        gt: torch.Tensor,
        pred: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        norm_dim: int = -1,
        unsqueeze_gt: bool = True,
        unsqueeze_pred: bool = True,
        device: torch.device = torch.device("cpu"),
    ) -> float:
        """Compute masked error between ground truth and predicted tensors.

        Args:
            gt: Ground truth tensor
            pred: Predicted tensor
            mask: Optional mask tensor. If None, creates all-ones mask
            norm_dim: Dimension along which to compute norm
            unsqueeze_gt: Whether to unsqueeze gt to [b t d]
            unsqueeze_pred: Whether to unsqueeze pred to [b t d]

        Returns:
            Masked mean error as float
        """
        # Reshape tensors to [b t d] format if needed
        if unsqueeze_gt:
            gt = gt.unsqueeze(0) if gt.dim() == 2 else gt.unsqueeze(1)
        if unsqueeze_pred:
            pred = pred.unsqueeze(0) if pred.dim() == 2 else pred.unsqueeze(1)

        # Create mask if not provided
        if mask is None:
            mask = torch.ones(
                (gt.shape[0], gt.shape[1]), dtype=torch.bool, device=device
            )
        mask_sum = mask.sum()

        # Compute error
        diff = torch.mean((gt - pred) ** 2, dim=norm_dim)  # [b t]
        return float((diff * mask).sum() / mask_sum)
