import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import typeguard

# from egoallo import fncsmpl
from egoallo.config.train.train_config import EgoAlloTrainConfig
from egoallo.constants import FOOT_HEIGHT_THRESHOLDS
from egoallo.constants import FOOT_INDICES
from egoallo.constants import HEAD_JOINT_INDEX
from egoallo.evaluation.metrics import EgoAlloEvaluationMetrics
from egoallo.transforms import SO3
from egoallo.types import BatchedJointTransforms
from egoallo.types import FloatArray
from egoallo.types import PathLike
from egoallo.types import ProcrustesMode
from egoallo.types import ProcrustesOutput
from egoallo.utilities import procrustes_align
from egoallo.utils.setup_logger import setup_logger
from jaxtyping import Float
from jaxtyping import jaxtyped
from torch import Tensor

from .base import BaseEvaluator

logger = setup_logger(output="logs/evaluation", name=__name__, level=logging.INFO)


class BodyEvaluator(BaseEvaluator):
    """Evaluates body pose metrics between predicted and ground truth data."""

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
            torch.abs(foot_positions_diff),
            dim=-1,
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

        return fs_per_sample.cpu().numpy(force=True)

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

        return contact_ratio.cpu().numpy(force=True)

    @classmethod
    @jaxtyped(typechecker=typeguard.typechecked)
    def compute_head_ori(
        cls,
        label_Ts_world_joint: Float[Tensor, "batch time 21 7"],
        pred_Ts_world_joint: Float[Tensor, "batch time 21 7"],
        device: torch.device,
    ) -> FloatArray:
        """Compute head orientation error."""
        pred_head_rot = pred_Ts_world_joint[:, :, HEAD_JOINT_INDEX, :4]
        label_head_rot = label_Ts_world_joint[:, :, HEAD_JOINT_INDEX, :4]

        pred_matrix = SO3(pred_head_rot).as_matrix()
        label_matrix = SO3(label_head_rot).as_matrix()

        # Use transpose() to only transpose last two dimensions
        matrix_errors = pred_matrix @ label_matrix.transpose(-2, -1) - torch.eye(
            3,
            device=device,
        )
        errors = torch.linalg.norm(
            matrix_errors.reshape(pred_Ts_world_joint.shape[0], -1, 9),
            dim=-1,
        )
        mean_errors = torch.mean(errors, dim=-1)

        return mean_errors.cpu().numpy(force=True)

    @classmethod
    def compute_head_trans(
        cls,
        label_Ts_world_joint: Float[Tensor, "batch time 21 7"],
        pred_Ts_world_joint: Float[Tensor, "batch time 21 7"],
        device: torch.device,
    ) -> FloatArray:
        """Compute head translation error in millimeters."""
        errors = (
            pred_Ts_world_joint[:, :, HEAD_JOINT_INDEX, 4:]
            - label_Ts_world_joint[:, :, HEAD_JOINT_INDEX, 4:]
        )
        mean_errors = (
            torch.mean(torch.linalg.norm(errors, dim=-1), dim=-1) * 1000.0
        )  # Convert to mm
        return mean_errors.cpu().numpy(force=True)

    @classmethod
    @jaxtyped(typechecker=typeguard.typechecked)
    def compute_mpjpe(
        cls,
        label_root_pos: Float[Tensor, "batch time 3"],
        label_joint_pos: Float[Tensor, "batch time num_joints 3"],
        pred_root_pos: Float[Tensor, "batch time 3"],
        pred_joint_pos: Float[Tensor, "batch time num_joints 3"],
        per_frame_procrustes_align: bool,
        device: torch.device,
    ) -> FloatArray:
        """Compute Mean Per Joint Position Error in millimeters."""
        num_samples = pred_joint_pos.shape[0]

        # Concatenate root and joints
        label_positions = torch.cat(
            [label_root_pos.unsqueeze(-2), label_joint_pos],
            dim=-2,
        )  # [batch, T, J+1, 3]
        pred_positions = torch.cat(
            [pred_root_pos.unsqueeze(-2), pred_joint_pos],
            dim=-2,
        )  # [batch, T, J+1, 3]

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

        return mpjpe.cpu().numpy(force=True)

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
                pt_paths[i],
                None,
                use_mean_body_shape=use_mean_body_shape,
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

    @classmethod
    @jaxtyped(typechecker=typeguard.typechecked)
    def compute_masked_error(
        cls,
        gt: Float[Tensor, "*batch time d"],
        pred: Float[Tensor, "*batch time d"],
        mask: Optional[Float[Tensor, "*batch time"]] = None,
        norm_dim: int = -1,
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
        # Create mask if not provided
        if mask is None:
            mask = torch.ones(
                (gt.shape[0], gt.shape[1]),
                dtype=torch.bool,
                device=device,
            )
        mask_sum = mask.sum()

        # Compute error
        diff = torch.mean((gt - pred) ** 2, dim=norm_dim)  # [b t]
        return float((diff * mask).sum() / mask_sum)
