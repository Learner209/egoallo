import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm
from egoallo.transforms import SO3

from egoallo import fncsmpl

from .base import BaseEvaluator
from .constants import (
    BODY_METRICS,
    FOOT_HEIGHT_THRESHOLDS,
    FOOT_INDICES,
    HEAD_JOINT_INDEX,
)
from .types import (
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
from egoallo.utilities import procrustes_align
from egoallo.setup_logger import setup_logger

logger = setup_logger(output="logs/evaluation", name=__name__)


class BodyEvaluator(BaseEvaluator):
    """Evaluates body pose metrics between predicted and ground truth data."""

    def _load_body_model(self, model_path: Path) -> torch.nn.Module:
        """Load the SMPL body model."""
        return fncsmpl.SmplhModel.load(model_path).to(self.device)

    def compute_foot_skate(
        self,
        pred_Ts_world_joint: BatchedJointTransforms,
    ) -> FloatArray:
        """Compute foot skate metric in millimeters."""
        num_samples, time = pred_Ts_world_joint.shape[:2]  # pred_Ts_world_joint: [N, T, J, 7]

        # Adjust position to floor
        pred_Ts_world_joint = pred_Ts_world_joint.clone()
        pred_Ts_world_joint[..., 6] -= torch.min(pred_Ts_world_joint[..., 6])

        foot_indices = torch.tensor(FOOT_INDICES, device=self.device)  # [4]
        foot_positions = pred_Ts_world_joint[:, :, foot_indices, 4:7]  # [N, T, 4, 3]
        foot_positions_diff = foot_positions[:, 1:, :, :2] - foot_positions[:, :-1, :, :2]  # [N, T-1, 4, 2]
        foot_positions_diff_norm = torch.sum(torch.abs(foot_positions_diff), dim=-1)  # [N, T-1, 4]

        H_thresh = torch.tensor(
            FOOT_HEIGHT_THRESHOLDS,
            device=self.device,
            dtype=torch.float32,
        )

        foot_contact = foot_positions[:, 1:, :, 2] < H_thresh  # [N, T-1, 4]
        foot_positions_diff_norm *= foot_contact  # [N, T-1, 4]

        exponent = 2 - 2 ** (foot_positions[:, 1:, :, 2] / H_thresh)  # [N, T-1, 4]
        fs_per_sample = torch.sum(
            torch.sum(foot_positions_diff_norm * exponent, dim=-1), dim=-1
        ) * 1000.0  # Convert to mm

        return fs_per_sample.cpu().numpy()

    def compute_foot_contact(
        self,
        pred_Ts_world_joint: BatchedJointTransforms,
    ) -> FloatArray:
        """Compute foot contact metric as a ratio of frames with proper ground contact."""
        foot_indices = torch.tensor(FOOT_INDICES, device=self.device)
        H_thresh = torch.tensor(
            FOOT_HEIGHT_THRESHOLDS,
            device=self.device,
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

    def compute_head_ori(
        self,
        label_Ts_world_joint: JointTransforms,
        pred_Ts_world_joint: BatchedJointTransforms,
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
            3, device=self.device
        )
        errors = torch.linalg.norm(
            matrix_errors.reshape(pred_Ts_world_joint.shape[0], -1, 9), dim=-1
        )
        mean_errors = torch.mean(errors, dim=-1)

        return mean_errors.cpu().numpy()

    def compute_head_trans(
        self,
        label_Ts_world_joint: JointTransforms,
        pred_Ts_world_joint: BatchedJointTransforms,
    ) -> FloatArray:
        """Compute head translation error in millimeters."""
        errors = (
            pred_Ts_world_joint[:, :, HEAD_JOINT_INDEX, 4:7]
            - label_Ts_world_joint[:, HEAD_JOINT_INDEX, 4:7]
        )
        mean_errors = torch.mean(torch.linalg.norm(errors, dim=-1), dim=-1) * 1000.0  # Convert to mm
        return mean_errors.cpu().numpy()

    def compute_mpjpe(
        self,
        label_T_world_root: RootTransforms,
        label_Ts_world_joint: JointTransforms,
        pred_T_world_root: BatchedRootTransforms,
        pred_Ts_world_joint: BatchedJointTransforms,
        per_frame_procrustes_align: bool,
    ) -> FloatArray:
        """Compute Mean Per Joint Position Error."""
        num_samples = pred_Ts_world_joint.shape[0]

        # Concatenate root and joints
        label_Ts_world_joint = torch.cat(
            [label_T_world_root.unsqueeze(1), label_Ts_world_joint], dim=1
        )  # [T, J+1, 7]
        pred_Ts_world_joint = torch.cat(
            [pred_T_world_root.unsqueeze(2), pred_Ts_world_joint], dim=2
        )  # [N, T, J+1, 7]

        pred_joint_positions = pred_Ts_world_joint[:, :, :, 4:7]  # [N, T, J+1, 3]
        label_joint_positions = label_Ts_world_joint[:, :, 4:7].unsqueeze(0).repeat(
            num_samples, 1, 1, 1
        )  # [N, T, J+1, 7]

        if per_frame_procrustes_align:
            pred_joint_positions = self.procrustes_align(
                points_y=label_joint_positions,
                points_x=pred_joint_positions,
                output="aligned_x",
            )

        position_differences = pred_joint_positions - label_joint_positions  # [N, T, J+1, 3]
        pjpe = torch.linalg.norm(position_differences, dim=-1) * 1000.0  # [N, T, J+1]
        mpjpe = torch.mean(pjpe.reshape(num_samples, -1), dim=-1)  # [N]

        return mpjpe.cpu().numpy()

    def procrustes_align(
        self,
        points_y: torch.Tensor,
        points_x: torch.Tensor,
        output: ProcrustesMode,
        fix_scale: bool = False,
    ) -> ProcrustesOutput:
        """Perform Procrustes alignment between point sets."""
        s, R, t = procrustes_align(points_y, points_x, fix_scale)
        
        if output == "transforms":
            return s, R, t
        elif output == "aligned_x":
            aligned_x = s[..., None, :] * torch.einsum(
                "...ij,...nj->...ni", R, points_x
            ) + t[..., None, :]
            return aligned_x

    def evaluate_directory(
        self,
        dir_with_pt_files: PathLike,
        use_mean_body_shape: bool = False,
        coco_regressor_path: Optional[PathLike] = None,
        skip_confirm: bool = False,
    ) -> None:
        """Evaluate all .pt files in directory."""
        dir_path = Path(dir_with_pt_files)
        assert dir_path.is_dir(), f"{dir_path} is not a directory."

        # Setup output paths
        suffix = "_meanbody" if use_mean_body_shape else ""
        out_disagg_pt_path = dir_path / f"_eval_cached_disaggregated_metrics{suffix}.pt"
        out_yaml_path = dir_path / f"_eval_cached_summary{suffix}.yaml"

        # Load COCO regressor if provided
        coco_regressor = None
        if coco_regressor_path is not None:
            coco_regressor = torch.load(coco_regressor_path)
            assert coco_regressor.shape == (17, 6890), "Invalid COCO regressor shape"
            coco_regressor = coco_regressor.to(self.device)

        # Check existing metrics
        if out_disagg_pt_path.exists() or out_yaml_path.exists():
            logger.info("Found existing metrics:")
            logger.info(out_yaml_path.read_text())
            if not skip_confirm:
                confirm = input("Metrics already computed. Overwrite? (y/n) ")
                if confirm.lower() != "y":
                    logger.info("Aborting evaluation.")
                    return

        # Get PT files
        pt_paths = [
            p
            for p in dir_path.glob("**/*.pt")
            if not p.name.startswith("_eval_cached_")
        ]
        num_sequences = len(pt_paths)

        # Initialize metrics
        metrics_list = BODY_METRICS.copy()
        if coco_regressor is not None:
            metrics_list.append("coco_mpjpe")

        stats_per_subsequence: Dict[str, FloatArray] = {
            metric: np.zeros(num_sequences) for metric in metrics_list
        }

        # Process files
        # with ThreadPoolExecutor() as executor:
        #     futures = [
        #         executor.submit(
        #             self.process_file,
        #             pt_paths[i],
        #             coco_regressor,
        #             use_mean_body_shape,
        #         )
        #         for i in range(num_sequences)
        #     ]

        #     for i, future in enumerate(tqdm(futures, total=num_sequences)):
        #         metrics = future.result()
        #         for key in metrics_list:
        #             stats_per_subsequence[key][i] = metrics.get(key, np.nan)

        for i in tqdm(range(num_sequences)):
            metrics = self.process_file(
                pt_paths[i],
                coco_regressor,
                use_mean_body_shape,
            )
            for key in metrics_list:
                stats_per_subsequence[key][i] = metrics.get(key, np.nan)

        # Save results
        torch.save(stats_per_subsequence, out_disagg_pt_path)
        logger.info(f"Wrote disaggregated metrics to {out_disagg_pt_path}")

        # Write summary
        summary = {
            key: {
                "average_sample_mean": float(np.nanmean(values)),
                "stddev_sample": float(np.nanstd(values)),
                "stderr_sample": float(
                    np.nanstd(values) / np.sqrt(np.count_nonzero(~np.isnan(values)))
                ),
            }
            for key, values in stats_per_subsequence.items()
        }

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary_text = yaml.dump(summary)
        out_yaml_path.write_text(f"# Written at {timestamp}\n\n{summary_text}")
        logger.info(f"Wrote summary to {out_yaml_path}")
        logger.info(summary_text)

    def process_file(
        self,
        pt_path: PathLike,
        coco_regressor: Optional[torch.Tensor],
        use_mean_body_shape: bool,
    ) -> MetricsDict:
        """Process a single PT file and compute metrics."""
        # Load file
        path = Path(pt_path)
        if not path.suffix == '.pt':
            raise ValueError(f"Expected .pt file, got: {path.suffix}")
        
        outputs = torch.load(path)
        
        # Verify required keys exist
        required_keys = [
            "groundtruth_betas", "groundtruth_T_world_root", "groundtruth_body_quats",
            "sampled_betas", "sampled_T_world_root", "sampled_body_quats"
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
        
        gt_shaped = self.body_model.with_shape(gt_betas)
        gt_posed = gt_shaped.with_pose_decomposed(
            T_world_root=gt_T_world_root,
            body_quats=gt_body_quats,
        )

        # Load predicted data
        sampled_betas = to_device(outputs["sampled_betas"])
        sampled_T_world_root = to_device(outputs["sampled_T_world_root"])
        sampled_body_quats = to_device(outputs["sampled_body_quats"])

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
        # import ipdb; ipdb.set_trace()
        
        # Add batch dimension to predicted tensors if needed
        if len(sampled_posed.T_world_root.shape) == 2:  # [T, 7]
            sampled_T_world_root = sampled_posed.T_world_root.unsqueeze(0)  # Add batch dim [1, T, 7]
            sampled_Ts_world_joint = sampled_posed.Ts_world_joint[..., :21, :].unsqueeze(0)  # Add batch dim [1, T, 21, 7]
        else:
            sampled_T_world_root = sampled_posed.T_world_root  # Keep as [N, T, 7]
            sampled_Ts_world_joint = sampled_posed.Ts_world_joint[..., :21, :]  # Keep as [N, T, 21, 7]

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

        if coco_regressor is not None:
            gt_mesh = gt_posed.lbs()
            gt_coco_joints = torch.einsum("ij,...jk->...ik", coco_regressor, gt_mesh.verts)

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

        # logger.info(f"Computed metrics for {path}: {metrics}")

        return metrics
