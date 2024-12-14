from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.utils.data
import typeguard
from jaxtyping import jaxtyped
from torch.utils.data import DataLoader
from tqdm import tqdm

from egoallo import fncsmpl, fncsmpl_extensions
from egoallo import transforms as tf
from egoallo.config.inference.inference_defaults import InferenceConfig
from egoallo.data.amass import EgoAlloTrainConfig, EgoAmassHdf5Dataset
from egoallo.data.dataclass import EgoTrainingData, collate_dataclass
from egoallo.evaluation.body_evaluator import BodyEvaluator
from egoallo.guidance_optimizer_jax import GuidanceMode
from egoallo.inference_utils import (
    create_masked_training_data,
    load_denoiser,
    load_runtime_config,
)
from egoallo.network import EgoDenoiser, EgoDenoiserConfig, EgoDenoiseTraj
from egoallo.sampling import (
    CosineNoiseScheduleConstants,
    quadratic_ts,
    run_sampling_with_masked_data,
)
from egoallo.transforms import SE3, SO3
from egoallo.utils.setup_logger import setup_logger
from egoallo.evaluation.metrics import EgoAlloEvaluationMetrics

logger = setup_logger(output="logs/test", name=__name__)


@dataclass
class ProcessingResult:
    """Container for sequence processing results."""

    gt_data: EgoTrainingData
    denoised_traj: EgoDenoiseTraj
    denoised_data: EgoTrainingData


class DataVisualizer:
    """Handles visualization of trajectory data."""

    @staticmethod
    def save_visualization(
        gt_data: EgoTrainingData,
        denoised_data: EgoTrainingData,
        body_model: fncsmpl.SmplhModel,
        output_dir: Path,
        timestamp: str,
    ) -> Tuple[Path, Path]:
        """Save visualization of ground truth and inferred trajectories."""
        output_dir.mkdir(exist_ok=True, parents=True)

        gt_path = output_dir / f"gt_traj_{timestamp}.mp4"
        inferred_path = output_dir / f"inferred_traj_{timestamp}.mp4"

        EgoTrainingData.visualize_ego_training_data(gt_data, body_model, str(gt_path))
        EgoTrainingData.visualize_ego_training_data(
            denoised_data, body_model, str(inferred_path)
        )

        return gt_path, inferred_path


class SequenceProcessor:
    """Handles processing of individual sequences."""

    def __init__(self, body_model: fncsmpl.SmplhModel, device: torch.device):
        self.body_model = body_model
        self.device = device

    def process_sequence(
        self,
        batch: EgoTrainingData,
        denoiser: EgoDenoiser,
        inference_config: InferenceConfig,
        model_config: EgoDenoiserConfig,
        device: torch.device,
    ) -> ProcessingResult:
        """Process a single sequence."""

        # Create masked training data
        masked_data = create_masked_training_data(
            body_model=self.body_model,
            data=batch,
            mask_ratio=model_config.mask_ratio,
            device=self.device,
        )

        # Run denoising with guidance
        denoised_traj = run_sampling_with_masked_data(
            denoiser_network=denoiser,
            body_model=self.body_model,
            masked_data=masked_data,
            guidance_mode=inference_config.guidance_mode,
            guidance_post=inference_config.guidance_post,
            guidance_inner=inference_config.guidance_inner,
            floor_z=0.0,
            hamer_detections=None,
            aria_detections=None,
            num_samples=1,
            device=self.device,
        )

        # Convert denoised trajectory back to EgoTrainingData format
        T_world_root = SE3.from_rotation_and_translation(
            SO3.from_matrix(denoised_traj.R_world_root), denoised_traj.t_world_root
        ).parameters()

        body_quats = SO3.from_matrix(denoised_traj.body_rotmats).wxyz

        if denoised_traj.hand_rotmats is not None:
            hand_quats = SO3.from_matrix(denoised_traj.hand_rotmats).wxyz
            left_hand_quats = hand_quats[..., :15, :]
            right_hand_quats = hand_quats[..., 15:30, :]
            full_hand_quats = torch.cat([left_hand_quats, right_hand_quats], dim=-2)
        else:
            full_hand_quats = None

        # Get forward kinematics results
        shaped = self.body_model.with_shape(
            torch.mean(denoised_traj.betas, dim=1, keepdim=True)
        )
        fk_outputs = shaped.with_pose_decomposed(
            T_world_root=T_world_root,
            body_quats=body_quats,
            left_hand_quats=left_hand_quats
            if denoised_traj.hand_rotmats is not None
            else None,
            right_hand_quats=right_hand_quats
            if denoised_traj.hand_rotmats is not None
            else None,
        )

        T_world_cpf = fncsmpl_extensions.get_T_world_cpf_from_root_pose(
            fk_outputs, T_world_root
        )

        Ts_world_joints = tf.SE3(
            torch.cat(
                [T_world_root[..., None, :], fk_outputs.Ts_world_joint[..., :21, :]],
                dim=-2,
            )[..., :22, :]
        )
        # breakpoint()
        Ts_cpf_joints = tf.SE3(T_world_cpf[:, :, None, :]).inverse() @ (Ts_world_joints)
        denoised_data = EgoTrainingData(
            T_world_root=T_world_root,
            contacts=denoised_traj.contacts,
            betas=denoised_traj.betas.mean(dim=1, keepdim=True),  # (1, 16)
            joints_wrt_world=Ts_world_joints.translation(),
            body_quats=body_quats,
            T_world_cpf=T_world_cpf,
            height_from_floor=T_world_cpf[..., 6:7],
            joints_wrt_cpf=Ts_cpf_joints.translation(),
            mask=torch.ones(denoised_traj.contacts.shape[:2], dtype=torch.bool),
            hand_quats=full_hand_quats if full_hand_quats is not None else None,
            visible_joints_mask=None,
        )

        return ProcessingResult(batch, denoised_traj, denoised_data)


class TestRunner:
    """Main class for running the test pipeline."""

    def __init__(self, inference_config: InferenceConfig):
        self.config = inference_config
        self.device = torch.device(inference_config.device)
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize all required components."""
        self.denoiser, self.model_config = load_denoiser(self.config.checkpoint_dir)
        self.denoiser = self.denoiser.to(self.device)

        runtime_config: EgoAlloTrainConfig = load_runtime_config(
            self.config.checkpoint_dir
        )
        self.body_model = fncsmpl.SmplhModel.load(runtime_config.smplh_npz_path).to(
            self.device
        )

        runtime_config.dataset_slice_strategy = "full_sequence"
        runtime_config.train_splits = ("test",)  # Sorry for the naming
        self.test_dataset = EgoAmassHdf5Dataset(runtime_config, cache_files=False)
        # breakpoint()
        self.dataloader = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_dataclass,
            drop_last=False,
        )

    def _prepare_output_dir(self, output_dir: Path) -> None:
        """Prepare output directory by cleaning existing .pt files."""
        if output_dir.exists():
            for pt_file in output_dir.glob("*.pt"):
                pt_file.unlink()
        output_dir.mkdir(exist_ok=True, parents=True)

    def _save_sequence_data(
        self,
        batch: EgoTrainingData,
        traj: EgoTrainingData,
        seq_idx: int,
        output_path: Path,
    ) -> None:
        """Save sequence data for evaluation."""
        assert batch.check_shapes(
            traj
        ), f"Shape mismatch: batch={batch.get_batch_size()} vs traj={traj.get_batch_size()}"

        torch.save(
            {
                # Ground truth data
                "groundtruth_betas": batch.betas[seq_idx, :].cpu(),
                "groundtruth_T_world_root": batch.T_world_root[seq_idx, :].cpu(),
                "groundtruth_body_quats": batch.body_quats[seq_idx, ..., :21, :].cpu(),
                # Sampled/predicted data
                "sampled_betas": traj.betas[0].cpu(),
                "sampled_T_world_root": traj.T_world_root[0].cpu(),
                "sampled_body_quats": traj.body_quats[0, ..., :21, :].cpu(),
            },
            output_path,
        )

        logger.info(f"Saved sequence to {output_path}")

    def _process_batch(
        self,
        batch: EgoTrainingData,
        batch_idx: int,
        processor: SequenceProcessor,
        visualizer: DataVisualizer,
        output_dir: Path,
    ) -> None:
        """Process a batch of sequences."""
        for seq_idx in range(batch.T_world_cpf.shape[0]):
            # try:
            # Process sequence
            result = processor.process_sequence(
                batch, self.denoiser, self.config, self.model_config, self.device
            )

            # Save visualizations if requested
            if self.config.visualize_traj:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                gt_path, inferred_path = visualizer.save_visualization(
                    result.gt_data,
                    result.denoised_data,
                    self.body_model,
                    output_dir,
                    timestamp,
                )
                logger.info(f"Saved ground truth video to: {gt_path}")
                logger.info(f"Saved inferred video to: {inferred_path}")

            # Save sequence data
            output_path = output_dir / f"sequence_{batch_idx}_{seq_idx}.pt"
            self._save_sequence_data(batch, result.denoised_data, seq_idx, output_path)

            # except Exception as e:
            #     logger.error(
            #         f"Error processing sequence {batch_idx}_{seq_idx}: {str(e)}"
            #     )
            #     continue

    def _compute_metrics(self) -> Optional[EgoAlloEvaluationMetrics]:
        """Compute evaluation metrics on processed sequences."""
        try:
            runtime_config = load_runtime_config(self.config.checkpoint_dir)
            evaluator = BodyEvaluator(
                body_model_path=runtime_config.smplh_npz_path, device=self.device
            )

            return evaluator.evaluate_directory(
                dir_with_pt_files=self.config.output_dir,
                use_mean_body_shape=self.config.use_mean_body_shape,
                skip_confirm=self.config.skip_eval_confirm,
            )
        except Exception as e:
            logger.error(f"Error computing metrics: {str(e)}")
            return None

    def run(self) -> Optional[EgoAlloEvaluationMetrics]:
        """Run the test pipeline.

        Returns:
            Dict containing paths to metrics files if metrics computed, None otherwise
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        processor = SequenceProcessor(self.body_model, self.device)
        visualizer = DataVisualizer()

        for batch_idx, batch in enumerate(self.dataloader):
            # if batch_idx == 5:
            #     break

            batch = batch.to(self.device)
            self._process_batch(batch, batch_idx, processor, visualizer, output_dir)

        if self.config.compute_metrics:
            return self._compute_metrics()

        return None


def main(inference_config: InferenceConfig) -> None:
    """Main entry point."""
    try:
        runner = TestRunner(inference_config)
        runner.run()
    except Exception as e:
        logger.error(f"Test run failed: {str(e)}")
        raise


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
