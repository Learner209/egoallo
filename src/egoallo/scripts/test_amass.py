from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from typing_extensions import runtime

import torch
import torch.utils.data
import typeguard
from jaxtyping import jaxtyped
from torch.utils.data import DataLoader
from tqdm import tqdm

from egoallo import fncsmpl, fncsmpl_extensions
from egoallo import transforms as tf
from egoallo.config import CONFIG_FILE, make_cfg
from egoallo.config.inference.inference_defaults import InferenceConfig
from egoallo.data.amass import EgoAmassHdf5Dataset
from egoallo.config.train.train_config import EgoAlloTrainConfig
from egoallo.data.dataclass import EgoTrainingData, collate_dataclass
from egoallo.evaluation.body_evaluator import BodyEvaluator
from egoallo.evaluation.metrics import EgoAlloEvaluationMetrics
from egoallo.guidance_optimizer_jax import GuidanceMode
from egoallo.inference_utils import (
    load_denoiser,
    _load_denoiser,
    load_runtime_config,
)
from egoallo.training_utils import ipdb_safety_net
from egoallo.network import EgoDenoiser, EgoDenoiserConfig, EgoDenoiseTraj
from egoallo.sampling import (
    CosineNoiseScheduleConstants,
    quadratic_ts,
    run_sampling_with_stitching,
)
from egoallo.transforms import SE3, SO3
from egoallo.utils.setup_logger import setup_logger

local_config_file = CONFIG_FILE
CFG = make_cfg(config_name="defaults", config_file=local_config_file, cli_args=[])


logger = setup_logger(output="logs/test", name=__name__)


class DataVisualizer:
    """Handles visualization of trajectory data."""

    @staticmethod
    def save_visualization(
        gt_data: EgoTrainingData,
        denoised_traj: EgoDenoiseTraj,
        body_model: fncsmpl.SmplhModel,
        output_dir: Path,
        timestamp: str,
    ) -> Tuple[Path, Path]:
        """Save visualization of ground truth and denoised trajectories."""
        output_dir.mkdir(exist_ok=True, parents=True)

        # TODO: hack way to debug.
        output_dir = Path("./exp/amass_visualization")
        output_dir.mkdir(parents=True, exist_ok=True)
        gt_path = output_dir / f"gt_traj_{timestamp}.mp4"
        inferred_path = output_dir / f"inferred_traj_{timestamp}.mp4"

        # Visualize ground truth
        # FIXME: the visualizeation utilities now accepts the `EgoDenoiseTraj` object, a `EgoTrainingData` object is used here.
        # EgoTrainingData.visualize_ego_training_data(gt_data, body_model, str(gt_path))

        EgoTrainingData.visualize_ego_training_data(
            gt_data, denoised_traj, body_model, str(inferred_path)
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
    ) -> EgoDenoiseTraj:
        """Process a single sequence and return denoised trajectory."""
        # Run denoising with guidance
        # breakpoint()
        denoised_traj: EgoDenoiseTraj = run_sampling_with_stitching(
            denoiser_network=denoiser,
            body_model=self.body_model,
            guidance_mode=inference_config.guidance_mode,
            guidance_post=inference_config.guidance_post,
            guidance_inner=inference_config.guidance_inner,
            Ts_world_cpf=batch.T_world_cpf[0],
            hamer_detections=None,
            aria_detections=None,
            num_samples=1,
            device=self.device,
            floor_z=0.0,
        )
        # denoised_traj.betas = denoised_traj.betas.mean(
        #     dim=1, keepdim=True
        # )  # average across time dimensions
        return denoised_traj


class TestRunner:
    """Main class for running the test pipeline."""

    config: InferenceConfig
    device: torch.device
    denoiser: EgoDenoiser
    model_config: EgoDenoiserConfig
    body_model: fncsmpl.SmplhModel
    test_dataset: EgoAmassHdf5Dataset
    dataloader: DataLoader

    def __init__(self, inference_config: InferenceConfig):
        self.config = inference_config
        self.device = torch.device(inference_config.device)
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize all required components."""
        self.denoiser, self.model_config = _load_denoiser(self.config.checkpoint_dir)
        self.denoiser = self.denoiser.to(self.device)

        runtime_config: EgoAlloTrainConfig = load_runtime_config(
            self.config.checkpoint_dir
        )
        self.body_model = fncsmpl.SmplhModel.load(runtime_config.smplh_npz_path).to(
            self.device
        )

        # runtime_config.dataset_slice_strategy = "full_sequence"
        runtime_config.train_splits = ("val",)  # Sorry for the naming
        self.test_dataset = EgoAmassHdf5Dataset(
            hdf5_path=runtime_config.dataset_hdf5_path,
            file_list_path=runtime_config.dataset_files_path,
            splits=runtime_config.train_splits,
            subseq_len=runtime_config.subseq_len,
            cache_files=False,
            slice_strategy=runtime_config.dataset_slice_strategy,
            min_subseq_len=runtime_config.subseq_len,
        )
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
        denoised_traj: EgoDenoiseTraj,
        body_model: fncsmpl.SmplhModel,
        seq_idx: int,
        output_path: Path,
    ) -> None:
        """Save sequence data for evaluation."""
        # TODO: change this super hacky way to modify pred and gt data in the saving procedure.
        # Convert rotation matrices to quaternions for saving
        body_quats = SO3.from_matrix(denoised_traj.body_rotmats).wxyz
        posed = denoised_traj.apply_to_body(body_model)
        Ts_world_root = fncsmpl_extensions.get_T_world_root_from_cpf_pose(
            posed, batch.T_world_cpf[..., 1:, :]
        )
        # breakpoint()
        torch.save(
            {
                # Ground truth data
                "groundtruth_betas": batch.betas[
                    seq_idx, 1:, :
                ].cpu(),  # timesteps-1 16
                "groundtruth_T_world_root": batch.T_world_root[
                    seq_idx, 1:, :
                ].cpu(),  # (timesteps-1, 7)
                "groundtruth_body_quats": batch.body_quats[
                    seq_idx, 1:, :21, :
                ].cpu(),  # (timesteps-1, 21, 4)
                # Denoised trajectory data
                "sampled_betas": denoised_traj.betas.cpu(),  # (num_samples, timesteps-1, 16)
                "sampled_T_world_root": Ts_world_root,  # (num_samples, timesteps-1, 7)
                "sampled_body_quats": body_quats[
                    ..., :21, :
                ].cpu(),  # (num_samples, timesteps-1, 21, 4)
            },
            output_path,
        )

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
            # Process sequence to get denoised trajectory
            denoised_traj = processor.process_sequence(
                batch, self.denoiser, self.config, self.model_config, self.device
            )

            # Save visualizations if requested
            if self.config.visualize_traj:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                gt_path, inferred_path = visualizer.save_visualization(
                    batch[seq_idx],
                    denoised_traj[seq_idx],
                    self.body_model,
                    output_dir,
                    timestamp,
                )
                logger.info(f"gt_path: {gt_path}, pred_path: {inferred_path}")

            # Save sequence data
            output_path = output_dir / f"sequence_{batch_idx}_{seq_idx}.pt"
            self._save_sequence_data(
                batch, denoised_traj, self.body_model, seq_idx, output_path
            )

    def _compute_metrics(
        self, dir_with_pt_files: Path
    ) -> Optional[EgoAlloEvaluationMetrics]:
        """Compute evaluation metrics on processed sequences."""
        # try:
        runtime_config = load_runtime_config(self.config.checkpoint_dir)
        evaluator = BodyEvaluator(
            body_model_path=runtime_config.smplh_npz_path, device=self.device
        )

        return evaluator.evaluate_directory(
            dir_with_pt_files=dir_with_pt_files,
            use_mean_body_shape=self.config.use_mean_body_shape,
            skip_confirm=self.config.skip_eval_confirm,
        )
        # except Exception as e:
        #     logger.error(f"Error computing metrics: {str(e)}")
        #     return None

    def run(self) -> Optional[EgoAlloEvaluationMetrics]:
        """Run the test pipeline.

        Returns:
            Dict containing paths to metrics files if metrics computed, None otherwise
        """
        import tempfile

        processor = SequenceProcessor(self.body_model, self.device)
        visualizer = DataVisualizer()

        # Create temporary directory for intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_output_dir = Path(temp_dir)

            for batch_idx, batch in tqdm(
                enumerate(self.dataloader),
                total=len(self.dataloader),
                desc="Enumerating test loader",
                ascii=" >=",
            ):
                # if batch_idx == 5:
                #     break

                batch = batch.to(self.device)
                self._process_batch(
                    batch, batch_idx, processor, visualizer, temp_output_dir
                )

            if self.config.compute_metrics:
                # Create final output directory for saving metrics
                final_output_dir = Path(self.config.output_dir)
                final_output_dir.mkdir(exist_ok=True, parents=True)

                # Compute metrics using temp files and save to final directory
                metrics = self._compute_metrics(temp_output_dir)
                if metrics:
                    metrics.save(final_output_dir)
                return metrics

        return None


def main(inference_config: InferenceConfig) -> None:
    # """Main entry point."""
    # try:
    ipdb_safety_net()
    runner = TestRunner(inference_config)
    eval_metrics = runner.run()
    eval_metrics.print_metrics(logger=logger, level="info")
    # except Exception as e:
    # logger.error(f"Test run failed: {str(e)}")
    # raise


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
