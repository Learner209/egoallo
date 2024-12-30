from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.utils.data
from traitlets import default
import typeguard
from jaxtyping import jaxtyped
from torch.utils.data import DataLoader
from tqdm import tqdm

from egoallo import fncsmpl, fncsmpl_extensions
from egoallo import transforms as tf
from egoallo.config import CONFIG_FILE, make_cfg
from egoallo.config.inference.inference_defaults import InferenceConfig
from egoallo.data import make_batch_collator, build_dataset
from egoallo.config.train.train_config import (
    EgoAlloTrainConfig,
)
from egoallo.joints2smpl.fit_seq import main_call, Joints2SmplFittingConfig
from egoallo.data.dataclass import EgoTrainingData, collate_dataclass
from egoallo.evaluation.body_evaluator import BodyEvaluator
from egoallo.evaluation.metrics import EgoAlloEvaluationMetrics
from egoallo.guidance_optimizer_jax import GuidanceMode
from egoallo.inference_utils import (
    load_denoiser,
    load_runtime_config,
)
from egoallo.network import EgoDenoiser, EgoDenoiserConfig, AbsoluteDenoiseTraj, JointsOnlyTraj, VelocityDenoiseTraj
from egoallo.sampling import (
    CosineNoiseScheduleConstants,
    quadratic_ts,
    run_sampling_with_masked_data,
)
from egoallo.transforms import SE3, SO3
from egoallo.utils.setup_logger import setup_logger
from egoallo.training_utils import ipdb_safety_net

local_config_file = CONFIG_FILE
CFG = make_cfg(config_name="defaults", config_file=local_config_file, cli_args=[])


logger = setup_logger(output="logs/test", name=__name__)


class DataVisualizer:
    """Handles visualization of trajectory data."""

    @staticmethod
    def save_visualization(
        gt_traj: AbsoluteDenoiseTraj,
        denoised_traj: AbsoluteDenoiseTraj,
        body_model: fncsmpl.SmplhModel,
        output_dir: Path,
        output_name: str,
        save_gt: bool = False,
    ) -> Tuple[Path, Path]:
        """Save visualization of ground truth and denoised trajectories."""
        output_dir.mkdir(exist_ok=True, parents=True)

        gt_path = output_dir / f"gt_traj_{output_name}.mp4"
        inferred_path = output_dir / f"inferred_traj_{output_name}.mp4"

        # Visualize ground truth
        if save_gt:
            EgoTrainingData.visualize_ego_training_data(
                gt_traj, body_model, str(gt_path)
            )

        EgoTrainingData.visualize_ego_training_data(
            denoised_traj, body_model, str(inferred_path)
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
        runtime_config: EgoAlloTrainConfig,
        inference_config: InferenceConfig,
        device: torch.device,
    ) -> Tuple[Union[AbsoluteDenoiseTraj, JointsOnlyTraj], Union[AbsoluteDenoiseTraj, JointsOnlyTraj]]:
        """Process a single sequence and return denoised trajectory."""
        # Run denoising with guidance
        # breakpoint()
        denoised_traj = run_sampling_with_masked_data(
            denoiser_network=denoiser,
            body_model=self.body_model,
            masked_data=batch,
            runtime_config=runtime_config,
            guidance_mode=inference_config.guidance_mode,
            guidance_post=inference_config.guidance_post,
            guidance_inner=inference_config.guidance_inner,
            floor_z=0.0,
            hamer_detections=None,
            aria_detections=None,
            num_samples=1,
            device=self.device,
        )
        gt_traj = batch.to_denoise_traj(denoising_config=runtime_config.denoising)
        return gt_traj, denoised_traj


class TestRunner:
    """Main class for running the test pipeline."""

    def __init__(self, inference_config: InferenceConfig):
        self.inference_config = inference_config
        self.device = torch.device(inference_config.device)
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize all required components."""
        runtime_config: EgoAlloTrainConfig = load_runtime_config(
            self.inference_config.checkpoint_dir
        )
        self.runtime_config = runtime_config
        self.denoiser, self.model_config = load_denoiser(self.inference_config.checkpoint_dir, runtime_config)
        self.denoiser = self.denoiser.to(self.device)

        self.body_model = fncsmpl.SmplhModel.load(runtime_config.smplh_npz_path).to(
            self.device
        )
        # Override runtime config with inference config values
        for field in dataclasses.fields(type(self.inference_config)):
            if hasattr(runtime_config, field.name):
                setattr(runtime_config, field.name, getattr(self.inference_config, field.name))

        self.dataloader = torch.utils.data.DataLoader(
            dataset=build_dataset(cfg=runtime_config)(config=runtime_config),
            batch_size=1,
            shuffle=False,
            # num_workers=runtime_config.num_workers,
            num_workers=0,
            # persistent_workers=runtime_config.num_workers > 0,
            pin_memory=True,
            collate_fn=make_batch_collator(runtime_config),
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
        gt_traj: Union[AbsoluteDenoiseTraj, JointsOnlyTraj, VelocityDenoiseTraj],
        denoised_traj: Union[AbsoluteDenoiseTraj, JointsOnlyTraj, VelocityDenoiseTraj], 
        seq_idx: int,
        output_path: Path,
    ) -> None:
        """Save sequence data for evaluation."""
        save_dict = {}

        if isinstance(gt_traj, (AbsoluteDenoiseTraj, VelocityDenoiseTraj)):
            # Convert rotation matrices to quaternions for saving
            denoised_body_quats = SO3.from_matrix(denoised_traj.body_rotmats).wxyz # type: ignore
            gt_body_quats = SO3.from_matrix(gt_traj.body_rotmats).wxyz # type: ignore

            save_dict.update({
                # Ground truth data
                "groundtruth_betas": gt_traj.betas[seq_idx, :]
                .mean(dim=0, keepdim=True)
                .cpu(),
                "groundtruth_T_world_root": SE3.from_rotation_and_translation(
                    SO3.from_matrix(gt_traj.R_world_root[seq_idx]),
                    gt_traj.t_world_root[seq_idx],
                )
                .parameters()
                .cpu(),
                "groundtruth_body_quats": gt_body_quats[seq_idx, ..., :21, :].cpu(),
                # Denoised trajectory data
                "sampled_betas": denoised_traj.betas.mean(dim=1, keepdim=True).cpu(),
                "sampled_T_world_root": SE3.from_rotation_and_translation(
                    SO3.from_matrix(denoised_traj.R_world_root),
                    denoised_traj.t_world_root,
                )
                .parameters()
                .cpu(),
                "sampled_body_quats": denoised_body_quats[..., :21, :].cpu(),
            })

        elif isinstance(gt_traj, JointsOnlyTraj):
            save_dict.update({
                # Ground truth data
                "groundtruth_joints": gt_traj.joints[seq_idx].cpu(),
                # Denoised trajectory data  
                "sampled_joints": denoised_traj.joints.cpu(),
            })

        torch.save(save_dict, output_path)

    def _process_batch(
        self,
        batch: EgoTrainingData,
        batch_idx: int,
        processor: SequenceProcessor,
        visualizer: DataVisualizer,
        output_dir: Path,
        save_gt_vis: bool = False,
        output_name: Optional[str] = None,
    ) -> None:
        """Process a batch of sequences."""
        for seq_idx in range(batch.T_world_cpf.shape[0]):
            # Process sequence to get denoised trajectory
            gt_traj, denoised_traj = processor.process_sequence(
                batch, self.denoiser, self.runtime_config, self.inference_config, self.device
            )
            if self.runtime_config.denoising.denoising_mode == "joints_only":
                # import ipdb; ipdb.set_trace()
                denoised_traj = denoised_traj[seq_idx]
                gt_traj = gt_traj[seq_idx]
                # main_call(Joints2SmplFittingConfig(), denoised_traj.joints.shape[0], denoised_traj.joints.cpu(), output_dir)
                main_call(Joints2SmplFittingConfig(), gt_traj.joints.shape[0], gt_traj.joints.cpu(), output_dir)

            # Save visualizations if requested
            if self.inference_config.visualize_traj and self.runtime_config.denoising.denoising_mode != "joints_only":
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                gt_path, inferred_path = visualizer.save_visualization(
                    gt_traj[seq_idx],
                    denoised_traj[seq_idx],
                    self.body_model,
                    output_dir,
                    output_name
                    if output_name
                    else f"sequence_{batch_idx}_{seq_idx}.mp4",
                    save_gt=save_gt_vis,
                )
                logger.info(f"pred_path: {inferred_path}")
                if save_gt_vis:
                    logger.info(f"gt_path: {gt_path}")

            # Save sequence data
            assert (
                output_name is not None
                and output_name.endswith(".pt")
                or output_name is None
            )
            filename = (
                output_name if output_name else f"sequence_{batch_idx}_{seq_idx}.pt"
            )
            output_path = output_dir / filename
            if self.runtime_config.denoising.denoising_mode != "joints_only":
                self._save_sequence_data(gt_traj, denoised_traj, seq_idx, output_path)

    def _compute_metrics(
        self, dir_with_pt_files: Path
    ) -> Optional[EgoAlloEvaluationMetrics]:
        """Compute evaluation metrics on processed sequences."""
        # try:
        evaluator = BodyEvaluator(
            body_model_path=self.runtime_config.smplh_npz_path, device=self.device
        )

        return evaluator.evaluate_directory(
            runtime_config=self.runtime_config,
            dir_with_pt_files=dir_with_pt_files,
            use_mean_body_shape=self.inference_config.use_mean_body_shape,
            # Initialize experiment.
            skip_confirm=self.inference_config.skip_eval_confirm,
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
                if batch_idx == 5:
                    break

                batch = batch.to(self.device)
                save_gt_vis = (
                    False if self.inference_config.dataset_type == "EgoExoDataset" else True
                )
                self._process_batch(
                    batch,
                    batch_idx,
                    processor,
                    visualizer,
                    temp_output_dir,
                    save_gt_vis=save_gt_vis,
                )

            # TODO: this is a hack to avoid computing metrics for EgoExo dataset.
            # breakpoint()
            if (
                self.inference_config.compute_metrics
                and self.inference_config.dataset_type != "EgoExoDataset"
                and self.runtime_config.denoising.denoising_mode != "joints_only"
            ):
                # Create final output directory for saving metrics
                final_output_dir = Path(self.inference_config.output_dir)
                final_output_dir.mkdir(exist_ok=True, parents=True)

                # Compute metrics using temp files and save to final directory
                metrics = self._compute_metrics(temp_output_dir)
                if metrics:
                    metrics.save(final_output_dir)
                return metrics
            elif self.runtime_config.denoising.denoising_mode == "joints_only":
                return None

        return None


def main(inference_config: InferenceConfig) -> None:
    """Main entry point."""
    try:
        runner = TestRunner(inference_config)
        eval_metrics = runner.run()
        eval_metrics.print_metrics(logger=logger, level="info")
    except Exception as e:
        logger.error(f"Test run failed: {str(e)}")
        raise


if __name__ == "__main__":
    import tyro

    ipdb_safety_net()

    tyro.cli(main)
