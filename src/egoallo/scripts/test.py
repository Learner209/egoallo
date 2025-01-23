from __future__ import annotations

import dataclasses
import copy
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, TYPE_CHECKING

import torch
import torch.utils.data
import typeguard
from jaxtyping import jaxtyped
from torch.utils.data import DataLoader
from tqdm import tqdm
import multiprocessing
import subprocess
import os
from multiprocessing import Process
if TYPE_CHECKING:
	from egoallo.types import DenoiseTrajType
	from egoallo.egoexo.egoexo_utils import EgoExoUtils

from egoallo import fncsmpl, fncsmpl_extensions
from egoallo import transforms as tf
from egoallo.config import CONFIG_FILE, make_cfg
from egoallo.config.inference.inference_defaults import InferenceConfig
from egoallo.data import make_batch_collator, build_dataset
from egoallo.config.train.train_config import EgoAlloTrainConfig
from egoallo.joints2smpl.fit_seq import joints2smpl_fit_seq, Joints2SmplFittingConfig
from egoallo.data.dataclass import EgoTrainingData
from egoallo.evaluation.body_evaluator import BodyEvaluator
from egoallo.evaluation.metrics import EgoAlloEvaluationMetrics
from egoallo.guidance_optimizer_jax import GuidanceMode
from egoallo.inference_utils import (
	load_denoiser,
	load_runtime_config,
)
from egoallo.network import (
	EgoDenoiser,
	EgoDenoiserConfig,
	AbsoluteDenoiseTraj,
	JointsOnlyTraj,
	VelocityDenoiseTraj,
)
from egoallo.sampling import (
	CosineNoiseScheduleConstants,
	quadratic_ts,
	run_sampling_with_masked_data,
)
from egoallo.transforms import SE3, SO3
from egoallo.utils.setup_logger import setup_logger
from egoallo.training_utils import ipdb_safety_net
# from egoallo.egoexo import EGOEXO_UTILS_INST

torch.multiprocessing.set_sharing_strategy('file_system')

local_config_file = CONFIG_FILE
CFG = make_cfg(config_name="defaults", config_file=local_config_file, cli_args=[])


logger = setup_logger(output="logs/test", name=__name__)


# Some helper functions to ensure compatibility with multiprocessing module, used in `TestRunner.run` class.
# reference: https://stackoverflow.com/questions/72766345/attributeerror-cant-pickle-local-object-in-multiprocessing
# ! FIXME: the `compute_single_metrics` function is set as top-level functions to ensure compatibility with multiprocessing module, since the latter relies on pickling objects to work.
def compute_single_metrics(
	*, 
	gt_traj: DenoiseTrajType,
	est_traj: DenoiseTrajType, 
	body_model: fncsmpl.SmplhModel,
	device: torch.device
) -> Dict[str, float]:
	return est_traj._compute_metrics(gt_traj, body_model=body_model, device=device)

def save_single_traj(*, 
	traj: DenoiseTrajType,
	take_name: str, 
	is_gt: bool,
	processor: SequenceProcessor,
	output_dir: Path
) -> None:
	prefix = "gt" if is_gt else "est"
	save_path = output_dir / f"{prefix}_{take_name}.pt"
	processor.save_sequence(traj=traj, output_path=save_path)


def compute_single_metrics_helper(kwargs: Dict[str, Union[DenoiseTrajType, fncsmpl.SmplhModel, torch.device]]) -> Dict[str, float]:
	return compute_single_metrics(**kwargs)

def save_single_traj_helper(kwargs: Dict[str, Union[DenoiseTrajType, str, bool, SequenceProcessor, Path]]) -> None:
	return save_single_traj(**kwargs)

class SequenceProcessor:
	"""Handles processing of individual sequences."""

	def __init__(self, body_model: fncsmpl.SmplhModel, device: torch.device):
		self.body_model = body_model 
		self.device = device

	def save_sequence(self, traj: DenoiseTrajType, output_path: Path) -> None:
		"""Save trajectory data to disk.
		
		Args:
			traj: Trajectory data to save
			output_path: Path to save the trajectory file
		"""
		# Create parent directories if they don't exist
		output_path.parent.mkdir(parents=True, exist_ok=True)
		
		# Move trajectory to CPU using map function with force=True
		traj_cpu = traj.map(lambda x: x.cpu().detach() if isinstance(x, torch.Tensor) else x)
		torch.save(traj_cpu, output_path)

	def process_sequence(
		self,
		batch: EgoTrainingData,
		denoiser: EgoDenoiser, 
		runtime_config: EgoAlloTrainConfig,
		inference_config: InferenceConfig,
		device: torch.device,
	) -> Tuple[DenoiseTrajType, DenoiseTrajType]:
		"""Process a single sequence and return denoised trajectory."""
		# Run denoising with guidance
		denoised_traj = run_sampling_with_masked_data(
			denoiser_network=denoiser,
			body_model=self.body_model,
			masked_data=copy.deepcopy(batch),
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
		gt_traj = runtime_config.denoising.from_ego_data(batch, include_hands=True)
		

		
		post_batch = batch.postprocess()
		# no need to postprocess denoised_traj since its' already been postprocessed.
		denoised_traj = post_batch._set_traj(denoised_traj)
		gt_traj = post_batch._post_process(gt_traj)
		gt_traj = post_batch._set_traj(gt_traj)

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
		self.denoiser, self.model_config = load_denoiser(
			self.inference_config.checkpoint_dir, runtime_config
		)
		self.denoiser = self.denoiser.to(self.device)

		self.body_model = fncsmpl.SmplhModel.load(runtime_config.smplh_npz_path).to(
			self.device
		)
		# ! Override runtime config with inference config values
		for field in dataclasses.fields(type(self.inference_config)):
			if hasattr(runtime_config, field.name):
				setattr(
					runtime_config,
					field.name,
					getattr(self.inference_config, field.name),
				)

		# Set collate function based on dataset type
		if runtime_config.dataset_type == "AriaDataset" or runtime_config.dataset_type == "AriaInferenceDataset":
			# runtime_config.data_collate_fn = "DefaultBatchCollator"
			runtime_config.data_collate_fn = "TensorOnlyDataclassBatchCollator"
			ds_init_config = self.inference_config.egoexo
		else:
			runtime_config.data_collate_fn = "TensorOnlyDataclassBatchCollator"
			ds_init_config = runtime_config
			# ds_init_config.splits = ("train", "val")
			
		# ! Temporal masking is disabled for testing, since it can cause RuntimeError no frames found within visible joints.
		runtime_config.temporal_mask_ratio = 0.0
	 
		self.dataloader = torch.utils.data.DataLoader(
			dataset=build_dataset(cfg=runtime_config)(config=ds_init_config),
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
		gt_traj: DenoiseTrajType,
		denoised_traj: DenoiseTrajType,
		seq_idx: int,
		output_path: Path,
	) -> None:
		"""Save sequence data for evaluation."""
		save_dict = {}

		if isinstance(gt_traj, (AbsoluteDenoiseTraj, VelocityDenoiseTraj)):
			# Convert rotation matrices to quaternions for saving
			denoised_body_quats = SO3.from_matrix(denoised_traj.body_rotmats).wxyz  # type: ignore
			gt_body_quats = SO3.from_matrix(gt_traj.body_rotmats).wxyz  # type: ignore

			save_dict.update(
				{
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
					"sampled_betas": denoised_traj.betas.mean(
						dim=1, keepdim=True
					).cpu(),
					"sampled_T_world_root": SE3.from_rotation_and_translation(
						SO3.from_matrix(denoised_traj.R_world_root),
						denoised_traj.t_world_root,
					)
					.parameters()
					.cpu(),
					"sampled_body_quats": denoised_body_quats[..., :21, :].cpu(),
				}
			)

		elif isinstance(gt_traj, JointsOnlyTraj):
			save_dict.update(
				{
					# Ground truth data
					"groundtruth_joints": gt_traj.joints[seq_idx].cpu(),
					# Denoised trajectory data
					"sampled_joints": denoised_traj.joints.cpu(),
				}
			)

		torch.save(save_dict, output_path)

	def _process_batch(
		self,
		batch: EgoTrainingData,
		batch_idx: int,
		processor: SequenceProcessor,
	) -> Tuple[DenoiseTrajType, DenoiseTrajType]:
		"""Process a batch of sequences."""

		gt_trajs = None  # shape: (batch_size, num_timesteps, ...)
		denoised_trajs = None  # shape: (num_samples==batch_size, num_timesteps, ...)

		for seq_idx in range(batch.T_world_cpf.shape[0]):
			torch.cuda.empty_cache()
			# Process sequence to get denoised trajectory
			gt_traj, denoised_traj = processor.process_sequence(
				batch,
				self.denoiser,
				self.runtime_config,
				self.inference_config,
				self.device,
			)
			# convert to cpu to avoid GPU OOM.
			gt_traj = gt_traj.to(torch.device("cpu"))
			denoised_traj = denoised_traj.to(torch.device("cpu"))

			if gt_trajs is None:
				gt_trajs = gt_traj
			else:
				gt_trajs = gt_trajs._dict_map(lambda key, value: torch.cat([value, getattr(gt_traj, key)], dim=0) if isinstance(value, torch.Tensor) else value)

			if denoised_trajs is None:
				denoised_trajs = denoised_traj
			else:
				denoised_trajs = denoised_trajs._dict_map(lambda key, value: torch.cat([value, getattr(denoised_traj, key)], dim=0) if isinstance(value, torch.Tensor) else value)

			# metrics = denoised_trajs._compute_metrics(
			#     gt_trajs, body_model=self.body_model, device=self.device)
			# metrics = EgoAlloEvaluationMetrics(**metrics)

			
			# TODO: this is previous implementation of visualization routine, which would cause GPU OOM.
			# if self.runtime_config.denoising.denoising_mode == "joints_only":
			#     # import ipdb; ipdb.set_trace()
			#     denoised_traj = denoised_traj[seq_idx]
			#     # joints2smpl_fit_seq(Joints2SmplFittingConfig(), self.body_model, denoised_traj.joints.shape[0], denoised_traj.joints.cpu(), output_dir)
			#     
			#     fit_seq_data: "EgoTrainingData" = joints2smpl_fit_seq(
			#         Joints2SmplFittingConfig(),
			#         self.body_model,
			#         gt_traj.joints.shape[0],
			#         gt_traj.joints.cpu(),
			#         output_dir,
			#     )
			#     # FIXME: this is a temporary fix to visualize the fit_seq_traj, set denoising mode to absolute to use from_ego_data function from `AbsoluteDenoiseTraj`.
			#     _ = self.runtime_config.denoising.denoising_mode
			#     self.runtime_config.denoising.denoising_mode = "absolute"
			#     fit_seq_traj = self.runtime_config.denoising.from_ego_data(
			#         fit_seq_data, include_hands=True
			#     )
			#     self.runtime_config.denoising.denoising_mode = _

			#     fit_seq_traj = fit_seq_traj.map(lambda x: x.to(self.device))
			#     self.body_model = self.body_model.to(self.device)

			#     output_path = Path("./exp/debug_frame_rate_diff/")
			#     output_path.mkdir(parents=True, exist_ok=True)

			#     EgoTrainingData.visualize_ego_training_data(
			#         # fit_seq_traj,
			#         gt_traj,
			#         self.body_model,
			#         output_path=str(output_path / "fit_seq_traj.mp4"),
			#     )

			# TODO: this is previous implementation of visualization routine, which would cause GPU OOM.
			# Save visualizations if requested
			# if (
			#     self.inference_config.visualize_traj
			#     and self.runtime_config.denoising.denoising_mode != "joints_only"
			# ):
			#     logger.info(f"this take name is: {batch.take_name}")
			#     timestamp = time.strftime("%Y%m%d-%H%M%S")

			#     # Adaptive sampling if sequence is longer than 1500 frames
			#     denoised_traj = denoised_traj[seq_idx]
			#     if len(denoised_traj.t_world_root) > 1500:
			#         # Calculate stride to get under 1500 frames
			#         stride = len(denoised_traj.t_world_root) // 1499 + 1
			#         denoised_traj = denoised_traj[::stride]
			#     # Run visualization in a separate process
			#     import multiprocessing
			#     # TODO: this is a temporary fix to avoid OOM of GPU memory as the OpenGL python bindings context would eat up GPU MeM too fast.
				
			#     breakpoint()
			#     vis_process = multiprocessing.Process(
			#         target=visualizer.save_visualization,
			#         args=(
			#             gt_traj[seq_idx],
			#             denoised_traj,
			#             self.body_model,
			#             output_dir,
			#             output_name
			#             if output_name
			#             else f"sequence_{batch_idx}_{seq_idx}.mp4",
			#             save_gt_vis,
			#         ),
			#     )
			#     vis_process.start()
			#     vis_process.join()  # Wait for visualization to complete

			#     # Construct paths that would have been returned
			#     gt_path = (
			#         output_dir
			#         / f"gt_traj_{output_name if output_name else f'sequence_{batch_idx}_{seq_idx}.mp4'}"
			#     )
			#     inferred_path = (
			#         output_dir
			#         / f"inferred_traj_{output_name if output_name else f'sequence_{batch_idx}_{seq_idx}.mp4'}"
			#     )

			#     logger.info(f"pred_path: {inferred_path}")
			#     if save_gt_vis:
			#         logger.info(f"gt_path: {gt_path}")

			# Save sequence data
			# assert (
			#     output_name is not None
			#     and output_name.endswith(".pt")
			#     or output_name is None
			# )
			# filename = (
			#     output_name if output_name else f"sequence_{batch_idx}_{seq_idx}.pt"
			# )
			# output_path = output_dir / filename
			# if self.runtime_config.denoising.denoising_mode != "joints_only":
			#     self._save_sequence_data(gt_traj, denoised_traj, seq_idx, output_path)

		# processor.save_sequence(gt_trajs, output_dir / f"gt_traj_{output_name}.pt")
		# processor.save_sequence(denoised_trajs, output_dir / f"denoised_traj_{output_name}.pt")

		return gt_trajs, denoised_trajs

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
		"""Run the test pipeline."""
		import tempfile
		import shutil

		# Add debug logging
		if self.inference_config.debug_max_iters:
			logger.warning(f"Running in debug mode with max {self.inference_config.debug_max_iters} iterations")

		processor = SequenceProcessor(self.body_model, self.device)

		# Create temporary directory for intermediate files
		with tempfile.TemporaryDirectory() as temp_dir:
			temp_output_dir = Path(temp_dir)
			# temp_output_dir = Path("./exp/test-debug-too-many-open-files")

			gt_trajs = []
			denoised_trajs = []
			identifiers = []

			for batch_idx, batch in tqdm(
				enumerate(self.dataloader),
				total=len(self.dataloader),
				desc="Enumerating test loader",
				ascii=" >=",
			):
				# Add debug iteration limit check
				if self.inference_config.debug_max_iters and batch_idx >= self.inference_config.debug_max_iters:
					logger.warning(f"Stopping after {batch_idx} iterations due to debug_max_iters={self.inference_config.debug_max_iters}")
					break

				batch.metadata.scope = "test"

				batch = batch.to(self.device)
				assert batch.metadata.stage == "preprocessed", f"Expected preprocessed data, got {batch.metadata.stage}"

				temp_output_dir.mkdir(parents=True, exist_ok=True)
				gt_traj, denoised_traj = self._process_batch(
					batch,
					batch_idx,
					processor,
				)

				# TODO: the current implementation assumes that the leading `TensorDataClass` batch size dim() returns `1`.
				gt_trajs.append(gt_traj)
				denoised_trajs.append(denoised_traj)
				identifiers.append(batch.metadata.take_name[0][0])

				torch.cuda.empty_cache()
			# Prepare arguments for parallel metric computation
			metric_args = [
				({
					'gt_traj': gt_traj,
					'est_traj': est_traj,
					'body_model': self.body_model.to(torch.device("cpu")),
					'device': torch.device("cpu")
				},)
				for gt_traj, est_traj in zip(gt_trajs, denoised_trajs)
			]

			# Compute metrics in parallel using DillProcess
			with torch.multiprocessing.get_context('spawn').Pool(processes=1) as pool:
				trajectory_metrics = pool.starmap(
					compute_single_metrics_helper,
					metric_args
				)
			
			# parallel compute metrics for debugging.
			# for gt_traj, est_traj in zip(gt_trajs, denoised_trajs):
			#     metrics = gt_traj._compute_metrics(est_traj, body_model=self.body_model, device=torch.device("cpu"))

			# Aggregate metrics across all trajectories
			aggregated_metrics = {}
			for metric_dict in trajectory_metrics:
				for metric_name, value in metric_dict.items():
					if metric_name not in aggregated_metrics:
						aggregated_metrics[metric_name] = []
					aggregated_metrics[metric_name].append(value)

			# Calculate mean and std for each metric
			final_metrics = {}
			for metric_name, values in aggregated_metrics.items():
				values_tensor = torch.FloatTensor(values)
				final_metrics[metric_name] = values_tensor.mean()
				final_metrics[f"{metric_name}_std"] = values_tensor.std()

			# Create evaluation metrics object with aggregated results
			metrics = EgoAlloEvaluationMetrics(**final_metrics)
			# Prepare arguments for parallel saving
			save_args = []

			for gt_traj, est_traj, take_name in zip(gt_trajs, denoised_trajs, identifiers):
				save_args.append(({
					"traj": gt_traj[0],
					"take_name": take_name,
					"is_gt": True,
					"processor": processor,
					"output_dir": temp_output_dir / take_name
				},))
				save_args.append(({
					"traj": est_traj[0],
					"take_name": take_name,
					"is_gt": False,
					"processor": processor,
					"output_dir": temp_output_dir / take_name
				},))

			# Execute saves in parallel
			with multiprocessing.get_context('spawn').Pool(processes=20) as pool:
				pool.starmap(save_single_traj_helper, save_args)

			# Run visualizations using subprocess for each trajectory
			if self.inference_config.visualize_traj:
				denoise_traj_type: str = self.runtime_config.denoising._repr_denoise_traj_type()
				for take_name in identifiers:
					gt_path = temp_output_dir / take_name / f"gt_{take_name}.pt"
					est_path = temp_output_dir / take_name / f"est_{take_name}.pt" 
					# egoexo_utils: EgoExoUtils = EGOEXO_UTILS_INST
					# Parse out take_uid from take_name
					take_uid = take_name.split("uid_")[1].split("_t")[0]
					this_take_name = take_name.split("name_")[1].split("_uid_")[0]
					
					this_take_path = Path(self.inference_config.egoexo_dataset_path) / "takes" / Path(this_take_name)

					cmd = [
						"python",
						"src/egoallo/scripts/visualize_inference.py",
						"--trajectory-path", str(gt_path), str(est_path),
						"--trajectory-type", denoise_traj_type,
						"--smplh-model-path", str(self.runtime_config.smplh_npz_path),
						"--output-dir", str(temp_output_dir / take_name),
						"--dataset-type", self.inference_config.dataset_type,
						"--config.egoexo.traj_root", str(this_take_path)
					]
					
					# Remove empty arguments
					cmd = [arg for arg in cmd if arg]
					logger.info(f"Running command: {' '.join(cmd)}")
					
					
					# Call visualization process
					subprocess.call(cmd, env=os.environ.copy())

			# After all operations complete successfully, copy temp dir contents to persistent location
			persistent_output_dir = Path(self.inference_config.output_dir)
			persistent_output_dir.mkdir(parents=True, exist_ok=True)
			
			# Move contents from temp dir to persistent dir, overwriting existing files
			for item in temp_output_dir.glob("*"):
				dest = persistent_output_dir / item.name
				if dest.exists():
					if dest.is_file():
						dest.unlink()
					else:
						shutil.rmtree(dest)
				shutil.move(str(item), str(dest))

		return metrics


def main(inference_config: InferenceConfig, debug: bool = False) -> None:
	"""Main entry point."""
	# try:
	if debug:
		import ipdb; ipdb.set_trace()
		
	runner = TestRunner(inference_config)
	eval_metrics = runner.run()
	eval_metrics.print_metrics(logger=logger, level="info")
	# except Exception as e:
	#     logger.error(f"Test run failed: {str(e)}")
	#     raise


if __name__ == "__main__":
	import tyro

	ipdb_safety_net()
	tyro.cli(main)
