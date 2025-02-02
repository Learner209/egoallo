"""Training script for EgoAllo diffusion model using HuggingFace accelerate."""

import os

from torch._dynamo import eval_frame

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import dataclasses
import shutil
from pathlib import Path
from typing import Literal
import time

import sys
import pytest
from pathlib import Path
import train_motion_prior
from unittest.mock import patch

import torch.optim.lr_scheduler
import torch.utils.data
import tyro
import yaml
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import ProjectConfiguration
from loguru import logger

from jaxtyping import install_import_hook

# Install hook before importing any modules you want to typecheck
# with install_import_hook("egoallo", "typeguard.typechecked"):
from egoallo import network, training_loss, training_utils
from egoallo.data import make_batch_collator, build_dataset
from egoallo.config.train.train_config import EgoAlloTrainConfig
from egoallo.utils.utils import make_source_code_snapshot
from egoallo.utils.setup_logger import setup_logger

import wandb
import datetime
import tempfile
from egoallo.config.inference.inference_defaults import InferenceConfig
from egoallo.scripts.test import TestRunner
import json
import numpy as np


def get_experiment_dir(experiment_name: str, version: int = 0) -> Path:
    """Creates a directory to put experiment files in, suffixed with a version
    number. Similar to PyTorch lightning."""
    # Use timestamp if experiment name not specified
    if not experiment_name:
        experiment_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment_dir = (
        Path(__file__).absolute().parent
        / "experiments"
        / experiment_name
        / f"v{version}"
    )
    if experiment_dir.exists():
        return get_experiment_dir(experiment_name, version + 1)
    else:
        return experiment_dir


def run_training(
    config: EgoAlloTrainConfig,
    restore_checkpoint_dir: Path | None = None,
    debug_mode: bool = False,
) -> None:
    # Set up experiment directory + HF accelerate.
    # We're getting to manage logging, checkpoint directories, etc manually,
    # and just use `accelerate` for distibuted training.
    experiment_dir = get_experiment_dir(config.experiment_name)
    assert not experiment_dir.exists()
    config.experiment_dir = experiment_dir
    accelerator = Accelerator(
        project_config=ProjectConfiguration(project_dir=str(experiment_dir)),
        dataloader_config=DataLoaderConfiguration(split_batches=True),
    )

    # Initialize wandb instead of tensorboardX
    if accelerator.is_main_process:
        wandb.init(
            project="egoallo",
            name=config.experiment_name
            or datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            config=dataclasses.asdict(config),
            dir=str(experiment_dir),
        )

        # Save experiment files
        experiment_dir.mkdir(exist_ok=True, parents=True)
        (experiment_dir / "git_commit.txt").write_text(
            training_utils.get_git_commit_hash()
        )
        (experiment_dir / "git_diff.txt").write_text(training_utils.get_git_diff())
        (experiment_dir / "run_config.yaml").write_text(yaml.dump(config))
        (experiment_dir / "model_config.yaml").write_text(yaml.dump(config.model))

    device = accelerator.device

    if accelerator.is_main_process:
        training_utils.ipdb_safety_net()

        # Save various things that might be useful.
        experiment_dir.mkdir(exist_ok=True, parents=True)
        (experiment_dir / "git_commit.txt").write_text(
            training_utils.get_git_commit_hash()
        )
        (experiment_dir / "git_diff.txt").write_text(training_utils.get_git_diff())
        (experiment_dir / "run_config.yaml").write_text(yaml.dump(config))
        (experiment_dir / "model_config.yaml").write_text(yaml.dump(config.model))

        # source_code_log_dir = experiment_dir / "logs"

        # llogger = setup_logger(output=None, name=__name__)
        # make_source_code_snapshot(source_code_log_dir, logger=llogger)

        # Write logs to file.
        logger.add(experiment_dir / "trainlog.log", rotation="100 MB")

    # Setup.
    model = network.EgoDenoiser(
        config.model,
        modality_dims=config.denoising.fetch_modality_dict(config.model.include_hands),
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=build_dataset(cfg=config)(config=config),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
        pin_memory=True,
        collate_fn=make_batch_collator(config),
        drop_last=True,
    )

    # breakpoint()
    optim = torch.optim.AdamW(  # type: ignore
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda step: min(1.0, step / config.warmup_steps)
    )

    # HF accelerate setup. We use this for parallelism, etc!
    model, train_loader, optim, scheduler = accelerator.prepare(
        model, train_loader, optim, scheduler
    )
    accelerator.register_for_checkpointing(scheduler)

    # Restore an existing model checkpoint.
    if restore_checkpoint_dir is not None:
        accelerator.load_state(str(restore_checkpoint_dir))

    # Get the initial step count.
    if restore_checkpoint_dir is not None and restore_checkpoint_dir.name.startswith(
        "checkpoint_"
    ):
        step = int(restore_checkpoint_dir.name.partition("_")[2])
    else:
        step = int(scheduler.state_dict()["last_epoch"])
        assert step == 0 or restore_checkpoint_dir is not None, step

    # Save an initial checkpoint. Not a big deal but currently this has an
    # off-by-one error, in that `step` means something different in this
    # checkpoint vs the others.
    accelerator.save_state(str(experiment_dir / f"checkpoints_{step}"))

    # Run training loop!
    loss_helper = training_loss.TrainingLossComputer(config.loss, device=device)
    loop_metrics_gen = training_utils.loop_metric_generator(counter_init=step)
    prev_checkpoint_path: Path | None = None
    training_start_time = time.time()
    batch_start_time = time.time()
    epoch_start_time = time.time()
    epoch_time = time.time() - epoch_start_time
    epoch = 0

    while True:
        for train_batch in train_loader:
            # Record batch loading time
            batch_load_time = time.time() - batch_start_time
            batch_start_time = time.time()

            loop_metrics = next(loop_metrics_gen)
            step = loop_metrics.counter

            loss, log_outputs = loss_helper.compute_denoising_loss(
                model,
                unwrapped_model=accelerator.unwrap_model(model),
                train_config=config,
                train_batch=train_batch,
            )

            # Add learning rate to outputs
            log_outputs["learning_rate"] = scheduler.get_last_lr()[0]

            # Wrap optimization steps in debug_mode check
            if not debug_mode:
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), config.max_grad_norm
                    )
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)

            if not accelerator.is_main_process:
                continue

            if step % 400 == 0:
                log_msg = (
                    f"step: {step} ({loop_metrics.iterations_per_sec:.2f} it/sec)"
                    f" epoch: {epoch} (time: {epoch_time:.1f}s)"
                    f" time: {loop_metrics.time_elapsed:.1f}s"
                    f" gpus: {loop_metrics.num_gpus}"
                    f" batch/gpu: {loop_metrics.per_gpu_batch_size}"
                    f" total_batch: {loop_metrics.total_batch_size}"
                    f" gpu_util: {[f'{u:.1f}%' for u in loop_metrics.gpu_utilization]}"
                    f" gpu_mem: {[f'{m:.1f}GB' for m in loop_metrics.gpu_memory_used]}"
                    f" lr: {scheduler.get_last_lr()[0]:.7f}"
                    f" loss: {loss.item():.6f}"
                )

                # Add all loss terms from log_outputs
                for key, value in log_outputs.items():
                    if key.startswith("loss_term/"):
                        # Extract term name after loss_term/
                        term_name = key.split("/")[-1]
                        # Add formatted loss term
                        log_msg += f" {term_name}: {value:.6f}"

                logger.info(log_msg)
                # Log metrics to wandb
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train/step": step,
                        "performance/batch_load_time_ms": batch_load_time * 1000,
                        "system/gpu_utilization": {
                            f"gpu_{i}": util
                            for i, util in enumerate(loop_metrics.gpu_utilization)
                        },
                        "system/gpu_memory_used": {
                            f"gpu_{i}": mem
                            for i, mem in enumerate(loop_metrics.gpu_memory_used)
                        },
                        "system/total_batch_size": loop_metrics.total_batch_size,
                        "system/per_gpu_batch_size": loop_metrics.per_gpu_batch_size,
                        "system/num_gpus": loop_metrics.num_gpus,
                        "time/batch_time_ms": loop_metrics.batch_time * 1000,
                        "time/iterations_per_sec": loop_metrics.iterations_per_sec,
                        "time/epoch_time": epoch_time,
                        "time/total_time": time.time() - training_start_time,
                    },
                    step=step,
                )

                # Add individual loss terms
                for key, value in log_outputs.items():
                    if key.startswith("loss_term/"):
                        term_name = key.split("/")[-1]
                        wandb.log({f"losses/{term_name}": value}, step=step)

                # Add gradient norms if not in debug mode
                if not debug_mode and step % 400 == 0:
                    total_grad_norm = 0.0
                    param_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm += p.norm(2).item() ** 2
                            grad_norm = p.grad.norm(2).item() ** 2
                            total_grad_norm += grad_norm

                    wandb.log(
                        {
                            "gradients/total_grad_norm": np.sqrt(total_grad_norm),
                            "gradients/param_norm": np.sqrt(param_norm),
                            "gradients/grad_to_param_ratio": np.sqrt(total_grad_norm)
                            / (np.sqrt(param_norm) + 1e-8),
                        },
                        step=step,
                    )

                # Log model parameter statistics periodically
                if step % 2000 == 0:
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            wandb.log(
                                {
                                    f"parameters/{name}/mean": param.mean().item(),
                                    f"parameters/{name}/std": param.std().item(),
                                    f"parameters/{name}/norm": param.norm().item(),
                                },
                                step=step,
                            )
            # Checkpointing
            steps_to_save = 1e4
            if step % steps_to_save == 0:
                # Save checkpoint.
                checkpoint_path = experiment_dir / f"checkpoints_{step}"
                accelerator.save_state(str(checkpoint_path))
                logger.info(f"Saved checkpoint to {checkpoint_path}")

                # Keep checkpoints from only every 100k steps
                if prev_checkpoint_path is not None:
                    shutil.rmtree(prev_checkpoint_path)
                prev_checkpoint_path = (
                    None if step % steps_to_save == 0 else checkpoint_path
                )

            # Evaluation
            steps_to_eval = 1e4
            if step % steps_to_eval == 0 and step != 0:
                # Create temporary directory for evaluation outputs
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Create inference config for evaluation
                    inference_config = InferenceConfig(
                        checkpoint_dir=experiment_dir / f"checkpoints_{step}",
                        output_dir=Path(temp_dir),
                        device=device,
                        visualize_traj=False,  # Don't generate videos during training
                        compute_metrics=True,
                        skip_eval_confirm=True,
                        use_mean_body_shape=False,  # use_mean_body_shape would fail assertion
                    )

                    # Run evaluation
                    try:
                        test_runner = TestRunner(inference_config)
                        # TODO: just for debugging.
                        # test_runner.denoiser = accelerator.unwrap_model(model)
                        metrics = test_runner.run()

                        assert metrics is not None
                        # Log summary metrics to wandb
                        for metric_name, metric_stats in metrics.summary.items():
                            for stat_name, stat_value in metric_stats.items():
                                wandb.log(
                                    {f"eval/{metric_name}/{stat_name}": stat_value},
                                    step=step,
                                )
                                logger.info(
                                    f"Step {step}, Loss: {log_outputs['train_loss']:.6f}, Eval: {metric_name} {stat_name}: {stat_value:.4f}"
                                )

                    except Exception as e:
                        logger.error(f"Evaluation failed at step {step}: {str(e)}")
                        logger.exception("Detailed error:")

                del checkpoint_path

        # End of epoch
        epoch += 1
        epoch_time = time.time() - epoch_start_time
        # if accelerator.is_main_process:
        #     logger.info(f"Epoch {epoch} completed in {epoch_time:.1f} seconds")
        epoch_start_time = time.time()

    # Finish wandb run
    if accelerator.is_main_process:
        wandb.finish()


def test_run_training_cli():
    # Store original argv
    original_argv = sys.argv.copy()

    try:
        # Create test config directly instead of using CLI args
        test_config = EgoAlloTrainConfig(
            batch_size=64,
            experiment_name="test_experiment",
            learning_rate=1e-4,
            dataset_hdf5_path=Path(
                "./data/amass_rich_hps/processed_amass_rich_hps.hdf5"
            ),
            dataset_files_path=Path(
                "./data/amass_rich_hps/processed_amass_rich_hps.txt"
            ),
            mask_ratio=0.0,
            splits=("train", "val"),
            joint_cond_mode="absrel",
            use_fourier_in_masked_joints=False,
            random_sample_mask_ratio=True,
            data_collate_fn="TensorOnlyDataclassBatchCollator",
        )

        # Mock wandb to prevent actual wandb initialization
        with patch("wandb.init"), patch("wandb.log"), patch("wandb.finish"):
            # Run the main function
            run_training(test_config)

    finally:
        # Restore original argv
        sys.argv = original_argv


if __name__ == "__main__":
    # breakpoint()
    tyro.cli(run_training)
