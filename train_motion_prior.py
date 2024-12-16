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

import tensorboardX
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
from egoallo.data.amass import EgoAmassHdf5Dataset, AdaptiveAmassHdf5Dataset
from egoallo.data.dataclass import collate_dataclass
from egoallo.config.train.train_config import EgoAlloTrainConfig

import wandb
import datetime
import tempfile
from egoallo.config.inference.inference_defaults import InferenceConfig
from egoallo.scripts.test_amass import TestRunner
import json


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

    if config.debug:
        import ipdb

        ipdb.set_trace()

    # Initialize experiment.
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

        # Write logs to file.
        logger.add(experiment_dir / "trainlog.log", rotation="100 MB")

    # Setup.
    model = network.EgoDenoiser(config.model)

    train_loader = torch.utils.data.DataLoader(
        # dataset=AdaptiveAmassHdf5Dataset(config=config),
        dataset=EgoAmassHdf5Dataset(config=config, cache_files=True),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
        pin_memory=True,
        collate_fn=collate_dataclass,
        drop_last=True,
    )

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
    batch_start_time = time.time()
    epoch_start_time = time.time()
    epoch = 0

    while True:
        # breakpoint()
        for train_batch in train_loader:
            # Record batch loading time
            batch_load_time = time.time() - batch_start_time

            loop_metrics = next(loop_metrics_gen)
            step = loop_metrics.counter

            loss, log_outputs = loss_helper.compute_denoising_loss(
                model,
                unwrapped_model=accelerator.unwrap_model(model),
                train_batch=train_batch,
            )

            # Add learning rate to outputs
            log_outputs["learning_rate"] = scheduler.get_last_lr()[0]

            # Log metrics to wandb instead of tensorboard
            if accelerator.is_main_process:
                # Log all outputs
                wandb.log(log_outputs, step=step)

                # Log additional training metrics
                wandb.log(
                    {
                        "batch_loading_time": batch_load_time,
                        "iterations_per_sec": loop_metrics.iterations_per_sec,
                        "batch_time": loop_metrics.batch_time,
                        "forward_time": loop_metrics.forward_time,
                        "backward_time": loop_metrics.backward_time,
                        "optimizer_time": loop_metrics.optimizer_time,
                        "gpu_utilization": loop_metrics.gpu_utilization,
                        "gpu_memory_used": loop_metrics.gpu_memory_used,
                        "total_batch_size": loop_metrics.total_batch_size,
                        "per_gpu_batch_size": loop_metrics.per_gpu_batch_size,
                    },
                    step=step,
                )

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

            if step % 200 == 0 and accelerator.is_main_process:
                mem_free, mem_total = torch.cuda.mem_get_info()
                epoch_time = time.time() - epoch_start_time
                # Build base log message with metrics
                log_msg = (
                    f"step: {step} ({loop_metrics.iterations_per_sec:.2f} it/sec)"
                    f" epoch: {epoch} (time: {epoch_time:.1f}s)"
                    f" time: {loop_metrics.time_elapsed:.1f}s"
                    f" batch_load: {batch_load_time*1000:.1f}ms"
                    f" batch: {loop_metrics.batch_time*1000:.1f}ms"
                    f" fwd: {loop_metrics.forward_time*1000:.1f}ms"
                    f" bwd: {loop_metrics.backward_time*1000:.1f}ms"
                    f" opt: {loop_metrics.optimizer_time*1000:.1f}ms"
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

            # Checkpointing and evaluation
            steps_to_save = 5000
            if step % steps_to_save == 0 and step != 0:
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

                # Create temporary directory for evaluation outputs
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Create inference config for evaluation
                    inference_config = InferenceConfig(
                        checkpoint_dir=checkpoint_path,
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


if __name__ == "__main__":
    # breakpoint()
    tyro.cli(run_training)
