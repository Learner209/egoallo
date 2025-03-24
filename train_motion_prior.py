"""Training script for EgoAllo diffusion model using HuggingFace accelerate."""

import os


from egoallo.inference_utils import load_runtime_config

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import dataclasses
import shutil
from pathlib import Path
import time


import torch.optim.lr_scheduler
import torch.utils.data
import yaml
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import ProjectConfiguration
from loguru import logger

from hydra.utils import instantiate

# Install hook before importing any modules you want to typecheck
# with install_import_hook("egoallo", "typeguard.typechecked"):
from egoallo import network, training_loss, training_utils
from egoallo.data import make_batch_collator, build_dataset
from egoallo.config.train.train_config import EgoAlloTrainConfig

import wandb
from torch.amp import autocast
import datetime
import tempfile
from egoallo.config.inference.defaults import InferenceConfig
from egoallo.scripts.test import TestRunner
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
    train_cfg: EgoAlloTrainConfig,
    inference_cfg: InferenceConfig,
    debug_mode: bool = False,
) -> None:
    # Set up experiment directory + HF accelerate.
    # We're getting to manage logging, checkpoint directories, etc manually,
    # and just use `accelerate` for distibuted training.
    restore_checkpoint_dir = (
        Path(train_cfg.restore_checkpoint_dir)
        if train_cfg.restore_checkpoint_dir
        else None
    )

    if debug_mode:
        import builtins

        builtins.breakpoint()  # noqa

    if restore_checkpoint_dir:
        train_cfg: EgoAlloTrainConfig = load_runtime_config(restore_checkpoint_dir)
        train_cfg.batch_size = 64  # FIXME: this is a temporary fix to distill a large model trained on thecluster to local machine.
        # experiment_dir =  restore_checkpoint_dir.parent
        experiment_dir = get_experiment_dir(train_cfg.experiment_name)
    else:
        experiment_dir = get_experiment_dir(train_cfg.experiment_name)
        assert not experiment_dir.exists()

    train_cfg.experiment_dir = experiment_dir

    accelerator = Accelerator(
        project_config=ProjectConfiguration(project_dir=str(experiment_dir)),
        dataloader_config=DataLoaderConfiguration(split_batches=True),
    )

    # Initialize wandb instead of tensorboardX
    if accelerator.is_main_process:
        wandb.init(
            project="egoallo",
            name=train_cfg.experiment_name
            or datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            config=dataclasses.asdict(train_cfg),
            dir=str(experiment_dir),
        )

        # Save experiment files
        experiment_dir.mkdir(exist_ok=True, parents=True)
        if not (experiment_dir / "git_commit.txt").exists():
            (experiment_dir / "git_commit.txt").write_text(
                training_utils.get_git_commit_hash(),
            )
        if not (experiment_dir / "git_diff.txt").exists():
            (experiment_dir / "git_diff.txt").write_text(training_utils.get_git_diff())
        if not (experiment_dir / "run_config.yaml").exists():
            (experiment_dir / "run_config.yaml").write_text(yaml.dump(train_cfg))
        if not (experiment_dir / "model_config.yaml").exists():
            (experiment_dir / "model_config.yaml").write_text(
                yaml.dump(train_cfg.model),
            )

    device = accelerator.device

    if accelerator.is_main_process:
        training_utils.ipdb_safety_net()

        # Save various things that might be useful.
        experiment_dir.mkdir(exist_ok=True, parents=True)

        if not (experiment_dir / "git_commit.txt").exists():
            (experiment_dir / "git_commit.txt").write_text(
                training_utils.get_git_commit_hash(),
            )
        if not (experiment_dir / "git_diff.txt").exists():
            (experiment_dir / "git_diff.txt").write_text(training_utils.get_git_diff())
        if not (experiment_dir / "run_config.yaml").exists():
            (experiment_dir / "run_config.yaml").write_text(yaml.dump(train_cfg))
        if not (experiment_dir / "model_config.yaml").exists():
            (experiment_dir / "model_config.yaml").write_text(
                yaml.dump(train_cfg.model),
            )

        # source_code_log_dir = experiment_dir / "logs"

        # llogger = setup_logger(output=None, name=__name__)
        # make_source_code_snapshot(source_code_log_dir, logger=llogger)

        # Write logs to file.
        logger.add(experiment_dir / "trainlog.log", rotation="100 MB")

    # Setup.
    model = network.EgoDenoiser(
        train_cfg.model,
        modality_dims=train_cfg.denoising.fetch_modality_dict(
            train_cfg.model.include_hands,
        ),
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=build_dataset(cfg=train_cfg)(config=train_cfg),
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        persistent_workers=train_cfg.num_workers > 0,
        pin_memory=True,
        collate_fn=make_batch_collator(train_cfg),
        drop_last=True,
    )

    optim = torch.optim.AdamW(  # type: ignore
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lr_lambda=lambda step: min(1.0, step / train_cfg.warmup_steps),
    )

    # HF accelerate setup. We use this for parallelism, etc!
    model, train_loader, optim, scheduler = accelerator.prepare(
        model,
        train_loader,
        optim,
        scheduler,
    )
    accelerator.register_for_checkpointing(scheduler)

    # Restore an existing model checkpoint.
    if restore_checkpoint_dir is not None:
        accelerator.load_state(str(restore_checkpoint_dir))

    # Get the initial step count.
    if restore_checkpoint_dir is not None and restore_checkpoint_dir.name.startswith(
        "checkpoint_",
    ):
        step = int(restore_checkpoint_dir.name.partition("_")[2])
    else:
        step = int(scheduler.state_dict()["last_epoch"])
        assert step == 0 or restore_checkpoint_dir is not None, step

    loss_helper = training_loss.TrainingLossComputer(train_cfg.loss, device=device)
    loop_metrics_gen = training_utils.loop_metric_generator(counter_init=step)
    # prev_checkpoint_path: Path | None = None
    training_start_time = time.time()
    batch_start_time = time.time()
    epoch_start_time = time.time()
    epoch_time = time.time() - epoch_start_time
    epoch = 0

    # Track previous loss for spike detection
    previous_loss = None

    while True:
        for idx, train_batch in enumerate(train_loader):
            # Record batch loading time
            batch_load_time = time.time() - batch_start_time
            batch_start_time = time.time()

            loop_metrics = next(loop_metrics_gen)
            step = loop_metrics.counter

            if step >= train_cfg.max_steps:
                break

            with autocast(device_type=device.type, dtype=torch.float32):
                loss, log_outputs = loss_helper.compute_denoising_loss(
                    model,
                    unwrapped_model=accelerator.unwrap_model(model),
                    train_config=train_cfg,
                    train_batch=train_batch,
                )

            # Add learning rate to outputs
            log_outputs["learning_rate"] = scheduler.get_last_lr()[0]

            # Check for loss spike
            current_loss = loss.item()
            if previous_loss is not None:
                # Define what constitutes a "significant" spike (e.g., 2x increase)
                spike_threshold = 3.0
                if (
                    step > train_cfg.detect_loss_spike_start_step
                    and current_loss > previous_loss * spike_threshold
                ):
                    if accelerator.is_main_process:
                        spike_checkpoint_path = (
                            experiment_dir
                            / f"checkpoints_{step}_loss_spike_{previous_loss:.6f}_{current_loss:.6f}"
                        )
                        logger.warning(
                            f"Loss spike detected! Previous: {previous_loss:.6f}, Current: {current_loss:.6f}",
                        )
                        logger.warning(
                            f"Saving spike checkpoint to {spike_checkpoint_path}",
                        )

                        accelerator.save_state(str(spike_checkpoint_path))

                        batch_save_path = (
                            spike_checkpoint_path / "anomaly_train_batch.pt"
                        )
                        batch_save_path.parent.mkdir(exist_ok=True, parents=True)

                        cpu_batch = train_batch.to(torch.device("cpu"))
                        torch.save(cpu_batch, batch_save_path)

                        logger.info(f"Saved loss spike data to {spike_checkpoint_path}")

                        if step > train_cfg.discard_loss_spike_start_step:
                            continue

            # Update previous loss
            previous_loss = current_loss

            # Wrap optimization steps in debug_mode check
            if not debug_mode:
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        train_cfg.max_grad_norm,
                    )
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)

            if not accelerator.is_main_process:
                continue

            if step % 200 == 0:
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
                # if prev_checkpoint_path is not None:
                #     shutil.rmtree(prev_checkpoint_path)
                # prev_checkpoint_path = None if step == 0 else checkpoint_path

            # Evaluation
            steps_to_eval = 1e4
            if step % steps_to_eval == 0:
                # if step % steps_to_eval == 0 and step != 0:
                # Create temporary directory for evaluation outputs
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Create inference config for evaluation
                    inference_cfg.checkpoint_dir = (
                        experiment_dir / f"checkpoints_{step}"
                    )
                    inference_cfg.output_dir = Path(temp_dir)

                    # Run evaluatin
                    try:
                        test_runner = TestRunner(inference_cfg)
                        metrics = test_runner.run()

                        assert metrics is not None

                        for metric_name, metric_stats in metrics.summary.items():
                            for stat_name, stat_value in metric_stats.items():
                                wandb.log(
                                    {f"eval/{metric_name}/{stat_name}": stat_value},
                                    step=step,
                                )
                                logger.info(
                                    f"Step {step}, Loss: {log_outputs['train_loss']:.6f}, Eval: {metric_name} {stat_name}: {stat_value:.4f}",
                                )

                        persistent_output_dir = Path(
                            experiment_dir / f"evaluation_{step}",
                        )
                        persistent_output_dir.mkdir(parents=True, exist_ok=True)

                        # Move contents from temp dir to persistent dir, overwriting existing files
                        for item in Path(temp_dir).glob("*"):
                            dest = persistent_output_dir / item.name
                            if dest.exists():
                                if dest.is_file():
                                    dest.unlink()
                                else:
                                    shutil.rmtree(dest)
                            shutil.move(str(item), str(dest))

                    except Exception as e:
                        logger.error(f"Evaluation failed at step {step}: {str(e)}")
                        logger.exception("Detailed error:")

                del checkpoint_path

        if step >= train_cfg.max_steps:
            break

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
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base="1.3", config_path="config")
    def main(cfg: DictConfig) -> None:
        train_config: EgoAlloTrainConfig = instantiate(cfg.train)
        inference_config: InferenceConfig = instantiate(cfg.inference)
        run_training(train_config, inference_config)

    main()
