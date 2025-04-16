"""Training script for EgoAllo diffusion model using HuggingFace accelerate."""

import os


from egoallo.inference_utils import load_runtime_config
from egoallo.scripts.simple_test import test_fn

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import dataclasses
import shutil
from pathlib import Path
import time
import copy

import pickle
import cv2

import torch.optim.lr_scheduler
import torch.utils.data
import yaml
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import ProjectConfiguration
from loguru import logger

from hydra.utils import instantiate

from egoallo import network, training_loss, training_utils
from egoallo.data import make_batch_collator, build_dataset
from egoallo.config.train.train_config import EgoAlloTrainConfig

import wandb
from torch.amp import autocast
import datetime
import tempfile
from egoallo.config.inference.defaults import InferenceConfig
import numpy as np
from egoallo.utils.optimization import get_scheduler
from egoallo.utils.ema_model import EMAModel


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
) -> None:
    restore_checkpoint_dir = (
        Path(train_cfg.restore_checkpoint_dir)
        if train_cfg.restore_checkpoint_dir
        else None
    )

    debug_mode = train_cfg.debug
    if debug_mode:
        import builtins

        builtins.breakpoint()  # noqa

    if restore_checkpoint_dir:
        train_cfg: EgoAlloTrainConfig = load_runtime_config(restore_checkpoint_dir)
        experiment_dir = get_experiment_dir(train_cfg.experiment_name)
    else:
        experiment_dir = get_experiment_dir(train_cfg.experiment_name)
        assert not experiment_dir.exists()

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

    if train_cfg.use_ema:
        ema_model = EMAModel(
            copy.deepcopy(model),
            update_after_step=train_cfg.ema_update_after_step,
            inv_gamma=train_cfg.ema_inv_gamma,
            power=train_cfg.ema_power,
            min_value=train_cfg.ema_min_value,
            max_value=train_cfg.ema_max_value,
        )
        ema_model: EMAModel = accelerator.prepare(ema_model)

    train_cfg.splits = ("train",)
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

    val_cfg = dataclasses.replace(train_cfg, splits=("val",))
    val_loader = torch.utils.data.DataLoader(
        dataset=build_dataset(cfg=val_cfg)(config=val_cfg),
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        persistent_workers=train_cfg.num_workers > 0,
        pin_memory=True,
        collate_fn=make_batch_collator(val_cfg),
        drop_last=True,
    )

    optim = torch.optim.AdamW(  # type: ignore
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = get_scheduler(
        train_cfg.lr_scheduler,
        optimizer=optim,
        num_warmup_steps=train_cfg.warmup_steps,
        num_training_steps=len(train_loader) * train_cfg.num_epochs,
        # pytorch assumes stepping LRScheduler every epoch
        # however huggingface diffusers steps it every batch
        last_epoch=-1,
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

    # Initialize early stopping
    early_stopping = training_utils.EarlyStopping(
        patience=train_cfg.early_stopping_patience,
        verbose=True,
        delta=train_cfg.early_stopping_delta,
    )

    # Track previous loss for spike detection
    previous_loss = None
    prev_ckpt_path = None

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

            # Define what constitutes a "significant" spike
            spike_threshold = 4.0

            if previous_loss is not None:
                if (
                    step > train_cfg.detect_loss_spike_start_step
                    and current_loss > previous_loss * spike_threshold
                ):
                    if accelerator.is_main_process:
                        # Find and delete any existing spike checkpoints
                        existing_spike_checkpoints = list(
                            experiment_dir.glob("checkpoints_*_loss_spike_*"),
                        )
                        for checkpoint in existing_spike_checkpoints:
                            if checkpoint.is_dir():
                                # Delete anomaly batch file if it exists
                                anomaly_batch = checkpoint / "anomaly_train_batch.pt"
                                if anomaly_batch.exists():
                                    anomaly_batch.unlink()
                                # Delete checkpoint directory
                                shutil.rmtree(checkpoint)
                                logger.info(
                                    f"Deleted previous spike checkpoint: {checkpoint}",
                                )

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
                            # Clear gradients before skipping
                            optim.zero_grad(set_to_none=True)
                            # Detach loss and other tensors
                            loss = loss.detach()
                            # Clear computation graph references
                            del train_batch, loss, log_outputs
                            # Now try to free memory
                            torch.cuda.empty_cache()
                            continue

            if previous_loss is None or current_loss <= previous_loss * spike_threshold:
                previous_loss = current_loss

            # Wrap optimization steps in debug_mode check
            if not debug_mode:
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        train_cfg.max_grad_norm,
                    )
                # Add gradient norms if not in debug mode
                if step % 10 == 0:
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

                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)

                if train_cfg.use_ema:
                    ema_model.step(model)

            if not accelerator.is_main_process:
                continue

            if step % 5 == 0:
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

            steps_to_eval = train_cfg.eval_every_step
            if step % steps_to_eval == 0:
                # Compute validation loss
                if not train_cfg.use_ema:
                    model.eval()
                    eval_model = model
                else:
                    ema_model.eval()
                    eval_model = ema_model

                total_val_loss = 0.0
                num_val_batches = 0
                with torch.no_grad():
                    for val_batch in val_loader:
                        with autocast(device_type=device.type, dtype=torch.float32):
                            val_loss, _ = loss_helper.compute_denoising_loss(
                                eval_model,
                                unwrapped_model=accelerator.unwrap_model(eval_model),
                                train_config=train_cfg,
                                train_batch=val_batch,
                            )
                        total_val_loss += val_loss.item()
                        num_val_batches += 1

                avg_val_loss = total_val_loss / num_val_batches

                eval_model.train()

                # Log validation loss
                if accelerator.is_main_process:
                    wandb.log({"val/loss": avg_val_loss}, step=step)
                    logger.info(f"Validation loss at step {step}: {avg_val_loss:.6f}")

                # Check early stopping
                if accelerator.is_main_process:
                    checkpoint_path = experiment_dir / f"checkpoints_{step}"
                    save_ckpt_flag = early_stopping(
                        avg_val_loss,
                        lambda: accelerator.save_state(str(checkpoint_path)),
                    )
                    if save_ckpt_flag:
                        prev_ckpt_path = checkpoint_path
                    if early_stopping.early_stop:
                        logger.info("Early stopping triggered")
                        break

            steps_to_test = train_cfg.test_every_step
            if step % steps_to_test == 0:
                try:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        inference_cfg.checkpoint_dir = prev_ckpt_path
                        inference_cfg.output_dir = Path(temp_dir)
                        test_fn(inference_cfg, device)

                        persistent_output_dir = Path(
                            experiment_dir / f"evaluation_{step}",
                        )
                        persistent_output_dir.mkdir(parents=True, exist_ok=True)

                        for item in Path(temp_dir).glob("*"):
                            dest = persistent_output_dir / item.name
                            if dest.exists():
                                if dest.is_file():
                                    dest.unlink()
                                else:
                                    shutil.rmtree(dest)
                            shutil.move(str(item), str(dest))

                        metrics_file = (
                            persistent_output_dir
                            / "all_pred_and_gt_traj_and_metrics.pkl"
                        )
                        if metrics_file.exists():
                            with open(metrics_file, "rb") as f:
                                metrics_data = pickle.load(f)
                                agg_metrics = metrics_data["agg_metrics"]

                                # Log aggregated metrics to wandb
                                for metric_name, metric_value in agg_metrics.items():
                                    wandb.log(
                                        {f"test/{metric_name}": metric_value},
                                        step=step,
                                    )

                                # Log visualization videos
                                for i, (take_name, _) in enumerate(
                                    metrics_data["all_metrics"].items(),
                                ):
                                    if i >= 10:  # Only log first 10 takes
                                        break

                                    video_path = (
                                        persistent_output_dir / f"{take_name}.mp4"
                                    )
                                    if video_path.exists():
                                        # Read video using OpenCV
                                        cap = cv2.VideoCapture(str(video_path))
                                        frames = []

                                        while True:
                                            ret, frame = cap.read()
                                            if not ret:
                                                break
                                            # Convert BGR to RGB
                                            frame = cv2.cvtColor(
                                                frame,
                                                cv2.COLOR_BGR2RGB,
                                            )
                                            frames.append(frame)

                                        cap.release()

                                        if frames:
                                            # Convert to numpy array and reshape for wandb.Video
                                            video_array = np.array(frames)
                                            video_array = np.transpose(
                                                video_array,
                                                (0, 3, 1, 2),
                                            )  # (timesteps, channel, H, W)
                                            # implemetn some kind of downsampling to ensure proper rendering on wandb website panel.
                                            # timesteps = video_array.shape[0]
                                            # video_array = video_array[::timesteps//30, :, :2, :2]

                                            wandb.log(
                                                {
                                                    f"test/media/{take_name}": wandb.Video(
                                                        video_array,
                                                        fps=30,
                                                    ),
                                                },
                                                step=step,
                                            )

                except Exception as e:
                    logger.error(f"Evaluation failed at step {step}: {str(e)}")
                    logger.exception("Detailed error:")

        if step >= train_cfg.max_steps:
            break

        if early_stopping.early_stop and accelerator.is_main_process:
            accelerator.save_state(str(experiment_dir / f"val_best_checkpoints_{step}"))
            break

        epoch += 1
        epoch_time = time.time() - epoch_start_time
        epoch_start_time = time.time()

    # Finish wandb run
    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig
    import faulthandler

    faulthandler.enable()

    training_utils.ipdb_safety_net()

    @hydra.main(version_base="1.3", config_path="config")
    def main(cfg: DictConfig) -> None:
        train_config: EgoAlloTrainConfig = instantiate(cfg.train)
        inference_config: InferenceConfig = instantiate(cfg.inference)
        run_training(train_config, inference_config)

    main()
