"""Training script for EgoAllo diffusion model using HuggingFace accelerate."""

import dataclasses
import shutil
from pathlib import Path
from typing import Literal

import tensorboardX
import torch.optim.lr_scheduler
import torch.utils.data
import tyro
import yaml
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import ProjectConfiguration
from diffusers import DDPMScheduler
from diffusers.training_utils import EMAModel

from egoallo import network, training_loss, training_utils
from egoallo.data.amass_dataset_dynamic import EgoAmassHdf5DatasetDynamic
from egoallo.data.dataclass import collate_dataclass
from egoallo.setup_logger import setup_logger

logger = setup_logger(output=None, name=__name__)


@dataclasses.dataclass(frozen=True)
class EgoAlloTrainConfig:
    experiment_name: str = "april13"
    dataset_hdf5_path: Path = Path("./data/egoalgo_no_skating_dataset.hdf5")
    dataset_files_path: Path = Path("./data/egoalgo_no_skating_dataset_files.txt")

    model: network.EgoDenoiserConfig = network.EgoDenoiserConfig()
    loss: training_loss.TrainingLossConfig = training_loss.TrainingLossConfig()

    # Dataset arguments
    batch_size: int = 256
    """Effective batch size."""
    num_workers: int = 2
    subseq_len: int = 128
    dataset_slice_strategy: Literal[
        "deterministic", "random_uniform_len", "random_variable_len"
    ] = "deterministic"
    dataset_slice_random_variable_len_proportion: float = 0.3
    """Only used if dataset_slice_strategy == 'random_variable_len'."""
    train_splits: tuple[Literal["train", "val", "test", "just_humaneva"], ...] = (
        "train",
    )

    # Training arguments
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    use_ema: bool = True
    ema_decay: float = 0.9999
    mixed_precision: Literal["no", "fp16", "bf16"] = "fp16"


def get_experiment_dir(experiment_name: str, version: int = 0) -> Path:
    """Creates a directory to put experiment files in, suffixed with a version number."""
    experiment_dir = (
        Path(__file__).absolute().parent / "experiments" / experiment_name / f"v{version}"
    )
    if experiment_dir.exists():
        return get_experiment_dir(experiment_name, version + 1)
    return experiment_dir


def run_training(
    config: EgoAlloTrainConfig,
    restore_checkpoint_dir: Path | None = None,
) -> None:
    """Run training loop with given configuration."""
    # Set up experiment directory + HF accelerate
    experiment_dir = get_experiment_dir(config.experiment_name)
    assert not experiment_dir.exists()
    
    accelerator = Accelerator(
        project_config=ProjectConfiguration(project_dir=str(experiment_dir)),
        dataloader_config=DataLoaderConfiguration(split_batches=True),
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    
    writer = (
        tensorboardX.SummaryWriter(logdir=str(experiment_dir), flush_secs=10)
        if accelerator.is_main_process
        else None
    )
    device = accelerator.device

    # Initialize experiment
    if accelerator.is_main_process:
        training_utils.pdb_safety_net()
        experiment_dir.mkdir(exist_ok=True, parents=True)
        
        # Save configs and git info
        (experiment_dir / "git_commit.txt").write_text(
            training_utils.get_git_commit_hash()
        )
        (experiment_dir / "git_diff.txt").write_text(training_utils.get_git_diff())
        (experiment_dir / "run_config.yaml").write_text(yaml.dump(config))
        (experiment_dir / "model_config.yaml").write_text(yaml.dump(config.model))

        # Add hyperparameters to TensorBoard
        assert writer is not None
        writer.add_hparams(
            hparam_dict=training_utils.flattened_hparam_dict_from_dataclass(config),
            metric_dict={},
            name=".",  # Hack to avoid timestamped subdirectory
        )

    # Setup model and optimizer
    model = network.EgoDenoiser(config.model)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="sample",
    )
    
    train_dataset = EgoAmassHdf5DatasetDynamic(config)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
        pin_memory=True,
        collate_fn=collate_dataclass,
        drop_last=True,
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lambda step: min(1.0, step / config.warmup_steps)
    )

    # Setup EMA if enabled
    ema = None
    if config.use_ema:
        ema = EMAModel(
            model.parameters(),
            decay=config.ema_decay,
            use_ema_warmup=True,
        )
        ema.to(device)

    # Prepare for distributed training
    model, train_loader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_loader, optimizer, lr_scheduler
    )
    accelerator.register_for_checkpointing(lr_scheduler)

    # Restore checkpoint if provided
    if restore_checkpoint_dir is not None:
        accelerator.load_state(str(restore_checkpoint_dir))

    # Get initial step count
    if restore_checkpoint_dir is not None and restore_checkpoint_dir.name.startswith(
        "checkpoint_"
    ):
        step = int(restore_checkpoint_dir.name.partition("_")[2])
    else:
        step = int(lr_scheduler.state_dict()["last_epoch"])
        assert step == 0 or restore_checkpoint_dir is not None, step

    # Save initial checkpoint
    accelerator.save_state(str(experiment_dir / f"checkpoints_{step}"))

    # Training loop
    loss_helper = training_loss.TrainingLossComputer(config.loss, device=device)
    loop_metrics_gen = training_utils.loop_metric_generator(counter_init=step)
    prev_checkpoint_path: Path | None = None

    while True:
        for train_batch in train_loader:
            loop_metrics = next(loop_metrics_gen)
            step = loop_metrics.counter

            with accelerator.accumulate(model):
                loss, log_outputs = loss_helper.compute_denoising_loss(
                    model,
                    unwrapped_model=accelerator.unwrap_model(model),
                    train_batch=train_batch,
                )
                log_outputs["learning_rate"] = lr_scheduler.get_last_lr()[0]
                accelerator.log(log_outputs, step=step)
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if ema is not None:
                    ema.step(model)

            # Rest of loop only executed by main process
            if not accelerator.is_main_process:
                continue

            # Logging
            if step % 10 == 0 and writer is not None:
                for k, v in log_outputs.items():
                    writer.add_scalar(k, v, step)

            # Print status
            if step % 20 == 0:
                mem_free, mem_total = torch.cuda.mem_get_info()
                logger.info(
                    f"step: {step} ({loop_metrics.iterations_per_sec:.2f} it/sec)"
                    f" mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G"
                    f" lr: {lr_scheduler.get_last_lr()[0]:.7e}"
                    f" loss: {loss.item():.6e}"
                )

            # Checkpointing
            if step % 5000 == 0:
                checkpoint_path = experiment_dir / f"checkpoints_{step}"
                accelerator.save_state(str(checkpoint_path))
                logger.info(f"Saved checkpoint to {checkpoint_path}")

                # Keep checkpoints from only every 100k steps
                if prev_checkpoint_path is not None:
                    shutil.rmtree(prev_checkpoint_path)
                prev_checkpoint_path = None if step % 100_000 == 0 else checkpoint_path


if __name__ == "__main__":
    tyro.cli(run_training)
