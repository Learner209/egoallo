"""Training script for EgoAllo diffusion model using HuggingFace accelerate."""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import shutil
import math
from pathlib import Path
import torch
import yaml
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import ProjectConfiguration
from diffusers.training_utils import EMAModel
from diffusers.schedulers import DDPMScheduler
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
import wandb

from egoallo.motion_diffusion_pipeline import MotionDiffusionPipeline, MotionUNet
from egoallo.data.amass_dataset import AmassHdf5Dataset
from egoallo.data.dataclass import collate_dataclass, EgoTrainingData
import egoallo.training_utils as training_utils
from egoallo.setup_logger import setup_logger
from egoallo.config.train_config import EgoAlloTrainConfig
from egoallo.training_loss import MotionLossComputer, TrainingLossConfig

logger = setup_logger(output=None, name=__name__)

class MotionPriorTrainer:
    """Handles the training of the Motion Prior model."""
    
    def __init__(
        self,
        config: EgoAlloTrainConfig,
        gradient_accumulation_steps: int = 1,
        use_ema: bool = True,
    ):
        self.config = config
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_ema = use_ema
        self.experiment_dir = training_utils.get_experiment_dir(config.experiment_name)
        
        # Initialize accelerator
        self.accelerator = self._setup_accelerator()
        self.device = self.accelerator.device
                        
        # Calculate effective batch size and learning rate
        self.total_batch_size = self._calculate_total_batch_size()
        self.learning_rate = self._calculate_scaled_learning_rate()
        self.step = 0
        
        # Initialize pipeline components and move to device
        self.unet = MotionUNet(config.model).to(self.device)
        self.pipeline = MotionDiffusionPipeline(
            unet=self.unet,
            scheduler=DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
        ).to(self.device)
        
        # Initialize training components
        self.train_loader = self._setup_dataloader()
        self.optimizer = self._setup_optimizer()
        self.lr_scheduler = self._setup_lr_scheduler()
        self.ema = self._setup_ema() if use_ema else None
        self.writer = None
        
        # Prepare components
        (
            self.pipeline,
            self.train_loader,
            self.optimizer,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.pipeline,
            self.train_loader,
            self.optimizer,
            self.lr_scheduler,
        )

        self.loop_metrics_gen = training_utils.loop_metric_generator(counter_init=self.step)
        self.prev_checkpoint_path = None

        # Initialize loss computer
        self.loss_computer = MotionLossComputer(
            config=TrainingLossConfig(),
            device=self.device,
            scheduler=self.pipeline.scheduler
        )

    def _setup_accelerator(self) -> Accelerator:
        """Initialize and configure the Accelerator."""
        return Accelerator(
            project_config=ProjectConfiguration(project_dir=str(self.experiment_dir)),
            dataloader_config=DataLoaderConfiguration(split_batches=False),
        )

    def _setup_dataloader(self) -> DataLoader:
        """Initialize the data loader."""
        dataset = AmassHdf5Dataset(self.config)
        return DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            persistent_workers=self.config.num_workers > 0,
            pin_memory=True,
            collate_fn=collate_dataclass,
            drop_last=True,
        )

    def _setup_optimizer(self) -> torch.optim.AdamW:
        """Initialize the optimizer."""
        return torch.optim.AdamW(
            self.pipeline.unet.parameters(),
            lr=self.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _setup_lr_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        """Initialize the learning rate scheduler."""
        return torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, step / self.config.warmup_steps)
        )

    def _setup_ema(self) -> EMAModel:
        """Initialize EMA model."""
        if not self.use_ema:
            return None
        return EMAModel(
            self.pipeline.unet.parameters(),
            decay=self.config.ema_decay,
            use_ema_warmup=True,
        ).to(self.device)

    def _restore_checkpoint(self, restore_checkpoint_dir: Path | None) -> int:
        """Restore from checkpoint if provided and return current step."""
        if restore_checkpoint_dir is not None:
            self.accelerator.load_state(str(restore_checkpoint_dir))
            if restore_checkpoint_dir.name.startswith("checkpoint_"):
                return int(restore_checkpoint_dir.name.partition("_")[2])
        return int(self.lr_scheduler.state_dict()["last_epoch"])

    def _save_initial_state(self) -> None:
        """Save initial experiment state."""
        if self.accelerator.is_main_process:
            training_utils.ipdb_safety_net()
            self.experiment_dir.mkdir(exist_ok=True, parents=True)
            
            # Save configs and git info
            (self.experiment_dir / "git_commit.txt").write_text(
                training_utils.get_git_commit_hash()
            )
            (self.experiment_dir / "git_diff.txt").write_text(training_utils.get_git_diff())
            (self.experiment_dir / "run_config.yaml").write_text(yaml.dump(self.config))
            (self.experiment_dir / "model_config.yaml").write_text(yaml.dump(self.config.model))

    def _train_step(
        self,
        train_batch: EgoTrainingData,
        loop_metrics: training_utils.LoopMetrics
    ) -> tuple[torch.Tensor, dict]:
        """Execute a single training step with proper loss computation."""
        with self.accelerator.accumulate(self.pipeline):
            # Move the entire batch to the correct device
            train_batch = train_batch.to(self.device)
            
            with autocast():
                # Sample noise and timesteps
                batch_size = train_batch.T_world_cpf.shape[0]
                noise = torch.randn(
                    [*train_batch.T_world_cpf.shape[:2], self.pipeline.unet.config.d_state], 
                    device=self.device
                )
                timesteps = torch.randint(
                    0, 
                    self.pipeline.scheduler.config.num_train_timesteps, 
                    (batch_size,), 
                    device=self.device
                )
                
                # Add noise to input
                noisy_motion = self.pipeline.scheduler.add_noise(
                    train_batch.pack().pack(),
                    noise,
                    timesteps
                )
                
                # Get model prediction
                model_pred = self.pipeline.unet.forward(
                    sample=noisy_motion,
                    timestep=timesteps,
                    train_batch=train_batch,
                    return_dict=False
                )
                
                # Compute losses using loss computer
                losses, joint_losses = self.loss_computer.compute_loss(
                    t=timesteps,
                    x0_pred=model_pred,
                    batch=train_batch,
                    unwrapped_model=self.accelerator.unwrap_model(self.pipeline.unet),
                    return_joint_losses=True
                )

            # Ensure loss is valid before backward pass
            if losses.total_loss is None or not losses.total_loss.requires_grad:
                raise ValueError("Loss is None or doesn't require gradients")
            
            # Backward pass and optimization
            self.accelerator.backward(losses.total_loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    self.pipeline.unet.parameters(),
                    self.config.max_grad_norm
                )
            
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            if self.ema is not None:
                self.ema.step(self.pipeline.unet)

            # Prepare logging outputs
            log_outputs = {
                "train/total_loss": losses.total_loss.item(),
                "train/betas_loss": losses.betas_loss.item(),
                "train/body_rot6d_loss": losses.body_rot6d_loss.item(),
                "train/contacts_loss": losses.contacts_loss.item(),
                "train/lr": self.lr_scheduler.get_last_lr()[0],
                "train/step": self.step,
                "train/iterations_per_sec": loop_metrics.iterations_per_sec
            }
            
            # Add per-joint losses if available
            if joint_losses:
                log_outputs.update({
                    f"train/{k}": v for k, v in joint_losses.items()
                })

        return losses.total_loss, log_outputs

    def train(self) -> None:
        """Execute the training loop."""
        self._save_initial_state()
        
        while self.step < self.config.max_steps:
            for train_batch in self.train_loader:
                loop_metrics = next(self.loop_metrics_gen)
                self.step = loop_metrics.counter
                
                loss, log_outputs = self._train_step(train_batch, loop_metrics)
                
                if self.accelerator.is_main_process:
                    self._handle_logging(log_outputs)
                    self._handle_checkpointing()
                
                if self.step >= self.config.max_steps:
                    break

    def _handle_logging(self, log_outputs):
        """Handle logging to tensorboard and console."""
        if self.step % 10 == 0 and self.writer is not None:
            for k, v in log_outputs.items():
                self.writer.add_scalar(k, v, self.step)

        if self.step % 20 == 0:
            iterations_per_sec = log_outputs['train/iterations_per_sec']
            mem_free, mem_total = torch.cuda.mem_get_info()
            logger.info(
                f"step: {self.step} ({iterations_per_sec:.2f} it/sec)"
                f" mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G"
                f" lr: {self.lr_scheduler.get_last_lr()[0]:.7e}"
                f" loss: {log_outputs['train/total_loss']:.6e}"
            )

    def _handle_checkpointing(self):
        """Handle model checkpointing."""
        if self.step % 5000 == 0:
            checkpoint_path = self.experiment_dir / f"checkpoints_{self.step}"
            self.accelerator.save_state(str(checkpoint_path))
            logger.info(f"Saved checkpoint to {checkpoint_path}")

            if self.prev_checkpoint_path is not None:
                shutil.rmtree(self.prev_checkpoint_path)
            self.prev_checkpoint_path = None if self.step % 100_000 == 0 else checkpoint_path

    def _calculate_total_batch_size(self) -> int:
        """Calculate the total effective batch size across all GPUs."""
        return (
            self.config.batch_size *  # per-GPU batch size
            torch.cuda.device_count() *  # number of GPUs
            self.gradient_accumulation_steps  # gradient accumulation
        )

    def _calculate_scaled_learning_rate(self) -> float:
        """Calculate the scaled learning rate based on total batch size."""
        batch_size_ratio = self.total_batch_size / self.config.base_batch_size
        
        if self.config.learning_rate_scaling == "sqrt":
            # Square root scaling (recommended for most cases)
            scale_factor = math.sqrt(batch_size_ratio)
        elif self.config.learning_rate_scaling == "linear":
            # Linear scaling (used in some cases, especially with very large batches)
            scale_factor = batch_size_ratio
        else:  # "none"
            # No scaling
            scale_factor = 1.0
        
        scaled_lr = self.config.base_learning_rate * scale_factor
        
        if self.accelerator.is_main_process:
            logger.info(f"Base learning rate: {self.config.base_learning_rate}")
            logger.info(f"Total batch size: {self.total_batch_size}")
            logger.info(f"Batch size ratio: {batch_size_ratio}")
            logger.info(f"Learning rate scale factor: {scale_factor}")
            logger.info(f"Scaled learning rate: {scaled_lr}")
            
        return scaled_lr


def train_motion_prior(
    config: EgoAlloTrainConfig,
    restore_checkpoint_dir: Path | None = None,
) -> None:
    """Main training function."""
    trainer = MotionPriorTrainer(
        config=config,
        gradient_accumulation_steps=1,
        use_ema=config.use_ema
    )
    
    if restore_checkpoint_dir is not None:
        trainer.step = trainer._restore_checkpoint(restore_checkpoint_dir)
        
    trainer.train()

if __name__ == "__main__":
    import tyro
    tyro.cli(train_motion_prior)