"""Training script for EgoAllo diffusion model using HuggingFace accelerate."""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import shutil
from pathlib import Path
import deepspeed
import tensorboardX
import torch
import yaml
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import ProjectConfiguration
from diffusers.training_utils import EMAModel
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

from egoallo.motion_diffusion_pipeline import MotionDiffusionPipeline, MotionUNet
from egoallo.training_utils import ipdb_safety_net
from egoallo.data.amass_dataset import AmassHdf5Dataset
from egoallo.data.dataclass import collate_dataclass, EgoTrainingData
from egoallo.network import EgoDenoiseTraj
from egoallo.training_utils import (
    get_experiment_dir,
    flattened_hparam_dict_from_dataclass,
    loop_metric_generator,
)
from egoallo import training_loss, network
from egoallo.setup_logger import setup_logger
from egoallo.config.train_config import EgoAlloTrainConfig
from egoallo.training_utils import get_experiment_dir, LoopMetrics
from egoallo.data.dataclass import EgoTrainingData

import math

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
        self.experiment_dir = get_experiment_dir(config.experiment_name)
        
        # Initialize accelerator
        self.accelerator = self._setup_accelerator()
        self.device = self.accelerator.device
                        
        # Calculate effective batch size and learning rate before other initialization
        self.total_batch_size = self._calculate_total_batch_size()
        self.learning_rate = self._calculate_scaled_learning_rate()
        # Initialize step counter before other components
        self.step = 0  # Initialize step to 0 by default
        
        # Initialize components
        self.model = self._setup_model()
        self.train_loader = self._setup_dataloader()
        self.optimizer = self._setup_optimizer()
        self.lr_scheduler = self._setup_lr_scheduler()
        self.ema = self._setup_ema() if config.use_ema else None

        # Add gradient scaler for mixed precision training
        self.scaler = GradScaler()
        
        # Prepare components including scaler
        self.model, self.train_loader, self.optimizer, self.lr_scheduler, self.scaler = \
            self.accelerator.prepare(
                self.model, self.train_loader, self.optimizer, self.lr_scheduler, self.scaler
            )

        # Setup writer for main process
        self.writer = self._setup_writer()
        
        # Initialize training components
        self.loss_helper = training_loss.TrainingLossComputer(config.loss, device=self.device)
        self.loop_metrics_gen = training_utils.loop_metric_generator(counter_init=self.step)
        self.prev_checkpoint_path = None

    def _setup_accelerator(self) -> Accelerator:
        """Initialize and configure the Accelerator."""
        return Accelerator(
            project_config=ProjectConfiguration(project_dir=str(self.experiment_dir)),
            dataloader_config=DataLoaderConfiguration(split_batches=False),
        )

    def _setup_model(self) -> network.EgoDenoiser:
        """Initialize the model."""
        return network.EgoDenoiser(self.config.model)

    def _setup_dataloader(self) -> torch.utils.data.DataLoader:
        """Initialize the data loader."""
        train_dataset = EgoAmassHdf5Dataset(
            hdf5_path=self.config.dataset_hdf5_path,
            file_list_path=self.config.dataset_files_path,
            splits=self.config.train_splits,
            subseq_len=self.config.subseq_len,
            cache_files=True,
            slice_strategy=self.config.dataset_slice_strategy,
            random_variable_len_proportion=self.config.dataset_slice_random_variable_len_proportion,
            random_variable_len_min=16
        )
        return torch.utils.data.DataLoader(
            dataset=train_dataset,
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
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _setup_lr_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        """Initialize the learning rate scheduler."""
        return torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, step / self.config.warmup_steps)
        )

    def _setup_ema(self) -> EMAModel | None:
        """Initialize EMA if enabled."""
        if not self.config.use_ema:
            return None
        ema = EMAModel(
            self.model.parameters(),
            decay=self.config.ema_decay,
            use_ema_warmup=True,
        )
        ema.to(self.device)
        return ema

    def _setup_dataloader(self) -> DataLoader:
        """Initialize the data loader."""
        dataset = AmassHdf5Dataset(self.config)
        effective_batch_size = self.total_batch_size
        per_gpu_batch_size = effective_batch_size // torch.cuda.device_count()
        return DataLoader(
            dataset,
            batch_size=per_gpu_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers * torch.cuda.device_count(),
            persistent_workers=self.config.num_workers > 0,
            pin_memory=True,
            collate_fn=collate_dataclass,
            drop_last=True
        )

    def _initialize_wandb(self) -> None:
        """Initialize Weights & Biases logging with additional batch size metrics."""
        config_dict = flattened_hparam_dict_from_dataclass(self.config)
        # Add computed values to wandb config
        config_dict.update({
            "total_batch_size": self.total_batch_size,
            "scaled_learning_rate": self.learning_rate,
            "num_gpus": torch.cuda.device_count(),
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
        })
        
        wandb.init(
            project="motion_diffusion",
            name=self.config.experiment_name,
            config=config_dict
        )

    def _save_config(self) -> None:
        """Save model configuration."""
        self.experiment_dir.mkdir(exist_ok=True, parents=True)
        (self.experiment_dir / "model_config.yaml").write_text(
            yaml.dump(self.config.model)
        )

    def _compute_joint_group_losses(self, joint_losses: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """Compute average losses per joint group."""
        joint_groups = {
            'spine': [3, 6, 9],
            'neck_head': [12, 15],
            'left_arm': [13, 16, 18, 20],
            'right_arm': [14, 17, 19, 21],
            'left_leg': [1, 4, 7, 10],
            'right_leg': [2, 5, 8, 11],
        }
        
        return {
            group_name: torch.mean(torch.stack([
                torch.tensor(joint_losses[f'body_rot6d_j{i-1}'], device=self.device)
                for i in joint_indices
            ]))
            for group_name, joint_indices in joint_groups.items()
        }

    def _log_to_wandb(self, losses: Any, group_losses: Dict[str, torch.Tensor]) -> None:
        """Log metrics to Weights & Biases."""
        log_dict = {
            "train/total_loss": losses.total_loss.item(),
            "train/betas_loss": losses.betas_loss.item(),
            "train/body_rot6d_loss": losses.body_rot6d_loss.item(),
            "train/contacts_loss": losses.contacts_loss.item(),
            # "train/hand_rot6d_loss": losses.hand_rot6d_loss.item(),
            # "train/fk_loss": losses.fk_loss.item(),
            # "train/foot_skating_loss": losses.foot_skating_loss.item(),
            # "train/velocity_loss": losses.velocity_loss.item(),
            "train/global_step": self.global_step,
        }
        
        # Add any group losses if present
        if group_losses:
            log_dict.update(group_losses)
            
        wandb.log(log_dict)

    def _log_to_console(self, losses: MotionLosses, group_losses: Dict[str, torch.Tensor]) -> None:
        """Log metrics to console using the pre-configured logger."""
        logger.info(
            f"Step {self.global_step}: "
            f"Total: {losses.total_loss.item():.4f}, "
            f"Betas: {losses.betas_loss.item():.4f}, "
            f"Body Rot: {losses.body_rot6d_loss.item():.4f}, "
            f"Contacts: {losses.contacts_loss.item():.4f}, "
            # f"FK: {losses.fk_loss.item():.4f}, "
            # f"Foot Skating: {losses.foot_skating_loss.item():.4f}, "
            # f"Velocity: {losses.velocity_loss.item():.4f}"
        )

    def _log_training_progress(
        self,
        losses: Any,
        group_losses: Dict[str, torch.Tensor]
    ) -> None:
        """Log training progress to wandb and console."""
        if self.global_step % 10 == 0:
            self._log_to_wandb(losses, group_losses)
            
        if self.global_step % 60 == 0:
            self._log_to_console(losses, group_losses)

    def _save_checkpoint(self) -> None:
        """Save model checkpoint and training state."""
        if not self.accelerator.is_main_process:
            return None
            
        writer = tensorboardX.SummaryWriter(logdir=str(self.experiment_dir), flush_secs=10)
        writer.add_hparams(
            hparam_dict=training_utils.flattened_hparam_dict_from_dataclass(self.config),
            metric_dict={},
            name=".",
        )
        return writer

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

    def train(self) -> None:
        """Execute the training loop."""
        self._save_initial_state()
        self.accelerator.save_state(str(self.experiment_dir / f"checkpoints_{self.step}"))

        while True:
            for train_batch in self.train_loader:
                loop_metrics = next(self.loop_metrics_gen)
                self.step = loop_metrics.counter
                
                loss, log_outputs = self._train_step(train_batch, loop_metrics)
                
                if self.accelerator.is_main_process:
                    self._handle_logging(log_outputs)
                    self._handle_checkpointing()

    def _train_step(
        self,
        train_batch: EgoTrainingData,
        loop_metrics: LoopMetrics
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """Execute a single training step with mixed precision."""
        with self.accelerator.accumulate(self.model):
            # Compute loss with autocast
            with autocast():
                loss, log_outputs = self.loss_helper.compute_denoising_loss(
                    self.model,
                    unwrapped_model=self.accelerator.unwrap_model(self.model),
                    train_batch=train_batch,
                )
            
            log_outputs["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
            log_outputs["iterations_per_sec"] = loop_metrics.iterations_per_sec
            
            self.accelerator.log(log_outputs, step=self.step)
            
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            if self.ema is not None:
                self.ema.step(self.model)
            
        return loss, log_outputs

    def _handle_logging(self, log_outputs):
        """Handle logging to tensorboard and console."""
        if self.step % 10 == 0 and self.writer is not None:
            for k, v in log_outputs.items():
                self.writer.add_scalar(k, v, self.step)

        if self.step % 20 == 0:
            iterations_per_sec = log_outputs['iterations_per_sec']
            mem_free, mem_total = torch.cuda.mem_get_info()
            logger.info(
                f"step: {self.step} ({iterations_per_sec:.2f} it/sec)"
                f" mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G"
                f" lr: {self.lr_scheduler.get_last_lr()[0]:.7e}"
                f" loss: {log_outputs['train/total_loss'].item():.6e}"
            )

    def _handle_checkpointing(self):
        """Handle model checkpointing including scaler state."""
        if self.step % 5000 == 0:
            checkpoint_path = self.experiment_dir / f"checkpoints_{self.step}"
            # Save scaler state along with other states
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
        gradient_accumulation_steps=1,  # You can make this configurable if needed
        use_ema=config.use_ema
    )
    if restore_checkpoint_dir is not None:
        trainer.step = trainer._restore_checkpoint(restore_checkpoint_dir)
    trainer.train()

if __name__ == "__main__":
    import tyro
    tyro.cli(train_motion_prior)
