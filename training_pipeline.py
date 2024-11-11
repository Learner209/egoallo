from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Dict, Any
import logging
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import wandb
import yaml
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import ProjectConfiguration
from diffusers import DDPMScheduler
from diffusers.training_utils import EMAModel

from egoallo.motion_diffusion_pipeline import MotionDiffusionPipeline, MotionUNet
from egoallo.data.amass_dataset_dynamic import EgoAmassHdf5DatasetDynamic
from egoallo.data.dataclass import collate_dataclass, EgoTrainingData
from egoallo.network import EgoDenoiseTraj
from egoallo.training_utils import (
    get_experiment_dir,
    flattened_hparam_dict_from_dataclass,
    loop_metric_generator,
)
from egoallo import training_loss, network
from egoallo.setup_logger import setup_logger
from egoallo.training_loss import MotionLosses

logger = setup_logger(output=None, name=__name__)

# Original config class remains the same
@dataclass(frozen=False)
class EgoAlloTrainConfig:
    experiment_name: str = "april13"
    dataset_hdf5_path: Path = Path("./data/egoalgo_no_skating_dataset.hdf5")
    dataset_files_path: Path = Path("./data/egoalgo_no_skating_dataset_files.txt")
    smplh_npz_path: Path = Path("./data/smplh/neutral/model.npz")
    use_ipdb: bool = False

    loss: training_loss.TrainingLossConfig = training_loss.TrainingLossConfig()
    device: Literal["cpu", "cuda"] = "cuda"

    # Dataset arguments.
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
    condition_on_prev_window: bool = True
    """Whether to condition on previous motion window."""
    model: network.EgoDenoiserConfig = network.EgoDenoiserConfig(
        condition_on_prev_window=condition_on_prev_window
    )

    # Optimizer options.
    learning_rate: float = 4e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
class MotionDiffusionTrainer:
    """Handles the training of the Motion Diffusion model."""
    
    def __init__(
        self,
        config: EgoAlloTrainConfig,
        num_epochs: int,
        gradient_accumulation_steps: int = 1,
        use_ema: bool = True,
    ):
        self.config = config
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_ema = use_ema
        self.experiment_dir = get_experiment_dir(config.experiment_name)
        
        # Initialize accelerator
        self.accelerator = self._setup_accelerator()
        self.device = self.accelerator.device
        
        # Initialize components
        self.model = self._setup_model()
        self.noise_scheduler = self._setup_scheduler()
        self.loss_computer = self._setup_loss_computer()
        self.optimizer = self._setup_optimizer()
        self.ema = self._setup_ema() if use_ema else None
        self.dataloader = self._setup_dataloader()
        
        # Prepare for distributed training
        self.model, self.dataloader, self.optimizer = self.accelerator.prepare(
            self.model, self.dataloader, self.optimizer
        )
        
        self.global_step = 0
        self.loop_metrics_gen = loop_metric_generator()
        
        # Initialize wandb if main process
        if self.accelerator.is_main_process:
            self._initialize_wandb()
            self._save_config()

        # Create checkpoint directory
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

    def _setup_accelerator(self) -> Accelerator:
        """Initialize and configure the Accelerator."""
        return Accelerator(
            project_config=ProjectConfiguration(project_dir=str(self.experiment_dir)),
            dataloader_config=DataLoaderConfiguration(split_batches=True),
            cpu=self.config.device == "cpu"
        )

    def _setup_model(self) -> MotionUNet:
        """Initialize the UNet model."""
        return MotionUNet(
            self.config.model,
            smplh_npz_path=self.config.smplh_npz_path,
            device=self.device
        )

    def _setup_scheduler(self) -> DDPMScheduler:
        """Initialize the noise scheduler."""
        return DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="sample"
        )

    def _setup_loss_computer(self) -> training_loss.MotionLossComputer:
        """Initialize the loss computer."""
        return training_loss.MotionLossComputer(
            self.config.loss,
            self.device,
            self.noise_scheduler
        )

    def _setup_optimizer(self) -> torch.optim.AdamW:
        """Initialize the optimizer."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

    def _setup_ema(self) -> Optional[EMAModel]:
        """Initialize EMA if enabled."""
        if not self.use_ema:
            return None
        ema = EMAModel(
            self.model.parameters(),
            decay=0.9999,
            use_ema_warmup=True
        )
        ema.to(self.device)
        return ema

    def _setup_dataloader(self) -> DataLoader:
        """Initialize the data loader."""
        dataset = EgoAmassHdf5DatasetDynamic(self.config)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            persistent_workers=self.config.num_workers > 0,
            pin_memory=True,
            collate_fn=collate_dataclass,
            drop_last=True
        )

    def _initialize_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        wandb.init(
            project="motion_diffusion",
            name=self.config.experiment_name,
            config=flattened_hparam_dict_from_dataclass(self.config)
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
            "train/hand_rot6d_loss": losses.hand_rot6d_loss.item(),
            "train/fk_loss": losses.fk_loss.item(),
            "train/foot_skating_loss": losses.foot_skating_loss.item(),
            "train/velocity_loss": losses.velocity_loss.item(),
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
            f"FK: {losses.fk_loss.item():.4f}, "
            f"Foot Skating: {losses.foot_skating_loss.item():.4f}, "
            f"Velocity: {losses.velocity_loss.item():.4f}"
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
            return

        # Create checkpoint subdirectory
        ckpt_path = self.checkpoint_dir / f"checkpoint-{self.global_step}"
        ckpt_path.mkdir(exist_ok=True, parents=True)

        # Unwrap model
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # Create pipeline instance for saving
        pipeline = MotionDiffusionPipeline(
            unet=unwrapped_model,
            scheduler=self.noise_scheduler
        )

        # Save the pipeline (includes model weights and config)
        pipeline.save_pretrained(str(ckpt_path))

        # Save optimizer state
        torch.save(
            self.optimizer.state_dict(),
            ckpt_path / "optimizer.pt"
        )

        # Save EMA state if used
        if self.ema is not None:
            torch.save(
                self.ema.state_dict(),
                ckpt_path / "ema.pt"
            )

        # Save training state
        torch.save(
            {
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            },
            ckpt_path / "training_state.pt"
        )

        logger.info(f"Saved checkpoint at step {self.global_step} to {ckpt_path}")

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load model and training state from checkpoint."""
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint path {checkpoint_path} does not exist. Starting from scratch.")
            return

        # Load pipeline (includes model weights and config)
        pipeline = MotionDiffusionPipeline.from_pretrained(str(checkpoint_path))
        self.model.load_state_dict(pipeline.unet.state_dict())
        self.noise_scheduler = pipeline.scheduler

        # Load optimizer state if exists
        optimizer_path = checkpoint_path / "optimizer.pt"
        if optimizer_path.exists():
            self.optimizer.load_state_dict(torch.load(optimizer_path))

        # Load EMA state if exists
        if self.ema is not None:
            ema_path = checkpoint_path / "ema.pt"
            if ema_path.exists():
                self.ema.load_state_dict(torch.load(ema_path))

        # Load training state
        training_state_path = checkpoint_path / "training_state.pt"
        if training_state_path.exists():
            training_state = torch.load(training_state_path)
            self.global_step = training_state["global_step"]
            self.current_epoch = training_state["epoch"]

        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def train(self) -> None:
        """Execute the training loop."""
        self.current_epoch = 0
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            self.model.train()
            for batch in self.dataloader:
                self._train_step(batch, epoch)

    def _train_step(self, batch: EgoTrainingData, epoch: int) -> None:
        """Execute a single training step."""
        loop_metrics = next(self.loop_metrics_gen)
        
        # Prepare input data
        clean_motion = batch.pack().pack()
        noise = torch.randn_like(clean_motion)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (clean_motion.shape[0],),
            device=self.device,
            dtype=torch.long
        )
        
        # Add noise
        noisy_motion = self.noise_scheduler.add_noise(clean_motion, noise, timesteps)
        
        # Forward pass and loss computation
        with self.accelerator.accumulate(self.model):
            x0_pred = self.model.forward(
                sample=noisy_motion,
                timestep=timesteps,
                train_batch=batch,
            )
            
            losses, joint_losses = self.loss_computer.compute_loss(
                x0_pred=x0_pred.sample,
                batch=batch,
                unwrapped_model=self.accelerator.unwrap_model(self.model),
                t=timesteps,
                return_joint_losses=True
            )
            
            # Backward pass
            self.accelerator.backward(losses.total_loss)
            
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
            
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.use_ema:
                self.ema.step(self.model.parameters())
        
        # Logging
        if self.accelerator.is_main_process:
            group_losses = self._compute_joint_group_losses(joint_losses)
            self._log_training_progress(losses, group_losses)
            
            # Save checkpoint
            if self.global_step % 5000 == 0:
                self._save_checkpoint()
        
        self.global_step += 1

def train_motion_diffusion(
    config: EgoAlloTrainConfig,
    num_epochs: int,
    gradient_accumulation_steps: int = 1,
    use_ema: bool = True,
    use_ipdb: bool = False,
) -> None:
    """Main training function."""
    if use_ipdb:
        import ipdb; ipdb.set_trace()
    
    trainer = MotionDiffusionTrainer(
        config,
        num_epochs,
        gradient_accumulation_steps,
        use_ema
    )
    trainer.train()

if __name__ == "__main__":
    import tyro
    tyro.cli(train_motion_diffusion)

