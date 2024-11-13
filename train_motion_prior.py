"""Training script for EgoAllo diffusion model using HuggingFace accelerate."""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import shutil
from pathlib import Path
import tensorboardX
import torch
import yaml
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import ProjectConfiguration
from diffusers.training_utils import EMAModel

from egoallo import network, training_loss, training_utils
from egoallo.data.amass_dataset_dynamic import EgoAmassHdf5DatasetDynamic
from egoallo.data.dataclass import collate_dataclass
from egoallo.setup_logger import setup_logger
from egoallo.config.train_config import EgoAlloTrainConfig

logger = setup_logger(output=None, name=__name__)

class MotionPriorTrainer:
    """Handles the training of the Motion Prior model."""
    
    def __init__(
        self,
        config: EgoAlloTrainConfig,
        restore_checkpoint_dir: Path | None = None,
    ):
        self.config = config
        self.experiment_dir = training_utils.get_experiment_dir(config.experiment_name)
        assert not self.experiment_dir.exists()
        
        # Initialize accelerator
        self.accelerator = self._setup_accelerator()
        self.device = self.accelerator.device
        
        # Initialize components
        self.model = self._setup_model()
        self.train_loader = self._setup_dataloader()
        self.optimizer = self._setup_optimizer()
        self.lr_scheduler = self._setup_lr_scheduler()
        self.ema = self._setup_ema() if config.use_ema else None
        
        # Prepare for distributed training
        self.model, self.train_loader, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.train_loader, self.optimizer, self.lr_scheduler
        )
        self.accelerator.register_for_checkpointing(self.lr_scheduler)
        
        # Setup writer for main process
        self.writer = self._setup_writer()
        
        # Restore checkpoint if provided
        self.step = self._restore_checkpoint(restore_checkpoint_dir)
        
        # Initialize training components
        self.loss_helper = training_loss.TrainingLossComputer(config.loss, device=self.device)
        self.loop_metrics_gen = training_utils.loop_metric_generator(counter_init=self.step)
        self.prev_checkpoint_path = None

    def _setup_accelerator(self) -> Accelerator:
        """Initialize and configure the Accelerator."""
        return Accelerator(
            project_config=ProjectConfiguration(project_dir=str(self.experiment_dir)),
            dataloader_config=DataLoaderConfiguration(split_batches=True),
        )

    def _setup_model(self) -> network.EgoDenoiser:
        """Initialize the model."""
        return network.EgoDenoiser(self.config.model)

    def _setup_dataloader(self) -> torch.utils.data.DataLoader:
        """Initialize the data loader."""
        train_dataset = EgoAmassHdf5DatasetDynamic(self.config)
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
            lr=self.config.learning_rate,
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

    def _setup_writer(self) -> tensorboardX.SummaryWriter | None:
        """Initialize tensorboard writer for main process."""
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
                
                self._train_step(train_batch, loop_metrics)
                
                if self.accelerator.is_main_process:
                    self._handle_logging(loop_metrics)
                    self._handle_checkpointing()

    def _train_step(self, train_batch, loop_metrics):
        """Execute a single training step."""
        with self.accelerator.accumulate(self.model):
            loss, log_outputs = self.loss_helper.compute_denoising_loss(
                self.model,
                unwrapped_model=self.accelerator.unwrap_model(self.model),
                train_batch=train_batch,
            )
            log_outputs["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
            self.accelerator.log(log_outputs, step=self.step)
            
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            if self.ema is not None:
                self.ema.step(self.model)

    def _handle_logging(self, loop_metrics):
        """Handle logging to tensorboard and console."""
        if self.step % 10 == 0 and self.writer is not None:
            for k, v in log_outputs.items():
                self.writer.add_scalar(k, v, self.step)

        if self.step % 20 == 0:
            mem_free, mem_total = torch.cuda.mem_get_info()
            logger.info(
                f"step: {self.step} ({loop_metrics.iterations_per_sec:.2f} it/sec)"
                f" mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G"
                f" lr: {self.lr_scheduler.get_last_lr()[0]:.7e}"
                f" loss: {loss.item():.6e}"
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

def train_motion_prior(
    config: EgoAlloTrainConfig,
    restore_checkpoint_dir: Path | None = None,
) -> None:
    """Main training function."""
    trainer = MotionPriorTrainer(config, restore_checkpoint_dir)
    trainer.train()

if __name__ == "__main__":
    import tyro
    tyro.cli(train_motion_prior)
