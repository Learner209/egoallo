from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import ProjectConfiguration
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
import torch
import wandb
import yaml
import tyro
from pathlib import Path

from egoallo.motion_diffusion_pipeline import MotionDiffusionPipeline, MotionUNet
from egoallo.network import EgoDenoiserConfig
from egoallo.data.amass_dataset_dynamic import EgoAmassHdf5DatasetDynamic
from egoallo.data.dataclass import collate_dataclass
from egoallo.training_utils import (
    get_experiment_dir,
    ipdb_safety_net,
    flattened_hparam_dict_from_dataclass,
    loop_metric_generator
)
from egoallo.setup_logger import setup_logger
from egoallo import training_loss, training_utils, network
from egoallo.data.dataclass import EgoTrainingData
from egoallo.network import EgoDenoiseTraj
from egoallo.fncsmpl import SmplhModel

logger = setup_logger(output=None, name=__name__)

import dataclasses
from typing import Literal

@dataclasses.dataclass(frozen=True)
class EgoAlloTrainConfig:
    experiment_name: str = "april13"
    dataset_hdf5_path: Path = Path("./data/egoalgo_no_skating_dataset.hdf5")
    dataset_files_path: Path = Path("./data/egoalgo_no_skating_dataset_files.txt")
    smplh_npz_path: Path = Path("./data/smplh/neutral/model.npz")

    loss: training_loss.TrainingLossConfig = training_loss.TrainingLossConfig()

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
    
def train_motion_diffusion(
    config: EgoAlloTrainConfig,
    num_epochs: int,
    gradient_accumulation_steps: int = 1,
    use_ema: bool = True,
    use_ipdb: bool = False,
):
    # Set up experiment directory + HF accelerate
    experiment_dir = get_experiment_dir(config.experiment_name)
    accelerator = Accelerator(
        project_config=ProjectConfiguration(project_dir=str(experiment_dir)),
        dataloader_config=DataLoaderConfiguration(split_batches=True),
    )
    device = accelerator.device
    if use_ipdb:
        import ipdb; ipdb.set_trace()

    # Initialize wandb
    if accelerator.is_main_process:
        wandb.init(
            project="motion_diffusion",
            name=config.experiment_name,
            config=flattened_hparam_dict_from_dataclass(config)
        )
        ipdb_safety_net()
        experiment_dir.mkdir(exist_ok=True, parents=True)
        (experiment_dir / "model_config.yaml").write_text(yaml.dump(config.model))

    # Initialize model and optimizer
    model = MotionUNet(config.model, smplh_npz_path=config.smplh_npz_path, device=device)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2"
    )
    loss_computer = training_loss.TrainingLossComputer(config.loss, device)
    
    if use_ema:
        ema = EMAModel(
            model.parameters(),
            decay=0.9999,
            use_ema_warmup=True
        )
        ema.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Setup dataset and dataloader
    train_dataset = EgoAmassHdf5DatasetDynamic(config)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
        pin_memory=True,
        collate_fn=collate_dataclass,
        drop_last=True
    )

    # Prepare for distributed training
    model, train_dataloader, optimizer = accelerator.prepare(
        model, train_dataloader, optimizer
    )
    model: MotionUNet

    # Training loop
    global_step = 0
    loop_metrics_gen = loop_metric_generator()
    
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            batch: EgoTrainingData
            loop_metrics = next(loop_metrics_gen)
            
            clean_motion: Float[Tensor, "*batch seq_len d_state"] = batch.pack().pack()
            noise = torch.randn_like(clean_motion)

            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, 
                (clean_motion.shape[0],),
                device=device,
                dtype=torch.long
            )
            
            # Add noise to get noisy sample
            noisy_motion = noise_scheduler.add_noise(
                clean_motion, 
                noise, 
                timesteps
            )
            
            with accelerator.accumulate(model):
                # Model predicts x_0 directly instead of noise
                x0_pred = model.forward(
                    sample=noisy_motion,
                    timestep=timesteps,
                    train_batch=batch,
                )
                
                # Get noise prediction from x0_pred using standard DDPM formulation
                # noise_pred = (noisy_motion - sqrt(alpha_t) * x0_pred) / sqrt(1-alpha_t)
                alphas_cumprod = noise_scheduler.alphas_cumprod[timesteps]
                alphas_cumprod = alphas_cumprod.flatten()
                while len(alphas_cumprod.shape) < len(noisy_motion.shape):
                    alphas_cumprod = alphas_cumprod.unsqueeze(-1)
                
                noise_pred = (noisy_motion - (alphas_cumprod ** 0.5) * x0_pred['sample']) / ((1 - alphas_cumprod) ** 0.5)
                
                losses = loss_computer.compute_loss(
                    noise_pred=noise_pred,
                    gt_noise=noise,
                    x0_pred=x0_pred['sample'],
                    batch=batch,
                    unwrapped_model=accelerator.unwrap_model(model),
                    t=timesteps
                )
                
                accelerator.backward(losses.total_loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                optimizer.step()
                optimizer.zero_grad()
                
                if use_ema:
                    ema.step(model.parameters())  # Use unwrapped model parameters
            # Logging
            if accelerator.is_main_process:
                if global_step % 10 == 0:
                    wandb.log({
                        "train/noise_pred_loss": losses.noise_pred_loss.item(),
                        "train/betas_loss": losses.betas_loss.item(),
                        "train/body_rot6d_loss": losses.body_rot6d_loss.item(),
                        "train/contacts_loss": losses.contacts_loss.item(),
                        "train/hand_rot6d_loss": losses.hand_rot6d_loss.item(),
                        "train/fk_loss": losses.fk_loss.item(),
                        "train/foot_skating_loss": losses.foot_skating_loss.item(),
                        "train/velocity_loss": losses.velocity_loss.item(),
                        "train/total_loss": losses.total_loss.item(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "iterations_per_sec": loop_metrics.iterations_per_sec
                    })


                
                if global_step % 60 == 0:
                    mem_free, mem_total = torch.cuda.mem_get_info()
                    logger.info(
                        f"Step: {global_step} ({loop_metrics.iterations_per_sec:.2f} it/sec)\n"
                        f"Memory: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}GB\n"
                        f"Losses:\n"
                        f"  Total:          {losses.total_loss.item():.6e}\n"
                        f"  Noise Pred:     {losses.noise_pred_loss.item():.6e}\n"
                        f"  Betas:          {losses.betas_loss.item():.6e}\n" 
                        f"  Body Rot6d:     {losses.body_rot6d_loss.item():.6e}\n"
                        f"  Contacts:       {losses.contacts_loss.item():.6e}\n"
                        f"  Hand Rot6d:     {losses.hand_rot6d_loss.item():.6e}\n"
                        f"  FK:             {losses.fk_loss.item():.6e}\n"
                        f"  Foot Skating:   {losses.foot_skating_loss.item():.6e}\n"
                        f"  Velocity:       {losses.velocity_loss.item():.6e}"
                    )
                
                if global_step % 5000 == 0:
                    # Save the EMA model if using EMA; else save the current model
                    if use_ema:
                        # Store the current model parameters
                        ema.store(model.parameters())
                        # Copy EMA parameters to the model
                        ema.copy_to(model.parameters())
                        # Save the model
                        pipeline = MotionDiffusionPipeline(
                            unet=accelerator.unwrap_model(model),
                            scheduler=noise_scheduler
                        )
                        pipeline.save_pretrained(experiment_dir / f"checkpoint-{global_step}")
                        # Restore the original parameters
                        ema.restore(model.parameters())
                    else:
                        pipeline = MotionDiffusionPipeline(
                            unet=accelerator.unwrap_model(model),
                            scheduler=noise_scheduler
                        )
                        pipeline.save_pretrained(experiment_dir / f"checkpoint-{global_step}")
                
            global_step += 1

if __name__ == "__main__":
    tyro.cli(train_motion_diffusion)

