from dataclasses import dataclass
from typing import Optional, Union, Dict, assert_never

import os
import json
import torch
from diffusers import DiffusionPipeline, UNet2DModel, ModelMixin
from diffusers.utils import BaseOutput
from diffusers.schedulers import DDPMScheduler, DDIMScheduler
from torch import nn

from .network import EgoDenoiserConfig, EgoDenoiseTraj
from .data.dataclass import EgoTrainingData
from .transforms import SE3, SO3
from .network import TransformerBlock, TransformerBlockConfig, make_positional_encoding, project_rot6d
from pathlib import Path
from egoallo.fncsmpl import SmplhModel

@dataclass
class MotionDiffusionPipelineOutput(BaseOutput):
    """Output of the motion diffusion pipeline"""
    motion: EgoDenoiseTraj
    intermediate_states: Optional[list[EgoDenoiseTraj]] = None

def _make_encoder(input_dim: int, config: EgoDenoiserConfig) -> nn.Sequential:
    """Helper function to create encoder blocks"""
    Activation = {"gelu": nn.GELU, "relu": nn.ReLU}[config.activation]
    return nn.Sequential(
        nn.Linear(input_dim, config.d_latent),
        Activation(),
        nn.Linear(config.d_latent, config.d_latent),
        Activation(),
        nn.Linear(config.d_latent, config.d_latent),
    )

def _make_decoder(output_dim: int, config: EgoDenoiserConfig) -> nn.Sequential:
    """Helper function to create decoder blocks"""
    Activation = {"gelu": nn.GELU, "relu": nn.ReLU}[config.activation]
    return nn.Sequential(
        nn.Linear(config.d_latent, config.d_latent),
        nn.LayerNorm(normalized_shape=config.d_latent),
        Activation(),
        nn.Linear(config.d_latent, config.d_latent),
        Activation(),
        nn.Linear(config.d_latent, output_dim),
    )

class MotionUNet(ModelMixin, nn.Module):
    """Custom UNet architecture for motion diffusion"""
    def __init__(self, 
                 config: EgoDenoiserConfig,
                 smplh_npz_path: Path = Path("./data/smplh/neutral/model.npz"),
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()

        self.config = config
        self.smpl_model = SmplhModel.load(smplh_npz_path).to(device)
        
        # MLP encoders and decoders for each modality
        modality_dims = {
            "betas": 16,
            "body_rot6d": 21 * 6,
            "contacts": 21,
        }
        if config.include_hand_motion:
            modality_dims["hand_rot6d"] = 30 * 6

        self.encoders = nn.ModuleDict({
            k: _make_encoder(dim, config)
            for k, dim in modality_dims.items()
        })
        
        self.decoders = nn.ModuleDict({
            k: _make_decoder(dim, config)
            for k, dim in modality_dims.items()
        })

        # Noise embedder
        self.noise_emb = nn.Embedding(
            num_embeddings=config.max_t,
            embedding_dim=config.d_noise_emb,
        )
        
        # Optional noise embedding token projection
        self.noise_emb_token_proj = nn.Linear(config.d_noise_emb, config.d_latent) if config.noise_conditioning == "token" else None
        
        # Encoder / decoder layers.
        # Inputs are conditioning (current noise level, observations); output
        # is encoded conditioning information.
        self.encoder_layers = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        d_latent=config.d_latent,
                        d_noise_emb=config.d_noise_emb,
                        d_feedforward=config.d_feedforward,
                        n_heads=config.num_heads,
                        dropout_p=config.dropout_p,
                        activation=config.activation,
                        include_xattn=False,  # No conditioning for encoder.
                        use_rope_embedding=config.positional_encoding == "rope",
                        use_film_noise_conditioning=config.noise_conditioning == "film",
                        xattn_mode=config.xattn_mode,
                    )
                )
                for _ in range(config.encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        d_latent=config.d_latent,
                        d_noise_emb=config.d_noise_emb,
                        d_feedforward=config.d_feedforward,
                        n_heads=config.num_heads,
                        dropout_p=config.dropout_p,
                        activation=config.activation,
                        include_xattn=True,  # Include conditioning for the decoder.
                        use_rope_embedding=config.positional_encoding == "rope",
                        use_film_noise_conditioning=config.noise_conditioning == "film",
                        xattn_mode=config.xattn_mode,
                    )
                )
                for _ in range(config.decoder_layers)
            ]
        )


        # Other components from EgoDenoiser
        self.latent_from_cond = nn.Linear(
            len(config.cond_component_names()) * config.d_latent,
            config.d_latent
        )
        
        # Embedding layers for conditional tokens
        self.cond_embeddings = nn.ModuleDict()
        for name in self.config.cond_component_names():
            self.cond_embeddings[name] = nn.Linear(
                self.config.get_cond_component_dim(name),
                config.d_latent
            )

        # Add encoder for previous window if enabled
        if config.condition_on_prev_window:
            # Projection for previous window features
            self.prev_window_proj = nn.Linear(
                config.d_latent,
                config.d_latent
            )

    def _encode_conditioning(
        self,
        conditioning: Dict[str, torch.Tensor],
        batch: int,
        time: int,
        device: torch.device,
        noise_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Encode conditioning information."""
        # Process conditioning
        if conditioning is not None:
            cond_embeds = []
            for name in self.config.cond_component_names():
                if name in conditioning:
                    component = conditioning[name]
                    embed = self.cond_embeddings[name](component)
                    cond_embeds.append(embed)
            cond = torch.cat(cond_embeds, dim=-1)
            encoder_out = self.latent_from_cond(cond)
        else:
            assert_never(conditioning)

        # Add positional encoding
        if self.config.positional_encoding == "rope":
            pos_enc = 0
        elif self.config.positional_encoding == "transformer":
            pos_enc = make_positional_encoding(
                d_latent=self.config.d_latent,
                length=time,
                dtype=encoder_out.dtype,
            )[None, ...].to(device)
        else:
            raise ValueError(f"Unknown positional encoding: {self.config.positional_encoding}")
            
        encoder_out = encoder_out + pos_enc

        # Process through encoder layers
        for layer in self.encoder_layers:
            encoder_out = layer(encoder_out, None, noise_emb=noise_emb)

        return encoder_out

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: torch.FloatTensor,
        train_batch: EgoTrainingData,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, BaseOutput]:
        config = self.config
        
        # Process timesteps
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=sample.device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timestep = timestep[None]
            
        # Unpack sample and get dimensions
        x_t = EgoDenoiseTraj.unpack(sample, include_hands=config.include_hand_motion)
        batch, time = x_t.body_rot6d.shape[:2]
        device = sample.device
        
        # Embed noise level
        noise_emb = self.noise_emb(timestep) # 1 x D
        
        # Initial projection of noisy input
        decoder_out = (
            self.encoders["betas"](x_t.betas.reshape(batch, time, -1))
            + self.encoders["body_rot6d"](x_t.body_rot6d.reshape(batch, time, -1))
            + self.encoders["contacts"](x_t.contacts)
        )
        
        if config.include_hand_motion and x_t.hand_rot6d is not None:
            decoder_out += self.encoders["hand_rot6d"](
                x_t.hand_rot6d.reshape(batch, time, -1)
            )

        # Add positional encoding to decoder input
        if config.positional_encoding == "rope":
            pos_enc = 0
        elif config.positional_encoding == "transformer":
            pos_enc = make_positional_encoding(
                d_latent=config.d_latent,
                length=time,
                dtype=decoder_out.dtype,
            )[None, ...].to(device)
        else:
            raise ValueError(f"Unknown positional encoding: {config.positional_encoding}")
            
        decoder_out = decoder_out + pos_enc
        
        # Process conditioning
        conditioning = config.make_cond(
            train_batch.T_cpf_tm1_cpf_t,
            T_world_cpf=train_batch.T_world_cpf,
            hand_positions_wrt_cpf=train_batch.joints_wrt_cpf[:, :, 19:21, :].reshape(batch, time, 6) 
                if config.include_hand_positions_cond else None
        )
        
        # Encode conditioning
        encoder_out = self._encode_conditioning(conditioning, batch, time, device, noise_emb)
        
        # Add previous window conditioning if configured
        if config.condition_on_prev_window and x_t.prev_window is not None:
            prev_encoded = self._encode_conditioning(
                config.make_cond(
                    x_t.prev_window.T_cpf_tm1_cpf_t,
                    T_world_cpf=x_t.prev_window.T_world_cpf,
                    hand_positions_wrt_cpf=None
                ),
                batch, time, device, torch.zeros_like(noise_emb)
            )
            encoder_out = encoder_out + self.prev_window_proj(prev_encoded)
        
        # Add noise token if configured
        if self.noise_emb_token_proj is not None:
            noise_emb_token = self.noise_emb_token_proj(noise_emb)
            encoder_out = torch.cat([noise_emb_token[:, None, :], encoder_out], dim=1)
            decoder_out = torch.cat([noise_emb_token[:, None, :], decoder_out], dim=1)
            num_tokens = time + 1
        else:
            num_tokens = time
        
        # Process through decoder layers with cross-attention to encoded conditioning
        for layer in self.decoder_layers:
            decoder_out = layer(decoder_out, None, noise_emb=noise_emb, cond=encoder_out)
        
        # Remove noise token if added
        if self.noise_emb_token_proj is not None:
            decoder_out = decoder_out[:, 1:, :]
        
        # Decode output
        outputs = []
        for key, decoder in self.decoders.items():
            out = decoder(decoder_out)
            if key in ("body_rot6d", "hand_rot6d"):
                out = out.reshape(batch, time, -1)
                # Project to valid rot6d representation
                out = project_rot6d(out.view(*out.shape[:-1], -1, 6))
                out = out.reshape(batch, time, -1)
            outputs.append(out)
            
        packed_output = torch.cat(outputs, dim=-1)
        
        if return_dict:
            return BaseOutput(sample=packed_output)
        return packed_output


    def save_config(self, save_directory: str):
        config = self.config  # Assuming you have a config attribute
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config.to_json(), f)

class MotionDiffusionPipeline(DiffusionPipeline):
    """Pipeline for generating human motion using diffusion models"""

    unet: MotionUNet
    scheduler: DDIMScheduler  # Change the type hint to DDIMScheduler

    def __init__(
        self,
        unet: MotionUNet,
        scheduler: DDIMScheduler,  # Accept a DDIMScheduler
    ):
        super().__init__()
        self.register_modules(
            unet=unet,
            scheduler=scheduler
        )

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        num_inference_steps: int = 50,  # Use fewer steps for DDIM
        generator: Optional[torch.Generator] = None,
        train_batch: Optional[EgoTrainingData] = None,
        return_intermediates: bool = False,
    ) -> MotionDiffusionPipelineOutput:
        # Set up scheduler
        self.scheduler.set_timesteps(num_inference_steps)

        # Initialize noise
        batch_size, time = train_batch.T_world_cpf.shape[:2]
        shape = (batch_size, time, self.unet.config.d_state)
        sample = torch.randn(shape, generator=generator, device=self.device)

        # Initialize storage for intermediate states if requested
        intermediates = [] if return_intermediates else None

        # Denoising loop
        timesteps = self.scheduler.timesteps.to(self.device)
        for t in timesteps:
            # Get model prediction
            model_output = self.unet.forward(
                sample=sample,
                timestep=t,
                train_batch=train_batch,
                return_dict=False
            )

            # Scheduler step
            sample = self.scheduler.step(
                model_output=model_output,
                timestep=t,
                sample=sample,
                generator=generator
            ).prev_sample

            if return_intermediates:
                intermediates.append(
                    EgoDenoiseTraj.unpack(
                        sample,
                        include_hands=self.unet.config.include_hand_motion,
                        should_project_rot6d=False
                    )
                )

        # Convert final sample to motion
        motion = EgoDenoiseTraj.unpack(
            sample,
            include_hands=self.unet.config.include_hand_motion,
            should_project_rot6d=False
        )

        return MotionDiffusionPipelineOutput(
            motion=motion,
            intermediate_states=intermediates
        )

    def save_custom_config(self, save_directory: str):
        """Save the UNet config to the specified directory"""
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, "unet_config.json")
        with open(config_path, "w") as f:
            json.dump(self.unet.config.to_json(), f)

    @classmethod
    def load_custom_config(cls, save_directory: str) -> EgoDenoiserConfig:
        """Load the UNet config from the specified directory"""
        config_path = os.path.join(save_directory, "unet_config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"Config file not found at {config_path}")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return EgoDenoiserConfig.from_json(config_dict)

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save the pipeline's models and configuration"""
        super().save_pretrained(save_directory, **kwargs)
        self.save_custom_config(save_directory)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Union[str, Path],
        **kwargs
    ) -> "MotionDiffusionPipeline":
        # Load the custom config first
        config = cls.load_custom_config(pretrained_model_path)

        # Create UNet with loaded config
        unet = MotionUNet(config)

        # Create DDIM scheduler
        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="sample",
            clip_sample=False,
        )

        # Load the pretrained weights
        pipeline = super().from_pretrained(
            pretrained_model_path,
            unet=unet,
            scheduler=scheduler,
            **kwargs
        )

        return pipeline