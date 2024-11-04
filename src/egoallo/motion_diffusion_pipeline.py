from dataclasses import dataclass
from typing import Optional, Union, Dict

import os
import json
import torch
from diffusers import DiffusionPipeline, UNet2DModel, ModelMixin
from diffusers.utils import BaseOutput
from diffusers.schedulers import DDPMScheduler
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
        if config.include_hands:
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

    def _encode_window(
        self,
        window: EgoDenoiseTraj,
        batch: int,
        time: int,
        device: torch.device,
        noise_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Encode window if provided."""
            
        # Encode window components
        encoded = (
            self.encoders["betas"](window.betas.reshape(batch, time, -1))
            + self.encoders["body_rot6d"](window.body_rot6d.reshape(batch, time, -1))
            + self.encoders["contacts"](window.contacts)
        )
        
        if self.config.include_hands and window.hand_rot6d is not None:
            encoded += self.encoders["hand_rot6d"](
                window.hand_rot6d.reshape(batch, time, -1)
            )

        # Add positional encoding
        if self.config.positional_encoding == "rope":
            pos_enc = 0
        elif self.config.positional_encoding == "transformer":
            pos_enc = make_positional_encoding(
                d_latent=self.config.d_latent,
                length=time,
                dtype=encoded.dtype,
            )[None, ...].to(device)
        else:
            raise ValueError(f"Unknown positional encoding: {self.config.positional_encoding}")
            
        encoded = encoded + pos_enc

        # Process through transformer layers
        for layer in self.encoder_layers:
            encoded = layer(encoded, None, noise_emb=noise_emb)
            
       
        return encoded

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
        x_t = EgoDenoiseTraj.unpack(sample, include_hands=config.include_hands)
        batch, time = x_t.betas.shape[:2]
        device = sample.device
        
        # Embed noise level
        noise_emb = self.noise_emb(timestep)
        
        # Encode current window
        decoder_out = self._encode_window(
            window=x_t,
            batch=batch,
            time=time,
            device=device,
            noise_emb=noise_emb
        )
        
        # Process conditioning
        conditioning = config.make_cond(
            train_batch.T_cpf_tm1_cpf_t,
            T_world_cpf=train_batch.T_world_cpf,
            hand_positions_wrt_cpf=train_batch.joints_wrt_cpf[:, :, 19:21, :].reshape(batch, time, 6)
        )
        
        # Process conditioning if provided
        if conditioning is not None:
            cond_embeds = []
            for name in config.cond_component_names():
                if name in conditioning:
                    component = conditioning[name]
                    embed = self.cond_embeddings[name](component)
                    cond_embeds.append(embed)
            cond = torch.cat(cond_embeds, dim=-1)
            encoder_out = self.latent_from_cond(cond)
        else:
            encoder_out = torch.zeros((batch, time, config.d_latent), device=device)
            
        # Add positional encoding
        if config.positional_encoding == "rope":
            pos_enc = 0
        elif config.positional_encoding == "transformer":
            pos_enc = make_positional_encoding(
                d_latent=config.d_latent,
                length=time,
                dtype=sample.dtype,
            )[None, ...].to(device)
        else:
            raise ValueError(f"Unknown positional encoding: {config.positional_encoding}")
            
        encoder_out = encoder_out + pos_enc
        
        # Encode previous window if configured
        if config.condition_on_prev_window and x_t.prev_window is not None:
            prev_encoded = self._encode_window(
                window=x_t.prev_window,
                batch=batch,
                time=time,
                device=device,
                noise_emb=None  # No noise embedding needed for clean motion
            )
            # Project and add to encoder output
            prev_encoded = self.prev_window_proj(prev_encoded)
            encoder_out = encoder_out + prev_encoded
            
        # Add noise token if configured
        if self.noise_emb_token_proj is not None:
            noise_emb_token = self.noise_emb_token_proj(noise_emb)
            encoder_out = torch.cat([noise_emb_token[:, None, :], encoder_out], dim=1)
            decoder_out = torch.cat([noise_emb_token[:, None, :], decoder_out], dim=1)
            num_tokens = time + 1
        else:
            num_tokens = time
            
        # Forward pass through transformer
        for layer in self.encoder_layers:
            encoder_out = layer(encoder_out, None, noise_emb=noise_emb)
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
                out = out.reshape(batch, time, -1, 6)
                out = project_rot6d(out).reshape(batch, time, -1)
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
    
    def __init__(
        self,
        unet: MotionUNet,
        scheduler: DDPMScheduler,
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
        num_inference_steps: int = 1000,
        generator: Optional[torch.Generator] = None,
        conditioning: Optional[Dict[str, torch.Tensor]] = None,
        return_intermediates: bool = False,
    ) -> MotionDiffusionPipelineOutput:
        # Set up scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        
        # Initialize noise
        shape = (batch_size, self.unet.config.get_d_state())
        noise = torch.randn(shape, generator=generator, device=self.device)
        
        # Initialize storage for intermediate states if requested
        intermediates = [] if return_intermediates else None
        
        # Denoising loop
        sample = noise
        for t in self.scheduler.timesteps:
            # Get model prediction
            model_output = self.unet(
                sample=sample,
                timestep=t,
                conditioning=conditioning
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
                        include_hands=self.unet.config.include_hands,
                        project_rot6d=True
                    )
                )
        
        # Convert final sample to motion
        motion = EgoDenoiseTraj.unpack(
            sample,
            include_hands=self.unet.config.include_hands,
            project_rot6d=True
        )
        
        return MotionDiffusionPipelineOutput(
            motion=motion,
            intermediate_states=intermediates
        )

    @classmethod
    def from_config(
        cls,
        config: EgoDenoiserConfig,
    ) -> "MotionDiffusionPipeline":
        """Create a pipeline from a configuration"""
        unet = MotionUNet(config)
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2"
        )
        return cls(unet=unet, scheduler=scheduler) 