from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from functools import cache, cached_property
from typing import Literal, Optional, Dict, List

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Bool, Float
from rotary_embedding_torch import RotaryEmbedding
from egoallo.setup_logger import setup_logger
from torch import Tensor, nn

from .fncsmpl import SmplhModel, SmplhShapedAndPosed
from .tensor_dataclass import TensorDataclass
from .transforms import SE3, SO3

logger = setup_logger(output=None, name=__name__)

def project_rotmats_via_svd(
    rotmats: Float[Tensor, "*batch 3 3"],
) -> Float[Tensor, "*batch 3 3"]:
    u, s, vh = torch.linalg.svd(rotmats)
    del s
    return torch.einsum("...ij,...jk->...ik", u, vh)

def project_rot6d(rot6d: Float[Tensor, "*batch joints 6"]) -> Float[Tensor, "*batch joints 6"]:
    """Project rot6d representations to valid rotations."""
    a = rot6d[..., :3]
    b = rot6d[..., 3:]
    r1 = torch.nn.functional.normalize(a, dim=-1)
    b_proj = b - torch.sum(r1 * b, dim=-1, keepdim=True) * r1
    r2 = torch.nn.functional.normalize(b_proj, dim=-1)
    projected_rot6d = torch.cat([r1, r2], dim=-1)
    return projected_rot6d



class EgoDenoiseTraj(TensorDataclass):
    """
    Data structure for denoising. Contains tensors that we are denoising, as
    well as utilities for packing and unpacking them.
    """

    betas: Float[Tensor, "*batch timesteps 16"]
    """Body shape parameters."""

    body_rot6d: Float[Tensor, "*batch timesteps 21 6"]
    """Local orientations for each body joint in rot6d representation."""

    contacts: Float[Tensor, "*batch timesteps 21"]
    """Contact boolean for each joint."""

    hand_rot6d: Float[Tensor, "*batch timesteps 30 6"] | None
    """Local orientations for each hand joint in rot6d representation."""

    prev_window: Optional[EgoDenoiseTraj] = None
    """Previous window trajectory for conditioning."""

    @staticmethod
    def get_packed_dim(include_hands: bool) -> int:
        """Get dimension of packed representation."""
        packed_dim = 16 + 21 * 6 + 21  # betas + body_rot6d + contacts
        if include_hands:
            packed_dim += 30 * 6  # hand_rot6d
        return packed_dim

    def pack(self) -> Float[Tensor, "*batch timesteps d_state"]:
        """Pack trajectory into a single flattened vector."""
        (*batch, time, num_joints, rot_dim) = self.body_rot6d.shape
        assert num_joints == 21 and rot_dim == 6

        to_cat = [
            self.betas.reshape((*batch, time, -1)),
            self.body_rot6d.reshape((*batch, time, -1)),
            self.contacts,
        ]

        if self.hand_rot6d is not None:
            to_cat.append(self.hand_rot6d.reshape((*batch, time, -1)))

        return torch.cat(to_cat, dim=-1)

    @classmethod
    def unpack(
        cls,
        x: Float[Tensor, "*batch timesteps d_state"],
        include_hands: bool,
        project_rot6d: bool = False,
        prev_window: Optional["EgoDenoiseTraj"] = None,
    ) -> "EgoDenoiseTraj":
        """Unpack trajectory from a single flattened vector."""
        (*batch, time, d_state) = x.shape
        assert d_state == cls.get_packed_dim(include_hands)

        if include_hands:
            betas, body_rot6d_flat, contacts, hand_rot6d_flat = torch.split(
                x, [16, 21 * 6, 21, 30 * 6], dim=-1
            )
            body_rot6d = body_rot6d_flat.reshape((*batch, time, 21, 6))
            hand_rot6d = hand_rot6d_flat.reshape((*batch, time, 30, 6))
        else:
            betas, body_rot6d_flat, contacts = torch.split(
                x, [16, 21 * 6, 21], dim=-1
            )
            body_rot6d = body_rot6d_flat.reshape((*batch, time, 21, 6))
            hand_rot6d = None

        if project_rot6d:
            body_rot6d = project_rot6d(body_rot6d)
            if hand_rot6d is not None:
                hand_rot6d = project_rot6d(hand_rot6d)

        return cls(
            betas=betas,
            body_rot6d=body_rot6d,
            contacts=contacts,
            hand_rot6d=hand_rot6d,
            prev_window=prev_window,
        )

    def with_prev_window(self, prev_window: "EgoDenoiseTraj") -> "EgoDenoiseTraj":
        """Create a new instance with a previous window for conditioning."""
        return EgoDenoiseTraj(
            betas=self.betas,
            body_rot6d=self.body_rot6d,
            contacts=self.contacts,
            hand_rot6d=self.hand_rot6d,
            prev_window=prev_window,
        )

    def apply_to_body(self, body_model: SmplhModel) -> SmplhShapedAndPosed:
        """Apply trajectory to SMPL body model."""
        device = self.betas.device
        dtype = self.betas.dtype
        assert self.hand_rot6d is not None
        shaped = body_model.with_shape(self.betas)

        # Convert rot6d to rotation matrices
        body_rotmats = SO3.from_rot6d(
            self.body_rot6d.reshape(-1, 6)
        ).as_matrix().reshape(self.body_rot6d.shape[:-1] + (3, 3))

        hand_rotmats = SO3.from_rot6d(
            self.hand_rot6d.reshape(-1, 6)
        ).as_matrix().reshape(self.hand_rot6d.shape[:-1] + (3, 3))

        posed = shaped.with_pose(
            T_world_root=SE3.identity(device=device, dtype=dtype).parameters(),
            local_quats=SO3.from_matrix(
                torch.cat([body_rotmats, hand_rotmats], dim=-3)
            ).wxyz,
        )
        return posed

@dataclass(frozen=True)
class EgoDenoiserConfig:
    max_t: int = 1000
    fourier_enc_freqs: int = 3
    d_latent: int = 512
    d_feedforward: int = 2048
    d_noise_emb: int = 1024
    num_heads: int = 4
    encoder_layers: int = 6
    decoder_layers: int = 6
    dropout_p: float = 0.0
    activation: Literal["gelu", "relu"] = "gelu"

    positional_encoding: Literal["transformer", "rope"] = "rope"
    noise_conditioning: Literal["token", "film"] = "token"

    xattn_mode: Literal["kv_from_cond_q_from_x", "kv_from_x_q_from_cond"] = (
        "kv_from_cond_q_from_x"
    )

    include_canonicalized_cpf_rotation_in_cond: bool = True
    include_hands: bool = True
    """Whether to include hand joints (+15 per hand) in the denoised state."""

    cond_param: Literal[
        "ours", "canonicalized", "absolute", "absrel", "absrel_global_deltas"
    ] = "ours"
    """Which conditioning parameterization to use.

    "ours" is the default, we try to be clever and design something with nice
        equivariance properties.
    "canonicalized" contains a transformation that's canonicalized to aligned
        to the first frame.
    "absolute" is the naive case, where we just pass in transformations
        directly.
    """

    include_hand_positions_cond: bool = False
    """Whether to include hand positions in the conditioning information."""

    condition_on_prev_window: bool = False
    """Whether to condition on previous motion window."""
    
    prev_window_encoder_layers: int = 2
    """Number of transformer layers for encoding previous window."""

    @cached_property
    def d_state(self) -> int:
        """Dimensionality of the state vector."""
        return EgoDenoiseTraj.get_packed_dim(self.include_hands)
        
    @cached_property
    def d_cond(self) -> int:
        """Dimensionality of conditioning vector."""
        if self.cond_param == "ours":
            d_cond = 0
            d_cond += 6  # Relative CPF rotation in rot6d representation.
            d_cond += 3  # Relative CPF translation.
            d_cond += 1  # Floor height.
            if self.include_canonicalized_cpf_rotation_in_cond:
                d_cond += 6  # Canonicalized CPF rotation in rot6d.
        elif self.cond_param == "canonicalized":
            d_cond = 6 + 3  # Rotation (rot6d) and translation.
        elif self.cond_param == "absolute":
            d_cond = 6 + 3  # Rotation (rot6d) and translation.
        elif self.cond_param == "absrel":
            # Both absolute and relative (rot6d + translation for each).
            d_cond = (6 + 3) * 2
        elif self.cond_param == "absrel_global_deltas":
            # Absolute rotation and translation, plus relative rotation and translation.
            d_cond = 6 + 3 + 6 + 3
        else:
            assert False

        # Add two 3D positions to the conditioning dimension if we're including
        # hand conditioning.
        if self.include_hand_positions_cond:
            d_cond += 6

        # Apply Fourier encoding
        d_cond = d_cond + d_cond * self.fourier_enc_freqs * 2  # Fourier encoding.

        return d_cond

    def make_cond(
        self,
        T_cpf_tm1_cpf_t: Tensor,  # Shape: (batch, time, 7)
        T_world_cpf: Tensor,       # Shape: (batch, time, 7)
        T_world_cpf_0: Optional[Tensor] = None,  # Shape: (batch, 7)
        hand_positions_wrt_cpf: Optional[Tensor] = None,  # Shape: (batch, time, 6)
    ) -> Dict[str, Tensor]:
        """
        Construct conditioning information based on the selected cond_param.
        Returns a dictionary with separate tensors for each semantic component.
        """
        batch, time, _ = T_cpf_tm1_cpf_t.shape
        device = T_cpf_tm1_cpf_t.device
        dtype = T_cpf_tm1_cpf_t.dtype
        cond_dict = {}

        # Compute floor height (common to all cond_param options)
        height_from_floor = T_world_cpf[..., 6:7]  # Shape: (batch, time, 1)
        cond_dict['floor_height'] = height_from_floor  # Shape: (batch, time, 1)

        if self.cond_param == "ours":
            # Relative CPF pose in rot6d and translation
            rel_pose = SE3(T_cpf_tm1_cpf_t)
            rel_rot6d = SO3(rel_pose.rotation().wxyz).as_rot6d()  # Shape: (batch, time, 6)
            rel_trans = rel_pose.translation()                    # Shape: (batch, time, 3)
            cond_dict['rel_rot6d'] = rel_rot6d
            cond_dict['rel_trans'] = rel_trans

            if self.include_canonicalized_cpf_rotation_in_cond:
                # Canonicalized CPF rotation in rot6d
                R_world_cpf = SE3(T_world_cpf).rotation()
                forward_cpf = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
                forward_world = R_world_cpf @ forward_cpf
                yaw_angles = -torch.atan2(forward_world[..., 1], forward_world[..., 0])
                R_canonical_world = SO3.from_z_radians(yaw_angles)
                R_canonical_cpf = R_canonical_world @ R_world_cpf
                canonical_rot6d = R_canonical_cpf.as_rot6d()
                cond_dict['canonical_rot6d'] = canonical_rot6d  # Shape: (batch, time, 6)

        elif self.cond_param == "canonicalized":
            assert T_world_cpf_0 is not None, "T_world_cpf_0 is required for 'canonicalized' cond_param."
            # Compute canonicalized transformations relative to the first frame
            T_world_cpf_0_inv = SE3.inv(SE3(T_world_cpf_0))  # Shape: (batch, 7)
            T_cpf_0_cpf_t = SE3.compose(T_world_cpf_0_inv[:, None, :], T_world_cpf)  # Shape: (batch, time, 7)
            canonical_pose = SE3(T_cpf_0_cpf_t)
            canonical_rot6d = SO3(canonical_pose.rotation().wxyz).as_rot6d()  # Shape: (batch, time, 6)
            canonical_trans = canonical_pose.translation()                    # Shape: (batch, time, 3)
            cond_dict['canonical_rot6d'] = canonical_rot6d
            cond_dict['canonical_trans'] = canonical_trans

        elif self.cond_param == "absolute":
            # Use absolute CPF pose in world coordinates
            abs_pose = SE3(T_world_cpf)
            abs_rot6d = SO3(abs_pose.rotation().wxyz).as_rot6d()  # Shape: (batch, time, 6)
            abs_trans = abs_pose.translation()                    # Shape: (batch, time, 3)
            cond_dict['abs_rot6d'] = abs_rot6d
            cond_dict['abs_trans'] = abs_trans

        elif self.cond_param == "absrel":
            # Combine absolute and relative transformations
            # Absolute CPF pose
            abs_pose = SE3(T_world_cpf)
            abs_rot6d = SO3(abs_pose.rotation().wxyz).as_rot6d()  # Shape: (batch, time, 6)
            abs_trans = abs_pose.translation()                    # Shape: (batch, time, 3)
            cond_dict['abs_rot6d'] = abs_rot6d
            cond_dict['abs_trans'] = abs_trans
            # Relative CPF pose
            rel_pose = SE3(T_cpf_tm1_cpf_t)
            rel_rot6d = SO3(rel_pose.rotation().wxyz).as_rot6d()
            rel_trans = rel_pose.translation()
            cond_dict['rel_rot6d'] = rel_rot6d
            cond_dict['rel_trans'] = rel_trans

        elif self.cond_param == "absrel_global_deltas":
            # Similar to 'absrel' but includes global deltas
            # Absolute CPF pose
            abs_pose = SE3(T_world_cpf)
            abs_rot6d = SO3(abs_pose.rotation().wxyz).as_rot6d()
            abs_trans = abs_pose.translation()
            cond_dict['abs_rot6d'] = abs_rot6d
            cond_dict['abs_trans'] = abs_trans
            # Relative CPF pose
            rel_pose = SE3(T_cpf_tm1_cpf_t)
            rel_rot6d = SO3(rel_pose.rotation().wxyz).as_rot6d()
            rel_trans = rel_pose.translation()
            cond_dict['rel_rot6d'] = rel_rot6d
            cond_dict['rel_trans'] = rel_trans
            # Global deltas
            delta_rot6d = abs_rot6d[:, 1:, :] - abs_rot6d[:, :-1, :]
            delta_trans = abs_trans[:, 1:, :] - abs_trans[:, :-1, :]
            # Pad to match sequence length
            delta_rot6d = torch.cat([delta_rot6d[:, :1, :], delta_rot6d], dim=1)
            delta_trans = torch.cat([delta_trans[:, :1, :], delta_trans], dim=1)
            cond_dict['delta_rot6d'] = delta_rot6d
            cond_dict['delta_trans'] = delta_trans

        else:
            raise ValueError(f"Unknown cond_param: {self.cond_param}")

        # Include hand positions in conditioning if applicable
        if self.include_hand_positions_cond and hand_positions_wrt_cpf is not None:
            cond_dict['hand_positions'] = hand_positions_wrt_cpf  # Shape: (batch, time, 6)

        return cond_dict

    def cond_component_names(self) -> List[str]:
        """List of conditional component names based on configuration."""
        names = ['floor_height']  # Common to all cond_param options

        if self.cond_param == 'ours':
            names.extend(['rel_rot6d', 'rel_trans'])
            if self.include_canonicalized_cpf_rotation_in_cond:
                names.append('canonical_rot6d')
        elif self.cond_param == 'canonicalized':
            names.extend(['canonical_rot6d', 'canonical_trans'])
        elif self.cond_param == 'absolute':
            names.extend(['abs_rot6d', 'abs_trans'])
        elif self.cond_param == 'absrel':
            names.extend(['abs_rot6d', 'abs_trans', 'rel_rot6d', 'rel_trans'])
        elif self.cond_param == 'absrel_global_deltas':
            names.extend([
                'abs_rot6d', 'abs_trans',
                'rel_rot6d', 'rel_trans',
                'delta_rot6d', 'delta_trans'
            ])
        else:
            raise ValueError(f"Unknown cond_param: {self.cond_param}")

        # Include hand positions if applicable
        if self.include_hand_positions_cond:
            names.append('hand_positions')

        return names

    def get_cond_component_dim(self, name: str) -> int:
        """Return the dimension of each conditional component."""
        if name == 'floor_height':
            return 1
        elif name in ['rel_rot6d', 'canonical_rot6d', 'abs_rot6d', 'delta_rot6d']:
            return 6
        elif name in ['rel_trans', 'canonical_trans', 'abs_trans', 'delta_trans']:
            return 3
        elif name == 'hand_positions':
            return 6
        else:
            raise ValueError(f"Unknown conditional component: {name}")

    def to_json(self) -> str:
        """Serialize the config to a JSON string."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> "EgoDenoiserConfig":
        """Create a config from a JSON string."""
        config_dict = json.loads(json_str)
        return cls(**config_dict)


class EgoDenoiser(nn.Module):
    """Denoising network for human motion.

    Inputs are noisy trajectory, conditioning information, and timestep.
    Output is denoised trajectory.
    """

    def __init__(self, config: EgoDenoiserConfig):
        super().__init__()

        self.config = config
        Activation = {"gelu": nn.GELU, "relu": nn.ReLU}[config.activation]

        # MLP encoders and decoders for each modality we want to denoise.
        modality_dims: dict[str, int] = {
            "betas": 16,
            "body_rot6d": 21 * 6,
            "contacts": 21,
        }
        if config.include_hands:
            modality_dims["hand_rot6d"] = 30 * 6

        assert sum(modality_dims.values()) == self.get_d_state()
        self.encoders = nn.ModuleDict(
            {
                k: nn.Sequential(
                    nn.Linear(modality_dim, config.d_latent),
                    Activation(),
                    nn.Linear(config.d_latent, config.d_latent),
                    Activation(),
                    nn.Linear(config.d_latent, config.d_latent),
                )
                for k, modality_dim in modality_dims.items()
            }
        )
        self.decoders = nn.ModuleDict(
            {
                k: nn.Sequential(
                    nn.Linear(config.d_latent, config.d_latent),
                    nn.LayerNorm(normalized_shape=config.d_latent),
                    Activation(),
                    nn.Linear(config.d_latent, config.d_latent),
                    Activation(),
                    nn.Linear(config.d_latent, modality_dim),
                )
                for k, modality_dim in modality_dims.items()
            }
        )

        # Helpers for converting between input dimensionality and latent dimensionality.
        self.latent_from_cond = nn.Linear(len(config.cond_component_names()) * config.d_latent, config.d_latent)

        # Noise embedder.
        self.noise_emb = nn.Embedding(
            # index 0 will be t=1
            # index 999 will be t=1000
            num_embeddings=config.max_t,
            embedding_dim=config.d_noise_emb,
        )
        self.noise_emb_token_proj = (
            nn.Linear(config.d_noise_emb, config.d_latent, bias=False)
            if config.noise_conditioning == "token"
            else None
        )

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

        # Embedding layers for the conditional tokens
        self.cond_embeddings = nn.ModuleDict()
        for name in self.config.cond_component_names():
            self.cond_embeddings[name] = nn.Linear(
                self.config.get_cond_component_dim(name), config.d_latent
            )

    def get_d_state(self) -> int:
        return EgoDenoiseTraj.get_packed_dim(self.config.include_hands)

    def forward(
        self,
        x_t_packed: Float[Tensor, "batch time state_dim"],
        t: Float[Tensor, "batch"],
        *,
        T_world_cpf: Float[Tensor, "batch time 7"],
        T_cpf_tm1_cpf_t: Float[Tensor, "batch time 7"],
        project_output_rot6d: bool,
        hand_positions_wrt_cpf: Float[Tensor, "batch time 6"] | None,
        mask: Bool[Tensor, "batch time"] | None,
        cond_dropout_keep_mask: Bool[Tensor, "batch"] | None = None,
    ) -> Float[Tensor, "batch time state_dim"]:
        """
        Predict a denoised trajectory. Note that `t` refers to a noise
        level, not a timestep.
        """
        config = self.config
        x_t = EgoDenoiseTraj.unpack(
            x_t_packed, include_hands=self.config.include_hands
        )
        (batch, time, num_body_joints, rot_dim) = x_t.body_rot6d.shape
        assert num_body_joints == 21 and rot_dim == 6

        # Encode the trajectory into a single vector per timestep
        x_t_encoded = (
            self.encoders["betas"](x_t.betas.reshape((batch, time, -1)))
            + self.encoders["body_rot6d"](x_t.body_rot6d.reshape((batch, time, -1)))
            + self.encoders["contacts"](x_t.contacts)
        )

        if self.config.include_hands:
            assert x_t.hand_rot6d is not None
            x_t_encoded += self.encoders["hand_rot6d"](
                x_t.hand_rot6d.reshape((batch, time, -1))
            )
        assert x_t_encoded.shape == (batch, time, config.d_latent)

        # Embed the diffusion noise level.
        assert t.shape == (batch,)
        noise_emb = self.noise_emb(t - 1)
        assert noise_emb.shape == (batch, config.d_noise_emb)

        # Prepare conditioning information.
        cond = config.make_cond(
            T_cpf_tm1_cpf_t,
            T_world_cpf=T_world_cpf,
            hand_positions_wrt_cpf=hand_positions_wrt_cpf,
        )
        # Embed each conditional component separately
        cond_embeds = []
        for name in self.config.cond_component_names():
            component = cond[name]  # Shape: (batch, time, d)
            embed = self.cond_embeddings[name](component)  # Shape: (batch, time, d_latent)
            cond_embeds.append(embed)
        cond = torch.cat(cond_embeds, dim=-1)

        # Randomly drop out conditioning information; this serves as a
        # regularizer that aims to improve sample diversity.
        if cond_dropout_keep_mask is not None:
            assert cond_dropout_keep_mask.shape == (batch,)
            cond = cond * cond_dropout_keep_mask[:, None, None]

        # Prepare encoder and decoder inputs.
        if config.positional_encoding == "rope":
            pos_enc = 0
        elif config.positional_encoding == "transformer":
            pos_enc = make_positional_encoding(
                d_latent=config.d_latent,
                length=time,
                dtype=cond.dtype,
            )[None, ...].to(x_t_encoded.device)
            assert pos_enc.shape == (1, time, config.d_latent)
        else:
            assert False

        encoder_out = self.latent_from_cond(cond) + pos_enc
        decoder_out = x_t_encoded + pos_enc

        # Append the noise embedding to the encoder and decoder inputs.
        # This is weird if we're using rotary embeddings!
        if self.noise_emb_token_proj is not None:
            noise_emb_token = self.noise_emb_token_proj(noise_emb)
            assert noise_emb_token.shape == (batch, config.d_latent)
            encoder_out = torch.cat([noise_emb_token[:, None, :], encoder_out], dim=1)
            decoder_out = torch.cat([noise_emb_token[:, None, :], decoder_out], dim=1)
            assert (
                encoder_out.shape
                == decoder_out.shape
                == (batch, time + 1, config.d_latent)
            )
            num_tokens = time + 1
        else:
            num_tokens = time

        # Compute attention mask. This needs to be a fl
        if mask is None:
            attn_mask = None
        else:
            assert mask.shape == (batch, time)
            assert mask.dtype == torch.bool
            if self.noise_emb_token_proj is not None:  # Account for noise token.
                mask = torch.cat([mask.new_ones((batch, 1)), mask], dim=1)
            # Last two dimensions of mask are (query, key). We're masking out only keys;
            # it's annoying for the softmax to mask out entire rows without getting NaNs.
            attn_mask = mask[:, None, None, :].repeat(1, 1, num_tokens, 1)
            assert attn_mask.shape == (batch, 1, num_tokens, num_tokens)
            assert attn_mask.dtype == torch.bool

        # Forward pass through transformer.
        for layer in self.encoder_layers:
            encoder_out = layer(encoder_out, attn_mask, noise_emb=noise_emb)
        for layer in self.decoder_layers:
            decoder_out = layer(
                decoder_out, attn_mask, noise_emb=noise_emb, cond=encoder_out
            )

        # Remove the extra token corresponding to the noise embedding.
        if self.noise_emb_token_proj is not None:
            decoder_out = decoder_out[:, 1:, :]
        assert isinstance(decoder_out, Tensor)
        assert decoder_out.shape == (batch, time, config.d_latent)

        packed_output = torch.cat(
            [
                (
                    project_rot6d(
                        modality_decoder(decoder_out).reshape(batch, time, -1, 6)
                    ).reshape(batch, time, -1)
                    if project_output_rot6d and key in ("body_rot6d", "hand_rot6d")
                    else modality_decoder(decoder_out)
                )
                for key, modality_decoder in self.decoders.items()
            ],
            dim=-1,
        )
        assert packed_output.shape == (batch, time, self.get_d_state())

        # Return packed output.
        return packed_output


@cache
def make_positional_encoding(
    d_latent: int, length: int, dtype: torch.dtype
) -> Float[Tensor, "length d_latent"]:
    """Computes standard Transformer positional encoding."""
    pe = torch.zeros(length, d_latent, dtype=dtype)
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_latent, 2).float() * (-np.log(10000.0) / d_latent)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    assert pe.shape == (length, d_latent)
    return pe


def fourier_encode(
    x: Float[Tensor, "*#batch channels"], freqs: int
) -> Float[Tensor, "*#batch channels+2*freqs*channels"]:
    """Apply Fourier encoding to a tensor."""
    *batch_axes, x_dim = x.shape
    coeffs = 2.0 ** torch.arange(freqs, device=x.device)
    scaled = (x[..., None] * coeffs).reshape((*batch_axes, x_dim * freqs))
    return torch.cat(
        [
            x,
            torch.sin(torch.cat([scaled, scaled + torch.pi / 2.0], dim=-1)),
        ],
        dim=-1,
    )


@dataclass(frozen=True)
class TransformerBlockConfig:
    d_latent: int
    d_noise_emb: int
    d_feedforward: int
    n_heads: int
    dropout_p: float
    activation: Literal["gelu", "relu"]
    include_xattn: bool
    use_rope_embedding: bool
    use_film_noise_conditioning: bool
    xattn_mode: Literal["kv_from_cond_q_from_x", "kv_from_x_q_from_cond"]


class TransformerBlock(nn.Module):
    """An even-tempered Transformer block."""

    def __init__(self, config: TransformerBlockConfig) -> None:
        super().__init__()
        self.sattn_qkv_proj = nn.Linear(
            config.d_latent, config.d_latent * 3, bias=False
        )
        self.sattn_out_proj = nn.Linear(config.d_latent, config.d_latent, bias=False)

        self.layernorm1 = nn.LayerNorm(config.d_latent)
        self.layernorm2 = nn.LayerNorm(config.d_latent)

        assert config.d_latent % config.n_heads == 0
        self.rotary_emb = (
            RotaryEmbedding(config.d_latent // config.n_heads)
            if config.use_rope_embedding
            else None
        )

        if config.include_xattn:
            self.xattn_kv_proj = nn.Linear(
                config.d_latent, config.d_latent * 2, bias=False
            )
            self.xattn_q_proj = nn.Linear(config.d_latent, config.d_latent, bias=False)
            self.xattn_layernorm = nn.LayerNorm(config.d_latent)
            self.xattn_out_proj = nn.Linear(
                config.d_latent, config.d_latent, bias=False
            )

        self.norm_no_learnable = nn.LayerNorm(
            config.d_feedforward, elementwise_affine=False, bias=False
        )
        self.activation = {"gelu": nn.GELU, "relu": nn.ReLU}[config.activation]()
        self.dropout = nn.Dropout(config.dropout_p)

        self.mlp0 = nn.Linear(config.d_latent, config.d_feedforward)
        self.mlp_film_cond_proj = (
            zero_module(
                nn.Linear(config.d_noise_emb, config.d_feedforward * 2, bias=False)
            )
            if config.use_film_noise_conditioning
            else None
        )
        self.mlp1 = nn.Linear(config.d_feedforward, config.d_latent)
        self.config = config

    def forward(
        self,
        x: Float[Tensor, "batch tokens d_latent"],
        attn_mask: Bool[Tensor, "batch 1 tokens tokens"] | None,
        noise_emb: Float[Tensor, "batch d_noise_emb"],
        cond: Float[Tensor, "batch tokens d_latent"] | None = None,
    ) -> Float[Tensor, "batch tokens d_latent"]:
        config = self.config
        (batch, time, d_latent) = x.shape

        # Self-attention.
        # We put layer normalization after the residual connection.
        x = self.layernorm1(x + self._sattn(x, attn_mask))

        # Include conditioning.
        if config.include_xattn:
            assert cond is not None
            x = self.xattn_layernorm(x + self._xattn(x, attn_mask, cond=cond))

        mlp_out = x
        mlp_out = self.mlp0(mlp_out)
        mlp_out = self.activation(mlp_out)

        # FiLM-style conditioning.
        if self.mlp_film_cond_proj is not None:
            scale, shift = torch.chunk(
                self.mlp_film_cond_proj(noise_emb), chunks=2, dim=-1
            )
            assert scale.shape == shift.shape == (batch, config.d_feedforward)
            mlp_out = (
                self.norm_no_learnable(mlp_out) * (1.0 + scale[:, None, :])
                + shift[:, None, :]
            )

        mlp_out = self.dropout(mlp_out)
        mlp_out = self.mlp1(mlp_out)

        x = self.layernorm2(x + mlp_out)
        assert x.shape == (batch, time, d_latent)
        return x

    def _sattn(self, x: Tensor, attn_mask: Tensor | None) -> Tensor:
        """Multi-head self-attention."""
        config = self.config
        q, k, v = rearrange(
            self.sattn_qkv_proj(x),
            "b t (qkv nh dh) -> qkv b nh t dh",
            qkv=3,
            nh=config.n_heads,
        )
        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q, seq_dim=-2)
            k = self.rotary_emb.rotate_queries_or_keys(k, seq_dim=-2)
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=config.dropout_p, attn_mask=attn_mask
        )
        x = self.dropout(x)
        x = rearrange(x, "b nh t dh -> b t (nh dh)", nh=config.n_heads)
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=config.dropout_p
        )
        x = self.dropout(x)
        x = rearrange(x, "b nh t dh -> b t (nh dh)", nh=config.n_heads)
        x = self.sattn_out_proj(x)
        return x

    def _xattn(self, x: Tensor, attn_mask: Tensor | None, cond: Tensor) -> Tensor:
        """Multi-head cross-attention."""
        config = self.config
        k, v = rearrange(
            self.xattn_kv_proj(
                {
                    "kv_from_cond_q_from_x": cond,
                    "kv_from_x_q_from_cond": x,
                }[self.config.xattn_mode]
            ),
            "b t (qk nh dh) -> qk b nh t dh",
            qk=2,
            nh=config.n_heads,
        )
        q = rearrange(
            self.xattn_q_proj(
                {
                    "kv_from_cond_q_from_x": x,
                    "kv_from_x_q_from_cond": cond,
                }[self.config.xattn_mode]
            ),
            "b t (nh dh) -> b nh t dh",
            nh=config.n_heads,
        )
        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q, seq_dim=-2)
            k = self.rotary_emb.rotate_queries_or_keys(k, seq_dim=-2)
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=config.dropout_p, attn_mask=attn_mask
        )
        x = rearrange(x, "b nh t dh -> b t (nh dh)")
        x = self.xattn_out_proj(x)

        return x


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module
