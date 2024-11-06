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
        should_project_rot6d: bool = False,
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

        if should_project_rot6d:
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
