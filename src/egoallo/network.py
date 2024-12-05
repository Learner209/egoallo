from __future__ import annotations

from dataclasses import dataclass
from functools import cache, cached_property
from math import ceil
from typing import Literal, assert_never, Optional

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Bool, Float
from loguru import logger
from rotary_embedding_torch import RotaryEmbedding
from torch import Tensor, nn

from .fncsmpl import SmplhModel, SmplhShapedAndPosed
from .tensor_dataclass import TensorDataclass
from .transforms import SE3, SO3

from egoallo.config import make_cfg, CONFIG_FILE
local_config_file = CONFIG_FILE
CFG = make_cfg(config_name="defaults", config_file=local_config_file, cli_args=[])

from egoallo.utils.setup_logger import setup_logger

logger = setup_logger(output=None, name=__name__)


def project_rotmats_via_svd(
    rotmats: Float[Tensor, "*batch 3 3"],
) -> Float[Tensor, "*batch 3 3"]:
    u, s, vh = torch.linalg.svd(rotmats)
    del s
    return torch.einsum("...ij,...jk->...ik", u, vh)


class EgoDenoiseTraj(TensorDataclass):
    """Data structure for denoising with MAE-style masking."""

    betas: Float[Tensor, "*#batch timesteps 16"]
    """Body shape parameters. We don't really need the timesteps axis here,
    it's just for convenience."""

    body_rotmats: Float[Tensor, "*#batch timesteps 21 3 3"]
    """Local orientations for each body joint."""

    contacts: Float[Tensor, "*#batch timesteps 21"]
    """Contact boolean for each joint."""

    hand_rotmats: Float[Tensor, "*#batch timesteps 30 3 3"] | None
    """Local orientations for each body joint."""

    R_world_root: Float[Tensor, "*#batch timesteps 3 3"]
    """Global rotation matrix of the root joint."""

    t_world_root: Float[Tensor, "*#batch timesteps 3"]
    """Global translation vector of the root joint."""

    @staticmethod
    def get_packed_dim(include_hands: bool) -> int:
        """Get dimension of packed state vector.
        
        Args:
            include_hands: Whether to include hand rotations in packed dimension.
        
        Returns:
            Total dimension of packed state vector.
        """
        # 16 (betas) + 21*9 (body_rotmats) + 21 (contacts) + 9 (R_world_root) + 3 (t_world_root)
        packed_dim = 16 + 21 * 9 + 21 + 9 + 3
        if include_hands:
            packed_dim += 30 * 9  # hand_rotmats
        return packed_dim

    def apply_to_body(self, body_model: SmplhModel) -> SmplhShapedAndPosed:
        """Apply the trajectory data to a SMPL-H body model."""
        device = self.betas.device
        dtype = self.betas.dtype
        assert self.hand_rotmats is not None
        
        shaped = body_model.with_shape(self.betas)
        T_world_root = SE3.from_rotation_and_translation(
            SO3.from_matrix(self.R_world_root),
            self.t_world_root,
        ).parameters()
        posed = shaped.with_pose(
            T_world_root=T_world_root,
            local_quats=SO3.from_matrix(
                torch.cat([self.body_rotmats, self.hand_rotmats], dim=-3)
            ).wxyz,
        )
        return posed

    def pack(self) -> Float[Tensor, "*#batch timesteps d_state"]:
        """Pack trajectory into a single flattened vector."""
        (*batch, time, num_joints, _, _) = self.body_rotmats.shape
        assert num_joints == 21
        
        # Create list of tensors to pack
        tensors_to_pack = [
            self.betas.reshape((*batch, time, -1)),
            self.body_rotmats.reshape((*batch, time, -1)),
            self.contacts.reshape((*batch, time, -1)),
            self.R_world_root.reshape((*batch, time, -1)),
            self.t_world_root.reshape((*batch, time, -1)),
        ]
        
        if self.hand_rotmats is not None:
            tensors_to_pack.append(self.hand_rotmats.reshape((*batch, time, -1)))
        
        return torch.cat(tensors_to_pack, dim=-1)

    @classmethod
    def unpack(
        cls,
        x: Float[Tensor, "*#batch timesteps d_state"],
        include_hands: bool,
        project_rotmats: bool = False,
    ) -> EgoDenoiseTraj:
        """Unpack trajectory from a single flattened vector."""
        (*batch, time, d_state) = x.shape
        assert d_state == cls.get_packed_dim(include_hands)

        if include_hands:
            betas, body_rotmats_flat, contacts, R_world_root, t_world_root, hand_rotmats_flat = torch.split(
                x, [16, 21 * 9, 21, 9, 3, 30 * 9], dim=-1
            )
            body_rotmats = body_rotmats_flat.reshape((*batch, time, 21, 3, 3))
            hand_rotmats = hand_rotmats_flat.reshape((*batch, time, 30, 3, 3))
        else:
            betas, body_rotmats_flat, contacts, R_world_root, t_world_root = torch.split(
                x, [16, 21 * 9, 21, 9, 3], dim=-1
            )
            body_rotmats = body_rotmats_flat.reshape((*batch, time, 21, 3, 3))
            hand_rotmats = None

        if project_rotmats:
            body_rotmats = project_rotmats_via_svd(body_rotmats)
            if hand_rotmats is not None:
                hand_rotmats = project_rotmats_via_svd(hand_rotmats)

        R_world_root = R_world_root.reshape(*batch, time, 3, 3)
        return cls(
            betas=betas,
            body_rotmats=body_rotmats,
            contacts=contacts,
            hand_rotmats=hand_rotmats,
            R_world_root=R_world_root,
            t_world_root=t_world_root,
        )


@dataclass
class EgoDenoiserConfig:
    # Basic parameters
    max_t: int = 1000
    fourier_enc_freqs: int = 3
    d_latent: int = 512
    d_feedforward: int = 2048
    d_noise_emb: int = 1024
    num_heads: int = 4
    encoder_layers: int = 6
    decoder_layers: int = 6
    dropout_p: float = 0.0
    
    # MAE parameters
    mask_ratio: float = 0.75  # Ratio of joints to mask during training
    include_hands: bool = True
    
    # Model settings
    activation: Literal["gelu", "relu"] = "gelu"
    positional_encoding: Literal["transformer", "rope"] = "rope"
    noise_conditioning: Literal["token", "film"] = "token"
    xattn_mode: Literal["kv_from_cond_q_from_x", "kv_from_x_q_from_cond"] = (
        "kv_from_cond_q_from_x"
    )

    # Joint position conditioning settings
    joint_cond_mode: Literal["absolute", "absrel_jnts", "absrel", "absrel_global_deltas"] = "absrel"

    @cached_property
    def d_cond(self) -> int:
        """Dimensionality of conditioning vector."""
        # Calculate number of visible joints based on mask ratio
        num_joints = CFG.smplh.num_joints
        num_visible_joints = ceil((1 - self.mask_ratio) * num_joints)
        # Base dimension for joint position (3) + joint index embedding (16)
        d_per_joint = 3 + 16

        if self.joint_cond_mode == "absolute":
            d_cond = d_per_joint * num_visible_joints + 1  # +1 for floor height
        elif self.joint_cond_mode == "absrel_jnts":
            d_cond = d_per_joint * num_visible_joints + 1  # +1 for floor height
        elif self.joint_cond_mode == "absrel":
            d_cond = (d_per_joint * 2 - 16) * num_visible_joints + 1  # +1 for floor height
        elif self.joint_cond_mode == "absrel_global_deltas":
            d_cond = (d_per_joint * num_visible_joints) + 9 + 3 + 1  # +1 for floor height
        else:
            assert_never(self.joint_cond_mode)
            
        d_cond = d_cond + d_cond * self.fourier_enc_freqs * 2  # Fourier encoding
        return d_cond

    def make_cond_with_masked_joints(
        self,
        joints: Float[Tensor, "batch time num_joints 3"],
        visible_joints_mask: Bool[Tensor, "batch time 21"],
    ) -> Float[Tensor, "batch time d_cond"]:
        """Construct conditioning using all joints (zeroing invisible ones) and floor height.
        
        Args:
            joints: Joint positions of shape (batch, time, num_joints, 3)
            visible_joints_mask: Boolean mask of shape (batch, time, 21) indicating joint visibility
        
        Returns:
            Conditioning tensor of shape (batch, time, d_cond)
        """
        batch, time = visible_joints_mask.shape[:2]
        device = joints.device
        dtype = joints.dtype

        # Create joint index embeddings for all joints
        joint_embeddings = nn.Embedding(CFG.smplh.num_joints, 16).to(device)
        all_indices = torch.arange(21, device=device)
        index_embeddings = joint_embeddings(all_indices).expand(batch, time, -1, -1)  # (batch, time, 21, 16)
        
        # Zero out invisible joints and their embeddings
        mask_expanded = visible_joints_mask.unsqueeze(-1)  # (batch, time, 21, 1)
        masked_joints = joints * mask_expanded
        masked_embeddings = index_embeddings * mask_expanded
        
        # Extract floor height from visible joints
        visible_heights = joints[..., 2] * visible_joints_mask  # Zero out invisible joints
        floor_height = visible_heights.max(dim=2, keepdim=True)[0]  # (batch, time, 1)
        
        if self.joint_cond_mode == "absolute":
            cond = torch.cat([
                masked_joints.reshape(batch, time, -1),  # Joint positions
                masked_embeddings.reshape(batch, time, -1),  # Joint embeddings
                floor_height,  # Floor height
            ], dim=-1)
                
        elif self.joint_cond_mode == "absrel_jnts":
            first_joint = masked_joints[..., 0:1, :]
            other_joints = masked_joints[..., 1:, :]
            local_coords = other_joints - first_joint
            
            cond = torch.cat([
                first_joint.reshape(batch, time, -1),
                local_coords.reshape(batch, time, -1),
                masked_embeddings.reshape(batch, time, -1),
                floor_height,
            ], dim=-1)

        elif self.joint_cond_mode == "absrel":
            abs_pos = masked_joints
            rel_pos = torch.zeros_like(masked_joints)
            rel_pos[:, 1:] = masked_joints[:, 1:] - masked_joints[:, :-1]
            
            cond = torch.cat([
                abs_pos.reshape(batch, time, -1),
                rel_pos.reshape(batch, time, -1),
                masked_embeddings.reshape(batch, time, -1),
                floor_height,
            ], dim=-1)
                
        elif self.joint_cond_mode == "absrel_global_deltas":
            first_frame = masked_joints[:, 0]  # (batch, joints, 3)
            current_frames = masked_joints  # (batch, time, joints, 3)
            
            # Calculate centroids using only visible joints
            mask_sum = visible_joints_mask.sum(dim=-1, keepdim=True).unsqueeze(-1)  # (batch, time, 1, 1)
            mask_sum = torch.clamp(mask_sum, min=1.0)  # Avoid division by zero
            
            first_centroid = (first_frame * visible_joints_mask[:, 0:1]).sum(dim=1, keepdim=True) / mask_sum[:, 0:1]
            current_centroids = (current_frames * mask_expanded).sum(dim=2, keepdim=True) / mask_sum
            
            first_centered = first_frame - first_centroid
            current_centered = current_frames - current_centroids
            
            first_centered_exp = first_centered.unsqueeze(1).expand(-1, time, -1, -1)
            H = first_centered_exp.transpose(-2, -1) @ current_centered
            
            U, _, Vh = torch.linalg.svd(H)
            r_mat = Vh.transpose(-2, -1) @ U.transpose(-2, -1)
            
            det = torch.linalg.det(r_mat)
            reflection_fix = torch.eye(3, device=r_mat.device).unsqueeze(0).unsqueeze(0).expand(batch, time, -1, -1).clone()
            reflection_fix[..., 2, 2] = det
            r_mat = r_mat @ reflection_fix
            
            t = current_centroids.squeeze(2) - torch.einsum('btij,bkj->bti', r_mat, first_centroid)
            
            cond = torch.cat([
                torch.cat([
                    masked_joints,
                    masked_embeddings
                ], dim=-1).reshape(batch, time, -1),
                r_mat.reshape(batch, time, 9),
                t.reshape(batch, time, 3),
                floor_height,
            ], dim=-1)
        
        else:
            assert_never(self.joint_cond_mode)

        # Apply Fourier encoding
        cond = fourier_encode(cond, freqs=self.fourier_enc_freqs)
        assert cond.shape == (batch, time, self.d_cond)
        return cond

    def make_cond(
        self,
        visible_joints: Float[Tensor, "batch time num_visible_joints 3"],
        visible_joints_mask: Bool[Tensor, "batch time 21"],
    ) -> Float[Tensor, "batch time d_cond"]:
        """Construct conditioning using visible joints, their indices, and floor height."""
        batch, time = visible_joints_mask.shape[:2]
        device = visible_joints.device
        dtype = visible_joints.dtype

        # Get indices of visible joints
        visible_indices = visible_joints_mask.nonzero(as_tuple=True)[-1]
        
        # Create learnable joint index embeddings
        joint_embeddings = nn.Embedding(CFG.smplh.num_joints, 16).to(device)
        num_visible_per_frame = visible_joints.shape[2]
        index_embeddings = joint_embeddings(visible_indices).reshape(batch, time, num_visible_per_frame, 16)

        # Extract floor height from visible joints (assuming lowest joint represents floor contact)
        floor_height = visible_joints[..., :, 2].min(dim=2, keepdim=True)[0]  # shape: (batch, time, 1)

        if self.joint_cond_mode == "absolute":
            # Concatenate positions, index embeddings, and floor height
            cond = torch.cat([
                visible_joints.reshape(batch, time, -1),  # Joint positions
                index_embeddings.reshape(batch, time, -1),  # Joint embeddings
                floor_height,  # Floor height
            ], dim=-1)
            
        elif self.joint_cond_mode == "absrel_jnts":
            first_visible_jnt = visible_joints[..., 0:1, :]
            other_visible_jnts = visible_joints[..., 1:, :]
            local_coords = other_visible_jnts - first_visible_jnt
            
            cond = torch.cat([
                first_visible_jnt.reshape(batch, time, -1),
                local_coords.reshape(batch, time, -1),
                index_embeddings.reshape(batch, time, -1),
                floor_height,
            ], dim=-1)

        elif self.joint_cond_mode == "absrel":
            abs_pos = visible_joints
            rel_pos = torch.zeros_like(visible_joints)
            rel_pos[:, 1:] = visible_joints[:, 1:] - visible_joints[:, :-1]
            
            cond = torch.cat([
                abs_pos.reshape(batch, time, -1),
                rel_pos.reshape(batch, time, -1),
                index_embeddings.reshape(batch, time, -1),
                floor_height,
            ], dim=-1)
            
        elif self.joint_cond_mode == "absrel_global_deltas":
            first_frame = visible_joints[:, 0]
            current_frames = visible_joints
            
            first_centroid = first_frame.mean(dim=1, keepdim=True)
            current_centroids = current_frames.mean(dim=2, keepdim=True)
            
            first_centered = first_frame - first_centroid
            current_centered = current_frames - current_centroids
            
            first_centered_exp = first_centered.unsqueeze(1).expand(-1, time, -1, -1)
            H = first_centered_exp.transpose(-2, -1) @ current_centered
            
            U, _, Vh = torch.linalg.svd(H)
            r_mat = Vh.transpose(-2, -1) @ U.transpose(-2, -1)
            
            det = torch.linalg.det(r_mat)
            reflection_fix = torch.eye(3, device=r_mat.device).unsqueeze(0).unsqueeze(0).expand(batch, time, -1, -1).clone()
            reflection_fix[..., 2, 2] = det
            r_mat = r_mat @ reflection_fix
            
            t = current_centroids.squeeze(2) - torch.einsum('btij,bkj->bti', r_mat, first_centroid)
            
            cond = torch.cat([
                torch.cat([
                    visible_joints,
                    index_embeddings
                ], dim=-1).reshape(batch, time, -1),
                r_mat.reshape(batch, time, 9),
                t.reshape(batch, time, 3),
                floor_height,
            ], dim=-1)
        
        else:
            assert_never(self.joint_cond_mode)

        # Apply Fourier encoding
        cond = fourier_encode(cond, freqs=self.fourier_enc_freqs)
        assert cond.shape == (batch, time, self.d_cond)
        return cond


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
            "body_rotmats": 21 * 9,
            "contacts": 21,
            "R_world_root": 9,
            "t_world_root": 3,
        }
        if config.include_hands:
            modality_dims["hand_rotmats"] = 30 * 9

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
        self.latent_from_cond = nn.Linear(config.d_cond, config.d_latent)

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

    def get_d_state(self) -> int:
        return EgoDenoiseTraj.get_packed_dim(self.config.include_hands)

    def forward(
        self,
        x_t_packed: Float[Tensor, "batch time state_dim"],
        t: Float[Tensor, "batch"],
        *,
        project_output_rotmats: bool,
        visible_joints: Float[Tensor, "batch time num_visible_joints 3"],
        visible_joints_mask: Bool[Tensor, "batch time 21"],
        mask: Bool[Tensor, "batch time"] | None,
        cond_dropout_keep_mask: Bool[Tensor, "batch"] | None = None,
    ) -> Float[Tensor, "batch time state_dim"]:
        """Forward pass with MAE-style masking."""
        config = self.config

        x_t = EgoDenoiseTraj.unpack(x_t_packed, include_hands=self.config.include_hands)
        (batch, time, num_body_joints, _, _) = x_t.body_rotmats.shape
        assert num_body_joints == 21

        # Encode the trajectory into a single vector per timestep.
        x_t_encoded = (
            self.encoders["betas"](x_t.betas.reshape((batch, time, -1)))
            + self.encoders["body_rotmats"](x_t.body_rotmats.reshape((batch, time, -1)))
            + self.encoders["contacts"](x_t.contacts)
            + self.encoders["R_world_root"](x_t.R_world_root.reshape((batch, time, -1)))
            + self.encoders["t_world_root"](x_t.t_world_root.reshape((batch, time, -1)))
        )
        if self.config.include_hands:
            assert x_t.hand_rotmats is not None
            x_t_encoded = x_t_encoded + self.encoders["hand_rotmats"](
                x_t.hand_rotmats.reshape((batch, time, -1))
            )
        assert x_t_encoded.shape == (batch, time, config.d_latent)

        # Embed the diffusion noise level.
        assert t.shape == (batch,)
        noise_emb = self.noise_emb(t - 1)
        assert noise_emb.shape == (batch, config.d_noise_emb)

        # Create conditioning from visible joints only
        cond = config.make_cond(
            visible_joints=visible_joints,
            visible_joints_mask=visible_joints_mask,
        )

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
            assert_never(config.positional_encoding)

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
                # Project rotation matrices for body_rotmats via SVD,
                (
                    project_rotmats_via_svd(
                        modality_decoder(decoder_out).reshape((-1, 3, 3))
                    ).reshape(
                        (batch, time, {"body_rotmats": 21, "hand_rotmats": 30}[key] * 9)
                    )
                    # if enabled,
                    if project_output_rotmats
                    and key in ("body_rotmats", "hand_rotmats")
                    # otherwise, just decode normally.
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
    coeffs = 2.0 ** torch.arange(freqs, device=x.device) # shape: (freqs,)
    scaled = (x[..., None] * coeffs).reshape((*batch_axes, x_dim * freqs)) # shape: (*batch_axes, x_dim * freqs)
    return torch.cat(
        [
            x,
            torch.sin(torch.cat([scaled, scaled + torch.pi / 2.0], dim=-1)),
        ],
        dim=-1,
    ) # shape: (*batch_axes, x_dim + 2 * freqs * x_dim)


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
            RotaryEmbedding(
                config.d_latent // config.n_heads,
                learned_freq=False,
            )
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
