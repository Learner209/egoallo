"""Network definitions."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import cache
from functools import cached_property
from pathlib import Path
from typing import Any, Type
from typing import assert_never
from typing import Dict
from typing import Literal
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from egoallo.config import CONFIG_FILE
from egoallo.config import make_cfg
from einops import rearrange
from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import jaxtyped
from rotary_embedding_torch import RotaryEmbedding
from torch import nn
from torch import Tensor
import typeguard

from egoallo.type_stubs import DenoiseTrajTypeLiteral
from egoallo.denoising.base_traj import BaseDenoiseTraj

# Move type imports inside TYPE_CHECKING block to avoid circular imports
if TYPE_CHECKING:
    from egoallo.type_stubs import DenoiseTrajType, EgoTrainingDataType, JointCondMode

from egoallo.utils.setup_logger import setup_logger
from egoallo.constants import SmplFamilyMetaModelZoo, SmplFamilyMetaModelName

from egoallo.denoising import (
    AbsoluteDenoiseTraj,
    JointsOnlyTraj,
    VelocityDenoiseTraj,
    AbsoluteDenoiseTrajAADecomp,
)


local_config_file = CONFIG_FILE
CFG = make_cfg(config_name="defaults", config_file=local_config_file, cli_args=[])

logger = setup_logger(output=None, name=__name__)


def get_kinematic_chain(use_smplh: bool = True) -> list[tuple[int, int]]:
    """Get kinematic chain for SMPL-H model based on SMPL_JOINT_NAMES."""
    # Define parent-child relationships for joints based on SMPL_JOINT_NAMES order:
    # ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    # 'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    # 'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    # 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand']
    # Each tuple is (child_idx, parent_idx)
    kinematic_chain = [
        (1, 0),  # left_hip -> pelvis
        (2, 0),  # right_hip -> pelvis
        (3, 0),  # spine1 -> pelvis
        (4, 1),  # left_knee -> left_hip
        (5, 2),  # right_knee -> right_hip
        (6, 3),  # spine2 -> spine1
        (7, 4),  # left_ankle -> left_knee
        (8, 5),  # right_ankle -> right_knee
        (9, 6),  # spine3 -> spine2
        (10, 7),  # left_foot -> left_ankle
        (11, 8),  # right_foot -> right_ankle
        (12, 9),  # neck -> spine3
        (13, 12),  # left_collar -> neck
        (14, 12),  # right_collar -> neck
        (15, 12),  # head -> neck
        (16, 13),  # left_shoulder -> left_collar
        (17, 14),  # right_shoulder -> right_collar
        (18, 16),  # left_elbow -> left_shoulder
        (19, 17),  # right_elbow -> right_shoulder
        (20, 18),  # left_wrist -> left_elbow
        (21, 19),  # right_wrist -> right_elbow
        (22, 20),  # left_hand -> left_wrist
        (23, 21),  # right_hand -> right_wrist
    ]
    if use_smplh:
        return kinematic_chain
    else:
        return kinematic_chain[:-2]


T = TypeVar("T", bound="BaseDenoiseTraj")


@dataclasses.dataclass
class DenoisingConfig:
    """Configuration for denoising.

    This class handles configuration for both absolute and velocity-based denoising,
    and provides factory methods for creating appropriate trajectory objects.
    """

    temporal_window: int = 2
    use_acceleration: bool = False
    loss_weights: dict[str, float] | None = None
    joint_cond_mode: "JointCondMode" = "absrel"
    include_hands: bool = False

    denoising_mode: "DenoiseTrajTypeLiteral" = "AbsoluteDenoiseTraj"
    DenoiseTrajTypeMetaDict: Dict["DenoiseTrajTypeLiteral", Type["DenoiseTrajType"]] = (
        dataclasses.field(
            default_factory=lambda: {
                "AbsoluteDenoiseTraj": AbsoluteDenoiseTraj,
                "JointsOnlyTraj": JointsOnlyTraj,
                "VelocityDenoiseTraj": VelocityDenoiseTraj,
                "AbsoluteDenoiseTrajAADecomp": AbsoluteDenoiseTrajAADecomp,
            },
        )
    )

    def __post_init__(self):
        if self.loss_weights is None:
            self.loss_weights = self.DenoiseTrajTypeMetaDict[
                self.denoising_mode
            ].loss_weights

    def is_velocity_mode(self) -> bool:
        """Check if we're using velocity-based denoising."""
        return self.denoising_mode in ("VelocityDenoiseTraj")

    def get_d_state(self) -> int:
        """Get the dimension of the state vector."""
        return self.DenoiseTrajTypeMetaDict[self.denoising_mode].get_packed_dim(
            include_hands=self.include_hands,
        )

    @jaxtyped(typechecker=typeguard.typechecked)
    def from_ego_data(
        self,
        ego_data: "EgoTrainingDataType",
        smpl_family_model_basedir: Path,
        include_hands: bool = True,
    ) -> "DenoiseTrajType":
        """Convert EgoTrainingData instance to appropriate DenoiseTraj based on config.

        This method serves as a wrapper that calls the appropriate conversion method
        on the ego_data instance. This design allows for more flexible and scalable
        handling of different EgoTrainingDataType classes and DenoiseTrajType classes.

        Args:
            ego_data: Input EgoTrainingData instance
            include_hands: Whether to include hand data in the output trajectory

        Returns:
            Appropriate trajectory object based on denoising mode
        """
        # Verify data is in the correct processing stage
        assert ego_data.metadata.stage == "preprocessed", (
            "EgoTrainingData should be preprocessed before being used to create trajectories. \
            , The logic is traj should be sent to network so that the ego_data should be between pre and post."
        )

        # Call the to_denoise_traj method on the ego_data instance
        # This delegates the conversion logic to the appropriate EgoTrainingDataType class
        return ego_data.to_denoise_traj(
            denoising_mode=self.denoising_mode,
            include_hands=include_hands,
            smpl_family_model_basedir=smpl_family_model_basedir,
        )

    def unpack_traj(
        self,
        x: Float[Tensor, "*batch timesteps d_state"],
        include_hands: bool = False,
        project_rotmats: bool = False,
    ) -> "BaseDenoiseTraj[Any]":
        """Unpack trajectory data using appropriate trajectory class based on configuration."""
        return self.DenoiseTrajTypeMetaDict[self.denoising_mode].unpack(
            x,
            include_hands=include_hands,
            project_rotmats=project_rotmats,
        )

    def fetch_modality_dict(self, include_hands: bool = False) -> dict[str, int]:
        # Use the Strategy Pattern: delegate to the appropriate trajectory class
        return self.DenoiseTrajTypeMetaDict[self.denoising_mode].get_modality_dict(
            include_hands=include_hands,
        )


@jaxtyped(typechecker=typeguard.typechecked)
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
    include_hands: bool = False

    # Model settings
    activation: Literal["gelu", "relu"] = "gelu"
    positional_encoding: Literal["transformer", "rope"] = "rope"
    noise_conditioning: Literal["token", "film"] = "token"
    xattn_mode: Literal["kv_from_cond_q_from_x", "kv_from_x_q_from_cond"] = (
        "kv_from_cond_q_from_x"
    )

    # Joint position conditioning settings
    joint_cond_mode: "JointCondMode" = "absrel"
    joint_emb_dim: int = 8

    use_fourier_in_masked_joints: bool = (
        True  # Whether to apply Fourier encoding in make_cond_with_masked_joints
    )

    use_joint_embeddings: bool = (
        True  # Whether to use joint index embeddings in conditioning
    )

    # batch size and sequence length, used only for initializing "SmplFamilyModelType".
    batch_size: int = 64
    seq_length: int = 128

    smpl_family_model_basedir: Union[str, Path] = "assets/smpl_based_model/"

    def __post_init__(self):
        if isinstance(self.smpl_family_model_basedir, str):
            self.smpl_family_model_basedir = Path(self.smpl_family_model_basedir)

    @cached_property
    def d_cond(self) -> int:
        """Dimensionality of conditioning vector."""
        num_joints = CFG.smplh.num_joints  # Assuming num_joints is 22
        spatial_dim = 4  # x, y, z, visibility

        # Only include joint embedding dimensions if enabled
        joint_emb_contribution = (
            (self.joint_emb_dim * num_joints) if self.use_joint_embeddings else 0
        )

        if self.joint_cond_mode == "vel_acc":
            # velocities (22*4) + accelerations (22*4) + index_embeddings (22*16 if enabled)
            d_cond = (spatial_dim * num_joints * 2) + joint_emb_contribution
        elif self.joint_cond_mode == "vel_acc_plus":
            # velocities (22*4) + accelerations (22*4) + index_embeddings (22*16 if enabled)
            d_cond = (spatial_dim * num_joints * 2) + joint_emb_contribution
        elif self.joint_cond_mode == "absolute":
            # joints_with_vis (22*4) + index_embeddings (22*16 if enabled)
            d_cond = (spatial_dim * num_joints) + joint_emb_contribution
        elif self.joint_cond_mode == "absrel_jnts":
            # first_joint (4) + local_coords (21*4) + index_embeddings (22*16 if enabled)
            d_cond = (
                spatial_dim + (spatial_dim * (num_joints - 1)) + joint_emb_contribution
            )
        elif self.joint_cond_mode in ("absrel", "joints_only"):
            # abs_pos (22*4) + rel_pos (22*4) + index_embeddings (22*16 if enabled)
            d_cond = (
                (spatial_dim * num_joints)
                + (spatial_dim * num_joints)
                + joint_emb_contribution
            )
        elif self.joint_cond_mode == "absrel_global_deltas":
            # joints_with_vis (22*spatial_dim) + index_embeddings (22*16 if enabled) + r_mat (9) + t (3)
            d_cond = (spatial_dim * num_joints) + joint_emb_contribution + 9 + 3
        else:
            assert_never(self.joint_cond_mode)

        # Apply Fourier encoding multiplier if enabled
        if self.use_fourier_in_masked_joints:
            d_cond = d_cond * (2 * self.fourier_enc_freqs + 1)

        return d_cond

    # @jaxtyped(typechecker=typeguard.typechecked)
    def make_cond_with_masked_joints(
        self,
        joints: Float[Tensor, "batch time 22 3"],
        visible_joints_mask: Bool[Tensor, "batch time 22"],
    ) -> Float[Tensor, "batch time d_cond"]:
        batch, time = visible_joints_mask.shape[:2]
        device = joints.device
        dtype = joints.dtype

        # !joints must be masked to prevent further motion information from being used
        masked_joints = joints.clone()

        # Create joint embeddings if enabled
        if self.use_joint_embeddings:
            joint_embeddings = nn.Embedding(
                CFG.smplh.num_joints,
                self.joint_emb_dim,
            ).to(device)
            all_indices = torch.arange(CFG.smplh.num_joints, device=device)
            index_embeddings = joint_embeddings(all_indices).expand(batch, time, -1, -1)
        else:
            index_embeddings = None

        if self.joint_cond_mode == "vel_acc_plus":
            # TODO: this should be deprectaed now, as it lacks logical soundness now.
            # 1. Compute hierarchical joint representation
            # Root (pelvis) as base
            root_pos = masked_joints[..., 0:1, :]

            # Get relative positions to parent joints using kinematic chain
            # This removes redundant global motion information
            rel_to_parent = torch.zeros_like(masked_joints)
            kinematic_chain = get_kinematic_chain(
                use_smplh=True,
            )  # Define this based on SMPL skeleton
            for joint_idx, parent_idx in kinematic_chain:
                rel_to_parent[..., joint_idx, :] = (
                    masked_joints[..., joint_idx, :] - masked_joints[..., parent_idx, :]
                )

            # Define the rel_to_parent mask
            rel_to_parent_visible_mask: Float[Tensor, "batch time 22"] = (
                torch.zeros_like(visible_joints_mask)
            )
            # Handle pelvis (root) joint separately
            rel_to_parent_visible_mask[..., 0] = visible_joints_mask[..., 0]
            # For all other joints, require both joint and parent to be visible
            for joint_idx, parent_idx in kinematic_chain:
                rel_to_parent_visible_mask[..., joint_idx] = (
                    visible_joints_mask[..., joint_idx]
                    & visible_joints_mask[..., parent_idx]
                )

            # 2. Compute motion features and masks using relative parent joint positions
            # Calculate velocity mask (both current and previous frame must have valid relative parent positions)
            vel_mask = torch.zeros_like(rel_to_parent_visible_mask)  # [batch, time, 22]
            vel_mask[:, 1:] = (
                rel_to_parent_visible_mask[:, 1:] & rel_to_parent_visible_mask[:, :-1]
            )

            # Calculate acceleration mask (current and two previous frames must have valid relative parent positions)
            acc_mask = torch.zeros_like(rel_to_parent_visible_mask)  # [batch, time, 22]
            acc_mask[:, 2:] = (
                rel_to_parent_visible_mask[:, 2:]
                & rel_to_parent_visible_mask[:, 1:-1]
                & rel_to_parent_visible_mask[:, :-2]
            )

            # Velocity (first order temporal difference of relative parent positions)
            velocity = torch.zeros_like(rel_to_parent)  # [batch, time, 22, 3]
            velocity[:, 1:] = rel_to_parent[:, 1:] - rel_to_parent[:, :-1]
            velocity = velocity * vel_mask.unsqueeze(-1)  # [batch, time, 22, 3]

            # Acceleration (second order temporal difference of relative parent positions)
            accel = torch.zeros_like(rel_to_parent)  # [batch, time, 22, 3]
            accel[:, 2:] = velocity[:, 2:] - velocity[:, 1:-1]
            accel = accel * acc_mask.unsqueeze(-1)  # [batch, time, 22, 3]

            # 3. Compute motion attention scores based on relative motion
            motion_magnitude = torch.norm(velocity, dim=-1)  # [batch, time-1, 22]
            attention_scores = torch.sigmoid(
                motion_magnitude - motion_magnitude.mean(dim=1, keepdim=True),
            )  # [batch, time-1, 22]

            # 4. Create decorrelated features
            components = [
                # Global root motion and its configuration
                root_pos.reshape(batch, time, -1),
                # Local joint configurations and visibility
                rel_to_parent.reshape(batch, time, -1),
                rel_to_parent_visible_mask.reshape(batch, time, -1),
                # Motion features with attention weights
                (velocity * attention_scores[..., None]).reshape(batch, time, -1),
                (accel * attention_scores[..., None]).reshape(batch, time, -1),
                # Motion feature masks
                vel_mask.reshape(batch, time, -1),
                acc_mask.reshape(batch, time, -1),
                # Optional semantic information
                index_embeddings.reshape(batch, time, -1)
                if self.use_joint_embeddings
                else None,
            ]
            components = [c for c in components if c is not None]

            # Pad temporal features to match time dimension
            pad_size = time - components[2].shape[1]
            components[2] = F.pad(components[2], (0, 0, 0, pad_size))
            pad_size = time - components[3].shape[1]
            components[3] = F.pad(components[3], (0, 0, 0, pad_size))

            cond = torch.cat(components, dim=-1)

        elif self.joint_cond_mode == "vel_acc":
            # Calculate velocity mask (both current and previous frame must be visible)
            vel_mask = torch.zeros_like(visible_joints_mask)
            vel_mask[:, 1:] = visible_joints_mask[:, 1:] & visible_joints_mask[:, :-1]

            # Calculate acceleration mask (current and two previous frames must be visible)
            acc_mask = torch.zeros_like(visible_joints_mask)
            acc_mask[:, 2:] = (
                visible_joints_mask[:, 2:]
                & visible_joints_mask[:, 1:-1]
                & visible_joints_mask[:, :-2]
            )

            # Calculate velocities and mask invisible parts
            velocities = torch.zeros_like(joints)
            velocities[:, 1:] = joints[:, 1:] - joints[:, :-1]
            velocities = velocities * vel_mask.unsqueeze(
                -1,
            )  # Mask out invisible velocities

            # Calculate accelerations and mask invisible parts
            accelerations = torch.zeros_like(joints)
            accelerations[:, 2:] = velocities[:, 2:] - velocities[:, 1:-1]
            accelerations = accelerations * acc_mask.unsqueeze(
                -1,
            )  # Mask out invisible accelerations

            # Combine with visibility masks
            velocities_with_vis = torch.cat(
                [velocities, vel_mask.unsqueeze(-1).to(dtype)],
                dim=-1,
            )

            accelerations_with_vis = torch.cat(
                [accelerations, acc_mask.unsqueeze(-1).to(dtype)],
                dim=-1,
            )

            # Combine all components
            components = [
                velocities_with_vis.reshape(batch, time, -1),
                accelerations_with_vis.reshape(batch, time, -1),
            ]
            if self.use_joint_embeddings:
                components.append(index_embeddings.reshape(batch, time, -1))

            cond = torch.cat(components, dim=-1)

        elif self.joint_cond_mode == "absolute":
            # joints_with_vis (22*4) + index_embeddings (22*16 if enabled)
            joints_with_vis = torch.cat(
                [masked_joints, visible_joints_mask.unsqueeze(-1).to(dtype)],
                dim=-1,
            )
            components = [joints_with_vis.reshape(batch, time, -1)]
            if self.use_joint_embeddings:
                components.append(index_embeddings.reshape(batch, time, -1))
            cond = torch.cat(components, dim=-1)

        elif self.joint_cond_mode == "absrel_jnts":
            joints_with_vis = torch.cat(
                [masked_joints, visible_joints_mask.unsqueeze(-1).to(dtype)],
                dim=-1,
            )
            first_joint = joints_with_vis[..., 0:1, :]
            other_joints = joints_with_vis[..., 1:, :]
            local_coords = other_joints - first_joint[..., :3].expand_as(
                other_joints[..., :3],
            )
            local_coords = torch.cat(
                [
                    local_coords[..., :3],
                    other_joints[..., 3:],
                ],
                dim=-1,
            )

            components = [
                first_joint.reshape(batch, time, -1),
                local_coords.reshape(batch, time, -1),
            ]
            if self.use_joint_embeddings:
                components.append(index_embeddings.reshape(batch, time, -1))
            cond = torch.cat(components, dim=-1)

        elif self.joint_cond_mode in ("absrel", "joints_only"):
            joints_with_vis = torch.cat(
                [masked_joints, visible_joints_mask.unsqueeze(-1).to(dtype)],
                dim=-1,
            )
            abs_pos = joints_with_vis
            rel_pos = torch.zeros_like(joints_with_vis)
            rel_pos[:, 1:] = joints_with_vis[:, 1:] - joints_with_vis[:, :-1]

            components = [
                abs_pos.reshape(batch, time, -1),
                rel_pos.reshape(batch, time, -1),
            ]
            if self.use_joint_embeddings:
                components.append(index_embeddings.reshape(batch, time, -1))
            cond = torch.cat(components, dim=-1)

        elif self.joint_cond_mode == "absrel_global_deltas":
            joints_with_vis = torch.cat(
                [masked_joints, visible_joints_mask.unsqueeze(-1).to(dtype)],
                dim=-1,
            )
            first_frame = joints_with_vis[:, 0]
            current_frames = joints_with_vis

            first_centroid = first_frame.mean(dim=1, keepdim=True)
            current_centroids = current_frames.mean(dim=2, keepdim=True)

            first_centered = first_frame - first_centroid
            current_centered = current_frames - current_centroids

            first_centered_exp = first_centered.unsqueeze(1).expand(-1, time, -1, -1)
            H = first_centered_exp.transpose(-2, -1) @ current_centered

            U, _, Vh = torch.linalg.svd(H)
            r_mat = Vh.transpose(-2, -1) @ U.transpose(-2, -1)

            det = torch.linalg.det(r_mat)
            reflection_fix = (
                torch.eye(3, device=r_mat.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(batch, time, -1, -1)
                .clone()
            )
            reflection_fix[..., 2, 2] = det
            r_mat = r_mat @ reflection_fix

            t = current_centroids.squeeze(2) - torch.einsum(
                "btij,bkj->bti",
                r_mat,
                first_centroid,
            )

            components = [
                torch.cat([joints_with_vis, index_embeddings], dim=-1).reshape(
                    batch,
                    time,
                    -1,
                ),
                r_mat.reshape(batch, time, 9),
                t.reshape(batch, time, 3),
            ]
            if self.use_joint_embeddings:
                components.append(index_embeddings.reshape(batch, time, -1))
            cond = torch.cat(components, dim=-1)

        else:
            assert_never(self.joint_cond_mode)

        # Apply Fourier encoding if enabled
        if self.use_fourier_in_masked_joints:
            cond = fourier_encode(cond, freqs=self.fourier_enc_freqs)

        assert cond.shape == (batch, time, self.d_cond)
        return cond


class EgoDenoiser(nn.Module):
    """Denoising network for human motion.

    Inputs are noisy trajectory, conditioning information, and timestep.
    Output is denoised trajectory.
    """

    def __init__(self, config: EgoDenoiserConfig, modality_dims: dict[str, int]):
        super().__init__()

        self.config = config

        self.body_model = SmplFamilyMetaModelZoo[SmplFamilyMetaModelName].load(
            config.smpl_family_model_basedir,
        )

        Activation = {"gelu": nn.GELU, "relu": nn.ReLU}[config.activation]

        self.encoders = nn.ModuleDict(
            {
                k: nn.Sequential(
                    nn.Linear(modality_dim, config.d_latent // 2),
                    nn.LayerNorm(config.d_latent // 2),
                    Activation(),
                    nn.Dropout(p=config.dropout_p),
                    nn.Linear(config.d_latent // 2, config.d_latent),
                    nn.LayerNorm(config.d_latent),
                    Activation(),
                    nn.Dropout(p=config.dropout_p),
                    nn.Linear(config.d_latent, config.d_latent),
                )
                for k, modality_dim in modality_dims.items()
            },
        )

        self.decoders = nn.ModuleDict(
            {
                k: nn.Sequential(
                    nn.Linear(config.d_latent, config.d_latent),
                    nn.LayerNorm(config.d_latent),
                    Activation(),
                    nn.Dropout(p=config.dropout_p),
                    nn.Linear(config.d_latent, config.d_latent // 2),
                    nn.LayerNorm(config.d_latent // 2),
                    Activation(),
                    nn.Dropout(p=config.dropout_p),
                    nn.Linear(config.d_latent // 2, modality_dim),
                )
                for k, modality_dim in modality_dims.items()
            },
        )

        # Helpers for converting between input dimensionality and latent dimensionality.
        self.latent_from_cond = nn.Sequential(
            nn.Linear(config.d_cond, config.d_latent),
            nn.LayerNorm(config.d_latent),
            nn.ReLU(),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=config.d_latent,
                    nhead=8,
                    dim_feedforward=config.d_latent * 4,
                    dropout=0.1,
                ),
                num_layers=2,
            ),
        )

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
                    ),
                )
                for _ in range(config.encoder_layers)
            ],
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
                    ),
                )
                for _ in range(config.decoder_layers)
            ],
        )

    @jaxtyped(typechecker=typeguard.typechecked)
    def forward(
        self,
        x_t_unpacked: Union[VelocityDenoiseTraj, AbsoluteDenoiseTraj, JointsOnlyTraj],
        t: Int[Tensor, "batch"],
        project_output_rotmats: bool,
        joints: Float[Tensor, "batch time num_joints 3"],
        visible_joints_mask: Bool[Tensor, "batch time 22"],
        mask: Bool[Tensor, "batch time"] | None,
        cond_dropout_keep_mask: Bool[Tensor, "batch"] | None = None,
    ) -> Float[Tensor, "batch time state_dim"]:
        """Forward pass with MAE-style masking."""
        config = self.config

        (batch, time, num_body_joints, _) = joints.shape
        assert num_body_joints == 22

        # Encode the trajectory into a single vector per timestep.
        x_t_encoded = x_t_unpacked.encode(self.encoders, batch, time)

        # Embed the diffusion noise level.
        assert t.shape == (batch,)
        noise_emb = self.noise_emb(t - 1)
        assert noise_emb.shape == (batch, config.d_noise_emb)

        cond = config.make_cond_with_masked_joints(
            joints=joints,
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
                decoder_out,
                attn_mask,
                noise_emb=noise_emb,
                cond=encoder_out,
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
                        modality_decoder(decoder_out).reshape((-1, 3, 3)),
                    ).reshape(
                        (
                            batch,
                            time,
                            {"body_rotmats": 21, "hand_rotmats": 30}[key] * 9,
                        ),
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

        # Return packed output.
        return packed_output


@jaxtyped(typechecker=typeguard.typechecked)
@cache
def make_positional_encoding(
    d_latent: int,
    length: int,
    dtype: torch.dtype,
) -> Float[Tensor, "length d_latent"]:
    """Computes standard Transformer positional encoding."""
    pe = torch.zeros(length, d_latent, dtype=dtype)
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_latent, 2).float() * (-np.log(10000.0) / d_latent),
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    assert pe.shape == (length, d_latent)
    return pe


@jaxtyped(typechecker=typeguard.typechecked)
def fourier_encode(
    x: Float[Tensor, "*batch channels"],
    freqs: int,
) -> Float[Tensor, "*batch channels_plus_2_mul_freqs_mul_channels"]:
    """Apply Fourier encoding to a tensor."""
    *batch_axes, x_dim = x.shape
    coeffs = 2.0 ** torch.arange(freqs, device=x.device)  # shape: (freqs,)
    scaled = (x[..., None] * coeffs).reshape(
        (*batch_axes, x_dim * freqs),
    )  # shape: (*batch_axes, x_dim * freqs)
    return torch.cat(
        [
            x,
            torch.sin(torch.cat([scaled, scaled + torch.pi / 2.0], dim=-1)),
        ],
        dim=-1,
    )  # shape: (*batch_axes, x_dim + 2 * freqs * x_dim)


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
            config.d_latent,
            config.d_latent * 3,
            bias=False,
        )
        self.sattn_out_proj = nn.Linear(config.d_latent, config.d_latent, bias=False)

        self.layernorm1 = nn.LayerNorm(config.d_latent)
        self.layernorm2 = nn.LayerNorm(config.d_latent)

        assert config.d_latent % config.n_heads == 0
        self.rotary_emb = (
            RotaryEmbedding(
                config.d_latent // config.n_heads,
                learned_freq=True,
            )
            if config.use_rope_embedding
            else None
        )

        if config.include_xattn:
            self.xattn_kv_proj = nn.Linear(
                config.d_latent,
                config.d_latent * 2,
                bias=False,
            )
            self.xattn_q_proj = nn.Linear(config.d_latent, config.d_latent, bias=False)
            self.xattn_layernorm = nn.LayerNorm(config.d_latent)
            self.xattn_out_proj = nn.Linear(
                config.d_latent,
                config.d_latent,
                bias=False,
            )

        self.norm_no_learnable = nn.LayerNorm(
            config.d_feedforward,
            elementwise_affine=False,
            bias=False,
        )
        self.activation = {"gelu": nn.GELU, "relu": nn.ReLU}[config.activation]()
        self.dropout = nn.Dropout(config.dropout_p)

        self.mlp0 = nn.Linear(config.d_latent, config.d_feedforward)
        self.mlp_film_cond_proj = (
            zero_module(
                nn.Linear(config.d_noise_emb, config.d_feedforward * 2, bias=False),
            )
            if config.use_film_noise_conditioning
            else None
        )
        self.mlp1 = nn.Linear(config.d_feedforward, config.d_latent)
        self.config = config

    @jaxtyped(typechecker=typeguard.typechecked)
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
                self.mlp_film_cond_proj(noise_emb),
                chunks=2,
                dim=-1,
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
            q,
            k,
            v,
            dropout_p=config.dropout_p,
            attn_mask=attn_mask,
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
                }[self.config.xattn_mode],
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
                }[self.config.xattn_mode],
            ),
            "b t (nh dh) -> b nh t dh",
            nh=config.n_heads,
        )
        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q, seq_dim=-2)
            k = self.rotary_emb.rotate_queries_or_keys(k, seq_dim=-2)
        x = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=config.dropout_p,
            attn_mask=attn_mask,
        )
        x = rearrange(x, "b nh t dh -> b t (nh dh)")
        x = self.xattn_out_proj(x)

        return x


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module
