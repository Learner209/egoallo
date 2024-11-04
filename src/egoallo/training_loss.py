"""Training loss configuration and computation."""
import dataclasses
from typing import Literal, NamedTuple

import torch
from torch import Tensor
import torch.utils.data
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel
from jaxtyping import Bool, Float, Int

from .data.amass import EgoTrainingData
from .fncsmpl import forward_kinematics
from .network import EgoDenoiseTraj
from .sampling import CosineNoiseScheduleConstants
from .transforms import SO3
from .motion_diffusion_pipeline import MotionUNet
from . import network

class MotionLosses(NamedTuple):
    """Container for all loss components."""
    noise_pred_loss: Tensor  # MSE between predicted and target noise
    betas_loss: Tensor      # Loss on SMPL shape parameters
    body_rot6d_loss: Tensor # Loss on body joint rotations
    contacts_loss: Tensor   # Loss on contact states
    hand_rot6d_loss: Tensor # Loss on hand joint rotations (if enabled)
    fk_loss: Tensor        # Forward kinematics loss
    foot_skating_loss: Tensor  # Foot skating prevention loss
    velocity_loss: Tensor   # Joint velocity smoothness loss
    total_loss: Tensor     # Weighted sum of all losses

@dataclasses.dataclass(frozen=True)
class TrainingLossConfig:
    """Configuration for training losses."""
    # Noise prediction loss weight
    noise_pred_weight: float = 1.0
    
    # Original loss weights
    cond_dropout_prob: float = 0.0
    beta_coeff_weights: tuple[float, ...] = tuple(1 / (i + 1) for i in range(16))
    loss_weights: dict[str, float] = dataclasses.field(
        default_factory=lambda: {
            "betas": 0.1,
            "body_rot6d": 1.0,
            "contacts": 0.1,
            "hand_rot6d": 0.01,
            # Geometric loss weights
            "fk": 0.1,
            "foot_skating": 0.05,
            "velocity": 0.05
        }
    )
    
    # Loss weighting strategy
    weight_loss_by_t: Literal["emulate_eps_pred"] = "emulate_eps_pred"
    """Weights to apply to the loss at each noise level."""

class TrainingLossComputer:
    """Helper class for computing the training loss."""

    def __init__(self, config: TrainingLossConfig, device: torch.device) -> None:
        self.config = config
        # Emulate loss weight for epsilon prediction
        assert self.config.weight_loss_by_t == "emulate_eps_pred"

    def compute_loss(
        self,
        t: Tensor,
        x0_pred: torch.FloatTensor,
        noise_pred: torch.FloatTensor,
        gt_noise: torch.FloatTensor,
        batch: EgoTrainingData,
        unwrapped_model: MotionUNet,
    ) -> MotionLosses:
        """Compute all training losses.
        
        Args:
            t: Timesteps for each batch element
            x0_pred: Model's predicted denoised motion
            noise_pred: Model's predicted noise
            gt_noise: Ground truth noise that was added
            batch: Training batch data
            unwrapped_model: Unwrapped model for accessing config
            
        Returns:
            MotionLosses containing all loss components
        """
        device = gt_noise.device
        batch_size, seq_len = batch.betas.shape[:2]

        # 1. Noise prediction loss
        noise_pred_loss = torch.nn.functional.mse_loss(noise_pred, gt_noise)

        # 2. Unpack predicted and ground truth motions
        x_0_pred = network.EgoDenoiseTraj.unpack(x0_pred, include_hands=unwrapped_model.config.include_hands)
        clean_motion = batch.pack()

        def weight_and_mask_loss(
            loss_per_step: Float[Tensor, "b t d"],
            bt_mask: Bool[Tensor, "b t"] = batch.mask,
            bt_mask_sum: Int[Tensor, ""] = torch.sum(batch.mask),
        ) -> Float[Tensor, ""]:
            """Helper to compute masked and weighted loss.
            
            Args:
                loss_per_step: Per-step loss values
                bt_mask: Binary mask for batch/time dimensions
                bt_mask_sum: Sum of mask for normalization
            
            Returns:
                Weighted and masked scalar loss
            """
            _, _, d = loss_per_step.shape
            return (
                torch.sum(
                    torch.sum(
                        torch.mean(loss_per_step, dim=-1) * bt_mask,
                        dim=-1,
                    )
                )
                / bt_mask_sum
            )

        # 3. Compute original losses between prediction and ground truth
        betas_loss = weight_and_mask_loss(
            (x_0_pred.betas - clean_motion.betas) ** 2 * 
            torch.tensor(self.config.beta_coeff_weights, device=device)
        )
        
        body_rot6d_loss = weight_and_mask_loss(
            (x_0_pred.body_rot6d - clean_motion.body_rot6d).reshape(
                (batch_size, seq_len, 21 * 6)
            ) ** 2
        )
        
        contacts_loss = weight_and_mask_loss(
            (x_0_pred.contacts - clean_motion.contacts) ** 2
        )

        # 4. Forward Kinematics Loss
        shaped_model = unwrapped_model.smpl_model.with_shape(clean_motion.betas)
        predicted_joint_positions = forward_kinematics(
            T_world_root=batch.T_world_cpf,
            Rs_parent_joint=SO3.from_rot6d(torch.cat([x_0_pred.body_rot6d, x_0_pred.hand_rot6d], dim=-2).reshape(batch_size, seq_len, -1, 6)).wxyz,
            t_parent_joint=shaped_model.t_parent_joint,
            parent_indices=unwrapped_model.smpl_model.parent_indices
)

        fk_loss = weight_and_mask_loss(
            (predicted_joint_positions[..., :batch.joints_wrt_world.shape[-2], 4:7] - batch.joints_wrt_world).reshape(batch_size, seq_len, -1) ** 2 # joints_wrt_world: (B, T, 21*3)
        )

        # 5. Foot Skating Loss
        foot_indices = [7, 8, 10, 11]  # Indices for foot joints
        foot_positions = predicted_joint_positions[..., foot_indices, 4:7] # (B, T, 4, 3)
        foot_velocities = foot_positions[:, 1:] - foot_positions[:, :-1] # (B, T-1, 4, 3)
        
        foot_contacts = clean_motion.contacts[..., foot_indices] # (B, T, 4)
        foot_skating_mask = foot_contacts[:, 1:] * batch.mask[:, 1:, None] # (B, T-1, 4)

        # Compute loss for each foot joint separately (B,T-1,4,3) -> (B,T-1,4)
        foot_skating_losses = torch.stack([
            weight_and_mask_loss(
                foot_velocities[..., i, :].pow(2), # (B,T-1,3) for each joint
                bt_mask=foot_skating_mask[..., i], # (B,T-1) for each joint
                bt_mask_sum=torch.maximum(
                    torch.sum(foot_skating_mask[..., i]) * 3, # Multiply by 3 for x,y,z
                    torch.tensor(1, device=device)
                )
            )
            for i in range(4)  # For each foot joint
        ]).sum()  # Sum losses from all joints
        foot_skating_loss = foot_skating_losses / 4  # Average across joints

        # 6. Velocity Loss
        joint_velocities = (
            predicted_joint_positions[..., 4:7][:, 1:] -
            predicted_joint_positions[..., 4:7][:, :-1]
        )
        gt_velocities = batch.joints_wrt_world[:, 1:] - batch.joints_wrt_world[:, :-1]

        velocity_loss = weight_and_mask_loss(
            (joint_velocities[..., :batch.joints_wrt_world.shape[-2], :] - gt_velocities).reshape(batch_size, seq_len-1, -1) ** 2,
            bt_mask=batch.mask[:, 1:],
            bt_mask_sum=torch.sum(batch.mask[:, 1:])
        )

        # 7. Hand Rotation Loss (if enabled)
        hand_rot6d_loss = torch.tensor(0.0, device=device)
        if unwrapped_model.config.include_hands:
            assert x_0_pred.hand_rot6d is not None
            assert clean_motion.hand_rot6d is not None
            pred_hand_flat = x_0_pred.hand_rot6d.reshape((batch_size, seq_len, -1))
            gt_hand_flat = clean_motion.hand_rot6d.reshape((batch_size, seq_len, -1))
            
            # Only compute loss for sequences with hand motion
            hand_motion = (
                torch.sum(
                    torch.sum(
                        torch.abs(gt_hand_flat - gt_hand_flat[:, 0:1, :]), dim=-1
                    )
                    * batch.mask,
                    dim=-1,
                )
                > 1e-5
            )
            hand_bt_mask = torch.logical_and(hand_motion[:, None], batch.mask)
            hand_rot6d_loss = weight_and_mask_loss(
                (pred_hand_flat - gt_hand_flat).reshape(
                    batch_size, seq_len, 30 * 6
                ) ** 2,
                bt_mask=hand_bt_mask,
                bt_mask_sum=torch.maximum(
                    torch.sum(hand_bt_mask), 
                    torch.tensor(256, device=device)
                ),
            )

        # 8. Combine all losses with weights
        total_loss = (
            self.config.noise_pred_weight * noise_pred_loss +
            self.config.loss_weights["betas"] * betas_loss +
            self.config.loss_weights["body_rot6d"] * body_rot6d_loss +
            self.config.loss_weights["contacts"] * contacts_loss +
            self.config.loss_weights["hand_rot6d"] * hand_rot6d_loss +
            self.config.loss_weights["fk"] * fk_loss +
            self.config.loss_weights["foot_skating"] * foot_skating_loss +
            self.config.loss_weights["velocity"] * velocity_loss
        )

        return MotionLosses(
            noise_pred_loss=noise_pred_loss,
            betas_loss=betas_loss,
            body_rot6d_loss=body_rot6d_loss,
            contacts_loss=contacts_loss,
            hand_rot6d_loss=hand_rot6d_loss,
            fk_loss=fk_loss,
            foot_skating_loss=foot_skating_loss,
            velocity_loss=velocity_loss,
            total_loss=total_loss
        )