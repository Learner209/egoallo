"""Training loss configuration and computation."""
import dataclasses
from typing import Literal, NamedTuple, Union, Tuple, Dict, Optional

import torch
from torch import Tensor
import torch.utils.data
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel
from diffusers import DDPMScheduler
from jaxtyping import Bool, Float, Int

from .data.amass import EgoTrainingData
from .fncsmpl import forward_kinematics
from .network import EgoDenoiseTraj
from .transforms import SO3
from .motion_diffusion_pipeline import MotionUNet
from . import network

class MotionLosses(NamedTuple):
    """Container for all loss components."""
    betas_loss: Tensor      # Loss on SMPL shape parameters
    body_rot6d_loss: Tensor # Loss on body joint rotations
    contacts_loss: Tensor   # Loss on contact states
    # hand_rot6d_loss: Tensor # Loss on hand joint rotations (if enabled)
    # fk_loss: Tensor        # Forward kinematics loss
    # foot_skating_loss: Tensor  # Foot skating prevention loss
    # velocity_loss: Tensor   # Joint velocity smoothness loss
    total_loss: Tensor     # Weighted sum of all losses

@dataclasses.dataclass(frozen=True)
class TrainingLossConfig:
    """Configuration for training losses."""
    
    # Dropout probability for conditional inputs
    cond_dropout_prob: float = 0.0
    
    # Use uniform weights (1.0) for all beta coefficients
    beta_coeff_weights: tuple[float, ...] = tuple(1.0 for _ in range(16))
    
    # Individual loss component weights with uniform weighting
    loss_weights: dict[str, float] = dataclasses.field(
        default_factory=lambda: {
            "body_rot6d": 1.0,    # Primary rotation loss
            "betas": 0.1,         # Body shape parameters 
            "contacts": 0.1,      # Contact states
            "hand_rot6d": 0.0,    # Hand rotation loss
            "fk": 0.0,           # Forward kinematics (disabled)
            "foot_skating": 0.0,  # Foot skating prevention (disabled)
            "velocity": 0.0       # Velocity consistency (disabled)
        }
    )
    
    # Loss weighting strategy
    weight_loss_by_t: Literal["emulate_eps_pred"] = "emulate_eps_pred"
    """Weights to apply to the loss at each noise level."""

class MotionLossComputer:
    """Improved loss computer with better stability and convergence."""
    
    def __init__(self, config: TrainingLossConfig, device: torch.device, scheduler: DDPMScheduler):
        self.config = config
        self.device = device
        self.scheduler = scheduler
        
        # Initialize loss weights with validation
        # self._validate_and_normalize_weights()
        
    def _validate_and_normalize_weights(self):
        """Validate and normalize loss weights to prevent dominance."""
        total = sum(self.config.loss_weights.values())
        if total == 0:
            raise ValueError("At least one loss weight must be non-zero")
        
        # Normalize weights if needed
        if total > 1:
            self.config.loss_weights = {
                k: v/total for k, v in self.config.loss_weights.items()
            }

    def compute_snr_weights(self, t: Int[Tensor, "b"]) -> Float[Tensor, "b"]:
        """Compute SNR-based weights for timesteps."""
        alpha_t = self.scheduler.alphas_cumprod[t]
        snr = alpha_t / (1 - alpha_t)
        weights = snr / (snr + 1.0 + 1e-8)
        return weights

    def weight_and_mask_loss(
        self,
        loss_per_step: Float[Tensor, "b t *d"],
        t: Int[Tensor, "b"],
        bt_mask: Optional[Bool[Tensor, "b t"]] = None,
        bt_mask_sum: Optional[Int[Tensor, ""]] = None,
        reduction: str = 'mean'
    ) -> Float[Tensor, "..."]:
        """Apply timestep-dependent weighting and masking to loss values."""
        weights = self.compute_snr_weights(t)
        
        # Reshape weights to match the dimensions of loss_per_step
        weights = weights.view(*weights.shape, *([1] * (loss_per_step.dim() - 1)))
        
        weighted_loss = weights * loss_per_step  # (B, T, *)
        
        if bt_mask is not None:
            # Expand bt_mask to match the dimensions of weighted_loss
            expanded_bt_mask = bt_mask.view(*bt_mask.shape, *([1] * (loss_per_step.dim() - 2)))
            expanded_bt_mask = expanded_bt_mask.expand_as(weighted_loss)
            weighted_loss = weighted_loss * expanded_bt_mask  # (B, T, *)
            
            if reduction == 'mean':
                mask_sum = bt_mask_sum if bt_mask_sum is not None else expanded_bt_mask.sum()
                return weighted_loss.sum() / mask_sum
                
        if reduction == 'mean':
            return weighted_loss.mean()
            
        return weighted_loss.sum(dim=1)

    def compute_rotation_loss(
        self,
        pred_rot6d: torch.Tensor,  # (B, T, J, 6)
        target_rot6d: torch.Tensor,  # (B, T, J, 6)
        t: torch.Tensor,  # (B,)
        mask: torch.Tensor,  # (B, T)
        return_per_joint: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute weighted geodesic rotation loss between predicted and target 6D rotations.
        
        Args:
            pred_rot6d: Predicted 6D rotations with shape (batch, time, joints, 6)
            target_rot6d: Target 6D rotations with shape (batch, time, joints, 6) 
            t: Timesteps for diffusion weighting with shape (batch,)
            mask: Binary mask indicating valid frames with shape (batch, time)
            return_per_joint: Whether to return per-joint losses
            
        Returns:
            Total weighted geodesic rotation loss if return_per_joint=False
            Tuple of (total loss, per-joint losses) if return_per_joint=True
        """
        # Convert 6D rotation representation to SO3 objects
        pred_rot_so3 = SO3.from_rot6d(pred_rot6d)  # (B, T, J)
        target_rot_so3 = SO3.from_rot6d(target_rot6d)  # (B, T, J)
        
        # Compute geodesic distance between rotations
        rot_loss = pred_rot_so3.log() - target_rot_so3.log()  # (B, T, J, 3)
        rot_loss = torch.norm(rot_loss, dim=-1)  # (B, T, J)
        
        # Apply timestep-dependent weighting using weight_and_mask_loss
        weighted_loss = self.weight_and_mask_loss(
            rot_loss,
            t,
            mask,
            reduction='none'  # Keep per-joint dimension
        )  # (B, J)
        
        if return_per_joint:
            # Return both total loss and per-joint losses
            total_loss = weighted_loss.mean()
            return total_loss, weighted_loss
        
        # Return only total loss
        return weighted_loss.mean()

    def compute_loss(
        self,
        t: Tensor,
        x0_pred: torch.FloatTensor,
        batch: EgoTrainingData,
        unwrapped_model: MotionUNet,
        return_joint_losses: bool = False
    ):
        """
        Compute training losses for motion prediction.
        
        Args:
            t: Timesteps tensor
            x0_pred: Predicted denoised motion
            batch: Training data batch
            unwrapped_model: Unwrapped model for accessing parameters
            return_joint_losses: Whether to return individual joint losses
            
        Returns:
            losses: Combined losses
            joint_losses: Individual joint losses if requested
        """
        device = x0_pred.device
        batch_size, seq_len = batch.betas.shape[:2]

        # 2. Unpack predicted and ground truth motions
        x_0_pred = network.EgoDenoiseTraj.unpack(x0_pred, include_hands=unwrapped_model.config.include_hand_motion, should_project_rot6d=True)
        clean_motion = batch.pack()

        # 3. Compute original losses between prediction and ground truth
        betas_loss = self.weight_and_mask_loss(
            (x_0_pred.betas - clean_motion.betas) ** 2,
            t,
            batch.mask
        )
        
        body_rot6d_loss = self.weight_and_mask_loss(
            (x_0_pred.body_rot6d - clean_motion.body_rot6d) ** 2,
            t,
            batch.mask
        ) # (B, T)
        per_joint_losses = (x_0_pred.body_rot6d - clean_motion.body_rot6d) ** 2 # (B, T, 21*3)
        
        contacts_loss = self.weight_and_mask_loss(
            (x_0_pred.contacts - clean_motion.contacts) ** 2,
            t,
            batch.mask
        )

        # 4. Forward Kinematics Loss
#         shaped_model = unwrapped_model.smpl_model.with_shape(clean_motion.betas)
#         predicted_joint_positions = forward_kinematics(
#             T_world_root=batch.T_world_cpf,
#             Rs_parent_joint=SO3.from_rot6d(torch.cat([x_0_pred.body_rot6d, x_0_pred.hand_rot6d], dim=-2).reshape(batch_size, seq_len, -1, 6)).wxyz,
#             t_parent_joint=shaped_model.t_parent_joint,
#             parent_indices=unwrapped_model.smpl_model.parent_indices
# )

#         fk_loss = self.weight_and_mask_loss(
#             (predicted_joint_positions[..., :batch.joints_wrt_world.shape[-2], 4:7] - batch.joints_wrt_world).reshape(batch_size, seq_len, -1) ** 2, # joints_wrt_world: (B, T, 21*3)
#             t,
#             batch.mask
#         )

#         # 5. Foot Skating Loss
#         foot_indices = [7, 8, 10, 11]  # Indices for foot joints
#         foot_positions = predicted_joint_positions[..., foot_indices, 4:7] # (B, T, 4, 3)
#         foot_velocities = foot_positions[:, 1:] - foot_positions[:, :-1] # (B, T-1, 4, 3)
        
#         foot_contacts = clean_motion.contacts[..., foot_indices] # (B, T, 4)
#         foot_skating_mask = foot_contacts[:, 1:] * batch.mask[:, 1:, None] # (B, T-1, 4)

#         # Compute loss for each foot joint separately (B,T-1,4,3) -> (B,T-1,4)
#         foot_skating_losses = torch.stack([
#             self.weight_and_mask_loss(
#                 foot_velocities[..., i, :].pow(2), # (B,T-1,3) for each joint
#                 t,
#                 bt_mask=foot_skating_mask[..., i], # (B,T-1) for each joint
#                 bt_mask_sum=torch.maximum(
#                     torch.sum(foot_skating_mask[..., i]) * 3, # Multiply by 3 for x,y,z
#                     torch.tensor(1, device=device)
#                 )
#             )
#             for i in range(4)  # For each foot joint
#         ]).sum()  # Sum losses from all joints
#         foot_skating_loss = foot_skating_losses / 4  # Average across joints

#         # 6. Velocity Loss
#         joint_velocities = (
#             predicted_joint_positions[..., 4:7][:, 1:] -
#             predicted_joint_positions[..., 4:7][:, :-1]
#         )
#         gt_velocities = batch.joints_wrt_world[:, 1:] - batch.joints_wrt_world[:, :-1]

#         # Fix: Use t[:, None].repeat(1, seq_len-1) to match the velocity sequence length
#         velocity_t = t  # The timesteps tensor needs to match the velocity sequence length
#         velocity_loss = self.weight_and_mask_loss(
#             (joint_velocities[..., :batch.joints_wrt_world.shape[-2], :] - gt_velocities).reshape(batch_size, seq_len-1, -1) ** 2,
#             t,  # Original timesteps tensor
#             bt_mask=batch.mask[:, 1:],
#             bt_mask_sum=torch.sum(batch.mask[:, 1:])
#         )

#         # 7. Hand Rotation Loss (if enabled)
#         hand_rot6d_loss = torch.tensor(0.0, device=device)
#         if unwrapped_model.config.include_hand_motion:
#             assert x_0_pred.hand_rot6d is not None
#             assert clean_motion.hand_rot6d is not None
#             pred_hand_flat = x_0_pred.hand_rot6d.reshape((batch_size, seq_len, -1))
#             gt_hand_flat = clean_motion.hand_rot6d.reshape((batch_size, seq_len, -1))
            
#             # Only compute loss for sequences with hand motion
#             hand_motion = (
#                 torch.sum(
#                     torch.sum(
#                         torch.abs(gt_hand_flat - gt_hand_flat[:, 0:1, :]), dim=-1
#                     )
#                     * batch.mask,
#                     dim=-1,
#                 )
#                 > 1e-5
#             )
#             hand_bt_mask = torch.logical_and(hand_motion[:, None], batch.mask)
#             hand_rot6d_loss = self.weight_and_mask_loss(
#                 (pred_hand_flat - gt_hand_flat).reshape(
#                     batch_size, seq_len, 30 * 6
#                 ) ** 2,
#                 t,
#                 bt_mask=hand_bt_mask,
#                 bt_mask_sum=torch.maximum(
#                     torch.sum(hand_bt_mask), 
#                     torch.tensor(256, device=device)
#                 ),
#             )

        # 8. Combine all losses with weights
        total_loss = (
            self.config.loss_weights["betas"] * betas_loss +
            self.config.loss_weights["body_rot6d"] * body_rot6d_loss +
            self.config.loss_weights["contacts"] * contacts_loss 
            # self.config.loss_weights["hand_rot6d"] * hand_rot6d_loss +
            # self.config.loss_weights["fk"] * fk_loss +
            # self.config.loss_weights["foot_skating"] * foot_skating_loss +
            # self.config.loss_weights["velocity"] * velocity_loss
        )

        losses = MotionLosses(
            betas_loss=betas_loss * self.config.loss_weights["betas"],
            body_rot6d_loss=body_rot6d_loss * self.config.loss_weights["body_rot6d"],
            contacts_loss=contacts_loss * self.config.loss_weights["contacts"],
            # hand_rot6d_loss=hand_rot6d_loss * self.config.loss_weights["hand_rot6d"],
            # fk_loss=fk_loss * self.config.loss_weights["fk"],
            # foot_skating_loss=foot_skating_loss * self.config.loss_weights["foot_skating"],
            # velocity_loss=velocity_loss * self.config.loss_weights["velocity"],
            total_loss=total_loss
        )

        if return_joint_losses:
            # Permute to iterate over joints
            num_joints = x_0_pred.body_rot6d.shape[2]
            per_joint_losses = per_joint_losses.view(batch_size, seq_len, num_joints, 6)  # Reshape to (B, T, J, 6)
            joint_losses_dict = {
                f"body_rot6d_j{i}": loss.mean().item()  # Take mean across batch dimension
                for i, loss in enumerate(per_joint_losses.permute(2, 0, 1, 3).mean(dim=[1, 3]))  # Average over batch and last 6D dimension
            }
            return losses, joint_losses_dict

        return losses, None