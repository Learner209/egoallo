"""Training loss configuration and computation."""
import dataclasses
from typing import Literal, NamedTuple, Union, Tuple, Dict, Optional, Any

import torch
import torch.utils.data
from torch import Tensor
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel
from jaxtyping import Bool, Float, Int
from dataclasses import dataclass

from . import network
from .data.amass import EgoTrainingData
from .sampling import CosineNoiseScheduleConstants
from .transforms import SO3

class MotionLosses(NamedTuple):
    """Container for all loss components."""
    betas_loss: Tensor
    body_rot6d_loss: Tensor
    contacts_loss: Tensor
    hand_rot6d_loss: Tensor
    fk_loss: Tensor
    foot_skating_loss: Tensor
    velocity_loss: Tensor
    total_loss: Tensor

@dataclass
class TrainingLossConfig:
    """Configuration for training losses."""
    
    # Dropout probability for conditional inputs
    cond_dropout_prob: float = 0.0
    
    # Use decreasing weights for beta coefficients
    beta_coeff_weights: tuple[float, ...] = tuple(1 / (i + 1) for i in range(16))
    
    # Individual loss component weights
    loss_weights: dict[str, float] = dataclasses.field(
        default_factory=lambda: {
            "betas": 0.1,
            "body_rot6d": 1.0,
            "contacts": 0.1,
            "hand_rot6d": 0.0,
            "fk": 0.0,
            "foot_skating": 0.0,
            "velocity": 0.0
        }
    )
    
    weight_loss_by_t: Literal["emulate_eps_pred"] = "emulate_eps_pred"

@dataclass
class RotationLossInputs:
    """Container for rotation loss computation inputs."""
    pred_rotmats: Float[Tensor, "batch time joints 3 3"]
    target_rotmats: Float[Tensor, "batch time joints 3 3"]
    mask: Bool[Tensor, "batch time"]
    return_per_joint: bool = False

@dataclass
class RotationLossOutputs:
    """Container for rotation loss computation outputs."""
    overall_loss: Float[Tensor, "batch time"]
    per_joint_losses: Optional[Float[Tensor, "joints"]] = None

class RotationLossComputer:
    """Handles computation of geodesic losses between rotation matrices."""
    
    def __init__(self, eps: float = 1e-7):
        self.eps = eps
    
    @staticmethod
    def _validate_inputs(inputs: RotationLossInputs) -> None:
        """Validates input tensors for shape and value consistency."""
        assert inputs.pred_rotmats.shape == inputs.target_rotmats.shape, \
            "Predicted and target rotation matrices must have same shape"
        assert inputs.pred_rotmats.shape[-2:] == (3, 3), \
            "Last two dimensions must be 3x3 for rotation matrices"
        assert inputs.mask.shape == inputs.pred_rotmats.shape[:2], \
            "Mask shape must match batch and time dimensions"
    
    @staticmethod
    def compute_rotation_difference(
        pred: Float[Tensor, "... 3 3"],
        target: Float[Tensor, "... 3 3"]
    ) -> Float[Tensor, "... 3 3"]:
        """Computes the rotation difference matrix R1.T @ R2."""
        return torch.matmul(pred.transpose(-2, -1), target)
    
    @staticmethod
    def compute_geodesic_distance(
        rot_diff: Float[Tensor, "... 3 3"],
        eps: float = 1e-7
    ) -> Float[Tensor, "..."]:
        """Computes the geodesic distance from rotation difference matrix."""
        # Compute matrix trace
        trace = torch.diagonal(rot_diff, dim1=-2, dim2=-1).sum(-1)
        
        # Clamp for numerical stability
        trace_normalized = torch.clamp(
            (trace - 1.0) / 2.0,
            min=-1.0 + eps,
            max=1.0 - eps
        )
        
        return torch.acos(trace_normalized)

    def __call__(self, inputs: RotationLossInputs) -> RotationLossOutputs:
        """Computes geodesic loss between rotation matrices for all joints.
        
        Args:
            inputs: RotationLossInputs containing predicted and target rotations,
                   mask, and computation options
        
        Returns:
            RotationLossOutputs containing overall loss and optionally per-joint losses
        """
        # Validate inputs
        self._validate_inputs(inputs)
        
        # Compute rotation differences
        rot_diff = self.compute_rotation_difference(
            inputs.pred_rotmats,
            inputs.target_rotmats
        )
        
        # Compute geodesic distances
        geodesic_dist = self.compute_geodesic_distance(rot_diff, self.eps)
        
        # Apply mask to distances
        masked_dist = geodesic_dist * inputs.mask.unsqueeze(-1)
        
        # Compute losses
        overall_loss = masked_dist.mean(-1)  # Average across joints
        
        if not inputs.return_per_joint:
            return RotationLossOutputs(overall_loss=overall_loss)
            
        # Compute per-joint losses
        per_joint_losses = masked_dist.sum((0, 1)) / inputs.mask.sum()
        return RotationLossOutputs(
            overall_loss=overall_loss,
            per_joint_losses=per_joint_losses
        )

class MotionLossComputer:
    """Helper class for computing the training loss. Contains a single method
    for computing a training loss."""

    def __init__(self, config: TrainingLossConfig, device: torch.device) -> None:
        self.config = config
        self.noise_constants = (
            CosineNoiseScheduleConstants.compute(timesteps=1000)
            .to(device)
            .map(lambda tensor: tensor.to(torch.float32))
        )

        # Emulate loss weight that would be ~equivalent to epsilon prediction.
        #
        # This will penalize later errors (close to the end of sampling) much
        # more than earlier ones (at the start of sampling).
        assert self.config.weight_loss_by_t == "emulate_eps_pred"
        weight_t = self.noise_constants.alpha_bar_t / (
            1 - self.noise_constants.alpha_bar_t
        )
        # Pad for numerical stability, and scale between [padding, 1.0].
        padding = 0.01
        self.weight_t = weight_t / weight_t[1] * (1.0 - padding) + padding

    def compute_rotation_loss(
        self,
        pred_rotmats: Float[Tensor, "batch time joints 3 3"],
        target_rotmats: Float[Tensor, "batch time joints 3 3"],
        mask: Bool[Tensor, "batch time"],
        return_per_joint: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Compute geodesic loss between rotation matrices for all joints."""
        # Create rotation loss computer instance
        loss_computer = RotationLossComputer()
        
        # Prepare inputs
        inputs = RotationLossInputs(
            pred_rotmats=pred_rotmats,
            target_rotmats=target_rotmats,
            mask=mask,
            return_per_joint=return_per_joint
        )
        
        # Compute losses
        outputs = loss_computer(inputs)
        
        if return_per_joint:
            return outputs.overall_loss, outputs.per_joint_losses
        return outputs.overall_loss

    def compute_denoising_loss(
        self,
        model: network.EgoDenoiser | DistributedDataParallel | OptimizedModule,
        unwrapped_model: network.EgoDenoiser,
        train_batch: EgoTrainingData,
        return_per_joint_losses: bool = True
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Compute denoising loss with detailed logging outputs."""
        log_outputs: dict[str, Tensor | float] = {}

        batch, time, num_joints, _ = train_batch.body_quats.shape
        assert num_joints == 21
        if unwrapped_model.config.include_hands:
            assert train_batch.hand_quats is not None
            x_0 = network.EgoDenoiseTraj(
                betas=train_batch.betas.expand((batch, time, 16)),
                body_rotmats=SO3(train_batch.body_quats).as_matrix(),
                contacts=train_batch.contacts,
                hand_rotmats=SO3(train_batch.hand_quats).as_matrix(),
            )
        else:
            x_0 = network.EgoDenoiseTraj(
                betas=train_batch.betas.expand((batch, time, 16)),
                body_rotmats=SO3(train_batch.body_quats).as_matrix(),
                contacts=train_batch.contacts,
                hand_rotmats=None,
            )
        x_0_packed = x_0.pack()
        device = x_0_packed.device
        assert x_0_packed.shape == (batch, time, unwrapped_model.get_d_state())

        # Diffuse.
        t = torch.randint(
            low=1,
            high=unwrapped_model.config.max_t + 1,
            size=(batch,),
            device=device,
        )
        eps = torch.randn(x_0_packed.shape, dtype=x_0_packed.dtype, device=device)
        assert self.noise_constants.alpha_bar_t.shape == (
            unwrapped_model.config.max_t + 1,
        )
        alpha_bar_t = self.noise_constants.alpha_bar_t[t, None, None]
        assert alpha_bar_t.shape == (batch, 1, 1)
        x_t_packed = (
            torch.sqrt(alpha_bar_t) * x_0_packed + torch.sqrt(1.0 - alpha_bar_t) * eps
        )

        hand_positions_wrt_cpf: Tensor | None = None
        if unwrapped_model.config.include_hand_positions_cond:
            # Joints 19 and 20 are the hand positions.
            hand_positions_wrt_cpf = train_batch.joints_wrt_cpf[:, :, 19:21, :].reshape(
                (batch, time, 6)
            )

            # Exclude hand positions for some items in the batch. We'll just do
            # this by passing in zeros.
            hand_positions_wrt_cpf = torch.where(
                # Uniformly drop out with some uniformly sampled probability.
                # :)
                (
                    torch.rand((batch, time, 1), device=device)
                    < torch.rand((batch, 1, 1), device=device)
                ),
                hand_positions_wrt_cpf,
                0.0,
            )

        # Denoise.
        x_0_packed_pred = model.forward(
            x_t_packed=x_t_packed,
            t=t,
            T_world_cpf=train_batch.T_world_cpf,
            T_cpf_tm1_cpf_t=train_batch.T_cpf_tm1_cpf_t,
            hand_positions_wrt_cpf=hand_positions_wrt_cpf,
            project_output_rotmats=False,
            mask=train_batch.mask,
            cond_dropout_keep_mask=torch.rand((batch,), device=device)
            > self.config.cond_dropout_prob
            if self.config.cond_dropout_prob > 0.0
            else None,
        )
        assert isinstance(x_0_packed_pred, torch.Tensor)
        x_0_pred = network.EgoDenoiseTraj.unpack(
            x_0_packed_pred, include_hands=unwrapped_model.config.include_hands
        )

        weight_t = self.weight_t[t].to(device)
        assert weight_t.shape == (batch,)

        def weight_and_mask_loss(
            loss_per_step: Float[Tensor, "b t d"],
            # bt stands for "batch time"
            bt_mask: Bool[Tensor, "b t"] = train_batch.mask,
            bt_mask_sum: Int[Tensor, ""] = torch.sum(train_batch.mask),
        ) -> Float[Tensor, ""]:
            """Weight and mask per-timestep losses (squared errors)."""
            _, _, d = loss_per_step.shape
            assert loss_per_step.shape == (batch, time, d)
            assert bt_mask.shape == (batch, time)
            assert weight_t.shape == (batch,)
            return (
                # Sum across b axis.
                torch.sum(
                    # Sum across t axis.
                    torch.sum(
                        # Mean across d axis.
                        torch.mean(loss_per_step, dim=-1) * bt_mask,
                        dim=-1,
                    )
                    * weight_t
                )
                / bt_mask_sum
            )

        body_rotmats_loss, per_joint_losses = self.compute_rotation_loss(
            x_0_pred.body_rotmats.reshape(batch, time, -1, 3, 3),
            x_0.body_rotmats.reshape(batch, time, -1, 3, 3),
            train_batch.mask,
            return_per_joint=True
        )
        

        # Calculate and store individual loss components
        loss_terms = {
            "body_rotmats": weight_and_mask_loss(body_rotmats_loss),
            "betas": weight_and_mask_loss((x_0_pred.betas - x_0.betas) ** 2),
            "contacts": weight_and_mask_loss((x_0_pred.contacts - x_0.contacts) ** 2),
            "hand_rotmats": self._compute_hand_loss(x_0_pred, x_0, train_batch) if unwrapped_model.config.include_hands else 0.0
        }

        # Compute total weighted loss
        total_loss = sum(
            self.config.loss_weights[key] * loss for key, loss in loss_terms.items()
        )

        # Create a MotionLosses object to encapsulate all loss components
        losses = MotionLosses(
            betas_loss=loss_terms["betas"] * self.config.loss_weights["betas"],
            body_rot6d_loss=loss_terms["body_rotmats"] * self.config.loss_weights["body_rot6d"],
            contacts_loss=loss_terms["contacts"] * self.config.loss_weights["contacts"],
            hand_rot6d_loss=loss_terms["hand_rotmats"] * self.config.loss_weights["hand_rot6d"],
            fk_loss=torch.tensor(0.0, device=device),
            foot_skating_loss=torch.tensor(0.0, device=device),
            velocity_loss=torch.tensor(0.0, device=device),
            total_loss=total_loss
        )

        # Prepare detailed log outputs
        log_outputs = {
            "train/betas_loss": losses.betas_loss.item(),
            "train/body_rot6d_loss": losses.body_rot6d_loss.item(),
            "train/contacts_loss": losses.contacts_loss.item(),
            "train/hand_rot6d_loss": losses.hand_rot6d_loss.item(),
            "train/fk_loss": losses.fk_loss.item(),
            "train/foot_skating_loss": losses.foot_skating_loss.item(),
            "train/velocity_loss": losses.velocity_loss.item(),
            "train/total_loss": losses.total_loss.item(),
        }

        # Add joint group losses if requested
        if return_per_joint_losses:
            for group_name, loss_value in joint_losses.items():
                log_outputs[f"train/body_rot6d_loss/{group_name}"] = loss_value

        return losses.total_loss, log_outputs

    def _compute_hand_loss(self, x_0_pred, x_0, train_batch):
        """Compute the hand rotation loss if hands are included."""
        assert x_0_pred.hand_rotmats is not None
        assert x_0.hand_rotmats is not None
        assert x_0.hand_rotmats.shape == (batch, time, 30, 3, 3)

        # Detect hand motion in the sequence
        gt_hand_flatmat = x_0.hand_rotmats.reshape((batch, time, -1))
        hand_motion = (
            torch.sum(
                torch.sum(
                    torch.abs(gt_hand_flatmat - gt_hand_flatmat[:, 0:1, :]), dim=-1
                ) * train_batch.mask,
                dim=-1,
            ) > 1e-5
        )
        assert hand_motion.shape == (batch,)

        hand_bt_mask = torch.logical_and(hand_motion[:, None], train_batch.mask)
        return torch.sum(
            weight_and_mask_loss(
                (x_0_pred.hand_rotmats - x_0.hand_rotmats).reshape(batch, time, 30 * 3 * 3) ** 2,
                bt_mask=hand_bt_mask,
                bt_mask_sum=torch.maximum(
                    torch.sum(hand_bt_mask), torch.tensor(256, device=device)
                ),
            )
        )