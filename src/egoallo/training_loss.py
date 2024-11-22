"""Training loss configuration."""

import dataclasses
from typing import Literal, Union, Tuple

import torch.utils.data
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel

from . import network
from .data.amass_dataset import EgoTrainingData
from .sampling import CosineNoiseScheduleConstants
from .transforms import SO3


@dataclasses.dataclass(frozen=True)
class TrainingLossConfig:
    cond_dropout_prob: float = 0.0
    beta_coeff_weights: tuple[float, ...] = tuple(1 / (i + 1) for i in range(16))
    loss_weights: dict[str, float] = dataclasses.field(
        default_factory={
            "betas": 0.0,
            "body_rotmats": 1.0,
            "contacts": 0.0,
            # We don't have many hands in the AMASS dataset...
            "hand_rotmats": 0.00,
        }.copy
    )
    weight_loss_by_t: Literal["emulate_eps_pred"] = "emulate_eps_pred"
    """Weights to apply to the loss at each noise level."""


class TrainingLossComputer:
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

    def weight_and_mask_loss(
        self,
        loss_per_step: Float[Tensor, "b t d"],
        t: torch.Tensor,  # Added parameter for timestep weights
        bt_mask: Bool[Tensor, "b t"],
        bt_mask_sum: Int[Tensor, ""] | None = None,
        reduction: str = 'mean'
    ) -> Float[Tensor, ""] | Float[Tensor, "b d"]:
        """Weight and mask per-timestep losses (squared errors).
        
        Args:
            loss_per_step: Per-step losses with shape (batch, time, dim)
            t: Timesteps for diffusion weighting with shape (batch,)
            bt_mask: Binary mask with shape (batch, time)
            bt_mask_sum: Optional pre-computed mask sum
            reduction: Either 'mean' or 'none' to control output format
            
        Returns:
            Weighted and masked loss, either scalar or per-batch/dim tensor
        """
        batch, time, d = loss_per_step.shape
        assert bt_mask.shape == (batch, time)
        
        weight_t = self.weight_t[t].to(loss_per_step.device)
        assert weight_t.shape == (batch,)
        
        if bt_mask_sum is None:
            bt_mask_sum = torch.sum(bt_mask)

        # Mean across d axis
        per_step = torch.mean(loss_per_step, dim=-1) if reduction == 'mean' else loss_per_step
        
        # Sum across t axis
        per_batch = torch.sum(per_step * bt_mask, dim=-1)
        
        # Weight by timestep
        weighted = per_batch * weight_t
        
        if reduction == 'mean':
            # Sum across b axis and normalize
            return torch.sum(weighted) / bt_mask_sum
        else:
            # Return per-batch/dim losses
            return weighted

    def compute_denoising_loss(
        self,
        model: network.EgoDenoiser | DistributedDataParallel | OptimizedModule,
        unwrapped_model: network.EgoDenoiser,
        train_batch: EgoTrainingData,
    ) -> tuple[Tensor, dict[str, Tensor | float]]:
        """Compute a training loss for the EgoDenoiser model.

        Returns:
            A tuple (total loss tensor, dictionary of metrics to log).
        """
        metrics: dict[str, Tensor | float] = {}
        
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
        if unwrapped_model.config.include_hands:
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

        # Compute individual loss terms
        loss_terms: dict[str, Tensor | float] = {
            "betas": self.weight_and_mask_loss(
                (x_0_pred.betas - x_0.betas) ** 2 
                * x_0.betas.new_tensor(self.config.beta_coeff_weights),
                t,
                train_batch.mask
            ),
            "body_rotmats": self.weight_and_mask_loss(
                (x_0_pred.body_rotmats - x_0.body_rotmats).reshape(
                    (batch, time, 21 * 3 * 3)
                ) ** 2,
                t,
                train_batch.mask
            ),
            "contacts": self.weight_and_mask_loss(
                (x_0_pred.contacts - x_0.contacts) ** 2,
                t,
                train_batch.mask
            ),
        }

        # Include hand objective.
        # We didn't use this in the paper.
        if unwrapped_model.config.include_hands:
            assert x_0_pred.hand_rotmats is not None
            assert x_0.hand_rotmats is not None
            assert x_0.hand_rotmats.shape == (batch, time, 30, 3, 3)

            # Detect whether or not hands move in a sequence.
            # We should only supervise sequences where the hands are actully tracked / move;
            # we mask out hands in AMASS sequences where they are not tracked.
            gt_hand_flatmat = x_0.hand_rotmats.reshape((batch, time, -1))
            hand_motion = (
                torch.sum(  # (b,) from (b, t)
                    torch.sum(  # (b, t) from (b, t, d)
                        torch.abs(gt_hand_flatmat - gt_hand_flatmat[:, 0:1, :]), dim=-1
                    )
                    # Zero out changes in masked frames.
                    * train_batch.mask,
                    dim=-1,
                )
                > 1e-5
            )
            assert hand_motion.shape == (batch,)

            hand_bt_mask = torch.logical_and(hand_motion[:, None], train_batch.mask)
            loss_terms["hand_rotmats"] = torch.sum(
                self.weight_and_mask_loss(
                    (x_0_pred.hand_rotmats - x_0.hand_rotmats).reshape(
                        batch, time, 30 * 3 * 3
                    )
                    ** 2,
                    t,
                    hand_bt_mask,
                    # We want to weight the loss by the number of frames where
                    # the hands actually move, but gradients here can be too
                    # noisy and put NaNs into mixed-precision training when we
                    # inevitably sample too few frames. So we clip the
                    # denominator.
                    bt_mask_sum=torch.maximum(
                        torch.sum(hand_bt_mask), torch.tensor(256, device=device)
                    ),
                )
            )
        else:
            loss_terms["hand_rotmats"] = 0.0

        assert loss_terms.keys() == self.config.loss_weights.keys()

        # Log individual loss terms
        for name, term in loss_terms.items():
            metrics[f"train/loss_{name}"] = term

        # Compute weighted total loss
        total_loss = sum([loss_terms[k] * self.config.loss_weights[k] for k in loss_terms])
        assert isinstance(total_loss, Tensor)
        assert total_loss.shape == ()
        
        # Add total loss to metrics
        metrics["train/total_loss"] = total_loss

        return total_loss, metrics
