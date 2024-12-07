"""Training loss configuration."""

import dataclasses
from typing import Literal

import torch.utils.data
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel

from . import network
from .data.dataclass import EgoTrainingData
from .sampling import CosineNoiseScheduleConstants
from .transforms import SO3


from egoallo.config import make_cfg, CONFIG_FILE
local_config_file = CONFIG_FILE
CFG = make_cfg(config_name="defaults", config_file=local_config_file, cli_args=[])


from egoallo.utils.setup_logger import setup_logger
logger = setup_logger(output=None, name=__name__)

@dataclasses.dataclass(frozen=True)
class TrainingLossConfig:
    cond_dropout_prob: float = 0.0
    beta_coeff_weights: tuple[float, ...] = tuple(1 / (i + 1) for i in range(16))
    loss_weights: dict[str, float] = dataclasses.field(
        default_factory={
            "betas": 0.00,
            "body_rotmats": 1.0,
            "contacts": 0.00,
            "hand_rotmats": 0.00,
            "R_world_root": 1.0,
            "t_world_root": 1.0,
            "joints": 0.00,
            "foot_skating": 0.00,
            "velocity": 0.00,
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

    def compute_geodesic_distance(self, R1: Float[Tensor, "... 3 3"], R2: Float[Tensor, "... 3 3"]) -> Float[Tensor, "..."]:
        """Compute geodesic distance between rotation matrices.
        
        Args:
            R1: First rotation matrix
            R2: Second rotation matrix
            
        Returns:
            Geodesic distance in radians
        """
        # Compute R1 @ R2.T
        R_diff = R1 @ R2.transpose(-2, -1)
        
        # Get trace of the difference matrix
        trace = R_diff.diagonal(dim1=-2, dim2=-1).sum(-1)
        
        # Clamp for numerical stability
        cos_theta = (trace - 1) / 2
        cos_theta = torch.clamp(cos_theta, -1 + 1e-7, 1 - 1e-7)
        
        # Return geodesic distance
        return torch.acos(cos_theta)

    def compute_denoising_loss(
        self,
        model: network.EgoDenoiser | DistributedDataParallel | OptimizedModule,
        unwrapped_model: network.EgoDenoiser,
        train_batch: EgoTrainingData,
    ) -> tuple[Tensor, dict[str, Tensor | float]]:
        """Compute a training loss for the EgoDenoiser model.

        Returns:
            A tuple (loss tensor, dictionary of things to log).
        """
            
        log_outputs: dict[str, Tensor | float] = {}

        batch, time, _, _ = train_batch.body_quats.shape

        num_joints = CFG.smplh.num_joints
        # assert num_joints == 22
        
        T_world_root = train_batch.T_world_root
        R_world_root = SO3(T_world_root[..., :4]).as_matrix()
        t_world_root = T_world_root[..., 4:7]
        if unwrapped_model.config.include_hands:
            assert train_batch.hand_quats is not None
            x_0 = network.EgoDenoiseTraj(
                betas=train_batch.betas.expand((batch, time, 16)),
                body_rotmats=SO3(train_batch.body_quats).as_matrix(),
                contacts=train_batch.contacts,
                hand_rotmats=SO3(train_batch.hand_quats).as_matrix(),
                R_world_root=R_world_root,
                t_world_root=t_world_root,
            )
        else:
            x_0 = network.EgoDenoiseTraj(
                betas=train_batch.betas.expand((batch, time, 16)),
                body_rotmats=SO3(train_batch.body_quats).as_matrix(),
                contacts=train_batch.contacts,
                hand_rotmats=None,
                R_world_root=R_world_root,
                t_world_root=t_world_root,
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

        
        # define per-timestep weightting scheme and construct weighting function.
        weight_t = self.weight_t[t].to(device)
        assert weight_t.shape == (batch,)

        def weight_and_mask_loss(
            loss_per_step: Float[Tensor, "b t d"],
            # bt stands for "batch time"
            bt_mask: Bool[Tensor, "b t"] = train_batch.mask,
            bt_mask_sum: Int[Tensor, ""] = torch.sum(train_batch.mask),
        ) -> Float[Tensor, ""]:
            """Weight and mask per-timestep losses (squared errors)."""
            _, time, d = loss_per_step.shape
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
            joints=train_batch.joints_wrt_world,
            visible_joints_mask=train_batch.visible_joints_mask,
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
        # breakpoint()
        # import ipdb; ipdb.set_trace()

        # Add joint position loss calculation
        # Get predicted joint positions through forward kinematics
        x_0_posed = x_0_pred.apply_to_body(unwrapped_model.body_model.to(device)) # (b, t, 22, 3)
        pred_joints = torch.cat([x_0_posed.Ts_world_joint[..., :num_joints-1, 4:7], x_0_posed.T_world_root[..., 4:7].unsqueeze(dim=-2)], dim=-2)  # (b, t, 22, 3)
        assert pred_joints.shape == (batch, time, num_joints, 3)
        
        # Get ground truth joints from training batch
        gt_joints = train_batch.joints_wrt_world  # (b, t, 22, 3)
        assert gt_joints.shape == (batch, time, num_joints, 3)
        
        # Calculate joint position loss with masking
        joint_loss = (pred_joints - gt_joints) ** 2  # (b, t, 22, 3)
        
        # Apply joint visibility mask and average
        # visible_joints_mask: shape (b, t, 22), joint_loss: shape (b, t, 22, 3)
        # Compute masked joint loss while handling shape alignment
        # joint_loss: (b, t, 22, 3), visible_joints_mask: (b, t, 22)
        # breakpoint()
        invisible_joint_loss = (
            joint_loss * (~train_batch.visible_joints_mask[..., None].bool())  # Add channel dim to inverted mask
        ).sum(dim=(-2, -1)) / (  # Sum over joints (22) and xyz (3)
            ((~train_batch.visible_joints_mask.bool()).sum(dim=-1) * 3) + 1e-8  # Multiply by 3 for xyz channels
        )  # Result: (b, t)

        
        # Foot skating loss
        foot_indices = [7, 8, 10, 11]  # Indices for foot joints
        foot_positions = pred_joints[..., foot_indices, :]  # (batch, time, 4, 3)
        foot_velocities = foot_positions[:, 1:] - foot_positions[:, :-1]  # (batch, time-1, 4, 3)
        
        # Get foot contacts from x_0_pred
        foot_contacts = x_0.contacts[..., foot_indices]  # (batch, time, 4)
        foot_skating_mask = foot_contacts[:, 1:] * train_batch.mask[:, 1:, None]  # (batch, time-1, 4)

        # Compute foot skating loss for each foot joint
        foot_skating_losses = []
        for i in range(len(foot_indices)):
            foot_loss = weight_and_mask_loss(
                foot_velocities[..., i, :].pow(2),  # (batch, time-1, 3)
                bt_mask=foot_skating_mask[..., i],  # (batch, time-1)
                bt_mask_sum=torch.maximum(
                    torch.sum(foot_skating_mask[..., i]) * 3,  # Multiply by 3 for x,y,z
                    torch.tensor(1, device=device)
                )
            )
            foot_skating_losses.append(foot_loss)
        foot_skating_loss = sum(foot_skating_losses) / len(foot_indices)

        # Velocity loss
        joint_velocities = pred_joints[:, 1:] - pred_joints[:, :-1]  # (batch, time-1, num_joints, 3)
        gt_velocities = train_batch.joints_wrt_world[:, 1:] - train_batch.joints_wrt_world[:, :-1]


        loss_terms: dict[str, Tensor | float] = {
            "betas": weight_and_mask_loss(
                # (b, t, 16)
                (x_0_pred.betas - x_0.betas) ** 2
                # (16,)
                * x_0.betas.new_tensor(self.config.beta_coeff_weights),
            ),
            "body_rotmats": weight_and_mask_loss(
                # Reshape inputs to (batch, time, 21, 3, 3)
                (x_0_pred.body_rotmats - x_0.body_rotmats).reshape(batch, time, -1) ** 2
            ),
            "contacts": weight_and_mask_loss((x_0_pred.contacts.float() - x_0.contacts.float()) ** 2),
            "t_world_root": weight_and_mask_loss(
                (x_0_pred.t_world_root - x_0.t_world_root) ** 2
            ),
            "R_world_root": weight_and_mask_loss(
                (x_0_pred.R_world_root - x_0.R_world_root).reshape(batch, time, -1) ** 2
            ),
            "joints": weight_and_mask_loss(invisible_joint_loss.unsqueeze(-1),
                                           train_batch.mask,
                                           torch.sum(train_batch.mask)),
            "foot_skating": foot_skating_loss,
            "velocity": weight_and_mask_loss(
                (joint_velocities - gt_velocities).reshape(batch, time-1, -1) ** 2,
                bt_mask=train_batch.mask[:, 1:],
                bt_mask_sum=torch.sum(train_batch.mask[:, 1:])
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
                weight_and_mask_loss(
                    (x_0_pred.hand_rotmats - x_0.hand_rotmats).reshape(
                        batch, time, 30 * 3 * 3
                    )
                    ** 2,
                    bt_mask=hand_bt_mask,
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
            # self.log(
            #     "train/hand_motion_proportion",
            #     torch.sum(hand_motion) / batch,
            # )
        else:
            loss_terms["hand_rotmats"] = 0.0


        assert loss_terms.keys() == self.config.loss_weights.keys()

        # Log loss terms.
        for name, term in loss_terms.items():
            log_outputs[f"loss_term/{name}"] = term * self.config.loss_weights[name]

        # Return loss.
        loss = sum([loss_terms[k] for k in loss_terms])
        assert isinstance(loss, Tensor)
        assert loss.shape == ()
        log_outputs["train_loss"] = loss

        return loss, log_outputs
