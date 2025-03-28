"""Training loss configuration."""

import dataclasses
from typing import Literal
from typing import Union, Tuple
from typing import TYPE_CHECKING
from jaxtyping import Float, Bool, Int

import torch.utils.data
from egoallo.config import CONFIG_FILE
from egoallo.config import make_cfg
from egoallo.utils.setup_logger import setup_logger
from torch import Tensor
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel

if TYPE_CHECKING:
    from egoallo.config.train.train_config import EgoAlloTrainConfig
    from egoallo.types import DenoiseTrajType

from . import network
from .data.dataclass import EgoTrainingData
from .sampling import CosineNoiseScheduleConstants
from .transforms import SO3

local_config_file = CONFIG_FILE
CFG = make_cfg(config_name="defaults", config_file=local_config_file, cli_args=[])

logger = setup_logger(output=None, name=__name__)


@dataclasses.dataclass
class TrainingLossConfig:
    cond_dropout_prob: float = 0.0
    beta_coeff_weights: tuple[float, ...] = tuple(1 / (i + 1) for i in range(16))
    weight_loss_by_t: Literal["emulate_eps_pred"] = "emulate_eps_pred"
    """Weights to apply to the loss at each noise level."""


class TrainingLossComputer:
    """Helper class for computing the training loss."""

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
        return_per_joint: bool = False,
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
            reduction="none",  # Keep per-joint dimension
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
        reduction: str = "mean",
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
        per_step = (
            torch.mean(loss_per_step, dim=-1) if reduction == "mean" else loss_per_step
        )

        # Sum across t axis
        per_batch = torch.sum(per_step * bt_mask, dim=-1)

        # Weight by timestep
        weighted = per_batch * weight_t

        if reduction == "mean":
            # Sum across b axis and normalize
            return torch.sum(weighted) / bt_mask_sum
        else:
            # Return per-batch/dim losses
            return weighted

    def compute_denoising_loss(
        self,
        model: network.EgoDenoiser | DistributedDataParallel | OptimizedModule,
        unwrapped_model: network.EgoDenoiser,
        train_config: "EgoAlloTrainConfig",
        train_batch: EgoTrainingData,
    ) -> tuple[Tensor, dict[str, Tensor | float]]:
        """Compute a training loss for the EgoDenoiser model.

        Returns:
            A tuple (total loss tensor, dictionary of metrics to log).
        """

        log_outputs: dict[str, Tensor | float] = {}

        batch, time, _, _ = train_batch.body_quats.shape

        num_joints = CFG.smplh.num_joints
        assert num_joints == 22
        # Create trajectory using denoising config factory method
        x_0: DenoiseTrajType = train_config.denoising.create_trajectory(
            **train_config.denoising.from_ego_data(
                train_batch,
                include_hands=unwrapped_model.config.include_hands,
            ).__dict__,
        )

        x_0_packed = x_0.pack()
        device = x_0_packed.device
        assert x_0_packed.shape == (
            batch,
            time,
            x_0.get_packed_dim(include_hands=unwrapped_model.config.include_hands),
        )

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
        x_t_unpacked: DenoiseTrajType = train_config.denoising.unpack_traj(
            x_t_packed,
            include_hands=unwrapped_model.config.include_hands,
            project_rotmats=False,
        )  # type: ignore

        # define per-time step weighting scheme and construct weighting function.
        weight_t = self.weight_t[t].to(device)
        assert weight_t.shape == (batch,)

        hand_positions_wrt_cpf: Tensor | None = None
        if unwrapped_model.config.include_hands:
            # Joints 19 and 20 are the hand positions.
            wrist_start_index = 20
            hand_positions_wrt_cpf = train_batch.joints_wrt_cpf[
                :,
                :,
                wrist_start_index : wrist_start_index + 2,
                :,
            ].reshape((batch, time, 6))

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

        x_0_packed_pred = model.forward(
            x_t_unpacked=x_t_unpacked,
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
        assert isinstance(x_0_packed_pred, torch.Tensor) and x_0_packed_pred.shape == (
            batch,
            time,
            x_0.get_packed_dim(include_hands=unwrapped_model.config.include_hands),
        )

        x_0_pred = train_config.denoising.unpack_traj(
            x_0_packed_pred,
            include_hands=unwrapped_model.config.include_hands,
            project_rotmats=False,
        )
        # Compute loss using x_0_pred and x_0

        # postprocessing
        train_batch = train_batch.postprocess()
        x_0_pred = train_batch._post_process(x_0_pred)
        x_0_pred = train_batch._set_traj(x_0_pred)
        x_0 = train_batch._post_process(x_0)
        x_0 = train_batch._set_traj(x_0)

        loss_terms: dict[str, Tensor | float] = x_0_pred.compute_loss(
            other=x_0,
            mask=train_batch.mask,
            weight_t=weight_t,
        )

        # Add joint position loss calculation
        # Get predicted joint positions through forward kinematics
        x_0_pred_posed = x_0_pred.apply_to_body(
            unwrapped_model.body_model.to(device),
        )  # (b, t, 22, 3)
        pred_joints = torch.cat(
            [
                x_0_pred_posed.T_world_root[..., 4:7].unsqueeze(dim=-2),
                x_0_pred_posed.Ts_world_joint[..., : num_joints - 1, 4:7],
            ],
            dim=-2,
        )  # (b, t, 22, 3)
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
        if train_batch.visible_joints_mask is not None:
            invisible_joint_loss = (
                joint_loss
                * (
                    (~train_batch.visible_joints_mask)[..., None]
                )  # Use only invisible joints by inverting the visible joints mask
            ).sum(dim=(-2, -1)) / (  # Sum over joints (22) and xyz (3)
                ((~train_batch.visible_joints_mask).sum(dim=-1) * 3)
                + 1e-8  # Multiply by 3 for xyz channels
            )  # Result: (b, t)
            vis_jnt_loss = (
                joint_loss * ((train_batch.visible_joints_mask)[..., None])
            ).sum(dim=(-2, -1)) / (
                ((train_batch.visible_joints_mask).sum(dim=-1) * 3) + 1e-8
            )
        else:
            logger.warning(
                "No visible joints mask found, using all joints for loss calculation, there should be no scenarios when visible_joints_mask is None",
            )
            invisible_joint_loss = torch.zeros((batch, time), device=device)
            vis_jnt_loss = joint_loss

        # Foot skating loss
        foot_indices = [7, 8, 10, 11]  # Indices for foot joints
        foot_positions = pred_joints[..., foot_indices, :]  # (batch, time, 4, 3)
        foot_velocities = (
            foot_positions[:, 1:] - foot_positions[:, :-1]
        )  # (batch, time-1, 4, 3)

        # Get foot contacts from x_0_pred
        foot_contacts = x_0.contacts[..., foot_indices]  # (batch, time, 4)
        foot_skating_mask = (
            foot_contacts[:, 1:] * train_batch.mask[:, 1:, None]
        ).bool()  # (batch, time-1, 4)

        # Compute foot skating loss for each foot joint
        foot_skating_losses = []
        for i in range(len(foot_indices)):
            foot_loss = x_0_pred._weight_and_mask_loss(
                foot_velocities[..., i, :].pow(2),  # (batch, time-1, 3)
                bt_mask=foot_skating_mask[..., i],  # (batch, time-1)
                weight_t=weight_t,
                bt_mask_sum=torch.maximum(
                    torch.sum(foot_skating_mask[..., i]) * 3,  # Multiply by 3 for x,y,z
                    torch.tensor(1, device=device),
                ),
            )
            foot_skating_losses.append(foot_loss)

        # Average foot skating losses
        foot_skating_loss = torch.stack(foot_skating_losses).mean()

        # Velocity loss
        joint_velocities = (
            pred_joints[:, 1:] - pred_joints[:, :-1]
        )  # (batch, time-1, num_joints, 3)
        gt_velocities = (
            train_batch.joints_wrt_world[:, 1:] - train_batch.joints_wrt_world[:, :-1]
        )

        loss_terms.update(
            {
                # empirically, invisible joints loss should be more important than visible joints loss.
                "joints": x_0_pred._weight_and_mask_loss(
                    invisible_joint_loss.unsqueeze(-1),
                    train_batch.mask,
                    weight_t,
                    torch.sum(train_batch.mask),
                )
                + x_0_pred._weight_and_mask_loss(
                    vis_jnt_loss.unsqueeze(-1),
                    train_batch.mask,
                    weight_t,
                    torch.sum(train_batch.mask),
                ),
                "foot_skating": foot_skating_loss,
                "velocity": x_0_pred._weight_and_mask_loss(
                    ((joint_velocities - gt_velocities) ** 2).reshape(
                        batch,
                        time - 1,
                        -1,
                    ),
                    train_batch.mask[:, 1:],
                    weight_t,
                    torch.sum(train_batch.mask[:, 1:]),
                ),
            },
        )

        # Include hand objective.
        # We didn't use this in the paper.
        # TODO: hand-rotmats loss is incorporated in the network.py DenoiseTraj class, keep it here for reference.
        # if unwrapped_model.config.include_hands:
        #     assert x_0_pred.hand_rotmats is not None
        #     assert x_0.hand_rotmats is not None
        #     assert x_0.hand_rotmats.shape == (batch, time, 30, 3, 3)

        #     # Detect whether or not hands move in a sequence.
        #     # We should only supervise sequences where the hands are actully tracked / move;
        #     # we mask out hands in AMASS sequences where they are not tracked.
        #     gt_hand_flatmat = x_0.hand_rotmats.reshape((batch, time, -1))
        #     hand_motion = (
        #         torch.sum(  # (b,) from (b, t)
        #             torch.sum(  # (b, t) from (b, t, d)
        #                 torch.abs(gt_hand_flatmat - gt_hand_flatmat[:, 0:1, :]), dim=-1
        #             )
        #             # Zero out changes in masked frames.
        #             * train_batch.mask,
        #             dim=-1,
        #         )
        #         > 1e-5
        #     )
        #     assert hand_motion.shape == (batch,)

        #     hand_bt_mask = torch.logical_and(hand_motion[:, None], train_batch.mask)
        #     loss_terms["hand_rotmats"] = torch.sum(
        #         weight_and_mask_loss(
        #             (x_0_pred.hand_rotmats - x_0.hand_rotmats).reshape(
        #                 batch, time, 30 * 3 * 3
        #             )
        #             ** 2,
        #             bt_mask=hand_bt_mask,
        #             # We want to weight the loss by the number of frames where
        #             # the hands actually move, but gradients here can be too
        #             # noisy and put NaNs into mixed-precision training when we
        #             # inevitably sample too few frames. So we clip the
        #             # denominator.
        #             bt_mask_sum=torch.maximum(
        #                 torch.sum(hand_bt_mask), torch.tensor(256, device=device)
        #             ),
        #         )
        #     )
        #     # self.log(
        #     #     "train/hand_motion_proportion",
        #     #     torch.sum(hand_motion) / batch,
        #     # )
        # else:
        #     loss_terms["hand_rotmats"] = 0.0

        assert all(
            k in train_config.denoising.loss_weights.keys() for k in loss_terms.keys()
        ), (
            f"Missing loss weights for terms: {set(loss_terms.keys()) - set(train_config.denoising.loss_weights.keys())}"
        )
        # Log loss terms.
        for name, term in loss_terms.items():
            loss_term = term * train_config.denoising.loss_weights[name]
            loss_terms[name] = loss_term
            log_outputs[f"loss_term/{name}"] = loss_term

        # Return loss.
        loss = sum([loss_terms[k] for k in loss_terms])
        assert isinstance(loss, Tensor)
        assert loss.shape == ()
        log_outputs["train_loss"] = loss

        return loss, log_outputs
