from typing import Literal, Tuple

import torch
from torch import Tensor
from jaxtyping import Float

from .network import EgoDenoiseTraj

from .middleware.third_party.HybrIK.hybrik.models.layers.smplh.fncsmplh import (
    SmplhModel,
)
from .hand_detection_structs import (
    CorrespondedAriaHandWristPoseDetections,
    CorrespondedHamerDetections,
)

GuidanceMode = Literal["off", "hands", "full"]


def compute_hand_loss(
    body_model: SmplhModel,
    traj: EgoDenoiseTraj,
    Ts_world_cpf: Float[Tensor, "time 7"],
    aria_detections: CorrespondedAriaHandWristPoseDetections,
) -> Tuple[Float[Tensor, ""], dict]:
    """Compute loss between predicted and detected hand positions."""
    # Get predicted hand positions
    smpl_output = body_model(
        global_orient=traj.root_orient,
        body_pose=traj.body_pose,
        left_hand_pose=traj.left_hand_pose,
        right_hand_pose=traj.right_hand_pose,
        transl=traj.root_pos,
    )

    # Extract hand joint positions
    pred_joints = smpl_output.joints
    pred_left_hand = pred_joints[:, body_model.left_hand_idxs]
    pred_right_hand = pred_joints[:, body_model.right_hand_idxs]

    # Compute MSE loss between predicted and detected positions
    loss = 0.0
    metrics = {}

    if aria_detections is not None:
        # Transform predictions to world space
        T_world_cpf = Ts_world_cpf.unsqueeze(1)  # [time, 1, 7]
        pred_left_world = transform_points(pred_left_hand, T_world_cpf)
        pred_right_world = transform_points(pred_right_hand, T_world_cpf)

        # Compute losses for valid detections
        if aria_detections.left_valid is not None:
            left_loss = torch.nn.functional.mse_loss(
                pred_left_world[aria_detections.left_valid],
                aria_detections.left_positions[aria_detections.left_valid],
            )
            loss += left_loss
            metrics["left_hand_loss"] = left_loss.item()

        if aria_detections.right_valid is not None:
            right_loss = torch.nn.functional.mse_loss(
                pred_right_world[aria_detections.right_valid],
                aria_detections.right_positions[aria_detections.right_valid],
            )
            loss += right_loss
            metrics["right_hand_loss"] = right_loss.item()

    return loss, metrics


def compute_full_body_loss(
    body_model: SmplhModel,
    traj: EgoDenoiseTraj,
    Ts_world_cpf: Float[Tensor, "time 7"],
    hamer_detections: CorrespondedHamerDetections,
) -> Tuple[Float[Tensor, ""], dict]:
    """Compute loss between predicted and detected full body poses."""
    # Get predicted body positions
    smpl_output = body_model(
        global_orient=traj.root_orient,
        body_pose=traj.body_pose,
        left_hand_pose=traj.left_hand_pose,
        right_hand_pose=traj.right_hand_pose,
        transl=traj.root_pos,
    )

    pred_joints = smpl_output.joints
    loss = 0.0
    metrics = {}

    if hamer_detections is not None and hamer_detections.joint_valid is not None:
        # Transform predictions to world space
        T_world_cpf = Ts_world_cpf.unsqueeze(1)
        pred_joints_world = transform_points(pred_joints, T_world_cpf)

        # Compute loss for valid joint detections
        joints_loss = torch.nn.functional.mse_loss(
            pred_joints_world[hamer_detections.joint_valid],
            hamer_detections.joint_positions[hamer_detections.joint_valid],
        )
        loss += joints_loss
        metrics["joints_loss"] = joints_loss.item()

    return loss, metrics


def do_guidance_optimization(
    Ts_world_cpf: Float[Tensor, "time 7"],
    traj: EgoDenoiseTraj,
    body_model: SmplhModel,
    guidance_mode: GuidanceMode,
    phase: Literal["inner", "post"],
    hamer_detections: None | CorrespondedHamerDetections,
    aria_detections: None | CorrespondedAriaHandWristPoseDetections,
    num_iters: int = 50,
    learning_rate: float = 0.01,
) -> Tuple[EgoDenoiseTraj, dict]:
    """Optimize trajectory to match detected hand and body positions."""
    if guidance_mode == "off":
        return traj, {}

    # Create optimizable parameters
    params = traj.to_dict()
    optimizable_params = {
        k: torch.nn.Parameter(v, requires_grad=True) for k, v in params.items()
    }

    optimizer = torch.optim.Adam(
        [p for p in optimizable_params.values()],
        lr=learning_rate,
    )

    metrics_history = []

    for i in range(num_iters):
        optimizer.zero_grad()

        # Create trajectory from current parameters
        current_traj = EgoDenoiseTraj(**optimizable_params)

        total_loss = 0.0
        current_metrics = {}

        # Compute losses based on guidance mode
        if guidance_mode in ["hands", "full"]:
            hand_loss, hand_metrics = compute_hand_loss(
                body_model,
                current_traj,
                Ts_world_cpf,
                aria_detections,
            )
            total_loss += hand_loss
            current_metrics.update(hand_metrics)

        if guidance_mode == "full":
            body_loss, body_metrics = compute_full_body_loss(
                body_model,
                current_traj,
                Ts_world_cpf,
                hamer_detections,
            )
            total_loss += body_loss
            current_metrics.update(body_metrics)

        # Optimization step
        total_loss.backward()
        optimizer.step()

        current_metrics["total_loss"] = total_loss.item()
        metrics_history.append(current_metrics)

    # Create final trajectory from optimized parameters
    optimized_traj = EgoDenoiseTraj(
        **{k: v.detach() for k, v in optimizable_params.items()},
    )

    return optimized_traj, {
        "metrics_history": metrics_history,
        "final_metrics": metrics_history[-1],
    }


def transform_points(
    points: Float[Tensor, "... N 3"],
    transforms: Float[Tensor, "... 7"],
) -> Float[Tensor, "... N 3"]:
    """Transform points using SE3 transforms in quaternion format."""
    # Extract quaternion and translation
    quat = transforms[..., :4]
    trans = transforms[..., 4:]

    # Normalize quaternion
    quat = quat / torch.norm(quat, dim=-1, keepdim=True)

    # Convert quaternion to rotation matrix
    R = quaternion_to_matrix(quat)

    # Apply rotation and translation
    transformed_points = torch.matmul(points, R.transpose(-1, -2))
    transformed_points = transformed_points + trans.unsqueeze(-2)

    return transformed_points


def quaternion_to_matrix(
    quaternions: Float[Tensor, "... 4"],
) -> Float[Tensor, "... 3 3"]:
    """Convert quaternions to rotation matrices."""
    # Normalize quaternions
    quaternions = quaternions / torch.norm(quaternions, dim=-1, keepdim=True)

    # Extract components
    w, x, y, z = (
        quaternions[..., 0],
        quaternions[..., 1],
        quaternions[..., 2],
        quaternions[..., 3],
    )

    # Compute rotation matrix elements
    R = torch.stack(
        [
            1 - 2 * y * y - 2 * z * z,
            2 * x * y - 2 * w * z,
            2 * x * z + 2 * w * y,
            2 * x * y + 2 * w * z,
            1 - 2 * x * x - 2 * z * z,
            2 * y * z - 2 * w * x,
            2 * x * z - 2 * w * y,
            2 * y * z + 2 * w * x,
            1 - 2 * x * x - 2 * y * y,
        ],
        dim=-1,
    ).reshape(quaternions.shape[:-1] + (3, 3))

    return R
