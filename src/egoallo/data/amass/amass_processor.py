"""AMASS dataset processor using functional SMPL-H implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
import torch
import typeguard
from egoallo.data.motion_processing import MotionProcessor
from egoallo.fncsmpl import SmplhModel
from egoallo.fncsmpl import SmplhShaped
from egoallo.fncsmpl import SmplhShapedAndPosed
from egoallo.transforms import SE3
from egoallo.transforms import SO3
from egoallo.utils.setup_logger import setup_logger
from jaxtyping import Float
from jaxtyping import jaxtyped
from numpy import ndarray as Array
from torch import Tensor

logger = setup_logger(output="logs/amass_processor", name=__name__)


class AMASSProcessor:
    """Process AMASS dataset sequences using functional SMPL-H implementation."""

    def __init__(
        self,
        amass_dir: str,
        smplh_dir: str,
        output_dir: str,
        fps: int = 30,
        include_velocities: bool = True,
        include_align_rot: bool = True,
        device: str = "cuda",
    ) -> None:
        """Initialize AMASS processor.

        Args:
            amass_dir: Path to AMASS dataset root
            smplh_dir: Path to SMPL model files
            output_dir: Output directory for processed sequences
            fps: Target frames per second
            include_velocities: Whether to compute velocities
            include_align_rot: Whether to compute alignment rotations
            device: Device to use for processing
        """
        self.amass_dir = Path(amass_dir)
        self.smplh_dir = Path(smplh_dir)
        self.output_dir = Path(output_dir)
        self.target_fps = fps
        self.include_velocities = include_velocities
        self.include_align_rot = include_align_rot
        self.device = device

        # Initialize motion processor
        self.motion_processor = MotionProcessor()

        # Joint indices for SMPL model
        self.joint_indices = {
            # TODO: change thoes joints indices with reference to mapping.py.
            "left_knee": 4,
            "right_knee": 5,
            "left_ankle": 7,
            "right_ankle": 8,
            "left_foot": 10,
            "right_foot": 11,
            "left_elbow": 18,
            "right_elbow": 19,
            "left_wrist": 20,
            "right_wrist": 21,
        }

        # Load SMPL-H models for each gender
        self.body_models = {}
        for gender in ["male", "female", "neutral"]:
            model_path = self.smplh_dir / f"{gender}/model.npz"
            self.body_models[gender] = SmplhModel.load(model_path)

    @jaxtyped(typechecker=typeguard.typechecked)
    def _convert_rotations(
        self,
        root_orient: Float[Tensor, "... 3"],
        body_pose: Float[Tensor, "... 63"],
        hand_pose: Float[Tensor, "... 90"],
        trans: Float[Tensor, "... 3"],
    ) -> tuple[
        Float[Tensor, "... 7"],
        Float[Tensor, "... 21 4"],
        Float[Tensor, "... 15 4"],
        Float[Tensor, "... 15 4"],
    ]:
        """Convert rotation representations."""
        # Convert root orientation and translation to SE(3)
        T_world_root = SE3.from_rotation_and_translation(
            rotation=SO3.exp(root_orient),
            translation=trans,
        ).parameters()  # (..., 7)

        # Convert body pose to quaternions (21 joints)
        body_rots = body_pose.reshape(*body_pose.shape[:-1], 21, 3)
        body_quats = SO3.exp(body_rots).wxyz  # (..., 21, 4)

        # Convert hand poses to quaternions (15 joints each)
        left_hand_rots = hand_pose[..., :45].reshape(*hand_pose.shape[:-1], 15, 3)
        right_hand_rots = hand_pose[..., 45:].reshape(*hand_pose.shape[:-1], 15, 3)
        left_hand_quats = SO3.exp(left_hand_rots).wxyz  # (..., 15, 4)
        right_hand_quats = SO3.exp(right_hand_rots).wxyz  # (..., 15, 4)

        return T_world_root, body_quats, left_hand_quats, right_hand_quats

    def process_sequence(
        self,
        seq_path: Path,
        min_frames: int = 30,
    ) -> Optional[Dict[str, Any]]:
        """Process a single AMASS sequence."""
        required_keys = ["mocap_framerate", "poses", "betas", "trans"]
        # Load sequence data
        seq_data = dict(np.load(seq_path, allow_pickle=True))

        for required_key in required_keys:
            if required_key not in seq_data.keys():
                return None

        # Get sequence info and handle mislabeled data
        gender = seq_data.get("gender", "invalid").item()
        if isinstance(gender, bytes):
            gender = gender.decode("utf-8")
        else:
            gender = str(gender)

        # Correct mislabeled framerates
        fps = int(seq_data["mocap_framerate"])
        if "BMLhandball" in str(seq_path):
            fps = 240
        if "20160930_50032" in str(seq_path) or "20161014_50033" in str(seq_path):
            fps = 59

        # Get poses and shape
        poses = torch.from_numpy(seq_data["poses"]).float().to(self.device)  # (N, 156)
        trans = torch.from_numpy(seq_data["trans"]).float().to(self.device)  # (N, 3)
        betas = (
            torch.from_numpy(seq_data["betas"][:16]).float().to(self.device)
        )  # (16,)

        # Trim sequence to middle 80% to avoid redundant static poses
        num_frames = len(poses)
        start_idx = int(0.1 * num_frames)
        end_idx = int(0.9 * num_frames)
        poses = poses[start_idx:end_idx]
        trans = trans[start_idx:end_idx]
        num_frames = end_idx - start_idx

        if num_frames < min_frames:
            logger.warning(f"Sequence too short: {num_frames} frames")
            return None

        # Resample if target FPS is different from source FPS
        assert self.target_fps <= fps, (
            f"target_fps: {self.target_fps}, fps: {fps}, seq_path: {seq_path}"
        )
        if self.target_fps != fps and self.target_fps < fps:
            fps_ratio = float(self.target_fps) / fps
            new_num_frames = int(fps_ratio * num_frames)
            downsamp_inds = np.linspace(
                0,
                num_frames - 1,
                num=new_num_frames,
                dtype=int,
            )

            poses = poses[downsamp_inds]
            trans = trans[downsamp_inds]
            num_frames = new_num_frames
            fps = self.target_fps

        # Split pose parameters
        root_orient = poses[:, :3]  # (N, 3)
        body_pose = poses[:, 3:66]  # (N, 63)
        hand_pose = poses[:, 66:]  # (N, 90)

        # Convert rotations to required format
        (
            T_world_root,
            body_quats,
            left_hand_quats,
            right_hand_quats,
        ) = self._convert_rotations(root_orient, body_pose, hand_pose, trans)

        # Process through SMPL-H pipeline
        body_model: SmplhModel = self.body_models[gender].to(self.device)
        shaped: SmplhShaped = body_model.with_shape(betas[None])  # Add batch dim
        posed: SmplhShapedAndPosed = shaped.with_pose_decomposed(
            T_world_root=T_world_root,
            body_quats=body_quats,
            left_hand_quats=left_hand_quats,
            right_hand_quats=right_hand_quats,
        )
        # mesh: SmplMesh = posed.lbs()

        # Extract joint positions (22 SMPL-H joints)
        joints = (
            torch.cat(
                [
                    posed.T_world_root[..., None, 4:7],  # Root position
                    posed.Ts_world_joint[..., :21, 4:7],  # Other joint positions
                ],
                dim=-2,
            )
            .detach()
            .cpu()
            .numpy()
        )  # (N, 22, 3)

        assert joints.ndim == 3 and joints.shape[-2:] == (
            22,
            3,
        ), f"joints shape is {joints.shape}"

        # Process floor height and contacts
        floor_height, contacts = self.motion_processor.process_floor_and_contacts(
            joints,
            self.joint_indices,
        )
        contacts: Float[Array, "*batch timesteps 22"] = contacts[..., :22]

        # Adjust heights
        trans[:, 2] -= floor_height
        joints[..., 2] -= floor_height

        # Compute velociti

        # Adjust heights
        trans[:, 2] -= floor_height
        joints[..., 2] -= floor_height

        # Compute velocities if requested
        velocities = None
        if self.include_velocities:
            dt = 1.0 / fps
            joint_vel = np.stack(
                [
                    self.motion_processor.compute_joint_velocity(joints[:, i])
                    for i in range(joints.shape[1])
                ],
                axis=1,
            )

            trans_vel = self.motion_processor.compute_joint_velocity(
                trans.cpu().numpy(),
            )

            root_orient_mat = SO3.exp(root_orient).as_matrix().detach().cpu().numpy()
            root_ang_vel = self.motion_processor.compute_angular_velocity(
                root_orient_mat,
                dt,
            )

            velocities = {
                "joints": joint_vel,
                "trans": trans_vel,
                "root_orient": root_ang_vel,
            }

        # Compute alignment rotations if requested
        align_rot = None
        if self.include_align_rot:
            forward_dir = joints[:, 1] - joints[:, 0]  # Pelvis to spine
            forward_dir = forward_dir / np.linalg.norm(forward_dir)
            align_rot = self.motion_processor.compute_alignment_rotation(forward_dir)

        # Prepare output data
        sequence_data = {
            "poses": poses.cpu().numpy(),
            "trans": trans.cpu().numpy(),
            "betas": betas.cpu().numpy(),
            "gender": gender,
            "fps": fps,
            "joints": joints,
            "contacts": contacts.astype(
                np.float32,
            ),  # contacts server as a boolean label, but for compatiblity with `load_from_npz` function, convert it to flaot32
            "pose_hand": hand_pose.cpu().numpy(),
            "root_orient": root_orient.cpu().numpy(),
            "pose_body": body_pose.cpu().numpy(),
        }

        if velocities is not None:
            sequence_data["velocities"] = velocities
        if align_rot is not None:
            sequence_data["align_rot"] = align_rot

        return sequence_data

    def save_sequence(self, sequence_data: Dict[str, Any], output_path: Path) -> None:
        """Save processed sequence data."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **sequence_data)
        logger.info(f"Saved processed sequence to {output_path}")
