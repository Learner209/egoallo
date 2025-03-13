"""HPS dataset processor using functional SMPL-H implementation."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
import torch
import typeguard
from egoallo.data.motion_processing import MotionProcessor

# from egoallo.fncsmpl import SmplhModel
from egoallo.fncsmpl_library import SmplhModel
from egoallo.transforms import SE3
from egoallo.transforms import SO3
from egoallo.utils.setup_logger import setup_logger
from jaxtyping import Float
from jaxtyping import jaxtyped
from numpy import ndarray as Array
from torch import Tensor

logger = setup_logger(output="logs/hps_processor", name=__name__)


class HPSProcessor:
    """Process HPS dataset sequences using functional SMPL-H."""

    def __init__(
        self,
        hps_dir: str,
        smplh_dir: str,
        output_dir: str,
        fps: int = 30,
        include_hands: bool = True,
        device: str = "cuda",
    ) -> None:
        self.hps_dir = Path(hps_dir)
        self.smplh_dir = Path(smplh_dir)
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.include_hands = include_hands
        self.device = device

        # Initialize motion processor
        self.motion_processor = MotionProcessor()

        # Load SMPL-H models for each gender
        self.body_models = {}
        for gender in ["male", "female", "neutral"]:
            model_path = self.smplh_dir / f"SMPLH_{gender.upper()}.pkl"
            self.body_models[gender] = SmplhModel.load(model_path, use_pca=False).to(
                self.device,
            )

        # Joint indices for contact detection
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

    def _load_camera_trajectory(
        self,
        sequence_name: str,
    ) -> tuple[list[dict[str, float]], float, float]:
        """Load camera trajectory from HPS format."""
        camera_path = (
            self.hps_dir / "head_camera_localizations" / f"{sequence_name}.json"
        )
        with open(camera_path, "r") as f:
            trajectory_data = json.load(f)

        trajectory_data = {
            int(k): {**v, "time": float(k) / self.fps}
            for k, v in trajectory_data.items()
            if v is not None
        }

        first_key, last_key = min(trajectory_data.keys()), max(trajectory_data.keys())
        return (
            list(trajectory_data.values()),
            trajectory_data[first_key]["time"],
            trajectory_data[last_key]["time"],
        )

    @jaxtyped(typechecker=typeguard.typechecked)
    def _convert_rotations(
        self,
        root_orient: Float[Tensor, "... 3"],
        body_pose: Float[Tensor, "... 63"],
        trans: Float[Tensor, "... 3"],
    ) -> tuple[
        Float[Tensor, "... 7"],
        Float[Tensor, "... 21 4"],
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

        return T_world_root, body_quats

    def validate_sequence(self, sequence_name: str) -> bool:
        """Validate that all required files exist for a sequence."""
        # HPS dataset has a few sequences starting with "Double" that may interfere with the subject name.
        if sequence_name.startswith("Double"):
            subject = sequence_name.split("_")[1]
        else:
            subject = sequence_name.split("_")[0]
        paths = [
            self.hps_dir / "hps_smpl" / f"{sequence_name}.pkl",
            self.hps_dir / "hps_betas" / f"{subject}.json",
            self.hps_dir / "head_camera_localizations" / f"{sequence_name}.json",
        ]
        return all(p.exists() for p in paths)

    def process_sequence(
        self,
        sequence_name: str,
        min_frames: int = 30,
    ) -> Optional[Dict[str, Any]]:
        """Process a single HPS sequence."""
        if not self.validate_sequence(sequence_name):
            logger.warning(f"Invalid sequence: {sequence_name}")
            return None

        # Load sequence data
        seq_path = self.hps_dir / "hps_smpl" / f"{sequence_name}.pkl"
        with open(seq_path, "rb") as f:
            seq_data = pickle.load(f)

        # Load subject betas
        if sequence_name.startswith("Double"):
            subject = sequence_name.split("_")[1]
        else:
            subject = sequence_name.split("_")[0]

        betas = (
            torch.from_numpy(
                np.array(
                    json.load(open(self.hps_dir / "hps_betas" / f"{subject}.json"))[
                        "betas"
                    ],
                ),
            )
            .float()
            .to(self.device)
        )
        assert betas.dim() == 1 and betas.shape[-1] == 10, (
            f"betas shape is {betas.shape}"
        )  # HPS dataset betas shape is (10,)

        # Convert sequence data to tensors
        poses = torch.from_numpy(seq_data["poses"]).float().to(self.device)
        trans = torch.from_numpy(seq_data["transes"]).float().to(self.device)

        num_frames = len(poses)
        if num_frames < min_frames:
            logger.warning(f"Sequence too short: {num_frames} frames")
            return None

        # Split pose parameters - HPS dataset only has root and body poses
        root_orient = poses[:, :3]  # (N, 3)
        body_pose = poses[:, 3:66]  # (N, 63)
        hand_pose = torch.zeros(poses.shape[0], 90).to(self.device)

        # Convert rotations to required format
        T_world_root, body_quats = self._convert_rotations(
            root_orient,
            body_pose,
            trans,
        )

        # Process through SMPL-H pipeline
        gender = "male"  # Get from metadata if available
        body_model = self.body_models[gender]
        shaped = body_model.with_shape(betas[None])
        posed = shaped.with_pose_decomposed(
            T_world_root=T_world_root,
            body_quats=body_quats,
            left_hand_quats=None,  # No hand poses in HPS dataset
            right_hand_quats=None,  # No hand poses in HPS dataset
        )

        # Extract joint positions
        joints = (
            torch.cat(
                [
                    posed.T_world_root[..., None, 4:7],
                    posed.Ts_world_joint[..., :21, 4:7],  # discard the hand joints.
                ],
                dim=-2,
            )
            .detach()
            .cpu()
            .numpy()
        )

        assert joints.ndim == 3 and joints.shape[-2:] == (
            22,
            3,
        ), f"joints shape is {joints.shape}"

        # Process floor height and contacts
        # floor_height = self.motion_processor.detect_floor_height(
        #     joints,
        #     list(self.joint_indices.values())
        # )

        floor_height, contacts = self.motion_processor.process_floor_and_contacts(
            joints,
            self.joint_indices,
        )
        contacts: Float[Array, "*batch timesteps 22"] = contacts[..., :22]

        # Adjust heights
        trans[:, 2] -= floor_height
        joints[..., 2] -= floor_height

        # Load camera trajectory
        camera_traj, start_time, end_time = self._load_camera_trajectory(sequence_name)

        # Prepare sequence data
        sequence_data = {
            "poses": poses.cpu().numpy(force=True),
            "trans": trans.cpu().numpy(force=True),
            "betas": betas.cpu().numpy(force=True),
            "contacts": contacts.astype(
                np.float32,
            ),  # contacts server as a boolean label, but for compatiblity with `load_from_npz` function, convert it to flaot32
            "gender": gender,
            "fps": self.fps,
            "joints": joints,
            "camera_trajectory": camera_traj,
            "time_range": (start_time, end_time),
            "root_orient": root_orient.cpu().numpy(force=True),
            "pose_body": body_pose.cpu().numpy(force=True),
            "pose_hand": hand_pose.cpu().numpy(force=True),
        }

        return sequence_data

    def save_sequence(self, sequence_data: Dict[str, Any], output_path: Path) -> None:
        """Save processed sequence data."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **sequence_data)
        logger.info(f"Saved processed sequence to {output_path}")
