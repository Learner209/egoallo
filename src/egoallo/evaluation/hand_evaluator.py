import json
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
import yaml

from egoallo.constants import EGOEXO_NAMES_LEFT
from egoallo.constants import EGOEXO_NAMES_RIGHT
from egoallo.constants import VERTEX_IDS
from egoallo.type_stubs import EvalMode
from egoallo.type_stubs import FloatArray
from egoallo.type_stubs import HandSide
from egoallo.type_stubs import MetricsDict
from egoallo.type_stubs import ModelType
from egoallo.type_stubs import PathLike
from egoallo.type_stubs import ProcrustesMode
from egoallo.type_stubs import ProcrustesOutput
from egoallo.utilities import procrustes_align
from egoallo.utils.setup_logger import setup_logger
import typeguard
from jaxtyping import jaxtyped

from .base import BaseEvaluator


logger = setup_logger(output="logs/evaluation", name=__name__)


class HandEvaluator(BaseEvaluator):
    """Evaluates hand pose metrics by comparing predicted keypoints with ground truth annotations."""

    def __init__(
        self,
        egoexo_dir: PathLike,
        egoexo_reorg_dir: PathLike,
        body_npz_path: PathLike,
        device: Optional[torch.device] = None,
    ):
        """Initialize the HandEvaluator."""
        super().__init__(body_npz_path, device)
        self.egoexo_dir = Path(egoexo_dir)
        self.egoexo_reorg_dir = Path(egoexo_reorg_dir)

    @staticmethod
    def get_mano_from_openpose_indices(include_tips: bool = True) -> FloatArray:
        """Get indices mapping from MANO joints to OpenPose format."""
        mano_to_openpose = [
            0,
            13,
            14,
            15,
            16,
            1,
            2,
            3,
            17,
            4,
            5,
            6,
            18,
            10,
            11,
            12,
            19,
            7,
            8,
            9,
            20,
        ]
        if not include_tips:
            mano_to_openpose = mano_to_openpose[:16]
        openpose_from_mano_idx = {
            mano_idx: openpose_idx
            for openpose_idx, mano_idx in enumerate(mano_to_openpose)
        }
        indices = np.array(
            [openpose_from_mano_idx[i] for i in range(len(mano_to_openpose))],
        )
        return indices

    def tips_from_vertices(
        self,
        vertices: FloatArray,
        model_type: ModelType,
        side: HandSide,
    ) -> FloatArray:
        """Select finger tips from SMPL vertices."""
        side_short = "" if model_type == "mano" else side[0]
        tip_names = ["thumb", "index", "middle", "ring", "pinky"]
        tips_idxs = [
            VERTEX_IDS[model_type][side_short + tip_name] for tip_name in tip_names
        ]
        finger_tips = vertices[..., tips_idxs, :]
        return finger_tips

    @lru_cache(maxsize=None)
    def load_mano_annotations(self, gt_joints_path: Path) -> Dict[str, FloatArray]:
        """Load ground truth MANO annotations from JSON."""
        with gt_joints_path.open("r") as f:
            anno = json.load(f)

        frames = np.array(list(anno.keys())).astype(int)

        def get_keypoints(order, frame_data):
            keypoints, mask = [], []
            for joint in order:
                cc = frame_data.get(joint, None)
                if cc is not None:
                    keypoints.append([cc["x"], cc["y"], cc["z"]])
                    mask.append(True)
                else:
                    keypoints.append([0, 0, 0])
                    mask.append(False)
            return np.array(keypoints), np.array(mask)

        def batch_get_keypoints(side: str):
            names = EGOEXO_NAMES_LEFT if side == "left" else EGOEXO_NAMES_RIGHT
            hand_kpts, hand_masks = [], []
            for frame in frames:
                frame_data = anno[str(frame)][0]["annotation3D"]
                kpts, mask = get_keypoints(names, frame_data)
                hand_kpts.append(kpts)
                hand_masks.append(mask)
            return np.array(hand_kpts), np.array(hand_masks)

        left_kpts, left_masks = batch_get_keypoints("left")
        right_kpts, right_masks = batch_get_keypoints("right")

        return {
            "frames": frames,
            "left_kpts": left_kpts,
            "left_mask": left_masks,
            "right_kpts": right_kpts,
            "right_mask": right_masks,
        }

    def aligned_subtract(
        self,
        a: FloatArray,
        b: FloatArray,
    ) -> FloatArray:
        """Subtract two sets of keypoints after Procrustes alignment."""
        if a.shape[0] == 1:
            return np.zeros_like(a)

        a_tensor = torch.tensor(a, dtype=torch.float32, device=self.device)
        b_tensor = torch.tensor(b, dtype=torch.float32, device=self.device)

        aligned_b = (
            procrustes_align(
                points_y=a_tensor,
                points_x=b_tensor,
                fix_scale=False,
            )[0]
            .cpu()
            .numpy()
        )

        return a - aligned_b

    @jaxtyped(typechecker=typeguard.typechecked)
    def procrustes_align(
        self,
        points_y: Float[Tensor, "*batch time 3"],
        points_x: Float[Tensor, "*batch time 3"],
        output: ProcrustesMode,
        fix_scale: bool = False,
    ) -> ProcrustesOutput:
        """Perform Procrustes alignment between point sets."""
        s, R, t = procrustes_align(points_y, points_x, fix_scale)

        if output == "transforms":
            return s, R, t
        elif output == "aligned_x":
            aligned_x = (
                s[..., None, :] * torch.einsum("...ij,...nj->...ni", R, points_x)
                + t[..., None, :]
            )
            return aligned_x

    def evaluate_take(
        self,
        take_index: int,
        eval_mode: EvalMode,
        hamer_frames_only: bool,
    ) -> Tuple[
        int,
        Dict[str, Dict[int, Tuple[float, ...]]],
        Dict[str, Dict[int, Tuple[float, ...]]],
        int,
        int,
    ]:
        """Evaluate a single take."""
        mpjpes = {}
        pampjpes = {}
        matched_kp = 0
        total_kp = 0

        # Load ground truth data
        gt_path = self.egoexo_dir / f"take{take_index:03d}" / "gt_joints.json"
        if not gt_path.exists():
            return take_index, mpjpes, pampjpes, matched_kp, total_kp

        gt_data = self.load_mano_annotations(gt_path)
        frames = gt_data["frames"]

        # Process each frame
        for frame in frames:
            for side in ["left", "right"]:
                gt_kpts = gt_data[f"{side}_kpts"][frame]
                gt_mask = gt_data[f"{side}_mask"][frame]

                if not gt_mask.any():
                    continue

                # Get predicted keypoints based on eval mode
                pred_kpts = self._get_predicted_keypoints(
                    take_index,
                    frame,
                    side,
                    eval_mode,
                    hamer_frames_only,
                )
                if pred_kpts is None:
                    continue

                # Compute metrics
                mpjpe = np.linalg.norm(gt_kpts - pred_kpts, axis=1) * 1000.0
                pampjpe = (
                    np.linalg.norm(self.aligned_subtract(gt_kpts, pred_kpts), axis=1)
                    * 1000.0
                )

                # Store results
                if take_index not in mpjpes:
                    mpjpes[take_index] = {}
                    pampjpes[take_index] = {}

                mpjpes[take_index][frame] = tuple(mpjpe[gt_mask])
                pampjpes[take_index][frame] = tuple(pampjpe[gt_mask])

                matched_kp += gt_mask.sum()
                total_kp += len(gt_mask)

        return take_index, mpjpes, pampjpes, matched_kp, total_kp

    def evaluate(
        self,
        results_save_path: Path,
        results_save_path_all: Path,
        eval_modes: List[EvalMode],
        hamer_frames_only_options: List[bool],
        write: bool = True,
        num_workers: int = 1,
    ) -> None:
        """Evaluate hand pose metrics for different evaluation modes."""
        stats_from_exp = {}
        stats_from_exp_all = {}

        for eval_mode in eval_modes:
            for hamer_frames_only in hamer_frames_only_options:
                exp = f"{eval_mode=}-{hamer_frames_only=}"
                logger.info(f"Processing experiment: {exp}")

                mpjpes = {}
                pampjpes = {}
                matched_kp = 0
                total_kp = 0

                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [
                        executor.submit(
                            self.evaluate_take,
                            take_index,
                            eval_mode,
                            hamer_frames_only,
                        )
                        for take_index in range(65)
                    ]

                    for future in as_completed(futures):
                        result = future.result()
                        take_index = result[0]
                        mpjpes[take_index] = result[1]
                        pampjpes[take_index] = result[2]
                        matched_kp += result[3]
                        total_kp += result[4]

                # Aggregate results
                mpjpe_values = []
                pampjpe_values = []
                for take_results in mpjpes.values():
                    for side_results in take_results.values():
                        for frame_values in side_results.values():
                            mpjpe_values.extend(frame_values)
                for take_results in pampjpes.values():
                    for side_results in take_results.values():
                        for frame_values in side_results.values():
                            pampjpe_values.extend(frame_values)

                stats = {
                    "mpjpe": float(np.mean(mpjpe_values)),
                    "mpjpe_stderr": float(
                        np.std(mpjpe_values) / np.sqrt(len(mpjpe_values)),
                    ),
                    "pampjpe": float(np.mean(pampjpe_values)),
                    "pampjpe_stderr": float(
                        np.std(pampjpe_values) / np.sqrt(len(pampjpe_values)),
                    ),
                    "matched": matched_kp,
                    "total": total_kp,
                }
                stats_all = {
                    "mpjpe": mpjpes,
                    "pampjpe": pampjpes,
                    "matched": matched_kp,
                    "total": total_kp,
                }

                logger.info(f"Experiment results for {exp}:")
                logger.info(stats)

                stats_from_exp[exp] = stats
                stats_from_exp_all[exp] = stats_all

                if write:
                    results_save_path.write_text(yaml.dump(stats_from_exp))
                    results_save_path_all.write_text(yaml.dump(stats_from_exp_all))

    def _get_predicted_keypoints(
        self,
        take_index: int,
        frame: int,
        side: str,
        eval_mode: EvalMode,
        hamer_frames_only: bool,
    ) -> Optional[FloatArray]:
        """Get predicted keypoints based on evaluation mode."""
        # Implementation depends on specific prediction method
        # This is a placeholder - actual implementation would load predictions
        return None

    def evaluate_directory(self, *args, **kwargs) -> None:
        """Not implemented for HandEvaluator."""
        raise NotImplementedError("HandEvaluator does not support directory evaluation")

    def process_file(self, *args, **kwargs) -> MetricsDict:
        """Not implemented for HandEvaluator."""
        raise NotImplementedError("HandEvaluator does not support file processing")
