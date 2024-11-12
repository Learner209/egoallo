from __future__ import annotations

import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union, overload

import numpy as np
import torch
import tyro
import yaml
from jaxtyping import Float
from torch import Tensor
from tqdm.auto import tqdm
from typing_extensions import assert_never

from egoallo import fncsmpl
from egoallo.eval_structs import load_relevant_outputs
from egoallo.transforms import SO3


class BodyEvaluator:
    """
    Evaluates body pose metrics between predicted and ground truth data.
    """

    def __init__(
        self,
        body_model_path: Path,
        device: Optional[torch.device] = None,
    ):
        """
        Initializes the BodyEvaluator.

        Args:
            body_model_path: Path to the SMPL body model (.npz file).
            device: Torch device to use (e.g., 'cpu' or 'cuda').
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.body_model = fncsmpl.SmplhModel.load(body_model_path).to(self.device)

    def compute_foot_skate(
        self,
        pred_Ts_world_joint: Float[Tensor, "num_samples time 21 7"],
    ) -> np.ndarray:
        """
        Computes the foot skate metric for predicted joint transforms.

        Args:
            pred_Ts_world_joint: Predicted world joint transforms (num_samples, time, 21, 7).

        Returns:
            Foot skate metric per sample as a NumPy array (num_samples,).
        """
        num_samples, time = pred_Ts_world_joint.shape[:2]

        # Adjust the person's position to the floor.
        pred_Ts_world_joint = pred_Ts_world_joint.clone()
        pred_Ts_world_joint[..., 6] -= torch.min(pred_Ts_world_joint[..., 6])

        foot_indices = torch.tensor([6, 7, 9, 10], device=self.device)

        foot_positions = pred_Ts_world_joint[:, :, foot_indices, 4:7]
        foot_positions_diff = foot_positions[:, 1:, :, :2] - foot_positions[:, :-1, :, :2]
        foot_positions_diff_norm = torch.sum(torch.abs(foot_positions_diff), dim=-1)

        # Thresholds from EgoEgo / kinpoly.
        H_thresh = torch.tensor(
            [0.08, 0.08, 0.04, 0.04],
            device=self.device,
            dtype=torch.float32,
        )

        foot_contact = foot_positions[:, 1:, :, 2] < H_thresh
        foot_positions_diff_norm *= foot_contact

        exponent = 2 - 2 ** (foot_positions[:, 1:, :, 2] / H_thresh)
        fs_per_sample = torch.sum(torch.sum(foot_positions_diff_norm * exponent, dim=-1), dim=-1)

        return fs_per_sample.cpu().numpy()

    def compute_foot_contact(
        self,
        pred_Ts_world_joint: Float[Tensor, "num_samples time 21 7"],
    ) -> np.ndarray:
        """
        Computes the foot contact metric for predicted joint transforms.

        Args:
            pred_Ts_world_joint: Predicted world joint transforms (num_samples, time, 21, 7).

        Returns:
            Foot contact metric per sample as a NumPy array (num_samples,).
        """
        num_samples = pred_Ts_world_joint.shape[0]

        foot_indices = torch.tensor([6, 7, 9, 10], device=self.device)

        # Thresholds from EgoEgo / kinpoly.
        H_thresh = torch.tensor(
            [0.08, 0.08, 0.04, 0.04],
            device=self.device,
            dtype=torch.float32,
        )

        foot_positions = pred_Ts_world_joint[:, :, foot_indices, 4:7]
        any_contact = torch.any(
            torch.any(foot_positions[..., 2] < H_thresh, dim=-1), dim=-1
        ).to(torch.float32)

        return any_contact.cpu().numpy()

    def compute_head_ori(
        self,
        label_Ts_world_joint: Float[Tensor, "time 21 7"],
        pred_Ts_world_joint: Float[Tensor, "num_samples time 21 7"],
    ) -> np.ndarray:
        """
        Computes the head orientation error between predicted and ground truth joints.

        Args:
            label_Ts_world_joint: Ground truth world joint transforms (time, 21, 7).
            pred_Ts_world_joint: Predicted world joint transforms (num_samples, time, 21, 7).

        Returns:
            Head orientation error per sample as a NumPy array (num_samples,).
        """
        num_samples, time = pred_Ts_world_joint.shape[:2]
        pred_head_rot = SO3(pred_Ts_world_joint[:, :, 14, :4]).as_matrix()
        label_head_rot = SO3(label_Ts_world_joint[:, 14, :4]).inverse().as_matrix()

        matrix_errors = (pred_head_rot @ label_head_rot) - torch.eye(3, device=self.device)
        errors = torch.linalg.norm(matrix_errors.reshape((num_samples, time, 9)), dim=-1)
        mean_errors = torch.mean(errors, dim=-1)

        return mean_errors.cpu().numpy()

    def compute_head_trans(
        self,
        label_Ts_world_joint: Float[Tensor, "time 21 7"],
        pred_Ts_world_joint: Float[Tensor, "num_samples time 21 7"],
    ) -> np.ndarray:
        """
        Computes the head translation error between predicted and ground truth joints.

        Args:
            label_Ts_world_joint: Ground truth world joint transforms (time, 21, 7).
            pred_Ts_world_joint: Predicted world joint transforms (num_samples, time, 21, 7).

        Returns:
            Head translation error per sample as a NumPy array (num_samples,).
        """
        errors = pred_Ts_world_joint[:, :, 14, 4:7] - label_Ts_world_joint[:, 14, 4:7]
        mean_errors = torch.mean(torch.linalg.norm(errors, dim=-1), dim=-1)

        return mean_errors.cpu().numpy()

    def compute_mpjpe(
        self,
        label_T_world_root: Float[Tensor, "time 7"],
        label_Ts_world_joint: Float[Tensor, "time 21 7"],
        pred_T_world_root: Float[Tensor, "num_samples time 7"],
        pred_Ts_world_joint: Float[Tensor, "num_samples time 21 7"],
        per_frame_procrustes_align: bool,
    ) -> np.ndarray:
        """
        Computes the Mean Per Joint Position Error (MPJPE) between predicted and ground truth joints.

        Args:
            label_T_world_root: Ground truth root transforms (time, 7).
            label_Ts_world_joint: Ground truth world joint transforms (time, 21, 7).
            pred_T_world_root: Predicted root transforms (num_samples, time, 7).
            pred_Ts_world_joint: Predicted world joint transforms (num_samples, time, 21, 7).
            per_frame_procrustes_align: Whether to perform per-frame Procrustes alignment.

        Returns:
            MPJPE per sample as a NumPy array (num_samples,).
        """
        num_samples, time = pred_Ts_world_joint.shape[:2]

        # Concatenate the root to the joints.
        label_Ts_world_joint = torch.cat(
            [label_T_world_root.unsqueeze(1), label_Ts_world_joint], dim=1
        )
        pred_Ts_world_joint = torch.cat(
            [pred_T_world_root.unsqueeze(2), pred_Ts_world_joint], dim=2
        )

        pred_joint_positions = pred_Ts_world_joint[:, :, :, 4:7]
        label_joint_positions = label_Ts_world_joint.unsqueeze(0).repeat(num_samples, 1, 1, 1)

        if per_frame_procrustes_align:
            pred_joint_positions = self.procrustes_align(
                points_y=label_joint_positions,
                points_x=pred_joint_positions,
                output="aligned_x",
            )

        position_differences = pred_joint_positions - label_joint_positions

        # Per-joint position errors in millimeters.
        pjpe = torch.linalg.norm(position_differences, dim=-1) * 1000.0

        # Mean per-joint position errors.
        mpjpe = torch.mean(pjpe.reshape((num_samples, -1)), dim=-1)

        return mpjpe.cpu().numpy()

    @overload
    def procrustes_align(
        self,
        points_y: Float[Tensor, "*batch N 3"],
        points_x: Float[Tensor, "*batch N 3"],
        output: Literal["transforms"],
        fix_scale: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        ...

    @overload
    def procrustes_align(
        self,
        points_y: Float[Tensor, "*batch N 3"],
        points_x: Float[Tensor, "*batch N 3"],
        output: Literal["aligned_x"],
        fix_scale: bool = False,
    ) -> Tensor:
        ...

    def procrustes_align(
        self,
        points_y: Float[Tensor, "*batch N 3"],
        points_x: Float[Tensor, "*batch N 3"],
        output: Literal["transforms", "aligned_x"],
        fix_scale: bool = False,
    ) -> Union[Tuple[Tensor, Tensor, Tensor], Tensor]:
        """
        Performs similarity transform alignment using the Umeyama method.

        Minimizes:
            mean(|| Y - s * (R @ X) + t ||^2)

        Args:
            points_y: Target points Y (..., N, 3).
            points_x: Source points X (..., N, 3).
            output: 'transforms' to return (s, R, t), 'aligned_x' to return transformed X.
            fix_scale: Whether to fix the scale s to 1.

        Returns:
            Either (s, R, t) tuple or aligned X, depending on the output parameter.
        """
        *dims, N, _ = points_y.shape
        device = points_y.device
        dtype = points_y.dtype
        N_tensor = torch.tensor(N, device=device, dtype=dtype)

        # Subtract mean
        my = points_y.mean(dim=-2)
        mx = points_x.mean(dim=-2)
        y0 = points_y - my[..., None, :]
        x0 = points_x - mx[..., None, :]

        # Correlation
        C = torch.matmul(y0.transpose(-1, -2), x0) / N_tensor
        U, D, Vh = torch.linalg.svd(C, full_matrices=False)

        S = torch.eye(3, device=device, dtype=dtype).expand(*dims, 3, 3)
        det = torch.det(U) * torch.det(Vh)
        S[..., -1, -1] = torch.where(det < 0, -1.0, 1.0)

        R = torch.matmul(U, torch.matmul(S, Vh))

        if fix_scale:
            s = torch.ones(*dims, 1, device=device, dtype=dtype)
        else:
            var = torch.sum(x0 ** 2, dim=(-1, -2), keepdim=True) / N_tensor
            s = (torch.sum(D * S.diagonal(dim1=-2, dim2=-1), dim=-1, keepdim=True) / var[..., 0])

        t = my - s * torch.matmul(R, mx[..., None])[..., 0]

        if output == "transforms":
            return s, R, t
        elif output == "aligned_x":
            aligned_x = s[..., None, :] * torch.einsum("...ij,...nj->...ni", R, points_x) + t[..., None, :]
            return aligned_x
        else:
            assert_never(output)

    def evaluate_directory(
        self,
        dir_with_npz_files: Path,
        use_mean_body_shape: bool = False,
        coco_regressor_path: Optional[Path] = None,
        skip_confirm: bool = False,
    ) -> None:
        """
        Evaluates all .npz files in the specified directory.

        Args:
            dir_with_npz_files: Directory containing .npz files to evaluate.
            use_mean_body_shape: Whether to use mean body shape.
            coco_regressor_path: Path to COCO regressor (.npy file), if needed.
            skip_confirm: Whether to skip confirmation prompts.
        """
        assert dir_with_npz_files.is_dir(), f"{dir_with_npz_files} is not a directory."

        if use_mean_body_shape:
            out_disagg_npz_path = dir_with_npz_files / "_eval_cached_disaggregated_metrics_meanbody.npz"
            out_yaml_path = dir_with_npz_files / "_eval_cached_summary_meanbody.yaml"
        else:
            out_disagg_npz_path = dir_with_npz_files / "_eval_cached_disaggregated_metrics.npz"
            out_yaml_path = dir_with_npz_files / "_eval_cached_summary.yaml"

        # Load COCO regressor if provided
        if coco_regressor_path is not None:
            coco_regressor = np.load(coco_regressor_path)
            assert coco_regressor.shape == (17, 6890), "Invalid COCO regressor shape."
            coco_regressor = torch.from_numpy(coco_regressor.astype(np.float32)).to(self.device)
        else:
            coco_regressor = None

        # Check if metrics are already computed
        if out_disagg_npz_path.exists() or out_yaml_path.exists():
            print("Found existing metrics:")
            print(out_yaml_path.read_text())
            if not skip_confirm:
                confirm = input("Metrics already computed. Overwrite? (y/n) ")
                if confirm.lower() != "y":
                    print("Aborting evaluation.")
                    return

        npz_paths = [
            p
            for p in dir_with_npz_files.glob("**/*.npz")
            if not p.name.startswith("_eval_cached_")
        ]
        num_sequences = len(npz_paths)

        # Initialize metrics dictionary
        stats_per_subsequence: Dict[str, np.ndarray] = {}
        metrics_list = ["mpjpe", "pampjpe", "head_ori", "head_trans", "foot_skate", "foot_contact"]
        if coco_regressor is not None:
            metrics_list.append("coco_mpjpe")

        for metric in metrics_list:
            stats_per_subsequence[metric] = np.zeros((num_sequences,))

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self.process_npz_file,
                    npz_paths[i],
                    coco_regressor,
                    use_mean_body_shape,
                )
                for i in range(num_sequences)
            ]

            for i, future in enumerate(tqdm(futures, total=num_sequences)):
                metrics = future.result()
                for key in metrics_list:
                    stats_per_subsequence[key][i] = metrics.get(key, np.nan)

        # Save metrics
        np.savez(out_disagg_npz_path, **stats_per_subsequence)
        print(f"Wrote disaggregated metrics to {out_disagg_npz_path}")

        # Write summary
        summary = {
            key: {
                "average_sample_mean": float(np.nanmean(values)),
                "stddev_sample": float(np.nanstd(values)),
                "stderr_sample": float(np.nanstd(values) / np.sqrt(np.count_nonzero(~np.isnan(values)))),
            }
            for key, values in stats_per_subsequence.items()
        }
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary_text = yaml.dump(summary)
        out_yaml_path.write_text(f"# Written by {__file__} at {timestamp}\n\n{summary_text}")
        print(f"Wrote summary to {out_yaml_path}")
        print(summary_text)

    def process_npz_file(
        self,
        npz_path: Path,
        coco_regressor: Optional[Tensor],
        use_mean_body_shape: bool,
    ) -> Dict[str, float]:
        """
        Processes a single .npz file and computes metrics.

        Args:
            npz_path: Path to the .npz file.
            coco_regressor: COCO regressor tensor, if any.
            use_mean_body_shape: Whether to use mean body shape.

        Returns:
            A dictionary containing computed metrics for the sequence.
        """
        relevant_outputs = load_relevant_outputs(npz_path)
        device = self.device

        # Load ground truth data
        gt_betas = torch.from_numpy(relevant_outputs["groundtruth_betas"]).to(device)
        gt_T_world_root = torch.from_numpy(relevant_outputs["groundtruth_T_world_root"]).to(device)
        gt_body_quats = torch.from_numpy(relevant_outputs["groundtruth_body_quats"]).to(device)

        gt_shaped = self.body_model.with_shape(gt_betas)
        gt_posed = gt_shaped.with_pose_decomposed(
            T_world_root=gt_T_world_root,
            body_quats=gt_body_quats,
        )

        # Load predicted data
        sampled_betas = torch.from_numpy(relevant_outputs["sampled_betas"]).to(device)
        sampled_T_world_root = torch.from_numpy(relevant_outputs["sampled_T_world_root"]).to(device)
        sampled_body_quats = torch.from_numpy(relevant_outputs["sampled_body_quats"]).to(device)

        if use_mean_body_shape:
            mean_betas = torch.zeros_like(sampled_betas.mean(dim=1, keepdim=True))
            sampled_shaped = self.body_model.with_shape(mean_betas)
        else:
            mean_betas = sampled_betas.mean(dim=1, keepdim=True)
            sampled_shaped = self.body_model.with_shape(mean_betas)

        sampled_posed = sampled_shaped.with_pose_decomposed(
            T_world_root=sampled_T_world_root,
            body_quats=sampled_body_quats,
        )

        # Compute metrics
        metrics = {}
        metrics["mpjpe"] = self.compute_mpjpe(
            label_T_world_root=gt_posed.T_world_root,
            label_Ts_world_joint=gt_posed.Ts_world_joint[..., :21, :],
            pred_T_world_root=sampled_posed.T_world_root,
            pred_Ts_world_joint=sampled_posed.Ts_world_joint[..., :21, :],
            per_frame_procrustes_align=False,
        ).mean()

        metrics["pampjpe"] = self.compute_mpjpe(
            label_T_world_root=gt_posed.T_world_root,
            label_Ts_world_joint=gt_posed.Ts_world_joint[..., :21, :],
            pred_T_world_root=sampled_posed.T_world_root,
            pred_Ts_world_joint=sampled_posed.Ts_world_joint[..., :21, :],
            per_frame_procrustes_align=True,
        ).mean()

        metrics["head_ori"] = self.compute_head_ori(
            label_Ts_world_joint=gt_posed.Ts_world_joint[..., :21, :],
            pred_Ts_world_joint=sampled_posed.Ts_world_joint[..., :21, :],
        ).mean()

        metrics["head_trans"] = self.compute_head_trans(
            label_Ts_world_joint=gt_posed.Ts_world_joint[..., :21, :],
            pred_Ts_world_joint=sampled_posed.Ts_world_joint[..., :21, :],
        ).mean()

        metrics["foot_skate"] = self.compute_foot_skate(
            pred_Ts_world_joint=sampled_posed.Ts_world_joint[..., :21, :]
        ).mean()

        metrics["foot_contact"] = self.compute_foot_contact(
            pred_Ts_world_joint=sampled_posed.Ts_world_joint[..., :21, :]
        ).mean()

        if coco_regressor is not None:
            gt_mesh = gt_posed.lbs()
            gt_coco_joints = torch.einsum("ij,...jk->...ik", coco_regressor, gt_mesh.verts)

            num_samples = sampled_T_world_root.shape[0]
            sampled_coco_joints = []
            for j in range(num_samples):
                sample_posed = sampled_posed.map(
                    lambda t: t[j] if t.shape[0] == num_samples else t
                )
                sample_mesh = sample_posed.lbs()
                sample_coco_joints = torch.einsum(
                    "ij,...jk->...ik", coco_regressor, sample_mesh.verts
                )
                sampled_coco_joints.append(sample_coco_joints)

            sampled_coco_joints = torch.stack(sampled_coco_joints, dim=0)
            coco_errors = torch.linalg.norm(gt_coco_joints - sampled_coco_joints, dim=-1) * 1000.0
            metrics["coco_mpjpe"] = coco_errors.mean().item()

        return metrics


def main(
    dir_with_npz_files: Path,
    body_npz_path: Path = Path("./data/smplh/neutral/model.npz"),
    use_mean_body_shape: bool = False,
    coco_regressor_path: Optional[Path] = None,
    skip_confirm: bool = False,
) -> None:
    """
    Main function to evaluate body pose metrics.

    Args:
        dir_with_npz_files: Directory containing .npz files to evaluate.
        body_npz_path: Path to the SMPL body model (.npz file).
        use_mean_body_shape: Whether to use mean body shape.
        coco_regressor_path: Path to COCO regressor (.npy file), if needed.
        skip_confirm: Whether to skip confirmation prompts.
    """
    evaluator = BodyEvaluator(body_model_path=body_npz_path)
    evaluator.evaluate_directory(
        dir_with_npz_files=dir_with_npz_files,
        use_mean_body_shape=use_mean_body_shape,
        coco_regressor_path=coco_regressor_path,
        skip_confirm=skip_confirm,
    )


if __name__ == "__main__":
    tyro.cli(main)
