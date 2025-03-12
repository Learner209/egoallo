from __future__ import annotations

import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Literal, overload

import numpy as np
import torch
import tyro
import yaml
from jaxtyping import Float
from torch import Tensor
from tqdm.auto import tqdm
from typing_extensions import assert_never

from egoalgo import fncsmpl
from egoalgo.eval_structs import load_relevant_outputs
from egoalgo.transforms import SO3


def compute_foot_skate(
    pred_Ts_world_joint: Float[Tensor, "num_samples time 21 7"],
) -> np.ndarray:
    (num_samples, time) = pred_Ts_world_joint.shape[:2]

    # Drop the person to the floor.
    # This is necessary for the foot skating metric to make sense for floating people...!
    pred_Ts_world_joint = pred_Ts_world_joint.clone()
    pred_Ts_world_joint[..., 6] -= torch.min(pred_Ts_world_joint[..., 6])

    foot_indices = torch.tensor([6, 7, 9, 10], device=pred_Ts_world_joint.device)

    foot_positions = pred_Ts_world_joint[:, :, foot_indices, 4:7]
    foot_positions_diff = foot_positions[:, 1:, :, :2] - foot_positions[:, :-1, :, :2]
    assert foot_positions_diff.shape == (num_samples, time - 1, 4, 2)

    foot_positions_diff_norm = torch.sum(torch.abs(foot_positions_diff), dim=-1)
    assert foot_positions_diff_norm.shape == (num_samples, time - 1, 4)

    # From EgoEgo / kinpoly.
    H_thresh = torch.tensor(
        # To match indices above: (ankle, ankle, toe, toe)
        [0.08, 0.08, 0.04, 0.04],
        device=pred_Ts_world_joint.device,
        dtype=torch.float32,
    )

    foot_positions_diff_norm = torch.sum(torch.abs(foot_positions_diff), dim=-1)
    assert foot_positions_diff_norm.shape == (num_samples, time - 1, 4)

    # Threshold.
    foot_positions_diff_norm = foot_positions_diff_norm * (
        foot_positions[..., 1:, :, 2] < H_thresh
    )
    fs_per_sample = torch.sum(
        torch.sum(
            foot_positions_diff_norm
            * (2 - 2 ** (foot_positions[..., 1:, :, 2] / H_thresh)),
            dim=-1,
        ),
        dim=-1,
    )
    assert fs_per_sample.shape == (num_samples,)

    return fs_per_sample.numpy(force=True)


def compute_foot_contact(
    pred_Ts_world_joint: Float[Tensor, "num_samples time 21 7"],
) -> np.ndarray:
    (num_samples, time) = pred_Ts_world_joint.shape[:2]

    foot_indices = torch.tensor([6, 7, 9, 10], device=pred_Ts_world_joint.device)

    # From EgoEgo / kinpoly.
    H_thresh = torch.tensor(
        # To match indices above: (ankle, ankle, toe, toe)
        [0.08, 0.08, 0.04, 0.04],
        device=pred_Ts_world_joint.device,
        dtype=torch.float32,
    )

    foot_positions = pred_Ts_world_joint[:, :, foot_indices, 4:7]

    any_contact = torch.any(
        torch.any(foot_positions[..., 2] < H_thresh, dim=-1),
        dim=-1,
    ).to(torch.float32)
    assert any_contact.shape == (num_samples,)

    return any_contact.numpy(force=True)


def compute_head_ori(
    label_Ts_world_joint: Float[Tensor, "time 21 7"],
    pred_Ts_world_joint: Float[Tensor, "num_samples time 21 7"],
) -> np.ndarray:
    (num_samples, time) = pred_Ts_world_joint.shape[:2]
    matrix_errors = (
        SO3(pred_Ts_world_joint[:, :, 14, :4]).as_matrix()
        @ SO3(label_Ts_world_joint[:, 14, :4]).inverse().as_matrix()
    ) - torch.eye(3, device=label_Ts_world_joint.device)
    assert matrix_errors.shape == (num_samples, time, 3, 3)

    return torch.mean(
        torch.linalg.norm(matrix_errors.reshape((num_samples, time, 9)), dim=-1),
        dim=-1,
    ).numpy(force=True)


def compute_head_trans(
    label_Ts_world_joint: Float[Tensor, "time 21 7"],
    pred_Ts_world_joint: Float[Tensor, "num_samples time 21 7"],
) -> np.ndarray:
    (num_samples, time) = pred_Ts_world_joint.shape[:2]
    errors = pred_Ts_world_joint[:, :, 14, 4:7] - label_Ts_world_joint[:, 14, 4:7]
    assert errors.shape == (num_samples, time, 3)

    return torch.mean(
        torch.linalg.norm(errors, dim=-1),
        dim=-1,
    ).numpy(force=True)


def compute_mpjpe(
    label_T_world_root: Float[Tensor, "time 7"],
    label_Ts_world_joint: Float[Tensor, "time 21 7"],
    pred_T_world_root: Float[Tensor, "num_samples time 7"],
    pred_Ts_world_joint: Float[Tensor, "num_samples time 21 7"],
    per_frame_procrustes_align: bool,
) -> np.ndarray:
    num_samples, time, _, _ = pred_Ts_world_joint.shape

    # Concatenate the world root to the joints.
    label_Ts_world_joint = torch.cat(
        [label_T_world_root[..., None, :], label_Ts_world_joint],
        dim=-2,
    )
    pred_Ts_world_joint = torch.cat(
        [pred_T_world_root[..., None, :], pred_Ts_world_joint],
        dim=-2,
    )
    del label_T_world_root, pred_T_world_root

    pred_joint_positions = pred_Ts_world_joint[:, :, :, 4:7]
    label_joint_positions = label_Ts_world_joint[None, :, :, 4:7].repeat(
        num_samples,
        1,
        1,
        1,
    )

    if per_frame_procrustes_align:
        pred_joint_positions = procrustes_align(
            points_y=pred_joint_positions,
            points_x=label_joint_positions,
            output="aligned_x",
        )

    position_differences = pred_joint_positions - label_joint_positions
    assert position_differences.shape == (num_samples, time, 22, 3)

    # Per-joint position errors, in millimeters.
    pjpe = torch.linalg.norm(position_differences, dim=-1) * 1000.0
    assert pjpe.shape == (num_samples, time, 22)

    # Mean per-joint position errors.
    mpjpe = torch.mean(pjpe.reshape((num_samples, -1)), dim=-1)
    assert mpjpe.shape == (num_samples,)

    return mpjpe.cpu().numpy()


@overload
def procrustes_align(
    points_y: Float[Tensor, "*#batch N 3"],
    points_x: Float[Tensor, "*#batch N 3"],
    output: Literal["transforms"],
    fix_scale: bool = False,
) -> tuple[Tensor, Tensor, Tensor]: ...


@overload
def procrustes_align(
    points_y: Float[Tensor, "*#batch N 3"],
    points_x: Float[Tensor, "*#batch N 3"],
    output: Literal["aligned_x"],
    fix_scale: bool = False,
) -> Tensor: ...


def procrustes_align(
    points_y: Float[Tensor, "*#batch N 3"],
    points_x: Float[Tensor, "*#batch N 3"],
    output: Literal["transforms", "aligned_x"],
    fix_scale: bool = False,
) -> tuple[Tensor, Tensor, Tensor] | Tensor:
    """Similarity transform alignment using the Umeyama method. Adapted from
    SLAHMR: https://github.com/vye16/slahmr/blob/main/slahmr/geometry/pcl.py

    Minimizes:

        mean( || Y - s * (R @ X) + t ||^2 )

    with respect to s, R, and t.

    Returns an (s, R, t) tuple.
    """
    *dims, N, _ = points_y.shape
    device = points_y.device
    N = torch.ones((*dims, 1, 1), device=device) * N

    # subtract mean
    my = points_y.sum(dim=-2) / N[..., 0]  # (*, 3)
    mx = points_x.sum(dim=-2) / N[..., 0]
    y0 = points_y - my[..., None, :]  # (*, N, 3)
    x0 = points_x - mx[..., None, :]

    # correlation
    C = torch.matmul(y0.transpose(-1, -2), x0) / N  # (*, 3, 3)
    U, D, Vh = torch.linalg.svd(C)  # (*, 3, 3), (*, 3), (*, 3, 3)

    S = (
        torch.eye(3, device=device)
        .reshape(*(1,) * (len(dims)), 3, 3)
        .repeat(*dims, 1, 1)
    )
    neg = torch.det(U) * torch.det(Vh.transpose(-1, -2)) < 0
    S = torch.where(
        neg.reshape(*dims, 1, 1),
        S * torch.diag(torch.tensor([1, 1, -1], device=device)),
        S,
    )

    R = torch.matmul(U, torch.matmul(S, Vh))  # (*, 3, 3)

    D = torch.diag_embed(D)  # (*, 3, 3)
    if fix_scale:
        s = torch.ones(*dims, 1, device=device, dtype=torch.float32)
    else:
        var = torch.sum(torch.square(x0), dim=(-1, -2), keepdim=True) / N  # (*, 1, 1)
        s = (
            torch.diagonal(torch.matmul(D, S), dim1=-2, dim2=-1).sum(
                dim=-1,
                keepdim=True,
            )
            / var[..., 0]
        )  # (*, 1)

    t = my - s * torch.matmul(R, mx[..., None])[..., 0]  # (*, 3)

    assert s.shape == (*dims, 1)
    assert R.shape == (*dims, 3, 3)
    assert t.shape == (*dims, 3)

    if output == "transforms":
        return s, R, t
    elif output == "aligned_x":
        aligned_x = (
            s[..., None, :] * torch.einsum("...ij,...nj->...ni", R, points_x)
            + t[..., None, :]
        )
        assert aligned_x.shape == points_x.shape
        return aligned_x
    else:
        assert_never(output)


def main(
    dir_with_npz_files: Path,
    /,
    body_npz_path: Path = Path("./data/smplh/neutral/model.npz"),
    use_mean_body_shape: bool = False,
    coco_regressor_path: Path | None = None,
    skip_confirm: Annotated[bool, tyro.conf.arg(aliases=["-y"])] = False,
) -> None:
    assert dir_with_npz_files.is_dir()
    assert dir_with_npz_files.exists()

    if use_mean_body_shape:
        out_disagg_npz_path = (
            dir_with_npz_files / "_eval_cached_disaggregated_metrics_meanbody.npz"
        )
        out_yaml_path = dir_with_npz_files / "_eval_cached_summary_meanbody.yaml"
    else:
        out_disagg_npz_path = (
            dir_with_npz_files / "_eval_cached_disaggregated_metrics.npz"
        )
        out_yaml_path = dir_with_npz_files / "_eval_cached_summary.yaml"

    # If we want COCO joint MPJPE. :)
    coco_regressor = (
        np.load(coco_regressor_path) if coco_regressor_path is not None else None
    )
    assert coco_regressor is None or coco_regressor.shape == (17, 6890)

    # Check if metrics are already written.
    if out_disagg_npz_path.exists() or out_yaml_path.exists():
        print("Found existing metrics:")
        print(out_yaml_path.read_text())
        if not skip_confirm:
            confirm = input("Metrics already computed. Overwrite? (y/n) ")
            if confirm != "y":
                print("Aborting!")
                return

    npz_paths = tuple(
        p
        for p in dir_with_npz_files.glob("**/*.npz")
        if not p.name.startswith("_eval_cached_")
    )
    device = torch.device("cuda")
    body_model = fncsmpl.SmplModel.load(body_npz_path).to(device)

    if coco_regressor is not None:
        coco_regressor = torch.from_numpy(coco_regressor.astype(np.float32)).to(device)

    # Metrics.
    stats_per_subsequence: dict[str, Float[np.ndarray, "num_seq num_samples"]] = {}

    with ThreadPoolExecutor() as executor:
        next_file = executor.submit(load_relevant_outputs, npz_paths[0])
        pbar = tqdm(range(len(npz_paths)))
        for i in pbar:
            assert next_file is not None
            relevant_outputs = next_file.result()

            if i + 1 < len(npz_paths):
                next_file = executor.submit(load_relevant_outputs, npz_paths[i + 1])
            else:
                next_file = None

            gt_shaped = body_model.with_shape(
                torch.from_numpy(relevant_outputs["groundtruth_betas"]).to(
                    device=device,
                ),
            )
            gt_posed = gt_shaped.with_pose_decomposed(
                T_world_root=torch.from_numpy(
                    relevant_outputs["groundtruth_T_world_root"],
                ).to(device),
                body_quats=torch.from_numpy(
                    relevant_outputs["groundtruth_body_quats"],
                ).to(device),
            )

            sampled_shaped = body_model.with_shape(
                torch.zeros(relevant_outputs["sampled_betas"].shape).to(device=device)
                if use_mean_body_shape
                else torch.from_numpy(relevant_outputs["sampled_betas"])
                .to(device=device)
                .mean(dim=1, keepdim=True),
            )
            sampled_posed = sampled_shaped.with_pose_decomposed(
                T_world_root=torch.from_numpy(
                    relevant_outputs["sampled_T_world_root"],
                ).to(device),
                body_quats=torch.from_numpy(relevant_outputs["sampled_body_quats"]).to(
                    device,
                ),
            )
            num_samples = relevant_outputs["sampled_T_world_root"].shape[0]

            # Compute body joint statistics.
            if i == 0:
                stats_template = np.zeros((len(npz_paths), num_samples))
                stats_per_subsequence["mpjpe"] = stats_template.copy()
                stats_per_subsequence["coco_mpjpe"] = stats_template.copy()
                stats_per_subsequence["pampjpe"] = stats_template.copy()
                stats_per_subsequence["head_ori"] = stats_template.copy()
                stats_per_subsequence["head_trans"] = stats_template.copy()
                stats_per_subsequence["foot_skate"] = stats_template.copy()
                stats_per_subsequence["foot_contact"] = stats_template.copy()

            if coco_regressor is not None:
                gt_mesh = gt_posed.lbs()
                sampled_coco = []

                # We do LBS + COCO regression one sample at a time to save memory.
                for j in range(num_samples):
                    sampled_mesh = sampled_posed.map(
                        # TODO: this heuristic is a hack. It will fail if
                        # num_samples happens to match the leading dimension of
                        # any tensors that aren't actually prefixed with the
                        # batch axes.
                        lambda t: t[j] if t.shape[0] == num_samples else t,
                    ).lbs()
                    sampled_coco_single = torch.einsum(
                        "ij,...jk->...ik",
                        coco_regressor,
                        sampled_mesh.verts,
                    )
                    sampled_coco.append(sampled_coco_single)

                gt_coco = torch.einsum("ij,...jk->...ik", coco_regressor, gt_mesh.verts)
                sampled_coco = torch.stack(sampled_coco, dim=0)
                assert sampled_coco.shape == (num_samples, sampled_coco.shape[1], 17, 3)
                stats_per_subsequence["coco_mpjpe"][i] = torch.mean(
                    torch.linalg.norm(gt_coco - sampled_coco, dim=-1).reshape(
                        (num_samples, -1),
                    ),
                    dim=1,
                ).numpy(force=True)

            stats_per_subsequence["mpjpe"][i] = compute_mpjpe(
                label_T_world_root=gt_posed.T_world_root,
                label_Ts_world_joint=gt_posed.Ts_world_joint[..., :21, :],
                pred_T_world_root=sampled_posed.T_world_root,
                pred_Ts_world_joint=sampled_posed.Ts_world_joint[..., :21, :],
                per_frame_procrustes_align=False,
            )
            stats_per_subsequence["pampjpe"][i] = compute_mpjpe(
                label_T_world_root=gt_posed.T_world_root,
                label_Ts_world_joint=gt_posed.Ts_world_joint[..., :21, :],
                pred_T_world_root=sampled_posed.T_world_root,
                pred_Ts_world_joint=sampled_posed.Ts_world_joint[..., :21, :],
                per_frame_procrustes_align=True,
            )
            stats_per_subsequence["head_ori"][i] = compute_head_ori(
                label_Ts_world_joint=gt_posed.Ts_world_joint[..., :21, :],
                pred_Ts_world_joint=sampled_posed.Ts_world_joint[..., :21, :],
            )
            stats_per_subsequence["head_trans"][i] = compute_head_trans(
                label_Ts_world_joint=gt_posed.Ts_world_joint[..., :21, :],
                pred_Ts_world_joint=sampled_posed.Ts_world_joint[..., :21, :],
            )
            stats_per_subsequence["foot_skate"][i] = compute_foot_skate(
                pred_Ts_world_joint=sampled_posed.Ts_world_joint[..., :21, :],
            )
            stats_per_subsequence["foot_contact"][i] = compute_foot_contact(
                pred_Ts_world_joint=sampled_posed.Ts_world_joint[..., :21, :],
            )

            if i % 10 == 0:
                stat_strings = []
                for k, v in stats_per_subsequence.items():
                    stat_strings.append(f"{k}: {np.mean(v[: i + 1]):.3f}")
                pbar.set_postfix_str(", ".join(stat_strings))

    # Write core metrics.
    np.savez(out_disagg_npz_path, **stats_per_subsequence)
    print("Wrote disaggregated metrics to", out_disagg_npz_path)

    # Write summary.
    summary = yaml.dump(
        {
            k: {
                "average_sample_mean": float(np.mean(v)),
                "average_sample_min": float(np.mean(np.min(v, axis=1))),
                "stddev_sample": float(np.std(v)),
                "stderr_sample": float(np.std(v) / np.sqrt(len(v))),
            }
            for k, v in stats_per_subsequence.items()
        },
    )
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    (out_yaml_path).write_text(f"# Written by {__file__} at {timestamp}\n\n" + summary)
    print("Wrote summary to", out_yaml_path)
    print()
    print()
    print(summary)


if __name__ == "__main__":
    tyro.cli(main)
