import json
import os
import pickle
from functools import cache
from pathlib import Path
from typing import Literal, overload

import numpy as onp
import numpy as np
import torch
import tyro
import viser
import viser.transforms as vtf
import yaml
from jaxtyping import Float, Int
from projectaria_tools.core import mps  # type: ignore
from projectaria_tools.core.mps.utils import get_nearest_pose
from torch import Tensor
from tqdm import tqdm

from egoalgo import fncsmpl
from egoalgo.constraint_optimizers_jax import GuidanceMode
from egoalgo.data.egopose_meta import TakeMeta
from egoalgo.hand_detection_structs import SavedHamerOutputs

vertex_ids = {
    "smplh": {
        "nose": 332,
        "reye": 6260,
        "leye": 2800,
        "rear": 4071,
        "lear": 583,
        "rthumb": 6191,
        "rindex": 5782,
        "rmiddle": 5905,
        "rring": 6016,
        "rpinky": 6133,
        "lthumb": 2746,
        "lindex": 2319,
        "lmiddle": 2445,
        "lring": 2556,
        "lpinky": 2673,
        "LBigToe": 3216,
        "LSmallToe": 3226,
        "LHeel": 3387,
        "RBigToe": 6617,
        "RSmallToe": 6624,
        "RHeel": 6787,
    },
    "smplx": {
        "nose": 9120,
        "reye": 9929,
        "leye": 9448,
        "rear": 616,
        "lear": 6,
        "rthumb": 8079,
        "rindex": 7669,
        "rmiddle": 7794,
        "rring": 7905,
        "rpinky": 8022,
        "lthumb": 5361,
        "lindex": 4933,
        "lmiddle": 5058,
        "lring": 5169,
        "lpinky": 5286,
        "LBigToe": 5770,
        "LSmallToe": 5780,
        "LHeel": 8846,
        "RBigToe": 8463,
        "RSmallToe": 8474,
        "RHeel": 8635,
    },
    "mano": {
        "thumb": 744,
        "index": 320,
        "middle": 443,
        "ring": 554,
        "pinky": 671,
    },
}


# joint names MANO
# Assuming this joint order in MANO is equal to the SMPL-H hand joint order (??)
MANO_LEFT_JOINT_NAMES = [
    "left_wrist",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "left_thumb4",
    "left_index4",
    "left_middle4",
    "left_ring4",
    "left_pinky4",
]
EGOEXO_NAMES_LEFT = [
    "left_wrist",
    "left_index_1",
    "left_index_2",
    "left_index_3",
    "left_middle_1",
    "left_middle_2",
    "left_middle_3",
    "left_pinky_1",
    "left_pinky_2",
    "left_pinky_3",
    "left_ring_1",
    "left_ring_2",
    "left_ring_3",
    "left_thumb_1",
    "left_thumb_2",
    "left_thumb_3",
    "left_thumb_4",
    "left_index_4",
    "left_middle_4",
    "left_ring_4",
    "left_pinky_4",
]

MANO_RIGHT_JOINT_NAMES = [
    "right_wrist",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "right_thumb4",
    "right_index4",
    "right_middle4",
    "right_ring4",
    "right_pinky4",
]
EGOEXO_NAMES_RIGHT = [
    "right_wrist",
    "right_index_1",
    "right_index_2",
    "right_index_3",
    "right_middle_1",
    "right_middle_2",
    "right_middle_3",
    "right_pinky_1",
    "right_pinky_2",
    "right_pinky_3",
    "right_ring_1",
    "right_ring_2",
    "right_ring_3",
    "right_thumb_1",
    "right_thumb_2",
    "right_thumb3",
    "right_thumb_4",
    "right_index_4",
    "right_middle_4",
    "right_ring_4",
    "right_pinky_4",
]


def get_mano_from_openpose_indices(include_tips: bool) -> Int[onp.ndarray, "21"]:
    # https://github.com/geopavlakos/hamer/blob/272d68f176e0ea8a506f761663dd3dca4a03ced0/hamer/models/mano_wrapper.py#L20
    # fmt: off
    mano_to_openpose = [
        0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20
    ]
    # fmt: on
    openpose_from_mano_idx = {
        mano_idx: openpose_idx for openpose_idx, mano_idx in enumerate(mano_to_openpose)
    }
    return onp.array(
        [openpose_from_mano_idx[i] for i in range(21 if include_tips else 16)]
    )


def tips_from_vertices(
    vertices: np.ndarray,
    model_type: Literal["smplh", "mano", "smplx"],
    side: Literal["left", "right"],
):
    """
    Select finger tips from SMPL vertices.
    vertices: B x N x 3 array
    model_type: 'smplh', 'smplx', 'mano'
    """
    if model_type == "mano":
        side_short = ""
    else:
        side_short = side[0]
    tip_names = ["thumb", "index", "middle", "ring", "pinky"]
    tips_idxs = []
    for tip_name in tip_names:
        tips_idxs.append(vertex_ids[model_type][side_short + tip_name])
    tips_idxs = np.array(tips_idxs)
    finger_tips = vertices[..., tips_idxs, :]
    return finger_tips


# def joints_with_tips(
#     vertices: np.ndarray,
#     joints: np.ndarray,
#     model_type: Literal["smplh", "mano"],
#     side: Literal["left", "right"],
# ) -> np.ndarray:
#     tips = tips_from_vertices(vertices, model_type, side)
#     if model_type  == "smplh":
#         assert joints.shape[-2] in (51, 52)
#         offset = joints.shape[-2] - 51
#         wrist = joints.shape
#         if joints.shape[-2] == 21:
#             joints = joints[..., 15 + :, :]
#         # joints =
#     return np.concatenate([joints, tips], axis=-2)


@cache
def egoexo_to_mano(gt_joints_path: Path) -> dict[str, np.ndarray]:
    """
    Convert a dictionary of 2D or 3D keypoints into OpenPose COCO 25 keypoint format, including body and hands.

    Args:
        mano_joints_with_tip_left: (F x N x 3); number of frames in sequence x (MANO joints + 5 finger tips) of left hand
        mano_joints_with_tip_right: (F x N x 3); number of frames in sequence x (MANO joints + 5 finger tips) of right hand
        keypoints_dict (dict): A dictionary where the key is the joint name and the value is a dict with
                            {'x': value, 'y': value, 'z': value}.

    Returns:
        dict: A dictionary with 'body', 'left_hand', and 'right_hand' keypoints in OpenPose format.
    """

    # Read gronu truth annotation
    anno = json.load(gt_joints_path.open("r"))
    frames = np.array(list(anno.keys())).astype(int)

    def get_keypoints(order, gt_kpts):
        keypoints, mask = [], []
        for joint in order:
            cc = gt_kpts.get(joint, None)  # Default to (0,0) if joint not found
            if cc is not None:
                keypoints.append([cc["x"], cc["y"], cc["z"]])
                mask.append(True)
            else:
                keypoints.append([0, 0, 0])
                mask.append(False)
        return keypoints, mask

    # get ordered joints
    def batch_get_keypoints(side):
        if side == "right":
            EGOEXO_NAMES = EGOEXO_NAMES_RIGHT
        elif side == "left":
            EGOEXO_NAMES = EGOEXO_NAMES_LEFT
        else:
            assert False

        egoexo_hand, egoexo_hand_mask = [], []
        for frame, x in anno.items():
            kp, mask = get_keypoints(EGOEXO_NAMES, x[0]["annotation3D"])
            egoexo_hand.append(kp)
            egoexo_hand_mask.append(mask)

        return np.array(egoexo_hand), np.array(egoexo_hand_mask)

    # GT JOINTS
    left_kpts, left_mask = batch_get_keypoints("left")
    right_kpts, right_mask = batch_get_keypoints("right")

    return {
        "frames": frames,
        "left_kpts": left_kpts,
        "left_mask": left_mask,
        "right_kpts": right_kpts,
        "right_mask": right_mask,
    }


def main(
    # take_index: int,
    # eval_mode: Literal["hamer"] | GuidanceMode,
    results_save_path: Path = Path("./data/hand_eval_stats.yaml"),
    results_save_path_all: Path = Path("./data/hand_eval_results.yaml"),
    egoexo_dir: Path = Path("/bluesclues-data/brent/egoexo4d"),
    egoexo_reorg_dir: Path = Path("/bluesclues-data/brent/egoalgo_egoexo4d_data"),
    body_npz_path: Path = Path("./data/smplh/neutral/model.npz"),
    write: bool = True,
    num_workers: int = 1,
    # hamer_frames_only: bool = True,
) -> None:
    assert not results_save_path.exists()
    assert not results_save_path_all.exists()
    stats_from_exp = {}
    stats_from_exp_all = {}

    for eval_mode in (
        "hamer",
        "no_hands",
        "aria_wrist_only",
        "aria_hamer",
        "hamer_wrist",
        "hamer_reproj2",
    ):
        for hamer_frames_only in (True, False) if eval_mode != "hamer" else (True,):
            mpjpes = dict[int, dict[str, dict[int, tuple[float, ...]]]]()
            pampjpes = dict[int, dict[str, dict[int, tuple[float, ...]]]]()
            matched_kp = 0
            total_kp = 0
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def process_take(
                take_index,
                eval_mode,
                egoexo_dir,
                egoexo_reorg_dir,
                body_npz_path,
                hamer_frames_only,
            ):
                take_mpjpes, take_pampjpes, take_matched_kp, take_total_kp = eval_take(
                    take_index,
                    eval_mode,
                    egoexo_dir,
                    egoexo_reorg_dir,
                    body_npz_path,
                    hamer_frames_only,
                )
                return (
                    take_index,
                    take_mpjpes,
                    take_pampjpes,
                    take_matched_kp,
                    take_total_kp,
                )

            exp = f"{eval_mode=}-{hamer_frames_only=}"
            print(exp)
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(
                        process_take,
                        take_index,
                        eval_mode,
                        egoexo_dir,
                        egoexo_reorg_dir,
                        body_npz_path,
                        hamer_frames_only,
                    )
                    for take_index in range(65)
                    # for take_index in (0, 1)
                ]

                for future in as_completed(futures):
                    (
                        take_index,
                        take_mpjpes,
                        take_pampjpes,
                        take_matched_kp,
                        take_total_kp,
                    ) = future.result()
                    mpjpes[take_index] = take_mpjpes
                    pampjpes[take_index] = take_pampjpes
                    matched_kp += take_matched_kp
                    total_kp += take_total_kp
            mpjpe_concat = []
            pampjpe_concat = []
            for _take_idx, v in mpjpes.items():
                for _side, v in v.items():
                    for _frame_idx, v in v.items():
                        mpjpe_concat.extend(v)
            for _take_idx, v in pampjpes.items():
                for _side, v in v.items():
                    for _frame_idx, v in v.items():
                        pampjpe_concat.extend(v)

            stats = {
                "mpjpe": float(np.mean(mpjpe_concat)),
                "mpjpe_stderr": float(
                    np.std(mpjpe_concat) / np.sqrt(len(mpjpe_concat))
                ),
                "pampjpe": float(np.mean(pampjpe_concat)),
                "pampjpe_stderr": float(
                    np.std(mpjpe_concat) / np.sqrt(len(mpjpe_concat))
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
            print()
            print()
            print()
            print(exp)
            print(stats)
            print()
            print()
            print()
            # breakpoint()
            stats_from_exp[exp] = stats
            stats_from_exp_all[exp] = stats_all

            if write:
                results_save_path.write_text(yaml.dump(stats_from_exp))
                results_save_path_all.write_text(yaml.dump(stats_from_exp_all))


cached_pickle_load = cache(pickle.load)
viser_server = cache(viser.ViserServer)


def eval_take(
    take_index: int,
    eval_mode: Literal["hamer"] | GuidanceMode,
    egoexo_dir: Path = Path("/bluesclues-data/brent/egoexo4d"),
    egoexo_reorg_dir: Path = Path("/bluesclues-data/brent/egoalgo_egoexo4d_data"),
    body_npz_path: Path = Path("./data/smplh/neutral/model.npz"),
    hamer_frames_only: bool = True,
) -> tuple[
    dict[str, dict[int, tuple[float, ...]]],
    dict[str, dict[int, tuple[float, ...]]],
    int,
    int,
]:
    device = torch.device("cuda")
    body_model = fncsmpl.SmplModel.load(body_npz_path).to(device)

    # Get paths.
    ego_pose_split_dir = egoexo_dir / "annotations" / "ego_pose" / "val"
    hand_anno_dir = ego_pose_split_dir / "hand" / "annotation"

    # Which take are we using?
    hand_annotated_uids = sorted([p.stem for p in hand_anno_dir.iterdir()])
    take_uid = hand_annotated_uids[take_index]
    take_meta = TakeMeta.load(egoexo_dir)
    take_name = take_meta.name_from_uid[take_uid]
    print(f"Processing take {take_name} with UID {take_uid}")

    print("Load hand annotations")
    mano_anno = egoexo_to_mano(hand_anno_dir / f"{take_uid}.json")

    traj_dir = egoexo_reorg_dir / take_name

    print("Load hand annotations")
    hamer_outputs: SavedHamerOutputs = cached_pickle_load(
        (traj_dir / "hamer_outputs.pkl").open("rb")
    )
    image_timestamps_ns = sorted(
        t for t in hamer_outputs["detections_left_wrt_cam"].keys()
    )
    if eval_mode == "hamer":
        slam_root_dir = traj_dir / "mps"
        closed_loop_path = slam_root_dir / "closed_loop_trajectory.csv"
        closed_loop_traj = mps.read_closed_loop_trajectory(str(closed_loop_path))  # type: ignore

        def get_keypoints(
            frame_idx: int,
        ) -> tuple[None | np.ndarray, None | np.ndarray]:
            T_world_cam = vtf.SE3(
                get_nearest_pose(closed_loop_traj, image_timestamps_ns[frame_idx])
                .transform_world_device.to_quat_and_translation()
                .squeeze(axis=0)
            ) @ vtf.SE3(hamer_outputs["T_device_cam"])
            time_ns = image_timestamps_ns[frame_idx]

            left_det = hamer_outputs["detections_left_wrt_cam"].get(time_ns, None)
            left_keypoints = None
            if left_det is not None:
                left_keypoints_wrt_cam = left_det["keypoints_3d"][0][
                    mano_from_openpose_indices
                ]
                left_keypoints = T_world_cam @ left_keypoints_wrt_cam

            right_det = hamer_outputs["detections_right_wrt_cam"].get(time_ns, None)
            right_keypoints = None
            if right_det is not None:
                right_keypoints_wrt_cam = right_det["keypoints_3d"][0][
                    mano_from_openpose_indices
                ]
                right_keypoints = T_world_cam @ right_keypoints_wrt_cam

            return left_keypoints, right_keypoints
    else:
        print("Loading saved outputs")
        saved = []
        for saved_npz_path in traj_dir.glob(f"egoalgo_outputs/{eval_mode}*-*.npz"):
            print(saved_npz_path)
            outputs = dict(np.load(saved_npz_path))
            expected_keys = [
                "Ts_world_cpf",
                "Ts_world_root",
                "body_quats",
                "left_hand_quats",
                "right_hand_quats",
                "betas",
                "frame_nums",
                "timestamps_ns",
            ]
            assert all(
                key in outputs for key in expected_keys
            ), f"Missing keys in NPZ file. Expected: {expected_keys}, Found: {list(outputs.keys())}"

            # NOTE: this is because I had a bug when saving some of the frame_nums.
            if outputs["frame_nums"].shape[0] != outputs["body_quats"].shape[1]:
                outputs["frame_nums"] = (
                    np.arange(outputs["body_quats"].shape[1]) + outputs["frame_nums"][0]
                )
            saved.append(outputs)

        if len(saved) == 0:
            return {}, {}, 0, 0

        sample_idx = 0
        betas = torch.tensor(
            np.concatenate([s["betas"][sample_idx] for s in saved], axis=0),
            device=device,
        )

        frame_nums = np.concatenate([s["frame_nums"] for s in saved], axis=0)
        Ts_world_root = torch.tensor(
            np.concatenate([s["Ts_world_root"][sample_idx] for s in saved], axis=0),
            device=device,
        )

        body_quats = np.concatenate(
            [s["body_quats"][sample_idx] for s in saved], axis=0
        )
        left_hand_quats = np.concatenate(
            [s["left_hand_quats"][sample_idx] for s in saved], axis=0
        )
        right_hand_quats = np.concatenate(
            [s["right_hand_quats"][sample_idx] for s in saved], axis=0
        )
        local_quats = torch.tensor(
            np.concatenate([body_quats, left_hand_quats, right_hand_quats], axis=1)
        ).to(device)
        del body_quats
        del left_hand_quats
        del right_hand_quats

        mesh = body_model.with_shape(betas).with_pose(Ts_world_root, local_quats).lbs()

        def get_keypoints(
            frame_idx: int,
        ) -> tuple[None | np.ndarray, None | np.ndarray]:
            assert frame_nums.shape == betas.shape[:1]
            (indices,) = np.nonzero(frame_nums == frame_idx)

            if indices.shape == () or indices.shape == (0,):
                return None, None

            assert indices.shape == (1,)
            (idx,) = indices

            time_ns = image_timestamps_ns[frame_idx]

            sides: list[Literal["left", "right"]] = ["left", "right"]
            if hamer_frames_only:
                sides.clear()
                left_det = hamer_outputs["detections_left_wrt_cam"].get(time_ns, None)
                if left_det is not None:
                    sides.append("left")

                right_det = hamer_outputs["detections_right_wrt_cam"].get(time_ns, None)
                if right_det is not None:
                    sides.append("right")

            out: dict[str, None | np.ndarray] = {"left": None, "right": None}
            for side in sides:
                wrist_idx = {"left": 19, "right": 20}[side]
                hand_start_idx = {"left": 21, "right": 36}[side]
                keypoints = np.concatenate(
                    [
                        mesh.posed_model.Ts_world_joint[
                            idx, wrist_idx : wrist_idx + 1, 4:7
                        ].numpy(force=True),
                        mesh.posed_model.Ts_world_joint[
                            idx, hand_start_idx : hand_start_idx + 15, 4:7
                        ].numpy(force=True),
                        tips_from_vertices(
                            mesh.verts[idx, ...].numpy(force=True), "smplh", side
                        ),
                    ],
                    axis=0,
                )
                assert keypoints.shape == (21, 3)
                out[side] = keypoints
            return out["left"], out["right"]

    mano_from_openpose_indices = get_mano_from_openpose_indices(include_tips=True)

    left_mpjpe_from_frame = dict[int, tuple[float, ...]]()
    left_pampjpe_from_frame = dict[int, tuple[float, ...]]()
    right_mpjpe_from_frame = dict[int, tuple[float, ...]]()
    right_pampjpe_from_frame = dict[int, tuple[float, ...]]()

    matched_keypoints = 0
    total_keypoints = 0

    # server = viser_server()
    # server.scene.reset()
    # server.gui.reset()
    # point_clouds = list[viser.PointCloudHandle]()
    #
    # size_inp = server.gui.add_number("Point size", initial_value=0.01, step=0.001)
    #
    # @size_inp.on_update
    # def _(_) -> None:
    #     for pc in point_clouds:
    #         pc.point_size = size_inp.value

    left_kp_all = []
    right_kp_all = []
    left_kp_label = []
    right_kp_label = []

    assert len(mano_anno["frames"].shape) == 1
    for i, frame_idx in (pbar := tqdm(enumerate(mano_anno["frames"]))):
        left_anno = mano_anno["left_kpts"][i]
        right_anno = mano_anno["right_kpts"][i]

        left_mask = mano_anno["left_mask"][i]
        right_mask = mano_anno["right_mask"][i]

        left_count = int(np.sum(left_mask))
        right_count = int(np.sum(right_mask))

        if left_count == 0 and right_count == 0:
            continue

        left_kp, right_kp = get_keypoints(frame_idx)

        total_keypoints += left_count
        total_keypoints += right_count

        def astup(array: np.ndarray) -> tuple[float, ...]:
            assert len(array.shape) == 1
            return tuple(float(x) for x in array)

        if left_kp is not None and left_count > 0:
            assert left_kp.shape == (21, 3)
            assert left_kp.shape == left_anno.shape
            left_kp_all.append(left_kp)
            left_kp_label.append(left_anno[left_mask])

            if not np.any(np.abs(left_kp) > 100.0):
                matched_keypoints += left_count

                # for j in range(21):
                #     server.scene.add_label(
                #         f"/est/{j}/label", text=f"le{j}", position=left_kp[j]
                #     )
                # for j in range(21):
                #     server.scene.add_label(
                #         f"/gt/{j}/label",
                #         text=f"la{j}",
                #         position=left_anno[j],
                #         visible=bool(left_mask[j]),
                #     )
                # point_clouds.clear()
                # point_clouds.extend(
                #     [
                #         server.scene.add_point_cloud(
                #             "/left_kp_label",
                #             points=left_kp,
                #             colors=(255, 0, 127),
                #             point_size=0.01,
                #         ),
                #         server.scene.add_point_cloud(
                #             "/right_kp_label",
                #             points=left_anno[left_mask],
                #             colors=(127, 0, 255),
                #             point_size=0.01,
                #         ),
                #     ]
                # )

                left_mpjpe_from_frame[frame_idx] = astup(
                    np.linalg.norm((left_kp - left_anno), axis=1)[left_mask]
                )
                left_pampjpe_from_frame[frame_idx] = astup(
                    np.linalg.norm(
                        aligned_subtract(
                            left_kp[left_mask],
                            left_anno[left_mask],
                            device=device,
                        ),
                        axis=1,
                    )
                )

        if right_kp is not None and right_count > 0:
            assert right_kp.shape == (21, 3)
            assert right_kp.shape == right_anno.shape
            right_kp_all.append(right_kp)
            right_kp_label.append(right_anno[right_mask])

            # Visualize the right hand the same way as the left
            # for j in range(21):
            #     server.scene.add_label(
            #         f"/est_right/{j}/label", text=f"re{j}", position=right_kp[j]
            #     )
            # for j in range(21):
            #     server.scene.add_label(
            #         f"/gt_right/{j}/label",
            #         text=f"ra{j}",
            #         position=right_anno[j],
            #         visible=bool(right_mask[j]),
            #     )
            # point_clouds.extend(
            #     [
            #         server.scene.add_point_cloud(
            #             "/right_kp_est",
            #             points=right_kp,
            #             colors=(0, 255, 0),
            #             point_size=0.01,
            #         ),
            #         server.scene.add_point_cloud(
            #             "/right_kp_gt",
            #             points=right_anno[right_mask],
            #             colors=(0, 255, 255),
            #             point_size=0.01,
            #         ),
            #     ]
            # )

            if not np.any(np.abs(right_kp) > 100.0):
                matched_keypoints += right_count

                right_mpjpe_from_frame[frame_idx] = astup(
                    np.linalg.norm((right_kp - right_anno), axis=1)[right_mask]
                )
                right_pampjpe_from_frame[frame_idx] = astup(
                    np.linalg.norm(
                        aligned_subtract(
                            right_kp[right_mask],
                            right_anno[right_mask],
                            device=device,
                        ),
                        axis=1,
                    )
                )

    # point_clouds.extend(
    #     [
    #         server.scene.add_point_cloud(
    #             "/left_kp_all",
    #             points=np.concatenate(left_kp_all),
    #             colors=(255, 0, 0),
    #             point_size=0.01,
    #         ),
    #         server.scene.add_point_cloud(
    #             "/right_kp_all",
    #             points=np.concatenate(right_kp_all),
    #             colors=(0, 0, 255),
    #             point_size=0.01,
    #         ),
    #         server.scene.add_point_cloud(
    #             "/left_kp_label",
    #             points=np.concatenate(left_kp_label),
    #             colors=(255, 0, 127),
    #             point_size=0.01,
    #         ),
    #         server.scene.add_point_cloud(
    #             "/right_kp_label",
    #             points=np.concatenate(right_kp_label),
    #             colors=(127, 0, 255),
    #             point_size=0.01,
    #         ),
    #     ]
    # )

    # next_btn = server.gui.add_button("Next")
    #
    # while next_btn.value is False:
    #     import time
    #
    #     time.sleep(0.1)

    return (
        {"left": left_mpjpe_from_frame, "right": right_mpjpe_from_frame},
        {"left": left_pampjpe_from_frame, "right": right_pampjpe_from_frame},
        matched_keypoints,
        total_keypoints,
    )


def aligned_subtract(a: np.ndarray, b: np.ndarray, device: torch.device) -> np.ndarray:
    if a.shape[0] == 1:
        assert b.shape[0] == 1
        return np.zeros_like(a)
    aligned_b = procrustes_align(
        points_y=torch.tensor(a, dtype=torch.float64, device=device),
        points_x=torch.tensor(b, dtype=torch.float64, device=device),
        output="aligned_x",
        fix_scale=False,
    ).numpy(force=True)

    if np.any(np.isnan(aligned_b)):
        breakpoint()

    return a - aligned_b


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
    dtype = points_y.dtype
    *dims, N, _ = points_y.shape
    device = points_y.device
    N = torch.ones((*dims, 1, 1), device=device, dtype=dtype) * N

    # subtract mean
    my = points_y.sum(dim=-2) / N[..., 0]  # (*, 3)
    mx = points_x.sum(dim=-2) / N[..., 0]
    y0 = points_y - my[..., None, :]  # (*, N, 3)
    x0 = points_x - mx[..., None, :]

    # correlation
    C = torch.matmul(y0.transpose(-1, -2), x0) / N  # (*, 3, 3)
    U, D, Vh = torch.linalg.svd(C)  # (*, 3, 3), (*, 3), (*, 3, 3)

    S = (
        torch.eye(3, device=device, dtype=dtype)
        .reshape(*(1,) * (len(dims)), 3, 3)
        .repeat(*dims, 1, 1)
    )
    neg = torch.det(U) * torch.det(Vh.transpose(-1, -2)) < 0
    S = torch.where(
        neg.reshape(*dims, 1, 1),
        S * torch.diag(torch.tensor([1, 1, -1], device=device, dtype=dtype)),
        S,
    )

    R = torch.matmul(U, torch.matmul(S, Vh))  # (*, 3, 3)

    D = torch.diag_embed(D)  # (*, 3, 3)
    if fix_scale:
        s = torch.ones(*dims, 1, device=device, dtype=dtype)
    else:
        var = (
            torch.sum(torch.square(x0), dim=(-1, -2), keepdim=True, dtype=dtype) / N
        )  # (*, 1, 1)
        s = (
            torch.diagonal(torch.matmul(D, S), dim1=-2, dim2=-1).sum(
                dim=-1, keepdim=True
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


if __name__ == "__main__":
    tyro.cli(main)