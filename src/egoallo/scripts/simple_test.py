from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING

import torch.utils.data
from tqdm import tqdm

from egoallo.denoising.abs_aadecomp_traj import AbsoluteDenoiseTrajAADecomp
from egoallo.type_stubs import EgoTrainingDataType
from egoallo.sampling import quadratic_ts
from egoallo.utilities import get_class_from_path
from egoallo.constants import EgoTrainingDataZoo

if TYPE_CHECKING:
    pass

from egoallo.data import make_batch_collator, build_dataset
from egoallo.config.train.train_config import EgoAlloTrainConfig
from egoallo.inference_utils import (
    load_denoiser,
    load_runtime_config,
)
from egoallo.denoising.abs_traj import AbsoluteDenoiseTraj
from egoallo.constants import SmplFamilyMetaModelZoo
from egoallo.sampling import CosineNoiseScheduleConstants
from egoallo.viz.hybrik_twist_angle_visualizer import InteractiveSMPLViewer

if __name__ == "__main__":
    checkpoint_dir = Path("experiments/Apr_11_hybrik/v1/checkpoints_10000")
    device = torch.device("cpu")
    runtime_config: EgoAlloTrainConfig = load_runtime_config(
        checkpoint_dir,
    )
    denoiser, model_config = load_denoiser(
        checkpoint_dir,
        runtime_config,
    )
    denoiser = denoiser.to(device)

    bs = 32
    win_size = runtime_config.subseq_len
    noise_constants = CosineNoiseScheduleConstants.compute(timesteps=1000).to(
        device=device,
    )
    alpha_bar_t = noise_constants.alpha_bar_t
    alpha_t = noise_constants.alpha_t

    body_model = (
        SmplFamilyMetaModelZoo[runtime_config.smpl_family_meta_model_name]
        .load(
            runtime_config.smpl_family_model_basedir,
        )
        .to(
            device,
        )
    )

    all_post_pred_x_0_list = []
    all_post_gt_x_0_list = []
    all_metrics = {}

    runtime_config.splits = ("test",)
    runtime_config.temporal_mask_ratio = 0.0
    runtime_config.batch_size = 1
    runtime_config.dataset_slice_strategy = "full_sequence"

    runtime_config.dataset_type = "AriaInferenceDataset"

    dataloader = torch.utils.data.DataLoader(
        dataset=build_dataset(cfg=runtime_config)(config=runtime_config),
        batch_size=bs,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=make_batch_collator(runtime_config),
        drop_last=True,
    )

    for batch_idx, batch in tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="Enumerating test loader",
        ascii=" >=",
    ):
        assert batch.joints_wrt_world.shape[0] == 1
        batch = batch.to(device)
        preprocessed_batch = copy.deepcopy(batch)
        post_processed_batch: EgoTrainingDataType = batch.postprocess()
        seq_len = batch.joints_wrt_world.shape[1]
        window_size = runtime_config.subseq_len
        overlap_size = int(window_size / 4)

        window_size = 128
        overlap_size = 0
        seq_len = 128

        x_t_packed = torch.randn(
            (
                bs,
                seq_len,
                runtime_config.denoising.d_state,
            ),
            device=device,
        )

        # canonical_overlap_weights = (
        #     torch.from_numpy(
        #         np.minimum(
        #             overlap_size,
        #             np.minimum(
        #                 np.arange(1, seq_len + 1),
        #                 np.arange(1, seq_len + 1)[::-1],
        #             ),
        #         )
        #         / overlap_size,
        #     )
        #     .to(device)
        #     .to(torch.float32)
        # )

        # Prepare window data in advance
        window_data = []
        overlap_weights = torch.zeros((1, seq_len, 1), device=x_t_packed.device)

        # for start_t in range(0, seq_len, window_size - overlap_size):
        #     end_t = min(start_t + window_size, seq_len)
        #     overlap_weights_slice = canonical_overlap_weights[
        #         None,
        #         : end_t - start_t,
        #         None,
        #     ]
        #     overlap_weights[:, start_t:end_t, :] += overlap_weights_slice

        #     win_data = copy.deepcopy(post_processed_batch[:, start_t:end_t])
        #     # FIXME: this is a hack to follow the state machine of EgoTrainingData dataclass.
        #     win_data.metadata.stage = "raw"
        #     win_data = win_data.preprocess()

        #     window_data.append((start_t, end_t, win_data, overlap_weights_slice))

        post_processed_batch = post_processed_batch[:, :seq_len]
        preprocessed_batch = preprocessed_batch[:, :seq_len]
        overlap_weights = torch.ones((1, seq_len, 1), device=x_t_packed.device)
        for start_t in range(0, seq_len, window_size - overlap_size):
            end_t = min(start_t + window_size, seq_len)

            win_data = copy.deepcopy(post_processed_batch[:, start_t:end_t])
            # FIXME: this is a hack to follow the state machine of EgoTrainingData dataclass.
            win_data.metadata.stage = "raw"
            win_data = win_data.preprocess()

            window_data.append(
                (
                    start_t,
                    end_t,
                    win_data,
                    torch.ones_like(overlap_weights[:, start_t:end_t, :]),
                ),
            )

        ts = quadratic_ts(timesteps=1000)
        x_t_packed = torch.randn(
            (
                bs,
                seq_len,
                runtime_config.denoising.d_state,
            ),
            device=device,
        )
        for i in range(len(ts) - 1):
            t = ts[i]
            t_next = ts[i + 1]

            with torch.inference_mode():
                x_0_packed_pred = torch.zeros_like(x_t_packed)

                # Process each window
                for start_t, end_t, win_data, overlap_weights_slice in window_data:
                    post_pred_x_0 = denoiser.forward(
                        x_t_unpacked=runtime_config.denoising.unpack_traj(
                            x_t_packed[:, start_t:end_t, :],
                            metadata=win_data.metadata,
                            include_hands=runtime_config.model.include_hands,
                        ),
                        t=torch.tensor([t], device=device).expand((bs,)),
                        joints=win_data.joints_wrt_world,
                        visible_joints_mask=win_data.visible_joints_mask,
                        project_output_rotmats=False,
                        mask=win_data.mask,
                    )

                    x_0_packed_pred[:, start_t:end_t, :] += (
                        post_pred_x_0 * overlap_weights_slice
                    )

                x_0_packed_pred /= overlap_weights
                assert not torch.any(torch.isnan(x_0_packed_pred)), (
                    "Found nan in x_0_packed_pred"
                )

                x_0_pred = runtime_config.denoising.unpack_traj(  # raw traj
                    x_0_packed_pred,
                    include_hands=runtime_config.model.include_hands,
                    project_rotmats=False,
                    metadata=preprocessed_batch.metadata,
                )

            if torch.any(torch.isnan(x_0_packed_pred)):
                print("found nan", i)
            sigma_t = torch.cat(
                [
                    torch.zeros((1,), device=device),
                    torch.sqrt(
                        (1.0 - alpha_bar_t[:-1])
                        / (1 - alpha_bar_t[1:])
                        * (1 - alpha_t),
                    )
                    * 0.0,
                ],
            )
            x_t_packed = (
                torch.sqrt(alpha_bar_t[t_next]) * x_0_packed_pred
                + (
                    torch.sqrt(1 - alpha_bar_t[t_next] - sigma_t[t] ** 2)
                    * (x_t_packed - torch.sqrt(alpha_bar_t[t]) * x_0_packed_pred)
                    / torch.sqrt(1 - alpha_bar_t[t] + 1e-8)
                )
                + sigma_t[t] * torch.randn_like(x_0_packed_pred)
            )

        pred_x_0 = x_0_pred

        if pred_x_0.joints_wrt_world is None or pred_x_0.visible_joints_mask is None:
            # Assigning placeholders to pred_x_0 in advance to prevent `__setitem__` impl of `TensorDataClass` ignoring None attribute.
            pred_x_0.joints_wrt_world = torch.zeros((bs, seq_len, 22, 3))
            pred_x_0.visible_joints_mask = torch.ones_like(
                pred_x_0.joints_wrt_world[..., 0],
            )

        post_pred_x_0 = copy.deepcopy(pred_x_0)

        for start_t, end_t, win_data, overlap_weights_slice in window_data:
            pred_x_0_window = copy.deepcopy(pred_x_0[:, start_t:end_t])

            win_data: EgoTrainingDataType = win_data.postprocess()
            post_pred_x_0_window = win_data.postprocess_denoise_traj(
                pred_x_0_window,
                unmask=False,
            )

            post_pred_x_0[:, start_t:end_t] = post_pred_x_0_window

        post_pred_x_0.metadata = post_processed_batch.metadata

        post_pred_posed = post_pred_x_0.apply_to_body(body_model)
        num_joints = post_processed_batch.joints_wrt_world.shape[-2]

        from egoallo.middleware.third_party.HybrIK.hybrik.models.layers.smpl.fncsmpl_aadecomp import (
            SmplShapedAndPosedAADecomp,
        )
        from egoallo.middleware.third_party.HybrIK.hybrik.models.layers.smplh.fncsmplh import (
            SmplhShapedAndPosed,
        )

        if isinstance(post_pred_posed, SmplhShapedAndPosed):
            post_pred_jts_wrt_world = torch.cat(
                [
                    post_pred_posed.T_world_root[..., None, :],
                    post_pred_posed.Ts_world_joint[..., : num_joints - 1, :],
                ],
                dim=-2,
            )[..., 4:]
        elif isinstance(post_pred_posed, SmplShapedAndPosedAADecomp):
            post_pred_jts_wrt_world = post_pred_posed.pose_skeleton

        input_jts_wrt_world = post_processed_batch.joints_wrt_world

        assert input_jts_wrt_world.shape == post_pred_jts_wrt_world.shape

        pred2gt_jts_offset = input_jts_wrt_world - post_pred_jts_wrt_world
        vis_pred2_gt_jts_offset = torch.where(
            post_processed_batch.visible_joints_mask.bool()
            .unsqueeze(-1)
            .expand(*post_processed_batch.visible_joints_mask.shape, 3),
            pred2gt_jts_offset,
            0,
        )  # *batch, timesteps, jts, 3
        vis_pred2_gt_jts_offset = vis_pred2_gt_jts_offset.sum(
            dim=(-2),
        ) / post_processed_batch.visible_joints_mask.sum(
            dim=-1,
            keepdim=True,
        )  # *batch, timesteps, 3

        x_0 = runtime_config.denoising.from_ego_data(
            ego_data=preprocessed_batch,
            smpl_family_model_basedir=runtime_config.smpl_family_model_basedir,
            include_hands=runtime_config.model.include_hands,
        )
        if isinstance(post_pred_x_0, AbsoluteDenoiseTraj):
            post_pred_x_0.t_world_root += vis_pred2_gt_jts_offset
        elif isinstance(post_pred_x_0, AbsoluteDenoiseTrajAADecomp):
            # post_pred_x_0.joints_wrt_world += vis_pred2_gt_jts_offset[..., None, :]

            body_twists = torch.zeros_like(post_processed_batch.body_twists)
            cos_sin_phis = torch.cat(
                [torch.cos(body_twists), torch.sin(body_twists)],
                dim=-1,
            )
            post_pred_x_0.cos_sin_phis = cos_sin_phis

            ps_vis = False
            if ps_vis:
                ind = 0
                batch_ind = 0
                viewer = InteractiveSMPLViewer(
                    smpl_aadecomp_model=body_model,
                    pose_skeleton=post_pred_x_0.joints_wrt_world[batch_ind, ind] * 1,
                    betas=post_pred_x_0.betas[batch_ind, ind],
                    transl=None,
                    initial_phis=post_pred_x_0.cos_sin_phis[batch_ind, ind],
                    global_orient=None,
                    device=device,
                    num_hybrik_joints=24,  # Standard for SMPL output from hybrik
                    leaf_thetas=None,
                    coordinate_transform=True,
                )
                viewer.show()

        post_x_0 = post_processed_batch.postprocess_denoise_traj(x_0, unmask=True)

        for i in range(bs):
            metrics = post_pred_x_0[i : i + 1]._compute_metrics(
                other=post_x_0[i : i + 1],
                body_model=body_model,
                device=device,
            )
            all_metrics[post_pred_x_0.metadata.take_name[i]] = metrics

        all_post_pred_x_0_list.append(post_pred_x_0)
        all_post_gt_x_0_list.append(post_x_0)

        DataClass: EgoTrainingDataType = get_class_from_path(
            EgoTrainingDataZoo[runtime_config.ego_training_data_name],
        )

        # vis using pyrender
        for i in range(bs):
            output_path = (
                Path("exp")
                / save_dir_name
                / f"{post_processed_batch.metadata.take_name[i]}.mp4"
            )
            output_path.parent.mkdir(exist_ok=True, parents=True)

            from egoallo.viz.smpl_pyrender_viewer import SMPLViewer

            viewer = SMPLViewer(
                smpl_family_model_basedir=runtime_config.smpl_family_model_basedir,
                smpl_family_meta_model_name=runtime_config.smpl_family_meta_model_name,
                gender=post_pred_x_0.metadata.gender,
            )
            viewer.render_list_sequences(
                [post_pred_x_0[i], post_x_0[i]],
                output_path,
                online_render=False,
            )

    pickle.dump(
        {
            "all_post_pred_x_0_list": all_post_pred_x_0_list,
            "all_post_gt_x_0_list": all_post_gt_x_0_list,
            "all_metrics": all_metrics,
        },
        open(str(Path("exp") / Path(save_dir_name) / "all_pred_and_gt_traj.pkl"), "wb"),
    )
