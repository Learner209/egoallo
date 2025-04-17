from __future__ import annotations
import pickle

import copy
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np

import torch.utils.data
from tqdm import tqdm

from egoallo.denoising.abs_aadecomp_traj import AbsoluteDenoiseTrajAADecomp
from egoallo.type_stubs import EgoTrainingDataType
from egoallo.sampling import quadratic_ts

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
from egoallo.config.inference.defaults import InferenceConfig
from egoallo.config.inference.egoexo import EgoExoInferenceConfig
from typing import Union
from egoallo.setup_logger import setup_logger
import dataclasses
from egoallo import training_utils

logger = setup_logger(output=None, name=__name__)


def test_fn(
    inference_config: Union[InferenceConfig, EgoExoInferenceConfig],
    device: torch.device,
):
    checkpoint_dir = inference_config.checkpoint_dir
    save_dir_name = inference_config.output_dir
    runtime_config: EgoAlloTrainConfig = load_runtime_config(
        checkpoint_dir,
    )

    # ! Override runtime config with inference config values
    for field in dataclasses.fields(type(inference_config)):
        if hasattr(runtime_config, field.name):
            setattr(
                runtime_config,
                field.name,
                getattr(inference_config, field.name),
            )

    denoiser, model_config = load_denoiser(
        checkpoint_dir,
        runtime_config,
    )
    denoiser = denoiser.to(device)

    bs = 1
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

    all_post_pred_x_0_dict = {}
    all_post_gt_x_0_dict = {}
    all_metrics = {}

    runtime_config.temporal_mask_ratio = 0.0
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
        if (
            inference_config.debug_max_iters
            and batch_idx > inference_config.debug_max_iters - 1
        ):
            break

        assert batch.joints_wrt_world.shape[0] == 1
        batch = batch.to(device)
        preprocessed_batch = copy.deepcopy(batch)
        post_processed_batch: EgoTrainingDataType = batch.postprocess()
        seq_len = batch.joints_wrt_world.shape[1]
        window_size = runtime_config.subseq_len
        overlap_size = int(window_size / 4)

        x_t_packed = torch.randn(
            (
                bs,
                seq_len,
                runtime_config.denoising.d_state,
            ),
            device=device,
        )

        canonical_overlap_weights = (
            torch.from_numpy(
                np.minimum(
                    overlap_size,
                    np.minimum(
                        np.arange(1, seq_len + 1),
                        np.arange(1, seq_len + 1)[::-1],
                    ),
                )
                / overlap_size,
            )
            .to(device)
            .to(torch.float32)
        )

        # Prepare window data in advance
        overlap_weights = torch.zeros((1, seq_len, 1), device=x_t_packed.device)

        window_data = []
        for start_t in range(0, seq_len, window_size - overlap_size):
            end_t = min(start_t + window_size, seq_len)
            overlap_weights_slice = canonical_overlap_weights[
                None,
                : end_t - start_t,
                None,
            ]
            overlap_weights[:, start_t:end_t, :] += overlap_weights_slice

            win_data = copy.deepcopy(post_processed_batch[:, start_t:end_t])
            # FIXME: this is a hack to follow the state machine of EgoTrainingData dataclass.
            win_data.metadata.stage = "raw"
            win_data = win_data.preprocess()

            window_data.append((start_t, end_t, win_data, overlap_weights_slice))

        # post_processed_batch = post_processed_batch[:, :seq_len]
        # preprocessed_batch = preprocessed_batch[:, :seq_len]
        # overlap_weights = torch.ones((1, seq_len, 1), device=x_t_packed.device)
        # for start_t in range(0, seq_len, window_size - overlap_size):
        #     end_t = min(start_t + window_size, seq_len)

        #     win_data = copy.deepcopy(post_processed_batch[:, start_t:end_t])
        #     # FIXME: this is a hack to follow the state machine of EgoTrainingData dataclass.
        #     win_data.metadata.stage = "raw"
        #     win_data = win_data.preprocess()

        #     window_data.append(
        #         (
        #             start_t,
        #             end_t,
        #             win_data,
        #             torch.ones_like(overlap_weights[:, start_t:end_t, :]),
        #         ),
        #     )

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
                    x_t_unpacked = runtime_config.denoising.unpack_traj(
                        x_t_packed[:, start_t:end_t, :],
                        metadata=win_data.metadata,
                        include_hands=runtime_config.model.include_hands,
                    )
                    occ_mask = (
                        (~win_data.visible_joints_mask).unsqueeze(-1).repeat(1, 1, 1, 3)
                    )
                    x_t_unpacked.joints_wrt_world = torch.where(
                        occ_mask,
                        x_t_unpacked.joints_wrt_world,
                        win_data.joints_wrt_world,
                    )
                    post_pred_x_0 = denoiser.forward(
                        x_t_unpacked=x_t_unpacked,
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

        post_x_0 = post_processed_batch.postprocess_denoise_traj(x_0, unmask=True)

        for i in range(bs):
            metrics = post_pred_x_0[i : i + 1]._compute_metrics(
                other=post_x_0[i : i + 1],
                body_model=body_model,
                device=device,
            )
            all_metrics[post_pred_x_0.metadata.take_name[i]] = metrics
            all_post_pred_x_0_dict[post_pred_x_0.metadata.take_name[i]] = post_pred_x_0
            all_post_gt_x_0_dict[post_x_0.metadata.take_name[i]] = post_x_0

        if inference_config.visualize_traj:
            if batch_idx > 20:
                continue

            for i in range(bs):
                output_path = (
                    save_dir_name / f"{post_processed_batch.metadata.take_name[i]}.mp4"
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
                    online_render=inference_config.online_render,
                )

    # Aggregate metrics across all takes by computing mean for each metric type
    agg_metrics = {}
    for take_metrics in all_metrics.values():
        for metric_name, metric_value in take_metrics.items():
            if metric_name not in agg_metrics:
                agg_metrics[metric_name] = []
            agg_metrics[metric_name].append(metric_value)
    agg_metrics = {
        metric: sum(values) / len(values) for metric, values in agg_metrics.items()
    }

    pickle.dump(
        {
            "all_post_pred_x_0_dict": all_post_pred_x_0_dict,
            "all_post_gt_x_0_dict": all_post_gt_x_0_dict,
            "all_metrics": all_metrics,
            "agg_metrics": agg_metrics,
        },
        open(str(Path(save_dir_name) / "all_pred_and_gt_traj_and_metrics.pkl"), "wb"),
    )


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig
    from hydra.utils import instantiate

    training_utils.ipdb_safety_net()

    @hydra.main(version_base="1.3", config_path="../../../config")
    def test(cfg: DictConfig) -> None:
        inference_config: Union[InferenceConfig, EgoExoInferenceConfig] = instantiate(
            cfg.inference,
        )
        test_fn(inference_config, device=torch.device("cuda"))

    test()
