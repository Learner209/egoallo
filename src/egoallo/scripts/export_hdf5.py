"""Translate data from HuMoR-style npz format to an hdf5-based one."""

from dataclasses import dataclass
import dataclasses
import queue
from tarfile import AbsoluteLinkError
import threading
import time
from pathlib import Path

import h5py
import torch
import torch.cuda
import tyro
import numpy as np
from egoallo import network
from egoallo import fncsmpl
from egoallo.data.dataclass import EgoTrainingData
from egoallo import training_utils
import faulthandler

# faulthandler.enable()


def main(
    smplh_npz_path: Path = Path("./assets/smpl_based_model/smplh/neutral/model.npz"),
    data_npz_dirs: list[Path] = [Path("")],
    output_file: Path = Path(""),
    output_list_file: Path = Path(""),
    include_hands: bool = True,
) -> None:
    body_model = fncsmpl.SmplhModel.load(smplh_npz_path)

    # training_utils.ipdb_safety_net()
    assert torch.cuda.is_available()

    task_queue = queue.Queue[Path]()

    # Collect all npz files from all input directories
    for data_npz_dir in data_npz_dirs:
        for path in data_npz_dir.glob("**/*.npz"):
            task_queue.put_nowait(path)

    total_count = task_queue.qsize()
    start_time = time.time()

    output_hdf5 = h5py.File(output_file, "w")
    file_list: list[str] = []

    def worker(device_idx: int) -> None:
        device_body_model = body_model.to("cuda:" + str(device_idx))

        while True:
            try:
                npz_path = task_queue.get_nowait()
            except queue.Empty:
                break

            print(f"Processing {npz_path} on device {device_idx}...")
            train_data = EgoTrainingData.load_from_npz(
                device_body_model, npz_path, include_hands=include_hands
            )

            output_name = npz_path.stem + ".mp4"
            # FIXME: the from_ego_data function is an instance moethod of network.DenoisingConfig
            denoising_config = network.DenoisingConfig(
                denoising_mode="absolute", include_hands=True
            )
            train_traj = denoising_config.from_ego_data(train_data)

            # Adaptive sampling if sequence is longer than 1500 frames
            # if len(train_traj.t_world_root) > 1500:
            #     # Calculate stride to get under 1500 frames
            #     stride = len(train_traj.t_world_root) // 1499 + 1
            #     train_traj = train_traj[::stride]

            # output_path = Path("./exp/debug_frame_rate_diff/")
            # output_path.mkdir(parents=True, exist_ok=True)

            # if not (output_path / output_name).exists():
            #     EgoTrainingData.visualize_ego_training_data(
            #         train_traj,
            #         body_model,
            #         output_path=str(output_path / output_name),
            #     )

            for data_npz_dir in data_npz_dirs:
                if str(data_npz_dir) in str(npz_path):
                    group_name = str(npz_path).partition(str(data_npz_dir) + "/")[2]
                    break
            else:
                raise ValueError(
                    f"NPZ file {npz_path} not found in any input directory"
                )

            print(f"Writing to group {group_name} on {device_idx}...")
            group = output_hdf5.create_group(group_name)
            file_list.append(group_name)

            for k, v in vars(train_data).items():
                # No need to write the mask, which will always be ones when we
                # load from the npz file!
                if v is None or not isinstance(v, torch.Tensor):
                    continue

                # Chunk into 32 timesteps at a time.
                if k != "contacts" and k != "mask":
                    assert v.dtype == torch.float32, f"{k} {v.dtype}"

                if v.shape[0] == train_data.T_world_cpf.shape[0]:
                    chunks = (min(32, v.shape[0]),) + v.shape[1:]
                else:
                    assert v.shape[0] == 1
                    chunks = v.shape
                group.create_dataset(k, data=v.numpy(force=True), chunks=chunks)

            print(
                f"Finished ~{total_count - task_queue.qsize()}/{total_count},",
                f"{(total_count - task_queue.qsize())/total_count * 100:.2f}% in",
                f"{time.time() - start_time} seconds",
            )

    # workers = [
    #     threading.Thread(target=worker, args=(0,))
    #     # for i in range(torch.cuda.device_count())
    #     for i in range(40)
    # ]
    # for w in workers:
    #     w.start()
    # for w in workers:
    #     w.join()

    # Single-threaded version
    worker(0)
    output_list_file.write_text("\n".join(file_list))


if __name__ == "__main__":
    training_utils.ipdb_safety_net()
    tyro.cli(main)
