"""Translate data from HuMoR-style npz format to an hdf5-based one."""

import queue
import threading
import time
from pathlib import Path

import h5py
import torch
import torch.cuda
import tyro

from egoallo import fncsmpl
from egoallo.data.amass_dataset import EgoTrainingData
from egoallo.setup_logger import setup_logger
from egoallo import training_utils

logger = setup_logger(output=None, name=__name__)

def main(
    smplh_npz_path: Path = Path("./assets/smpl_based_model/smplh/SMPLH_MALE.pkl"),
    data_npz_dir: Path = Path("./data/amass/processed"),
    output_file: Path = Path("./data/amass/processed.hdf5"),
    output_list_file: Path = Path("./data/amass/processed_files.txt"),
    include_hands: bool = True,
    base_group_name: str = "",
) -> None:
    body_model = fncsmpl.SmplhModel.load(smplh_npz_path)

    assert torch.cuda.is_available()

    task_queue = queue.Queue[Path]()
    print(f"Scanning for NPZ files in: {data_npz_dir}")
    for path in data_npz_dir.rglob("*.npz"):
        task_queue.put_nowait(path)

    total_count = task_queue.qsize()
    start_time = time.time()

    with h5py.File(output_file, "w") as output_hdf5:
        file_list: list[str] = []

        def worker(device_idx: int) -> None:
            device_body_model = body_model.to(f"cuda:{device_idx}")

            while True:
                try:
                    npz_path = task_queue.get_nowait()
                except queue.Empty:
                    break

                rel_path = npz_path.relative_to(data_npz_dir)
                video_output_path = Path("./logs/amass_viz").joinpath(rel_path).with_suffix('.mp4')
                video_output_path.parent.mkdir(parents=True, exist_ok=True)
                
                print(f"Processing {npz_path} on device {device_idx}...")
                train_data = EgoTrainingData.load_from_npz(
                    device_body_model, npz_path, include_hands=include_hands
                )
                # import ipdb; ipdb.set_trace()
                EgoTrainingData.visualize_ego_training_data(
                    train_data, body_model, str(video_output_path)
                )

                rel_path = npz_path.relative_to(data_npz_dir)
                group_name = str(rel_path.with_suffix(""))
                if base_group_name:
                    group_name = f"{base_group_name}/{group_name}"
                
                print(f"Writing to group {group_name} on {device_idx}...")
                group = output_hdf5.create_group(group_name)
                file_list.append(group_name)

                for k, v in vars(train_data).items():
                    if k == "mask":
                        continue

                    if v.dtype == torch.float32:
                        if v.shape[0] > 1:
                            chunks = (min(32, v.shape[0]),) + v.shape[1:]
                        else:
                            chunks = v.shape
                        
                        group.create_dataset(
                            k, 
                            data=v.numpy(force=True), 
                            chunks=chunks,
                            compression="gzip",
                            compression_opts=4
                        )

                progress = (total_count - task_queue.qsize()) / total_count * 100
                elapsed_time = time.time() - start_time
                print(
                    f"Progress: {progress:.2f}% ({total_count - task_queue.qsize()}/{total_count})",
                    f"Time elapsed: {elapsed_time:.2f}s"
                )

        # Original multi-threaded version:
        num_workers = min(torch.cuda.device_count(), 20)
        workers = [
            threading.Thread(target=worker, args=(i % torch.cuda.device_count(),))
            for i in range(num_workers)
        ]
        
        for w in workers:
            w.start()
        for w in workers:
            w.join()

        # # Single threaded version
        # device_idx = 0 # Use first GPU
        # worker(device_idx)

        if output_list_file:
            output_list_file.write_text("\n".join(sorted(file_list)))


if __name__ == "__main__":
    training_utils.ipdb_safety_net()
    tyro.cli(main)
