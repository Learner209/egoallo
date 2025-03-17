"""Translate data from HuMoR-style npz format to an hdf5-based one."""

import queue
import time
from pathlib import Path

import h5py
import torch.cuda
import tyro

from egoallo import training_utils
from egoallo.data.dataclass import EgoTrainingData

# faulthandler.enable()
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


def main(
    smplh_model_path: Path = Path("assets/smpl_based_model/smplh/SMPLH_NEUTRAL.pkl"),
    data_npz_dirs: list[Path] = [Path("")],
    output_file: Path = Path(""),
    output_list_file: Path = Path(""),
    include_hands: bool = True,
) -> None:
    assert torch.cuda.is_available()

    task_queue = queue.Queue[Path]()

    for data_npz_dir in data_npz_dirs:
        for path in data_npz_dir.glob("**/*.npz"):
            task_queue.put_nowait(path)

    total_count = task_queue.qsize()
    start_time = time.time()

    output_hdf5 = h5py.File(output_file, "w")
    file_list: list[str] = []

    def worker(device_idx: int) -> None:
        while True:
            try:
                npz_path = task_queue.get_nowait()
            except queue.Empty:
                break

            print(f"Processing {npz_path} on device {device_idx}...")
            for test_data, (start_idx, end_idx) in EgoTrainingData.load_from_npz(
                smplh_model_path,
                npz_path,
                include_hands=include_hands,
                device=torch.device("cpu"),
            ):
                assert test_data.metadata.scope == "test", (
                    "The data should be test data."
                )

                # output_name = npz_path.stem + ".mp4"
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
                        group_name = str(npz_path).partition(str(data_npz_dir) + "/")[
                            2
                        ]  # get original path name
                        group_name = (
                            str(Path(group_name).with_suffix(""))
                            + f"_{start_idx}_{end_idx}"
                            + Path(group_name).suffix
                        )
                        break
                else:
                    raise ValueError(
                        f"NPZ file {npz_path} not found in any input directory",
                    )

                print(f"Writing to group {group_name} on {device_idx}...")
                group = output_hdf5.create_group(group_name)
                file_list.append(group_name)

                for k, v in vars(test_data).items():
                    # No need to write the mask, which will always be ones when we
                    # load from the npz file!
                    if v is None or not isinstance(v, torch.Tensor):
                        continue

                    # Chunk into 32 timesteps at a time.
                    if k not in ("contacts", "mask"):
                        assert v.dtype == torch.float32, f"{k} {v.dtype}"

                    if v.shape[0] == test_data.T_world_cpf.shape[0]:
                        chunks = (min(32, v.shape[0]),) + v.shape[1:]
                    else:
                        assert v.shape[0] == 1
                        chunks = v.shape
                    group.create_dataset(k, data=v.numpy(force=True), chunks=chunks)

            print(
                f"Finished ~{total_count - task_queue.qsize()}/{total_count},",
                f"{(total_count - task_queue.qsize()) / total_count * 100:.2f}% in",
                f"{time.time() - start_time} seconds",
            )

    # workers = [
    #     threading.Thread(target=worker, args=(0,))
    #     # for i in range(torch.cuda.device_count())
    #     for i in range(5)
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
