import argparse
import glob
import os
import os.path as osp
import random
import re
import shutil
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Optional
from typing import Tuple

import imageio
import numpy as np
import torch
import typeguard
from egoallo.utils.setup_logger import setup_logger
from jaxtyping import Float
from jaxtyping import jaxtyped
from torch import Tensor
from yacs.config import CfgNode as CN
# from egoego.config import make_cfg, CONFIG_FILE
# from egoego.utils.setup_logger import setup_logger
# local_config_file = CONFIG_FILE
# CFG = make_cfg(config_name="defaults", config_file=local_config_file, cli_args=[])

local_logger = setup_logger(output=None, name=__name__)


def find_numerical_key_in_dict(d):
    res = []
    for k in d.keys():
        try:
            int(k)
            res.append(int(k))
        except ValueError:
            pass
    return res


def deterministic():
    np.random.seed(0)
    random.seed(0)
    torch.use_deterministic_algorithms(True)
    torch.random.manual_seed(0)


def debug_on_error(debug, logger=local_logger):
    if debug:

        def exception_handler(type, value, traceback):
            logger.critical(
                "\nAn exception occurred while executing the requested command:",
            )
            hr(logger=logger)
            sys.__excepthook__(type, value, traceback)
            logger.critical("\nStarting interactive debugging session: ")
            hr(logger=logger)
            ipdb.post_mortem(traceback)

        sys.excepthook = exception_handler


def hr(logger=local_logger, char="-", width=None, **kwargs):
    if width is None:
        width = shutil.get_terminal_size()[0]
    logger.critical(char * width)


def signal_handler(signal_received, frame, logger, opt):
    logger.warning(f"Signal {signal_received} received, terminating the process.")
    # logger.debug(f"Current line number: {frame.f_lineno}")
    # logger.debug(f"Current function: {frame.f_code.co_name}")
    # logger.debug(f"Local variables: {frame.f_locals}")
    # logger.debug(f"Global variables: {frame.f_globals}")
    # logger.debug(f"Reference to the previous frame: {frame.f_back}")
    # logger.debug(f"Code object that represents the compiles: {frame.f_code}")

    # logger.info(opt)

    # Find the last valid ckpts in the ckpt_save_path.
    ckpt_save_dir = opt.io.diffusion.ckpt_save_path
    file_list = glob.glob(osp.join(ckpt_save_dir, "model_*.pt"))
    if file_list:
        iter_exist = []
        for file_ in file_list:
            iter_current = re.findall(r"model_(\d+).pt", file_)
            iter_exist.append(int(iter_current[0]))
        init_iter = max(iter_exist)
        init_path = os.path.join(ckpt_save_dir, "model_{}.pt".format(init_iter))
    else:
        init_iter = 0
        init_path = None
    logger.info(f"Last val_step: {init_iter:4,d} ckpt path: {init_path}")

    # Construct the val_file in YAML format.
    config_file = opt.config_file
    is_train = "train" in opt.phases
    if is_train:
        train_cfg = CN.load_cfg(open(config_file, "r"))
        val_cfg = train_cfg.clone()
        val_cfg.defrost()
        val_cfg.datasets.test.ckpt_weight_path = init_path
        val_cfg.datasets.inference.ckpt_weight_path = init_path
        val_cfg.logging.wandb.mode = "disabled"
        val_cfg.phases = ["vis"]
        val_cfg.freeze()
        cfg_basename = osp.splitext(osp.basename(config_file))[0]
        val_cfg_save_path = osp.join(
            opt.io.diffusion.val_save_path,
            f"{cfg_basename}_val.yaml",
        )
        with open(val_cfg_save_path, "w") as f:
            with redirect_stdout(f):
                print(val_cfg.dump())
        logger.info(
            "Corresponding val config file has been saved to: {}".format(
                val_cfg_save_path,
            ),
        )

    exp_save_root_dir = opt.io.diffusion.project_exp_name
    response = input(
        f"Are you sure you want to delete the directory '{exp_save_root_dir}'? (y/n): ",
    )

    if response.lower() == "yes" or response.lower() == "y":
        try:
            # Remove the directory using shutil.rmtree which deletes a directory and all its contents
            shutil.rmtree(exp_save_root_dir)
            logger.info(f"Directory '{exp_save_root_dir}' has been deleted.")
        except Exception as e:
            logger.info(f"Error: {e}")

    sys.exit(0)


def convert_to_dict(cfg_node, key_list=[]):
    _VALID_TYPES = {tuple, list, str, int, float, bool}
    """ Convert a config node to dictionary """
    if not isinstance(cfg_node, CN):
        if type(cfg_node) not in _VALID_TYPES:
            local_logger.warning(
                "Key {} with value {} is not a valid type; valid types: {}".format(
                    ".".join(key_list),
                    type(cfg_node),
                    _VALID_TYPES,
                ),
            )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
        return cfg_dict


def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == "type1":
        lr_adjust = {
            2: args.learning_rate * 0.5**1,
            4: args.learning_rate * 0.5**2,
            6: args.learning_rate * 0.5**3,
            8: args.learning_rate * 0.5**4,
            10: args.learning_rate * 0.5**5,
        }
    elif args.lradj == "type2":
        lr_adjust = {
            5: args.learning_rate * 0.5**1,
            10: args.learning_rate * 0.5**2,
            15: args.learning_rate * 0.5**3,
            20: args.learning_rate * 0.5**4,
            25: args.learning_rate * 0.5**5,
        }
    else:
        lr_adjust = {}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_model = None

    def __call__(self, val_loss, epoch):
        score = -val_loss
        should_save = False
        if self.best_score is None:
            self.best_score = score
            should_save = True
        elif score < self.best_score + self.delta:
            self.counter += 1
            local_logger.info(
                f"Epoch: [{epoch:4,d}], EarlyStopping counter: {self.counter} out of {self.patience}",
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            should_save = True
            self.counter = 0
        return should_save


def load_args(filename):
    with open(filename, "r") as f:
        args = json.load(f)
    return args


def string_split(str_for_split):
    str_no_space = str_for_split.replace(" ", "")
    str_split = str_no_space.split(",")
    value_list = [eval(x) for x in str_split]

    return value_list


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Invalid boolen type")


def images_to_video(img_folder, output_vid_file, ext=".png"):
    os.makedirs(img_folder, exist_ok=True)

    command = [
        "ffmpeg",
        "-r",
        "30",
        "-y",
        "-threads",
        "16",
        "-i",
        f"{img_folder}/%06d{ext}",
        "-profile:v",
        "baseline",
        "-level",
        "3.0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-v",
        "error",
        output_vid_file,
    ]

    # command = [
    #     'ffmpeg', '-r', '30', '-y', '-threads', '16', '-i', f'{img_folder}/%05d.png', output_vid_file,
    # ]

    local_logger.debug(f'Running "{" ".join(command)}"')
    subprocess.call(command)


def images_to_video_w_imageio(img_folder, output_vid_file):
    img_files = os.listdir(img_folder)
    img_files.sort()
    im_arr = []
    for img_name in img_files:
        img_path = os.path.join(img_folder, img_name)
        im = imageio.imread(img_path)
        im_arr.append(im)

    im_arr = np.asarray(im_arr)
    imageio.mimwrite(output_vid_file, im_arr, fps=30, quality=8)


def get_device(device: Optional[str | torch.device] = None) -> torch.device:
    """Get torch device, defaulting to CUDA if available."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_path(path: str | Path) -> Path:
    """Convert path-like object to Path."""
    return Path(path) if not isinstance(path, Path) else path


@jaxtyped(typechecker=typeguard.typechecked)
def procrustes_align(
    points_y: Float[Tensor, "*batch time 3"],
    points_x: Float[Tensor, "*batch time 3"],
    fix_scale: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform Procrustes alignment between two point sets.

    Args:
        points_y: Target points (..., N, 3)
        points_x: Source points (..., N, 3)
        fix_scale: Whether to fix scale to 1

    Returns:
        Tuple of (scale, rotation, translation)
    """
    *dims, N, _ = points_y.shape
    device = points_y.device
    dtype = points_y.dtype
    N_tensor = torch.tensor(N, device=device, dtype=dtype)

    # Center points
    my = points_y.mean(dim=-2)
    mx = points_x.mean(dim=-2)
    y0 = points_y - my[..., None, :]
    x0 = points_x - mx[..., None, :]

    # Correlation
    C = torch.matmul(y0.transpose(-1, -2), x0) / N_tensor
    U, D, Vh = torch.linalg.svd(C, full_matrices=False)

    # Fix: Clone S before modifying to avoid in-place operation conflicts
    S = torch.eye(3, device=device, dtype=dtype).expand(*dims, 3, 3).clone()
    det = torch.det(U) * torch.det(Vh)
    S[..., -1, -1] = torch.where(det < 0, -1.0, 1.0)

    R = torch.matmul(U, torch.matmul(S, Vh))

    if fix_scale:
        s = torch.ones(*dims, 1, device=device, dtype=dtype)
    else:
        var = torch.sum(x0**2, dim=(-1, -2), keepdim=True) / N_tensor
        s = (
            torch.sum(D * S.diagonal(dim1=-2, dim2=-1), dim=-1, keepdim=True)
            / var[..., 0]
        )

    t = my - s * torch.matmul(R, mx[..., None])[..., 0]

    return s, R, t


if __name__ == "__main__":
    debug_on_error(debug=True, logger=local_logger)
    raise ValueError("This is a test error")
