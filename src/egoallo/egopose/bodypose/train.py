import json
import math
import os.path
import random
import warnings

import numpy as np
import torch
from egoallo.egopose.bodypose.utils.utils_option import parse as parse_opt
from egoallo.utils.setup_logger import setup_logger
from egoallo.utils.utils import convert_to_dict
from egoallo.utils.utils import debug_on_error
from egoallo.utils.utils import deterministic
from models.model_egoexo import ModelEgoExo4D
from models.select_model import define_Model
from tqdm import tqdm
from utils import utils_option as option

import wandb
from data.select_dataset import define_Dataset


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from egoallo.config import make_cfg, CONFIG_FILE
from egoallo.data.build import MultiEpochsDataLoader

local_config_file = CONFIG_FILE
CFG = make_cfg(config_name="defaults", config_file=local_config_file, cli_args=[])


logger = setup_logger(output=None, name=__name__)

debug_on_error(debug=True, logger=logger)


def inference(test_loader, model: ModelEgoExo4D, dry_run=False):
    model.init_test()
    inference_dict = {}

    for index, test_data in tqdm(
        enumerate(test_loader),
        total=len(test_loader),
        desc="Enumerating inference loader",
        ascii=" >=",
    ):
        model.feed_data(test_data, inference=True)

        model.test(inference=True)

        body_parms_pred = model.current_prediction()
        predicted_position = body_parms_pred["position"]

        t_ = np.array(test_data["t"][0]).tolist()
        preds_ = dict(zip(t_, predicted_position.tolist()))
        inference_dict[test_data["take_uid"][0]] = {
            "take_name": test_data["take_name"][0],
            "body": preds_,
        }

    return inference_dict


def test(
    test_loader,
    model: ModelEgoExo4D,
    epoch,
    current_step,
    test_step,
    dry_run=False,
):
    pos_error = []
    vel_error = []
    tasks = []
    inference_dict = {}
    gt_dict = {}

    for index, test_data in tqdm(
        enumerate(test_loader),
        total=len(test_loader),
        desc="Enumerating test loader",
        ascii=" >=",
    ):
        model.feed_data(test_data, inference=False)

        model.test()

        body_parms_pred = model.current_prediction()
        body_parms_gt = model.current_gt()

        pred_ = body_parms_pred["position"]
        pred = pred_[model.visible[0].bool()]
        gt_ = body_parms_gt["position"]
        gt = gt_[model.visible[0].bool()]

        data = torch.sqrt(torch.sum(torch.square(gt - pred), axis=-1))
        pos_error_ = data.sum() / (data != 0).sum()

        vel_visible = (
            model.visible[0, 1:, ...].bool() & model.visible[0, :-1, ...].bool()
        )

        gt_vel = (gt_[1:, ...] - gt_[:-1, ...]) * 10
        gt_vel = gt_vel[vel_visible]

        pred_vel = (pred_[1:, ...] - pred_[:-1, ...]) * 10
        pred_vel = pred_vel[vel_visible]

        data_vel = torch.mean(
            torch.sqrt(torch.sum(torch.square(gt_vel - pred_vel), axis=-1)),
        )
        vel_error_ = data_vel.sum() / (data_vel != 0).sum()

        if model.visible.max() != 0:
            pos_error.append(pos_error_)
            vel_error.append(vel_error_)
            tasks.append(str(test_data["task"].numpy()[0])[0])

        visible = model.visible.squeeze(0).unsqueeze(2).repeat(1, 1, 3)  # T x 17 x 3
        visible[visible != 1] = torch.nan

        gt_nan = visible * gt_

        t_ = np.array(test_data["t"][0]).tolist()
        preds_ = dict(zip(t_, pred_.tolist()))
        gts_ = dict(zip(t_, gt_nan.tolist()))
        inference_dict[test_data["take_uid"][0]] = {
            "take_name": test_data["take_name"][0],
            "body": preds_,
        }
        gt_dict[test_data["take_uid"][0]] = {
            "take_name": test_data["take_name"][0],
            "body": gts_,
        }

    activities = {
        "1": "Cooking",
        "2": "Health",
        "3": "Campsite",
        "4": "Bike repair",
        "5": "Music",
        "6": "Basketball",
        "7": "Bouldering",
        "8": "Soccer",
        "9": "Dance",
    }
    for task_num in range(0, 10):
        task_ids = [i for i, j in enumerate(tasks) if j == str(task_num)]
        if len(task_ids) > 0:
            pos_ = torch.stack(pos_error)[task_ids].cpu().numpy()
            vel_ = torch.stack(vel_error)[task_ids].cpu().numpy()
            logger.info(
                "<epoch:{:3d}, iter:{:8,d}, Task: {}, Samples: {}, MPJPE[cm]: {:<.5f}, MPJVE [m/s]: {:<.5f}\n".format(
                    epoch,
                    current_step,
                    activities[str(task_num)],
                    len(task_ids),
                    (pos_.mean()) * 100,
                    (vel_.mean()),
                ),
            )

    pos_error = sum(pos_error) / len(pos_error)
    vel_error = sum(vel_error) / len(vel_error)
    wandb.log({"MPJPE": pos_error * 100, "MPJVE": vel_error, "test_step": test_step})
    # testing log
    logger.info(
        "<epoch:{:3d}, iter:{:8,d}, Average positional error [cm]: {:<.5f}, Average velocity error [m/s]: {:<.5f}\n".format(
            epoch,
            current_step,
            pos_error * 100,
            vel_error,
        ),
    )
    return inference_dict, gt_dict


def main(opt):
    """
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    """

    egopose_opt = opt.egopose
    wandb.init(
        project=egopose_opt.wandb.project_name,
        config=egopose_opt,
        mode=egopose_opt.wandb.mode,
        tags=egopose_opt.wandb.tags,
        name=egopose_opt.wandb.display_name,
        notes=egopose_opt.wandb.notes,
    )
    egopose_opt = convert_to_dict(egopose_opt)
    egopose_opt = parse_opt(egopose_opt)

    paths = (
        path for key, path in egopose_opt["path"].items() if "pretrained" not in key
    )
    if isinstance(paths, str):
        if not os.path.exists(paths):
            os.makedirs(paths)
    else:
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-

    init_iter, init_path_G = option.find_last_checkpoint(
        egopose_opt["path"]["models"],
        net_type="G",
    )
    if init_path_G is not None:
        egopose_opt["path"]["pretrained_netG"] = init_path_G
    current_step = init_iter

    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    option.save(egopose_opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    egopose_opt = option.dict_to_nonedict(egopose_opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    logger.info(option.dict2str(egopose_opt))
    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = egopose_opt["train"]["manual_seed"]
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    """
    # ----------------------------------------
    # Step--2 (create dataloader)
    # ----------------------------------------
    """
    phases = egopose_opt["phases"]
    phases = [
        phase.lower()
        for phase in phases
        if phase in list(egopose_opt["datasets"].keys())
    ]

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------

    for phase in phases:
        dataset_opt = egopose_opt["datasets"][phase]
        phase = dataset_opt["phase"]
        if phase == "train":
            train_set = define_Dataset(dataset_opt, opt)
            train_size = int(
                math.ceil(len(train_set) / dataset_opt["dataloader_batch_size"]),
            )
            logger.info(
                "Number of train datum: {:,d}, iters: {:,d}".format(
                    len(train_set),
                    train_size,
                ),
            )
            train_loader = MultiEpochsDataLoader(
                train_set,
                batch_size=dataset_opt["dataloader_batch_size"],
                shuffle=dataset_opt["dataloader_shuffle"],
                num_workers=dataset_opt["dataloader_num_workers"],
                drop_last=True,
                pin_memory=True,
            )
        elif phase == "test":
            test_set = define_Dataset(dataset_opt, opt)
            test_loader = MultiEpochsDataLoader(
                test_set,
                batch_size=dataset_opt["dataloader_batch_size"],
                shuffle=False,
                num_workers=1,
                drop_last=True,
                pin_memory=True,
            )
        elif phase == "inference":
            test_set = define_Dataset(dataset_opt, opt)
            test_loader = MultiEpochsDataLoader(
                test_set,
                batch_size=dataset_opt["dataloader_batch_size"],
                shuffle=False,
                num_workers=1,
                drop_last=True,
                pin_memory=True,
            )
        else:
            # raise NotImplementedError("Phase [%s] is not recognized." % phase)
            warnings.warn("Phase [%s] is not recognized." % phase, UserWarning)

    """
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    """

    model = define_Model(egopose_opt)

    if egopose_opt["merge_bn"] and current_step > egopose_opt["merge_bn_startpoint"]:
        logger.info("^_^ -----merging bnorm----- ^_^")
        model.merge_bnorm_test()

    logger.info(model.info_network())

    model.init_train()
    logger.info(model.info_params())

    if egopose_opt["instantiate"]["test"]["dry_run"] and (
        "test" in phases or "train" in phases
    ):
        test(test_loader, model, 0, current_step, 0, dry_run=False)

    """
    # ----------------------------------------
    # Step--4 (main phase)
    # ----------------------------------------
    """

    if "train" in phases:
        test_step = 0

        for epoch in range(100000):  # keep running
            for i, train_data in enumerate(train_loader):
                current_step += 1
                # -------------------------------
                # 1) feed patch pairs
                # -------------------------------

                model.feed_data(train_data)

                # -------------------------------
                # 2) optimize parameters
                # -------------------------------
                model.optimize_parameters(current_step)

                # -------------------------------
                # 3) update learning rate
                # -------------------------------
                model.update_learning_rate(current_step)
                wandb_dict = model.log_dict
                wandb_dict["train_step"] = current_step
                wandb.log(wandb_dict)

                # -------------------------------
                # merge bnorm
                # -------------------------------
                if (
                    egopose_opt["merge_bn"]
                    and egopose_opt["merge_bn_startpoint"] == current_step
                ):
                    logger.info("^_^ -----merging bnorm----- ^_^")
                    model.merge_bnorm_train()
                    model.print_network()

                # -------------------------------
                # 4) training information
                # -------------------------------
                if current_step % egopose_opt["train"]["checkpoint_print"] == 0:
                    logs = model.current_log()  # such as loss
                    message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                        epoch,
                        current_step,
                        model.current_learning_rate(),
                    )
                    for k, v in logs.items():  # merge log information into message
                        message += "{:s}: {:.3e} ".format(k, v)
                    logger.info(message)

                # -------------------------------
                # 5) save model
                # -------------------------------
                if current_step % egopose_opt["train"]["checkpoint_save"] == 0:
                    logger.info("Saving the model.")
                    model.save(current_step)

                # -------------------------------
                # 6) testing
                # -------------------------------
                if current_step % egopose_opt["train"]["checkpoint_test"] == 0:
                    test_step += 1
                    test(test_loader, model, epoch, current_step, test_step)

        logger.info("Saving the final model.")
        model.save("latest")
        logger.info("End of training.")

    """
    # ----------------------------------------
    # Step--4 (test phase)
    # ----------------------------------------
    """
    if "test" in phases:
        epoch = init_iter
        test_step = init_iter
        inference_dict, gt_dict = test(
            test_loader,
            model,
            epoch,
            current_step,
            test_step,
        )
        dataset_opt = egopose_opt["datasets"]["test"]
        pred_path = os.path.join(
            egopose_opt["path"]["images"],
            dataset_opt["split"] + "_pred.json",
        )
        gt_path = os.path.join(
            egopose_opt["path"]["images"],
            dataset_opt["split"] + "_gt.json",
        )

        with open(pred_path, "w") as fp:
            json.dump(inference_dict, fp)
        with open(gt_path, "w") as fp:
            json.dump(gt_dict, fp)

    """
    # ----------------------------------------
    # Step--4 (inference phase)
    # ----------------------------------------
    """
    inference_dict = inference(test_loader, model)
    dataset_opt = egopose_opt["datasets"]["inference"]
    pred_path = os.path.join(
        egopose_opt["path"]["images"],
        dataset_opt["split"] + "_pred.json",
    )

    with open(pred_path, "w") as fp:
        json.dump(inference_dict, fp)
    logger.info("Done with inference")


if __name__ == "__main__":
    deterministic()
    main(CFG)
