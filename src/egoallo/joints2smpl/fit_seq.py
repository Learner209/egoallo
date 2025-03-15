from __future__ import division
from __future__ import print_function

import os
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never
from typing import Literal

import h5py
import joblib
import numpy as np
import smplx
import torch
import trimesh
import tyro

# from egoallo import fncsmpl
from egoallo import fncsmpl_library as fncsmpl
from egoallo.data.dataclass import EgoTrainingData
from egoallo.joints2smpl import joints2smpl_config
from egoallo.joints2smpl import smplify
from egoallo.utils.setup_logger import setup_logger
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

logger = setup_logger(output=None, name=__name__)


@dataclass
class Joints2SmplFittingConfig:
    # Batch settings
    # ! IMPORTANT: batch_size should be 1, the batch_size can be mnodified to incoporate multiple temporal frames to optimize together, however, this could be changed to using jaxls afterwards since the latter provides a more structured optimization framework
    batch_size: int = 1

    # Optimization settings
    num_smplify_iters: int = 100

    # Hardware settings
    cuda: bool = True
    gpu_ids: int = 0

    # Joint settings
    num_joints: int = 22
    joint_category: Literal["AMASS"] = "AMASS"
    fix_foot: bool = False

    # Data paths
    data_folder: Path = Path("./demo/demo_data/")
    save_folder: Path = Path("./demo/demo_results/")
    files: str = "test_motion.npy"


def main(opt: Joints2SmplFittingConfig):
    print(opt)

    device = torch.device("cuda:" + str(opt.gpu_ids) if opt.cuda else "cpu")
    print(joints2smpl_config.SMPL_MODEL_DIR)
    smplmodel = smplx.create(
        joints2smpl_config.SMPL_MODEL_DIR,
        model_type="smpl",
        gender="neutral",
        ext="pkl",
        batch_size=opt.batch_size,
    ).to(device)

    # ## --- load the mean pose as original ----
    smpl_mean_file = joints2smpl_config.SMPL_MEAN_FILE

    file = h5py.File(smpl_mean_file, "r")
    init_mean_pose = torch.from_numpy(file["pose"][:]).unsqueeze(0).float()
    init_mean_shape = torch.from_numpy(file["shape"][:]).unsqueeze(0).float()
    cam_trans_zero = torch.Tensor([0.0, 0.0, 0.0]).to(device)
    pred_pose = torch.zeros(opt.batch_size, 72).to(device)
    pred_betas = torch.zeros(opt.batch_size, 10).to(device)
    pred_cam_t = torch.zeros(opt.batch_size, 3).to(device)
    keypoints_3d = torch.zeros(opt.batch_size, opt.num_joints, 3).to(device)

    # # #-------------initialize SMPLify
    smplify_model = smplify.SMPLify3D(
        smplxmodel=smplmodel,
        batch_size=opt.batch_size,
        joints_category=opt.joint_category,
        num_iters=opt.num_smplify_iters,
        device=device,
    )
    # print("initialize SMPLify3D done!")

    purename = os.path.splitext(opt.files)[0]
    # --- load data ---
    data = np.load(opt.data_folder / (purename + ".npy"))

    dir_save = opt.save_folder / purename
    if not os.path.isdir(dir_save):
        os.makedirs(dir_save, exist_ok=True)

    # run the whole seqs
    num_seqs = data.shape[0]

    for idx in range(num_seqs):
        print(f"idx={idx}")

        # ! IMPORTANT: *1.2 #scale problem [check first]
        joints3d = data[idx]
        keypoints_3d[0, :, :] = torch.Tensor(joints3d).to(device).float()

        if idx == 0:
            pred_betas[0, :] = init_mean_shape
            pred_pose[0, :] = init_mean_pose
            pred_cam_t[0, :] = cam_trans_zero
        else:
            data_param = joblib.load(dir_save / ("%04d" % (idx - 1) + ".pkl"))
            pred_betas[0, :] = torch.from_numpy(data_param["beta"]).unsqueeze(0).float()
            pred_pose[0, :] = torch.from_numpy(data_param["pose"]).unsqueeze(0).float()
            pred_cam_t[0, :] = torch.from_numpy(data_param["cam"]).unsqueeze(0).float()

        if opt.joint_category == "AMASS":
            confidence_input = torch.ones(opt.num_joints)
            # make sure the foot and ankle
            if opt.fix_foot is True:
                confidence_input[7] = 1.5
                confidence_input[8] = 1.5
                confidence_input[10] = 1.5
                confidence_input[11] = 1.5
        else:
            print("Such category not settle down!")

        # ----- from initial to fitting -------
        (
            new_opt_vertices,
            new_opt_joints,
            new_opt_pose,
            new_opt_betas,
            new_opt_cam_t,
            new_opt_joint_loss,
        ) = smplify_model(
            pred_pose.detach(),
            pred_betas.detach(),
            pred_cam_t.detach(),
            keypoints_3d,
            conf_3d=confidence_input.to(device),
            seq_ind=idx,
        )

        # # -- save the results to ply---
        outputp = smplmodel(
            betas=new_opt_betas,
            global_orient=new_opt_pose[:, :3],
            body_pose=new_opt_pose[:, 3:],
            transl=new_opt_cam_t,
            return_verts=True,
        )
        mesh_p = trimesh.Trimesh(
            vertices=outputp.vertices.detach().cpu().numpy(force=True).squeeze(),
            faces=smplmodel.faces,
            process=False,
        )
        mesh_p.export(dir_save / ("%04d" % idx + ".ply"))

        # save the pkl
        param = {}
        param["beta"] = new_opt_betas.detach().cpu().numpy(force=True)
        param["pose"] = new_opt_pose.detach().cpu().numpy(force=True)
        param["cam"] = new_opt_cam_t.detach().cpu().numpy(force=True)

        # save the root position
        # shape of keypoints_3d is torch.Size([1, 22, 3]) and root is the first one
        root_position = keypoints_3d[0, 0, :].detach().cpu().numpy(force=True)
        print(f"root at {root_position}, shape of keypoints_3d is {keypoints_3d.shape}")
        param["root"] = root_position

        joblib.dump(param, dir_save / ("%04d" % idx + ".pkl"), compress=3)


# @jaxtyped(typechecker=typeguard.typechecked)
def joints2smpl_fit_seq(
    opt: Joints2SmplFittingConfig,
    body_model: fncsmpl.SmplhModel,
    batch_size: int,
    joints3d: Float[Tensor, "batch_size 22 3"],
    output_dir: Path,
):
    # FIXME: the batch_size is set to passed in param instead of opt.batch_size, this is a temporary fix.
    device = torch.device("cuda:" + str(opt.gpu_ids) if opt.cuda else "cpu")
    smplmodel = smplx.create(
        joints2smpl_config.SMPL_MODEL_DIR,
        model_type="smpl",
        gender="neutral",
        ext="pkl",
        batch_size=1,
    ).to(device)

    smpl_mean_file = joints2smpl_config.SMPL_MEAN_FILE

    file = h5py.File(smpl_mean_file, "r")
    init_mean_pose = torch.from_numpy(file["pose"][:]).unsqueeze(0).float()
    init_mean_shape = torch.from_numpy(file["shape"][:]).unsqueeze(0).float()
    cam_trans_zero = torch.Tensor([0.0, 0.0, 0.0]).to(device)

    pred_pose = torch.zeros(batch_size, 72).to(device)
    pred_betas = torch.zeros(batch_size, 10).to(device)
    pred_cam_t = torch.zeros(batch_size, 3).to(device)
    torch.zeros(batch_size, 3).to(device)
    pred_joints = torch.zeros(batch_size, opt.num_joints, 3).to(device)

    keypoints_3d = torch.zeros(batch_size, opt.num_joints, 3).to(device)

    smplify_model = smplify.SMPLify3D(
        smplxmodel=smplmodel,
        batch_size=1,
        joints_category=opt.joint_category,
        num_iters=opt.num_smplify_iters,
        device=device,
    )

    # run the whole seqs
    num_seqs = batch_size

    # The for iteration is not efficient, however, we did want to have initial configuration to be set to the previous sequence.
    for idx in tqdm(range(num_seqs)):
        # ! IMPORTANT: *1.2 #scale problem [check first]
        _joints3d = joints3d[idx]
        keypoints_3d[idx, :, :] = torch.Tensor(_joints3d).to(device).float()

        if idx == 0:
            pred_betas[idx, :] = init_mean_shape
            pred_pose[idx, :] = init_mean_pose
            pred_cam_t[idx, :] = cam_trans_zero
        else:
            pred_betas[idx, :] = pred_betas[idx - 1, :]
            pred_pose[idx, :] = pred_pose[idx - 1, :]
            pred_cam_t[idx, :] = pred_cam_t[idx - 1, :]

        if opt.joint_category == "AMASS":
            confidence_input = torch.ones(opt.num_joints)
            # make sure the foot and ankle
            if opt.fix_foot is True:
                confidence_input[7] = 1.5
                confidence_input[8] = 1.5
                confidence_input[10] = 1.5
                confidence_input[11] = 1.5
        else:
            assert_never(opt.joint_category)

        # ----- from initial to fitting -------
        (
            new_opt_vertices,
            new_opt_joints,
            new_opt_pose,
            new_opt_betas,
            new_opt_cam_t,
            new_opt_joint_loss,
        ) = smplify_model(
            pred_pose[idx : idx + 1].detach(),
            pred_betas[idx : idx + 1].detach(),
            pred_cam_t[idx : idx + 1].detach(),
            keypoints_3d[idx : idx + 1],
            conf_3d=confidence_input.to(device),
            seq_ind=idx,
        )

        # # -- save the results to ply---
        outputp = smplmodel(
            betas=new_opt_betas,
            global_orient=new_opt_pose[:, :3],
            body_pose=new_opt_pose[:, 3:],
            transl=new_opt_cam_t,
            return_verts=True,
        )
        mesh_p = trimesh.Trimesh(
            vertices=outputp.vertices.detach().cpu().numpy(force=True).squeeze(),
            faces=smplmodel.faces,
            process=False,
        )
        mesh_p.export(output_dir / ("%04d" % idx + ".ply"))

        pred_pose[idx] = new_opt_pose
        pred_betas[idx] = new_opt_betas
        pred_cam_t[idx] = new_opt_cam_t
        pred_joints[idx] = new_opt_joints[:, :22]

    # Prepare output data
    sequence_data = {
        "poses": pred_pose[:, :66].numpy(force=True),
        "trans": pred_cam_t.numpy(force=True),
        "betas": pred_betas.mean(dim=0).numpy(force=True),
        "gender": "male",
        "fps": 30,
        "joints": pred_joints.numpy(force=True),
        "contacts": np.ones(
            (pred_joints.shape[0], 22),
        ),  # contacts server as a boolean label, but for compatiblity with `load_from_npz` function, convert it to flaot32
        "pose_hand": np.zeros((pred_joints.shape[0], 90)),
        "root_orient": pred_pose[:, :3].numpy(force=True),
        "pose_body": pred_pose[:, 3:66].numpy(force=True),
    }
    output_path = output_dir / "sequence.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **sequence_data)

    return EgoTrainingData.load_from_npz(
        body_model=body_model,
        path=output_path,
        include_hands=True,
    )


if __name__ == "__main__":
    tyro.cli(main)

# ---load predefined something
