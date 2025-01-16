import torch
from torch.utils.data import Dataset, DataLoader
import os.path as osp
import joblib
import pandas as pd
import numpy as np
import json
import os
import shutil
import random
from PIL import Image
from tqdm import tqdm
import multiprocessing
import random
import threading
import glob
import skimage.io as io
import torchvision.transforms as T
from tqdm import tqdm
import matplotlib.pyplot as plt
# from egoallo.data.build import EgoTrainingData
from typing import List, Dict, Any, Tuple
from jaxtyping import Float, Bool, jaxtyped
import typeguard
from torch import Tensor
from egoallo.mapping import EGOEXO4D_BODYPOSE_TO_SMPLH_INDICES
from egoallo.utils.setup_logger import setup_logger
from egoallo.utilities import find_numerical_key_in_dict
from pathlib import Path

logger = setup_logger(output=None, name=__name__)
import joblib


random.seed(1)

class Dataset_EgoExo(Dataset):
    def __init__(self, config: Dict[str, Any]):
        super(Dataset_EgoExo,self).__init__()
        
        self.root = config['dataset_path']
        self.root_takes = os.path.join(self.root, "takes")
        self.split = config['split']
        self.root_poses = os.path.join(self.root, "annotations", "ego_pose",self.split, "body")
        self.use_pseudo = config['use_pseudo']
        self.coord = config["coord"]
        gt_ground_height_anno_dir = config["gt_ground_height_anno_dir"]
        self.gt_ground_height = json.load(open(Path(gt_ground_height_anno_dir) / f"ego_pose_gt_anno_{self.split}_public_height.json"))
        # self.slice_window =  config["window_size"]
        self.slice_window = 128
        
        manually_annotated_takes = os.listdir(os.path.join(self.root_poses,"annotation"))
        self.manually_annotated_takes = [take.split(".")[0] for take in manually_annotated_takes]
        if self.use_pseudo:
            pseudo_annotated_takes = os.listdir(os.path.join(self.root_poses,"automatic"))
            self.pseudo_annotated_takes = [take.split(".")[0] for take in pseudo_annotated_takes]
        
        self.cameras = os.listdir(self.root_poses.replace("body", "camera_pose"))
        self.metadata = json.load(open(os.path.join(self.root,"takes.json")))
        
        self.takes_uids = self.pseudo_annotated_takes if self.use_pseudo else self.manually_annotated_takes
        self.takes_metadata = {}

        self.valid_take_uid_save_label = "valid_takes_{}_use_manual.pkl".format(self.split) if not self.use_pseudo else "valid_takes_{}_use_pseudo.pkl".format(self.split)

        for take_uid in self.takes_uids:
            take_temp = self.get_metadata_take(take_uid)
            if take_temp and 'bouldering' not in take_temp['take_name']:
                self.takes_metadata[take_uid] = take_temp

        if not osp.exists(self.valid_take_uid_save_label):
            self.valid_take_uids = []
            manually = 0
            no_man = 0
            no_cam = 0
            no_cam_list = []

            cnt = 0
            # breakpoint()
            for take_uid in tqdm(self.takes_metadata, total=len(self.takes_metadata), desc="takes_metadata", ascii=' >='):
        
                # if cnt > 50:
                #     break
                cnt += 1
                if take_uid+".json" in self.cameras:
                    camera_json = json.load(open(os.path.join(self.root_poses.replace("body", "camera_pose"),take_uid+".json")))
                    take_name = camera_json['metadata']['take_name']
                    if not take_uid in self.manually_annotated_takes:
                        no_man +=1
                        if self.use_pseudo and take_uid in self.pseudo_annotated_takes:
                            pose_json = json.load(open(os.path.join(self.root_poses,"automatic",take_uid+".json")))
                            if (len(pose_json) > (self.slice_window +2)) and self.split == "train":
                                ann, traj = self.translate_poses(pose_json, camera_json, self.coord)
                                if len(traj) > (self.slice_window +2):
                                    self.valid_take_uids.append(take_uid)
                            elif self.split != "train":
                                ann, traj = self.translate_poses(pose_json, camera_json, self.coord)
                                self.valid_take_uids.append(take_uid)
                    elif take_uid in self.manually_annotated_takes:
                        pose_json = json.load(open(os.path.join(self.root_poses,"annotation",take_uid+".json")))
                        if (len(pose_json) > (self.slice_window +2)) and self.split == "train":
                            ann, traj = self.translate_poses(pose_json, camera_json, self.coord)
                            if len(traj) > (self.slice_window +2):
                                self.valid_take_uids.append(take_uid)
                        elif self.split != "train":
                            ann, traj = self.translate_poses(pose_json, camera_json, self.coord)
                            self.valid_take_uids.append(take_uid)

                else:
                    #print("No take uid {} in camera poses".format(take_uid))
                    no_cam += 1
                    no_cam_list.append(take_uid)

            #self.joint_names = ['left-wrist', 'left-eye', 'nose', 'right-elbow', 'left-ear', 'left-shoulder', 'right-hip', 'right-ear', 'left-knee', 'left-hip', 'right-wrist', 'right-ankle', 'right-eye', 'left-elbow', 'left-ankle', 'right-shoulder', 'right-knee']
            if len(self.valid_take_uids) > 0:
                joblib.dump(self.valid_take_uids, self.valid_take_uid_save_label)
            else:
                raise UserWarning("No valid takes found")
        else:
            self.valid_take_uids = joblib.load(self.valid_take_uid_save_label)
            logger.info(f"Loaded valid take uids from {len(self.valid_take_uids)}")
            # self.valid_take_uids = self.valid_take_uids

        self.joint_idxs = [i for i in range(17)] # 17 keypoints in total

        self.joint_names = ['nose','left-eye','right-eye','left-ear','right-ear','left-shoulder','right-shoulder','left-elbow','right-elbow','left-wrist','right-wrist','left-hip','right-hip','left-knee','right-knee','left-ankle','right-ankle']
        # self.single_joint = opt['single_joint']
        logger.info(f"Dataset lenght: {len(self.valid_take_uids)}")
        logger.info(f"Split: {self.split}")
        # logger.info('No Manually: {}'.format(no_man))
        # logger.info('No camera: {}'.format(no_cam))
        # logger.info('No camera list: {}'.format(no_cam_list))

    def translate_poses(self, anno, cams, coord):
        """
        Translate poses from EgoExo4D to global coordinates.
        NOTE: the raw ['camera_extrinsics'] are in global coordinates, which transforms world coordinates to camera coordinates.
        """
        trajectory = {}
        to_remove = []
        for key in cams.keys():
            if "aria" in key:
                aria_key =  key
                break
        first = next(iter(anno))
        first_cam =  cams[aria_key]['camera_extrinsics'][first]
        T_first_camera = np.eye(4)
        T_first_camera[:3, :] = np.array(first_cam)
        for frame in anno:
            try:
                current_anno = anno[frame]
                current_cam =  cams[aria_key]['camera_extrinsics'][frame]
                T_world_camera_ = np.eye(4)
                T_world_camera_[:3, :] = np.array(current_cam)
                
                if coord == 'global':
                    T_world_camera = np.linalg.inv(T_world_camera_)
                elif coord == 'aria':
                    T_world_camera = np.dot(T_first_camera,np.linalg.inv(T_world_camera_))
                else:
                    T_world_camera = T_world_camera_
                assert len(current_anno) != 0 
                for idx in range(len(current_anno)):
                    joints = current_anno[idx]["annotation3D"]
                    for joint_name in joints:
                        joint4d = np.ones(4)
                        joint4d[:3] = np.array([joints[joint_name]["x"], joints[joint_name]["y"], joints[joint_name]["z"]])
                        if coord == 'global':
                            new_joint4d = joint4d
                        elif coord == 'aria':
                            new_joint4d = T_first_camera.dot(joint4d)
                        else:
                            new_joint4d = T_world_camera_.dot(joint4d) #The skels always stay in 0,0,0 wrt their camera frame
                        joints[joint_name]["x"] = new_joint4d[0]
                        joints[joint_name]["y"] = new_joint4d[1]
                        joints[joint_name]["z"] = new_joint4d[2]
                    current_anno[idx]["annotation3D"] = joints
                traj = T_world_camera[:3,3]
                trajectory[frame] = traj
            except:
                to_remove.append(frame)
            anno[frame] = current_anno
        keys_old = list(anno.keys())
        for frame in keys_old:
            if frame in to_remove:
                del anno[frame]
        return anno, trajectory

    def get_metadata_take(self, uid):
        for take in self.metadata:
            if take["take_uid"]==uid:
                return take

    def parse_skeleton(self, skeleton):
        poses = []
        flags = []
        keypoints = skeleton.keys()
        for keyp in self.joint_names:
            if keyp in keypoints:
                flags.append(1) #visible
                poses.append([skeleton[keyp]['x'], skeleton[keyp]['y'], skeleton[keyp]['z']]) #visible
            else:
                flags.append(0) #not visible
                poses.append([-1,-1,-1]) #not visible
        return poses, flags

    @jaxtyped(typechecker=typeguard.typechecked)
    def _process_joints(
        self,
        data: Float[Tensor, "timesteps 17 3"],
        vis: Float[Tensor, "timesteps 17"],
        ground_height: float = 0.0,
        return_smplh_joints: bool = True,
        num_joints: int = 22,
        debug_vis: bool = False,
    ) -> Tuple[
        Float[Tensor, "timesteps {num_joints} 3"],
        Bool[Tensor, "timesteps {num_joints}"],
    ]:
        """Process joint data from annotations.

        Args:
            data: List of frame dictionaries containing body pose data
            return_smplh_joints: If True, converts joints from EgoExo4D (17 joints) to SMPLH format (22 body joints).
                Invalid mappings will be filled with zeros.
            debug_vis: If True, visualize joints using polyscope (for debugging)

        Returns:
            Tuple of:
            - joints_world: World coordinate joint positions (timesteps x J x 3) where J is 17 for EgoExo4D or 22 for SMPLH
            - visible: Joint visibility mask (timesteps x J) where J is 17 for EgoExo4D or 22 for SMPLH
        """
        # Initialize SMPLH tensors with NaN for positions and False for visibility
        if return_smplh_joints:
            T = data.shape[0]
            smplh_world = torch.full((T, 22, 3), float("nan"), dtype=torch.float32)
            smplh_visible = torch.zeros((T, 22), dtype=torch.bool)

            # Map joints using EGOEXO4D_BODYPOSE_TO_SMPLH_INDICES
            for smplh_idx, ego_idx in enumerate(EGOEXO4D_BODYPOSE_TO_SMPLH_INDICES):
                if ego_idx != -1:
                    # Valid mapping - copy data
                    smplh_world[:, smplh_idx] = data[:, ego_idx]
                    smplh_visible[:, smplh_idx] = vis[:, ego_idx]
            return smplh_world, smplh_visible
        else:
            return data, vis.bool()

    def __getitem__(self, index):
        take_uid = self.valid_take_uids[index]

        camera_json = json.load(open(os.path.join(self.root_poses.replace("body", "camera_pose"),take_uid+".json")))
        take_name = camera_json['metadata']['take_name']
        gt_ground_height = self.gt_ground_height[take_uid] if take_uid in self.gt_ground_height else 0.0
        if self.use_pseudo and take_uid in self.pseudo_annotated_takes:
            pose_json = json.load(open(os.path.join(self.root_poses,"automatic",take_uid+".json")))
            if (len(pose_json) > (self.slice_window +2)) and self.split == "train":
                ann, traj = self.translate_poses(pose_json, camera_json, self.coord)
            elif self.split != "train":
                ann, traj = self.translate_poses(pose_json, camera_json, self.coord)
        elif take_uid in self.manually_annotated_takes:
            pose_json = json.load(open(os.path.join(self.root_poses,"annotation",take_uid+".json")))
            if (len(pose_json) > (self.slice_window +2)) and self.split == "train":
                ann, traj = self.translate_poses(pose_json, camera_json, self.coord)
            elif self.split != "train":
                ann, traj = self.translate_poses(pose_json, camera_json, self.coord)
        else:
            raise UserWarning("Take uid {} not found in any annotation folder".format(take_uid))
            

        pose = ann
        aria_trajectory =  traj

        capture_frames = find_numerical_key_in_dict(pose)
        # capture_frames =  list(pose.keys())
        # Create continuous frame sequence from min to max frame keys
        min_frame = min(capture_frames)
        max_frame = max(capture_frames)
        continuous_frames = list(range(min_frame, max_frame + 1))

        seq_len = len(continuous_frames)

        # Prepare data for interpolation
        frame_keys_list = list(capture_frames)
        skeletons_window = []
        flags_window = []
        aria_window = []

        for frame in frame_keys_list:
            skeleton = pose[str(frame)][0]["annotation3D"]
            skeleton, flags = self.parse_skeleton(skeleton)
            skeletons_window.append(skeleton)
            flags_window.append(flags)
            aria_window.append(aria_trajectory[str(frame)])

        skeletons_window = torch.Tensor(np.array(skeletons_window)) # T, 17, 3
        flags_window = torch.Tensor(np.array(flags_window)) # T, 17
        aria_window = torch.Tensor(np.array(aria_window)) # T, 3

        # Process original keyframes
        joints_world_orig, visible_mask_orig = self._process_joints(
            skeletons_window,
            flags_window.float(),
            ground_height=float(gt_ground_height),
            return_smplh_joints=True,
            num_joints=22,
            debug_vis=False,
        )
        # breakpoint()

        # Import scipy interpolation
        from scipy.interpolate import interp1d

        # Create interpolation functions for each joint dimension
        num_joints = joints_world_orig.shape[1]

        # Interpolate world coordinates
        joints_world = torch.zeros((seq_len, num_joints, 3))
        for j in range(num_joints):
            for d in range(3):
                interp_fn = interp1d(frame_keys_list, joints_world_orig[:, j, d].numpy(force=True),
                                   kind='linear', fill_value='extrapolate')
                joints_world[:, j, d] = torch.from_numpy(interp_fn(continuous_frames))

        # Create visibility mask based on non-nan values in world coordinates
        visible_mask = ~torch.isnan(joints_world).any(dim=-1)  # shape: (seq_len, num_joints)

        masked_joints = joints_world.clone()
        masked_joints[~visible_mask] = 0

        take_name = f"name_{take_name}_uid_{take_uid}_t{continuous_frames[0]}_{continuous_frames[-1]}"

        from egoallo.data.dataclass import EgoTrainingData
        
        ret = EgoTrainingData(
            joints_wrt_world=masked_joints,  # Already computed above
            joints_wrt_cpf=torch.zeros_like(masked_joints),  # Same shape as joints_world
            T_world_root=torch.zeros((seq_len, 7)), # T x 7 for translation + quaternion
            T_world_cpf=torch.zeros((seq_len, 7)),  # T x 7 for translation + quaternion
            visible_joints_mask=visible_mask,  # Already computed above
            mask=torch.ones(seq_len, dtype=torch.bool),  # T
            betas=torch.zeros((1, 16)),  # 1 x 16 for SMPL betas
            body_quats=torch.zeros((seq_len, 21, 4)),  # T x 21 x 4 for body joint rotations
            hand_quats=torch.zeros((seq_len, 30, 4)),  # T x 30 x 4 for hand joint rotations
            contacts=torch.zeros((seq_len, 22)),  # T x 22 for contact states
            height_from_floor=torch.full((seq_len, 1), gt_ground_height),  # T x 1
            metadata=EgoTrainingData.MetaData( # raw data.
                take_name=take_name,
                frame_keys=tuple(continuous_frames),  # Convert to tuple of ints
                stage="raw",
                scope="test",
            ),
        )
        ret = ret.preprocess()
        # breakpoint()
        return ret
       

    
    def __len__(self):
        return len(self.valid_take_uids)


class Dataset_EgoExo_inference(Dataset):
    def __init__(self, config: Dict[str, Any]):
        super(Dataset_EgoExo_inference,self).__init__()
        
        self.root = config['dataset_path']
        self.root_takes = os.path.join(self.root, "takes")
        self.split = config['split'] #val or test
        self.camera_poses = os.path.join(self.root, "annotations", "ego_pose",self.split, "camera_pose")
        self.use_pseudo = config['use_pseudo']
        self.coord = config["coord"]

        self.metadata = json.load(open(os.path.join(self.root,"takes.json")))
        
        self.dummy_json = json.load(open(config['dummy_json_path']))
        self.takes_uids = [*self.dummy_json]
        self.takes_metadata = {}

        for take_uid in self.takes_uids:
            take_temp = self.get_metadata_take(take_uid)
            if take_temp and 'bouldering' not in take_temp['take_name']:
                self.takes_metadata[take_uid] = take_temp

        self.trajectories = {}
        self.cameras = {}

        for take_uid in tqdm(self.takes_metadata):
            trajectory = {}
            camera_json = json.load(open(os.path.join(self.camera_poses,take_uid+".json")))
            take_name = camera_json['metadata']['take_name']
            self.cameras[take_uid] = camera_json
            traj = self.translate_camera([*self.dummy_json[take_uid]['body']], camera_json, self.coord)
            self.trajectories[take_uid] = traj

        print('Dataset lenght: {}'.format(len(self.trajectories)))
        print('Split: {}'.format(self.split))


    def translate_camera(self, frames, cams, coord):
        trajectory = {}
        for key in cams.keys():
            if "aria" in key:
                aria_key =  key
                break
        first = frames[0]
        first_cam =  cams[aria_key]['camera_extrinsics'][first]
        T_first_camera = np.eye(4)
        T_first_camera[:3, :] = np.array(first_cam)
        for frame in frames:
            current_cam =  cams[aria_key]['camera_extrinsics'][frame]
            T_world_camera_ = np.eye(4)
            T_world_camera_[:3, :] = np.array(current_cam)
            
            if coord == 'global':
                T_world_camera = np.linalg.inv(T_world_camera_)
            elif coord == 'aria':
                T_world_camera = np.dot(T_first_camera,np.linalg.inv(T_world_camera_))
            else:
                T_world_camera = T_world_camera_

            traj = T_world_camera[:3,3]
            trajectory[frame] = traj

        return trajectory

    def get_metadata_take(self, uid):
        for take in self.metadata:
            if take["take_uid"]==uid:
                return take

    def __getitem__(self, index):
        take_uid = self. takes_uids[index]
        aria_trajectory =  self.trajectories[take_uid]
        aria_window = []
        frames_window =  list(aria_trajectory.keys())
        for frame in frames_window:
            aria_window.append(aria_trajectory[frame])



        aria_window =  torch.Tensor(np.array(aria_window))
        head_offset = aria_window.unsqueeze(1).repeat(1,17,1)
        condition =  aria_window
        task = torch.tensor(self.takes_metadata[take_uid]['task_id'])
        take_name = self.takes_metadata[take_uid]['root_dir']

        return {'cond': condition, 
                't': frames_window,
                'aria': aria_window,
                'offset':head_offset,
                'task':task,
                'take_name':take_name,
                'take_uid':take_uid}

    
    def __len__(self):
        return len(self.trajectories)    