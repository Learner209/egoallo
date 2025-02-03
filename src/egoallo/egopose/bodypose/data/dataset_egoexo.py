import json
import os
import os.path as osp
import random

import joblib
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from egoallo.utils.setup_logger import setup_logger

logger = setup_logger(output=None, name=__name__)


random.seed(1)


class Dataset_EgoExo(Dataset):
    def __init__(self, opt):
        super(Dataset_EgoExo, self).__init__()

        self.root = opt["root"]
        self.root_takes = os.path.join(self.root, "takes")
        self.split = opt["split"]
        self.root_poses = os.path.join(
            self.root, "annotations", "ego_pose", self.split, "body"
        )
        self.use_pseudo = opt["use_pseudo"]
        self.coord = opt["coord"]
        self.slice_window = opt["window_size"]
        # load sequences paths

        manually_annotated_takes = os.listdir(
            os.path.join(self.root_poses, "annotation")
        )
        self.manually_annotated_takes = [
            take.split(".")[0] for take in manually_annotated_takes
        ]
        if self.use_pseudo:
            pseudo_annotated_takes = os.listdir(
                os.path.join(self.root_poses, "automatic")
            )
            self.pseudo_annotated_takes = [
                take.split(".")[0] for take in pseudo_annotated_takes
            ]

        self.cameras = os.listdir(self.root_poses.replace("body", "camera_pose"))
        self.metadata = json.load(open(os.path.join(self.root, "takes.json")))

        self.takes_uids = (
            self.pseudo_annotated_takes
            if self.use_pseudo
            else self.manually_annotated_takes
        )
        self.takes_metadata = {}

        self.valid_take_uid_save_label = (
            "valid_takes_{}_use_manual.pkl".format(self.split)
            if not self.use_pseudo
            else "valid_takes_{}_use_pseudo.pkl".format(self.split)
        )

        for take_uid in self.takes_uids:
            take_temp = self.get_metadata_take(take_uid)
            if take_temp and "bouldering" not in take_temp["take_name"]:
                self.takes_metadata[take_uid] = take_temp

        if not osp.exists(self.valid_take_uid_save_label):
            self.valid_take_uids = []
            no_man = 0
            no_cam = 0
            no_cam_list = []

            for take_uid in tqdm(
                self.takes_metadata,
                total=len(self.takes_metadata),
                desc="takes_metadata",
                ascii=" >=",
            ):
                if take_uid + ".json" in self.cameras:
                    camera_json = json.load(
                        open(
                            os.path.join(
                                self.root_poses.replace("body", "camera_pose"),
                                take_uid + ".json",
                            )
                        )
                    )
                    camera_json["metadata"]["take_name"]
                    if take_uid not in self.manually_annotated_takes:
                        no_man += 1
                        if self.use_pseudo and take_uid in self.pseudo_annotated_takes:
                            pose_json = json.load(
                                open(
                                    os.path.join(
                                        self.root_poses, "automatic", take_uid + ".json"
                                    )
                                )
                            )
                            if (
                                len(pose_json) > (self.slice_window + 2)
                            ) and self.split == "train":
                                ann, traj = self.translate_poses(
                                    pose_json, camera_json, self.coord
                                )
                                if len(traj) > (self.slice_window + 2):
                                    self.valid_take_uids.append(take_uid)
                            elif self.split != "train":
                                ann, traj = self.translate_poses(
                                    pose_json, camera_json, self.coord
                                )
                                self.valid_take_uids.append(take_uid)
                    elif take_uid in self.manually_annotated_takes:
                        pose_json = json.load(
                            open(
                                os.path.join(
                                    self.root_poses, "annotation", take_uid + ".json"
                                )
                            )
                        )
                        if (
                            len(pose_json) > (self.slice_window + 2)
                        ) and self.split == "train":
                            ann, traj = self.translate_poses(
                                pose_json, camera_json, self.coord
                            )
                            if len(traj) > (self.slice_window + 2):
                                self.valid_take_uids.append(take_uid)
                        elif self.split != "train":
                            ann, traj = self.translate_poses(
                                pose_json, camera_json, self.coord
                            )
                            self.valid_take_uids.append(take_uid)

                else:
                    # print("No take uid {} in camera poses".format(take_uid))
                    no_cam += 1
                    no_cam_list.append(take_uid)

            # self.joint_names = ['left-wrist', 'left-eye', 'nose', 'right-elbow', 'left-ear', 'left-shoulder', 'right-hip', 'right-ear', 'left-knee', 'left-hip', 'right-wrist', 'right-ankle', 'right-eye', 'left-elbow', 'left-ankle', 'right-shoulder', 'right-knee']
            if len(self.valid_take_uids) > 0:
                joblib.dump(self.valid_take_uids, self.valid_take_uid_save_label)
            else:
                raise UserWarning("No valid takes found")
        else:
            self.valid_take_uids = joblib.load(self.valid_take_uid_save_label)
            logger.info(f"Loaded valid take uids from {len(self.valid_take_uids)}")
            # self.valid_take_uids = self.valid_take_uids

        self.joint_idxs = [i for i in range(17)]  # 17 keypoints in total

        self.joint_names = [
            "nose",
            "left-eye",
            "right-eye",
            "left-ear",
            "right-ear",
            "left-shoulder",
            "right-shoulder",
            "left-elbow",
            "right-elbow",
            "left-wrist",
            "right-wrist",
            "left-hip",
            "right-hip",
            "left-knee",
            "right-knee",
            "left-ankle",
            "right-ankle",
        ]
        self.single_joint = opt["single_joint"]
        print("Dataset lenght: {}".format(len(self.valid_take_uids)))
        print("Split: {}".format(self.split))
        # print('No Manually: {}'.format(no_man))
        # print('No camera: {}'.format(no_cam))
        # print('No camera list: {}'.format(no_cam_list))

    def translate_poses(self, anno, cams, coord):
        trajectory = {}
        to_remove = []
        for key in cams.keys():
            if "aria" in key:
                aria_key = key
                break
        first = next(iter(anno))
        first_cam = cams[aria_key]["camera_extrinsics"][first]
        T_first_camera = np.eye(4)
        T_first_camera[:3, :] = np.array(first_cam)
        for frame in anno:
            try:
                current_anno = anno[frame]
                current_cam = cams[aria_key]["camera_extrinsics"][frame]
                T_world_camera_ = np.eye(4)
                T_world_camera_[:3, :] = np.array(current_cam)

                if coord == "global":
                    T_world_camera = np.linalg.inv(T_world_camera_)
                elif coord == "aria":
                    T_world_camera = np.dot(
                        T_first_camera, np.linalg.inv(T_world_camera_)
                    )
                else:
                    T_world_camera = T_world_camera_
                assert len(current_anno) != 0
                for idx in range(len(current_anno)):
                    joints = current_anno[idx]["annotation3D"]
                    for joint_name in joints:
                        joint4d = np.ones(4)
                        joint4d[:3] = np.array(
                            [
                                joints[joint_name]["x"],
                                joints[joint_name]["y"],
                                joints[joint_name]["z"],
                            ]
                        )
                        if coord == "global":
                            new_joint4d = joint4d
                        elif coord == "aria":
                            new_joint4d = T_first_camera.dot(joint4d)
                        else:
                            new_joint4d = T_world_camera_.dot(
                                joint4d
                            )  # The skels always stay in 0,0,0 wrt their camera frame
                        joints[joint_name]["x"] = new_joint4d[0]
                        joints[joint_name]["y"] = new_joint4d[1]
                        joints[joint_name]["z"] = new_joint4d[2]
                    current_anno[idx]["annotation3D"] = joints
                traj = T_world_camera[:3, 3]
                trajectory[frame] = traj
            except Exception:
                to_remove.append(frame)
            anno[frame] = current_anno
        keys_old = list(anno.keys())
        for frame in keys_old:
            if frame in to_remove:
                del anno[frame]
        return anno, trajectory

    def get_metadata_take(self, uid):
        for take in self.metadata:
            if take["take_uid"] == uid:
                return take

    def parse_skeleton(self, skeleton):
        poses = []
        flags = []
        keypoints = skeleton.keys()
        for keyp in self.joint_names:
            if keyp in keypoints:
                flags.append(1)  # visible
                poses.append(
                    [skeleton[keyp]["x"], skeleton[keyp]["y"], skeleton[keyp]["z"]]
                )  # visible
            else:
                flags.append(0)  # not visible
                poses.append([-1, -1, -1])  # not visible
        return poses, flags

    def __getitem__(self, index):
        take_uid = self.valid_take_uids[index]

        camera_json = json.load(
            open(
                os.path.join(
                    self.root_poses.replace("body", "camera_pose"), take_uid + ".json"
                )
            )
        )
        take_name = camera_json["metadata"]["take_name"]

        if self.use_pseudo and take_uid in self.pseudo_annotated_takes:
            pose_json = json.load(
                open(os.path.join(self.root_poses, "automatic", take_uid + ".json"))
            )
            if (len(pose_json) > (self.slice_window + 2)) and self.split == "train":
                ann, traj = self.translate_poses(pose_json, camera_json, self.coord)
            elif self.split != "train":
                ann, traj = self.translate_poses(pose_json, camera_json, self.coord)
        elif take_uid in self.manually_annotated_takes:
            pose_json = json.load(
                open(os.path.join(self.root_poses, "annotation", take_uid + ".json"))
            )
            if (len(pose_json) > (self.slice_window + 2)) and self.split == "train":
                ann, traj = self.translate_poses(pose_json, camera_json, self.coord)
            elif self.split != "train":
                ann, traj = self.translate_poses(pose_json, camera_json, self.coord)
        else:
            raise UserWarning(
                "Take uid {} not found in any annotation folder".format(take_uid)
            )

        pose = ann
        aria_trajectory = traj

        capture_frames = list(pose.keys())

        if self.split == "train":
            frames_idx = random.randint(self.slice_window, len(capture_frames) - 1)
            frames_window = capture_frames[frames_idx - self.slice_window : frames_idx]
        else:
            frames_window = capture_frames

        skeletons_window = []
        flags_window = []
        t_window = []
        aria_window = []

        for frame in frames_window:
            t_window.append(int(frame))
            skeleton = pose[frame][0]["annotation3D"]
            skeleton, flags = self.parse_skeleton(skeleton)
            skeletons_window.append(skeleton)
            flags_window.append(flags)
            aria_window.append(aria_trajectory[frame])

        skeletons_window = torch.Tensor(np.array(skeletons_window))
        flags_window = torch.Tensor(np.array(flags_window))
        aria_window = torch.Tensor(np.array(aria_window))
        head_offset = aria_window.unsqueeze(1).repeat(1, 17, 1)
        condition = aria_window
        task = torch.tensor(self.takes_metadata[take_uid]["task_id"])
        take_name = self.takes_metadata[take_uid]["root_dir"]

        return {
            "cond": condition,
            "gt": skeletons_window,
            "visible": flags_window,
            "t": frames_window,
            "aria": aria_window,
            "offset": head_offset,
            "task": task,
            "take_name": take_name,
            "take_uid": take_uid,
        }

    def __len__(self):
        return len(self.valid_take_uids)


class Dataset_EgoExo_inference(Dataset):
    def __init__(self, opt):
        super(Dataset_EgoExo_inference, self).__init__()

        self.root = opt["root"]
        self.root_takes = os.path.join(self.root, "takes")
        self.split = opt["split"]  # val or test
        self.camera_poses = os.path.join(
            self.root, "annotations", "ego_pose", self.split, "camera_pose"
        )
        self.use_pseudo = opt["use_pseudo"]
        self.coord = opt["coord"]

        self.metadata = json.load(open(os.path.join(self.root, "takes.json")))

        self.dummy_json = json.load(open(opt["dummy_json_path"]))
        self.takes_uids = [*self.dummy_json]
        self.takes_metadata = {}

        for take_uid in self.takes_uids:
            take_temp = self.get_metadata_take(take_uid)
            if take_temp and "bouldering" not in take_temp["take_name"]:
                self.takes_metadata[take_uid] = take_temp

        self.trajectories = {}
        self.cameras = {}

        for take_uid in tqdm(self.takes_metadata):
            camera_json = json.load(
                open(os.path.join(self.camera_poses, take_uid + ".json"))
            )
            camera_json["metadata"]["take_name"]
            self.cameras[take_uid] = camera_json
            traj = self.translate_camera(
                [*self.dummy_json[take_uid]["body"]], camera_json, self.coord
            )
            self.trajectories[take_uid] = traj

        print("Dataset lenght: {}".format(len(self.trajectories)))
        print("Split: {}".format(self.split))

    def translate_camera(self, frames, cams, coord):
        trajectory = {}
        for key in cams.keys():
            if "aria" in key:
                aria_key = key
                break
        first = frames[0]
        first_cam = cams[aria_key]["camera_extrinsics"][first]
        T_first_camera = np.eye(4)
        T_first_camera[:3, :] = np.array(first_cam)
        for frame in frames:
            current_cam = cams[aria_key]["camera_extrinsics"][frame]
            T_world_camera_ = np.eye(4)
            T_world_camera_[:3, :] = np.array(current_cam)

            if coord == "global":
                T_world_camera = np.linalg.inv(T_world_camera_)
            elif coord == "aria":
                T_world_camera = np.dot(T_first_camera, np.linalg.inv(T_world_camera_))
            else:
                T_world_camera = T_world_camera_

            traj = T_world_camera[:3, 3]
            trajectory[frame] = traj

        return trajectory

    def get_metadata_take(self, uid):
        for take in self.metadata:
            if take["take_uid"] == uid:
                return take

    def __getitem__(self, index):
        take_uid = self.takes_uids[index]
        aria_trajectory = self.trajectories[take_uid]
        aria_window = []
        frames_window = list(aria_trajectory.keys())
        for frame in frames_window:
            aria_window.append(aria_trajectory[frame])

        aria_window = torch.Tensor(np.array(aria_window))
        head_offset = aria_window.unsqueeze(1).repeat(1, 17, 1)
        condition = aria_window
        task = torch.tensor(self.takes_metadata[take_uid]["task_id"])
        take_name = self.takes_metadata[take_uid]["root_dir"]

        return {
            "cond": condition,
            "t": frames_window,
            "aria": aria_window,
            "offset": head_offset,
            "task": task,
            "take_name": take_name,
            "take_uid": take_uid,
        }

    def __len__(self):
        return len(self.trajectories)
