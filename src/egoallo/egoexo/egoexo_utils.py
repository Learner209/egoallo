import os.path as osp
from collections import defaultdict
from glob import glob
import joblib
import json
import numpy as np
from egoallo.utils.aria_utils.aria_calib import CalibrationUtilities
from projectaria_tools.core import data_provider
from projectaria_tools.core import calibration
from egoallo.utils.setup_logger import setup_logger
import cv2
import os


from egoallo.config import make_cfg, CONFIG_FILE

local_config_file = CONFIG_FILE
CFG = make_cfg(config_name="defaults", config_file=local_config_file, cli_args=[])


logger = setup_logger(output=None, name=__name__)

class EgoExoUtils:
    def __init__(self, opt, lazy_loading: bool = True, run_demo: bool = False):
        """
        Parameters
        ----------
        run_demo : bool, optional, default=False
            Whether to run the demo, more specifically, run the test-sample data rather than the whole set to save time.
        """
        self.root_path = opt.io.egoexo.root_path
        self.opt = opt

        # Loading dataset preprocessing options
        self.preprocess_opt = CFG.io.egoexo.preprocessing
        self._gt_handpose_output_dir = self.preprocess_opt.gt_handpose.output_dir
        self._gt_bodypose_output_dir = self.preprocess_opt.gt_bodypose.output.save_dir
        self._gt_bodypose_sample_ouput_dir = self.preprocess_opt.gt_bodypose.sample_output.save_dir
        self._aria_img_output_dir = self.preprocess_opt.aria_img_output_dir
        self._egoexo_train_data_output_dir = self.preprocess_opt.egoexo_train_data_output_dir
        self._exported_mp4_path = self.preprocess_opt.exported_mp4_path
        
        self.egoexo_metadata = {
            "takes": osp.join(self.root_path, "takes.json"),
            # "captures": osp.join(self.root_path, "captures.json"),
            # "physical_setting": osp.join(self.root_path, "physical_setting.json"),
            # "participants": osp.join(self.root_path, "participants.json"),
            # "visual_objects": osp.join(self.root_path, "visual_objects.json"),
            # "metadata": osp.join(self.root_path, "metadata.json"),
            "splits": osp.join(self.root_path, "annotations", "splits.json"),
        }

        for k, v in self.egoexo_metadata.items():
            self.egoexo_metadata[k] = json.load(open(v,'r'))

        self.raw_gt_anno_paths = defaultdict(dict)
        self.raw_gt_anno_ = defaultdict(dict)
        self._anno_types = self.preprocess_opt.anno_types
        self._splits = self.preprocess_opt.splits
        gt_bodypose_output_dir = self._gt_bodypose_output_dir if not run_demo else self._gt_bodypose_sample_ouput_dir
        for anno_type in self._anno_types:
            for split in self._splits:
                self.raw_gt_anno_paths[anno_type][split] = osp.join(
                    gt_bodypose_output_dir,
                    "annotation",
                    anno_type,
                    f"ego_pose_gt_anno_{split}_public.json",
                )
        

        self._all_takes = self.egoexo_metadata["takes"]
        self.take_uid_to_take_names = {take["take_uid"]: take["take_name"] for take in self._all_takes}
        self.take_names_to_take_uid = {take["take_name"]: take["take_uid"] for take in self._all_takes}
        if not lazy_loading:
            self._load_splits_metadata()


    def _load_splits_metadata(self):
        #! Validate whether only train_uids has gopro_calibs.csv
        # ! Validate the train_uids, val_uids, test_uids, and the total nums
        self.all_train_uids = self.egoexo_metadata["splits"]["split_to_take_uids"]["train"]
        self.all_val_uids = self.egoexo_metadata["splits"]["split_to_take_uids"]["val"]
        self.all_test_uids = self.egoexo_metadata["splits"]["split_to_take_uids"]["test"]
        self.all_take_uids = [take["take_uid"] for take in self.egoexo_metadata["takes"]] 
        self.all_take_names = [take["take_name"] for take in self.egoexo_metadata["takes"]]

        # print(f"train takes: {len(self.all_train_uids)}, val takes: {len(self.all_val_uids)}, test takes: {len(self.all_test_uids)}")
        # print(f"total: {len(self.all_take_uids)}")

        self.all_train_takes = [self.find_take_from_take_uid(take_uid) for take_uid in self.all_train_uids]
        self.all_val_takes = [self.find_take_from_take_uid(take_uid) for take_uid in self.all_val_uids]
        self.all_test_takes = [self.find_take_from_take_uid(take_uid) for take_uid in self.all_test_uids]

        # ! subsititute 
        traj_dirs = [os.path.join(self.root_path, take["root_dir"], "trajectory") for take in self.egoexo_metadata["takes"]]
        traj_dirs = list(filter(lambda traj_dir: os.path.exists(os.path.join(traj_dir, "gopro_calibs.csv")), traj_dirs))
        # print(f"Total takes with gopro_calibs.csv: {len(traj_dirs)}")
        exo_traj_paths = [os.path.join(traj_dir, "gopro_calibs.csv") for traj_dir in traj_dirs]

        # print(f"egoexo['splits']' keys are {self.egoexo_metadata['splits'].keys()}")

        self.take_uids_to_benchmark = self.egoexo_metadata["splits"]["take_uid_to_benchmark"]
        self.take_uids_to_benchmark_as_list = list(self.take_uids_to_benchmark.keys())

        # print(f"the length of take_uids_to_benchmark is {len(set(self.take_uids_to_benchmark.keys()))}")

        # TODO: Check why take_uids_to_benchmark contain 6173 take_uids whether there are only 5035 takes.

        self.all_take_uids_for_ego_hand_pose_benchmarks = set([take_uid for take_uid in self.all_take_uids if take_uid in self.take_uids_to_benchmark_as_list  and "ego_hand_pose" in self.take_uids_to_benchmark[take_uid]])
        self.all_take_uids_for_ego_body_pose_benchmarks = set([take_uid for take_uid in self.all_take_uids if  take_uid in self.take_uids_to_benchmark_as_list  and "ego_body_pose" in self.take_uids_to_benchmark[take_uid]])

        self.take_uids_for_ego_hand_pose_benchmarks = {
            "train": self.all_take_uids_for_ego_hand_pose_benchmarks.intersection(set(self.all_train_uids)),
               "val": self.all_take_uids_for_ego_hand_pose_benchmarks.intersection(set(self.all_val_uids)),
            "test": self.all_take_uids_for_ego_hand_pose_benchmarks.intersection(set(self.all_test_uids))
        }
        self.take_uids_for_ego_body_pose_benchmarks = {
            "train": self.all_take_uids_for_ego_body_pose_benchmarks.intersection(set(self.all_train_uids)),
               "val": self.all_take_uids_for_ego_body_pose_benchmarks.intersection(set(self.all_val_uids)),
            "test": self.all_take_uids_for_ego_body_pose_benchmarks.intersection(set(self.all_test_uids))
        }

        # print(f"Num of takes available for hand-pose benchmarks are {len(self.all_take_uids_for_ego_hand_pose_benchmarks)}, out of which train: {len(self.all_take_uids_for_ego_hand_pose_benchmarks.intersection(set(self.all_train_uids)))}, val: {len(self.all_take_uids_for_ego_hand_pose_benchmarks.intersection(set(self.all_val_uids)))}, test: {len(self.all_take_uids_for_ego_hand_pose_benchmarks.intersection(set(self.all_test_uids)))}")
        # print(f"Num of takes availabel for body-pose benchmarks are {len(self.all_take_uids_for_ego_body_pose_benchmarks)}, out of which train: {len(self.all_take_uids_for_ego_body_pose_benchmarks.intersection(set(self.all_train_uids)))}, val: {len(self.all_take_uids_for_ego_body_pose_benchmarks.intersection(set(self.all_val_uids)))}, test: {len(self.all_take_uids_for_ego_body_pose_benchmarks.intersection(set(self.all_test_uids)))}")

    @property
    def takes(self):
        return self.egoexo_metadata["takes"]

    @property
    def captures(self):
        return self.egoexo_metadata["captures"]

    @property
    def metadata(self):
        return self.egoexo_metadata["metadata"]
    
    @property
    def splits(self):
        return self.egoexo_metadata["splits"]

    @property
    def gt_handpose_output_dir(self):
        return self._gt_handpose_output_dir
    
    @property
    def gt_bodypose_output_dir(self):
        return self._gt_bodypose_output_dir
    
    @property
    def aria_img_output_dir(self):
        return self._aria_img_output_dir
    
    @property
    def egoexo_train_data_output_dir(self):
        return self._egoexo_train_data_output_dir
    
    @property
    def exported_mp4_path(self):
        return self._exported_mp4_path
    
    def get_egoexo_exported_data_output(self, anno_types, splits):
        """
        Get egoexo exported data output path
        """
        all_exported_data_paths = self.get_egoexo_exported_data_output_path(anno_types, splits)
        all_exported_data = {}
        for _, exported_data_path in enumerate(all_exported_data_paths):
            assert osp.exists(exported_data_path), f"EgoExo Exported data path: {exported_data_path} does not exist"
            all_exported_data = {**all_exported_data, **joblib.load(exported_data_path)}
        return all_exported_data

    def get_egoexo_exported_data_output_path(self, anno_types, splits):
        """
        Get egoexo exported data output path.
        """
        preprocess_cfg = self.preprocess_opt
        all_exported_data_paths = []
        for _, anno_type in enumerate(anno_types):
            if anno_type not in preprocess_cfg.all_anno_types:
                raise ValueError("Invalid anno_type:{0}".format(anno_type))
            this_type_egoexo_train_data_output_dir = os.path.join(
                self._egoexo_train_data_output_dir, "annotation", anno_type
            )
            for _, split in enumerate(splits):
                if split not in preprocess_cfg.all_splits:
                    raise ValueError("Invalid split:{0}".format(split))
                export_egoexo_output_obj_path = os.path.join(this_type_egoexo_train_data_output_dir, f"egopose_gt_anno_{split}_public.p")
                all_exported_data_paths.append(export_egoexo_output_obj_path)
        return all_exported_data_paths
                
    def get_all_uids_w_gt_anno(self, anno_type, split):
        """
        Get all take_uids with gt_anno accompanying.
        """
        # Load GT annotation
        if anno_type not in self.raw_gt_anno_.keys() or split not in self.raw_gt_anno_[anno_type].keys():
            self.raw_gt_anno_[anno_type][split] = json.load(open(self.raw_gt_anno_paths[anno_type][split]))
        gt_anno = self.raw_gt_anno_[anno_type][split]
        take_uids_w_gt_anno = list(gt_anno.keys())
        return take_uids_w_gt_anno

    def get_random_uid_w_gt_anno(self, anno_type, split):
        """
        Get random take_uid with gt_anno accompanying.
        """
        # Load GT annotation
        if anno_type not in self.raw_gt_anno_.keys() or split not in self.raw_gt_anno_[anno_type].keys():
            self.raw_gt_anno_[anno_type][split] = json.load(open(self.raw_gt_anno_paths[anno_type][split]))
        gt_anno = self.raw_gt_anno_[anno_type][split]
        take_uids_w_gt_anno = list(gt_anno.keys())
        return np.random.choice(take_uids_w_gt_anno)
        
    def get_random_uid(self, split):
        return np.random.choice(self.splits["split_to_take_uids"][split])
    
    def get_random_take(self, split):
        return self.find_take_from_take_uid(self.get_random_uid(split))

    def get_take_metadata_from_take_uid(self, take_uid):
        """
        Get metadata related to a specific take based on its unique identifier.

        Parameters
        ----------
            take_uid :  str,
                The unique identifier of the take.
        Returns
        -------
        - take_uid : str
            The unique identifier of the take.
        - take_name : str
            The name of the take.
        - take : dict
            The metadata of the take.
        - open_loop_traj_path : str
            The file path of the open loop trajectory CSV file.
        - close_loop_traj_path : str
            The file path of the closed loop trajectory CSV file.
        - gopro_calibs_path : str
            The file path of the GoPro calibrations CSV file.
        - cam_pose_anno_path : str
            The file path of the camera pose annotation JSON file.
        - online_calib_json_path : str
            The file path of the online calibration JSON file.
        - vrs_path : str
            The file path of the vrs_path
        - vrs_noimagestreams_path : str
            The file path of the vr_no_imagestreams_path.
        - semidense_observations_path : str
            The file path of the semidense_observations_path.

        Raises
        ------
            AssertionError: If any of the required files do not exist.
        """
        

        take_name = self.take_uid_to_take_names[take_uid]
        take = self.find_take_from_take_uid(take_uid)

        take = [t for t in self.egoexo_metadata["takes"] if t["take_uid"] == take_uid][0]
        open_loop_traj_path = None if not self.has_open_loop_traj(take_uid) else self.load_open_loop_traj(take_uid)	
        close_loop_traj_path = None if not self.has_close_loop_traj(take_uid) else self.load_close_loop_traj(take_uid)	
        gopro_calibs_path = None if not self.has_gopro_calibs(take_uid) else self.load_gopro_calibs(take_uid)
        cam_pose_anno_path = None if not self.has_cam_pose_anno(take_uid) else self.load_cam_pose_anno(take_uid)
        online_calib_json_path = None if not self.has_online_calib_json(take_uid) else self.load_online_calib_json(take_uid)
        vrs_path = None if not self.has_vrs(take_uid) else self.load_vrs(take_uid)
        no_image_streams_vrs_path = None if not self.has_vrs_noimagestreams(take_uid) else self.load_vrs_noimagestreams(take_uid)
        semidense_observations_path = None if not self.has_semidense_observations(take_uid) else self.load_semidense_observations(take_uid)
        
        return take_uid, take_name, take, open_loop_traj_path, close_loop_traj_path, gopro_calibs_path, cam_pose_anno_path, online_calib_json_path, vrs_path, no_image_streams_vrs_path, semidense_observations_path

    def get_take_metadata_from_take_name(self, take_name):
        """
        Get metadata related to a specific take based on its take name.

        Parameters
        ----------
            take_uid :  str,
                The unique identifier of the take.
        Returns
        -------
        - take_uid : str
            The unique identifier of the take.
        - take_name : str
            The name of the take.
        - take : dict
            The metadata of the take.
        - open_loop_traj_path : str
            The file path of the open loop trajectory CSV file.
        - close_loop_traj_path : str
            The file path of the closed loop trajectory CSV file.
        - gopro_calibs_path : str
            The file path of the GoPro calibrations CSV file.
        - cam_pose_anno_path : str
            The file path of the camera pose annotation JSON file.

        Raises
        ------
            AssertionError: If any of the required files do not exist.
        """
        
        take_uid = self.take_names_to_take_uid[take_name]
        return self.get_take_metadata_from_take_uid(take_uid)

    def get_exported_mp4_path_from_take_name(self, take_name, exported_mp4_output_dir):
        """
        Get the vrs-exported MP4 file path for a specific take based on its take name.

        Parameters
        ----------
        take_name :  str,
                The name of the take.
        Returns
        -------
        mp4_file_path : str
            The file path of the exported MP4 file.

        Raises
        ------
            AssertionError: If the MP4 file does not exist.
        """
        split = self.egoexo_metadata["splits"]["take_uid_to_split"][self.take_names_to_take_uid[take_name]]
        mp4_output_root = os.path.join(
            exported_mp4_output_dir, "exported_mp4", split
        )
        mp4_file_path = osp.join(mp4_output_root, f"{take_name}.mp4")
        res = mp4_file_path if osp.exists(mp4_file_path) else None
        return res

    def get_exported_mp4_path_from_take_uid(self, take_uid, gt_output_dir):
        """
        Get the vrs-exported MP4 file path for a specific take based on its take uid.

        Parameters
        ----------
        take_name :  str,
                The name of the take.
        Returns
        -------
        mp4_file_path : str
            The file path of the exported MP4 file.

        Raises
        ------
            AssertionError: If the MP4 file does not exist.
        """
        take_name = self.take_uid_to_take_names[take_uid]
        split = self.egoexo_metadata["splits"]["take_uid_to_split"][self.take_names_to_take_uid[take_name]]
        mp4_output_root = os.path.join(
            gt_output_dir, "exported_mp4", split
        )
        mp4_file_path = osp.join(mp4_output_root, f"{take_name}.mp4")
        assert osp.exists(mp4_file_path), f"mp4_file_path: {mp4_file_path} does not exist"
        return mp4_file_path

    def get_undistorted_aria_imgs_from_take_uid(self, take_uid):
        """
        Get the undistorted Aria images for a specific take based on its take uid.

        Parameters
        ----------
        take_uid :  str

        Returns
        -------
        undistorted_aria_imgs : dict of str(absolute path)
            each key represents frame num.
        """
        take_name = self.find_take_name_from_take_uid(take_uid)
        split = self.egoexo_metadata["splits"]["take_uid_to_split"][take_uid]
        aria_img_output_root = os.path.join(
            self._aria_img_output_dir, "image", "undistorted", split, take_name
        )
        undistorted_aria_imgs_paths = sorted(glob(osp.join(aria_img_output_root, f"*.png")))
        if len(undistorted_aria_imgs_paths) == 0:
            logger.warning(f"Take_name: {take_name} do not has any undistorted Aria images in {aria_img_output_root}")

        aria_img_frame_nums = [int(osp.splitext(osp.basename(img))[0]) for img in undistorted_aria_imgs_paths]
        undistorted_aria_imgs_dict = {frame_num: img_path for frame_num, img_path in zip(aria_img_frame_nums, undistorted_aria_imgs_paths)}
        
        return undistorted_aria_imgs_dict

    def get_undistorted_aria_imgs_from_take_name(self, take_name):
        """
        Get the undistorted Aria images for a specific take based on its take uid.

        Parameters
        ----------
        take_uid :  str

        Returns
        -------
        undistorted_aria_imgs : dict of str(absolute path)
            each key represents frame num.
        """
        take_uid = self.find_take_uid_from_take_name(take_name)
        return self.get_undistorted_aria_imgs_from_take_uid(take_uid)

    def find_take_uid_from_take_name(self, take_name):
        if not take_name in self.take_names_to_take_uid:
            return None
        else:
            return self.take_names_to_take_uid[take_name]

    def find_take_name_from_take_uid(self, take_uid):
        if not take_uid in self.take_uid_to_take_names:
            return None
        else:
            return self.take_uid_to_take_names[take_uid]


    def find_take_from_take_name(self, take_name):
        for take in self.takes:
            if take["take_name"] == take_name:
                return take
        return None

    def find_take_from_take_uid(self, take_uid):
        for take in self.takes:
            if take["take_uid"] == take_uid:
                return take
        return None

    @staticmethod
    def get_ego_aria_cam_name(take):
        ego_cam_names = [
            x["cam_id"]
            for x in take["capture"]["cameras"]
            if str(x["is_ego"]).lower() == "true"
        ]
        assert len(ego_cam_names) > 0, "No ego cameras found!"
        if len(ego_cam_names) > 1:
            ego_cam_names = [
                cam for cam in ego_cam_names if cam in take["frame_aligned_videos"].keys()
            ]
            assert len(ego_cam_names) > 0, "No frame-aligned ego cameras found!"
            if len(ego_cam_names) > 1:
                ego_cam_names_filtered = [
                    cam for cam in ego_cam_names if "aria" in cam.lower()
                ]
                if len(ego_cam_names_filtered) == 1:
                    ego_cam_names = ego_cam_names_filtered
            assert (
                len(ego_cam_names) == 1
            ), f"Found too many ({len(ego_cam_names)}) ego cameras: {ego_cam_names}"
        ego_cam_names = ego_cam_names[0]
        return ego_cam_names

    @staticmethod
    def get_exo_cam_names(take):
        exo_cam_names = [
            x["cam_id"]
            for x in take["capture"]["cameras"]
            if str(x["is_ego"]).lower() == "false"
        ]
        assert len(exo_cam_names) > 0, "No exo cameras found!"
        return exo_cam_names

    def has_open_loop_traj(self, take_uid):
        """
        Check if a specific take has an open loop trajectory.

        Parameters
        ----------
            take_uid :  str,
                The unique identifier of the take.
        Returns
        -------
        - bool
            True if the open loop trajectory exists, False otherwise.
        """
        take_name = self.take_uid_to_take_names[take_uid]
        open_loop_traj_path = osp.join(self.root_path, "takes", take_name, "trajectory", "open_loop_trajectory.csv")
        return osp.exists(open_loop_traj_path)

    def has_close_loop_traj(self, take_uid):
        """
        Check if a specific take has an close loop trajectory.

        Parameters
        ----------
            take_uid :  str,
                The unique identifier of the take.
        Returns
        -------
        - bool
            True if the close loop trajectory exists, False otherwise.
        """
        take_name = self.take_uid_to_take_names[take_uid]
        close_loop_traj_path = osp.join(self.root_path, "takes", take_name, "trajectory", "closed_loop_trajectory.csv")
        return osp.exists(close_loop_traj_path)

    def has_gopro_calibs(self, take_uid):
        """
        Check if a specific take has an gopro_calibs.

        Parameters
        ----------
            take_uid :  str,
                The unique identifier of the take.
        Returns
        -------
        - bool
            True if the close loop trajectory exists, False otherwise.
        """
        take_name = self.take_uid_to_take_names[take_uid]
        gopro_calibs_path = osp.join(self.root_path, "takes", take_name, "trajectory", "gopro_calibs.csv")
        return osp.exists(gopro_calibs_path)

    def has_cam_pose_anno(self, take_uid):
        """
        Check if a specific take has an cam pose annotation.

        Parameters
        ----------
            take_uid :  str,
                The unique identifier of the take.
        Returns
        -------
        - bool
            True if the close loop trajectory exists, False otherwise.
        """
        split = self.egoexo_metadata["splits"]["take_uid_to_split"][take_uid]
        take_name = self.take_uid_to_take_names[take_uid]
        cam_pose_anno_path = osp.join(self.root_path, "annotations", "ego_pose",split,"camera_pose", f"{take_uid}.json")
        return osp.exists(cam_pose_anno_path)

    def has_online_calib_json(self, take_uid):
        """
        Check if a specific take has an online_calib_json.

        Parameters
        ----------
            take_uid :  str,
                The unique identifier of the take.
        Returns
        -------
        - bool
            True if the close loop trajectory exists, False otherwise.
        """
        take_name = self.take_uid_to_take_names[take_uid]
        online_calib_json_path = osp.join(self.root_path, "takes", take_name, "trajectory","online_calibration.jsonl")
        return osp.exists(online_calib_json_path)

    def has_vrs(self, take_uid):
        """
        Check if a specific take has an cam pose annotation.

        Parameters
        ----------
            take_uid :  str,
                The unique identifier of the take.
        Returns
        -------
        - bool
            True if the close loop trajectory exists, False otherwise.
        """
        take_name = self.take_uid_to_take_names[take_uid]
        take = self.find_take_from_take_uid(take_uid)
        ego_cam_name = EgoExoUtils.get_ego_aria_cam_name(take)
        vrs_path =  osp.join(self.root_path, "takes", take_name, "{}.vrs".format(ego_cam_name))
        return osp.exists(vrs_path)

    def has_vrs_noimagestreams(self, take_uid):
        """
        Check if a specific take has an cam pose annotation.

        Parameters
        ----------
            take_uid :  str,
                The unique identifier of the take.
        Returns
        -------
        - bool
            True if the close loop trajectory exists, False otherwise.
        """
        take_name = self.take_uid_to_take_names[take_uid]
        take = self.find_take_from_take_uid(take_uid)
        ego_cam_name = EgoExoUtils.get_ego_aria_cam_name(take)
        vrs_noimagestreams_path =  osp.join(self.root_path, "takes", take_name,"{}_noimagestreams.vrs".format(ego_cam_name))
        return osp.exists(vrs_noimagestreams_path)


    def has_semidense_observations(self, take_uid):
        """
        Check if a specific take has an semidense_point_clouds.

        Parameters
        ----------
            take_uid :  str,
                The unique identifier of the take.
        Returns
        -------
        - bool
            True if the semidense_point_clouds exists, False otherwise.
        """
        take_name = self.take_uid_to_take_names[take_uid]
        take = self.find_take_from_take_uid(take_uid)
        semidense_observations_path = osp.join(self.root_path, "takes", take_name, "trajectory","semidense_observations.csv.gz")
        return osp.exists(semidense_observations_path)

    def load_open_loop_traj(self, take_uid):
        """
        Return if possible the path of an open loop trajectory.

        Parameters
        ----------
            take_uid :  str,
                The unique identifier of the take.
        Returns
        -------
        - bool
            True if the open loop trajectory exists, False otherwise.
        """
        take_name = self.take_uid_to_take_names[take_uid]
        open_loop_traj_path = osp.join(self.root_path, "takes", take_name, "trajectory", "open_loop_trajectory.csv")
        return open_loop_traj_path

    def load_close_loop_traj(self, take_uid):
        """
        Return if possible the path of an close loop trajectory.

        Parameters
        ----------
            take_uid :  str,
                The unique identifier of the take.
        Returns
        -------
        - bool
            True if the close loop trajectory exists, False otherwise.
        """
        take_name = self.take_uid_to_take_names[take_uid]
        close_loop_traj_path = osp.join(self.root_path, "takes", take_name, "trajectory", "closed_loop_trajectory.csv")
        return close_loop_traj_path

    def load_gopro_calibs(self, take_uid):
        """
        Return if possible the path of an gopro_calibs.

        Parameters
        ----------
            take_uid :  str,
                The unique identifier of the take.
        Returns
        -------
        - bool
            True if the close loop trajectory exists, False otherwise.
        """
        take_name = self.take_uid_to_take_names[take_uid]
        gopro_calibs_path = osp.join(self.root_path, "takes", take_name, "trajectory", "gopro_calibs.csv")
        return gopro_calibs_path

    def load_cam_pose_anno(self, take_uid):
        """
        Return if possible the path of an cam pose annotation.

        Parameters
        ----------
            take_uid :  str,
                The unique identifier of the take.
        Returns
        -------
        - bool
            True if the close loop trajectory exists, False otherwise.
        """
        split = self.egoexo_metadata["splits"]["take_uid_to_split"][take_uid]
        take_name = self.take_uid_to_take_names[take_uid]
        cam_pose_anno_path = osp.join(self.root_path, "annotations", "ego_pose",split,"camera_pose", f"{take_uid}.json")
        return cam_pose_anno_path

    def load_online_calib_json(self, take_uid):
        """
        Return if possible the path of an online_calib_json.

        Parameters
        ----------
            take_uid :  str,
                The unique identifier of the take.
        Returns
        -------
        - bool
            True if the close loop trajectory exists, False otherwise.
        """
        take_name = self.take_uid_to_take_names[take_uid]
        online_calib_json_path = osp.join(self.root_path, "takes", take_name, "trajectory","online_calibration.jsonl")
        return online_calib_json_path

    def load_vrs(self, take_uid):
        """
        Return if possible the path of an cam pose annotation.

        Parameters
        ----------
            take_uid :  str,
                The unique identifier of the take.
        Returns
        -------
        - bool
            True if the close loop trajectory exists, False otherwise.
        """
        take_name = self.take_uid_to_take_names[take_uid]
        take = self.find_take_from_take_uid(take_uid)
        ego_cam_name = EgoExoUtils.get_ego_aria_cam_name(take)
        vrs_path =  osp.join(self.root_path, "takes", take_name, "{}.vrs".format(ego_cam_name))
        return vrs_path

    def load_vrs_noimagestreams(self, take_uid):
        """
        Return if possible the path of an cam pose annotation.

        Parameters
        ----------
            take_uid :  str,
                The unique identifier of the take.
        Returns
        -------
        - bool
            True if the close loop trajectory exists, False otherwise.
        """
        take_name = self.take_uid_to_take_names[take_uid]
        take = self.find_take_from_take_uid(take_uid)
        ego_cam_name = EgoExoUtils.get_ego_aria_cam_name(take)
        vrs_noimagestreams_path =  osp.join(self.root_path, "takes", take_name,"{}_noimagestreams.vrs".format(ego_cam_name))
        return vrs_noimagestreams_path



    def load_semidense_observations(self, take_uid):
        """
        Check if a specific take has an semidense_point_clouds.

        Parameters
        ----------
            take_uid :  str,
                The unique identifier of the take.
        Returns
        -------
        - bool
            True if the semidense_point_clouds exists, False otherwise.
        """
        take_name = self.take_uid_to_take_names[take_uid]
        take = self.find_take_from_take_uid(take_uid)
        semidense_observations_path = osp.join(self.root_path, "takes", take_name, "trajectory","semidense_observations.csv.gz")
        return semidense_observations_path