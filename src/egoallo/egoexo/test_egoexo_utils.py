import pytest
from egoallo.egoexo import EGOEXO_UTILS_INST
from egoallo.egoexo.egoexo_utils import EgoExoUtils
from egoallo.utils.setup_logger import setup_logger
import cv2
import os


from egoallo.config import make_cfg, CONFIG_FILE

local_config_file = CONFIG_FILE
CFG = make_cfg(config_name="defaults", config_file=local_config_file, cli_args=[])



@pytest.mark.finished
@pytest.fixture(scope="function",name="setup_egoexo_utils",params=[pytest.param(CFG,id="local_cfg")])
def test_egoexo_utils():
    egoexo_utils_inst = EGOEXO_UTILS_INST
    return egoexo_utils_inst

@pytest.mark.finished
@pytest.mark.parametrize(
    "fixture1, cfg",
    [
        pytest.param(pytest.lazy_fixture("setup_egoexo_utils"), CFG, id="test_egoexo_utils"),
    ]
)
def test_egoexo_utils_examples(fixture1, cfg):
    egoexo_utils_inst: EgoExoUtils = fixture1
    all_take_uids_for_bodypose_benchmarks = egoexo_utils_inst.all_take_uids_for_ego_body_pose_benchmarks
    anno_types = cfg.io.egoexo.preprocessing.anno_types
    splits = cfg.io.egoexo.preprocessing.splits
    for anno_type in anno_types:
        for split in splits:
            all_exported_data_paths = egoexo_utils_inst.get_all_exported_data_paths(anno_type, split)
            all_exported_data = egoexo_utils_inst.get_egoexo_exported_data_output(anno_type, split)
            random_uid_w_gt_anno = egoexo_utils_inst.get_random_uid_w_gt_anno(anno_type, split)
            random_uid = egoexo_utils_inst.get_random_uid(split)
            random_take = egoexo_utils_inst.get_random_take(split)
    
    test_num = 5
    for split in splits:
        for frame_idx in range(test_num):
            random_uid = egoexo_utils_inst.get_random_uid(split=split)
            take_uid = random_uid
            take_name = egoexo_utils_inst.find_take_name_from_take_uid(take_uid)
            take_uid, take_name, take, open_loop_traj_path, close_loop_traj_path, gopro_calibs_path, cam_pose_anno_path, \
                online_calib_json_path, vrs_path, no_image_streams_vrs_path, semidense_observations_path = \
                egoexo_utils_inst.get_take_metadata_from_take_uid(take_uid)
            res = egoexo_utils_inst.get_take_metadata_from_take_name(take_name)
            exported_mp4_path = cfg.io.egoexo.preprocessing.exported_mp4_path
            mp4_file_path = egoexo_utils_inst.get_exported_mp4_path_from_take_name(take_name, exported_mp4_path)
            ego_cam_name = EgoExoUtils.get_ego_aria_cam_name(take)
            exo_cam_names = EgoExoUtils.get_exo_cam_names(take)


if __name__ == "__main__":
    pass