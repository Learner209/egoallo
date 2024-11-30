import subprocess

import numpy as np
from egoallo.utils.smpl_mapping.mapping import EGOEXO4D_BODYPOSE_KINTREE_PARENTS


# Customized order when processing each hand's annotation
HAND_ORDER = ["right", "left"]


def get_aria_camera_models(aria_path):
	try:
		from projectaria_tools.core import data_provider

		vrs_data_provider = data_provider.create_vrs_data_provider(aria_path)
		aria_camera_model = vrs_data_provider.get_device_calibration()
		slam_left = aria_camera_model.get_camera_calib("camera-slam-left")
		slam_right = aria_camera_model.get_camera_calib("camera-slam-right")
		rgb_cam = aria_camera_model.get_camera_calib("camera-rgb")
	except Exception as e:
		print(
			f"[Warning] Hitting exception {e}. Fall back to old projectaria_tools ..."
		)
		import projectaria_tools

		vrs_data_provider = projectaria_tools.dataprovider.AriaVrsDataProvider()
		vrs_data_provider.openFile(aria_path)

		aria_stream_id = projectaria_tools.dataprovider.StreamId(214, 1)
		vrs_data_provider.setStreamPlayer(aria_stream_id)
		vrs_data_provider.readFirstConfigurationRecord(aria_stream_id)

		aria_stream_id = projectaria_tools.dataprovider.StreamId(1201, 1)
		vrs_data_provider.setStreamPlayer(aria_stream_id)
		vrs_data_provider.readFirstConfigurationRecord(aria_stream_id)

		aria_stream_id = projectaria_tools.dataprovider.StreamId(1201, 2)
		vrs_data_provider.setStreamPlayer(aria_stream_id)
		vrs_data_provider.readFirstConfigurationRecord(aria_stream_id)

		assert vrs_data_provider.loadDeviceModel()

		aria_camera_model = vrs_data_provider.getDeviceModel()
		slam_left = aria_camera_model.getCameraCalib("camera-slam-left")
		slam_right = aria_camera_model.getCameraCalib("camera-slam-right")
		rgb_cam = aria_camera_model.getCameraCalib("camera-rgb")

	assert slam_left is not None
	assert slam_right is not None
	assert rgb_cam is not None

	return {
		"1201-1": slam_left,
		"1201-2": slam_right,
		"214-1": rgb_cam,
	}


def aria_landscape_to_portrait(kpts, img_shape=(1408, 1408)):
	"""
	Rotate kpts coordinates from landscape view to portrait view
	img_shape is the shape of landscape image
	"""
	H, _ = img_shape
	none_idx = np.any(np.isnan(kpts), axis=1)
	new_kpts = kpts.copy()
	new_kpts[~none_idx, 0] = H - kpts[~none_idx, 1] - 1
	new_kpts[~none_idx, 1] = kpts[~none_idx, 0]
	return new_kpts


def rand_bbox_from_kpts(kpts, img_shape, expansion_factor=1.5):
	"""
	Generate random body bbox based on body kpts; for testing purpose.
	"""
	img_H, img_W = img_shape[:2]
	# Get proposed body bounding box from hand keypoints
	xmin, ymin, xmax, ymax = (
		kpts[:, 0].min(),
		kpts[:, 1].min(),
		kpts[:, 0].max(),
		kpts[:, 1].max(),
	)
	# Get x-coordinate for bbox
	x_center = (xmin + xmax) / 2.0
	width = (xmax - xmin) * expansion_factor
	rand_w = width * np.random.uniform(low=0.9, high=1.1)
	rand_x = width * np.random.uniform(low=-0.1, high=0.1)
	xmin = x_center + rand_x - 0.5 * rand_w
	xmax = x_center + rand_x + 0.5 * rand_w
	# Get y-coordinate for bbox
	y_center = (ymin + ymax) / 2.0
	height = (ymax - ymin) * expansion_factor
	rand_h = height * np.random.uniform(low=0.9, high=1.1)
	rand_y = height * np.random.uniform(low=-0.1, high=0.1)
	ymin = y_center + rand_y - 0.5 * rand_h
	ymax = y_center + rand_y + 0.5 * rand_h
	# Clip bbox within image bound
	bbox_x1, bbox_y1, bbox_x2, bbox_y2 = (
		np.clip(xmin, 0, img_W - 1),
		np.clip(ymin, 0, img_H - 1),
		np.clip(xmax, 0, img_W - 1),
		np.clip(ymax, 0, img_H - 1),
	)
	bbox = np.array([bbox_x1, bbox_y1, bbox_x2, bbox_y2]).astype(np.float32)
	return bbox


def pad_bbox_from_kpts(kpts, img_shape, padding=20):
	"""
	Generate body bbox based on body kpts with padding; for train and val.
	"""
	img_H, img_W = img_shape[:2]
	# Get proposed body bounding box from body keypoints
	x1, y1, x2, y2 = (
		kpts[:, 0].min(),
		kpts[:, 1].min(),
		kpts[:, 0].max(),
		kpts[:, 1].max(),
	)

	# Proposed body bounding box with padding
	bbox_x1, bbox_y1, bbox_x2, bbox_y2 = (
		np.clip(x1 - padding, 0, img_W - 1),
		np.clip(y1 - padding, 0, img_H - 1),
		np.clip(x2 + padding, 0, img_W - 1),
		np.clip(y2 + padding, 0, img_H - 1),
	)

	# Return bbox result
	return np.array([bbox_x1, bbox_y1, bbox_x2, bbox_y2])


def xyxy2cs(x1, y1, x2, y2, img_shape, pixel_std):
	aspect_ratio = img_shape[1] * 1.0 / img_shape[0]

	center = np.zeros((2), dtype=np.float32)
	center[0] = (x1 + x2) / 2
	center[1] = (y1 + y2) / 2

	w = x2 - x1
	h = y2 - y1

	if w > aspect_ratio * h:
		h = w * 1.0 / aspect_ratio
	elif w < aspect_ratio * h:
		w = h * aspect_ratio
	scale = np.array([w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
	if center[0] != -1:
		scale = scale * 1.25

	return center, scale


def world_to_cam(kpts, extri):
	"""
	Transform 3D world kpts to camera coordinate system
	Input:
		kpts: (N,3)
		extri: (3,4) [R|t]
	Output:
		new_kpts: (N,3)
	"""
	new_kpts = kpts.copy()
	new_kpts = np.append(new_kpts, np.ones((new_kpts.shape[0], 1)), axis=1).T  # (4,N)
	new_kpts = (extri @ new_kpts).T  # (N,3)
	return new_kpts


def cam_to_img(kpts, intri):
	"""
	Project points in camera coordinate system to image plane
	Input:
		kpts: (N,3)
	Output:
		new_kpts: (N,2)
	"""
	new_kpts = kpts.copy()
	new_kpts = intri @ new_kpts.T  # (3,N)
	new_kpts = new_kpts / new_kpts[2, :]
	new_kpts = new_kpts[:2, :].T
	return new_kpts


def xywh2xyxy(bbox):
	"""
	Given bbox in [x1,y1,w,h], return bbox corners [x1, y1, x2, y2]
	"""
	x1, y1, w, h = bbox
	x2 = x1 + w
	y2 = y1 + h
	return np.array([x1, y1, x2, y2])


def unit_vector(vector):
	"""Returns the unit vector of the vector."""
	return vector / (np.linalg.norm(vector) + 1e-8)


def angle_between(v1, v2):
	"""Returns the angle between vectors v1 and v2"""
	v1_u = unit_vector(v1)
	v2_u = unit_vector(v2)
	return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * 180.0 / np.pi

def body_jnts_dist_angle_check(curr_body_pose3d):
	"""
	Check body biomechanical info: Discard distance based on kintree table relationship beyond low and high thresholds

	Parameters
	-----------
	curr_body_pose3d : np.ndarray of shape (17, 3)
		3D body pose estimation in world coordinate system
	
	Returns
	--------
	curr_body_pose3d : np.ndarray of shape (17, 3)
		Filtered 3D body pose estimation in world coordinate system, invalid jnts are set to None
	valid_flag : np.ndarray of shape (17,)
		Flag to indicate valid joints

	"""
	## Joint distance threshold ##
	kintree_table = EGOEXO4D_BODYPOSE_KINTREE_PARENTS
	assert len(kintree_table) == curr_body_pose3d.shape[0] and curr_body_pose3d.shape[1] == 3, f"Invalid input shape: {curr_body_pose3d.shape} or invalid kintree_table {len(kintree_table)}"
	num_of_jnts = curr_body_pose3d.shape[0]
	# nose <-> left-eye <-> right-eye <-> left-ear <-> right-ear are the four components that has short distance to each other, so their thresholds should be set differently.
	short_jnt_dist_index = [1, 2, 3, 4]
	joint_dist_min_threshold = np.full((num_of_jnts,), 0.02)
	joint_dist_min_threshold[short_jnt_dist_index] = 0.008
	joint_dist_max_threshold = np.full((num_of_jnts,), 0.8)
	joint_dist_max_threshold[short_jnt_dist_index] = 0.2

	##Joint angle threshold ##
	# joint_angle_min_threshold = np.array(
	# 	[100, 90, 90, 60, 70, 80, 60, 70, 80, 60, 70, 80, 60, 70, 80]
	# )
	# joint_angle_max_threshold = np.array([180] * 15)
	## Misc ##
	# wrist_conn_index = [1, 5, 9, 13, 17]
	joint_dist_index = list(range(0, num_of_jnts))
	# joint_angle_index = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]

	###### Joint distance check #######
	pr_jnt_pos = np.zeros((len(kintree_table), 3))
	for k in range(1, len(kintree_table)):
		if kintree_table[k] == -1:
			pr_jnt_pos[k,:] = curr_body_pose3d[k,:3]
		else:
			pr_jnt_pos[k,:] = curr_body_pose3d[kintree_table[k],:3]
	jnt_dists = curr_body_pose3d - pr_jnt_pos
	jnt_dists = np.linalg.norm(jnt_dists, axis=1)

	###### Joint angle check ######
	# joint_angle = []
	# for joint_idx in joint_angle_index:
	# 	# If current joint has pose3d estimation
	# 	if joint_idx in wrist_conn_index:
	# 		vec1 = curr_body_pose3d[0, :3] - curr_body_pose3d[joint_idx, :3]
	# 	else:
	# 		vec1 = curr_body_pose3d[joint_idx - 1, :3] - curr_body_pose3d[joint_idx, :3]
	# 	vec2 = curr_body_pose3d[joint_idx + 1, :3] - curr_body_pose3d[joint_idx, :3]
	# 	# Compute angle
	# 	joint_angle.append(angle_between(vec1, vec2))
	# joint_angle = np.array(joint_angle)

	# Filter invalid joints from valid joints (vis_flag)
	invalid_dist_flag = np.logical_or(
		jnt_dists < joint_dist_min_threshold,
		jnt_dists > joint_dist_max_threshold,
	)
	invalid_dist_flag_ = np.full((num_of_jnts,), False)
	invalid_dist_flag_[joint_dist_index] = invalid_dist_flag

	# invalid_angle_flag = np.logical_or(
	# 	joint_angle < joint_angle_min_threshold, joint_angle > joint_angle_max_threshold
	# )
	# invalid_angle_flag_ = np.full((num_of_jnts,), False)
	# invalid_angle_flag_[joint_angle_index] = invalid_angle_flag

	# invalid_flag = np.logical_or(invalid_dist_flag_, invalid_angle_flag_)
	invalid_flag = invalid_dist_flag_
	valid_flag = np.logical_not(invalid_flag)
	curr_body_pose3d[invalid_flag] = None
	return curr_body_pose3d, valid_flag



def hand_jnts_dist_angle_check(curr_hand_pose3d):
	"""
	Check hand biomechanical info: Distance and angle
	"""
	## Joint distance threshold ##
	num_of_jnts = curr_hand_pose3d.shape[0]
	long_joint_dist_index = [4, 8, 12, 16]
	joint_dist_min_threshold = np.full((num_of_jnts-1,), 0.005)
	joint_dist_min_threshold[long_joint_dist_index] = 0.06
	joint_dist_max_threshold = np.full((num_of_jnts-1,), 0.08)
	joint_dist_max_threshold[long_joint_dist_index] = 0.12
	##Joint angle threshold ##
	joint_angle_min_threshold = np.array(
		[100, 90, 90, 60, 70, 80, 60, 70, 80, 60, 70, 80, 60, 70, 80]
	)
	joint_angle_max_threshold = np.array([180] * 15)
	## Misc ##
	wrist_conn_index = [1, 5, 9, 13, 17]
	joint_dist_index = list(range(1, num_of_jnts))
	joint_angle_index = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]

	###### Joint distance check #######
	joint_distance = []
	for joint_idx in joint_dist_index:
		if joint_idx in wrist_conn_index:
			joint_distance.append(
				np.linalg.norm(
					curr_hand_pose3d[joint_idx][:3] - curr_hand_pose3d[0][:3]
				)
			)
		else:
			joint_distance.append(
				np.linalg.norm(
					curr_hand_pose3d[joint_idx][:3]
					- curr_hand_pose3d[joint_idx - 1][:3]
				)
			)
	joint_distance = np.array(joint_distance)

	###### Joint angle check ######
	joint_angle = []
	for joint_idx in joint_angle_index:
		# If current joint has pose3d estimation
		if joint_idx in wrist_conn_index:
			vec1 = curr_hand_pose3d[0, :3] - curr_hand_pose3d[joint_idx, :3]
		else:
			vec1 = curr_hand_pose3d[joint_idx - 1, :3] - curr_hand_pose3d[joint_idx, :3]
		vec2 = curr_hand_pose3d[joint_idx + 1, :3] - curr_hand_pose3d[joint_idx, :3]
		# Compute angle
		joint_angle.append(angle_between(vec1, vec2))
	joint_angle = np.array(joint_angle)

	# Filter invalid joints from valid joints (vis_flag)
	invalid_dist_flag = np.logical_or(
		joint_distance < joint_dist_min_threshold,
		joint_distance > joint_dist_max_threshold,
	)
	invalid_dist_flag_ = np.full((num_of_jnts,), False)
	invalid_dist_flag_[joint_dist_index] = invalid_dist_flag

	invalid_angle_flag = np.logical_or(
		joint_angle < joint_angle_min_threshold, joint_angle > joint_angle_max_threshold
	)
	invalid_angle_flag_ = np.full((num_of_jnts,), False)
	invalid_angle_flag_[joint_angle_index] = invalid_angle_flag

	invalid_flag = np.logical_or(invalid_dist_flag_, invalid_angle_flag_)
	curr_hand_pose3d[invalid_flag] = None
	return curr_hand_pose3d


def reproj_error_check(proj_2d_kpts, anno_2d_kpts, threshold):
	"""
	Reprojection error check between projected 2d kpts and annotation 2d kpts;
	Heuristics to filter good 3d kpts
	"""
	# Compute euclidean distance between each joint
	joint_dist = np.linalg.norm(proj_2d_kpts - anno_2d_kpts, axis=1)
	# Ignore hand wrist check due to large ambiguity region
	joint_dist[0] = 0
	# Thresholding
	valid_reproj_flag = joint_dist <= threshold
	return valid_reproj_flag


def get_interested_take(all_uids, takes_df):
	"""
	For hand ego-pose baseline model, we are only interested in takes with
	scenario in Health, Bike Repair, Music, Cooking
	"""
	interested_scenarios = ["Health", "Bike Repair", "Music", "Cooking"]
	scenario_take_dict = {scenario: [] for scenario in interested_scenarios}
	all_interested_scenario_uid = []
	for curr_local_cam_valid_uid in all_uids:
		curr_scenario = takes_df[takes_df["take_uid"] == curr_local_cam_valid_uid][
			"scenario_name"
		].item()
		if curr_scenario in interested_scenarios:
			scenario_take_dict[curr_scenario].append(curr_local_cam_valid_uid)
			all_interested_scenario_uid.append(curr_local_cam_valid_uid)
	all_interested_scenario_uid = sorted(all_interested_scenario_uid)
	return all_interested_scenario_uid, scenario_take_dict


def get_ego_pose_takes_from_splits(splits):
	"""
	Filter for ego pose takes from splits.json
	"""
	return [
		uid
		for uid in splits["take_uid_to_benchmark"]
		if "ego_pose" in splits["take_uid_to_benchmark"][uid]
	]




def extract_aria_calib_to_json(input_vrs, output_path):

	extract_calibration_json_cmd = f"vrs {input_vrs} | grep calib_json"

	p = subprocess.Popen(
		[extract_calibration_json_cmd], shell=True, stdout=subprocess.PIPE
	)
	out, err = p.communicate()
	out = out.decode("utf-8")

	calib_json_string = out[20:-1]

	with open(output_path, "w") as f:
		f.write(calib_json_string)
