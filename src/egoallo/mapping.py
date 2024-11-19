import numpy as np
"""
    smplx_keypoint_names: The names of the SMPL-X keypoints
    openpose25_keypoint_names: The names of the OpenPose keypoints. The order for OpenPose here is:
        25 body keypoints
        21 left hand keypoints
        21 right hand keypoints
        51 facial landmarks
        17 contour landmarks
    openpose_idxs: The indices of the OpenPose keypoint array.
    smplx_idxs: The corresponding SMPL-X indices.
"""

# 135 smplx ids
SMPLX_IDS =  [55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 20, 37, 38, 39, 66, 25, 26, 27, 67, 28, 29, 30, 68, 34, 35, 36, 69, 31, 32, 33, 70, 21, 52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45, 73, 49, 50, 51, 74, 46, 47, 48, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143]
# 144 smplx kpt names
# 144 smplx JOINT NAMES
SMPLX_JOINT_NAMES = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'jaw', 'left_eye_smplhf', 'right_eye_smplhf', 'left_index1', 'left_index2', 'left_index3', 'left_middle1', 'left_middle2', 'left_middle3', 'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_ring1', 'left_ring2', 'left_ring3', 'left_thumb1', 'left_thumb2', 'left_thumb3', 'right_index1', 'right_index2', 'right_index3', 'right_middle1', 'right_middle2', 'right_middle3', 'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_ring1', 'right_ring2', 'right_ring3', 'right_thumb1', 'right_thumb2', 'right_thumb3', 'nose', 'right_eye', 'left_eye', 'right_ear', 'left_ear', 'left_big_toe', 'left_small_toe', 'left_heel', 'right_big_toe', 'right_small_toe', 'right_heel', 'left_thumb', 'left_index', 'left_middle', 'left_ring', 'left_pinky', 'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_pinky', 'right_eye_brow1', 'right_eye_brow2', 'right_eye_brow3', 'right_eye_brow4', 'right_eye_brow5', 'left_eye_brow5', 'left_eye_brow4', 'left_eye_brow3', 'left_eye_brow2', 'left_eye_brow1', 'nose1', 'nose2', 'nose3', 'nose4', 'right_nose_2', 'right_nose_1', 'nose_middle', 'left_nose_1', 'left_nose_2', 'right_eye1', 'right_eye2', 'right_eye3', 'right_eye4', 'right_eye5', 'right_eye6', 'left_eye4', 'left_eye3', 'left_eye2', 'left_eye1', 'left_eye6', 'left_eye5', 'right_mouth_1', 'right_mouth_2', 'right_mouth_3', 'mouth_top', 'left_mouth_3', 'left_mouth_2', 'left_mouth_1', 'left_mouth_5', 'left_mouth_4', 'mouth_bottom', 'right_mouth_4', 'right_mouth_5', 'right_lip_1', 'right_lip_2', 'lip_top', 'left_lip_2', 'left_lip_1', 'left_lip_3', 'lip_bottom', 'right_lip_3', 'right_contour_1', 'right_contour_2', 'right_contour_3', 'right_contour_4', 'right_contour_5', 'right_contour_6', 'right_contour_7', 'right_contour_8', 'contour_middle', 'left_contour_8', 'left_contour_7', 'left_contour_6', 'left_contour_5', 'left_contour_4', 'left_contour_3', 'left_contour_2', 'left_contour_1']
# 135 openpose kpt names and ids
OPENPOSE25_KEYPOINT_NAMES = ['nose', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'left_shoulder', 'left_elbow', 'left_wrist', 'pelvis', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle', 'right_eye', 'left_eye', 'right_ear', 'left_ear', 'left_big_toe', 'left_small_toe', 'left_heel', 'right_big_toe', 'right_small_toe', 'right_heel', 'left_wrist', 'left_thumb1', 'left_thumb2', 'left_thumb3', 'left_thumb', 'left_index1', 'left_index2', 'left_index3', 'left_index', 'left_middle1', 'left_middle2', 'left_middle3', 'left_middle', 'left_ring1', 'left_ring2', 'left_ring3', 'left_ring', 'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_pinky', 'right_wrist', 'right_thumb1', 'right_thumb2', 'right_thumb3', 'right_thumb', 'right_index1', 'right_index2', 'right_index3', 'right_index', 'right_middle1', 'right_middle2', 'right_middle3', 'right_middle', 'right_ring1', 'right_ring2', 'right_ring3', 'right_ring', 'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_pinky', 'right_eye_brow1', 'right_eye_brow2', 'right_eye_brow3', 'right_eye_brow4', 'right_eye_brow5', 'left_eye_brow5', 'left_eye_brow4', 'left_eye_brow3', 'left_eye_brow2', 'left_eye_brow1', 'nose1', 'nose2', 'nose3', 'nose4', 'right_nose_2', 'right_nose_1', 'nose_middle', 'left_nose_1', 'left_nose_2', 'right_eye1', 'right_eye2', 'right_eye3', 'right_eye4', 'right_eye5', 'right_eye6', 'left_eye4', 'left_eye3', 'left_eye2', 'left_eye1', 'left_eye6', 'left_eye5', 'right_mouth_1', 'right_mouth_2', 'right_mouth_3', 'mouth_top', 'left_mouth_3', 'left_mouth_2', 'left_mouth_1', 'left_mouth_5', 'left_mouth_4', 'mouth_bottom', 'right_mouth_4', 'right_mouth_5', 'right_lip_1', 'right_lip_2', 'lip_top', 'left_lip_2', 'left_lip_1', 'left_lip_3', 'lip_bottom', 'right_lip_3', 'right_contour_1', 'right_contour_2', 'right_contour_3', 'right_contour_4', 'right_contour_5', 'right_contour_6', 'right_contour_7', 'right_contour_8', 'contour_middle', 'left_contour_8', 'left_contour_7', 'left_contour_6', 'left_contour_5', 'left_contour_4', 'left_contour_3', 'left_contour_2', 'left_contour_1']
OPENPOSE_IDXS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134]
# 10 smplx keypoints without openpose correspondences
SMPLX_KEYPOINTS_WO_OPENPOSE_CORRESPONDENCES=["spine1", "spine2", "spine3", "left_foot", "right_foot", "left_collar", "right_collar", "head", "left_eye_smplx", "right_eye_spmlx"]

# 22 smpl joints names
# ! 22 smpl joint names is not complete, just copying it here for integrity, refer to 24 smpl joints above.
SMPL_JOINTS = {'hips' : 0, 'leftUpLeg' : 1, 'rightUpLeg' : 2, 'spine' : 3, 'leftLeg' : 4, 'rightLeg' : 5,
                'spine1' : 6, 'leftFoot' : 7, 'rightFoot' : 8, 'spine2' : 9, 'leftToeBase' : 10, 'rightToeBase' : 11, 
                'neck' : 12, 'leftShoulder' : 13, 'rightShoulder' : 14, 'head' : 15, 'leftArm' : 16, 'rightArm' : 17,
                'leftForeArm' : 18, 'rightForeArm' : 19, 'leftHand' : 20, 'rightHand' : 21}
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12, 13, 14, 16, 17, 18, 19]

# chosen virtual mocap markers that are "keypoints" to work with
KEYPT_VERTS = [4404, 920, 3076, 3169, 823, 4310, 1010, 1085, 4495, 4569, 6615, 3217, 3313, 6713,
            6785, 3383, 6607, 3207, 1241, 1508, 4797, 4122, 1618, 1569, 5135, 5040, 5691, 5636,
            5404, 2230, 2173, 2108, 134, 3645, 6543, 3123, 3024, 4194, 1306, 182, 3694, 4294, 744]




# 73 smplh JOINT NAMES: 24-2+15*2
# ! smplh joint num should be smpl_joint_names - 2(because the last 2 smpl joint names are left hand and right hand.) + 15 * 2.
# ! Q: Why is 73 points instead of 52 joints in SMPLH? 15 joints for each hand + 22 body joints (same as the original SMPL)?
# ! A: The original smpl has 24 joints. Adding 15 hand joints, the using the 3 face joints and remaining are taken from the smplx leg joints to cover the whole body
SMPLH_JOINT_NAMES = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_index1', 'left_index2', 'left_index3', 'left_middle1', 'left_middle2', 'left_middle3', 'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_ring1', 'left_ring2', 'left_ring3', 'left_thumb1', 'left_thumb2', 'left_thumb3', 'right_index1', 'right_index2', 'right_index3', 'right_middle1', 'right_middle2', 'right_middle3', 'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_ring1', 'right_ring2', 'right_ring3', 'right_thumb1', 'right_thumb2', 'right_thumb3', 'nose', 'right_eye', 'left_eye', 'right_ear', 'left_ear', 'left_big_toe', 'left_small_toe', 'left_heel', 'right_big_toe', 'right_small_toe', 'right_heel', 'left_thumb', 'left_index', 'left_middle', 'left_ring', 'left_pinky', 'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_pinky']

SMPLH_BODY_JOINTS = SMPLH_JOINT_NAMES[:22]
SMPLH_HAND_JOINTS = SMPLH_JOINT_NAMES[22:52]


USED_SMPLH_JOINT_NAMES = SMPLH_JOINT_NAMES[:52]

# 24 smpl joint names
SMPL_JOINT_NAMES = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand']
NUM_SMPL_JNTS = len(SMPL_JOINT_NAMES)

# 15 hand joint names
# ! there may be additionally a `wrist` joint in the hand joints.
# ! the order is thumb, index, middle, ring, pinky. and if the finger tips are excluded, then the joint nums for each hand would decrease by 5.
HAND_JOINT_NAMES = ['index1', 'index2', 'index3', 'middle1', 'middle2', 'middle3', 'pinky1', 'pinky2', 'pinky3', 'ring1', 'ring2', 'ring3', 'thumb1', 'thumb2', 'thumb3']

COCO25_KEYPOINT_MAPPINGS = {0: 'Nose', 1: 'Neck', 2: 'RShoulder', 3: 'RElbow', 4: 'RWrist', 5: 'LShoulder', 6: 'LElbow', 7: 'LWrist', 8: 'MidHip', 9: 'RHip', 10: 'RKnee', 11: 'RAnkle', 12: 'LHip', 13: 'LKnee', 14: 'LAnkle', 15: 'REye', 16: 'LEye', 17: 'REar', 18: 'LEar', 19: 'LBigToe', 20: 'LSmallToe', 21: 'LHeel', 22: 'RBigToe', 23: 'RSmallToe', 24: 'RHeel', 25: 'Background'}
COCO18_KEYPOINT_MAPPINGS = {0: 'Nose', 1: 'Neck', 2: 'RShoulder', 3: 'RElbow', 4: 'RWrist', 5: 'LShoulder', 6: 'LElbow', 7: 'LWrist', 8: 'RHip', 9: 'RKnee', 10: 'RAnkle', 11: 'LHip', 12: 'LKnee', 13: 'LAnkle', 14: 'REye', 15: 'LEye', 16: 'REar', 17: 'LEar'}

# 17 in total
EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS = ['nose','left-eye','right-eye','left-ear','right-ear','left-shoulder','right-shoulder','left-elbow','right-elbow','left-wrist','right-wrist','left-hip','right-hip','left-knee','right-knee','left-ankle','right-ankle']
NUM_EGOEXO4D_EGOPOSE_JNTS = len(EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS)
# 42 in total, 21 for left hand, 21 for right hand.
EGOEXO4D_EGOPOSE_HANDPOSE_MAPPINGS = ['right_wrist', 'right_thumb_1', 'right_thumb_2', 'right_thumb_3', 'right_thumb_4', 'right_index_1', 'right_index_2', 'right_index_3', 'right_index_4', 'right_middle_1', 'right_middle_2', 'right_middle_3', 'right_middle_4', 'right_ring_1', 'right_ring_2', 'right_ring_3', 'right_ring_4', 'right_pinky_1', 'right_pinky_2', 'right_pinky_3', 'right_pinky_4', 'left_wrist', 'left_thumb_1', 'left_thumb_2', 'left_thumb_3', 'left_thumb_4', 'left_index_1', 'left_index_2', 'left_index_3', 'left_index_4', 'left_middle_1', 'left_middle_2', 'left_middle_3', 'left_middle_4', 'left_ring_1', 'left_ring_2', 'left_ring_3', 'left_ring_4', 'left_pinky_1', 'left_pinky_2', 'left_pinky_3', 'left_pinky_4']
OPENPOSE_HANDPOSE_MAPPINGS = EGOEXO4D_EGOPOSE_HANDPOSE_MAPPINGS

EGOEXO_HEAD_IDX = EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS.index('nose')
EGOEXO_ROOT_IDX = EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS.index('left-hip')

# ! The egoexo4d-bodypose <--> smplh-bodypose mapping is not consistent, as the head pose is mapped to left-eye.
# Totaling 24.
EGOEXO4D_BODYPOSE_TO_SMPL_INDICES = [-1, 11, 12, -1, 13, 14, -1, 15, 16, -1, -1, -1, -1, -1,-1, 1, 5, 6, 7, 8, 9, 10, -1, -1]
EGOEXO4D_BODYPOSE_TO_SMPLH_INDICES = EGOEXO4D_BODYPOSE_TO_SMPL_INDICES[:-2]

SMPLH_TO_EGOEXO4D_BODYPOSE_INDICES = [-1, 15, -1, -1, -1, 16,17, 18,19, 20, 21, 1, 2, 4, 5, 7, 8]

# ! IN general, the egoexo4d handpose <-> smplh hand part joints indices mapping should be consistent.
# 32 mapping indices to smplh-w-hand indice (left-wrist, right-wrist and 15*2 hand joints)
# this variable map to the higher-32-dimension of smplh_joint_names, starting from ind+22.
EGOEXO4D_HANDPOSE_TO_SMPLH_w_HAND_INDICES = [21, 0, 26, 27, 28, 30, 31, 32, 38,39, 40, 34, 35, 36, 22, 23, 24, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2,3 ]

SMPLH_w_HAND_TO_EGOEXO4D_HANDPOSE_INDICES = np.array([1, 29, 30, 31, -1, 17, 18, 19, -1, 20, 21, 22, -1, 11, 12, 13, -1, 23, 24, 25, -1, 0, 14, 15, 16, -1, 17, 18, 19, -1, 5, 6, 7, -1, 11, 12, 13, -1, 8, 9, 10 , -1]) + 20

# egoexo4d bodypose kintree parents mapping: NOTE that the kintree table is only used in hand jnts dist angle check func
# and it doesn't represents any kinematic or hierarchical grounded structure.
EGOEXO4D_BODYPOSE_KINTREE_PARENTS = [-1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 11, 11, 12, 13, 14]