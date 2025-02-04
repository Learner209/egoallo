from typing import Dict

# Hand evaluation constants
EGOEXO_NAMES_LEFT = [
    "left_wrist",
    "left_index_1",
    "left_index_2",
    "left_index_3",
    "left_middle_1",
    "left_middle_2",
    "left_middle_3",
    "left_pinky_1",
    "left_pinky_2",
    "left_pinky_3",
    "left_ring_1",
    "left_ring_2",
    "left_ring_3",
    "left_thumb_1",
    "left_thumb_2",
    "left_thumb_3",
    "left_thumb_4",
    "left_index_4",
    "left_middle_4",
    "left_ring_4",
    "left_pinky_4",
]

EGOEXO_NAMES_RIGHT = [name.replace("left", "right") for name in EGOEXO_NAMES_LEFT]

# Vertex indices for different model types
VERTEX_IDS: Dict[str, Dict[str, int]] = {
    "smplh": {
        "lthumb": 2746,
        "lindex": 2319,
        "lmiddle": 2445,
        "lring": 2556,
        "lpinky": 2673,
        "rthumb": 6191,
        "rindex": 5782,
        "rmiddle": 5905,
        "rring": 6016,
        "rpinky": 6133,
    },
}

# Body evaluation constants
FOOT_INDICES = [6, 7, 9, 10]
HEAD_JOINT_INDEX = 14
FOOT_HEIGHT_THRESHOLDS = [0.08, 0.08, 0.04, 0.04]

# Metrics names
BODY_METRICS = [
    "mpjpe",
    "pampjpe",
    "head_ori",
    "head_trans",
    "foot_skate",
    "foot_contact",
]

# Default file names
DEFAULT_METRICS_FILENAME = "_eval_cached_disaggregated_metrics{}.npz"
DEFAULT_SUMMARY_FILENAME = "_eval_cached_summary{}.yaml"
