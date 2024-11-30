import argparse


def create_egopose_processing_argparse():
    parser = argparse.ArgumentParser("Ego-pose baseline model dataset preparation")

    # Parameters of data preparation pipeline
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="train/val/test split of the dataset",
    )
    parser.add_argument(
        "--steps",
        type=str,
        nargs="+",
        default=["aria_calib", "hand_gt_anno", "body_gt_anno", "raw_image", "undistorted_image"],
        help="""
            Determine which step should be executed in data preparation:
            - aria_calib: Generate aria calibration JSON file for easier loading
            - gt_anno: Extract ground truth annotation file
            - raw_image: Extract raw ego-view (aria) images
            - undistorted_image: Undistort raw aria images
            """,
    )
    parser.add_argument(
        "--anno_types",
        type=str,
        nargs="+",
        default=["manual", "auto"],
        help="Type of annotation: use manual or automatic data",
    )
    # For merge from config file.
    parser.add_argument("--config_file", type=str, default="config/experiment.yaml")

    args = parser.parse_args()

    for split in args.splits:
        assert split in ["train", "val", "test"], f"Invalid split: {split}"

    for anno_type in args.anno_types:
        assert anno_type in [
            "manual",
            "auto",
        ], f"Invalid annotation type: {anno_type}"

    return args
