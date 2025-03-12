# Standard library imports
import json
import os
import os.path as osp
import pickle
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

import torch
from egoallo.config import CONFIG_FILE
from egoallo.config import make_cfg
from egoallo.egopose.bodypose.bodypose_dataloader import body_pose_anno_loader
from egoallo.egopose.handpose.data_preparation.utils.config import (
    create_egopose_processing_argparse,
)
from egoallo.utils.setup_logger import setup_logger
from egoallo.utils.smpl_mapping.mapping import EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS
from egoallo.utils.smpl_mapping.mapping import EGOEXO4D_EGOPOSE_HANDPOSE_MAPPINGS
from egoallo.utils.utils import debug_on_error
from yacs.config import CfgNode as CN
# Third-party imports


# Set environment variables for threading
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Configure torch multiprocessing
torch.multiprocessing.set_start_method("spawn", force=True)

# Local imports

local_config_file = CONFIG_FILE
CFG = make_cfg(config_name="defaults", config_file=local_config_file, cli_args=[])


logger = setup_logger(output=None, name=__name__)

debug_on_error(debug=True, logger=logger)

BODY_JOINTS = EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS
HAND_JOINTS = EGOEXO4D_EGOPOSE_HANDPOSE_MAPPINGS
NUM_OF_HAND_JOINTS = len(HAND_JOINTS) // 2
NUM_OF_BODY_JOINTS = len(BODY_JOINTS)
NUM_OF_JOINTS = NUM_OF_BODY_JOINTS + NUM_OF_HAND_JOINTS * 2


def save_stats_collector(
    stats_collector,
    output_path: str,
    split: str,
    anno_type: str,
) -> None:
    """Save stats collector to disk"""
    save_path = Path(output_path) / f"stats_collector_{split}_{anno_type}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(stats_collector, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved stats collector to {save_path}")


def load_stats_collector(output_path: str, split: str, anno_type: str):
    """Load stats collector from disk"""
    load_path = Path(output_path) / f"stats_collector_{split}_{anno_type}.pkl"
    with open(load_path, "rb") as f:
        return pickle.load(f)


def analyze_preprocessing_pipeline(args, output_path: str) -> None:
    """
    Performs detailed analysis of the body pose preprocessing pipeline and logs statistics
    """
    logger = setup_logger(
        output=osp.join(output_path, "preprocessing_analysis.txt"),
        name="preprocessing_analysis",
    )

    # Load data for each split and annotation type
    for anno_type in args.anno_types:
        for split in args.splits:
            logger.info(f"\nAnalyzing {split} split with {anno_type} annotations...")

            # Create data loader
            gt_anno = body_pose_anno_loader(args, split, anno_type)

            # Save stats collector
            save_stats_collector(gt_anno.stats_collector, output_path, split, anno_type)

            # Log analysis results
            summary_stats = gt_anno.stats_collector.get_summary_stats()
            log_analysis_results(summary_stats, split, anno_type, logger)

            gt_anno_output_dir = (
                osp.join(output_path, "annotation", anno_type)
                if split in ["train", "val"]
                else osp.join(output_path, "annotation")
            )
            os.makedirs(gt_anno_output_dir, exist_ok=True)
            # Save ground truth JSON file
            if split in ["train", "val"]:
                with open(
                    osp.join(
                        gt_anno_output_dir,
                        f"ego_pose_gt_anno_{split}_public.json",
                    ),
                    "w",
                ) as f:
                    json.dump(gt_anno.db, f)
            else:
                if len(gt_anno.db) == 0:
                    logger.info("[Warning] No gt-anno is found in the local file.")
                else:
                    with open(
                        osp.join(
                            gt_anno_output_dir,
                            f"ego_pose_gt_anno_{split}_public.json",
                        ),
                        "w",
                    ) as f:
                        json.dump(gt_anno.db, f)


def log_analysis_results(
    stats: Dict[str, Any],
    split: str,
    anno_type: str,
    logger,
) -> None:
    """Log detailed analysis results with enhanced filtering statistics"""
    logger.info(
        f"\n=== BODY POSE PREPROCESSING ANALYSIS {split.upper()} {anno_type.upper()} ===\n",
    )

    # Overall statistics
    logger.info("Overall Statistics:")
    logger.info(f"Total Takes Processed: {stats['total_takes']}")
    logger.info(
        f"Valid Takes: {stats['valid_takes']} ({stats['valid_takes'] / stats['total_takes'] * 100:.2f}%)",
    )
    logger.info(f"Total Frames: {stats['total_frames']}")
    logger.info(
        f"Valid Frames: {stats['valid_frames']} ({stats['valid_frames'] / stats['total_frames'] * 100:.2f}%)",
    )

    # Keypoint filtering breakdown
    logger.info("\nKeypoint Filtering Statistics:")
    percentages = stats["keypoint_percentages"]
    logger.info("Filtering Breakdown (% of total keypoints):")
    logger.info(f"├── Missing Annotations: {percentages['missing_annotation']:.2f}%")
    logger.info(
        f"├── Biomechanical Invalid: {percentages['biomechanical_invalid']:.2f}%",
    )
    logger.info(f"├── Visibility Invalid: {percentages['visibility_invalid']:.2f}%")
    logger.info(f"├── Projection Error: {percentages['projection_error']:.2f}%")
    logger.info(f"└── Final Valid: {percentages['final_valid']:.2f}%")

    # Add valid keypoint distribution analysis
    if "keypoint_distribution" in stats:
        logger.info("\nValid Keypoint Distribution:")
        distribution = stats["keypoint_distribution"]
        total_valid = sum(distribution.values())

        # Sort by keypoint index
        sorted_dist = dict(sorted(distribution.items()))

        for idx, count in sorted_dist.items():
            percentage = (count / total_valid) * 100
            keypoint_name = EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS[idx]
            logger.info(
                f"Keypoint {idx:2d} ({keypoint_name:12s}): {count:6d} occurrences ({percentage:5.2f}%)",
            )

    # Validation criteria explanation
    logger.info("\nValidation Criteria:")
    logger.info("1. Annotation Check:")
    logger.info("   - Verifies presence of 2D and 3D annotations")
    logger.info("2. Multi-view Check:")
    logger.info("   - Requires ≥2 camera views for triangulation")
    logger.info("3. Biomechanical Validation:")
    logger.info("   - Checks joint angles and distances")
    logger.info("   - Validates anatomical constraints")
    logger.info("4. Geometric Validation:")
    logger.info("   - Ensures realistic bone lengths")
    logger.info("   - Validates joint positions")
    logger.info("5. Visibility and Bounds Check:")
    logger.info("   - Validates keypoints within image bounds")
    logger.info("   - Checks visibility in camera masks")


def create_body_gt_anno(args):
    """Creates ground truth annotation file with detailed analysis"""
    logger.info("Generating ground truth annotation files...")

    # Run the original annotation creation process
    gt_output_dir = (
        args.gt_bodypose.output.save_dir
        if not args.gt_bodypose.run_demo
        else args.gt_bodypose.output_dir
    )

    # Add analysis
    analysis_output_path = os.path.join(gt_output_dir, "preprocessing_analysis.txt")
    analyze_preprocessing_pipeline(args, analysis_output_path)


def print_saved_stats(
    output_path: str,
    splits: List[str],
    anno_types: List[str],
) -> None:
    """Print summary statistics from saved stats collectors"""
    logger = setup_logger(output=output_path, name="saved_stats_analysis")

    for anno_type in anno_types:
        for split in splits:
            try:
                # Load saved stats collector
                stats_collector = load_stats_collector(output_path, split, anno_type)

                # Get and log summary stats
                summary_stats = stats_collector.get_summary_stats()
                logger.info(
                    f"\nLoaded statistics for {split} split with {anno_type} annotations:",
                )
                log_analysis_results(summary_stats, split, anno_type, logger)

            except FileNotFoundError:
                logger.warning(
                    f"No saved stats found for {split} split with {anno_type} annotations",
                )


def extract_ground_heights(
    splits: List[str],
    anno_types: List[str],
    output_path: str,
) -> None:
    """
    Extract ground heights from saved annotation files and create a mapping JSON.

    Args:
        args: Configuration arguments
        output_path: Base output directory path
    """
    logger.info("Extracting ground heights from annotation files...")

    # Initialize ground heights dictionary
    ground_heights = {}

    # Process each split and annotation type
    for split in splits:
        for anno_type in anno_types:
            # Determine annotation directory based on split
            if split in ["train", "val"]:
                gt_anno_path = osp.join(
                    output_path,
                    "annotation",
                    anno_type,
                    f"ego_pose_gt_anno_{split}_public.json",
                )
            else:
                gt_anno_path = osp.join(
                    output_path,
                    "annotation",
                    f"ego_pose_gt_anno_{split}_public.json",
                )

            # Read annotation file if it exists
            if osp.exists(gt_anno_path):
                logger.info(f"Processing {split} split with {anno_type} annotations...")
                with open(gt_anno_path, "r") as f:
                    annotations = json.load(f)

                # Extract ground heights from each take's metadata
                for take_uid, take_data in annotations.items():
                    if "metadata" in take_data:
                        ground_height = take_data["metadata"].get("ground_height")
                        if ground_height is not None:
                            ground_heights[take_uid] = ground_height
            else:
                logger.warning(f"Annotation file not found: {gt_anno_path}")

    # Save ground heights mapping
    output_file = osp.join(output_path, "ground_heights.json")
    with open(output_file, "w") as f:
        json.dump(ground_heights, f)
    logger.info(f"Saved ground heights mapping to {output_file}")
    logger.info(f"Processed {len(ground_heights)} takes with valid ground heights")


def main(cfg):
    """Main entry point for preprocessing pipeline"""
    output_path = cfg.gt_bodypose.output.log_save_dir

    # Run preprocessing analysis
    analyze_preprocessing_pipeline(cfg, output_path)

    # Optionally print saved stats
    if cfg.get("print_saved_stats", False):
        print_saved_stats(output_path, splits=cfg.splits, anno_types=cfg.anno_types)


if __name__ == "__main__":
    #     deterministic()
    cli_opt = create_egopose_processing_argparse()
    cli_opt_dict = vars(cli_opt)
    cli_opt_dict = CN(cli_opt_dict)

    # Load and merge configurations
    local_cfg = CFG
    local_cfg.defrost()
    preprocess_cfg = local_cfg.io.egoexo.preprocessing
    preprocess_cfg.merge_from_other_cfg(cli_opt_dict)
    preprocess_cfg.merge_from_file(cli_opt.config_file)
    preprocess_cfg.valid_kpts_num_thresh = 4
    local_cfg.io.egoexo.preprocessing = preprocess_cfg
    local_cfg.freeze()

    # Set up output directories
    if not preprocess_cfg.gt_bodypose.run_demo:
        gt_output_dir = preprocess_cfg.gt_bodypose.output.save_dir
        cfg_save_dir = preprocess_cfg.gt_bodypose.output.config_save_dir
        for dir_path in [
            gt_output_dir,
            cfg_save_dir,
            preprocess_cfg.gt_bodypose.output.log_save_dir,
        ]:
            os.makedirs(dir_path, exist_ok=True)
    else:
        gt_output_dir = preprocess_cfg.gt_bodypose.sample_output.save_dir
        cfg_save_dir = preprocess_cfg.gt_bodypose.sample_output.config_save_dir
        for dir_path in [
            gt_output_dir,
            cfg_save_dir,
            preprocess_cfg.gt_bodypose.sample_output.log_save_dir,
        ]:
            os.makedirs(dir_path, exist_ok=True)

    # Save configuration
    local_cfg_save_path = osp.join(cfg_save_dir, "preprocess_dataset.yaml")
    with open(local_cfg_save_path, "w") as f:
        with redirect_stdout(f):
            print(preprocess_cfg.dump())

    main(preprocess_cfg)
    # tyro.cli(extract_ground_heights)
