from __future__ import annotations
import dataclasses
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import torch
import tyro
from tqdm import tqdm

from egoallo.data.hps.hps_processor import HPSProcessor
from egoallo.setup_logger import setup_logger
from egoallo.training_utils import ipdb_safety_net

logger = setup_logger(output="logs/hps_preprocess", name=__name__)

@dataclasses.dataclass
class HPSPreprocessConfig:
    """Configuration for preprocessing HPS dataset."""
    # Dataset paths and options
    hps_data_dir: Path = Path("./datasets/HPS")
    smplh_model_dir: Path = Path("./assets/smpl_based_model/smplh/")
    output_dir: Path = Path("./data/hps/processed")
    output_list_file: Path = Path("./data/hps/hps_dataset_files.txt")
    
    # Processing options
    target_fps: int = 30
    min_sequence_length: int = 30
    max_sequence_length: int = 300
    include_hands: bool = True
    num_processes: int = 8
    
    # Data split options
    splits: list[str] = dataclasses.field(default_factory=lambda: ["train", "val", "test"])
    
    # Device options
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Debug options
    debug: bool = False

def process_sequence(args: tuple[HPSProcessor, str, Path]) -> Optional[str]:
    """Process a single HPS sequence and save as npz file.
    
    Args:
        args: Tuple of (processor, sequence_name, output_path)
        
    Returns:
        Relative path to the saved npz file if successful, None otherwise
    """
    processor, sequence_name, output_path = args

    # Incorporate caching strategy
    if output_path.exists():
        logger.info(f"Sequence {sequence_name} already processed, skipping")
        return str(output_path.relative_to(output_path.parent.parent) / f"{sequence_name}.npz")
    
    # try:
    # Process sequence with early exit check
    logger.info(f"Processing sequence {sequence_name}")
    processed_data = processor.process_sequence(sequence_name)
    
    # If None returned, sequence was invalid or too short
    if processed_data is None:
        return None
    
    # Save processed data
    processor.save_sequence(processed_data, output_path)
    
    # Return relative path for file list
    rel_path = output_path.relative_to(output_path.parent.parent)
    logger.info(f"Successfully processed sequence {sequence_name}")
    return str(rel_path)
    # except Exception as e:
    #     logger.error(f"Error processing {sequence_name}: {str(e)}")
    #     return None

def main(config: HPSPreprocessConfig) -> None:
    """Main preprocessing function."""
    start_time = time.time()
    
    # Create output directories
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = HPSProcessor(
        hps_dir=str(config.hps_data_dir),
        smplh_dir=str(config.smplh_model_dir),
        output_dir=str(config.output_dir),
        fps=config.target_fps,
        include_hands=config.include_hands,
        device=config.device
    )
    
    # Set up task queue for parallel processing
    task_queue = queue.Queue()
    processed_files: list[str] = []
    
    # Collect all sequences to process
    sequences = []
    for split in config.splits:
        split_dir = config.output_dir / split
        split_dir.mkdir(exist_ok=True)
        
        # Get sequences from HPS data directory structure
        seq_dir = config.hps_data_dir / 'hps_smpl'
        sequences.extend([
            p.stem for p in seq_dir.glob("*.pkl")
        ])
    
    # Add sequences to processing queue
    for seq_name in sequences:
        split = "train"  # Determine split based on your criteria
        output_path = config.output_dir / split / f"{seq_name}.npz"
        task_queue.put_nowait((processor, seq_name, output_path))
    
    total_count = task_queue.qsize()
    logger.info(f"Found {total_count} sequences to process")
    
    def worker(device_idx: int) -> None:
        while True:
            try:
                args = task_queue.get_nowait()
            except queue.Empty:
                break
                
            rel_path = process_sequence(args)
            if rel_path is not None:
                processed_files.append(rel_path)
                
            logger.info(
                f"Progress: {total_count - task_queue.qsize()}/{total_count} "
                f"({(total_count - task_queue.qsize())/total_count * 100:.2f}%)"
            )
    
    # Start worker threads
    workers = [
        threading.Thread(target=worker, args=(i,))
        for i in range(config.num_processes)
    ]
    # for w in workers:
    #     w.start()
    # for w in workers:
    #     w.join()
    
    worker(0)
    
    # Save file list
    config.output_list_file.write_text("\n".join(sorted(processed_files)))
    
    total_time = time.time() - start_time
    logger.info(f"Total processing time: {total_time/60:.2f} minutes")

if __name__ == "__main__":
    config = tyro.cli(HPSPreprocessConfig)
    if config.debug:
        import ipdb
        ipdb.set_trace()
    
    ipdb_safety_net()
    main(config)