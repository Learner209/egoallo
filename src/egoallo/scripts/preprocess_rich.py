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

from egoallo.data.rich.rich_processor import RICHDataProcessor
from egoallo.setup_logger import setup_logger
from egoallo.training_utils import ipdb_safety_net

logger = setup_logger(output="logs/rich_preprocess", name=__name__)

@dataclasses.dataclass
class RICHPreprocessConfig:
    """Configuration for preprocessing RICH dataset."""
    # Dataset paths and options
    rich_data_dir: Path = Path("./third_party/rich_toolkit")
    smplh_model_dir: Path = Path("./assets/smpl_based_model/smplh/")
    smplx_model_dir: Path = Path("./third_party/rich_toolkit/body_models/smplx")
    output_dir: Path = Path("./data/rich/processed")
    output_list_file: Path = Path("./data/rich/rich_dataset_files.txt")
    
    # Processing options
    target_fps: int = 30
    min_sequence_length: int = 30
    max_sequence_length: int = 300
    include_contact: bool = True
    use_pca: bool = True
    num_processes: int = 8
    
    # Data split options
    splits: list[str] = dataclasses.field(default_factory=lambda: ["train", "val", "test"])
    
    # Device options
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Debug options
    debug: bool = False

def process_sequence(args: tuple[RICHDataProcessor, str, str, Path]) -> Optional[str]:
    """Process a single RICH sequence and save as npz file.
    
    Args:
        args: Tuple of (processor, split, seq_name, output_path)
        
    Returns:
        Relative path to the saved npz file if successful, None otherwise
    """
    processor, split, seq_name, output_path = args
    
    # try:
    # Process sequence with early exit check
    processed_data = processor.process_sequence(split, seq_name, output_path)
    
    # If None returned, file already exists
    if processed_data is None:
        rel_path = output_path.relative_to(output_path.parent.parent)
        return str(rel_path)
    
    # Save as npz if new processing was needed
    processor.save_sequence(processed_data, output_path)
    
    # Return relative path for file list
    rel_path = output_path.relative_to(output_path.parent.parent)
    logger.info(f"Successfully processed sequence {seq_name}")
    return str(rel_path)
    # except Exception as e:
    #     logger.error(f"Error processing {seq_name}: {str(e)}")
    #     return None

def main(config: RICHPreprocessConfig) -> None:
    """Main preprocessing function."""
    start_time = time.time()
    
    # Create output directories
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = RICHDataProcessor(
        rich_data_dir=str(config.rich_data_dir),
        smplh_model_dir=str(config.smplh_model_dir),
        smplx_model_dir=str(config.smplx_model_dir),
        output_dir=str(config.output_dir),
        fps=config.target_fps,
        include_contact=config.include_contact,
        device=config.device
    )
    
    # Set up task queue for parallel processing
    task_queue = queue.Queue()
    processed_files: list[str] = []
    
    # Collect all sequences to process
    config.splits = ["train"]
    for split in config.splits:
        split_dir = config.output_dir / split
        split_dir.mkdir(exist_ok=True)
        
        sequences = [d.name for d in (config.rich_data_dir / 'data/bodies' / split).iterdir() if d.is_dir()]
        
        for seq_name in sequences:
            seq_name = "ParkingLot1_005_burpeejump2"
            output_path = split_dir / f"{seq_name}.npz"
            # No need to check existence here since it's handled in process_sequence
            task_queue.put_nowait((processor, split, seq_name, output_path))
    
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
    # workers = [
    #     threading.Thread(target=worker, args=(i,))
    #     for i in range(config.num_processes)
    # ]
    # for w in workers:
    #     w.start()
    # for w in workers:
    #     w.join()

    # Single thread version
    for i in range(total_count):
        args = task_queue.get()
        rel_path = process_sequence(args)
        if rel_path is not None:
            processed_files.append(rel_path)
            
        logger.info(
            f"Progress: {i+1}/{total_count} "
            f"({(i+1)/total_count * 100:.2f}%)"
        )
    
    # Add existing npz files to processed_files list
    for split in config.splits:
        split_dir = config.output_dir / split
        existing_files = split_dir.glob("*.npz")
        for file_path in existing_files:
            rel_path = file_path.relative_to(config.output_dir)
            if str(rel_path) not in processed_files:
                processed_files.append(str(rel_path))
    
    # Save file list
    config.output_list_file.write_text("\n".join(processed_files))
    
    total_time = time.time() - start_time
    logger.info(f"Total processing time: {total_time/60:.2f} minutes")

if __name__ == "__main__":
    # Parse config and run
    config = tyro.cli(RICHPreprocessConfig)
    if config.debug:
        import ipdb
        ipdb.set_trace()
    
    ipdb_safety_net()
    main(config) 