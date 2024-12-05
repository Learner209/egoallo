"""Script to preprocess AMASS dataset."""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Optional, List

import tyro
from tqdm import tqdm

from egoallo.data.amass.amass_processor import AMASSProcessor
from egoallo.data.amass.amass_dataset_config import AMASSDatasetConfig
from egoallo.utils.setup_logger import setup_logger
from egoallo.training_utils import ipdb_safety_net

logger = setup_logger(output="logs/amass_preprocess", name=__name__)


def process_sequence(
    args: tuple[AMASSProcessor, Path, Path]
) -> Optional[str]:
    """Process a single AMASS sequence.
    
    Args:
        args: Tuple of (processor, sequence_path, output_path)
        
    Returns:
        Relative path to processed file if successful
    """
    processor, seq_path, output_path = args
    
    # Skip if already processed
    if output_path.exists():
        rel_path = output_path.relative_to(output_path.parent.parent)
        logger.info(f"Skipping {seq_path.name}, {output_path} - already processed")
        return str(rel_path)
    
    try:
        # Process sequence
        sequence_data = processor.process_sequence(seq_path)
        if sequence_data is not None:
            processor.save_sequence(sequence_data, output_path)
            rel_path = output_path.relative_to(output_path.parent.parent)
            return str(rel_path)
    except Exception as e:
        logger.error(f"Error processing {seq_path}: {str(e)}")
    
    return None


def main(config: AMASSDatasetConfig) -> None:
    """Main preprocessing function.
    
    Args:
        config: Preprocessing configuration
    """
    start_time = time.time()
    
    # Initialize processor
    processor = AMASSProcessor(**config.get_processor_kwargs())
    
    # Set up processing queue
    task_queue = queue.Queue()
    processed_files: List[str] = []
    
    # Add sequences to queue
    all_datasets = config.train_datasets + config.val_datasets + config.test_datasets
    for dataset in all_datasets:
        dataset_dir = config.amass_dir / dataset
        if not dataset_dir.exists():
            logger.warning(f"Dataset directory not found: {dataset_dir}")
            continue
            
        # Determine output split
        if dataset in config.train_datasets:
            split = "train"
        elif dataset in config.val_datasets:
            split = "val"
        else:
            split = "test"
            
        # Create output directory
        output_dir = config.output_dir / split
        output_dir.mkdir(exist_ok=True)
        # Add sequences to queue
        for seq_path in dataset_dir.rglob("**/*.npz"):
            # Create unique identifier from full relative path
            rel_path = seq_path.relative_to(dataset_dir)
            unique_id = str(rel_path).replace('/', '_').replace('\\', '_')
            output_path = output_dir / f"{dataset}_{unique_id}"
            # assert output_path.exists()
            task_queue.put_nowait((processor, seq_path, output_path))
    
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
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    
    # Single-threaded for debugging
    # device_idx = 0
    # worker(device_idx)

    # Save file list
    config.output_list_file.write_text("\n".join(sorted(processed_files)))
    
    total_time = time.time() - start_time
    logger.info(f"Total processing time: {total_time/60:.2f} minutes")


if __name__ == "__main__":
    config = tyro.cli(AMASSDatasetConfig)
    if config.debug:
        import ipdb
        ipdb.set_trace()

    ipdb_safety_net()
    main(config) 