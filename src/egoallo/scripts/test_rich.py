from __future__ import annotations
import dataclasses
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import h5py
import torch
import tyro
from tqdm import tqdm

from egoallo.data.rich.rich_processor import RICHDataProcessor
from egoallo.data.rich.rich_dataset_config import RICHDatasetConfig
from egoallo.setup_logger import setup_logger

logger = setup_logger(output="logs/rich_preprocess", name=__name__)

@dataclasses.dataclass
class RICHPreprocessConfig:
    """Configuration for preprocessing RICH dataset."""
    # Dataset paths and options
    rich_data_dir: Path = Path("./third_party/rich_toolkit")
    smplx_model_dir: Path = Path("./third_party/rich_toolkit/body_models/smplx")
    output_dir: Path = Path("./data/rich/processed_data")
    output_hdf5: Path = Path("./data/rich/rich_dataset.hdf5")
    output_list_file: Path = Path("./data/rich/rich_dataset_files.txt")
    
    # Processing options
    target_fps: int = 30
    min_sequence_length: int = 30  # Minimum frames per sequence
    max_sequence_length: int = 300  # Maximum frames per sequence
    include_contact: bool = True
    use_pca: bool = True
    num_processes: int = 4
    
    # Data split options
    splits: list[str] = dataclasses.field(default_factory=lambda: ["train", "val", "test"])
    
    # Device options
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Debug options
    debug: bool = False

def process_sequence(args: tuple[RICHDataProcessor, str, str, Path]) -> Optional[str]:
    """Process a single RICH sequence and return its group name if successful.
    
    Args:
        processor: RICH data processor instance
        split: Dataset split name (train/val/test)
        seq_name: Sequence name
        output_path: Path to save processed sequence
        
    Returns:
        Group name for the sequence if processing successful, None otherwise
    """
    processor, split, seq_name, output_path = args
    try:
        # Process sequence
        processed_data = processor.process_sequence(split, seq_name)
        
        # Save sequence
        processor.save_sequence(processed_data, output_path)
        
        # Return group name for file list
        group_name = f"{split}/{seq_name}"
        logger.info(f"Successfully processed sequence {group_name}")
        return group_name
        
    except Exception as e:
        logger.error(f"Error processing sequence {seq_name}: {str(e)}")
        return None

def main(config: RICHPreprocessConfig) -> None:
    """Main preprocessing function.
    
    Args:
        config: Preprocessing configuration
    """
    start_time = time.time()
    
    # Create output directories
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = RICHDataProcessor(
        rich_data_dir=str(config.rich_data_dir),
        smplx_model_dir=str(config.smplx_model_dir),
        output_dir=str(config.output_dir),
        fps=config.target_fps,
        include_contact=config.include_contact,
        use_pca=config.use_pca,
        device=config.device
    )
    
    # Set up task queue for parallel processing
    task_queue = queue.Queue[tuple[RICHDataProcessor, str, str, Path]]()
    
    # Collect all sequences to process
    for split in config.splits:
        split_dir = config.rich_data_dir / 'data/bodies' / split
        sequences = [d.name for d in split_dir.iterdir() if d.is_dir()]
        
        for seq_name in sequences:
            output_path = config.output_dir / f"{split}_{seq_name}.pt"
            if not output_path.exists():  # Skip if already processed
                task_queue.put_nowait((processor, split, seq_name, output_path))
    
    total_count = task_queue.qsize()
    logger.info(f"Found {total_count} sequences to process")
    
    # Process sequences and collect group names for file list
    file_list: list[str] = []
    
    def worker(device_idx: int) -> None:
        """Worker function for parallel processing."""
        while True:
            try:
                args = task_queue.get_nowait()
            except queue.Empty:
                break
                
            group_name = process_sequence(args)
            if group_name is not None:
                file_list.append(group_name)
                
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
    
    # Save processed sequences to HDF5
    logger.info("Saving sequences to HDF5...")
    with h5py.File(config.output_hdf5, "w") as f:
        for split in config.splits:
            split_dir = config.output_dir / split
            for seq_path in tqdm(list(split_dir.glob("*.pt")), desc=f"Processing {split}"):
                # Load processed sequence
                seq_data = torch.load(seq_path, map_location="cpu")
                
                # Create HDF5 group and datasets
                group_name = f"{split}/{seq_path.stem}"
                group = f.create_group(group_name)
                
                for k, v in seq_data.items():
                    if isinstance(v, torch.Tensor):
                        v = v.numpy()
                    chunks = (min(32, v.shape[0]),) + v.shape[1:] if v.ndim > 1 else None
                    group.create_dataset(k, data=v, chunks=chunks)
    
    # Save file list
    config.output_list_file.write_text("\n".join(file_list))
    
    total_time = time.time() - start_time
    logger.info(f"Total processing time: {total_time/60:.2f} minutes")

if __name__ == "__main__":
    # Set up debugging
    if config.debug:
        import ipdb
        ipdb.set_trace()
    
    # Parse config and run
    config = tyro.cli(RICHPreprocessConfig)
    main(config) 