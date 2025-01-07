from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

from typing import Any
from egoallo.types import DatasetType, DatasetSliceStrategy, DatasetSplit
from egoallo.guidance_optimizer_jax import GuidanceMode


@dataclasses.dataclass
class InferenceConfig:
    """Configuration for inference."""

    start_index: int = 0
    """Start index of trajectory sequence to process"""

    traj_length: int = 128
    """Length of trajectory sequence to process"""

    num_samples: int = 1
    """Number of samples to generate during inference"""

    batch_size: int = 1
    """Batch size for inference"""

    smplh_npz_path: Path = Path("./data/smplh/neutral/model.npz")
    """Path to SMPL+H model file"""

    output_dir: Path = Path("./exp/test-amass")
    """Directory to save inference outputs"""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    """Device to run inference on"""

    annotation_path: Path = Path("./data/egoexo-default-gt-output")
    """Path to ground truth annotations"""

    checkpoint_dir: Path = Path("./experiments/nov_29_absrel_jnts_pilot/v2")
    """Directory containing model checkpoints"""

    # Evaluation configs
    compute_metrics: bool = True
    """Whether to compute evaluation metrics after inference"""

    use_mean_body_shape: bool = False
    """Whether to use mean body shape for evaluation"""

    skip_eval_confirm: bool = True
    """Whether to skip confirmation before evaluation"""

    num_workers: int = 4
    """Number of workers for data loading"""

    visualize_traj: bool = False
    """Whether to visualize trajectories"""

    guidance_post: bool = False
    """Post guidance weight"""

    guidance_inner: bool = False
    """Inner guidance weight"""

    guidance_mode: GuidanceMode = "aria_hamer"
    """Which guidance mode to use."""

    dataset_type: DatasetType = "AdaptiveAmassHdf5Dataset"
    """Dataset type to use"""

    dataset_slice_strategy: DatasetSliceStrategy = "full_sequence"
    """Dataset slice strategy to use"""

    splits: tuple[DatasetSplit, ...] = ("test",)
    """Dataset splits to use"""
    @dataclasses.dataclass
    class EgoExoConfig:
        """Configuration specific to EgoExo dataset."""
        
        dataset_path: Path = Path("./datasets/egoexo-default")
        """Path to EgoExo dataset"""
        
        bodypose_anno_dir: tuple[Path, ...] = (
            Path("./data/egoexo-default-gt-output/annotation/manual"),
        )  # type: ignore
        """Path to body pose annotation directory"""
        
        anno_type: str = "manual"
        """Type of annotations to use (e.g. 'manual', 'auto')"""

        # Test config attributes
        split: str = "train"
        """Dataset split to use"""

        use_pseudo: bool = False
        """Whether to use pseudo annotations"""

        coord: str = "null" 
        """Coordinate system to use"""

        def __getitem__(self, key: str) -> Any:
            """Enable dictionary-style access to attributes."""
            return getattr(self, key)

    # Add EgoExo config instance
    egoexo: EgoExoConfig = dataclasses.field(default_factory=EgoExoConfig)
    """EgoExo dataset specific configurations"""

    def __getitem__(self, key: str) -> Any:
        """Enable dictionary-style access to attributes."""
        return getattr(self, key)

    @property
    def egoexo_dataset_path(self) -> Path:
        """Backward compatibility property for egoexo_dataset_path"""
        return self.egoexo.dataset_path
    
    @property
    def bodypose_anno_dir(self) -> tuple[Path, ...]:
        """Backward compatibility property for bodypose_anno_dir"""
        return self.egoexo.bodypose_anno_dir
        
    @property
    def anno_type(self) -> str:
        """Backward compatibility property for anno_type"""
        return self.egoexo.anno_type

    @property
    def split(self) -> str:
        """Backward compatibility property for split"""
        return self.egoexo.split
    
    @property
    def use_pseudo(self) -> bool:
        """Backward compatibility property for use_pseudo"""
        return self.egoexo.use_pseudo
    
    @property
    def coord(self) -> str:
        """Backward compatibility property for coord"""
        return self.egoexo.coord
