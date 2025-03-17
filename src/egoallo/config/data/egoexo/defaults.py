from __future__ import annotations

import dataclasses
from pathlib import Path

from typing import Any


@dataclasses.dataclass
class EgoExoConfig:
    """Configuration specific to EgoExo dataset."""

    traj_root: Path = Path("./datasets/egoexo-default/takes/cmu_bike_0")
    """Path to trajectory root"""

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

    coord: str = "global"
    """Coordinate system to use"""

    gt_ground_height_anno_dir: Path = Path(
        "./exp/egoexo-default-exp/egoexo/egoexo-default-gt-output/bodypose/canonical/time_12_22_18_09_59/logs/annotation/manual",
    )
    """Ground height for ground truth"""

    batch_size: int = 1
    """batch-size"""

    def __getitem__(self, key: str) -> Any:
        """Enable dictionary-style access to attributes."""
        return getattr(self, key)
