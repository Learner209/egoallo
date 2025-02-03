from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from collections import defaultdict


@dataclass
class KeypointFilterStats:
    """Track keypoint filtering statistics for a single frame"""

    total: int = 0
    missing_annotation: int = 0
    biomechanical_invalid: int = 0
    visibility_invalid: int = 0
    projection_error: int = 0
    final_valid: int = 0
    valid_keypoint_indices: List[int] = field(default_factory=list)

    def update_from(self, other: "KeypointFilterStats") -> None:
        """Update statistics from another KeypointFilterStats object"""
        self.total += other.total
        self.missing_annotation += other.missing_annotation
        self.biomechanical_invalid += other.biomechanical_invalid
        self.visibility_invalid += other.visibility_invalid
        self.projection_error += other.projection_error
        self.final_valid += other.final_valid
        self.valid_keypoint_indices.extend(other.valid_keypoint_indices)

    def get_percentages(self) -> Dict[str, float]:
        """Calculate percentage statistics relative to total"""
        if self.total == 0:
            return {}

        return {
            "missing_annotation": (self.missing_annotation / self.total) * 100,
            "biomechanical_invalid": (self.biomechanical_invalid / self.total) * 100,
            "visibility_invalid": (self.visibility_invalid / self.total) * 100,
            "projection_error": (self.projection_error / self.total) * 100,
            "final_valid": (self.final_valid / self.total) * 100,
        }

    def get_valid_keypoint_distribution(self) -> Dict[int, int]:
        """Get distribution of valid keypoint indices"""
        if not self.valid_keypoint_indices:
            return {}

        distribution = defaultdict(int)
        for idx in self.valid_keypoint_indices:
            distribution[idx] += 1
        return dict(distribution)

    def __getstate__(self):
        """Return state for pickling"""
        return self.__dict__

    def __setstate__(self, state):
        """Set state when unpickling"""
        self.__dict__.update(state)


@dataclass
class FrameStats:
    """Statistics for a single frame"""

    keypoint_stats: KeypointFilterStats = field(default_factory=KeypointFilterStats)
    valid: bool = False
    camera_validations: Dict[str, bool] = field(default_factory=dict)


@dataclass
class TakeStats:
    """Statistics for a complete take"""

    frames: Dict[str, FrameStats] = field(
        default_factory=lambda: defaultdict(FrameStats)
    )
    valid: bool = False

    def get_aggregate_stats(self) -> KeypointFilterStats:
        """Aggregate keypoint statistics across all frames"""
        agg_stats = KeypointFilterStats()
        for frame_stats in self.frames.values():
            agg_stats.update_from(frame_stats.keypoint_stats)
        return agg_stats

    def count_valid_frames(self) -> int:
        """Count the number of valid frames in this take"""
        return sum(1 for frame in self.frames.values() if frame.valid)


class PreprocessingStatsCollector:
    """
    Collects and analyzes preprocessing statistics across takes and frames.
    Provides hierarchical organization of statistics at take and frame levels.
    """

    def __init__(self) -> None:
        self.takes: Dict[str, TakeStats] = defaultdict(TakeStats)

    def mark_take_processed(self, take_id: str, valid: bool) -> None:
        """
        Mark a take as processed with its validity status

        Args:
            take_id: Unique identifier for the take
            valid: Whether the take is valid
        """
        self.takes[take_id].valid = valid

    def update_frame_stats(
        self,
        take_id: str,
        frame_id: str,
        keypoint_stats: KeypointFilterStats,
        valid: bool = False,
        camera_validations: Optional[Dict[str, bool]] = None,
    ) -> None:
        """
        Update statistics for a specific frame within a take

        Args:
            take_id: Unique identifier for the take
            frame_id: Unique identifier for the frame
            keypoint_stats: Keypoint filtering statistics for this frame
            valid: Whether the frame is valid
            camera_validations: Optional camera validation results
        """
        frame_stats = FrameStats(
            keypoint_stats=keypoint_stats,
            valid=valid,
            camera_validations=camera_validations or {},
        )
        self.takes[take_id].frames[frame_id] = frame_stats

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Generate comprehensive summary statistics across all takes

        Returns:
            Dictionary containing aggregated statistics including:
            - Total keypoint statistics
            - Keypoint percentages
            - Take and frame validity counts
            - Valid keypoint distribution
        """
        total_stats = KeypointFilterStats()
        valid_takes = 0
        valid_frames = 0
        total_frames = 0

        # Aggregate statistics across all takes
        for take_stats in self.takes.values():
            # Update keypoint statistics
            take_kpt_stats = take_stats.get_aggregate_stats()
            total_stats.update_from(take_kpt_stats)

            # Update take and frame counts
            if take_stats.valid:
                valid_takes += 1
            valid_frames += take_stats.count_valid_frames()
            total_frames += len(take_stats.frames)

        return {
            "keypoint_stats": total_stats,
            "keypoint_percentages": total_stats.get_percentages(),
            "keypoint_distribution": total_stats.get_valid_keypoint_distribution(),
            "valid_takes": valid_takes,
            "total_takes": len(self.takes),
            "valid_frames": valid_frames,
            "total_frames": total_frames,
        }
