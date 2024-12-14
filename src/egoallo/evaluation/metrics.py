from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
import yaml
from jaxtyping import Float, Bool, jaxtyped
import typeguard


@jaxtyped(typechecker=typeguard.typechecked)
@dataclass
class EgoAlloEvaluationMetrics:
    """Container for EgoAllo evaluation metrics.

    Contains both per-sequence metrics and summary statistics.
    All arrays are runtime type-checked using typeguard and jaxtyping.
    """

    # Per-sequence metrics with shape annotations
    mpjpe: Float[np.ndarray, "N"]  # Mean per-joint position error
    pampjpe: Float[np.ndarray, "N"]  # PA-MPJPE
    head_ori: Float[np.ndarray, "N"]  # Head orientation error
    head_trans: Float[np.ndarray, "N"]  # Head translation error
    foot_skate: Float[np.ndarray, "N"]  # Foot skating metric
    foot_contact: Float[np.ndarray, "N"]  # Foot contact ratio
    coco_mpjpe: Optional[Float[np.ndarray, "N"]] = None  # Optional COCO joint error

    # File paths for saved data
    metrics_file: Optional[Path] = None
    summary_file: Optional[Path] = None

    @property
    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        metrics_dict = {}
        for field in [
            "mpjpe",
            "pampjpe",
            "head_ori",
            "head_trans",
            "foot_skate",
            "foot_contact",
        ]:
            values = getattr(self, field)
            if values is not None:
                metrics_dict[field] = {
                    "average_sample_mean": float(np.nanmean(values)),
                    "stddev_sample": float(np.nanstd(values)),
                    "stderr_sample": float(
                        np.nanstd(values) / np.sqrt(np.count_nonzero(~np.isnan(values)))
                    ),
                }

        if self.coco_mpjpe is not None:
            metrics_dict["coco_mpjpe"] = {
                "average_sample_mean": float(np.nanmean(self.coco_mpjpe)),
                "stddev_sample": float(np.nanstd(self.coco_mpjpe)),
                "stderr_sample": float(
                    np.nanstd(self.coco_mpjpe)
                    / np.sqrt(np.count_nonzero(~np.isnan(self.coco_mpjpe)))
                ),
            }
        return metrics_dict

    def save(self, output_dir: Path, suffix: str = "") -> None:
        """Save metrics to files."""
        # Save raw metrics
        self.metrics_file = (
            output_dir / f"_eval_cached_disaggregated_metrics{suffix}.pt"
        )
        torch.save(
            {
                field: getattr(self, field)
                for field in self.__dataclass_fields__
                if field not in ["metrics_file", "summary_file"]
            },
            self.metrics_file,
        )

        # Save summary
        self.summary_file = output_dir / f"_eval_cached_summary{suffix}.yaml"
        with open(self.summary_file, "w") as f:
            yaml.dump(self.summary, f)

    @classmethod
    def load(cls, metrics_file: Path) -> "EgoAlloEvaluationMetrics":
        """Load metrics from files."""
        data = torch.load(metrics_file)
        summary_file = metrics_file.parent / metrics_file.name.replace(
            "disaggregated_metrics", "summary"
        ).replace(".pt", ".yaml")

        return cls(
            **data,
            metrics_file=metrics_file,
            summary_file=summary_file if summary_file.exists() else None,
        )

