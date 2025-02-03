from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
import yaml
from jaxtyping import jaxtyped
import typeguard
import logging


@jaxtyped(typechecker=typeguard.typechecked)
@dataclass
class EgoAlloEvaluationMetrics:
    """Container for EgoAllo evaluation metrics.

    Contains both per-sequence metrics and summary statistics.
    All arrays are runtime type-checked using typeguard and jaxtyping.
    """

    # File paths for saved data
    metrics_file: Optional[Path] = None
    summary_file: Optional[Path] = None

    def __init__(self, **metrics):
        """Initialize with arbitrary metrics.

        Args:
            **metrics: Arbitrary keyword arguments containing metric arrays
        """
        # Set file paths
        self.metrics_file = None
        self.summary_file = None

        # Store all provided metrics
        for name, value in metrics.items():
            setattr(self, name, value)

        # Track metric names for iteration
        self._metric_names = [
            name
            for name in metrics.keys()
            if name not in ["metrics_file", "summary_file"]
        ]

    @property
    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        metrics_dict = {}
        for field in self._metric_names:
            values = getattr(self, field)
            if values is not None:
                metrics_dict[field] = {
                    "average_sample_mean": float(np.nanmean(values)),
                    "stddev_sample": float(np.nanstd(values)),
                    "stderr_sample": float(
                        np.nanstd(values) / np.sqrt(np.count_nonzero(~np.isnan(values)))
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
            {field: getattr(self, field) for field in self._metric_names},
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

    def print_metrics(
        self, logger=None, level: str = "info", verbose: bool = False
    ) -> None:
        """Print metrics information using the provided logger.

        Args:
            logger: Optional logger object. If None, uses print()
            level: Logging level ('debug', 'info', 'warning', etc.). Defaults to 'info'
            verbose: If True, prints raw metrics in addition to summary. Defaults to False
        """
        if logger is None:
            logger = logging.getLogger(__name__)
            if not logger.handlers:
                logger.addHandler(logging.StreamHandler())
                logger.setLevel(logging.INFO)

        log_func = getattr(logger, level.lower())

        # Print summary statistics
        log_func("=== EgoAllo Evaluation Metrics Summary ===")
        summary_data = self.summary
        for metric_name, stats in summary_data.items():
            log_func(f"\n{metric_name}:")
            for stat_name, value in stats.items():
                log_func(f"  {stat_name}: {value:.4f}")

        # Print raw metrics if verbose
        if verbose:
            log_func("\n=== Raw Metrics ===")
            for field in self._metric_names:
                values = getattr(self, field)
                if values is not None:
                    log_func(f"\n{field}:")
                    log_func(f"  shape: {values.shape}")
                    log_func(f"  min: {np.nanmin(values):.4f}")
                    log_func(f"  max: {np.nanmax(values):.4f}")
