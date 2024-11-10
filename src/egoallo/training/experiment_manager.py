"""Experiment management utilities for training."""
from pathlib import Path
import yaml
from typing import Dict, Any, Optional

from ..utils.git_utils import get_git_commit_hash, get_git_diff

class ExperimentManager:
    """Manages experiment directories, configurations and checkpoints."""
    
    def __init__(self, experiment_name: str, base_dir: Optional[Path] = None):
        self.experiment_name = experiment_name
        self.base_dir = base_dir or Path(__file__).parent.parent.parent / "experiments"
        self.experiment_dir = self._create_experiment_dir()

    def _create_experiment_dir(self, version: int = 0) -> Path:
        """Creates versioned experiment directory."""
        experiment_dir = self.base_dir / self.experiment_name / f"v{version}"
        if experiment_dir.exists():
            return self._create_experiment_dir(version + 1)
        return experiment_dir

    def setup_experiment(self, config: Any) -> None:
        """Initialize experiment directory and save configs."""
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save experiment artifacts
        self._save_git_info()
        self._save_configs(config)

    def _save_git_info(self) -> None:
        """Save git commit hash and diff."""
        (self.experiment_dir / "git_commit.txt").write_text(get_git_commit_hash())
        (self.experiment_dir / "git_diff.txt").write_text(get_git_diff())

    def _save_configs(self, config: Any) -> None:
        """Save experiment configurations."""
        (self.experiment_dir / "run_config.yaml").write_text(yaml.dump(config))
        if hasattr(config, 'model'):
            (self.experiment_dir / "model_config.yaml").write_text(yaml.dump(config.model))

    def get_checkpoint_path(self, step: int) -> Path:
        """Get path for saving model checkpoint."""
        return self.experiment_dir / f"checkpoint_{step}" 