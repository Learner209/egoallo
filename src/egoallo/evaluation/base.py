from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Optional

import torch
import typeguard
from egoallo import fncsmpl
from egoallo.types import Device
from egoallo.types import MetricsDict
from egoallo.types import PathLike
from egoallo.types import ProcrustesMode
from egoallo.types import ProcrustesOutput
from jaxtyping import Float
from jaxtyping import jaxtyped
from torch import Tensor

from .utils import ensure_path
from .utils import get_device


class BaseEvaluator(ABC):
    """Base class for pose evaluators."""

    def __init__(self, body_model_path: PathLike, device: Optional[Device] = None):
        """
        Initialize the evaluator.

        Args:
            body_model_path: Path to the body model file
            device: Torch device to use
        """
        self.device = get_device(device)
        self.body_model_path = ensure_path(body_model_path)
        self.body_model: fncsmpl.SmplhModel = self._load_body_model(
            self.body_model_path,
        )

    @abstractmethod
    def _load_body_model(self, model_path: Path) -> fncsmpl.SmplhModel:
        """Load the body model from file."""
        pass

    @classmethod
    @abstractmethod
    @jaxtyped(typechecker=typeguard.typechecked)
    def procrustes_align(
        cls,
        points_y: Float[Tensor, "*batch time 3"],
        points_x: Float[Tensor, "*batch time 3"],
        output: ProcrustesMode,
        fix_scale: bool = False,
        device: torch.device = torch.device("cpu"),
    ) -> ProcrustesOutput:
        """
        Perform Procrustes alignment between point sets.

        Args:
            points_y: Target points
            points_x: Source points
            output: Output mode ('transforms' or 'aligned_x')
            fix_scale: Whether to fix scale to 1

        Returns:
            Either transformation parameters or aligned points
        """
        pass

    @abstractmethod
    def evaluate_directory(self, *args, **kwargs) -> None:
        """Evaluate all files in a directory."""
        pass

    @abstractmethod
    def process_file(self, *args, **kwargs) -> MetricsDict:
        """Process a single file and compute metrics."""
        pass
