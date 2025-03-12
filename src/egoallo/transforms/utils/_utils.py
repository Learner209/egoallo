from typing import Callable
from typing import Type
from typing import TypeVar

import torch

from .._base import MatrixLieGroup
# assert TYPE_CHECKING
# if TYPE_CHECKING:


T = TypeVar("T", bound=MatrixLieGroup)


def get_epsilon(dtype: torch.dtype) -> float:
    """Helper for grabbing type-specific precision constants.

    Args:
        dtype: Datatype.

    Returns:
        Output float.
    """
    return {
        torch.float32: 1e-5,
        torch.float32: 1e-10,
    }[dtype]


def register_lie_group(
    *,
    matrix_dim: int,
    parameters_dim: int,
    tangent_dim: int,
    space_dim: int,
) -> Callable[[Type[T]], Type[T]]:
    """Decorator for registering Lie group dataclasses.

    Sets dimensionality class variables.
    """

    def _wrap(cls: Type[T]) -> Type[T]:
        # Register dimensions as class attributes.
        cls.matrix_dim = matrix_dim
        cls.parameters_dim = parameters_dim
        cls.tangent_dim = tangent_dim
        cls.space_dim = space_dim

        return cls

    return _wrap
