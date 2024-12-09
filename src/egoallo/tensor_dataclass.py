import dataclasses
from typing import Any, Callable, Self, dataclass_transform

import torch


@dataclass_transform()
class TensorDataclass:
    """A lighter version of nerfstudio's TensorDataclass:
    https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/utils/tensor_dataclass.py
    """

    def __init_subclass__(cls) -> None:
        dataclasses.dataclass(cls)

    def to(self, device: torch.device | str) -> Self:
        """Move the tensors in the dataclass to the given device.

        Args:
            device: The device to move to.

        Returns:
            A new dataclass.
        """
        return self.map(lambda x: x.to(device))
    def as_nested_dict(self, numpy: bool) -> dict[str, Any]:
        """Convert the dataclass to a nested dictionary.

        Recurses into lists, tuples, and dictionaries.
        """

        def _to_dict(obj: Any) -> Any:
            if isinstance(obj, TensorDataclass):
                return {k: _to_dict(v) for k, v in vars(obj).items()}
            elif isinstance(obj, (list, tuple)):
                return type(obj)(_to_dict(v) for v in obj)
            elif isinstance(obj, dict):
                return {k: _to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, torch.Tensor) and numpy:
                return obj.numpy(force=True)
            else:
                return obj

        return _to_dict(self)

    def map(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> Self:
        """Apply a function to all tensors in the dataclass.

        Also recurses into lists, tuples, and dictionaries.

        Args:
            fn: The function to apply to each tensor.

        Returns:
            A new dataclass.
        """

        def _map_impl[MapT](
            fn: Callable[[torch.Tensor], torch.Tensor],
            val: MapT,
        ) -> MapT:
            if isinstance(val, torch.Tensor):
                return fn(val)
            elif isinstance(val, TensorDataclass):
                return type(val)(**_map_impl(fn, vars(val)))
            elif isinstance(val, (list, tuple)):
                return type(val)(_map_impl(fn, v) for v in val)
            elif isinstance(val, dict):
                assert type(val) is dict  # No subclass support.
                return {k: _map_impl(fn, v) for k, v in val.items()}  # type: ignore
            else:
                return val

        return _map_impl(fn, self)

    def check_shapes(self, other: 'TensorDataclass') -> bool:
        """Check if two TensorDataclass instances have the same shape across all attributes.

        Args:
            other: The other TensorDataclass instance to compare with.

        Returns:
            True if all corresponding attributes have the same shape, False otherwise.
        """

        def _check_shapes_impl(val1: Any, val2: Any) -> bool:
            if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
                return val1.shape == val2.shape
            elif isinstance(val1, TensorDataclass) and isinstance(val2, TensorDataclass):
                return all(_check_shapes_impl(v1, v2) for v1, v2 in zip(vars(val1).values(), vars(val2).values()))
            elif isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
                return all(_check_shapes_impl(v1, v2) for v1, v2 in zip(val1, val2))
            elif isinstance(val1, dict) and isinstance(val2, dict):
                return all(_check_shapes_impl(val1[k], val2[k]) for k in val1.keys() & val2.keys())
            else:
                return True  # Non-tensor attributes are considered to have the same shape

        return _check_shapes_impl(self, other)
