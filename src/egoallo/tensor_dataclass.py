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

    def reduce[T](self, fn: Callable[[T, T], T]) -> T:
        """Reduce all tensors in the dataclass to a single value.

        Also recurses into lists, tuples, and dictionaries.

        Args:
            fn: The reduction function to apply between tensors.
                Should take two values and return a single value.

        Returns:
            The reduced value.
        """

        def _reduce_impl(
            fn: Callable[[T, T], T],
            val: Any,
        ) -> torch.Tensor | None:
            if isinstance(val, torch.Tensor):
                return val
            elif isinstance(val, TensorDataclass):
                tensors = [_reduce_impl(fn, v) for v in vars(val).values()]
                tensors = [t for t in tensors if t is not None]
                if not tensors:
                    return None
                result = tensors[0]
                for t in tensors[1:]:
                    result = fn(result, t)
                return result
            elif isinstance(val, (list, tuple)):
                tensors = [_reduce_impl(fn, v) for v in val]
                tensors = [t for t in tensors if t is not None]
                if not tensors:
                    return None
                result = tensors[0]
                for t in tensors[1:]:
                    result = fn(result, t)
                return result
            elif isinstance(val, dict):
                assert type(val) is dict  # No subclass support.
                tensors = [_reduce_impl(fn, v) for v in val.values()]
                tensors = [t for t in tensors if t is not None]
                if not tensors:
                    return None
                result = tensors[0]
                for t in tensors[1:]:
                    result = fn(result, t)
                return result
            else:
                return None

        result = _reduce_impl(fn, self)
        if result is None:
            raise ValueError("No tensors found in dataclass")
        return result

    def _dict_map(self, fn: Callable[[str, torch.Tensor], torch.Tensor]) -> Self:
        """Apply a function to all tensors in the dataclass.

        Also recurses into lists, tuples, and dictionaries.

        Args:
            fn: The function to apply to each tensor. Takes attribute name and tensor value.

        Returns:
            A new dataclass.
        """

        def _map_impl[MapT](
            fn: Callable[[str, torch.Tensor], torch.Tensor],
            val: MapT,
            name: str = "",
        ) -> MapT:
            if isinstance(val, torch.Tensor):
                return fn(name, val)
            elif isinstance(val, TensorDataclass):
                # raise NotImplementedError("Not implemented for TensorDataclass")
                # TODO: only implement for the first level of recursion.
                return type(val)(**{
                    k: _map_impl(fn, v, f"{name}" if name else k)
                    for k, v in vars(val).items()
                })
            elif isinstance(val, (list, tuple)):
                raise NotImplementedError("Not implemented for list or tuple")
                return type(val)(_map_impl(fn, v, f"{name}[{i}]")
                               for i, v in enumerate(val))
            elif isinstance(val, dict):
                raise NotImplementedError("Not implemented for dict")
                assert type(val) is dict  # No subclass support.
                return {k: _map_impl(fn, v, f"{name}[{k}]")
                       for k, v in val.items()}  # type: ignore
            else:
                return val

        return _map_impl(fn, self)

    def check_shapes(self, other: "TensorDataclass") -> bool:
        """Check if two TensorDataclass instances have the same shape across all attributes.

        Args:
            other: The other TensorDataclass instance to compare with.

        Returns:
            True if all corresponding attributes have the same shape, False otherwise.
        """

        def _check_shapes_impl(val1: Any, val2: Any) -> bool:
            if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
                return val1.shape == val2.shape
            elif isinstance(val1, TensorDataclass) and isinstance(
                val2, TensorDataclass
            ):
                return all(
                    _check_shapes_impl(v1, v2)
                    for v1, v2 in zip(vars(val1).values(), vars(val2).values())
                )
            elif isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
                return all(_check_shapes_impl(v1, v2) for v1, v2 in zip(val1, val2))
            elif isinstance(val1, dict) and isinstance(val2, dict):
                return all(
                    _check_shapes_impl(val1[k], val2[k])
                    for k in val1.keys() & val2.keys()
                )
            else:
                return (
                    True  # Non-tensor attributes are considered to have the same shape
                )

        return _check_shapes_impl(self, other)

    def get_batch_size(self) -> int | None:
        """Get the batch dimension (first dimension) that is consistent across all tensors.

        Returns:
            int | None: The batch size if tensors exist, None if no tensors found
        """
        # FIXME: the current implementation ignores the scenario where the preceding batch_size dimensions would be null, be aware of this bug.
        batch_size = None

        def _get_batch_size_impl(val: Any) -> int | None:
            nonlocal batch_size

            if isinstance(val, torch.Tensor):
                if len(val.shape) > 0:  # Skip 0-dim tensors
                    if batch_size is None:
                        batch_size = val.shape[0]
                    elif batch_size != val.shape[0]:
                        raise ValueError(
                            f"Inconsistent batch sizes found: {batch_size} vs {val.shape[0]}"
                        )
                    return val.shape[0]
            elif isinstance(val, TensorDataclass):
                for v in vars(val).values():
                    _get_batch_size_impl(v)
            elif isinstance(val, (list, tuple)):
                for v in val:
                    _get_batch_size_impl(v)
            elif isinstance(val, dict):
                for v in val.values():
                    _get_batch_size_impl(v)
            return None

        _get_batch_size_impl(self)
        return batch_size

    def __getitem__(self, index) -> Self:
        """Implements native Python slicing for TensorDataclass.

        Supports numpy/torch-style indexing including:
        - Single index: data[0]
        - Multiple indices: data[0,1]
        - Slices: data[0:10]
        - Mixed indexing: data[0, :10, 2:4]
        - Ellipsis: data[..., 0]

        Args:
            index: Index specification. Can be int, slice, tuple, or ellipsis.

        Returns:
            A new TensorDataclass with sliced data.

        Examples:
            >>> data = TensorDataclass(...)
            >>> # Single index
            >>> first_item = data[0]
            >>> # Multiple indices
            >>> specific_item = data[0, 10]
            >>> # Slice
            >>> first_ten = data[:10]
            >>> # Mixed indexing
            >>> subset = data[0, :10, 2:4]
        """
        # Convert single index to tuple for uniform handling
        if not isinstance(index, tuple):
            index = (index,)

        def _getitem_impl[GetItemT](val: GetItemT, idx: tuple) -> GetItemT:
            if isinstance(val, torch.Tensor):
                try:
                    return val[idx]
                except IndexError as e:
                    raise IndexError(
                        f"Invalid index {idx} for tensor of shape {val.shape}"
                    ) from e
            elif isinstance(val, TensorDataclass):
                return type(val)(**_getitem_impl(vars(val), idx))
            elif isinstance(val, (list, tuple)):
                return type(val)(_getitem_impl(v, idx) for v in val)
            elif isinstance(val, dict):
                assert type(val) is dict  # No subclass support
                return {k: _getitem_impl(v, idx) for k, v in val.items()}  # type: ignore
            else:
                return val

        return _getitem_impl(self, index)

    def __setitem__(self, index, value) -> None:
        """Implements setting values using native Python slicing.

        Args:
            index: Index specification (same as __getitem__)
            value: Value to set. Must be compatible with the dataclass structure.

        Raises:
            ValueError: If value structure doesn't match the dataclass
            TypeError: If value types are incompatible
        """
        if not isinstance(index, tuple):
            index = (index,)

        def _setitem_impl(target: Any, idx: tuple, val: Any) -> None:
            if isinstance(target, torch.Tensor):
                if not isinstance(val, torch.Tensor):
                    raise TypeError(f"Cannot set tensor with value of type {type(val)}")
                try:
                    target[idx] = val
                except IndexError as e:
                    raise IndexError(
                        f"Invalid index {idx} for tensor of shape {target.shape}"
                    ) from e
            elif isinstance(target, TensorDataclass):
                if not isinstance(val, TensorDataclass):
                    raise TypeError(
                        f"Cannot set TensorDataclass with value of type {type(val)}"
                    )
                for k, v in vars(target).items():
                    _setitem_impl(v, idx, getattr(val, k))
            elif isinstance(target, (list, tuple)):
                if not isinstance(val, (list, tuple)):
                    raise TypeError(
                        f"Cannot set {type(target)} with value of type {type(val)}"
                    )
                for t, v in zip(target, val):
                    _setitem_impl(t, idx, v)
            elif isinstance(target, dict):
                if not isinstance(val, dict):
                    raise TypeError(f"Cannot set dict with value of type {type(val)}")
                for k in target:
                    if k in val:
                        _setitem_impl(target[k], idx, val[k])

        _setitem_impl(self, index, value)
