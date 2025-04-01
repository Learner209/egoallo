import torch
from typing import Any, Callable, List, Tuple, TypeVar

T = TypeVar("T")
U = TypeVar("U")


class TensorDataclassBatchPlugin:
    """
    A plugin for TensorDataclass to handle batch dimension manipulation.
    Supports flattening arbitrary batch dimensions to a single batch dimension,
    and unflattening back to the original batch dimensions.
    """

    @staticmethod
    def flatten_batch_dims(
        tensor: torch.Tensor,
        batch_dims: List[int],
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Flatten the batch dimensions of a tensor.

        Args:
            tensor: The tensor to flatten.
            batch_dims: List of batch dimensions to use.

        Returns:
            Tuple of (flattened tensor, original batch dimensions)
        """
        non_batch_dims = tensor.shape[len(batch_dims) :]

        # Calculate the total batch size
        total_batch_size = 1
        for dim in batch_dims:
            total_batch_size *= dim

        # Reshape the tensor to flatten the batch dimensions
        flattened = tensor.reshape(total_batch_size, *non_batch_dims)

        return flattened, batch_dims

    @staticmethod
    def unflatten_batch_dims(
        tensor: torch.Tensor,
        original_batch_dims: List[int],
    ) -> torch.Tensor:
        """
        Unflatten the batch dimensions of a tensor.

        Args:
            tensor: The tensor to unflatten.
            original_batch_dims: The original batch dimensions to reshape to.

        Returns:
            Unflattened tensor with original batch dimensions.
        """
        non_batch_dims = tensor.shape[
            1:
        ]  # All dimensions except the first one (flattened batch dim)

        # Reshape the tensor to the original batch dimensions
        unflattened = tensor.reshape(*original_batch_dims, *non_batch_dims)

        return unflattened

    @staticmethod
    def with_flattened_batch_dims(
        fn: Callable[[Any], T],
        obj: Any,
        batch_dims: List[int],
    ) -> Tuple[T, List[int]]:
        """
        Apply a function to an object with flattened batch dimensions.

        Args:
            fn: The function to apply.
            obj: The object to apply the function to.
            batch_dims: List of batch dimensions to use.

        Returns:
            Tuple of (function result, original batch dimensions)
        """
        # Create a flattened version of the object
        flattened_obj = TensorDataclassBatchPlugin.flatten_obj(obj, batch_dims)

        # Apply the function to the flattened object
        result = fn(flattened_obj)

        return result, batch_dims

    @staticmethod
    def flatten_obj(obj: Any, batch_dims: List[int]) -> Any:
        """
        Flatten the batch dimensions of all tensors in an object.

        Args:
            obj: The object to flatten.
            batch_dims: The batch dimensions to flatten.

        Returns:
            Object with flattened tensors.
        """
        if isinstance(obj, torch.Tensor):
            if (
                obj.dim() >= len(batch_dims) + 1
            ):  # Only flatten if there are enough dimensions
                tensor_batch_dims = list(obj.shape[: len(batch_dims)])
                # Only flatten if batch dims match or can be broadcast
                if all(t in (b, 1) for t, b in zip(tensor_batch_dims, batch_dims)):
                    # Broadcast to the target batch dimensions if needed
                    if tensor_batch_dims != batch_dims:
                        # Create a new list for broadcasting
                        broadcast_shape = list(batch_dims) + list(
                            obj.shape[len(batch_dims) :],
                        )
                        obj = obj.expand(broadcast_shape)
                    flattened, _ = TensorDataclassBatchPlugin.flatten_batch_dims(
                        obj,
                        batch_dims,
                    )
                    return flattened
            return obj

        elif isinstance(obj, dict):
            return {
                k: TensorDataclassBatchPlugin.flatten_obj(v, batch_dims)
                for k, v in obj.items()
            }

        elif isinstance(obj, (list, tuple)):
            return type(obj)(
                TensorDataclassBatchPlugin.flatten_obj(item, batch_dims) for item in obj
            )

        elif hasattr(obj, "__dict__"):
            # For dataclasses or objects with __dict__
            flattened_vars = {
                k: TensorDataclassBatchPlugin.flatten_obj(v, batch_dims)
                for k, v in vars(obj).items()
            }

            # Create a new instance with the flattened attributes
            if hasattr(obj, "__class__"):
                try:
                    # Try to create a new instance with the same class
                    new_obj = obj.__class__.__new__(obj.__class__)
                    new_obj.__dict__.update(flattened_vars)
                    return new_obj
                except Exception:
                    pass

            # Fallback if creating a new instance fails
            return flattened_vars

        return obj

    @staticmethod
    def unflatten_obj(obj: Any, result: Any, original_batch_dims: List[int]) -> Any:
        """
        Unflatten the batch dimensions of all tensors in a result object.

        Args:
            obj: The original object (used for structure reference).
            result: The result object to unflatten.
            original_batch_dims: The original batch dimensions to reshape to.

        Returns:
            Object with unflattened tensors.
        """
        if not original_batch_dims:  # No batch dimensions to unflatten
            return result

        if isinstance(result, torch.Tensor) and result.dim() > 0:
            if isinstance(obj, torch.Tensor) and obj.dim() > len(original_batch_dims):
                # Only unflatten tensors that would have had batch dimensions
                return TensorDataclassBatchPlugin.unflatten_batch_dims(
                    result,
                    original_batch_dims,
                )
            return result

        elif isinstance(result, dict):
            return {
                k: TensorDataclassBatchPlugin.unflatten_obj(
                    obj.get(k) if isinstance(obj, dict) else obj,
                    v,
                    original_batch_dims,
                )
                for k, v in result.items()
            }

        elif isinstance(result, (list, tuple)):
            obj_items = obj if isinstance(obj, (list, tuple)) else [obj] * len(result)
            return type(result)(
                TensorDataclassBatchPlugin.unflatten_obj(o, r, original_batch_dims)
                for o, r in zip(obj_items, result)
            )

        elif hasattr(result, "__dict__"):
            # For dataclasses or objects with __dict__
            original_obj = obj if hasattr(obj, "__dict__") else {}
            unflattened_vars = {
                k: TensorDataclassBatchPlugin.unflatten_obj(
                    getattr(original_obj, k, None),
                    v,
                    original_batch_dims,
                )
                for k, v in vars(result).items()
            }

            # Create a new instance with the unflattened attributes
            if hasattr(result, "__class__"):
                try:
                    # Try to create a new instance with the same class
                    new_obj = result.__class__.__new__(result.__class__)
                    new_obj.__dict__.update(unflattened_vars)
                    return new_obj
                except Exception:
                    pass

            # Fallback if creating a new instance fails
            return unflattened_vars

        return result

    @staticmethod
    def apply_with_flattened_batch(
        fn: Callable[[Any], Any],
        obj: Any,
        batch_dims: List[int],
    ) -> Any:
        """
        Apply a function to an object with flattened batch dimensions, then unflatten the result.

        Args:
            fn: The function to apply.
            obj: The object to apply the function to.

        Returns:
            Unflattened function result.
        """
        # Flatten the object
        flattened_obj = TensorDataclassBatchPlugin.flatten_obj(obj, batch_dims)

        # Apply the function to the flattened object
        result = fn(flattened_obj)

        # Unflatten the result
        unflattened_result = TensorDataclassBatchPlugin.unflatten_obj(
            obj,
            result,
            batch_dims,
        )

        return unflattened_result
