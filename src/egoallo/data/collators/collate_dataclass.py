import numpy as np
import torch


def collate_dataclass[T](batch: list[T]) -> T:
    """Collate function that works for dataclasses."""
    keys = vars(batch[0]).keys()
    return type(batch[0])(
        **{k: torch.stack([getattr(b, k) for b in batch]) for k in keys},
    )


def collate_tensor_only_dataclass[T](batch: list[T]) -> T:
    """Collate function that only stacks tensor attributes in dataclasses.

    This is a more flexible version that:
    1. Only stacks torch.Tensor and np.ndarray attributes
    2. Ignores non-tensor attributes (preserves first item's value)
    3. Handles None values and optional fields
    4. Supports nested dataclasses

    Args:
        batch: List of dataclass instances to collate

    Returns:
        Collated dataclass with stacked tensor attributes

    Example:
        >>> @dataclass
        >>> class Data:
        >>>     x: torch.Tensor
        >>>     y: Optional[torch.Tensor] = None
        >>>     z: str = "test"
        >>> batch = [Data(x=torch.ones(3), y=None, z="a"),
        >>>         Data(x=torch.zeros(3), y=torch.ones(2), z="b")]
        >>> result = collate_tensor_only_dataclass(batch)
        >>> # result.x: tensor([[1,1,1], [0,0,0]])
        >>> # result.y: None
        >>> # result.z: "a"
    """
    if not batch:
        raise ValueError("Empty batch")

    # Get first item as reference
    first = batch[0]
    if not hasattr(first, "__dataclass_fields__"):
        raise TypeError("Expected dataclass instance")

    # Initialize output dict
    collated = {}

    # Get all field names from first item
    fields = vars(first).keys()

    for key in fields:
        # Get values for this field from all items
        values = [getattr(item, key) for item in batch]

        # Handle first non-None value as reference
        ref_val = next((v for v in values if v is not None), None)

        if ref_val is None:
            # If all values are None, keep as None
            collated[key] = None

        elif isinstance(ref_val, (torch.Tensor, np.ndarray)):
            # Stack tensors/arrays, filtering out None values
            valid_values = [v for v in values if v is not None]
            if valid_values:
                if isinstance(ref_val, torch.Tensor):
                    collated[key] = torch.stack(valid_values)
                else:
                    collated[key] = np.stack(valid_values)
            else:
                collated[key] = None

        elif hasattr(ref_val, "__dataclass_fields__"):
            # Recursively handle nested dataclasses
            valid_values = [v for v in values if v is not None]
            if valid_values:
                collated[key] = collate_tensor_only_dataclass(valid_values)
            else:
                collated[key] = None

        elif key == "take_name":
            collated[key] = tuple(values)
        else:
            # For non-tensor fields, keep first item's value
            collated[key] = values[0]

    return type(first)(**collated)
