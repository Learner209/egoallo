import collections

import torch
from torch.utils.data._utils.collate import default_collate_err_msg_format
from torch.utils.data._utils.collate import np_str_obj_array_pattern


def extended_collate[T](batch: list[T]) -> T:
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        if batch[0].ndim == 0 or all(a.shape[0] == batch[0].shape[0] for a in batch):
            return torch.stack(batch, 0, out=out)
        else:
            return batch
    elif elem_type.__module__ == "numpy" and elem_type.__name__ not in (
        "str_",
        "string_",
    ):
        if elem_type.__name__ in ("ndarray", "memmap"):
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return extended_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float32)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: extended_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(extended_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [extended_collate(samples) for samples in transposed]
    raise TypeError(default_collate_err_msg_format.format(elem_type))


class ExtendedBatchCollator(object):
    def __call__(self, batch):
        return extended_collate(batch)
