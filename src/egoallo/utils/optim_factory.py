# utils/optim_factory.py
from __future__ import annotations
import importlib
import warnings
from typing import Iterable, Mapping, Any


def _optional_import(module: str, cls: str):
    "Return the class or None if the package is absent."
    try:
        return getattr(importlib.import_module(module), cls)
    except (ImportError, AttributeError):
        return None


# -------------------------------------------------------------------------
# 1.  Dynamically resolve every optimiser we know about
# -------------------------------------------------------------------------
_OPTIM_TABLE: Mapping[str, str | tuple[str, str]] = {
    # ---- Native PyTorch --------------------------------------------------
    "sgd": ("torch.optim", "SGD"),
    "adam": ("torch.optim", "Adam"),
    "adamw": ("torch.optim", "AdamW"),
    "radam": ("torch.optim", "RAdam"),  # ≥ PyTorch 2.2
    "rmsprop": ("torch.optim", "RMSprop"),
    "adagrad": ("torch.optim", "Adagrad"),
    "adafactor": ("torch.optim", "Adafactor"),  # ≥ PyTorch 2.3
    # ---- NVIDIA-Apex fused kernels --------------------------------------
    "fusedadam": ("apex.optimizers", "FusedAdam"),
    "fusedlamb": ("apex.optimizers", "FusedLAMB"),
    # ---- torch-optimizer collection -------------------------------------
    "lamb": ("torch_optimizer", "Lamb"),
    "adamw_gc": ("torch_optimizer", "AdamWGC"),
    "ranger": ("torch_optimizer", "Ranger"),
    "ranger21": ("torch_optimizer", "Ranger21"),
    # ---- Stand-alone research repos -------------------------------------
    "lion": ("lion_pytorch", "Lion"),
    "adan": ("adan_pytorch", "Adan"),
    "adabelief": ("adabelief_pytorch", "AdaBelief"),
    "sophia": ("sophia", "SophiaG"),  # Li et al. 2023
}

_LOOKAHEAD = _optional_import("torch_optimizer", "Lookahead") or _optional_import(
    "lookahead_pytorch.lookahead",
    "Lookahead",
)


def get_optimizer(
    name: str,
    params: Iterable,
    lr: float | None = None,
    weight_decay: float = 0.0,
    *,
    fused: bool | None = None,
    lookahead: bool = False,
    foreach: bool | None = None,
    **kwargs: Any,
):
    """
    Generic optimizer factory.

    Parameters
    ----------
    name : str
        Key or alias (case-insensitive) from the table above.
        Prefix 'fused' will be honoured if Apex is present.
    params : iterable
        Model parameters or param-groups.
    lr : float, optional
        Learning rate; default taken from the underlying class.
    fused, foreach : bool | None
        Force Apex fused kernels / torch foreach or use library defaults.
    lookahead : bool
        If True, wrap the base optimizer in a Lookahead outer loop.
    **kwargs
        Forwarded verbatim to the optimiser constructor.

    Returns
    -------
    torch.optim.Optimizer
    """
    key = name.lower()
    if fused and not key.startswith("fused"):
        key = "fused" + key  # e.g. fusedadam

    if key not in _OPTIM_TABLE:
        raise ValueError(f"Unknown optimizer '{name}'. Available: {list(_OPTIM_TABLE)}")

    module, cls_name = _OPTIM_TABLE[key]
    cls = _optional_import(module, cls_name)
    if cls is None:
        raise ImportError(
            f"Requested optimiser '{name}' requires "
            f"`{module}`. Install it or choose another.",
        )

    # Torch built-ins get foreach / fused kwargs for speed (CUDA only)
    if module.startswith("torch.optim"):
        if foreach is not None:
            kwargs.setdefault("foreach", foreach)
        if fused is not None and "fused" in cls.__init__.__code__.co_varnames:
            kwargs.setdefault("fused", fused)

    # Standard constructor signature
    base_opt = cls(params, lr=lr, weight_decay=weight_decay, **kwargs)

    if lookahead:
        if _LOOKAHEAD is None:
            warnings.warn("Lookahead wrapper requested but package missing.")
        else:
            base_opt = _LOOKAHEAD(base_opt)

    return base_opt
