from __future__ import annotations

from abc import ABC, abstractmethod
from random import random, randint
from typing import Tuple, Dict, Type

import torch
from torch import Tensor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from egoallo.config.train.train_config import EgoAlloTrainConfig  # type: ignore


from egoallo.data.datasets.amass_dataset import AdaptiveAmassHdf5Dataset  # type: ignore

"""Flexible masking extension for AMASS HDF5 datasets.

This module introduces plug‑and‑play masking strategies that can be mixed and
matched at runtime, while preserving full backward‑compatibility with the
original ``AdaptiveAmassHdf5Dataset`` API.  The key ideas are:

* **Strategy pattern** – spatial and temporal masking logic is encapsulated in
  independent, easily swappable classes that share a minimal interface.
* **Non‑intrusive inheritance** – the new ``FlexibleMaskingDataset`` re‑uses all
  loading / augmentation code from ``AdaptiveAmassHdf5Dataset`` and only
  overrides the mask‑generation section; nothing else in your pipeline needs to
  change (optimisers, dataloaders, etc.).
* **Declarative configuration** – strategies are chosen via ``cfg.mask.*``
  strings or callables; adding a new scheme never touches this file.
"""
# ---------------------------------------------------------------------------
# Strategy interfaces
# ---------------------------------------------------------------------------


class SpatialMaskStrategy(ABC):
    """Mask generator that zeroes‑out joints across the entire time dimension."""

    @abstractmethod
    def __call__(
        self,
        *,
        num_joints: int,
        seq_len: int,
        device: torch.device,
    ) -> Tensor:  # noqa: D401,E501
        """Return a **boolean** tensor of shape ``(seq_len, num_joints)`` where
        ``False`` means *masked* and ``True`` means *visible*."""


class TemporalMaskStrategy(ABC):
    """Mask generator that zeroes‑out full frames or temporal patches."""

    @abstractmethod
    def __call__(self, *, seq_len: int, device: torch.device) -> Tensor:  # noqa: D401
        """Return a **boolean** tensor of shape ``(seq_len,)`` where ``False``
        means that the whole frame is masked for **all** joints."""


# ---------------------------------------------------------------------------
# Concrete spatial strategies
# ---------------------------------------------------------------------------


class RandomJointSpatialMask(SpatialMaskStrategy):
    """Original per‑joint random masking (unchanged behaviour)."""

    def __init__(self, mask_ratio: float):
        self.mask_ratio = float(mask_ratio)

    def __call__(
        self,
        *,
        num_joints: int,
        seq_len: int,
        device: torch.device,
    ) -> Tensor:  # noqa: D401,E501
        num_mask = int(num_joints * self.mask_ratio)
        idx = torch.randperm(num_joints, device=device)[:num_mask]
        mask = torch.ones((seq_len, num_joints), dtype=torch.bool, device=device)
        mask[:, idx] = False
        return mask


# SMPL‑(H/X) canonical 24‑joint split.  Adjust if you use a custom ordering.
_UPPER = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]  # 0 = pelvis/root
_LOWER = [1, 2, 4, 5, 7, 8, 10, 11]  # 0 = pelvis/root

# Additional joint definitions
_HEAD = [15]  # Head joint index
_HANDS = [20, 21]  # Left and right hand indices
_FEET = [7, 8, 10, 11]  # Left and right foot/ankle indices


class UpperLowerSpatialMask(SpatialMaskStrategy):
    """Masks **either** the full upper or full lower body on every call."""

    def __init__(self, selection: str | None = None):
        """If *selection* is ``"upper"`` or ``"lower"`` the choice is fixed.
        Pass ``None`` (default) for a random 50/50 decision each sample."""
        assert selection in (None, "upper", "lower")
        self.selection = selection

    def __call__(
        self,
        *,
        num_joints: int,
        seq_len: int,
        device: torch.device,
    ) -> Tensor:  # noqa: D401,E501
        part = self.selection or ("upper" if random() < 0.5 else "lower")
        idx = _UPPER if part == "upper" else _LOWER
        mask = torch.ones((seq_len, num_joints), dtype=torch.bool, device=device)
        mask[:, idx] = False
        return mask


class HeadOnlySpatialMask(SpatialMaskStrategy):
    """Masks all joints except for the head."""

    def __call__(
        self,
        *,
        num_joints: int,
        seq_len: int,
        device: torch.device,
    ) -> Tensor:
        mask = torch.zeros(
            (seq_len, num_joints),
            dtype=torch.bool,
            device=device,
        )  # Start with all masked
        mask[:, _HEAD] = True  # Unmask head joint
        return mask


class HeadHandsSpatialMask(SpatialMaskStrategy):
    """Masks all joints except for the head and hands."""

    def __call__(
        self,
        *,
        num_joints: int,
        seq_len: int,
        device: torch.device,
    ) -> Tensor:
        mask = torch.zeros((seq_len, num_joints), dtype=torch.bool, device=device)
        visible_joints = _HEAD + _HANDS  # Combine head and hands indices
        mask[:, visible_joints] = True
        return mask


class ExtremitiesSpatialMask(SpatialMaskStrategy):
    """Masks all joints except for head, hands, and feet."""

    def __call__(
        self,
        *,
        num_joints: int,
        seq_len: int,
        device: torch.device,
    ) -> Tensor:
        mask = torch.zeros((seq_len, num_joints), dtype=torch.bool, device=device)
        visible_joints = _HEAD + _HANDS + _FEET  # Combine all extremity indices
        mask[:, visible_joints] = True
        return mask


# ---------------------------------------------------------------------------
# Concrete temporal strategies
# ---------------------------------------------------------------------------


class PatchTemporalMask(TemporalMaskStrategy):
    """Original patch‑based masking (unchanged behaviour)."""

    def __init__(self, ratio: float, patch_size: int):
        self.ratio, self.patch_size = float(ratio), int(patch_size)

    def __call__(self, *, seq_len: int, device: torch.device) -> Tensor:  # noqa: D401
        pad = (self.patch_size - seq_len % self.patch_size) % self.patch_size
        total = seq_len + pad
        patches = total // self.patch_size
        mask = torch.ones((patches, self.patch_size), dtype=torch.bool, device=device)
        k = int(patches * self.ratio)
        # never mask patch 0 — preserves first frame information
        mask_idx = torch.randperm(patches - 1, device=device)[:k] + 1
        mask[mask_idx] = False
        return mask.view(-1)[:seq_len]


class WindowTemporalMask(TemporalMaskStrategy):
    """Masks **one contiguous window** of configurable length (default 10 %)."""

    def __init__(self, ratio: float = 0.10):
        assert 0 < ratio < 1, "ratio must be a fraction (e.g. 0.1 for 10 %)"
        self.ratio = float(ratio)

    def __call__(self, *, seq_len: int, device: torch.device) -> Tensor:  # noqa: D401
        win = max(1, int(round(seq_len * self.ratio)))
        start = randint(1, max(1, seq_len - win - 1))  # avoid masking very first/last
        mask = torch.ones((seq_len,), dtype=torch.bool, device=device)
        mask[start : start + win] = False
        return mask


# ---------------------------------------------------------------------------
# Strategy registry – add new schemes without touching the dataset class!
# ---------------------------------------------------------------------------

_SPATIAL_REGISTRY: Dict[str, Type[SpatialMaskStrategy]] = {
    "random_joint": RandomJointSpatialMask,
    "upper_lower": UpperLowerSpatialMask,
    "head_only": HeadOnlySpatialMask,
    "head_hands": HeadHandsSpatialMask,
    "extremities": ExtremitiesSpatialMask,
}
_TEMPORAL_REGISTRY: Dict[str, Type[TemporalMaskStrategy]] = {
    "patch": PatchTemporalMask,
    "window": WindowTemporalMask,
}

# ---------------------------------------------------------------------------
# Drop‑in replacement dataset
# ---------------------------------------------------------------------------


class FlexibleMaskingDataset(AdaptiveAmassHdf5Dataset):
    """Adaptive dataset **with** pluggable spatial/temporal masking strategies.

    Usage (no code changes needed downstream):
    >>> cfg.dataset_cls = "egoallo.datasets.flexible_masking_dataset.FlexibleMaskingDataset"  # noqa: E501
    >>> cfg.mask.spatial.name = "upper_lower"    # or "random_joint"
    >>> cfg.mask.spatial.kwargs = {"selection": None}  # optional
    >>> cfg.mask.temporal.name = "window"        # or "patch"
    >>> cfg.mask.temporal.kwargs = {"ratio": 0.15}
    """

    def __init__(self, config: "EgoAlloTrainConfig"):
        super().__init__(config)

        # ------------------------------------------------------------------
        # Build strategies from *cfg.mask.* namespaces.  Example YAML:
        # mask:
        #   spatial: {name: upper_lower, kwargs: {}}
        #   temporal: {name: window, kwargs: {ratio: 0.1}}
        # ------------------------------------------------------------------
        s_cfg = getattr(config, "mask_scheme", {}).get("spatial", {})  # type: ignore[attr-defined]
        t_cfg = getattr(config, "mask_scheme", {}).get("temporal", {})  # type: ignore[attr-defined]

        self.spatial_strategy: SpatialMaskStrategy = _SPATIAL_REGISTRY[
            s_cfg.get("name", "random_joint")
        ](**s_cfg.get("kwargs", {}))
        self.temporal_strategy: TemporalMaskStrategy = _TEMPORAL_REGISTRY[
            t_cfg.get("name", "patch")
        ](**t_cfg.get("kwargs", {}))

    # ------------------------------------------------------------------
    # The only method we need to override is ``_make_masks``; everything
    # else (loading, preprocessing, fps‑aug, etc.) remains untouched.
    # ------------------------------------------------------------------

    def _make_masks(
        self,
        *,
        seq_len: int,
        num_joints: int,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns (visible_joints_mask, temporal_mask)."""
        temporal_mask = self.temporal_strategy(seq_len=seq_len, device=device)  # (T,)
        spatial_mask = self.spatial_strategy(
            num_joints=num_joints,
            seq_len=seq_len,
            device=device,
        )  # (T, J)
        # Combine: a joint is visible iff both masks are *True*
        visible = spatial_mask & temporal_mask.unsqueeze(-1)
        return visible, temporal_mask, spatial_mask


# ---------------------------------------------------------------------------
# Helper function to register custom strategies at runtime (one‑liner in user
# code; avoids modifying this file when you invent a novel scheme).
# ---------------------------------------------------------------------------


def register_mask_strategy(
    *,
    kind: str,
    name: str,
    cls: Type[SpatialMaskStrategy] | Type[TemporalMaskStrategy],
):
    assert kind in ("spatial", "temporal"), "kind must be 'spatial' or 'temporal'"
    reg = _SPATIAL_REGISTRY if kind == "spatial" else _TEMPORAL_REGISTRY
    if name in reg:
        raise ValueError(f"Mask strategy '{name}' already registered.")
    reg[name] = cls
