"""Traj definitions."""

from abc import ABC, abstractmethod
from typing import Generic
from typing import TypeVar

import torch
from torch import nn
from jaxtyping import Bool
from jaxtyping import Float
from torch import Tensor
from egoallo.type_stubs import SmplFamilyModelType

from egoallo.tensor_dataclass import TensorDataclass

T = TypeVar("T", bound="BaseDenoiseTraj")


class BaseDenoiseTraj(TensorDataclass, ABC, Generic[T]):
    """Abstract base class for denoising trajectories."""

    @property
    def loss_weights(self) -> dict[str, float]:
        pass

    @abstractmethod
    def pack(self) -> Float[Tensor, "*batch timesteps d_state"]:
        """Pack trajectory into a single flattened vector."""
        pass

    @classmethod
    @abstractmethod
    def unpack(
        cls,
        x: Float[Tensor, "*batch timesteps d_state"],
        include_hands: bool = False,
        project_rotmats: bool = False,
    ) -> T:
        """Unpack trajectory from a single flattened vector."""
        pass

    def _weight_and_mask_loss(
        self,
        loss_per_step: Float[Tensor, "batch time d"],
        bt_mask: Bool[Tensor, "batch time"],
        weight_t: Float[Tensor, "batch"],
        bt_mask_sum: Float[Tensor, ""] | None = None,
    ) -> Float[Tensor, ""]:
        """Weight and mask per-timestep losses (squared errors)."""
        batch, time, d = loss_per_step.shape
        assert bt_mask.shape == (batch, time)
        assert weight_t.shape == (batch,)

        if bt_mask_sum is None:
            bt_mask_sum = torch.sum(bt_mask)

        return (
            torch.sum(
                torch.sum(
                    torch.mean(loss_per_step, dim=-1) * bt_mask,
                    dim=-1,
                )
                * weight_t,
            )
            / bt_mask_sum
        )

    @abstractmethod
    def compute_loss(
        self,
        other: T,
        mask: Bool[Tensor, "batch time"],
        weight_t: Float[Tensor, "batch"],
    ) -> dict[str, Float[Tensor, ""]]:
        """Compute loss between this trajectory and another."""
        pass

    @abstractmethod
    def apply_to_body(self, body_model: "SmplFamilyModelType") -> "SmplFamilyModelType":
        """Apply the trajectory data to a SMPL-H body model."""
        pass

    @abstractmethod
    def encode(
        self,
        encoders: nn.ModuleDict,
        batch: int,
        time: int,
    ) -> Float[Tensor, "batch time d_latent"]:
        """Encode trajectory into latent representation.

        Args:
            encoders: Dictionary of encoder networksF
            batch: Batch size
            time: Sequence length

        Returns:
            Encoded representation of shape (batch, time, d_latent)
        """
        pass

    @classmethod
    @abstractmethod
    def get_modality_dict(cls, include_hands: bool = False) -> dict[str, int]:
        """Get dictionary of modalities and their dimensions for this trajectory type.

        Args:
            include_hands: Whether to include hand modalities in the dictionary

        Returns:
            Dictionary mapping modality names to their dimensions
        """
        pass
