"""Wrapper for the SMPLH body model.

We break down the SMPLH into four stages, each with a corresponding data structure:
- Loading the model itself:
    `model = SmplhModelAADecomp.load(path to npz)`
- Applying a body shape to the model:
    `shaped = model.with_shape(betas)`
- Posing the body shape:
    `posed = shaped.with_pose(root pose, local joint poses)`
- Recovering the mesh with LBS:
    `mesh = posed.lbs()`

NOTE: only support 1 batch axes, not arbitrary batch axes.
"""

import torch
import numpy as np
import typeguard
from pathlib import Path
from jaxtyping import Float, Int
from typing import Self
from egoallo.transforms import SE3, SO3
from torch import Tensor
from egoallo.tensor_dataclass import TensorDataclass

from egoallo.middleware.third_party.HybrIK.hybrik.models.layers.smplh.SMPLH import SMPLH_Layer as SMPLH
import typeguard
from jaxtyping import jaxtyped


@jaxtyped(typechecker=typeguard.typechecked)
class SmplhModelAADecomp(TensorDataclass):
    """SMPLH Wrapper using smplh.SMPLH with original API structure."""

    model: SMPLH
    """The underlying SMPLH model."""

    @classmethod
    def load(cls, model_path: Path, **kwargs) -> "SmplhModelAADecomp":
        gender = kwargs.get("gender", "neutral")
        smplh_model_path = model_path / "smplh" / gender / f"model.npz"
        assert smplh_model_path.exists(), f"SMPLH model not found at {smplh_model_path}"
        model = SMPLH(str(smplh_model_path), **kwargs)
        return cls(model=model)

    @jaxtyped(typechecker=typeguard.typechecked)
    def with_shape(self, betas: Float[Tensor, "*batch num_betas"]) -> "SmplhShapedAADecomp":
        return SmplhShapedAADecomp(
            body_model=self,
            betas=betas,
        )

@jaxtyped(typechecker=typeguard.typechecked)
class SmplhShapedAADecomp(TensorDataclass):
    body_model: SmplhModelAADecomp
    """The underlying body model."""
    betas: Float[Tensor, "*#batch 16"]
    """betas"""

    @jaxtyped(typechecker=typeguard.typechecked)
    def with_pose(
        self,
        T_world_root: Float[Tensor, "*batch 7"],
        local_quats: Float[Tensor, "*batch joints 4"],
    ) -> "SmplhShapedAndPosedAADecomp":
        raise NotImplementedError()

    @jaxtyped(typechecker=typeguard.typechecked)
    def with_pose_decomposed(
        self,
        T_world_root: Float[Tensor, "*batch 7"],
        body_quats: Float[Tensor, "*batch 21 4"],
        left_hand_quats: Float[Tensor, "*batch 15 4"] | None = None,
        right_hand_quats: Float[Tensor, "*batch 15 4"] | None = None,
    ) -> "_SmplhShapedAndPosedAADecomp":
        batch_axes = T_world_root.shape[:-1]
        assert self.betas.shape[:-1] == batch_axes, f"betas shape {self.betas.shape} does not match batch axes {batch_axes}"
        global_orient = SE3(T_world_root).rotation().as_matrix().reshape(batch_axes + (1, 3, 3))
        transl = SE3(T_world_root).translation().reshape(batch_axes + (3,))
        output = self.body_model.model.forward(
            betas=self.betas,
            global_orient=global_orient,
            transl=transl,
            body_pose=SO3(body_quats).as_matrix().reshape(batch_axes + (21, 3, 3)),
            left_hand_pose=SO3(left_hand_quats).as_matrix().reshape(batch_axes + (15, 3, 3)),
            right_hand_pose=SO3(right_hand_quats).as_matrix().reshape(batch_axes + (15, 3, 3)),
        )
        ts_world_joint = output.joints[..., 1:, :]

        # Temporary fix for transl not aligning with output.joints[0]
        T_world_root[..., 4:7] = output.joints[..., 0, :]

        return _SmplhShapedAndPosedAADecomp(
            self,
            T_world_root=T_world_root,
            body_quats=body_quats,
            left_hand_quats=left_hand_quats,
            right_hand_quats=right_hand_quats,
            ts_world_joint=ts_world_joint,
        )

    @jaxtyped(typechecker=typeguard.typechecked)
    def with_pose_decomposed_twist_angles(
        self,
        transl: Float[Tensor, "*batch 3"],
        pose_skeleton: Float[Tensor, "*batch 71 3"],
        phis: Float[Tensor, "*batch 54 2"],
    ) -> "SmplhShapedAndPosedAADecomp":

        return SmplhShapedAndPosedAADecomp(
            self,
            transl=transl,
            pose_skeleton=pose_skeleton,
            phis=phis,
        )

@jaxtyped(typechecker=typeguard.typechecked)
class SmplhShapedAndPosedAADecomp(TensorDataclass):
    """Outputs from the SMPL-H model."""

    shaped_model: SmplhShapedAADecomp
    """Underlying shaped body model."""

    transl: Float[Tensor, "*#batch 3"]
    """Translation."""

    pose_skeleton: Float[Tensor, "*#batch 52 3"]
    """Pose skeleton."""

    phis: Float[Tensor, "*#batch 54 2"]
    """Translation."""

    def lbs(self) -> "SmplhMeshAADecomp":
        output = self.shaped_model.body_model.model.hybrik(
            betas=self.shaped_model.betas,
            pose_skeleton=self.pose_skeleton,
            transl=self.transl,
            phis=self.phis,
        )
        return SmplhMeshAADecomp(self, output.vertices, self.shaped_model.body_model.model.faces_tensor)


@jaxtyped(typechecker=typeguard.typechecked)
class SmplhMeshAADecomp(TensorDataclass):
    """Outputs from the SMPLX model."""

    posed_model: SmplhShapedAndPosedAADecomp
    """Posed model that this mesh was computed for."""

    vertices: Float[Tensor, "*batch verts 3"]
    """Vertices for mesh."""

    faces: Int[Tensor, "faces 3"]
    """Faces for mesh."""

@jaxtyped(typechecker=typeguard.typechecked)
class _SmplhShapedAndPosedAADecomp(TensorDataclass):
    """Outputs from the SMPL-H model."""

    shaped_model: SmplhShapedAADecomp
    """Underlying shaped body model."""

    T_world_root: Float[Tensor, "*#batch 7"]
    """Root transform."""
    ts_world_joint: Float[Tensor, "*#batch 51 3"]

    body_quats: Float[Tensor, "*#batch 21 4"]
    left_hand_quats: Float[Tensor, "*#batch 15 4"]
    right_hand_quats: Float[Tensor, "*#batch 15 4"]

    def lbs(self) -> "SmplhMeshAADecomp":
        raise NotImplementedError
