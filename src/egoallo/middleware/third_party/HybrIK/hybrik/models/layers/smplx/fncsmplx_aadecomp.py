"""Wrapper for the SMPLX body model.

We break down the SMPLX into four stages, each with a corresponding data structure:
- Loading the model itself:
    `model = SmplxModelAADecomp.load(path to npz)`
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
from egoallo.middleware.third_party.HybrIK.hybrik.models.layers.smplx.body_models import SMPLXLayer as SMPLX
import typeguard
from jaxtyping import jaxtyped


@jaxtyped(typechecker=typeguard.typechecked)
class SmplxModelAADecomp(TensorDataclass):
    """SMPLX Wrapper using smplx.SMPLX with original API structure."""

    model: SMPLX
    """The underlying SMPLX model."""

    @classmethod
    def load(cls, model_path: Path, kid_template_path: Path | None = None, **kwargs) -> "SmplxModelAADecomp":

        gender = kwargs.get("gender", "neutral")
        smplx_model_path = model_path / "smplx" / "smplx_v_1_1" / f"SMPLX_{gender.upper()}.npz"
        assert smplx_model_path.exists()

        if kid_template_path:
            model = SMPLX(smplx_model_path, age='kid', kid_template_path=kid_template_path, **kwargs)
        else:
            model = SMPLX(smplx_model_path, age='kid', kid_template_path=Path(smplx_model_path).parent.parent / 'smplx_kid_template.npy', **kwargs)
        return cls(model=model)

    @jaxtyped(typechecker=typeguard.typechecked)
    def with_shape(self, betas: Float[Tensor, "*batch num_betas"]) -> "SmplxShapedAADecomp":
        return SmplxShapedAADecomp(
            body_model=self,
            betas=betas,
        )

@jaxtyped(typechecker=typeguard.typechecked)
class SmplxShapedAADecomp(TensorDataclass):
    body_model: SmplxModelAADecomp
    """The underlying body model."""
    betas: Float[Tensor, "*#batch 11"]
    """betas"""

    @jaxtyped(typechecker=typeguard.typechecked)
    def with_pose(
        self,
        T_world_root: Float[Tensor, "*batch 7"],
        local_quats: Float[Tensor, "*batch joints 4"],
    ) -> "SmplxShapedAndPosedAADecomp":
        raise NotImplementedError()

    @jaxtyped(typechecker=typeguard.typechecked)
    def with_pose_decomposed(
        self,
        T_world_root: Float[Tensor, "*batch 7"],
        body_quats: Float[Tensor, "*batch 21 4"],
        left_hand_quats: Float[Tensor, "*batch 15 4"] | None = None,
        right_hand_quats: Float[Tensor, "*batch 15 4"] | None = None,
        expression: Float[Tensor, "*batch 10"] | None = None,
        leye_pose: Float[Tensor, "*batch 1 3"] | None = None,
        reye_pose: Float[Tensor, "*batch 1 3"] | None = None,
    ) -> "_SmplxShapedAndPosedAADecomp":
        batch_axes = T_world_root.shape[:-1]
        global_orient = SE3(T_world_root).rotation().as_matrix().reshape(batch_axes + (1, 3, 3))
        transl = SE3(T_world_root).translation().reshape(batch_axes + (3,))
        output = self.body_model.model.forward_simple_with_pose_decomposed(
            betas=self.betas,
            global_orient=global_orient,
            transl=transl,
            body_pose=SO3(body_quats).as_matrix().reshape(batch_axes + (21, 3, 3)),
            left_hand_pose=SO3(left_hand_quats).as_matrix().reshape(batch_axes + (15, 3, 3)),
            right_hand_pose=SO3(right_hand_quats).as_matrix().reshape(batch_axes + (15, 3, 3)),
            expression=expression,
            leye_pose=SO3.exp(leye_pose).as_matrix().reshape(batch_axes + (1, 3, 3)),
            reye_pose=SO3.exp(reye_pose).as_matrix().reshape(batch_axes + (1, 3, 3)),
        )
        ts_world_joint = output.joints[..., 1:, :]
        return _SmplxShapedAndPosedAADecomp(
            self,
            T_world_root=T_world_root,
            body_quats=body_quats,
            left_hand_quats=left_hand_quats,
            right_hand_quats=right_hand_quats,
            expression=expression,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            ts_world_joint=ts_world_joint,
        )

    @jaxtyped(typechecker=typeguard.typechecked)
    def with_pose_decomposed_twist_angles(
        self,
        transl: Float[Tensor, "*batch 3"],
        pose_skeleton: Float[Tensor, "*batch 71 3"],
        phis: Float[Tensor, "*batch 54 2"],
        expression: Float[Tensor, "*batch 10"] | None = None,
    ) -> "SmplxShapedAndPosedAADecomp":

        return SmplxShapedAndPosedAADecomp(
            self,
            transl=transl,
            expression=expression,
            pose_skeleton=pose_skeleton,
            phis=phis,
        )

@jaxtyped(typechecker=typeguard.typechecked)
class SmplxShapedAndPosedAADecomp(TensorDataclass):
    """Outputs from the SMPL-H model."""

    shaped_model: SmplxShapedAADecomp
    """Underlying shaped body model."""

    transl: Float[Tensor, "*#batch 3"]
    """Translation."""

    expression: Float[Tensor, "*#batch 10"] | None
    """Expression."""

    pose_skeleton: Float[Tensor, "*#batch 71 3"]
    """Pose skeleton."""

    phis: Float[Tensor, "*#batch 54 2"]
    """Translation."""

    def lbs(self) -> "SmplxMeshAADecomp":
        output = self.shaped_model.body_model.model.hybrik(
            betas=self.shaped_model.betas[..., :11],
            pose_skeleton=self.pose_skeleton,
            transl=self.transl,
            phis=self.phis,
            expression=self.expression,
        )
        return SmplxMeshAADecomp(self, output.vertices, self.shaped_model.body_model.model.faces_tensor)


@jaxtyped(typechecker=typeguard.typechecked)
class SmplxMeshAADecomp(TensorDataclass):
    """Outputs from the SMPLX model."""

    posed_model: SmplxShapedAndPosedAADecomp
    """Posed model that this mesh was computed for."""

    vertices: Float[Tensor, "*batch verts 3"]
    """Vertices for mesh."""

    faces: Int[Tensor, "faces 3"]
    """Faces for mesh."""

@jaxtyped(typechecker=typeguard.typechecked)
class _SmplxShapedAndPosedAADecomp(TensorDataclass):
    """Outputs from the SMPL-H model."""

    shaped_model: SmplxShapedAADecomp
    """Underlying shaped body model."""

    T_world_root: Float[Tensor, "*#batch 7"]
    """Root transform."""
    ts_world_joint: Float[Tensor, "*#batch 126 3"]

    body_quats: Float[Tensor, "*#batch 21 4"]
    left_hand_quats: Float[Tensor, "*#batch 15 4"]
    right_hand_quats: Float[Tensor, "*#batch 15 4"]

    leye_pose: Float[Tensor, "*#batch 1 3"]
    reye_pose: Float[Tensor, "*#batch 1 3"]

    expression: Float[Tensor, "*#batch 10"]

    def lbs(self) -> "SmplxMeshAADecomp":
        raise NotImplementedError
