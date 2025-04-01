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
from egoallo.tensor_dataclass_batch_plugins import TensorDataclassBatchPlugin

from egoallo.middleware.third_party.HybrIK.hybrik.models.layers.smpl.SMPL import SMPL_layer as SMPL
import typeguard
from jaxtyping import jaxtyped


@jaxtyped(typechecker=typeguard.typechecked)
class SmplModelAADecomp(TensorDataclass):
    """SMPLH Wrapper using smplh.SMPLH with original API structure."""

    model: SMPL
    """The underlying SMPLH model."""

    @classmethod
    def load(cls, model_path: Path, **kwargs) -> "SmplModelAADecomp":
        gender = kwargs.get("gender", "neutral")
        smpl_model_path = model_path / "smpl" / "SMPL_python_v.1.0.0" / "smpl" / "models" / f"basicModel_{gender}_lbs_10_207_0_v1.0.0.pkl"
        assert smpl_model_path.exists()

        h36m_jregressor_path = model_path / "smpl" / "J_regressor_h36m.npy"
        assert h36m_jregressor_path.exists()

        model = SMPL(smpl_model_path, h36m_jregressor=np.load(h36m_jregressor_path), **kwargs)
        return cls(model=model)

    @jaxtyped(typechecker=typeguard.typechecked)
    def with_shape(self, betas: Float[Tensor, "*batch num_betas"]) -> "SmplShapedAADecomp":
        return SmplShapedAADecomp(
            body_model=self,
            betas=betas,
        )

@jaxtyped(typechecker=typeguard.typechecked)
class SmplShapedAADecomp(TensorDataclass):
    body_model: SmplModelAADecomp
    """The underlying body model."""
    betas: Float[Tensor, "*batch 10"]
    """betas"""

    @jaxtyped(typechecker=typeguard.typechecked)
    def with_pose(
        self,
        T_world_root: Float[Tensor, "*batch 7"],
        local_quats: Float[Tensor, "*batch joints 4"],
    ) -> "SmplShapedAndPosedAADecomp":
        raise NotImplementedError()

    @jaxtyped(typechecker=typeguard.typechecked)
    def with_pose_decomposed(
        self,
        T_world_root: Float[Tensor, "*batch 7"],
        body_quats: Float[Tensor, "*batch 23 4"],
    ) -> "_SmplShapedAndPosedAADecomp":
        batch_axes = T_world_root.shape[:-1]
        global_orient = SE3(T_world_root).rotation().as_matrix().reshape(batch_axes + (1, 3, 3))
        transl = SE3(T_world_root).translation().reshape(batch_axes + (3,))

        flattened_global_orient, _ = TensorDataclassBatchPlugin.flatten_batch_dims(global_orient, batch_axes)
        flattened_transl, _ = TensorDataclassBatchPlugin.flatten_batch_dims(transl, batch_axes)
        flattened_aa, _ = TensorDataclassBatchPlugin.flatten_batch_dims(SO3(body_quats).log().reshape(batch_axes + (23 * 3,)), batch_axes)
        flattened_betas, _ = TensorDataclassBatchPlugin.flatten_batch_dims(self.betas, batch_axes)

        output = self.body_model.model.forward(
            betas=flattened_betas,
            global_orient=flattened_global_orient,
            transl=flattened_transl,
            pose_axis_angle=flattened_aa,
        )

        ts_world_joint = output.joints[..., 1:, :]

        unflattened_ts_world_joint = TensorDataclassBatchPlugin.unflatten_batch_dims(ts_world_joint, batch_axes)

        return _SmplShapedAndPosedAADecomp(
            self,
            T_world_root=T_world_root,
            ts_world_joint=unflattened_ts_world_joint,
        )

    @jaxtyped(typechecker=typeguard.typechecked)
    def with_pose_decomposed_twist_angles(
        self,
        global_orient: Float[Tensor, "*batch 3"] | None,
        transl: Float[Tensor, "*batch 3"],
        pose_skeleton: Float[Tensor, "*batch 24 3"] | Float[Tensor, "*batch 29 3"],
        phis: Float[Tensor, "*batch 23 2"],
    ) -> "SmplShapedAndPosedAADecomp":

        return SmplShapedAndPosedAADecomp(
            self,
            transl=transl,
            global_orient=global_orient,
            pose_skeleton=pose_skeleton,
            phis=phis,
        )

@jaxtyped(typechecker=typeguard.typechecked)
class SmplShapedAndPosedAADecomp(TensorDataclass):
    """Outputs from the SMPL-H model."""

    shaped_model: SmplShapedAADecomp
    """Underlying shaped body model."""

    transl: Float[Tensor, "*#batch 3"]
    """Translation."""

    global_orient: Float[Tensor, "*#batch 3"] | None
    """Global orientation."""

    pose_skeleton: Float[Tensor, "*#batch 24 3"] | Float[Tensor, "*#batch 29 3"]
    """Pose skeleton."""

    phis: Float[Tensor, "*#batch 23 2"]
    """Translation."""

    @property
    def rot_mats(self) -> Float[Tensor, "*#batch 24 3 3"] | Float[Tensor, "*#batch 29 3 3"]:
        """
        should support arbitrary batch dimensions, however, hybrik only supports one leading dim.
        """
        batch_dims = self.transl.shape[:-1]
        flattened_obj = TensorDataclassBatchPlugin.flatten_obj(self, batch_dims)
        output = self.shaped_model.body_model.model.hybrik(
            betas=flattened_obj.shaped_model.betas,
            global_orient=SO3.exp(flattened_obj.global_orient).as_matrix().reshape(flattened_obj.global_orient.shape[:-1] + (3, 3)) if flattened_obj.global_orient is not None else None,
            pose_skeleton=flattened_obj.pose_skeleton,
            transl=flattened_obj.transl,
            phis=flattened_obj.phis,
        )
        rot_mats = TensorDataclassBatchPlugin.unflatten_batch_dims(output.rot_mats, batch_dims)
        return rot_mats

    def lbs(self) -> "SmplMeshAADecomp":
        """
        should support arbitrary batch dimensions, however, hybrik only supports one leading dim.
        """
        batch_dims = self.transl.shape[:-1]
        flattened_obj = TensorDataclassBatchPlugin.flatten_obj(self, batch_dims)
        output = self.shaped_model.body_model.model.hybrik(
            betas=flattened_obj.shaped_model.betas,
            global_orient=SO3.exp(flattened_obj.global_orient).as_matrix().reshape(flattened_obj.global_orient.shape[:-1] + (3, 3)) if flattened_obj.global_orient is not None else None,
            pose_skeleton=flattened_obj.pose_skeleton,
            transl=flattened_obj.transl,
            phis=flattened_obj.phis,
        )
        vertices = TensorDataclassBatchPlugin.unflatten_batch_dims(output.vertices, batch_dims)
        rot_mats = TensorDataclassBatchPlugin.unflatten_batch_dims(output.rot_mats, batch_dims)
        return SmplMeshAADecomp(
            self,
            vertices=vertices,
            faces=self.shaped_model.body_model.model.faces_tensor,
            rot_mats=rot_mats,
        )


@jaxtyped(typechecker=typeguard.typechecked)
class SmplMeshAADecomp(TensorDataclass):
    """Outputs from the SMPLX model."""

    posed_model: SmplShapedAndPosedAADecomp
    """Posed model that this mesh was computed for."""

    rot_mats: Float[Tensor, "*batch 24 3 3"]
    """Rotation matrices for all joints"""

    vertices: Float[Tensor, "*batch verts 3"]
    """Vertices for mesh."""

    faces: Int[Tensor, "faces 3"]
    """Faces for mesh."""

@jaxtyped(typechecker=typeguard.typechecked)
class _SmplShapedAndPosedAADecomp(TensorDataclass):
    """Outputs from the SMPL-H model."""

    shaped_model: SmplShapedAADecomp
    """Underlying shaped body model."""

    T_world_root: Float[Tensor, "*#batch 7"]
    """Root transform."""
    ts_world_joint: Float[Tensor, "*#batch 23 3"]

    body_quats: Float[Tensor, "*#batch 23 4"]

    def lbs(self) -> "SmplMeshAADecomp":
        raise NotImplementedError
