"""Wrapper for the SMPL-H body model.

We break down the SMPL-H into four stages, each with a corresponding data structure:
- Loading the model itself:
    `model = SmplhModel.load(path to npz)`
- Applying a body shape to the model:
    `shaped = model.with_shape(betas)`
- Posing the body shape:
    `posed = shaped.with_pose(root pose, local joint poses)`
- Recovering the mesh with LBS:
    `mesh = posed.lbs()`

NOTE: only support 1 batch axes, not arbitrary batch axes.
"""

import torch
import smplx
import numpy as np
from pathlib import Path
from jaxtyping import Float, Int
from .transforms import SE3, SO3
from torch import Tensor
from .tensor_dataclass import TensorDataclass


# @jaxtyped(typechecker=typeguard.typechecked)
class SmplhModel(TensorDataclass):
    """SMPL-H Wrapper using smplx.SMPLH with original API structure."""

    model: smplx.SMPLH
    """The underlying SMPL-H model."""

    faces: Int[Tensor, "faces 3"]
    """Vertex indices for mesh faces."""
    J_regressor: Float[Tensor, "joints_plus_1 verts"]
    """Linear map from vertex to joint positions.
    For SMPL-H, 1 root + 21 body joints + 2 * 15 hand joints."""
    parent_indices: tuple[int, ...]
    """Defines kinematic tree. Index of 0 signifies that a joint is defined
    relative to the root."""
    weights: Float[Tensor, "verts joints_plus_1 "]
    """LBS weights."""
    posedirs: Float[Tensor, "verts 3 joints_times_9"]
    """Pose blend shape bases."""
    v_template: Float[Tensor, "verts 3"]
    """Canonical mesh verts."""
    shapedirs: Float[Tensor, "verts 3 n_betas"]
    """Shape bases."""
    hands_components_l: Float[Tensor, "num_pca 45"] | None = None
    """Left hand PCA components. Optional."""
    hands_components_r: Float[Tensor, "num_pca 45"] | None = None
    """Right hand PCA components. Optional."""

    @classmethod
    def load(cls, model_path: Path, **kwargs) -> "SmplhModel":
        model = smplx.SMPLH(model_path, **kwargs)
        parent_indices = tuple(model.parents[1:].tolist())  # Exclude root
        faces = torch.from_numpy(model.faces.astype(np.int32))
        return cls(
            faces=faces,
            J_regressor=model.J_regressor,
            parent_indices=parent_indices,
            weights=model.lbs_weights,
            posedirs=model.posedirs,
            v_template=model.v_template,
            shapedirs=model.shapedirs,
            hands_components_l=None,
            hands_components_r=None,
            model=model,
        )

    def get_num_joints(self) -> int:
        return len(self.parent_indices)

    # @jaxtyped(typechecker=typeguard.typechecked)
    def with_shape(self, betas: Float[Tensor, "*batch num_betas"]) -> "SmplhShaped":
        batch_axes = betas.shape[:-1]
        output = self.model(
            betas=betas.reshape(-1, betas.shape[-1]),
            return_verts=True,
        )

        root_offset = output.joints[..., 0, :]
        verts_zero = output.vertices - root_offset.unsqueeze(1)
        joints_zero = output.joints[
            ...,
            1 : self.get_num_joints() + 1,
            :,
        ] - root_offset.unsqueeze(1)
        t_parent_joint = (
            joints_zero
            - joints_zero[
                ...,
                torch.tensor(self.parent_indices).to(
                    joints_zero.device,
                ),
                :,
            ]
        )

        root_offset = root_offset.reshape(
            batch_axes + (root_offset.shape[-1],),
        )
        verts_zero = verts_zero.reshape(
            batch_axes + verts_zero.shape[-2:],
        )
        joints_zero = joints_zero.reshape(
            batch_axes + joints_zero.shape[-2:],
        )
        t_parent_joint = t_parent_joint.reshape(
            batch_axes + t_parent_joint.shape[-2:],
        )

        return SmplhShaped(
            body_model=self,
            root_offset=root_offset,
            verts_zero=verts_zero,
            joints_zero=joints_zero,
            t_parent_joint=t_parent_joint,
            betas=betas,
        )


# @jaxtyped(typechecker=typeguard.typechecked)
class SmplhShaped(TensorDataclass):
    body_model: SmplhModel
    """The underlying body model."""
    root_offset: Float[Tensor, "*#batch 3"]
    verts_zero: Float[Tensor, "*#batch verts 3"]
    """Vertices of shaped body _relative to the root joint_ at the zero
    configuration."""
    joints_zero: Float[Tensor, "*#batch joints 3"]
    """Joints of shaped body _relative to the root joint_ at the zero
    configuration."""
    t_parent_joint: Float[Tensor, "*#batch joints 3"]
    """Position of each shaped body joint relative to its parent. Does not
    include root."""
    betas: Float[Tensor, "*#batch 16"]
    """betas"""

    # @jaxtyped(typechecker=typeguard.typechecked)
    def with_pose(
        self,
        T_world_root: Float[Tensor, "*batch 7"],
        local_quats: Float[Tensor, "*batch joints 4"],
    ) -> "SmplhShapedAndPosed":
        batch_dim = local_quats.shape[:-2]

        root_se3 = SE3(T_world_root)
        translation = root_se3.translation()
        global_orient = root_se3.rotation().log().view(*batch_dim, -1)

        body_pose = SO3(local_quats[..., :21, :]).log().view(*batch_dim, -1)
        left_hand_pose = (
            SO3(local_quats[..., 21 : 21 + 15, :]).log().view(*batch_dim, -1)
        )
        right_hand_pose = SO3(local_quats[..., 21 + 15 :, :]).log().view(*batch_dim, -1)

        Ts_world_joint = forward_kinematics(
            T_world_root=T_world_root,
            Rs_parent_joint=local_quats,
            t_parent_joint=self.t_parent_joint,
            parent_indices=self.body_model.parent_indices,
        )
        return SmplhShapedAndPosed(
            self,
            T_world_root=T_world_root,
            local_quats=local_quats,
            Ts_world_joint=Ts_world_joint,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            translation=translation,
        )

    # @jaxtyped(typechecker=typeguard.typechecked)
    def with_pose_decomposed(
        self,
        T_world_root: Float[Tensor, "*batch 7"],
        body_quats: Float[Tensor, "*batch 21 4"],
        left_hand_quats: Float[Tensor, "*batch 15 4"] | None = None,
        right_hand_quats: Float[Tensor, "*batch 15 4"] | None = None,
    ) -> "SmplhShapedAndPosed":
        num_joints = self.body_model.get_num_joints()
        batch_axes = body_quats.shape[:-2]
        if left_hand_quats is None:
            left_hand_quats = body_quats.new_zeros((*batch_axes, 15, 4))
            left_hand_quats[..., 0] = 1.0
        if right_hand_quats is None:
            right_hand_quats = body_quats.new_zeros((*batch_axes, 15, 4))
            right_hand_quats[..., 0] = 1.0
        local_quats = broadcasting_cat(
            [body_quats, left_hand_quats, right_hand_quats],
            dim=-2,
        )
        assert local_quats.shape[-2:] == (num_joints, 4)
        return self.with_pose(T_world_root, local_quats)


# @jaxtyped(typechecker=typeguard.typechecked)
class SmplhShapedAndPosed(TensorDataclass):
    """Outputs from the SMPL-H model."""

    shaped_model: SmplhShaped
    """Underlying shaped body model."""

    T_world_root: Float[Tensor, "*#batch 7"]
    """Root coordinate frame."""

    local_quats: Float[Tensor, "*#batch joints 4"]
    """Local joint orientations."""

    Ts_world_joint: Float[Tensor, "*#batch joints 7"]
    """Absolute transform for each joint. Does not include the root."""

    global_orient: Float[Tensor, "*#batch 3"]
    """Global orientation of the body."""

    body_pose: Float[Tensor, "*#batch (joints - 1) 3"]
    """Local body pose."""

    left_hand_pose: Float[Tensor, "*#batch 15 3"]
    """Local left hand pose."""

    right_hand_pose: Float[Tensor, "*#batch 15 3"]
    """Local right hand pose."""

    translation: Float[Tensor, "*#batch 3"]
    """Translation of the body."""

    def with_new_T_world_root(
        self,
        T_world_root: Float[Tensor, "*batch 7"],
    ) -> "SmplhShapedAndPosed":
        raise NotImplementedError()
        # return SmplhShapedAndPosed(self.shaped, T_world_root, self.local_quats)

    def lbs(self) -> "SmplMesh":
        output = self.shaped_model.body_model.model(
            betas=self.shaped_model.betas,
            global_orient=self.global_orient,
            body_pose=self.body_pose,
            left_hand_pose=self.left_hand_pose,
            right_hand_pose=self.right_hand_pose,
            transl=self.translation,
            return_verts=True,
        )
        return SmplMesh(self, output.vertices, self.shaped_model.body_model.faces)


# @jaxtyped(typechecker=typeguard.typechecked)
class SmplMesh(TensorDataclass):
    """Outputs from the SMPL-H model."""

    posed_model: SmplhShapedAndPosed
    """Posed model that this mesh was computed for."""

    verts: Float[Tensor, "*batch verts 3"]
    """Vertices for mesh."""

    faces: Int[Tensor, "faces 3"]
    """Faces for mesh."""


# @jaxtyped(typechecker=typeguard.typechecked)
def forward_kinematics(
    T_world_root: Float[Tensor, "*#batch 7"],
    Rs_parent_joint: Float[Tensor, "*#batch joints 4"],
    t_parent_joint: Float[Tensor, "*#batch joints 3"],
    parent_indices: tuple[int, ...],
) -> Float[Tensor, "*#batch joints 7"]:
    """Run forward kinematics to compute absolute poses (T_world_joint) for
    each joint. The output array containts pose parameters
    (w, x, y, z, tx, ty, tz) for each joint. (this does not include the root!)

    Args:
        T_world_root: Transformation to world frame from root frame.
        Rs_parent_joint: Local orientation of each joint.
        t_parent_joint: Position of each joint with respect to its parent frame. (this does not
            depend on local joint orientations)
        parent_indices: Parent index for each joint. Index of 0 signifies that
            a joint is defined relative to the root. We assume that this array is
            sorted: parent joints should always precede child joints.

    Returns:
        Transformations to world frame from each joint frame.
    """

    # Check shapes.
    num_joints = len(parent_indices)
    assert Rs_parent_joint.shape[-2:] == (num_joints, 4)
    assert t_parent_joint.shape[-2:] == (num_joints, 3)

    # Get relative transforms.
    Ts_parent_child = broadcasting_cat([Rs_parent_joint, t_parent_joint], dim=-1)
    assert Ts_parent_child.shape[-2:] == (num_joints, 7)

    # Compute one joint at a time.
    list_Ts_world_joint: list[Tensor] = []
    for i in range(num_joints):
        if parent_indices[i] == 0:
            T_world_parent = T_world_root
        else:
            T_world_parent = list_Ts_world_joint[parent_indices[i] - 1]
        list_Ts_world_joint.append(
            (SE3(T_world_parent) @ SE3(Ts_parent_child[..., i, :])).wxyz_xyz,
        )

    Ts_world_joint = torch.stack(list_Ts_world_joint, dim=-2)
    assert Ts_world_joint.shape[-2:] == (num_joints, 7)
    return Ts_world_joint


def broadcasting_cat(tensors: list[Tensor], dim: int) -> Tensor:
    """Like torch.cat, but broadcasts."""
    assert len(tensors) > 0
    output_dims = max(map(lambda t: len(t.shape), tensors))
    tensors = [
        t.reshape((1,) * (output_dims - len(t.shape)) + t.shape) for t in tensors
    ]
    max_sizes = [max(t.shape[i] for t in tensors) for i in range(output_dims)]
    expanded_tensors = [
        tensor.expand(
            *(
                tensor.shape[i] if i == dim % len(tensor.shape) else max_size
                for i, max_size in enumerate(max_sizes)
            ),
        )
        for tensor in tensors
    ]
    return torch.cat(expanded_tensors, dim=dim)


def _normalize_dtype(v: np.ndarray) -> np.ndarray:
    """Normalize datatypes; all arrays should be either int32 or float32."""
    if "int" in str(v.dtype):
        return v.astype(np.int32)
    elif "float" in str(v.dtype):
        return v.astype(np.float32)
    else:
        return v
