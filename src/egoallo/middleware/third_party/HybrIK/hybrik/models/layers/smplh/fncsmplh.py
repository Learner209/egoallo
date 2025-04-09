"""NOTE: only support 1 batch axes, not arbitrary batch axes.
Update: 4/1
Problems:
1. if the left_hand_pose and right_hand_pose are not passed in during forward call, then smplx.SMPLH would use flat_hand_mean, which depends on the batch_size param when instantiating. This is an inconsistency across SmplFamilyModelType APIs! SmplFamilyModelType APIs should support inferring batch size while running forward call.

"""

import torch
import smplx
import typeguard
import numpy as np
from pathlib import Path
from jaxtyping import Float, Int
from typing import Self
from egoallo.transforms import SE3, SO3
from torch import Tensor
from egoallo.tensor_dataclass import TensorDataclass
from jaxtyping import jaxtyped
from einops import einsum


@jaxtyped(typechecker=typeguard.typechecked)
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
        gender = kwargs.get("gender", "neutral")
        model_path = model_path / "smplh" / f"SMPLH_{gender.upper()}.pkl"
        kwargs['use_pca'] = kwargs.get('use_pca', False)
        model = smplx.SMPLH(str(model_path), ext='pkl', **kwargs)
        parent_indices = tuple((model.parents[1:] - 1).tolist())  # Exclude root
        faces = torch.from_numpy(model.faces.astype(np.int32))
        num_verts = model.J_regressor.shape[1]
        num_jnts = model.J_regressor.shape[0] - 1
        return cls(
            faces=faces,
            J_regressor=model.J_regressor,
            parent_indices=parent_indices,
            weights=model.lbs_weights,
            posedirs=model.posedirs.transpose(0,1).reshape(num_verts, 3, num_jnts * 9),
            v_template=model.v_template,
            shapedirs=model.shapedirs,
            hands_components_l=model.left_hand_components if hasattr(model, "left_hand_components") else None,
            hands_components_r=model.right_hand_components if hasattr(model, "right_hand_components") else None,
            model=model,
        )

    @classmethod
    def pca_to_aa(
        cls,
        hand_pose_pca: Float[Tensor, "*#batch num_pca"],
        hand_components: Float[Tensor, "num_pca 45"],
    ) -> Float[Tensor, "*#batch 15 3"]:
        """Convert PCA coefficients to axis-angle rotations for hand poses.

        Args:
            hand_pose_pca: PCA coefficients for hand pose
            hand_components: Hand components matrix (left or right)

        Returns:
            Hand joint rotations in axis-angle format
        """
        # Multiply PCA coefficients with components to get axis-angle values
        hand_pose = einsum(
            hand_pose_pca,
            hand_components,
            "... num_pca, num_pca joints3 -> ... joints3",
        )
        # Reshape to (batch_size, 15, 3) format
        return hand_pose.reshape(*hand_pose.shape[:-1], 15, 3)

    def convert_hand_poses(
        self,
        left_hand_pca: Float[Tensor, "*#batch num_pca"] | None = None,
        right_hand_pca: Float[Tensor, "*#batch num_pca"] | None = None,
    ) -> tuple[
        Float[Tensor, "*#batch 15 3"] | None, Float[Tensor, "*#batch 15 3"] | None,
    ]:
        """Convert both hand PCA coefficients to axis-angle format.

        Args:
            left_hand_pca: PCA coefficients for left hand pose
            right_hand_pca: PCA coefficients for right hand pose

        Returns:
            Tuple of (left_hand_pose, right_hand_pose) in axis-angle format
        """
        device = left_hand_pca.device
        dtype = left_hand_pca.dtype

        left_hand_pose = None
        right_hand_pose = None

        if left_hand_pca is not None:
            left_hand_pose = self.pca_to_aa(left_hand_pca, self.hands_components_l.to(device, dtype))

        if right_hand_pca is not None:
            right_hand_pose = self.pca_to_aa(right_hand_pca, self.hands_components_r.to(device, dtype))

        return left_hand_pose, right_hand_pose

    def get_num_joints(self) -> int:
        return len(self.parent_indices)

    @jaxtyped(typechecker=typeguard.typechecked)
    def with_shape(self, betas: Float[Tensor, "*batch num_betas"]) -> "SmplhShaped":
        batch_axes = betas.shape[:-1]

        default_global_orient = torch.zeros(batch_axes + (3,), dtype=betas.dtype)
        default_body_pose = torch.zeros(batch_axes + (self.model.NUM_BODY_JOINTS * 3,), dtype=betas.dtype)
        default_transl = torch.zeros(batch_axes + (3,), dtype=betas.dtype)
        default_left_hand_pose = torch.zeros(batch_axes + (15 * 3,), dtype=betas.dtype)
        default_right_hand_pose = torch.zeros(batch_axes + (15 * 3,), dtype=betas.dtype)

        from egoallo.tensor_dataclass_batch_plugins import TensorDataclassBatchPlugin
        flattened_global_orient, _ = TensorDataclassBatchPlugin.flatten_batch_dims(default_global_orient, batch_axes)
        flattened_body_pose, _ = TensorDataclassBatchPlugin.flatten_batch_dims(default_body_pose, batch_axes)
        flattened_transl, _ = TensorDataclassBatchPlugin.flatten_batch_dims(default_transl, batch_axes)
        flattened_left_hand_pose, _ = TensorDataclassBatchPlugin.flatten_batch_dims(default_left_hand_pose, batch_axes)
        flattened_right_hand_pose, _ = TensorDataclassBatchPlugin.flatten_batch_dims(default_right_hand_pose, batch_axes)

        flattened_betas, _ = TensorDataclassBatchPlugin.flatten_batch_dims(betas, batch_axes)

        device = self.model.pose_mean.device
        output = self.model.forward(
            betas=flattened_betas,
            global_orient=flattened_global_orient.to(device),
            body_pose=flattened_body_pose.to(device),
            transl=flattened_transl.to(device),
            left_hand_pose=flattened_left_hand_pose.to(device),
            right_hand_pose=flattened_right_hand_pose.to(device),
            return_verts=True,
        )

        output_joints = TensorDataclassBatchPlugin.unflatten_batch_dims(output.joints, batch_axes)
        output_vertices = TensorDataclassBatchPlugin.unflatten_batch_dims(output.vertices, batch_axes)

        root_offset = output_joints[..., 0, :]
        verts_zero = output_vertices - root_offset.unsqueeze(-2)
        joints_zero = output_joints[
            ...,
            1 : self.get_num_joints() + 1,
            :,
        ] - root_offset.unsqueeze(-2)
        root_and_joints_zero = output_joints - root_offset.unsqueeze(-2)

        t_parent_joint = (
            joints_zero
            - root_and_joints_zero[
                ...,
                torch.tensor(np.array(self.parent_indices) + 1).to(
                    root_and_joints_zero.device,
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

    def to(self, device: torch.device) -> Self:
        super(SmplhModel, self).to(device)
        self.model = self.model.to(device)
        return self


@jaxtyped(typechecker=typeguard.typechecked)
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
    betas: Float[Tensor, "*#batch num_betas"]
    """betas"""

    @jaxtyped(typechecker=typeguard.typechecked)
    def with_pose(
        self,
        T_world_root: Float[Tensor, "*batch 7"],
        local_quats: Float[Tensor, "*batch joints 4"],
    ) -> "SmplhShapedAndPosed":
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
        )

    @jaxtyped(typechecker=typeguard.typechecked)
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


@jaxtyped(typechecker=typeguard.typechecked)
class SmplhShapedAndPosed(TensorDataclass):
    """Outputs from the SMPL-H model."""

    shaped_model: SmplhShaped
    """Underlying shaped body model."""

    T_world_root: Float[Tensor, "*batch 7"]
    """Root coordinate frame."""

    local_quats: Float[Tensor, "*batch joints 4"]
    """Local joint orientations."""

    Ts_world_joint: Float[Tensor, "*batch joints 7"]
    """Absolute transform for each joint. Does not include the root."""

    def with_new_T_world_root(
        self,
        T_world_root: Float[Tensor, "*batch 7"],
    ) -> "SmplhShapedAndPosed":
        raise NotImplementedError()
        # return SmplhShapedAndPosed(self.shaped, T_world_root, self.local_quats)

    def lbs(self) -> "SmplhMesh":
        batch_dim = self.local_quats.shape[:-2]
        # NOTE: intentionally left transl to be zeros to deal with root offset afterwards. since the root joint still has offset in the current coordinate.
        output = self.shaped_model.body_model.model(
            betas=self.shaped_model.betas,
            global_orient=SE3(self.T_world_root).rotation().log().view(*batch_dim, -1),
            body_pose=SO3(self.local_quats[..., :21, :]).log().view(*batch_dim, -1),
            left_hand_pose=SO3(self.local_quats[..., 21 : 21 + 15, :])
            .log()
            .view(*batch_dim, -1),
            right_hand_pose=SO3(self.local_quats[..., 21 + 15 :, :])
            .log()
            .view(*batch_dim, -1),
            transl=torch.zeros_like(self.T_world_root[..., 4:7]),
            return_verts=True,
        )
        root_offset = output.joints[..., 0:1, :]
        output.vertices -= root_offset
        output.joints -= root_offset

        # apply transl
        output.vertices += SE3(self.T_world_root).translation().unsqueeze(-2)
        output.joints += SE3(self.T_world_root).translation().unsqueeze(-2)

        return SmplhMesh(self, output.vertices, self.shaped_model.body_model.faces)

    def compute_joint_contacts(
        self, vertex_contacts: Float[Tensor, "*#batch verts"],
    ) -> Float[Tensor, "*#batch joints"]:
        """Convert per-vertex contact labels to per-joint contact labels using skinning weights.

        Args:
            vertex_contacts: Binary contact labels for each vertex (0 or 1)

        Returns:
            Joint contact labels (continuous values between 0 and 1)
        """
        device, dtype = vertex_contacts.device, vertex_contacts.dtype
        # Get skinning weights from the body model
        weights = self.shaped_model.body_model.weights.to(device, dtype)  # (verts, joints+1)

        # Weighted sum of contact labels
        weighted_contacts = einsum(
            vertex_contacts,
            weights,
            "... verts, verts joints -> ... joints",
        )

        # Normalize by sum of weights
        weight_sums = weights.sum(dim=0)  # (joints,)
        joint_contacts = weighted_contacts / weight_sums

        # Threshold to get binary labels (optional, adjust threshold as needed)
        # joint_contacts = (joint_contacts > 0.3).float()

        return joint_contacts



@jaxtyped(typechecker=typeguard.typechecked)
class SmplhMesh(TensorDataclass):
    """Outputs from the SMPL-H model."""

    posed_model: SmplhShapedAndPosed
    """Posed model that this mesh was computed for."""

    vertices: Float[Tensor, "*batch verts 3"]
    """Vertices for mesh."""

    faces: Int[Tensor, "faces 3"]
    """Faces for mesh."""


@jaxtyped(typechecker=typeguard.typechecked)
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

    num_joints = len(parent_indices)
    assert Rs_parent_joint.shape[-2:] == (num_joints, 4)
    assert t_parent_joint.shape[-2:] == (num_joints, 3)

    # Get relative transforms.
    Ts_parent_child = broadcasting_cat([Rs_parent_joint, t_parent_joint], dim=-1)
    assert Ts_parent_child.shape[-2:] == (num_joints, 7)

    # Compute one joint at a time.
    list_Ts_world_joint: list[Tensor] = []
    for i in range(num_joints):
        if parent_indices[i] == -1:
            T_world_parent = T_world_root
        else:
            T_world_parent = list_Ts_world_joint[parent_indices[i]]
        list_Ts_world_joint.append(
            (SE3(T_world_parent) @ SE3(Ts_parent_child[..., i, :])).wxyz_xyz,
        )

    Ts_world_joint = torch.stack(list_Ts_world_joint, dim=-2)
    assert Ts_world_joint.shape[-2:] == (num_joints, 7)
    return Ts_world_joint


@jaxtyped(typechecker=typeguard.typechecked)
def inverse_kinematics(
    T_world_root: Float[Tensor, "*#batch 7"],
    Ts_world_joints: Float[Tensor, "*#batch joints 7"],
    parent_indices: tuple[int, ...],
) -> tuple[Float[Tensor, "*#batch joints 4"], Float[Tensor, "*#batch joints 3"]]:
    """
    Run inverse kinematics to compute local joint rotations (Rs_parent_joint)
    and local joint translations (t_parent_joint) relative to their parents.

    This function reverses the process of forward_kinematics.

    Args:
        T_world_root: Absolute transformation (wxyz_xyz) to world frame from root frame.
        Ts_world_joint: Absolute transformations (wxyz_xyz) to world frame from each joint frame.
                       Shape must match the number of joints defined in parent_indices.
        parent_indices: Parent index for each joint. Index of -1 signifies that
                       a joint is defined relative to the root.

    Returns:
        A tuple containing:
        - Rs_parent_joint: Local orientation (wxyz quat) of each joint relative to its parent.
        - t_parent_joint: Position (xyz) of each joint relative to its parent,
                          expressed in the parent's coordinate frame.
    """
    num_joints = len(parent_indices)
    assert Ts_world_joints.shape[-2] == num_joints, \
        f"Ts_world_joint shape {Ts_world_joints.shape} inconsistent with num_joints {num_joints}"
    batch_shape = T_world_root.shape[:-1]
    assert Ts_world_joints.shape[:-2] == batch_shape, \
        f"Batch shapes mismatch: T_world_root {T_world_root.shape} vs Ts_world_joint {Ts_world_joints.shape}"

    device = T_world_root.device
    dtype = T_world_root.dtype

    Rs_parent_joint_list = []
    t_parent_joint_list = []

    for i in range(num_joints):
        parent_idx = parent_indices[i]

        T_world_child = Ts_world_joints[..., i, :]

        if parent_idx == -1:
            T_world_parent = T_world_root
        else:
            assert 0 <= parent_idx < num_joints, f"Invalid parent index {parent_idx} for joint {i}"
            T_world_parent = Ts_world_joints[..., parent_idx, :]

        T_parent_child = SE3(T_world_parent).inverse() @ SE3(T_world_child)

        R_parent_joint_i = T_parent_child.rotation().wxyz
        t_parent_joint_i = T_parent_child.translation()

        Rs_parent_joint_list.append(R_parent_joint_i)
        t_parent_joint_list.append(t_parent_joint_i)

    Rs_parent_joint_with_root = torch.stack(Rs_parent_joint_list, dim=-2)
    t_parent_joint_with_root = torch.stack(t_parent_joint_list, dim=-2)

    assert Rs_parent_joint_with_root.shape == batch_shape + (num_joints, 4)
    assert t_parent_joint_with_root.shape == batch_shape + (num_joints, 3)

    return Rs_parent_joint_with_root[..., 1:, :], t_parent_joint_with_root[..., 1:, :]


@jaxtyped(typechecker=typeguard.typechecked)
def inverse_kinematics_rotation_only(
    R_world_root: Float[Tensor, "*#batch 3 3"],
    Rs_world_joints: Float[Tensor, "*#batch joints 3 3"],
    parent_indices: tuple[int, ...], # parent indices should exclude root, and its length is num_joints - 1, parent indicator for root is -1.
) -> Float[Tensor, "*#batch joints 3 3"]:
    num_joints = len(parent_indices)
    assert Rs_world_joints.shape[-3] == num_joints, \
        f"Rs_world_joints shape {Rs_world_joints.shape} inconsistent with num_joints {num_joints}"
    batch_shape = R_world_root.shape[:-2]
    assert Rs_world_joints.shape[:-3] == batch_shape, \
        f"Batch shapes mismatch: R_world_root {R_world_root.shape} vs Rs_world_joints {Rs_world_joints.shape}"

    Rs_world_joint_with_root = torch.cat([R_world_root[..., None, :, :], Rs_world_joints], dim=-3)

    Rs_parent_joint = (SO3.from_matrix(Rs_world_joint_with_root[..., np.asarray(parent_indices)+1,  :, :]).inverse() @ SO3.from_matrix(Rs_world_joint_with_root[..., 1:, :, :])).as_matrix()
    return Rs_parent_joint

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
