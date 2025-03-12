"""Somewhat opinionated wrapper for the SMPL-H body model.

Very little of it is specific to SMPL-H. This could very easily be adapted for other models in SMPL family.

We break down the SMPL-H into four stages, each with a corresponding data structure:
- Loading the model itself:
    `model = SmplhModel.load(path to npz)`
- Applying a body shape to the model:
    `shaped = model.with_shape(betas)`
- Posing the body shape:
    `posed = shaped.with_pose(root pose, local joint poses)`
- Recovering the mesh with LBS:
    `mesh = posed.lbs()`

In contrast to other SMPL wrappers:
- Everything is stateless, so we can support arbitrary batch axes.
- The root is no longer ever called a joint.
- The `trans` and `root_orient` inputs are replaced by a single SE(3) root transformation.
- We're using (4,) wxyz quaternion vectors for all rotations, (7,) wxyz_xyz vectors for all
  rigid transforms.
"""

# TODO: this import may interfere with jaxtyping's runtime checking. comment out not.
# from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import torch
import typeguard
from einops import einsum
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import jaxtyped
from torch import Tensor

from .tensor_dataclass import TensorDataclass
from .transforms import SE3
from .transforms import SO3


@jaxtyped(typechecker=typeguard.typechecked)
class SmplhModel(TensorDataclass):
    """A human body model from the SMPL family."""

    faces: Int[Tensor, "faces 3"]
    """Vertex indices for mesh faces."""
    J_regressor: Float[Tensor, "joints_plus_1 verts"]
    """Linear map from vertex to joint positions.
    For SMPL-H, 1 root + 21 body joints + 2 * 15 hand joints."""
    parent_indices: tuple[int, ...]
    """Defines kinematic tree. Index of -1 signifies that a joint is defined
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

    @staticmethod
    def load(model_path: Path, num_pca_comps: int = 6) -> "SmplhModel":
        """Load a body model from an NPZ or PKL file.

        Args:
            model_path: Path to model file (.npz or .pkl)
            num_pca_comps: Number of PCA components to use for hands

        Returns:
            Loaded SMPL-H model
        """
        # Load data based on file extension
        ext = model_path.suffix.lower()[1:]
        if ext == "pkl":
            with open(model_path, "rb") as f:
                model_data = pickle.load(f, encoding="latin1")
        elif ext == "npz":
            model_data = np.load(model_path, allow_pickle=True)
        else:
            raise ValueError(f"Unknown file extension: {ext}")

        # Convert to dict if needed
        params_numpy = {k: _normalize_dtype(v) for k, v in model_data.items()}

        # Verify model type
        assert (
            "bs_style" not in params_numpy
            or params_numpy.pop("bs_style").item() == b"lbs"
        )
        assert (
            "bs_type" not in params_numpy
            or params_numpy.pop("bs_type").item() == b"lrotmin"
        )

        # Extract parent indices
        parent_indices = tuple(
            int(index) for index in params_numpy.pop("kintree_table")[0][1:] - 1
        )

        # Convert numpy arrays to tensors
        params = {
            k: torch.from_numpy(v)
            for k, v in params_numpy.items()
            if v.dtype in (np.int32, np.float32)
        }

        # Extract hand components if available, limiting to num_pca_comps
        hands_components_l = None
        hands_components_r = None
        if "hands_componentsl" in params:
            hands_components_l = params["hands_componentsl"][:num_pca_comps]
        if "hands_componentsr" in params:
            hands_components_r = params["hands_componentsr"][:num_pca_comps]

        return SmplhModel(
            faces=params["f"],
            J_regressor=params["J_regressor"],
            parent_indices=parent_indices,
            weights=params["weights"],
            posedirs=params["posedirs"],
            v_template=params["v_template"],
            shapedirs=params["shapedirs"],
            hands_components_l=hands_components_l,
            hands_components_r=hands_components_r,
        )

    @classmethod
    @jaxtyped(typechecker=typeguard.typechecked)
    def pca_to_aa(
        cls,
        hand_pose_pca: Float[Tensor, "*batch num_pca"],
        hand_components: Float[Tensor, "num_pca 45"],
    ) -> Float[Tensor, "*batch 15 3"]:
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

    @jaxtyped(typechecker=typeguard.typechecked)
    def convert_hand_poses(
        self,
        left_hand_pca: Float[Tensor, "*batch num_pca"] | None = None,
        right_hand_pca: Float[Tensor, "*batch num_pca"] | None = None,
    ) -> tuple[
        Float[Tensor, "*batch 15 3"] | None,
        Float[Tensor, "*batch 15 3"] | None,
    ]:
        """Convert both hand PCA coefficients to axis-angle format.

        Args:
            left_hand_pca: PCA coefficients for left hand pose
            right_hand_pca: PCA coefficients for right hand pose

        Returns:
            Tuple of (left_hand_pose, right_hand_pose) in axis-angle format
        """
        left_hand_pose = None
        right_hand_pose = None

        if left_hand_pca is not None:
            left_hand_pose = self.pca_to_aa(left_hand_pca, self.hands_components_l)

        if right_hand_pca is not None:
            right_hand_pose = self.pca_to_aa(right_hand_pca, self.hands_components_r)

        return left_hand_pose, right_hand_pose

    def get_num_joints(self) -> int:
        """Get the number of joints in this model."""
        return len(self.parent_indices)

    @jaxtyped(typechecker=typeguard.typechecked)
    def with_shape(self, betas: Float[Tensor, "*batch n_betas"]) -> "SmplhShaped":
        """Compute a new body model, with betas applied."""
        num_betas = betas.shape[-1]
        assert num_betas <= self.shapedirs.shape[-1]
        verts_with_shape = self.v_template + einsum(
            self.shapedirs[:, :, :num_betas],
            betas,
            "verts xyz beta, ... beta -> ... verts xyz",
        )
        assert isinstance(verts_with_shape, Float[Tensor, "*batch verts xyz"])
        root_and_joints_pred = einsum(
            self.J_regressor,
            verts_with_shape,
            "jointsp1 verts, ... verts xyz -> ... jointsp1 xyz",
        )
        assert isinstance(root_and_joints_pred, Float[Tensor, "*batch jointsp1 xyz"])
        root_offset = root_and_joints_pred[..., 0:1, :]  # shape: (*batch 1 xyz)

        return SmplhShaped(
            body_model=self,
            root_offset=root_offset.squeeze(-2),
            verts_zero=verts_with_shape - root_offset,
            joints_zero=root_and_joints_pred[..., 1:, :] - root_offset,
            t_parent_joint=root_and_joints_pred[..., 1:, :]
            - root_and_joints_pred[..., np.array(self.parent_indices) + 1, :],
        )


@jaxtyped(typechecker=typeguard.typechecked)
class SmplhShaped(TensorDataclass):
    """The SMPL-H body model with a body shape applied."""

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

    @jaxtyped(typechecker=typeguard.typechecked)
    def with_pose_decomposed(
        self,
        T_world_root: Float[Tensor, "*batch 7"],
        body_quats: Float[Tensor, "*batch 21 4"],
        left_hand_quats: Float[Tensor, "*batch 15 4"] | None = None,
        right_hand_quats: Float[Tensor, "*batch 15 4"] | None = None,
    ) -> "SmplhShapedAndPosed":
        """Pose our SMPL-H body model. Returns a set of joint and vertex outputs."""

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
    def with_pose(
        self,
        T_world_root: Float[Tensor, "*batch 7"],
        local_quats: Float[Tensor, "*batch joints 4"],
    ) -> "SmplhShapedAndPosed":
        """Pose our SMPL-H body model. Returns a set of joint and vertex outputs."""

        # Forward kinematics.
        num_joints = self.body_model.get_num_joints()
        assert local_quats.shape[-2:] == (num_joints, 4)

        Ts_world_joint = forward_kinematics(
            T_world_root=T_world_root,
            Rs_parent_joint=local_quats,
            t_parent_joint=self.t_parent_joint,
            parent_indices=self.body_model.parent_indices,
        )
        assert Ts_world_joint.shape[-2:] == (num_joints, 7)
        return SmplhShapedAndPosed(
            shaped_model=self,
            T_world_root=T_world_root,
            local_quats=local_quats,
            Ts_world_joint=Ts_world_joint,
        )


@jaxtyped(typechecker=typeguard.typechecked)
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

    @jaxtyped(typechecker=typeguard.typechecked)
    def with_new_T_world_root(
        self,
        T_world_root: Float[Tensor, "*batch 7"],
    ) -> "SmplhShapedAndPosed":
        return SmplhShapedAndPosed(
            shaped_model=self.shaped_model,
            T_world_root=T_world_root,
            local_quats=self.local_quats,
            Ts_world_joint=(
                SE3(T_world_root[..., None, :])
                @ SE3(self.T_world_root[..., None, :]).inverse()
                @ SE3(self.Ts_world_joint)
            ).parameters(),
        )

    def lbs(self) -> "SmplMesh":
        """Compute a mesh with LBS."""
        num_joints = self.local_quats.shape[-2]
        verts_with_blend = self.shaped_model.verts_zero + einsum(
            self.shaped_model.body_model.posedirs,
            (
                SO3(self.local_quats).as_matrix()
                - torch.eye(
                    3,
                    dtype=self.local_quats.dtype,
                    device=self.local_quats.device,
                )
            ).reshape((*self.local_quats.shape[:-2], num_joints * 9)),
            "... verts j joints_times_9, ... joints_times_9 -> ... verts j",
        )
        verts_transformed = einsum(
            broadcasting_cat(
                [
                    SE3(self.T_world_root).as_matrix()[..., None, :3, :],
                    SE3(self.Ts_world_joint).as_matrix()[..., :, :3, :],
                ],
                dim=-3,
            ),
            self.shaped_model.body_model.weights,
            broadcasting_cat(
                [
                    verts_with_blend[..., :, None, :]
                    - broadcasting_cat(  # Prepend root to joints zeros.
                        [
                            self.shaped_model.joints_zero.new_zeros(3),
                            self.shaped_model.joints_zero[..., None, :, :],
                        ],
                        dim=-2,
                    ),
                    verts_with_blend.new_ones(
                        (
                            *verts_with_blend.shape[:-1],
                            1 + self.shaped_model.joints_zero.shape[-2],
                            1,
                        ),
                    ),
                ],
                dim=-1,
            ),
            "... joints_p1 i j, verts joints_p1, ... verts joints_p1 j -> ... verts i",
        )
        assert (
            verts_transformed.shape[-2:]
            == self.shaped_model.body_model.v_template.shape
        )
        return SmplMesh(
            posed_model=self,
            verts=verts_transformed,
            faces=self.shaped_model.body_model.faces,
        )

    @jaxtyped(typechecker=typeguard.typechecked)
    def compute_joint_contacts(
        self,
        vertex_contacts: Float[Tensor, "*batch verts"],
    ) -> Float[Tensor, "*batch joints"]:
        """Convert per-vertex contact labels to per-joint contact labels using skinning weights.

        Args:
            vertex_contacts: Binary contact labels for each vertex (0 or 1)

        Returns:
            Joint contact labels (continuous values between 0 and 1)
        """
        # Get skinning weights from the body model
        weights = self.shaped_model.body_model.weights  # (verts, joints+1)

        # Remove root weights (first column) as we only want joint contacts
        joint_weights = weights  # (verts, joints)

        # Weighted sum of contact labels
        weighted_contacts = einsum(
            vertex_contacts,
            joint_weights,
            "... verts, verts joints -> ... joints",
        )

        # Normalize by sum of weights
        weight_sums = joint_weights.sum(dim=0)  # (joints,)
        joint_contacts = weighted_contacts / weight_sums

        # Threshold to get binary labels (optional, adjust threshold as needed)
        joint_contacts = (joint_contacts > 0.5).float()

        return joint_contacts


@jaxtyped(typechecker=typeguard.typechecked)
class SmplMesh(TensorDataclass):
    """Outputs from the SMPL-H model."""

    posed_model: SmplhShapedAndPosed
    """Posed model that this mesh was computed for."""

    verts: Float[Tensor, "*batch verts 3"]
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
        parent_indices: Parent index for each joint. Index of -1 signifies that
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
