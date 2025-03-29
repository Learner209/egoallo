from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from jaxtyping import Array, jaxtyped
import typeguard
from typing import Optional

from .lbs import lbs, hybrik, rotmat_to_quat, quat_to_rotmat
from ..smplx.lbs import lbs_get_twist
from ..smplx.vertex_ids import vertex_ids as VERTEX_IDS

try:
    import cPickle as pk
except ImportError:
    import pickle as pk


ModelOutput = namedtuple(
    'ModelOutput',
    [
        'vertices', 'joints', 'joints_from_verts',
        'rot_mats',
    ],
)
ModelOutput.__new__.__defaults__ = (None,) * len(ModelOutput._fields)


def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class SMPL_layer(nn.Module):
    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    NUM_BETAS = 10
    JOINT_NAMES = [
        'pelvis', 'left_hip', 'right_hip',      # 2
        'spine1', 'left_knee', 'right_knee',    # 5
        'spine2', 'left_ankle', 'right_ankle',  # 8
        'spine3', 'left_foot', 'right_foot',    # 11
        'neck', 'left_collar', 'right_collar',  # 14
        'jaw',                                  # 15
        'left_shoulder', 'right_shoulder',      # 17
        'left_elbow', 'right_elbow',            # 19
        'left_wrist', 'right_wrist',            # 21
        'left_thumb', 'right_thumb',            # 23
        'head', 'left_middle', 'right_middle',  # 26
        'left_bigtoe', 'right_bigtoe',           # 28
    ]
    LEAF_NAMES = [
        'head', 'left_middle', 'right_middle', 'left_bigtoe', 'right_bigtoe',
    ]
    LEAF_INDICES = [
        VERTEX_IDS['smplh']['nose'],
        VERTEX_IDS['smplh']['lmiddle'],
        VERTEX_IDS['smplh']['rmiddle'],
        VERTEX_IDS['smplh']['LBigToe'],
        VERTEX_IDS['smplh']['RBigToe'],
    ]
    root_idx_17 = 0
    root_idx_smpl = 0

    def __init__(
        self,
        model_path,
        h36m_jregressor,
        gender='neutral',
        dtype=torch.float32,
        num_joints=24,
    ):
        ''' SMPL model layers

        Parameters:
        ----------
        model_path: str
            The path to the folder or to the file where the model
            parameters are stored
        gender: str, optional
            Which gender to load
        '''
        super(SMPL_layer, self).__init__()

        self.ROOT_IDX = self.JOINT_NAMES.index('pelvis')
        self.LEAF_IDX = [self.JOINT_NAMES.index(name) for name in self.LEAF_NAMES]
        self.SPINE3_IDX = 9

        with open(model_path, 'rb') as smpl_file:
            self.smpl_data = Struct(**pk.load(smpl_file, encoding='latin1'))

        self.gender = gender

        self.dtype = dtype

        self.faces = self.smpl_data.f

        ''' Register Buffer '''
        # Faces
        self.register_buffer(
            'faces_tensor',
            to_tensor(to_np(self.smpl_data.f, dtype=np.int64), dtype=torch.long),
        )

        # The vertices of the template model, (6890, 3)
        self.register_buffer(
            'v_template',
            to_tensor(to_np(self.smpl_data.v_template), dtype=dtype),
        )

        # The shape components
        # Shape blend shapes basis, (6890, 3, 10)
        self.register_buffer(
            'shapedirs',
            to_tensor(to_np(self.smpl_data.shapedirs), dtype=dtype),
        )

        # Pose blend shape basis: 6890 x 3 x 23*9, reshaped to 6890*3 x 23*9
        num_pose_basis = self.smpl_data.posedirs.shape[-1]
        # 23*9 x 6890*3
        posedirs = np.reshape(self.smpl_data.posedirs, [-1, num_pose_basis]).T
        self.register_buffer(
            'posedirs',
            to_tensor(to_np(posedirs), dtype=dtype),
        )

        # Vertices to Joints location (23 + 1, 6890)
        self.register_buffer(
            'J_regressor',
            to_tensor(to_np(self.smpl_data.J_regressor), dtype=dtype),
        )
        # Vertices to Human3.6M Joints location (17, 6890)
        self.register_buffer(
            'J_regressor_h36m',
            to_tensor(to_np(h36m_jregressor), dtype=dtype),
        )

        self.num_joints = num_joints
        assert self.num_joints >= 24 and self.num_joints <= 29

        # indices of parents for each joints
        parents = torch.zeros(len(self.JOINT_NAMES), dtype=torch.long)
        parents[:(self.NUM_JOINTS + 1)] = to_tensor(to_np(self.smpl_data.kintree_table[0])).long()
        parents[0] = -1
        # extend kinematic tree
        parents[24] = 15
        parents[25] = 22
        parents[26] = 23
        parents[27] = 10
        parents[28] = 11
        if parents.shape[0] > self.num_joints:
            parents = parents[:self.num_joints]

        self.register_buffer(
            'children_map',
            self._parents_to_children(parents),
        )
        # (24,)
        self.register_buffer('parents', parents)

        # (6890, 23 + 1)
        self.register_buffer(
            'lbs_weights',
            to_tensor(to_np(self.smpl_data.weights), dtype=dtype),
        )

    def _parents_to_children(self, parents):
        children = torch.ones_like(parents) * -1
        for i in range(self.num_joints):
            if children[parents[i]] < 0:
                children[parents[i]] = i
        for i in self.LEAF_IDX:
            if i < children.shape[0]:
                children[i] = -1

        children[self.SPINE3_IDX] = -3
        children[0] = 3
        children[self.SPINE3_IDX] = self.JOINT_NAMES.index('neck')

        return children

    @jaxtyped(typechecker=typeguard.typechecked)
    def forward(
        self,
        pose_axis_angle: Float[torch.Tensor, "batch 23 3"],
        betas: Optional[Float[torch.Tensor, "batch 10"]] = None,
        global_orient: Optional[Float[torch.Tensor, "batch 3"]] = None,
        transl: Optional[Float[torch.Tensor, "batch 3"]] = None,
        return_verts: bool = True,
    ):
        ''' Forward pass for the SMPL model

            Parameters
            ----------
            pose_axis_angle: torch.tensor, optional, shape Bx(J*3)
                It should be a tensor that contains joint rotations in
                axis-angle format. (default=None)
            betas: torch.tensor, optional, shape Bx10
                It can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            global_orient: torch.tensor, optional, shape Bx3
                Global Orientations.
            transl: torch.tensor, optional, shape Bx3
                Global Translations.
            return_verts: bool, optional
                Return the vertices. (default=True)

            Returns
            -------
        '''
        # batch_size = pose_axis_angle.shape[0]

        # concate root orientation with thetas
        if global_orient is not None:
            full_pose = torch.cat([global_orient, pose_axis_angle], dim=1)
        else:
            full_pose = pose_axis_angle

        # Translate thetas to rotation matrics
        pose2rot = True
        # vertices: (B, N, 3), joints: (B, K, 3)
        vertices, joints, rot_mats, joints_from_verts_h36m = lbs(
            betas, full_pose, self.v_template,
            self.shapedirs, self.posedirs,
            self.J_regressor, self.J_regressor_h36m, self.parents,
            self.lbs_weights, pose2rot=pose2rot, dtype=self.dtype,
        )

        if transl is not None:
            root_offset = joints[..., self.root_idx_smpl, :]
            transl = transl - root_offset
            # apply translations
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            joints_from_verts_h36m += transl.unsqueeze(dim=1)
        else:
            vertices = vertices - joints_from_verts_h36m[:, self.root_idx_17, :].unsqueeze(1).detach()
            joints = joints - joints[:, self.root_idx_smpl, :].unsqueeze(1).detach()
            joints_from_verts_h36m = joints_from_verts_h36m - joints_from_verts_h36m[:, self.root_idx_17, :].unsqueeze(1).detach()

        output = ModelOutput(
            vertices=vertices, joints=joints, rot_mats=rot_mats, joints_from_verts=joints_from_verts_h36m,
        )
        return output

    @jaxtyped(typechecker=typeguard.typechecked)
    def hybrik(
            self,
            pose_skeleton: Float[torch.Tensor, "batch 29 3"] | Float[torch.Tensor, "batch 24 3"],
            betas: Float[torch.Tensor, "batch 10"],
            phis: Float[torch.Tensor, "batch 23 2"],
            global_orient: Optional[Float[torch.Tensor, "batch 3 3"]] = None,
            transl: Optional[Float[torch.Tensor, "batch 3"]] = None,
            return_verts: bool = True,
            leaf_thetas: Optional[Float[torch.Tensor, "batch 5 4"]] = None,
            naive=False,
    ):
        ''' Inverse pass for the SMPL model

            Parameters
            ----------
            pose_skeleton: torch.tensor, optional, shape Bx(J*3)
                It should be a tensor that contains joint locations in
                (X, Y, Z) format. (default=None)
            betas: torch.tensor, optional, shape Bx10
                It can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            global_orient: torch.tensor, optional, shape Bx3x3
                Global Orientations.
            transl: torch.tensor, optional, shape Bx3
                Global Translations.
            return_verts: bool, optional
                Return the vertices. (default=True)

            Returns
            -------
        '''
        batch_size = pose_skeleton.shape[0]

        if leaf_thetas is not None:
            leaf_thetas = leaf_thetas.reshape(batch_size * 5, 4)
            leaf_thetas = quat_to_rotmat(leaf_thetas)

        if self.training:
            naive = True

        vertices, new_joints, rot_mats, joints_from_verts = hybrik(
            betas, global_orient, pose_skeleton, phis,
            self.v_template, self.shapedirs, self.posedirs,
            self.J_regressor, self.J_regressor_h36m, self.parents, self.children_map,
            self.lbs_weights, dtype=self.dtype, train=self.training,
            leaf_thetas=leaf_thetas,
            naive=naive, num_joints=self.num_joints,
        )

        rot_mats = rot_mats.reshape(batch_size, self.num_joints, 3, 3)
        # rot_mats = rotmat_to_quat(rot_mats).reshape(batch_size, 24 * 4)

        if transl is not None:
            root_offset = new_joints[..., self.root_idx_smpl, :]
            transl = transl - root_offset
            new_joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            joints_from_verts += transl.unsqueeze(dim=1)
        else:
            vertices = vertices - joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()
            new_joints = new_joints - new_joints[:, self.root_idx_smpl, :].unsqueeze(1).detach()
            joints_from_verts = joints_from_verts - joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()

        output = ModelOutput(
            vertices=vertices, joints=new_joints, rot_mats=rot_mats, joints_from_verts=joints_from_verts,
        )
        return output

    @jaxtyped(typechecker=typeguard.typechecked)
    def forward_get_twist(
        self,
        betas: Optional[Float[torch.Tensor, f"*batch 10"]] = None,
        global_orient: Optional[Float[torch.Tensor, f"*batch 1 3 3"]] = None,
        body_pose: Optional[Float[torch.Tensor, f"*batch 23 3 3"]] = None,
        transl: Optional[Float[torch.Tensor, f"*batch 3"]] = None,
        full_pose: Optional[Float[torch.Tensor, f"*batch dim"]] = None,
    ):

        device, dtype = self.shapedirs.device, self.shapedirs.dtype

        model_vars = [
            betas, global_orient, body_pose, transl,
            full_pose,
        ]
        batch_size = 1
        for var in model_vars:
            if var is None:
                continue
            batch_size = max(batch_size, len(var))

        if full_pose is None:
            if global_orient is None:
                global_orient = torch.eye(3, device=device, dtype=dtype).view(
                    1, 1, 3, 3,
                ).expand(batch_size, -1, -1, -1).contiguous()
            if body_pose is None:
                body_pose = torch.eye(3, device=device, dtype=dtype).view(
                    1, 1, 3, 3,
                ).expand(
                        batch_size, self.NUM_BODY_JOINTS, -1, -1,
                ).contiguous()
            if betas is None:
                betas = torch.zeros(
                    [batch_size, self.NUM_BETAS],
                    dtype=dtype, device=device,
                )
            if transl is None:
                transl = torch.zeros([batch_size, 3], dtype=dtype, device=device)

            # Concatenate all pose vectors
            full_pose = torch.cat(
                [
                    global_orient.reshape(-1, 1, 3, 3),
                    body_pose.reshape(-1, self.NUM_BODY_JOINTS, 3, 3),
                ], dim=1,
            )

        if self.num_joints == self.NUM_JOINTS + 1:
            # No leaf jnts
            twist = lbs_get_twist(
                betas, full_pose, self.v_template, self.shapedirs, self.posedirs, self.J_regressor, self.parents, self.lbs_weights, leaf_indices=[], pose2rot=False,
            )
        else:
            # has 5 additional leaf jnts
            twist = lbs_get_twist(
                betas, full_pose, self.v_template,
                self.shapedirs, self.posedirs,
                self.J_regressor, self.parents,
                self.lbs_weights,
                leaf_indices=self.LEAF_INDICES, pose2rot=False,
            )

        return twist
