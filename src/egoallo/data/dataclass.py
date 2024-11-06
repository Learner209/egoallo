"""
=====================
Coordiante transforms
=====================

Aria rgb/slam-left/slam-right cam: X down, Z in front, Y Left, thus image portrait view is horizontal.
SMPL follows the convention of X left, Y up, Z front, thus egoego follows the same convention.
Blender/OpenGL Camera follows the convention of X right, Y up, Z back.
OpenCV Camera follows the convention of X right, Y down, Z front.

"""


from __future__ import annotations

from pathlib import Path

from videoio import VideoWriter
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
import typeguard
from jaxtyping import Bool, Float, jaxtyped
from typing import TypeVar, Optional, Callable
from torch import Tensor
import trimesh

import cv2

from .. import fncsmpl, fncsmpl_extensions
from .. import transforms as tf
from ..tensor_dataclass import TensorDataclass
import numpy as np
# from blendify import scene
# from blendify.colors import UniformColors, FacesUV, FileTextureColors
# from blendify.materials import PrincipledBSDFMaterial
# from blendify.utils.image import blend_with_background
# from blendify.utils.smpl_wrapper import SMPLWrapper

from egoallo.setup_logger import setup_logger
from egoallo.fncsmpl import SmplhModel, SmplhShaped, SmplhShapedAndPosed, SmplMesh
from egoallo import fncsmpl, transforms
from egoallo.network import EgoDenoiseTraj
logger = setup_logger(output=None, name=__name__)

@jaxtyped(typechecker=typeguard.typechecked)
class EgoTrainingData(TensorDataclass):
    """Dictionary of tensors we use for EgoAllo training."""

    T_world_root: Float[Tensor, "*#batch timesteps 7"]
    """Transformation from the world frame to the root frame at each timestep."""

    contacts: Float[Tensor, "*#batch timesteps 21"]
    """Contact boolean for each joint."""

    betas: Float[Tensor, "*#batch 1 16"]
    """Body shape parameters."""

    body_quats: Float[Tensor, "*#batch timesteps 21 4"]
    """Local orientations for each body joint."""

    T_cpf_tm1_cpf_t: Float[Tensor, "*#batch timesteps 7"]
    """Transformation to the next central pupil frame, from this timestep's
    central pupil frame."""

    T_world_cpf: Float[Tensor, "*#batch timesteps 7"]
    """Transformation from the world frame to the central pupil frame at each timestep."""

    height_from_floor: Float[Tensor, "*#batch timesteps 1"]
    """Distance from CPF to floor at each timestep."""

    joints_wrt_cpf: Float[Tensor, "*#batch timesteps 21 3"]
    """Joint positions relative to the central pupil frame."""

    mask: Bool[Tensor, "*#batch timesteps"]
    """Mask to support variable-length sequence."""

    hand_quats: Optional[Float[Tensor, "*#batch timesteps 30 4"]] = None
    """Local orientations for each hand joint."""

    prev_window: Optional["EgoTrainingData"] = None
    """Previous window of training data for conditioning."""

    def __post_init__(self):
        """Validate the dataclass after initialization."""
        # Ensure all required tensor fields are present and have correct types
        for field_name, field_type in self.__annotations__.items():
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, (torch.Tensor, EgoTrainingData)):
                raise TypeError(f"Field {field_name} must be a Tensor, None, or EgoTrainingData, got {type(value)}")

    def to(self, device: torch.device | str):
        """Override to handle prev_window correctly."""
        result = super().to(device)
        if result.prev_window is not None:
            result.prev_window = result.prev_window.to(device)
        return result

    def map(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        """Override to handle prev_window correctly."""
        result = super().map(fn)
        if result.prev_window is not None:
            result.prev_window = result.prev_window.map(fn)
        return result

    @property
    def joints_wrt_world(self) -> Tensor:
        return tf.SE3(self.T_world_cpf[..., None, :]) @ self.joints_wrt_cpf

    def with_prev_window(self, prev_window: Optional[EgoTrainingData]) -> EgoTrainingData:
        """Create a new EgoTrainingData instance with the given prev_window."""
        return EgoTrainingData(
            T_world_root=self.T_world_root,
            contacts=self.contacts,
            betas=self.betas,
            body_quats=self.body_quats,
            T_cpf_tm1_cpf_t=self.T_cpf_tm1_cpf_t,
            T_world_cpf=self.T_world_cpf,
            height_from_floor=self.height_from_floor,
            joints_wrt_cpf=self.joints_wrt_cpf,
            mask=self.mask,
            hand_quats=self.hand_quats,
            prev_window=prev_window
        )

    @staticmethod
    def load_from_npz(
        body_model: fncsmpl.SmplhModel,
        path: Path,
        include_hands: bool,
    ) -> EgoTrainingData:
        """Load a single trajectory from a (processed_30fps) npz file."""
        raw_fields = {
            k: torch.from_numpy(v.astype(np.float32) if v.dtype == np.float64 else v)
            for k, v in np.load(path).items()
            if v.dtype in (np.float32, np.float64)
        }

        timesteps = raw_fields["root_orient"].shape[0]
        assert raw_fields["root_orient"].shape == (timesteps, 3)
        assert raw_fields["pose_body"].shape == (timesteps, 63)
        assert raw_fields["pose_hand"].shape == (timesteps, 90)
        assert raw_fields["joints"].shape == (timesteps, 22, 3)

        T_world_root = torch.cat(
            [
                tf.SO3.exp(raw_fields["root_orient"]).wxyz,
                raw_fields["joints"][:, 0, :],
            ],
            dim=-1,
        )
        body_quats = tf.SO3.exp(raw_fields["pose_body"].reshape(timesteps, 21, 3)).wxyz
        hand_quats = tf.SO3.exp(raw_fields["pose_hand"].reshape(timesteps, 30, 3)).wxyz

        device = body_model.weights.device
        shaped = body_model.with_shape(raw_fields["betas"][None].to(device))

        # Batch the SMPL body model operations, this can be pretty memory-intensive...
        posed = shaped.with_pose_decomposed(
            T_world_root=T_world_root.to(device), body_quats=body_quats.to(device)
        )
        smplh_mesh = posed.lbs()

        T_world_cpf = (
            tf.SE3(posed.Ts_world_joint[:, 14, :])  # T_world_head
            @ tf.SE3(fncsmpl_extensions.get_T_head_cpf(shaped))
        ).parameters()
        assert T_world_cpf.shape == (timesteps, 7)


        # Construct the training data elements that we want to keep.
        ego_data = EgoTrainingData(
            T_world_root=T_world_root[1:].cpu(),
            contacts=raw_fields["contacts"][1:, 1:].cpu(),  # Root is no longer a joint.
            betas=raw_fields["betas"][None].cpu(),
            # joints_wrt_world=raw_fields["joints"][
            #     1:, 1:
            # ].cpu(),  # Root is no longer a joint.
            body_quats=body_quats[1:].cpu(),
            # CPF frame stuff.
            T_world_cpf=T_world_cpf[1:].cpu(),
            # Get translational z coordinate from wxyz_xyz.
            height_from_floor=T_world_cpf[1:, 6:7].cpu(),
            T_cpf_tm1_cpf_t=(
                tf.SE3(T_world_cpf[:-1, :]).inverse() @ tf.SE3(T_world_cpf[1:, :])
            )
            .parameters()
            .cpu(),
            joints_wrt_cpf=(
                # unsqueeze so both shapes are (timesteps, joints, dim)
                tf.SE3(T_world_cpf[1:, None, :]).inverse()
                @ raw_fields["joints"][1:, 1:, :].to(T_world_cpf.device)
            ).cpu(),
            mask=torch.ones((timesteps - 1,), dtype=torch.bool),
            hand_quats=hand_quats[1:].cpu() if include_hands else None,
        )
     
        return ego_data

    def pack(self) -> EgoDenoiseTraj:
        """Convert EgoTrainingData to EgoDenoiseTraj format."""
        # Convert quaternions to 6D rotation representation
        body_rot6d = tf.SO3(wxyz=self.body_quats).as_rot6d()
        
        # Convert hand quaternions if they exist
        hand_rot6d = None
        if self.hand_quats is not None:
            hand_rot6d = tf.SO3(wxyz=self.hand_quats).as_rot6d()

        # Pack the prev_window if it exists
        prev_window_packed = None
        if self.prev_window is not None:
            prev_window_packed = self.prev_window.pack()

        # Create EgoDenoiseTraj instance
        return EgoDenoiseTraj(
            betas=self.betas,  # Expand betas to match timesteps
            body_rot6d=body_rot6d,
            contacts=self.contacts,
            hand_rot6d=hand_rot6d,
            prev_window=prev_window_packed
        )

    @staticmethod
    def visualize_ego_training_data(
            ego_data: EgoTrainingData, 
            body_model: fncsmpl.SmplhModel
        ):
        """
        Visualize EgoTrainingData using the blendify API and SMPL model.

        :param ego_data: EgoTrainingData instance containing the pose and transformation data.
        :param smpl_path: Path to the SMPL model files.
        """
        device = body_model.weights.device
        shaped = body_model.with_shape(ego_data.betas.to(device))
        # Batch the SMPL body model operations, this can be pretty memory-intensive...
        posed = shaped.with_pose_decomposed(
            T_world_root=ego_data.T_world_root.to(device), body_quats=ego_data.body_quats.to(device)
        )
        smplh_mesh = posed.lbs()

        # Add the camera
        camera = scene.set_perspective_camera((1280,720), fov_y=np.deg2rad(75))
        # camera.location = (0, -3, 1)  # Adjust camera location for better view

        # Define the materials
        # Material and Colors for SMPL mesh
        smpl_material = PrincipledBSDFMaterial()
        smpl_colors = UniformColors((0.3, 0.3, 0.3))
        # Add the SMPL mesh, set the pose to zero for the first frame, just to initialize
        smpl_mesh = scene.renderables.add_mesh(smplh_mesh.verts[0].numpy(force=True), smplh_mesh.faces.numpy(force=True), smpl_material, smpl_colors)
        smpl_mesh.set_smooth()  # Force the surface of model to look smooth

        # Material and Colors for background scene mesh
        trimesh_mesh = trimesh.load(Path("./blendify_assets/scene_mesh.ply"))
        uv_map = np.load(Path("./blendify_assets/scene_face_uvmap.npy"))
        texture_path = "./blendify_assets/scene_texture.jpg"
        scene_mesh_material = PrincipledBSDFMaterial()
        scene_mesh_material.specular = 1.0
        scene_mesh_material.roughness = 1.0
        scene_mesh_colors = FileTextureColors(texture_path, FacesUV(uv_map))

        # Add the background scene mesh; turn off shadowing as the shadows are already baked in the texture
        scene_mesh = scene.renderables.add_mesh(
            vertices=np.array(trimesh_mesh.vertices), faces=np.array(trimesh_mesh.faces), material=scene_mesh_material, colors=scene_mesh_colors
        )
        scene_mesh.emit_shadows = False

        # Set the lights; one main sunlight and a secondary light without visible shadows to make the scene overall brighter
        sunlight = scene.lights.add_sun(
            strength=2.3, rotation_mode="euleryz", rotation=(-45, -90)
        )
        sunlight2 = scene.lights.add_sun(
            strength=3, rotation_mode="euleryz", rotation=(-45, 165)
        )
        sunlight2.cast_shadows = False

        # Rendering loop
        path = "output.mp4"
        logger.info("Entering the main drawing loop")
        total_frames= ego_data.T_world_root.shape[0]
        with VideoWriter(path, resolution=(1280,720), fps=30) as vw:
            # Loop over each timestep to visualize the pose
            for i in tqdm(range(ego_data.T_world_root.shape[0]-1)):
                # Get SMPL pose parameters
                smpl_pose = ego_data.body_quats[i].numpy()

                camera_quaternion = transforms.SO3.identity(device=torch.device("cuda"),dtype=torch.float64).wxyz.numpy(force=True)

                right_eye = (smplh_mesh.verts[i+1, 6260, :] + smplh_mesh.verts[i+1, 6262, :]) / 2.0
                left_eye = (smplh_mesh.verts[i+1, 2800, :] + smplh_mesh.verts[i+1, 2802, :]) / 2.0
                cpf_pos = (right_eye + left_eye) / 2.0
                cpf_pos[2] = cpf_pos[2] + 1.0

                smpl_mesh.update_vertices(smplh_mesh.verts[i+1].numpy(force=True))
                # Set the current camera position
                camera.set_position(rotation=camera_quaternion, translation=cpf_pos)
                # Render the scene to temporary image
                img = scene.render(use_gpu=True, samples=1)
                cv2.imshow("img", img)
                cv2.waitKey(20)
                
                # Frames have transparent background; perform an alpha blending with white background instead
                img_white_bkg = blend_with_background(img, (1.0, 1.0, 1.0))
                # Add the frame to the video
                vw.write(img_white_bkg)

        print("Visualization complete. Frames saved as 'frame_i.png'.")
        return

T = TypeVar("T")

def _collate_dataclass(batch: list[T]) -> T:
    """Collate function that works for dataclasses."""
    keys = vars(batch[0]).keys()
    return type(batch[-1])(
        **{k: torch.stack([getattr(b, k) for b in batch]) for k in keys}
    )

def collate_dataclass(batch: list[T]) -> T:
    """Collate function that works for dataclasses with optional prev_window."""
    if not batch:
        return None
    
    keys = vars(batch[0]).keys()
    result = {}
    
    for k in keys:
        # Get values for this key from all batch items
        values = [getattr(b, k) for b in batch]
        
        # Handle prev_window specially
        if k == "prev_window":
            # If any prev_window is None, all should be None
            if any(v is None for v in values):
                result[k] = None
            else:
                # Recursively collate prev_windows
                result[k] = collate_dataclass(values)
        else:
            # For tensor fields, stack them
            result[k] = torch.stack(values)
    
    return type(batch[0])(**result)