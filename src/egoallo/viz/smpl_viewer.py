from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import PIL
import torch
import trimesh
from torch import Tensor
from tqdm import tqdm
from videoio import VideoWriter

from egoallo.fncsmpl import (
    SE3,
    SO3,
    SmplhModel,
    SmplhShaped,
    SmplhShapedAndPosed,
    SmplMesh,
)
from egoallo.fncsmpl_extensions import get_T_world_cpf

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from egoallo.data.dataclass import EgoTrainingData
    from egoallo.fncsmpl import SmplhModel
    from egoallo.network import AbsoluteDenoiseTraj, VelocityDenoiseTraj, JointsOnlyTraj

# On some systems, EGL does not start properly if OpenGL was already initialized, that's why it's better
# to keep EGLContext import on top
from third_party.cloudrender.cloudrender.libegl import EGLContext

sys.path.append(str(Path(__file__).parent.parent.parent.parent))


import time

from egoallo.fncsmpl import SmplhShapedAndPosed, SmplMesh

VIZ_UTILS_IMPORT = True
try:
    from third_party.cloudrender.cloudrender.camera import PerspectiveCameraModel
    from third_party.cloudrender.cloudrender.capturing import AsyncPBOCapture
    from third_party.cloudrender.cloudrender.render import (
        DirectionalLight,
        SimplePointcloud,
    )
    from third_party.cloudrender.cloudrender.scene import Scene
    from third_party.cloudrender.cloudrender.utils import trimesh_load_from_zip
    from third_party.cloudrender.cloudrender.camera.trajectory import (
        Trajectory,
    )  # pragma:
    from third_party.cloudrender.cloudrender.render.smpl_legacy import (
        AnimatableSMPLModel,
    )

except ImportError:  # pragma: no cover
    VIZ_UTILS_IMPORT = False


import json

from .utils import blend_with_background

logger = logging.getLogger(__name__)

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from OpenGL import GL as gl

from egoallo.utils.setup_logger import setup_logger

logger = setup_logger(output=None, name=__name__)


@dataclass
class RendererConfig:
    """Configuration for the renderer."""

    resolution: Tuple[int, int] = (1280, 720)
    fps: float = 30.0
    fov: float = 75.0


class BaseRenderer:
    """Base class for OpenGL/EGL rendering setup."""

    def __init__(self, config: RendererConfig = RendererConfig()):
        self.config = config
        self.context = None
        self._setup_context()
        self._setup_buffers()
        self._configure_gl()

    def _setup_context(self):
        """Initialize EGL context."""
        logger.info("Initializing EGL and OpenGL")
        self.context = EGLContext()
        if not self.context.initialize(*self.config.resolution):
            raise RuntimeError("Failed to initialize EGL context")
        # Test OpenGL context

        version = gl.glGetString(gl.GL_VERSION)
        logger.info(f"OpenGL version: {version}")

    def _setup_buffers(self):
        """Set up OpenGL frame and render buffers."""
        self._main_cb, self._main_db = gl.glGenRenderbuffers(2)

        # Color buffer
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._main_cb)
        gl.glRenderbufferStorage(
            gl.GL_RENDERBUFFER, gl.GL_RGBA, *self.config.resolution
        )

        # Depth buffer
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._main_db)
        gl.glRenderbufferStorage(
            gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24, *self.config.resolution
        )

        # Frame buffer
        self._main_fb = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self._main_fb)
        gl.glFramebufferRenderbuffer(
            gl.GL_DRAW_FRAMEBUFFER,
            gl.GL_COLOR_ATTACHMENT0,
            gl.GL_RENDERBUFFER,
            self._main_cb,
        )
        gl.glFramebufferRenderbuffer(
            gl.GL_DRAW_FRAMEBUFFER,
            gl.GL_DEPTH_ATTACHMENT,
            gl.GL_RENDERBUFFER,
            self._main_db,
        )

        gl.glDrawBuffers([gl.GL_COLOR_ATTACHMENT0])

    def _configure_gl(self):
        """Configure OpenGL settings."""
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glClearColor(1.0, 1.0, 1.0, 0)
        gl.glViewport(0, 0, *self.config.resolution)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDepthFunc(gl.GL_LESS)
        gl.glDepthRange(0.0, 1.0)


class SMPLViewer(BaseRenderer):
    """
    SMPL model viewer with scene support.

    This class provides functionality to render SMPL body models in a 3D scene
    with proper lighting and camera setup.
    """

    def __init__(
        self,
        config: Optional[RendererConfig] = None,
        scene_path: Optional[Path] = None,
    ):
        """
        Initialize the SMPL viewer.

        Args:
            config: Renderer configuration for resolution, FPS, and FOV
            scene_path: Optional path to scene mesh file
        """
        super().__init__(config or RendererConfig())

        # Initialize scene components
        self.scene = Scene()
        self.scene_path = scene_path or Path(
            "./assets/cloudrender/test_assets/MPI_Etage6.zip"
        )

        # Initialize instance variables
        self.camera: Optional[PerspectiveCameraModel] = None
        self.pointcloud: Optional[SimplePointcloud] = None
        self.smpl_renderer: Optional[AnimatableSMPLModel] = None
        self.shadow_map = None

        # Setup rendering components
        self._setup_camera()
        self._setup_scene()
        self._setup_lighting()
        self._setup_smpl_renderer()

    def _setup_camera(self) -> None:
        """Initialize camera with perspective projection."""
        self.camera = PerspectiveCameraModel()
        self.camera.init_intrinsics(self.config.resolution, fov=self.config.fov, far=50)
        # Set initial camera position - combine rotation and translation into a single pose array
        self.camera.init_extrinsics(
            quat=np.array([1, np.pi / 5, 0, 0]),
            pose=np.array([0, -1, 2]),
        )

    def _setup_scene(self) -> None:
        """Set up the scene with background pointcloud."""
        # Initialize pointcloud renderer
        self.pointcloud = SimplePointcloud(camera=self.camera)
        self.pointcloud.generate_shadows = False
        self.pointcloud.init_context()

        # Load scene mesh if available
        pointcloud = trimesh_load_from_zip(str(self.scene_path), "*/pointcloud.ply")
        self.pointcloud.set_buffers(pointcloud)
        self.scene.add_object(self.pointcloud)
        logger.info(f"Loaded scene mesh from {self.scene_path}")

    def _setup_lighting(self) -> None:
        """Set up scene lighting with shadows."""
        # Create directional light from above-front
        self.light = DirectionalLight(
            direction=np.array([0.5, -1.0, -0.5]),  # Angled from above-front
            intensity=np.array([0.8, 0.8, 0.8]),
        )

        # Larger shadow map for better coverage
        self.shadow_map = self.scene.add_dirlight_with_shadow(
            light=self.light,
            shadowmap_texsize=(2048, 2048),  # Increased resolution
            shadowmap_worldsize=(6.0, 6.0, 12.0),  # Larger area
            shadowmap_center=np.array([0.0, 0.0, 1.0]).tolist(),  # Centered on subject
        )

    def _setup_smpl_renderer(self) -> None:
        """Initialize SMPL renderer."""
        self.smpl_renderer = AnimatableSMPLModel(
            camera=self.camera,
            gender="male",  # Can be parameterized if needed
            smpl_root="./assets/smpl_based_model",  # Update path as needed
            model_type="smplh",
        )
        self.smpl_renderer.draw_shadows = False
        self.smpl_renderer.init_context()

    def render_sequence(
        self,
        denoised_traj: Union[AbsoluteDenoiseTraj, VelocityDenoiseTraj, JointsOnlyTraj],
        body_model: SmplhModel,
        output_path: str = "output.mp4",
    ) -> None:
        """Render SMPL sequence to video using denoised trajectory data."""
        assert (
            denoised_traj.R_world_root.dim() == 3
        ), "The batch size should be zero when visualizing."
        device = body_model.weights.device

        # Prepare SMPL sequence
        # denoised_traj = denoised_traj

        T_world_root = SE3.from_rotation_and_translation(
            SO3.from_matrix(denoised_traj.R_world_root),
            denoised_traj.t_world_root,
        ).parameters()

        denoised_traj = denoised_traj.map(
            lambda x: x.unsqueeze(0)
        )  # prepend a new axis to incorporate changes in `apply_to_body` function.
        posed = denoised_traj.apply_to_body(body_model)
        posed = posed.map(
            lambda x: x.squeeze(0)
        )  # remove the first dim as a compensation for the denoised_traj unsqueeze operation.
        denoised_traj = denoised_traj.map(lambda x: x.squeeze(0))  # restore the state
        global_root_orient_aa = SO3(posed.T_world_root[..., :4]).log()
        pose = torch.cat(
            [
                global_root_orient_aa,
                SO3(posed.local_quats[..., :23, :])
                .log()
                .reshape(*posed.local_quats.shape[:-2], -1),
            ],
            dim=-1,
        )

        mesh = posed.lbs()
        T_world_cpf = SE3(get_T_world_cpf(mesh))
        # Convert SMPL data to sequence
        sequence = []
        for i in range(T_world_root.shape[0]):
            sequence.append(
                {
                    "pose": pose[i].cpu().numpy(),
                    "shape": denoised_traj.betas[0, :10].cpu().numpy(),
                    "translation": T_world_root[i, 4:].cpu().numpy(),
                }
            )

        # Setup SMPL renderer
        # import ipdb; ipdb.set_trace()
        self.smpl_renderer.set_sequence(
            sequence, default_frame_time=1 / self.config.fps
        )
        self.smpl_renderer.set_material(0.3, 1, 0, 0)
        self.scene.add_object(self.smpl_renderer)

        # Camera looks from behind and slightly above
        camera_offset = SE3.from_rotation_and_translation(
            rotation=SO3.from_rpy_radians(
                roll=torch.tensor(0.0),
                pitch=torch.tensor(-0.3),  # Look down more
                yaw=torch.tensor(0.0),
            ),
            translation=torch.tensor([0.0, 0.3, 2.5]),  # Further back, slightly higher
        )

        # Render frames
        with (
            VideoWriter(
                output_path, resolution=self.config.resolution, fps=self.config.fps
            ) as vw,
            AsyncPBOCapture(self.config.resolution, queue_size=40) as capturing,
        ):
            for i in tqdm(range(T_world_root.shape[0] - 1), desc="Rendering frames"):
                # Get current camera pose from ego_data
                T_world_cam = T_world_cpf @ camera_offset

                # Update camera and render frame
                self._render_frame(
                    frame_idx=i + 1,
                    device=device,
                    video_writer=vw,
                    capture=capturing,
                    camera_pose=T_world_cam,
                )

            self._flush_remaining_frames(video_writer=vw, capture=capturing)

    def _render_frame(
        self,
        frame_idx: int,
        device: torch.device,
        video_writer: VideoWriter,
        capture: AsyncPBOCapture,
        camera_pose: SE3,
    ) -> None:
        """Render a single frame with camera pose and shadow updates."""
        # Update SMPL frame
        self.smpl_renderer.set_current_frame(frame_idx)
        current_smpl_params = self.smpl_renderer.params_sequence[
            self.smpl_renderer.current_sequence_frame_ind
        ]
        cur_smpl_trans, cur_smpl_shape, cur_smpl_pose = (
            current_smpl_params["translation"],
            current_smpl_params["shape"],
            current_smpl_params["pose"],
        )

        # Calculate camera position relative to SMPL model
        # Keep camera at fixed distance and height from model
        distance = 2.0  # Distance from model
        height = 0.8  # Height above model
        angle = frame_idx * 0.01  # Rotate around model over time

        # Calculate camera position in world coordinates
        smpl_pos = torch.from_numpy(cur_smpl_trans).float()
        camera_offset = torch.tensor(
            [distance * np.cos(angle), distance * np.sin(angle), height]
        )
        camera_pos = smpl_pos + camera_offset

        # Make camera look at SMPL model
        look_dir = smpl_pos - camera_pos
        look_dir = look_dir / torch.norm(look_dir)

        # Calculate up vector (always pointing up in z direction)
        up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)

        # Calculate right vector
        right = torch.cross(look_dir, up, dim=-1)
        right = right / torch.norm(right)

        # Recalculate up to ensure orthogonality
        up = torch.cross(right, look_dir)

        # Create rotation matrix
        rot_matrix = torch.stack([right, up, -look_dir], dim=1)

        # Convert to SO3 and SE3
        rotation = SO3.from_matrix(rot_matrix)
        transform = SE3.from_rotation_and_translation(rotation, camera_pos)

        # Apply transform to camera
        pose = transform.translation().numpy(force=True)
        quat = transform.rotation().wxyz.numpy(force=True)
        self.camera.init_extrinsics(quat, pose)

        # Update shadow map position with higher elevation
        assert self.shadow_map is not None
        smpl_translation = self.smpl_renderer.translation_params.cpu().numpy()
        shadow_offset = -self.light.direction * 5  # Increased distance
        shadow_height_offset = np.array([0.0, 0.0, 2.0])  # Raise shadow camera
        self.shadow_map.camera.init_extrinsics(
            pose=smpl_translation + shadow_offset + shadow_height_offset
        )

        # Clear buffers and render
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        self.scene.draw()

        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self._main_fb)
        gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)

        # Capture and write frame
        color = capture.request_color_async()
        if color is not None:
            color = blend_with_background(color, (1.0, 1.0, 1.0))
            video_writer.write(color)

    def _flush_remaining_frames(
        self, video_writer: VideoWriter, capture: AsyncPBOCapture
    ) -> None:
        """Flush any remaining frames in the capture queue."""
        # Flush the remaining frames
        logger.info("Flushing PBO queue")
        color = capture.get_first_requested_color()
        while color is not None:
            video_writer.write(color)
            color = capture.get_first_requested_color()


def visualize_ego_training_data(
    denoised_traj: Union[AbsoluteDenoiseTraj, VelocityDenoiseTraj, JointsOnlyTraj],
    body_model: SmplhModel,
    output_path: str = "output.mp4",
) -> None:
    """
    Main visualization function for denoised trajectories.

    Args:
        denoised_traj: Denoised trajectory containing pose sequences
        body_model: SMPL body model for mesh generation
        output_path: Path to save the output video
    """
    viewer = SMPLViewer()
    viewer.render_sequence(denoised_traj, body_model, output_path)
