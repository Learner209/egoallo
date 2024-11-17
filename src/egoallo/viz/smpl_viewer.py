from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import trimesh
from torch import Tensor
from tqdm import tqdm
from videoio import VideoWriter

import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from third_party.cloudrender.cloudrender.scene import Scene
from third_party.cloudrender.cloudrender.camera import PerspectiveCameraModel
from third_party.cloudrender.cloudrender.render import SimplePointcloud, DirectionalLight
from third_party.cloudrender.cloudrender.capturing import AsyncPBOCapture

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from egoallo.data.dataclass import EgoTrainingData
    from egoallo.fncsmpl import SmplhModel

from .renderer import BaseRenderer, RendererConfig
from .utils import blend_with_background

logger = logging.getLogger(__name__)

class SMPLViewer(BaseRenderer):
    """
    SMPL model viewer with scene support.
    
    This class provides functionality to render SMPL body models in a 3D scene
    with proper lighting and camera setup.
    """
    
    def __init__(
        self, 
        config: Optional[RendererConfig] = None,
        scene_path: Optional[Path] = None
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
        self.scene_path = scene_path or Path("./assets/scene/MPI_Etage6.ply")
        
        # Setup rendering components
        self._setup_camera()
        self._setup_scene()
        self._setup_lighting()

    def _setup_camera(self) -> None:
        """Initialize camera with perspective projection."""
        self.camera = PerspectiveCameraModel()
        self.camera.init_intrinsics(
            self.config.resolution, 
            fov=self.config.fov, 
            far=50
        )
        # Set initial camera position
        self.camera.init_extrinsics(
            pose=np.array([1, np.pi/5, 0, 0]),  # rotation
            translation=np.array([0, -1, 2])     # position
        )

    def _setup_scene(self) -> None:
        """Set up the scene with background pointcloud."""
        # Initialize pointcloud renderer
        self.pointcloud = SimplePointcloud(camera=self.camera)
        self.pointcloud.generate_shadows = False
        self.pointcloud.init_context()
        
        # Load scene mesh if available
        if self.scene_path.exists():
            try:
                pointcloud = trimesh.load(self.scene_path)
                self.pointcloud.set_buffers(pointcloud)
                self.scene.add_object(self.pointcloud)
                logger.info(f"Loaded scene mesh from {self.scene_path}")
            except Exception as e:
                logger.warning(f"Failed to load scene mesh: {e}")

    def _setup_lighting(self) -> None:
        """Set up scene lighting with shadows."""
        # Create main directional light
        light = DirectionalLight(
            direction=np.array([0., -1., -1.]),  # Light direction
            color=np.array([0.8, 0.8, 0.8])      # Light color
        )
        
        # Add shadow-casting light to scene
        self.scene.add_dirlight_with_shadow(
            light=light,
            shadowmap_texsize=(1024, 1024),      # Shadow resolution
            shadowmap_worldsize=(4., 4., 10.),   # Shadow volume size
            shadowmap_center=np.zeros(3)         # Center of shadow volume
        )

    def render_sequence(
        self, 
        ego_data: EgoTrainingData, 
        body_model: SmplhModel, 
        output_path: str = "output.mp4"
    ) -> None:
        """
        Render SMPL sequence to video.

        Args:
            ego_data: Training data containing pose sequences
            body_model: SMPL body model for mesh generation
            output_path: Path to save the output video
        """
        device = body_model.weights.device
        
        # Prepare SMPL model with shape and pose
        shaped = body_model.with_shape(ego_data.betas.to(device))
        posed = shaped.with_pose_decomposed(
            T_world_root=ego_data.T_world_root.to(device),
            body_quats=ego_data.body_quats.to(device)
        )
        smplh_mesh = posed.lbs()

        # Setup video writer and frame capture
        with VideoWriter(output_path, resolution=self.config.resolution, fps=self.config.fps) as vw, \
             AsyncPBOCapture(self.config.resolution, queue_size=50) as capturing:
            
            # Render each frame
            for i in tqdm(range(ego_data.T_world_root.shape[0]-1), desc="Rendering frames"):
                self._render_frame(
                    smplh_mesh=smplh_mesh,
                    frame_idx=i+1,
                    device=device,
                    video_writer=vw,
                    capture=capturing
                )

            # Flush remaining frames
            self._flush_remaining_frames(video_writer=vw, capture=capturing)

    def _render_frame(
        self, 
        smplh_mesh: Tensor, 
        frame_idx: int,
        device: torch.device,
        video_writer: VideoWriter,
        capture: AsyncPBOCapture
    ) -> None:
        """Render a single frame of the sequence."""
        # Calculate camera position from eye positions
        right_eye = (smplh_mesh.verts[frame_idx, 6260, :] + 
                    smplh_mesh.verts[frame_idx, 6262, :]) / 2.0
        left_eye = (smplh_mesh.verts[frame_idx, 2800, :] + 
                   smplh_mesh.verts[frame_idx, 2802, :]) / 2.0
        camera_pos = (right_eye + left_eye) / 2.0
        camera_pos[2] += 1.0  # Offset camera above head

        # Update camera position
        self.camera.set_position(
            rotation=torch.eye(3, device=device),
            translation=camera_pos.cpu().numpy()
        )

        # Render and capture frame
        self.scene.draw()
        color = capture.request_color_async()
        if color is not None:
            color = blend_with_background(color, (1.0, 1.0, 1.0))
            video_writer.write(color)

    def _flush_remaining_frames(
        self, 
        video_writer: VideoWriter,
        capture: AsyncPBOCapture
    ) -> None:
        """Flush any remaining frames in the capture queue."""
        logger.info("Flushing remaining frames")
        while (color := capture.get_first_requested_color()) is not None:
            color = blend_with_background(color, (1.0, 1.0, 1.0))
            video_writer.write(color)

def visualize_ego_training_data(
    ego_data: EgoTrainingData, 
    body_model: SmplhModel, 
    output_path: str = "output.mp4"
) -> None:
    """
    Main visualization function for EgoTrainingData.

    Args:
        ego_data: Training data containing pose sequences
        body_model: SMPL body model for mesh generation
        output_path: Path to save the output video
    """
    viewer = SMPLViewer()
    viewer.render_sequence(ego_data, body_model, output_path) 