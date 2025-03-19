from __future__ import annotations


import sys
from pathlib import Path
from typing import Optional
from typing import TYPE_CHECKING
from typing import Tuple
from dataclasses import dataclass
from functools import reduce
from contextlib import nullcontext

import numpy as np
import torch
from egoallo.fncsmpl_library import SE3
from egoallo.fncsmpl_library import SO3
from egoallo.fncsmpl_extensions_library import get_T_world_cpf
from egoallo.setup_logger import setup_logger

import os
import cv2

import pyrender
from pyrender.trackball import Trackball

import trimesh
from egoallo.types import DenoiseTrajType

import egoallo.fncsmpl_library as fncsmpl
from .utils import create_skeleton_point_cloud

if TYPE_CHECKING:
    from egoallo.types import DenoiseTrajType

logger = setup_logger(output=None, name=__name__)
sys.path.append(str(Path(__file__).parent.parent.parent.parent))


@dataclass
class RendererConfig:
    """Configuration for the renderer."""

    resolution: Tuple[int, int] = (1280, 720)
    fps: float = 30.0
    fov: float = 75.0
    use_blending: bool = True


class SMPLViewer:
    """
    SMPL model viewer with scene support.

    This class provides functionality to render SMPL body models in a 3D scene
    with proper lighting and camera setup.
    """

    def __init__(
        self,
        config: Optional[RendererConfig] = None,
    ):
        """
        Initialize the SMPL viewer.

        Args:
            config: Renderer configuration for resolution, FPS, and FOV
            scene_path: Optional path to scene mesh file
        """
        self.config = config or RendererConfig()

    def render_sequence(
        self,
        traj: "DenoiseTrajType",
        smplh_model_path: Path,
        output_path: str = "output.mp4",
        online_render: bool = False,
    ):
        """Render SMPL sequence to video using pyrender (lightweight alternative to OpenGL).

        Args:
            traj: Denoised trajectory data
            smplh_model_path: Path to SMPL-H model
            output_path: Path to save the output video
            online_render: If True, use pyrender's interactive viewer for real-time rendering
        """

        # Choose rendering platform based on rendering mode
        if online_render:
            # For online rendering with a GUI, use the default platform
            if "PYOPENGL_PLATFORM" in os.environ:
                del os.environ["PYOPENGL_PLATFORM"]
        else:
            # For offscreen rendering, use EGL
            os.environ["PYOPENGL_PLATFORM"] = "egl"

        # Check that the trajectory is in the right format
        assert traj.R_world_root.dim() == 3, (
            "The batch size should be zero when visualizing."
        )
        assert traj.metadata.stage == "postprocessed", (
            "The trajectory should be postprocessed before visualization."
        )

        # Move trajectory to CPU
        device = torch.device("cpu")
        traj = traj.to(device)

        # Get transformation matrices
        T_world_root = SE3.from_rotation_and_translation(
            SO3.from_matrix(traj.R_world_root),
            traj.t_world_root,
        ).parameters()

        # Process SMPL trajectory data
        traj = traj.map(
            lambda x: x.unsqueeze(0),
        )  # prepend a new axis for apply_to_body
        batch_size = reduce(lambda x, y: x * y, traj.betas.shape[:-1])

        # Apply trajectory to body model
        posed = traj.apply_to_body(
            fncsmpl.SmplhModel.load(
                smplh_model_path,
                use_pca=False,
                batch_size=batch_size,
            ).to(device),
        )

        # Cleanup dimensions
        posed = posed.map(lambda x: x.squeeze(0))
        traj = traj.map(lambda x: x.squeeze(0))

        # Extract pose parameters
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

        # Get mesh data
        mesh = posed.lbs()
        vertices_seq = mesh.vertices.cpu().numpy(force=True)
        faces = mesh.faces.cpu().numpy(force=True)

        # Get camera transform
        T_world_cpf = SE3(get_T_world_cpf(mesh))

        # Get keypoints data
        if traj.metadata.dataset_type in ("AriaDataset", "EgoExoDataset"):
            seq_len = traj.metadata.aux_joints_wrt_world_placeholder.shape[1]
            jnts = (
                traj.metadata.aux_joints_wrt_world_placeholder[0, :, :]
                .cpu()
                .numpy(force=True)
            )
            vis_masks = (
                traj.metadata.aux_visible_joints_mask_placeholder[0, :]
                .cpu()
                .numpy(force=True)
                if traj.metadata.aux_visible_joints_mask_placeholder is not None
                else np.ones_like(jnts[..., 0], dtype=bool)
            )
            in_smplh_flag = False
        elif traj.metadata.dataset_type in (
            "AdaptiveAmassHdf5Dataset",
            "VanillaAmassHdf5Dataset",
        ):
            seq_len = traj.joints_wrt_world.shape[0]
            jnts = traj.joints_wrt_world.cpu().numpy(force=True)
            vis_masks = (
                traj.visible_joints_mask.cpu().numpy(force=True)
                if traj.visible_joints_mask is not None
                else np.ones_like(jnts[..., 0], dtype=bool)
            )
            in_smplh_flag = True
        else:
            raise ValueError(f"Unknown dataset type: {traj.metadata.dataset_type}")

        # Create keypoint visualization data
        vis_kpts_seq = []
        invis_kpts_seq = []
        for i in range(seq_len):
            # Get joints and visibility mask for this frame
            _jnt = jnts[i]  # [J, 3]
            _vis_m = vis_masks[i]  # [J]

            # Create skeleton point cloud
            (
                (visible_skeleton_points, visible_skeleton_colors),
                (invisible_skeleton_points, invisible_skeleton_colors),
            ) = create_skeleton_point_cloud(
                joints_wrt_world=_jnt,
                visible_joints_mask=_vis_m,
                input_smplh=in_smplh_flag,
                num_samples_per_bone=50,
                return_colors=True,
            )

            vis_kpts_seq.append(
                {
                    "vertices": visible_skeleton_points,
                    "colors": visible_skeleton_colors,
                },
            )
            invis_kpts_seq.append(
                {
                    "vertices": invisible_skeleton_points,
                    "colors": invisible_skeleton_colors,
                },
            )

        # Setup video writer (only if saving to disk)
        video_writer = None
        if output_path:
            # Handle different OpenCV versions
            try:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            except AttributeError:
                try:
                    fourcc = cv2.cv.CV_FOURCC(*"mp4v")
                except AttributeError:
                    # Fallback to a simple integer code for mp4v
                    fourcc = 0x7634706D  # mp4v in hex

            video_writer = cv2.VideoWriter(
                output_path,
                fourcc,
                self.config.fps,
                (self.config.resolution[0], self.config.resolution[1]),
            )

        # Material for SMPL mesh
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            roughnessFactor=0.8,
            alphaMode="BLEND",
            baseColorFactor=(0.7, 0.7, 0.9, 0.5),
        )
        # First create the scene for both rendering methods
        scene = pyrender.Scene(
            bg_color=[0.0, 0.0, 0.0, 1.0],
            ambient_light=[0.3, 0.3, 0.3],
        )
        # Create camera
        camera = pyrender.PerspectiveCamera(
            yfov=np.radians(self.config.fov),
            aspectRatio=self.config.resolution[0] / self.config.resolution[1],
        )
        camera_node = scene.add(camera)

        # Add lighting
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        scene.add(light, pose=np.eye(4))

        # Setup the interactive viewer if requested
        viewer = None
        if online_render:
            # Create interactive viewer with our scene
            import time

            viewer = pyrender.Viewer(
                scene,
                viewport_size=(self.config.resolution[0], self.config.resolution[1]),
                use_raymond_lighting=True,
                run_in_thread=True,
                show_world_axis=True,
            )

            # Set a slower viewing fps for interactive viewing
            render_fps = min(
                30,
                self.config.fps,
            )  # Cap at 30fps for smooth interactive viewing
            frame_time = 1.0 / render_fps
        else:
            # Setup offscreen renderer
            renderer = pyrender.OffscreenRenderer(
                viewport_width=self.config.resolution[0],
                viewport_height=self.config.resolution[1],
            )

        mesh_node = None
        vis_pc_node = None
        occ_pc_node = None

        cxt = viewer.render_lock if online_render else nullcontext()
        viewport_size = (self.config.resolution[0], self.config.resolution[1])

        try:
            for i in range(len(T_world_root)):
                # Create mesh for this frame
                mesh_trimesh = trimesh.Trimesh(
                    vertices=vertices_seq[i],
                    faces=faces,
                    process=False,
                )
                mesh_pyrender = pyrender.Mesh.from_trimesh(
                    mesh_trimesh,
                    material=material,
                )

                # Handle keypoints for this frame if available
                keypoints_ready = (
                    i < len(vis_kpts_seq) and len(vis_kpts_seq[i]["vertices"]) > 0
                )

                # Position camera
                angle = i * 0.05  # Rotate around model over time
                distance = 1.0
                height = 1.0

                # Get subject position from current pose
                subject_pos = T_world_root[i, 4:7].cpu().numpy(force=True)

                # Calculate camera position
                cam_x = subject_pos[0] + distance * np.sin(angle)
                cam_y = subject_pos[1] + distance * np.cos(angle)
                cam_z = subject_pos[2] + height

                # Create camera pose matrix (look at subject)
                cam_pos = np.array([cam_x, cam_y, cam_z])
                forward = subject_pos - cam_pos
                forward = forward / np.linalg.norm(forward)

                # Calculate camera orientation
                right = np.cross(forward, [0, 0, 1])
                right = right / np.linalg.norm(right)
                up = np.cross(right, forward)

                # Build camera pose matrix
                T_world_cam = np.eye(4)
                T_world_cam[:3, 0] = right
                T_world_cam[:3, 1] = up
                T_world_cam[:3, 2] = -forward
                T_world_cam[:3, 3] = cam_pos

                with cxt:
                    # Update or add mesh
                    if mesh_node is None:
                        mesh_node = scene.add(mesh_pyrender)
                    else:
                        scene.remove_node(mesh_node)
                        mesh_node = scene.add(mesh_pyrender)

                    # Update or add keypoints
                    if keypoints_ready:
                        points = np.array(vis_kpts_seq[i]["vertices"])
                        colors = np.ones((len(points), 3)) * np.array(
                            [0, 1.0, 0],
                        )  # Green points

                        pc = pyrender.Mesh.from_points(points, colors)
                        if vis_pc_node is None:
                            vis_pc_node = scene.add(pc)
                        else:
                            scene.remove_node(vis_pc_node)
                            vis_pc_node = scene.add(pc)

                        points = np.array(invis_kpts_seq[i]["vertices"])
                        colors = np.ones((len(points), 3)) * np.array(
                            [1.0, 0, 0],
                        )  # Green points

                        pc = pyrender.Mesh.from_points(points, colors)
                        if occ_pc_node is None:
                            occ_pc_node = scene.add(pc)
                        else:
                            scene.remove_node(occ_pc_node)
                            occ_pc_node = scene.add(pc)

                # If using online rendering with the viewer
                if online_render:
                    # For interactive viewing, control the framerate
                    time.sleep(frame_time)

                    # https://github.com/mmatl/pyrender/issues/165
                    viewer._trackball = Trackball(T_world_cam, viewport_size, 1.0)
                    viewer._trackball._scale = 1500.0

                    # Check if viewer is still active
                    if not viewer.is_active:
                        print("Viewer window closed. Stopping rendering.")
                        break

                else:
                    scene.set_pose(camera_node, T_world_cam)
                    flags = pyrender.RenderFlags.RGBA
                    color, depth = renderer.render(scene, flags=flags)
                    color_bgr = cv2.cvtColor(color, cv2.COLOR_RGBA2BGR)
                    if video_writer is not None:
                        video_writer.write(color_bgr)

                # Print progress
                if i % 10 == 0:
                    print(f"Rendering frame {i + 1}/{len(T_world_root)}")

            # If using online viewer, wait for user to close the window
            if online_render and viewer.is_active:
                print(
                    "Rendering complete. Interactive viewer is still open. Close the window to finish.",
                )
                while viewer.is_active:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("Rendering interrupted by user.")

        finally:
            # Clean up
            if video_writer is not None:
                video_writer.release()

            # Clean up renderer
            if not online_render and "renderer" in locals() and renderer is not None:
                try:
                    renderer.delete()
                except Exception:
                    pass

            # Close the viewer if it was created
            if online_render and viewer is not None:
                viewer.close_external()

            if output_path:
                print(f"Video saved to {output_path}")
