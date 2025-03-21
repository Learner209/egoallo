from __future__ import annotations

import time
from functools import reduce
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np
import torch
import open3d as o3d
import cv2
from egoallo.fncsmpl_library import SE3, SO3
from egoallo.setup_logger import setup_logger
from egoallo.types import DenoiseTrajType
import egoallo.fncsmpl_library as fncsmpl
from .utils import create_skeleton_point_cloud

logger = setup_logger(output=None, name=__name__)


@dataclass
class RendererConfig:
    """Configuration for the renderer."""

    resolution: Tuple[int, int] = (1280, 720)
    fps: float = 30.0
    fov: float = 75.0
    use_blending: bool = True


class SMPLViewer:
    """
    SMPL model viewer with scene support using Open3D.

    This class provides functionality to render SMPL body models in a 3D scene
    with proper lighting and camera setup.
    """

    def __init__(self, config: Optional[RendererConfig] = None):
        """
        Initialize the SMPL viewer.

        Args:
            config: Renderer configuration for resolution, FPS, and FOV
        """
        self.config = config or RendererConfig()

    def render_sequence(
        self,
        traj: "DenoiseTrajType",
        smplh_model_path: Path,
        output_path: str = "output.mp4",
        online_render: bool = False,
    ):
        """Render SMPL sequence to video using Open3D.

        Args:
            traj: Denoised trajectory data
            smplh_model_path: Path to SMPL-H model
            output_path: Path to save the output video
            online_render: If True, use Open3D's interactive viewer for real-time rendering
        """
        # Check trajectory format
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

        batch_size = reduce(lambda x, y: x * y, traj.betas.shape[:-1])

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
            _jnt = jnts[i]
            _vis_m = vis_masks[i]

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

        traj = traj.map(
            lambda x: x.unsqueeze(0),
        )

        posed = traj.apply_to_body(
            fncsmpl.SmplhModel.load(
                smplh_model_path,
                use_pca=False,
                batch_size=batch_size,
            ).to(device),
        )

        posed = posed.map(lambda x: x.squeeze(0))
        traj = traj.map(lambda x: x.squeeze(0))

        mesh = posed.lbs()
        vertices_seq = mesh.vertices.cpu().numpy(force=True)
        faces = mesh.faces.cpu().numpy(force=True)

        # Setup video writer
        video_writer = None
        if output_path:
            try:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            except AttributeError:
                try:
                    fourcc = cv2.cv.CV_FOURCC(*"mp4v")
                except AttributeError:
                    fourcc = 0x7634706D

            video_writer = cv2.VideoWriter(
                output_path,
                fourcc,
                self.config.fps,
                (self.config.resolution[0], self.config.resolution[1]),
            )

        vis = o3d.visualization.Visualizer()
        vis.create_window(
            width=self.config.resolution[0],
            height=self.config.resolution[1],
            visible=online_render,
        )

        # Setup camera
        ctr = vis.get_view_control()
        cam_params = ctr.convert_to_pinhole_camera_parameters()

        # Create mesh and point cloud geometries
        mesh_geo = o3d.geometry.TriangleMesh()
        vis_pc = o3d.geometry.PointCloud()
        invis_pc = o3d.geometry.PointCloud()

        # Add coordinate frame at origin
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)

        # Add geometries to scene
        vis.add_geometry(mesh_geo)
        vis.add_geometry(vis_pc)
        vis.add_geometry(invis_pc)
        vis.add_geometry(coordinate_frame)

        try:
            for i in range(len(T_world_root)):
                # Update mesh
                mesh_geo.vertices = o3d.utility.Vector3dVector(vertices_seq[i])
                mesh_geo.triangles = o3d.utility.Vector3iVector(faces)
                mesh_geo.compute_vertex_normals()
                mesh_geo.paint_uniform_color([0.7, 0.7, 0.9])

                # Update point clouds
                if i < len(vis_kpts_seq):
                    vis_points = vis_kpts_seq[i]["vertices"]
                    vis_colors = vis_kpts_seq[i]["colors"]
                    if len(vis_points) > 0:
                        vis_pc.points = o3d.utility.Vector3dVector(vis_points)
                        vis_pc.colors = o3d.utility.Vector3dVector(vis_colors)

                    invis_points = invis_kpts_seq[i]["vertices"]
                    invis_colors = invis_kpts_seq[i]["colors"]
                    if len(invis_points) > 0:
                        invis_pc.points = o3d.utility.Vector3dVector(invis_points)
                        invis_pc.colors = o3d.utility.Vector3dVector(invis_colors)

                # Update camera position
                angle = i * 0.05
                distance = 1.0
                height = 1.0
                subject_pos = T_world_root[i, 4:7].cpu().numpy(force=True)

                cam_x = subject_pos[0] + distance * np.sin(angle)
                cam_y = subject_pos[1] + distance * np.cos(angle)
                cam_z = subject_pos[2] + height

                cam_pos = np.array([cam_x, cam_y, cam_z])
                forward = subject_pos - cam_pos
                forward = forward / np.linalg.norm(forward)

                right = np.cross(forward, [0, 0, 1])
                right = right / np.linalg.norm(right)
                up = np.cross(right, forward)

                T_world_cam = np.eye(4)
                T_world_cam[:3, 0] = right
                T_world_cam[:3, 1] = -up
                T_world_cam[:3, 2] = forward
                T_world_cam[:3, 3] = cam_pos

                cam_params.extrinsic = np.linalg.inv(T_world_cam)

                ctr.convert_from_pinhole_camera_parameters(cam_params)

                # Update visualization
                vis.update_geometry(mesh_geo)
                vis.update_geometry(vis_pc)
                vis.update_geometry(invis_pc)
                vis.poll_events()
                vis.update_renderer()

                if online_render:
                    # For interactive viewing, control framerate
                    if i % 10 == 0:
                        print(f"Rendering frame {i + 1}/{len(T_world_root)}")
                    time.sleep(0.1)
                    continue

                # Capture frame for video
                if video_writer is not None:
                    img = np.asarray(vis.capture_screen_float_buffer()) * 255
                    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
                    video_writer.write(img)

                if i % 10 == 0:
                    print(f"Rendering frame {i + 1}/{len(T_world_root)}")

        finally:
            if video_writer is not None:
                video_writer.release()
            vis.destroy_window()
