from __future__ import annotations


import sys
from pathlib import Path
from typing import TYPE_CHECKING
from contextlib import nullcontext

import numpy as np
import torch
from tqdm import tqdm
from egoallo.middleware.third_party.HybrIK.hybrik.models.layers.smplh.fncsmplh import (
    SE3,
)
from egoallo.middleware.third_party.HybrIK.hybrik.models.layers.smplh.fncsmplh import (
    SO3,
)
from egoallo.setup_logger import setup_logger
from egoallo.constants import SmplFamilyMetaModelZoo

import os
import cv2

import pyrender
from pyrender.trackball import Trackball

import trimesh
from egoallo.type_stubs import DenoiseTrajType

from .base_viewer import SMPLBaseViewer
from .utils import create_skeleton_point_cloud

if TYPE_CHECKING:
    from egoallo.type_stubs import DenoiseTrajType

logger = setup_logger(output=None, name=__name__)
sys.path.append(str(Path(__file__).parent.parent.parent.parent))


class SMPLViewer(SMPLBaseViewer):
    """
    SMPL model viewer with scene support.

    This class provides functionality to render SMPL body models in a 3D scene
    with proper lighting and camera setup.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def render_sequence(
        self,
        traj: "DenoiseTrajType",
        output_path: str = "output.mp4",
        online_render: bool = False,
        **kwargs,
    ):
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
        traj: "DenoiseTrajType" = traj.to(device)

        # Get transformation matrices
        T_world_root = SE3.from_rotation_and_translation(
            SO3.from_matrix(traj.R_world_root),
            traj.t_world_root,
        ).parameters()

        seq_len = traj.joints_wrt_world.shape[0]
        jnts = traj.joints_wrt_world.cpu().numpy(force=True)
        vis_masks = (
            traj.visible_joints_mask.bool().cpu().numpy(force=True)
            if traj.visible_joints_mask is not None
            else np.ones_like(jnts[..., 0], dtype=bool)
        )
        in_smplh_flag = True

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
                joints_wrt_world=_jnt[:22],
                visible_joints_mask=_vis_m[:22],
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

        traj: "DenoiseTrajType" = traj.map(
            lambda x: x.unsqueeze(0),
        )

        posed = traj.apply_to_body(
            SmplFamilyMetaModelZoo[self.smpl_family_meta_model_name]
            .load(self.smpl_family_model_basedir, gender=kwargs.get("gender", "male"))
            .to(device),
        )

        posed = posed.map(lambda x: x.squeeze(0))
        traj = traj.map(lambda x: x.squeeze(0))

        mesh = posed.lbs()

        vertices_seq = mesh.vertices.cpu().numpy(force=True)
        faces = mesh.faces.cpu().numpy(force=True)

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

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            roughnessFactor=0.8,
            alphaMode="BLEND",
            baseColorFactor=(0.7, 0.7, 0.9, 0.5),
        )
        scene = pyrender.Scene(
            bg_color=[0.0, 0.0, 0.0, 1.0],
            ambient_light=[0.3, 0.3, 0.3],
        )
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
            for i in tqdm(range(len(T_world_root))):
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
                    if True:
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
                        )  # Red points

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

    def render_list_sequences(
        self,
        traj_list: list["DenoiseTrajType"],
        output_path: str = "output.mp4",
        online_render: bool = False,
        **kwargs,
    ):
        # Choose rendering platform based on rendering mode
        if online_render:
            if "PYOPENGL_PLATFORM" in os.environ:
                del os.environ["PYOPENGL_PLATFORM"]
        else:
            os.environ["PYOPENGL_PLATFORM"] = "egl"

        # Move trajectories to CPU and validate
        device = torch.device("cpu")
        processed_trajs = []
        max_seq_len = 0

        # Process each trajectory
        for traj in traj_list:
            assert traj.R_world_root.dim() == 3, (
                "The batch size should be zero when visualizing."
            )
            assert traj.metadata.stage == "postprocessed", (
                "The trajectory should be postprocessed before visualization."
            )

            traj = traj.to(device)

            # Get transformation matrices
            T_world_root = SE3.from_rotation_and_translation(
                SO3.from_matrix(traj.R_world_root),
                traj.t_world_root,
            ).parameters()

            seq_len = traj.joints_wrt_world.shape[0]
            jnts = traj.joints_wrt_world.cpu().numpy(force=True)
            vis_masks = (
                traj.visible_joints_mask.bool().cpu().numpy(force=True)
                if traj.visible_joints_mask is not None
                else np.ones_like(jnts[..., 0], dtype=bool)
            )
            in_smplh_flag = True

            max_seq_len = max(max_seq_len, seq_len)

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
                    joints_wrt_world=_jnt[:22],
                    visible_joints_mask=_vis_m[:22],
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

            # Process SMPL model
            traj = traj.map(lambda x: x.unsqueeze(0))
            posed = traj.apply_to_body(
                SmplFamilyMetaModelZoo[self.smpl_family_meta_model_name]
                .load(
                    self.smpl_family_model_basedir,
                    gender=kwargs.get("gender", "male"),
                )
                .to(device),
            )
            posed = posed.map(lambda x: x.squeeze(0))
            traj = traj.map(lambda x: x.squeeze(0))
            mesh = posed.lbs()

            processed_trajs.append(
                {
                    "T_world_root": T_world_root,
                    "vertices_seq": mesh.vertices.cpu().numpy(force=True),
                    "faces": mesh.faces.cpu().numpy(force=True),
                    "vis_kpts_seq": vis_kpts_seq,
                    "invis_kpts_seq": invis_kpts_seq,
                },
            )

        # Set up video writer
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

        # Set up scene
        scene = pyrender.Scene(
            bg_color=[0.0, 0.0, 0.0, 1.0],
            ambient_light=[0.3, 0.3, 0.3],
        )
        camera = pyrender.PerspectiveCamera(
            yfov=np.radians(self.config.fov),
            aspectRatio=self.config.resolution[0] / self.config.resolution[1],
        )
        camera_node = scene.add(camera)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        scene.add(light, pose=np.eye(4))

        # Create materials with different colors for each trajectory
        materials = []
        base_colors = [
            (0.7, 0.7, 0.9),  # Blue-ish
            (0.9, 0.7, 0.7),  # Red-ish
            (0.7, 0.9, 0.7),  # Green-ish
            (0.9, 0.9, 0.7),  # Yellow-ish
            (0.7, 0.9, 0.9),  # Cyan-ish
            (0.9, 0.7, 0.9),  # Magenta-ish
        ]
        for i in range(len(processed_trajs)):
            color = base_colors[i % len(base_colors)]
            materials.append(
                pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.0,
                    roughnessFactor=0.8,
                    alphaMode="BLEND",
                    baseColorFactor=(*color, 0.5),
                ),
            )

        # Setup viewer
        viewer = None
        if online_render:
            import time

            viewer = pyrender.Viewer(
                scene,
                viewport_size=(self.config.resolution[0], self.config.resolution[1]),
                use_raymond_lighting=True,
                run_in_thread=True,
                show_world_axis=True,
            )
            render_fps = min(30, self.config.fps)
            frame_time = 1.0 / render_fps
        else:
            renderer = pyrender.OffscreenRenderer(
                viewport_width=self.config.resolution[0],
                viewport_height=self.config.resolution[1],
            )

        # Initialize nodes for each trajectory
        mesh_nodes = [None] * len(processed_trajs)
        vis_pc_nodes = [None] * len(processed_trajs)
        occ_pc_nodes = [None] * len(processed_trajs)

        cxt = viewer.render_lock if online_render else nullcontext()
        viewport_size = (self.config.resolution[0], self.config.resolution[1])

        try:
            for frame_idx in tqdm(range(max_seq_len)):
                # Calculate scene bounds for camera positioning
                all_positions = []
                for traj_idx, traj_data in enumerate(processed_trajs):
                    if frame_idx < len(traj_data["T_world_root"]):
                        all_positions.append(
                            traj_data["T_world_root"][frame_idx, 4:7]
                            .cpu()
                            .numpy(force=True),
                        )

                if not all_positions:
                    break

                # Calculate camera position to view all subjects
                center = np.mean(all_positions, axis=0)
                max_dist = max(np.linalg.norm(pos - center) for pos in all_positions)
                distance = max(
                    2.0,
                    max_dist * 2.5,
                )  # Ensure minimum distance and scale based on spread
                height = distance * 0.5

                # Calculate camera position
                angle = frame_idx * 0.05
                cam_x = center[0] + distance * np.sin(angle)
                cam_y = center[1] + distance * np.cos(angle)
                cam_z = center[2] + height

                # Create camera pose matrix
                cam_pos = np.array([cam_x, cam_y, cam_z])
                forward = center - cam_pos
                forward = forward / np.linalg.norm(forward)
                right = np.cross(forward, [0, 0, 1])
                right = right / np.linalg.norm(right)
                up = np.cross(right, forward)

                T_world_cam = np.eye(4)
                T_world_cam[:3, 0] = right
                T_world_cam[:3, 1] = up
                T_world_cam[:3, 2] = -forward
                T_world_cam[:3, 3] = cam_pos

                with cxt:
                    # Update meshes and point clouds for each trajectory
                    for traj_idx, traj_data in enumerate(processed_trajs):
                        if frame_idx >= len(traj_data["T_world_root"]):
                            continue

                        # Update mesh
                        mesh_trimesh = trimesh.Trimesh(
                            vertices=traj_data["vertices_seq"][frame_idx],
                            faces=traj_data["faces"],
                            process=False,
                        )
                        mesh_pyrender = pyrender.Mesh.from_trimesh(
                            mesh_trimesh,
                            material=materials[traj_idx],
                        )

                        if mesh_nodes[traj_idx] is not None:
                            scene.remove_node(mesh_nodes[traj_idx])
                        mesh_nodes[traj_idx] = scene.add(mesh_pyrender)

                        # Update keypoints if available
                        if frame_idx < len(traj_data["vis_kpts_seq"]):
                            # Visible keypoints
                            points = np.array(
                                traj_data["vis_kpts_seq"][frame_idx]["vertices"],
                            )
                            colors = np.ones((len(points), 3)) * np.array([0, 1.0, 0])
                            pc = pyrender.Mesh.from_points(points, colors)

                            if vis_pc_nodes[traj_idx] is not None:
                                scene.remove_node(vis_pc_nodes[traj_idx])
                            vis_pc_nodes[traj_idx] = scene.add(pc)

                            # Invisible keypoints
                            points = np.array(
                                traj_data["invis_kpts_seq"][frame_idx]["vertices"],
                            )
                            colors = np.ones((len(points), 3)) * np.array([1.0, 0, 0])
                            pc = pyrender.Mesh.from_points(points, colors)

                            if occ_pc_nodes[traj_idx] is not None:
                                scene.remove_node(occ_pc_nodes[traj_idx])
                            occ_pc_nodes[traj_idx] = scene.add(pc)

                if online_render:
                    time.sleep(frame_time)
                    viewer._trackball = Trackball(T_world_cam, viewport_size, 1.0)
                    viewer._trackball._scale = 1500.0

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

            if online_render and viewer.is_active:
                print(
                    "Rendering complete. Interactive viewer is still open. Close the window to finish.",
                )
                while viewer.is_active:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("Rendering interrupted by user.")

        finally:
            if video_writer is not None:
                video_writer.release()

            if not online_render and "renderer" in locals() and renderer is not None:
                try:
                    renderer.delete()
                except Exception:
                    pass

            if online_render and viewer is not None:
                viewer.close_external()

            if output_path:
                print(f"Video saved to {output_path}")
