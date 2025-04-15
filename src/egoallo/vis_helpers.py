import time
from pathlib import Path
from typing import Callable
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import torch
import trimesh
import viser
from plyfile import PlyData

from typing import Literal
from egoallo.type_stubs import DenoiseTrajType

from .hand_detection_structs import CorrespondedAriaHandWristPoseDetections
from .hand_detection_structs import CorrespondedHamerDetections
from egoallo.transforms import SE3
from egoallo.transforms import SO3


class SplatArgs(TypedDict):
    centers: npt.NDArray[np.floating]
    """(N, 3)."""
    rgbs: npt.NDArray[np.floating]
    """(N, 3). Range [0, 1]."""
    opacities: npt.NDArray[np.floating]
    """(N, 1). Range [0, 1]."""
    covariances: npt.NDArray[np.floating]
    """(N, 3, 3)."""


def load_splat_file(splat_path: Path, center: bool = False) -> SplatArgs:
    """Load an antimatter15-style splat file."""
    start_time = time.time()
    splat_buffer = splat_path.read_bytes()
    bytes_per_gaussian = (
        # Each Gaussian is serialized as:
        # - position (vec3, float32)
        3 * 4
        # - xyz (vec3, float32)
        + 3 * 4
        # - rgba (vec4, uint8)
        + 4
        # - ijkl (vec4, uint8), where 0 => -1, 255 => 1.
        + 4
    )
    assert len(splat_buffer) % bytes_per_gaussian == 0
    num_gaussians = len(splat_buffer) // bytes_per_gaussian

    # Reinterpret cast to dtypes that we want to extract.
    splat_uint8 = np.frombuffer(splat_buffer, dtype=np.uint8).reshape(
        (num_gaussians, bytes_per_gaussian),
    )
    scales = splat_uint8[:, 12:24].copy().view(np.float32)
    wxyzs = splat_uint8[:, 28:32] / 255.0 * 2.0 - 1.0
    Rs = SO3(wxyzs).as_matrix()
    covariances = np.einsum(
        "nij,njk,nlk->nil",
        Rs,
        np.eye(3)[None, :, :] * scales[:, None, :] ** 2,
        Rs,
    )
    centers = splat_uint8[:, 0:12].copy().view(np.float32)
    if center:
        centers -= np.mean(centers, axis=0, keepdims=True)
    print(
        f"Splat file with {num_gaussians=} loaded in {time.time() - start_time} seconds",
    )
    return {
        "centers": centers,
        # Colors should have shape (N, 3).
        "rgbs": splat_uint8[:, 24:27] / 255.0,
        "opacities": splat_uint8[:, 27:28] / 255.0,
        # Covariances should have shape (N, 3, 3).
        "covariances": covariances,
    }


def load_ply_file(ply_file_path: Path, center: bool = False) -> SplatArgs:
    """Load Gaussians stored in a PLY file."""
    start_time = time.time()

    SH_C0 = 0.28209479177387814

    plydata = PlyData.read(ply_file_path)
    v = plydata["vertex"]
    positions = np.stack([v["x"], v["y"], v["z"]], axis=-1)
    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1))
    wxyzs = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1)
    colors = 0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
    opacities = 1.0 / (1.0 + np.exp(-v["opacity"][:, None]))

    Rs = SO3(wxyzs).as_matrix()
    covariances = np.einsum(
        "nij,njk,nlk->nil",
        Rs,
        np.eye(3)[None, :, :] * scales[:, None, :] ** 2,
        Rs,
    )
    if center:
        positions -= np.mean(positions, axis=0, keepdims=True)

    num_gaussians = len(v)
    print(
        f"PLY file with {num_gaussians=} loaded in {time.time() - start_time} seconds",
    )
    return {
        "centers": positions,
        "rgbs": colors,
        "opacities": opacities,
        "covariances": covariances,
    }


def add_splat_to_viser(
    splat_or_ply_path: Path,
    server: viser.ViserServer,
    z_offset: float = 0.0,
) -> None:
    """Add some Gaussian splats to the Viser server."""
    if splat_or_ply_path.suffix.lower() == ".ply":
        splat_args = load_ply_file(splat_or_ply_path)
    elif splat_or_ply_path.suffix.lower() == ".splat":
        splat_args = load_splat_file(splat_or_ply_path)
    else:
        assert False
    server.scene.add_gaussian_splats(
        "/gaussian_splats",
        centers=splat_args["centers"],
        rgbs=splat_args["rgbs"],
        opacities=splat_args["opacities"],
        covariances=splat_args["covariances"],
        position=(0.0, 0.0, z_offset),
    )


def visualize_traj_and_hand_detections(
    server: viser.ViserServer,
    traj: DenoiseTrajType,
    smpl_family_model_basedir: Path,
    gender: Literal["male", "female"],
    hamer_detections: CorrespondedHamerDetections | None = None,
    aria_detections: CorrespondedAriaHandWristPoseDetections | None = None,
    points_data: np.ndarray | None = None,
    splat_path: Path | None = None,
    floor_z: float = 0.0,
    show_joints: bool = False,
    get_ego_video: Callable[[int, int, float], bytes] | None = None,
    device: torch.device = torch.device("cuda"),
) -> Callable[[], int]:
    """Visualization function for trajectories and hand detections.
    Returns a callback that should be called repeatedly in a loop."""

    assert traj.joints_wrt_world.dim() == 4, (
        "There should be only one batch size when visualizing."
    )
    assert traj.metadata.stage == "postprocessed", (
        "The trajectory should be postprocessed before visualization."
    )

    from egoallo.constants import SmplFamilyMetaModelZoo

    smpl_family_meta_model_name = "SmplModelAADecomp"
    smpl_aadecomp_model = (
        SmplFamilyMetaModelZoo[smpl_family_meta_model_name]
        .load(smpl_family_model_basedir, gender=gender, num_joints=24)
        .to(device)
    )

    posed = traj.apply_to_body(smpl_aadecomp_model)

    sample_count = traj.joints_wrt_world.shape[0]
    timesteps = posed.pose_skeleton.shape[-3]
    num_joints = posed.pose_skeleton.shape[-2]

    verts_zero, joints_zero = smpl_aadecomp_model.verts_zero_and_jts_zero(
        betas=traj.betas.mean(dim=-2),
        num_joints=num_joints,
    )
    assert verts_zero.shape == (sample_count, 6890, 3)
    assert joints_zero.shape == (sample_count, 23, 3)

    Rs_world_joint_with_root = (
        SE3(posed.Ts_world_joint_with_root).rotation().as_matrix()
    )  # (sample_count, timesteps, num_joints, 3, 3)
    assert Rs_world_joint_with_root.shape == (sample_count, timesteps, num_joints, 3, 3)

    joint_positions = posed.pose_skeleton[
        ...,
        1:,
        :,
    ]  # (sample_count, timesteps, num_joints-1, 3)
    T_world_root = SE3.from_rotation_and_translation(
        rotation=SO3.from_matrix(Rs_world_joint_with_root[..., 0, :, :]),
        translation=posed.pose_skeleton[..., 0, :3],
    )
    T_world_cpf = SE3.from_rotation_and_translation(
        rotation=SO3.from_matrix(Rs_world_joint_with_root[..., 15, :, :]),
        translation=posed.pose_skeleton[..., 15, :3],
    )

    server.scene.add_grid(
        "/ground",
        plane="xy",
        cell_color=(80, 80, 80),
        section_color=(50, 50, 50),
        position=(0.0, 0.0, floor_z),
    )

    if points_data is not None:
        point_cloud = server.scene.add_point_cloud(
            "/aria_points",
            points=points_data,
            colors=np.cos(points_data + np.arange(3)) / 3.0
            + 0.7,  # Make points colorful :)
            point_size=0.01,
            # point_size=0.1,
            point_shape="sparkle",
        )
        size_slider = server.gui.add_slider(
            "Point cloud size",
            min=0.001,
            max=0.05,
            step=0.001,
            initial_value=0.005,
        )

        @size_slider.on_update
        def _(_) -> None:
            if point_cloud is not None:
                point_cloud.point_size = size_slider.value

    if splat_path is not None:
        add_splat_to_viser(splat_path, server)  # , z_offset=-floor_z)

    glasses_mesh = trimesh.load("./data/glasses.stl")
    assert isinstance(glasses_mesh, trimesh.Trimesh)
    glasses_mesh.visual.face_colors = [10, 20, 20, 255]  # type: ignore

    cpf_handle = server.scene.add_frame(
        "/cpf",
        show_axes=True,
        axes_length=0.05,
        axes_radius=0.004,
    )
    server.scene.add_mesh_trimesh("/cpf/glasses", glasses_mesh, scale=0.001 * 1.05)

    joint_position_handles: list[viser.SceneNodeHandle] = []
    timestep_handles: list[viser.FrameHandle] = []
    hamer_handles: list[viser.MeshHandle | viser.PointCloudHandle] = []
    aria_handles: list[viser.SceneNodeHandle] = []
    for t in range(timesteps):
        timestep_handles.append(
            server.scene.add_frame(f"/timesteps/{t}", show_axes=False),
        )

        # Joints.
        if show_joints and posed is not None:
            for j in range(sample_count):
                joints_colors = np.zeros((num_joints, 3))
                joints_colors[:, 0] = traj.contacts[j, t, :].numpy(force=True)
                joints_colors[:, 2] = 1.0 - traj.contacts[j, t, :].numpy(force=True)
                joint_position_handles.append(
                    server.scene.add_point_cloud(
                        f"/timesteps/{t}/joints",
                        points=posed.pose_skeleton[j, t, :num_joints, :3].numpy(
                            force=True,
                        ),
                        colors=joints_colors,
                        point_shape="circle",
                        point_size=0.02,
                    ),
                )

        # Visualize Aria detections.
        if aria_detections is not None:
            for side in ("left", "right"):
                detections = {
                    "left": aria_detections.detections_left_concat,
                    "right": aria_detections.detections_right_concat,
                }[side]
                if detections is None:
                    continue
                indices = detections.indices
                index = torch.searchsorted(indices, t)
                if index < len(indices) and indices[index] == t:  # found?
                    aria_handles.append(
                        server.scene.add_spline_catmull_rom(
                            f"/timesteps/{t}/aria_detections/{side}",
                            np.array(
                                [
                                    detections.wrist_position[index].numpy(force=True),
                                    detections.palm_position[index].numpy(force=True),
                                ],
                            ),
                            line_width=3.0,
                            color=(255, 0, 0) if side == "left" else (0, 255, 0),
                            visible=False,
                        ),
                    )

    body_handles = (
        [
            server.scene.add_mesh_skinned(
                f"/persons/{i}",
                vertices=verts_zero[i, :, :].numpy(force=True),
                faces=smpl_aadecomp_model.model.faces_tensor.numpy(force=True),
                bone_wxyzs=SO3.identity(device=device, dtype=traj.betas.dtype)
                .wxyz.repeat(num_joints, 1)
                .numpy(force=True),
                bone_positions=np.concatenate(
                    [
                        np.zeros((1, 3)),
                        # Indices are (batch, time, joint, positions).
                        joints_zero[i, :, :].numpy(force=True),
                    ],
                    axis=0,
                ),
                color=(152, 93, 229),
                skin_weights=smpl_aadecomp_model.model.lbs_weights.numpy(
                    force=True,
                ),  # (6890, 23+1)
            )
            for i in range(sample_count)
        ]
        if posed is not None
        else []
    )

    gui_attach = server.gui.add_checkbox("Attach camera to CPF", initial_value=False)
    gui_attach_dist = server.gui.add_number("Attach distance", initial_value=0.3)
    gui_show_body = server.gui.add_checkbox("Show body", initial_value=True)
    gui_show_glasses = server.gui.add_checkbox("Show glasses", initial_value=True)
    gui_show_cpf_axes = server.gui.add_checkbox("Show CPF axes", initial_value=False)
    gui_wireframe = server.gui.add_checkbox("Wireframe", initial_value=False)
    gui_smpl_opacity = server.gui.add_slider(
        "SMPL Opacity",
        initial_value=1.0,
        min=0.0,
        max=1.0,
        step=0.01,
    )
    gui_hamer_opacity = server.gui.add_slider(
        "HaMeR Opacity",
        initial_value=1.0,
        min=0.0,
        max=1.0,
        step=0.01,
    )

    @gui_smpl_opacity.on_update
    def _(_) -> None:
        for handle in body_handles:
            handle.opacity = gui_smpl_opacity.value

    @gui_hamer_opacity.on_update
    def _(_) -> None:
        for handle in hamer_handles:
            if isinstance(handle, viser.MeshHandle):
                handle.opacity = gui_hamer_opacity.value

    gui_show_hamer_hands = server.gui.add_checkbox(
        "Show HaMeR hands",
        initial_value=False,
    )
    gui_show_aria_hands = server.gui.add_checkbox(
        "Show wrist detections",
        initial_value=False,
    )
    gui_body_color = server.gui.add_rgb("Body color", initial_value=(152, 93, 229))

    if show_joints:
        gui_show_joints = server.gui.add_checkbox("Show joints", initial_value=True)

        @gui_show_joints.on_update
        def _(_) -> None:
            for handle in joint_position_handles:
                handle.visible = gui_show_joints.value

    @gui_show_body.on_update
    def _(_) -> None:
        for handle in body_handles:
            handle.visible = gui_show_body.value

    @gui_show_glasses.on_update
    def _(_) -> None:
        # The glasses are a child of the CPF frame.
        cpf_handle.visible = gui_show_glasses.value

    @gui_show_cpf_axes.on_update
    def _(_) -> None:
        cpf_handle.show_axes = gui_show_cpf_axes.value

    @gui_wireframe.on_update
    def _(_) -> None:
        for handle in body_handles:
            handle.wireframe = gui_wireframe.value

    @gui_show_hamer_hands.on_update
    def _(_) -> None:
        for handle in hamer_handles:
            handle.visible = gui_show_hamer_hands.value

    @gui_show_aria_hands.on_update
    def _(_) -> None:
        for handle in aria_handles:
            handle.visible = gui_show_aria_hands.value

    @gui_body_color.on_update
    def _(_) -> None:
        for handle in body_handles:
            handle.color = gui_body_color.value

    # Add playback UI.
    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=timesteps - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_start_end = server.gui.add_multi_slider(
            "Start/end",
            min=0,
            max=timesteps - 1,
            initial_value=(0, timesteps - 1),
            step=1,
        )
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider(
            "FPS",
            min=1,
            max=60,
            step=0.1,
            initial_value=15,
        )
        gui_framerate_options = server.gui.add_button_group(
            "FPS options",
            ("10", "20", "30", "60"),
        )

    # Frame step buttons.
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % timesteps

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % timesteps

    # Disable frame controls when we're playing.
    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

    # Set the framerate when we click one of the options.
    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)

    def do_update() -> None:
        t = gui_timestep.value

        _wxyz = T_world_cpf.rotation().wxyz[0, t]
        cpf_handle.wxyz = _wxyz.numpy(force=True)
        cpf_handle.position = T_world_cpf.translation()[0, t].numpy(force=True)

        if gui_attach.value:
            # buggy.
            for client in server.get_clients().values():
                client.camera.wxyz = (
                    SO3(_wxyz) @ SO3.from_z_radians(torch.tensor(np.pi).to(device))
                ).wxyz.numpy(force=True)
                client.camera.position = cpf_handle.position - SO3(
                    _wxyz,
                ).as_matrix().numpy(force=True) @ np.array(
                    [0.0, 0.0, gui_attach_dist.value],
                )

        if posed is not None:
            for i in range(sample_count):
                for b, bone_handle in enumerate(body_handles[i].bones):
                    if b == 0:
                        # Root bone
                        bone_handle.wxyz = (
                            T_world_root.rotation().wxyz[i, t].numpy(force=True)
                        )
                        bone_handle.position = T_world_root.translation()[i, t].numpy(
                            force=True,
                        )
                    else:
                        # Other bones
                        bone_handle.wxyz = SO3.from_matrix(
                            Rs_world_joint_with_root[i, t, b, :, :],
                        ).wxyz.numpy(force=True)
                        bone_handle.position = joint_positions[i, t, b - 1].numpy(
                            force=True,
                        )

        for ii, timestep_frame in enumerate(timestep_handles):
            timestep_frame.visible = t == ii

    get_viser_file = server.gui.add_button("Get .viser file")

    if get_ego_video is not None:
        ego_video = server.gui.add_button("Get Ego Video")

        @ego_video.on_click
        def _(event: viser.GuiEvent) -> None:
            assert event.client is not None
            notif = event.client.add_notification(
                "Getting video...",
                body="",
                loading=True,
                with_close_button=False,
            )
            ego_video_bytes = get_ego_video(
                gui_start_end.value[0],
                gui_start_end.value[1],
                (gui_start_end.value[1] - gui_start_end.value[0]) / gui_framerate.value,
            )
            notif.remove()
            event.client.send_file_download("ego_video.mp4", ego_video_bytes)

    prev_time = time.time()
    handle = None

    def loop_cb() -> int:
        start, end = gui_start_end.value
        duration = end - start

        if get_viser_file.value is False:
            nonlocal prev_time
            now = time.time()
            sleepdur = 1.0 / gui_framerate.value - (now - prev_time)
            if sleepdur > 0.0:
                time.sleep(sleepdur)
            prev_time = now
            if gui_playing.value:
                gui_timestep.value = (gui_timestep.value + 1 - start) % duration + start
            do_update()
            return gui_timestep.value
        else:
            # Save trajectory.
            nonlocal handle
            if handle is None:
                handle = server._start_scene_recording()
                handle.set_loop_start()
                gui_timestep.value = start

            assert handle is not None
            handle.insert_sleep(1.0 / gui_framerate.value)
            gui_timestep.value = (gui_timestep.value + 1 - start) % duration + start

            if gui_timestep.value == start:
                get_viser_file.value = False
                server.send_file_download(
                    "recording.viser",
                    content=handle.end_and_serialize(),
                )
                handle = None

            do_update()

            return gui_timestep.value

    return loop_cb


if __name__ == "__main__":
    device = torch.device("cuda")
    traj = torch.load("assets/toy_examples/infer_traj.pt").to(device)
    loop_cb = visualize_traj_and_hand_detections(
        server=viser.ViserServer(),
        traj=traj,
        smpl_family_model_basedir=Path("assets/smpl_based_model/"),
        gender="male",
        device=device,
    )

    while True:
        loop_cb()
