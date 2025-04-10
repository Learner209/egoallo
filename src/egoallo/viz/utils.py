import numpy as np
from egoallo.transforms import SE3, SO3

from jaxtyping import Float, Bool
from typing import List
import typeguard
from typing import Optional
from egoallo.mapping import EGOEXO4D_BODYPOSE_KINTREE_PARENTS, SMPLH_KINTREE
from jaxtyping import jaxtyped
import open3d as o3d
import time
import torch


def blend_with_background(image: np.ndarray, background_color: tuple) -> np.ndarray:
    """Blend RGBA image with solid background color."""
    if image.shape[2] != 4:
        return image

    alpha = image[:, :, 3:4] / 255.0
    background = np.ones_like(image[:, :, :3]) * np.array(background_color) * 255
    blended = image[:, :, :3] * alpha + background * (1 - alpha)
    return blended.astype(np.uint8)


@jaxtyped(typechecker=typeguard.typechecked)
def create_skeleton_point_cloud(
    joints_wrt_world: Float[np.ndarray, "num_joints 3"],
    visible_joints_mask: Bool[np.ndarray, "num_joints"],
    input_smplh: bool = False,
    num_samples_per_bone: int = 100,
    return_colors: Optional[bool] = False,
) -> (
    tuple[Float[np.ndarray, "num_points 3"], Float[np.ndarray, "num_points 3"]]
    | Float[np.ndarray, "num_points 3"]
):
    """Create a point cloud representing the skeleton by densely sampling along bones.

    Args:
        joints_wrt_world: Joint positions in world coordinates, shape [num_joints, 3]
        visible_joints_mask: Boolean mask indicating valid joints, shape [num_joints]
        input_smplh: If True, input uses SMPLH convention (52 joints), otherwise COCO (17 joints)
        num_samples_per_bone: Number of points to sample along each bone
        return_colors: If True, returns colors for points (green=valid, red=invalid)

    Returns:
        points: Sampled 3D points forming the skeleton, shape [num_points, 3]
        colors: Optional point colors, shape [num_points, 3]
    """
    # Verify input dimensions based on convention
    if input_smplh:
        assert joints_wrt_world.shape[0] == 22, (
            f"SMPLH joints should have 22 joints, got {joints_wrt_world.shape[0]}"
        )
        # SMPLH kinematic tree (parent indices for each joint)
        # Only includes body joints, not hand joints
        kintree = SMPLH_KINTREE
    else:
        assert joints_wrt_world.shape[0] == 17, (
            f"COCO joints should have 17 joints, got {joints_wrt_world.shape[0]}"
        )
        # COCO kinematic tree (parent indices for each joint)
        kintree = EGOEXO4D_BODYPOSE_KINTREE_PARENTS

    # Initialize list to store sampled points and colors
    vis_pts = []
    invis_pts = []
    visc = [] if return_colors else None
    invisc = [] if return_colors else None

    # Define colors
    green = np.array([0, 255, 0]).astype(np.uint8)
    red = np.array([0, 255, 0]).astype(np.uint8)

    # Iterate through kinematic tree
    for joint_idx, parent_idx in enumerate(kintree):
        if parent_idx == -1:
            continue

        # Get joint positions
        start = joints_wrt_world[parent_idx]
        end = joints_wrt_world[joint_idx]

        # Skip if either contains NaN
        if np.isnan(start).any() or np.isnan(end).any():
            continue

        # Get validity of both joints
        start_valid = visible_joints_mask[parent_idx]
        end_valid = visible_joints_mask[joint_idx]

        # Sample points along bone
        for t in np.linspace(0, 1, num_samples_per_bone):
            point = start + t * (end - start)
            if start_valid and end_valid:
                vis_pts.append(point)
                if return_colors:
                    visc.append(green)
            elif not start_valid and not end_valid:
                invis_pts.append(point)
                if return_colors:
                    invisc.append(red)
            else:
                if (start_valid and t > 0.5) or (end_valid and t <= 0.5):
                    invis_pts.append(point)
                    if return_colors:
                        invisc.append(red)
                else:
                    vis_pts.append(point)
                    if return_colors:
                        visc.append(green)

    if len(vis_pts) == 0:
        vis_ret = (
            (np.zeros((0, 3)), np.zeros((0, 3))) if return_colors else np.zeros((0, 3))
        )
    else:
        vis_ret = (
            (np.stack(vis_pts), np.stack(visc)) if return_colors else np.stack(vis_pts)
        )

    if len(invis_pts) == 0:
        invis_ret = (
            (np.zeros((0, 3)), np.zeros((0, 3))) if return_colors else np.zeros((0, 3))
        )
    else:
        invis_ret = (
            (np.stack(invis_pts), np.stack(invisc))
            if return_colors
            else np.stack(invis_pts)
        )

    return vis_ret, invis_ret


def plot_synchronous_3d_animations_multi_modal(
    num_of_sets: int,
    rows: int,
    cols: int,
    data: list,  # List of numpy arrays, one per modality
    modality_names: list = None,  # Optional list of names for modalities
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.colors

    """
    Visualizes multiple sets of synchronous 3D point cloud animations from multiple
    modalities using Plotly subplots.

    Args:
        num_of_sets (int): The total number of data sets to visualize.
        rows (int): The number of rows in the subplot grid.
        cols (int): The number of columns in the subplot grid.
        data (list): A list containing numpy arrays, one for each modality.
                     Each array must have the shape: (num_sets, timesteps, num_kpts, 3).
        modality_names (list, optional): A list of strings corresponding to the name
                                         of each modality for the legend. If None,
                                         modalities will be named "Mod 1", "Mod 2", etc.
                                         Defaults to None.
    """

    if rows * cols < num_of_sets:
        raise ValueError(
            f"Grid size ({rows}x{cols}) is too small for {num_of_sets} sets.",
        )

    num_modalities = len(data)
    if num_modalities == 0:
        raise ValueError("The 'data' list cannot be empty.")

    # --- Validate Data Shapes and Modality Names ---
    if modality_names and len(modality_names) != num_modalities:
        raise ValueError(
            f"Number of modality names ({len(modality_names)}) must match number of data arrays ({num_modalities}).",
        )

    # Use the shape of the first modality's data as reference
    ref_shape = data[0].shape
    if ref_shape[0] != num_of_sets:
        raise ValueError(
            f"Mismatch between num_of_sets ({num_of_sets}) and data dimension 0 ({ref_shape[0]}) for modality 0.",
        )
    if len(ref_shape) != 4 or ref_shape[3] != 3:
        raise ValueError(
            "Data arrays must be 4-dimensional (sets, timesteps, kpts, 3).",
        )

    timesteps = ref_shape[1]
    _num_kpts = ref_shape[2]

    # Check consistency across all modalities
    for mod_idx in range(1, num_modalities):
        mod_shape = data[mod_idx].shape
        if mod_shape[0] != num_of_sets:
            raise ValueError(
                f"Mismatch between num_of_sets ({num_of_sets}) and data dimension 0 ({mod_shape[0]}) for modality {mod_idx}.",
            )
        if mod_shape[1] != ref_shape[1]:
            raise ValueError(
                f"Shape mismatch (timesteps) between modality 0 {ref_shape[1:]} and modality {mod_idx} {mod_shape[1:]}.",
            )
        if mod_shape[-1] != ref_shape[-1]:
            raise ValueError(
                f"Shape mismatch (dims) between modality 0 {ref_shape[2:]} and modality {mod_idx} {mod_shape[2:]}.",
            )

    # --- Setup Visuals ---
    # Use a qualitative color scheme from Plotly
    colors = plotly.colors.qualitative.Vivid
    # Define a list of marker symbols
    markers = [
        "circle",
        "circle-open",
        "cross",
        "diamond",
        "diamond-open",
        "square",
        "square-open",
    ]

    # Generate default modality names if not provided
    if modality_names is None:
        modality_names = [f"Mod {i + 1}" for i in range(num_modalities)]

    # --- Create Figure ---
    specs = [[{"type": "scene"} for _ in range(cols)] for _ in range(rows)]
    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=specs,
        subplot_titles=[f"Set {i + 1}" for i in range(num_of_sets)],
    )

    # --- Add Initial Traces (Timestep 0) ---
    # The order traces are added here *must* be maintained when creating frames
    for i in range(num_of_sets):  # Iterate through sets first
        subplot_row = (i // cols) + 1
        subplot_col = (i % cols) + 1
        for mod_idx in range(num_modalities):  # Iterate through modalities second
            mod_data_array = data[mod_idx]
            data_t0 = mod_data_array[
                i,
                0,
                :,
                :,
            ]  # Data for set i, modality mod_idx, timestep 0

            color = colors[mod_idx % len(colors)]  # Cycle through colors
            marker_symbol = markers[mod_idx % len(markers)]  # Cycle through markers
            name = f"Set {i + 1} {modality_names[mod_idx]}"

            fig.add_trace(
                go.Scatter3d(
                    x=data_t0[:, 0],
                    y=data_t0[:, 1],
                    z=data_t0[:, 2],
                    mode="markers",
                    marker=dict(color=color, size=1.5, symbol=marker_symbol),
                    name=name,
                ),
                row=subplot_row,
                col=subplot_col,
            )

    # --- Create Animation Frames ---
    frames = []
    num_traces_total = num_of_sets * num_modalities  # Total traces added initially

    for t in range(timesteps):  # Loop through each timestep for the frame
        frame_data = []  # Holds the data for *all* traces for this specific frame/timestep

        # Iterate in the *exact same order* as traces were added initially
        for i in range(num_of_sets):  # Sets first
            for mod_idx in range(num_modalities):  # Modalities second
                mod_data_array = data[mod_idx]
                data_t = mod_data_array[
                    i,
                    t,
                    :,
                    :,
                ]  # Data for set i, modality mod_idx, timestep t

                # Append only the coordinate data for this trace for this frame
                frame_data.append(
                    go.Scatter3d(x=data_t[:, 0], y=data_t[:, 1], z=data_t[:, 2]),
                )

        frames.append(
            go.Frame(
                data=frame_data,  # Contains data for all num_traces_total traces
                name=f"t={t}",
                # Specify that this frame updates all the initially added traces
                traces=list(range(num_traces_total)),
            ),
        )

    fig.frames = frames

    # --- Configure Layout and Animation Controls --- (Identical to previous version)

    # Buttons
    fig.update_layout(
        title_text="Synchronous Multi-Modality 3D Point Cloud Animations",
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 100, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "showactive": False,
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            },
        ],
    )

    # Slider
    sliders = [{"pad": {"t": 30, "b": 10}, "len": 0.9, "x": 0.1, "y": 0, "steps": []}]
    for t in range(timesteps):
        slider_step = {
            "args": [
                [f"t={t}"],
                {
                    "frame": {"duration": 100, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
            "label": f"Time {t}",
            "method": "animate",
        }
        sliders[0]["steps"].append(slider_step)

    fig.update_layout(sliders=sliders)

    # --- Customize Scene Layouts (Identical to previous version) ---
    scene_config = dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        aspectratio=dict(x=1, y=1, z=1),
        aspectmode="data",
        # Consider setting axis ranges based on global min/max across all data if needed
    )
    layout_update_dict = {}
    for i in range(num_of_sets):
        scene_id = f"scene{i + 1}" if i > 0 else "scene"
        layout_update_dict[scene_id] = scene_config

    fig.update_layout(**layout_update_dict)
    fig.update_layout(
        height=max(400, rows * 350),
        width=max(600, cols * 350),
    )  # Slightly increase size per plot
    fig.update_layout(hovermode="closest", legend_title_text="Modalities")

    # Show legend
    fig.update_layout(showlegend=True)

    # --- Show Figure ---
    fig.show()


def test_plot_synchronous_3d_animations_mutli_modal():
    import math

    # --- Generate Example Data ---
    NUM_SETS = 6  # Reduced for clarity, use 8 if preferred
    TIMESTEPS = 60
    NUM_KPTS = 15

    # Create data for 3 modalities
    num_modalities_example = 3
    all_data = []  # List to hold data arrays for each modality

    # Modality 1: Expanding/Contracting Sphere
    data_mod1 = np.zeros((NUM_SETS, TIMESTEPS, NUM_KPTS, 3))
    # Modality 2: Figure-eight Motion
    data_mod2 = np.zeros((NUM_SETS, TIMESTEPS, NUM_KPTS, 3))
    # Modality 3: Random Walk within a Box
    data_mod3 = np.zeros((NUM_SETS, TIMESTEPS, NUM_KPTS, 3))

    # Generate points on a sphere for modality 1 base
    phi = np.pi * (np.sqrt(5.0) - 1.0)  # golden angle in radians
    indices = np.arange(0, NUM_KPTS)
    z_sphere = 1 - (indices / float(NUM_KPTS - 1)) * 2  # z goes from 1 to -1
    radius_sphere = np.sqrt(1 - z_sphere * z_sphere)  # radius at z
    theta_sphere = phi * indices  # golden angle increment
    x_sphere_base = radius_sphere * np.cos(theta_sphere)
    y_sphere_base = radius_sphere * np.sin(theta_sphere)

    for s in range(NUM_SETS):
        # Set-specific parameters
        set_offset = np.array(
            [s % 3 * 10, s // 3 * 10, 0],
        )  # Basic spatial offset for sets
        set_speed_factor = 1.0 + s * 0.1
        set_scale_factor = 5.0 + s  # Size factor

        # Initialize positions for random walk (Modality 3)
        current_pos_mod3 = (
            np.random.rand(NUM_KPTS, 3) * set_scale_factor * 0.5
            - set_scale_factor * 0.25
            + set_offset
        )

        for t in range(TIMESTEPS):
            time_angle = t * 0.1 * set_speed_factor

            # Modality 1: Pulsating Sphere
            scale = set_scale_factor * (
                1 + 0.3 * np.sin(time_angle * 2)
            )  # Pulsating radius
            data_mod1[s, t, :, 0] = x_sphere_base * scale + set_offset[0]
            data_mod1[s, t, :, 1] = y_sphere_base * scale + set_offset[1]
            data_mod1[s, t, :, 2] = z_sphere * scale + set_offset[2]

            # Modality 2: Figure Eight
            x_fig8 = set_scale_factor * 0.8 * np.sin(time_angle)
            y_fig8 = (
                set_scale_factor * 0.5 * np.sin(time_angle * 2)
            )  # Double frequency for figure eight
            z_fig8 = np.linspace(
                -set_scale_factor * 0.2,
                set_scale_factor * 0.2,
                NUM_KPTS,
            )  # Spread points in Z
            data_mod2[s, t, :, 0] = x_fig8 + set_offset[0]
            data_mod2[s, t, :, 1] = y_fig8 + set_offset[1]
            data_mod2[s, t, :, 2] = (
                z_fig8 + set_offset[2] + t * 0.05
            )  # Slow drift upwards

            # Modality 3: Random Walk step
            step = (
                (np.random.rand(NUM_KPTS, 3) - 0.5) * 0.1 * set_scale_factor
            )  # Small random step
            current_pos_mod3 += step
            # Simple boundary reflection
            box_size = set_scale_factor * 0.6
            min_bound = set_offset - box_size / 2
            max_bound = set_offset + box_size / 2
            current_pos_mod3 = np.clip(
                current_pos_mod3,
                min_bound,
                max_bound,
            )  # Keep within bounds (simple clip)
            data_mod3[s, t, :, :] = current_pos_mod3

    # Add generated data arrays to the list
    all_data.append(data_mod1)
    all_data.append(data_mod2)
    all_data.append(data_mod3)

    # Optional: Define names for the modalities
    mod_names = ["SpherePulse", "FigureEight", "RandWalk"]

    # --- Define grid layout ---
    total_plots = NUM_SETS
    cols_layout = math.ceil(math.sqrt(total_plots))
    rows_layout = math.ceil(total_plots / cols_layout)
    print(
        f"Using layout: {rows_layout} rows x {cols_layout} cols for {num_modalities_example} modalities.",
    )

    # --- Call the plotting function ---
    plot_synchronous_3d_animations_multi_modal(
        num_of_sets=NUM_SETS,
        rows=rows_layout,
        cols=cols_layout,
        data=all_data,  # Pass the list of numpy arrays
        modality_names=mod_names,  # Pass the list of names
    )

    print("Plot generation complete. Check the displayed Plotly figure.")


@jaxtyped(typechecker=typeguard.typechecked)
def visualize_smpl_skeleton_with_rotation(
    Rs_world_joints: Float[torch.Tensor, "*batch time joints 3 3"],
    ts_world_joints: Float[torch.Tensor, "*batch time joints 3"],
    parent_indices: List[int],
    batch_idx: int = 0,
    joint_sphere_radius: float = 0.01,
    axis_size: float = 0.05,
    skeleton_line_color: List[float] = [0.1, 0.8, 0.1],
    joint_sphere_color: List[float] = [0.8, 0.1, 0.1],
    window_title: str = "SMPL Skeleton Animation",
):
    """
    Visualizes a continuous animation of SMPL-like skeleton movements for a specific batch using Open3D.

    Args:
        Rs_world_joints (torch.Tensor): Joint rotations in world space. Shape: (bs, ts, num_jts, 3, 3).
        ts_world_joints (torch.Tensor): Joint translations in world space. Shape: (bs, ts, num_jts, 3).
        parent_indices (list or torch.Tensor): Parent indices for each joint.
    """
    num_sphere_verts = 0
    rel_Rs_world_joints = Rs_world_joints.clone()
    rel_Rs_world_joints[..., 1:, :, :, :] = (
        SO3.from_matrix(Rs_world_joints[..., :-1, :, :, :]).inverse()
        @ SO3.from_matrix(Rs_world_joints[..., 1:, :, :, :])
    ).as_matrix()
    rel_Rs_world_joints = rel_Rs_world_joints.cpu().numpy(force=True)
    Rs_world_joints = Rs_world_joints.cpu().numpy(force=True)
    rel_ts_world_joints = ts_world_joints.clone()
    rel_ts_world_joints[..., 1:, :, :] = (
        ts_world_joints[..., 1:, :, :] - ts_world_joints[..., :-1, :, :]
    )
    rel_ts_world_joints = rel_ts_world_joints.cpu().numpy(force=True)
    ts_world_joints = ts_world_joints.cpu().numpy(force=True)

    def create_skeleton_geometries(
        R_frame,
        t_frame,
        parent_indices,
        joint_sphere_radius,
        axis_size,
        skeleton_line_color,
        joint_sphere_color,
    ):
        geometries = []
        num_jts = t_frame.shape[0]

        # Joint Spheres
        joint_spheres = o3d.geometry.TriangleMesh()
        for i in range(num_jts):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=joint_sphere_radius)
            sphere.translate(t_frame[i])
            nonlocal num_sphere_verts
            num_sphere_verts = np.asarray(sphere.vertices).shape[0]
            joint_spheres += sphere
        joint_spheres.paint_uniform_color(joint_sphere_color)
        geometries.append(joint_spheres)

        # Skeleton Lines
        lines = [
            [parent, i]
            for i, parent in enumerate(parent_indices)
            if parent != -1 and 0 <= parent < num_jts
        ]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(t_frame),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.paint_uniform_color(skeleton_line_color)
        geometries.append(line_set)

        # Coordinate Frames
        for i in range(num_jts):
            T = np.eye(4)
            T[:3, :3] = R_frame[i]
            T[:3, 3] = t_frame[i]
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=axis_size,
            )
            coord_frame.transform(T)
            geometries.append(coord_frame)

        return geometries

    def update_skeleton_geometries(
        geometries,
        R_frame,
        t_frame,
        rel_R_frame,
        rel_t_frame,
        parent_indices,
        joint_sphere_radius,
        axis_size,
    ):
        num_jts = t_frame.shape[0]

        # Update Joint Spheres
        joint_spheres = geometries[0]
        vertices = np.asarray(joint_spheres.vertices)
        for i in range(num_jts):
            vertices[i * num_sphere_verts : (i + 1) * num_sphere_verts] = (
                vertices[i * num_sphere_verts : (i + 1) * num_sphere_verts]
                - vertices[i * num_sphere_verts]
                + t_frame[i]
            )
        joint_spheres.vertices = o3d.utility.Vector3dVector(vertices)

        # Update Skeleton Lines
        line_set = geometries[1]
        line_set.points = o3d.utility.Vector3dVector(t_frame)

        for i in range(num_jts):
            geometries[i + 2].translate(rel_t_frame[i], relative=True)
            geometries[i + 2].rotate(R=rel_R_frame[i], center=t_frame[i])

    # Initialize Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"{window_title} (Batch: {batch_idx})")

    # Prepare initial geometries
    geometries = create_skeleton_geometries(
        Rs_world_joints[batch_idx, 0],
        ts_world_joints[batch_idx, 0],
        parent_indices,
        joint_sphere_radius,
        axis_size,
        skeleton_line_color,
        joint_sphere_color,
    )

    # Add geometries to the visualizer
    for geom in geometries:
        vis.add_geometry(geom)

    vis.poll_events()
    vis.update_renderer()

    # Main animation loop
    for time_idx in range(1, ts_world_joints.shape[1]):
        update_skeleton_geometries(
            geometries,
            Rs_world_joints[batch_idx, time_idx],
            ts_world_joints[batch_idx, time_idx],
            rel_Rs_world_joints[batch_idx, time_idx],
            rel_ts_world_joints[batch_idx, time_idx],
            parent_indices,
            joint_sphere_radius,
            axis_size,
        )

        vis.update_geometry(geometries[0])  # Update joint spheres
        vis.update_geometry(geometries[1])  # Update skeleton lines
        for i in range(2, len(geometries)):  # Update coordinate frames
            vis.update_geometry(geometries[i])

        vis.poll_events()
        vis.update_renderer()

        # Optional: add a small delay to control animation speed
        time.sleep(0.1)

    vis.destroy_window()


def test_visualize_smpl_skeleton_with_rotation():
    import torch
    from egoallo import network
    from pathlib import Path
    from egoallo.constants import SmplFamilyMetaModelZoo
    from egoallo.mapping import SMPLH_KINTREE

    _ = torch.load("/tmp/test_batch.pt")
    body_model = SmplFamilyMetaModelZoo["SmplhModel"].load(
        Path("./assets/smpl_based_model"),
        gender="neutral",
    )
    denoising = network.DenoisingConfig(denoising_mode="AbsoluteDenoiseTraj")
    x_0 = denoising.from_ego_data(
        _,
        include_hands=False,
        smpl_family_model_basedir=Path("./assets/smpl_based_model"),
    )
    posed = x_0.apply_to_body(body_model)
    Rs = (
        SE3(torch.cat([posed.T_world_root[..., None, :], posed.Ts_world_joint], dim=-2))
        .rotation()
        .as_matrix()
        .cpu()[..., :22, :, :]
    )
    ts = (
        SE3(torch.cat([posed.T_world_root[..., None, :], posed.Ts_world_joint], dim=-2))
        .translation()
        .cpu()[..., :22, :]
    )
    parent_indices = SMPLH_KINTREE
    for i in range(32):
        visualize_smpl_skeleton_with_rotation(Rs, ts, parent_indices, batch_idx=i)


if __name__ == "__main__":
    test_visualize_smpl_skeleton_with_rotation()
