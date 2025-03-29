from pathlib import Path
import torch
import numpy as np
import polyscope as ps
import polyscope.imgui as psim  # Import the imgui submodule
from typing import Optional

from egoallo.constants import SmplFamilyMetaModelZoo, SmplFamilyMetaModelName
from egoallo.type_stubs import SmplFamilyModelType
from egoallo.setup_logger import setup_logger
from egoallo import training_utils
from jaxtyping import jaxtyped, Float
import typeguard
from typing import Tuple

logger = setup_logger(output=None, name=__name__)
training_utils.ipdb_safety_net()

# --- The Interactive Viewer Class ---

# Define highlight color and size
HIGHLIGHT_COLOR: Tuple[float, float, float] = (1.0, 0.0, 0.0)  # Red
HIGHLIGHT_RADIUS_FACTOR: float = 2.5  # Make highlight 2.5x bigger than skeleton radius


class InteractiveSMPLViewer:
    """
    An interactive Polyscope viewer for SMPL-like models using the hybrik function.

    Allows real-time adjustment of 'phis' (body twist angles) via sliders
    and visualizes the resulting mesh and skeleton.
    Uses polyscope.imgui for UI elements within the callback.
    """

    @jaxtyped(typechecker=typeguard.typechecked)
    def __init__(
        self,
        smpl_aadecomp_model,
        pose_skeleton: Float[torch.Tensor, "num_joints 3"],
        betas: Float[torch.Tensor, "10"],
        transl: Optional[Float[torch.Tensor, "3"]],
        initial_phis: Float[torch.Tensor, "23 2"],
        global_orient: Optional[Float[torch.Tensor, "3 3"]],
        leaf_thetas: Optional[Float[torch.Tensor, "5 4"]],
        device: torch.device,
        num_hybrik_joints: int = 24,
        coordinate_transform: bool = True,
    ):
        # ... (Initialization of model, device, constants - mostly unchanged) ...
        self.model: SmplFamilyModelType = smpl_aadecomp_model.to(device)
        self.device = device
        self.num_hybrik_joints = num_hybrik_joints
        self.coordinate_transform = coordinate_transform

        # --- Prepare Input Tensors ---
        self.pose_skeleton_batch = pose_skeleton.unsqueeze(0).to(device)
        self.betas_batch = betas.unsqueeze(0).to(device)
        self.transl_batch = (
            transl.unsqueeze(0).to(device) if transl is not None else None
        )
        self.initial_phis_cos_sin_batch = initial_phis.unsqueeze(0).to(
            device,
        )  # Keep for reset
        self.global_orient_batch = (
            global_orient.unsqueeze(0).to(device) if global_orient is not None else None
        )
        self.leaf_thetas_batch = (
            leaf_thetas.unsqueeze(0).to(device) if leaf_thetas is not None else None
        )
        if (
            self.initial_phis_cos_sin_batch.shape[1] != 23
            or self.initial_phis_cos_sin_batch.shape[2] != 2
        ):
            raise ValueError(
                f"initial_phis should have shape [1, 23, 2], but got {self.initial_phis_cos_sin_batch.shape}",
            )
        if (
            self.global_orient_batch is not None
            and self.global_orient_batch.shape[1] != 3
        ):
            raise ValueError(
                f"global_orient should have shape [1, 3], but got {self.global_orient_batch.shape}",
            )

        # --- Get Model Constants for Visualization ---
        try:
            self.faces = self.model.model.faces_tensor.cpu().numpy()
            parents_full = self.model.model.parents.cpu().numpy()
            self.parents = parents_full[: self.num_hybrik_joints]
            all_joint_names = self.model.model.JOINT_NAMES
            if all_joint_names is None or len(all_joint_names) < 24:
                logger.warning("Warning: Using generic joint names for sliders.")
                self.slider_joint_names = [f"Joint {i}" for i in range(1, 24)]
            else:
                self.slider_joint_names = all_joint_names[1:24]  # Joints 1 to 23
            self.all_joints_names = all_joint_names
        except AttributeError as e:
            raise AttributeError(
                f"Failed to access model attributes (faces_tensor, parents, joint_names). Please check model structure. Error: {e}",
            )

        # --- State Variables for UI ---
        self.initial_angles_deg = self._phis_cos_sin_to_angles_deg(
            self.initial_phis_cos_sin_batch[0],
        )
        self.current_angles_deg = (
            self.initial_angles_deg.copy()
        )  # Live state for sliders
        self.selected_joint_index: int = (
            -1
        )  # <<< Index of the highlighted joint (-1 for none)
        self.skeleton_node_radius: float = 0.005  # <<< Store base radius

        # --- Polyscope Setup ---
        self.ps_mesh = None
        self.ps_skeleton = None
        self.ps_selected_joint = (
            None  # <<< Structure for the highlighted joint point cloud
        )
        self.slider_names = [f"Phi: {name}" for name in self.slider_joint_names]
        self.skeleton_edges = self._build_skeleton_edges()
        self.polyscope_initialized = False

        # --- Internal state for geometry ---
        self.current_joints_ps: Optional[np.ndarray] = (
            None  # Store current joint positions for highlight update
        )

        logger.debug("InteractiveSMPLViewer initialized.")
        logger.debug(f" - Visualizing {self.num_hybrik_joints} joints.")
        logger.debug(
            f" - Total joint names available for buttons: {len(self.slider_joint_names)}",
        )

    def _phis_cos_sin_to_angles_deg(
        self,
        phis_cos_sin_batch0: torch.Tensor,
    ) -> np.ndarray:
        """Converts [23, 2] cos/sin tensor to [23] numpy array of degrees."""
        phis_cos_sin_cpu = phis_cos_sin_batch0.detach().cpu()
        angles_rad = torch.atan2(phis_cos_sin_cpu[:, 1], phis_cos_sin_cpu[:, 0])
        return np.degrees(angles_rad.numpy())

    def _angles_deg_to_phis_cos_sin(self, angles_deg: np.ndarray) -> torch.Tensor:
        """Converts [23] numpy array of degrees to [1, 23, 2] cos/sin tensor on the correct device."""
        angles_rad = np.radians(angles_deg)
        cos_phi = np.cos(angles_rad)
        sin_phi = np.sin(angles_rad)
        phis_cos_sin = np.stack([cos_phi, sin_phi], axis=-1)
        return torch.tensor(
            phis_cos_sin,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

    def _build_skeleton_edges(self) -> np.ndarray:
        """Builds edges for Polyscope curve network from parents."""
        edges = []
        for i, p in enumerate(self.parents):
            if p != -1:
                edges.append([p, i])
        if not edges:
            logger.info("Warning: No skeleton edges generated. Check parents array.")
            return np.array([]).reshape(0, 2)
        return np.array(edges, dtype=np.int32)

    def _transform_coords_for_polyscope(
        self,
        points_tensor: torch.Tensor,
    ) -> np.ndarray:
        """Detaches tensor, moves to CPU, converts to NumPy, and optionally transforms coords."""
        if points_tensor is None:
            return None
        points_np = points_tensor.detach().cpu().numpy()
        if self.coordinate_transform:
            return points_np[:, [0, 2, 1]]  # Y-up to Z-up
        else:
            return points_np

    # --- The Polyscope Callback Function ---
    @torch.no_grad()
    def _update_visualization_callback(self):
        """
        Core callback function for Polyscope.
        Defines UI using imgui (sliders, joint buttons), reads state, updates visualization.
        """
        # --- Define the UI using ImGui ---
        # Keep track if UI elements changed state that requires *model* update
        phi_angles_changed = False
        # Keep track if UI elements changed state that requires *highlight* update
        joint_selection_changed = False

        # --- Phi Sliders Section ---
        psim.PushItemWidth(180)
        if psim.CollapsingHeader(
            "Twist Angle Controls (Phis)",
            psim.ImGuiTreeNodeFlags_DefaultOpen,
        ):
            psim.Indent()
            psim.TextUnformatted("Adjust Body Twist:")
            psim.Separator()
            for i, name in enumerate(self.slider_names):
                # Ensure index i is valid for current_angles_deg
                if i < len(self.current_angles_deg):
                    changed, new_angle = psim.SliderFloat(
                        name,
                        self.current_angles_deg[i],
                        -180.0,
                        180.0,
                    )
                    if changed:
                        self.current_angles_deg[i] = new_angle
                        phi_angles_changed = True
                else:
                    psim.TextDisabled(
                        f"{name} (Index out of bounds)",
                    )  # Should not happen with correct setup

            psim.Separator()
            if psim.Button("Reset Phis"):
                logger.debug("Resetting phis via button.")
                self.current_angles_deg = self.initial_angles_deg.copy()
                phi_angles_changed = True
            psim.Unindent()
        psim.PopItemWidth()

        # --- Joint Selection Buttons Section ---
        psim.Separator()
        if psim.CollapsingHeader(
            "Joint Selection",
            psim.ImGuiTreeNodeFlags_DefaultOpen,
        ):
            psim.Indent()
            psim.TextUnformatted("Click to highlight a joint:")
            psim.Columns(3, "joint_cols", border=False)  # Arrange buttons in columns
            button_clicked_this_frame = (
                -1
            )  # Track which button was clicked *this frame*
            for i in range(self.num_hybrik_joints):
                joint_name = (
                    self.all_joints_names[i]
                    if i < len(self.all_joints_names)
                    else f"Joint {i}"
                )
                # Change button color if it's the currently selected one
                is_selected = i == self.selected_joint_index
                if is_selected:
                    # PushStyleColor allows temporary color change for specific widgets
                    psim.PushStyleColor(
                        psim.ImGuiCol_Button,
                        (0.8, 0.1, 0.1, 1.0),
                    )  # Darker red when selected
                    psim.PushStyleColor(
                        psim.ImGuiCol_ButtonHovered,
                        (1.0, 0.2, 0.2, 1.0),
                    )
                    psim.PushStyleColor(
                        psim.ImGuiCol_ButtonActive,
                        (1.0, 0.0, 0.0, 1.0),
                    )

                if psim.Button(f"{i}: {joint_name}"):
                    # If clicked, store the index. We update self.selected_joint_index *after* the loop
                    # to handle the case where the currently selected button is clicked again (toggle off).
                    button_clicked_this_frame = i
                    joint_selection_changed = (
                        True  # Mark that selection state might change
                    )

                if is_selected:
                    psim.PopStyleColor(3)  # Pop the 3 colors pushed

                psim.NextColumn()  # Move to the next column for the next button

            psim.Columns(1)  # Reset columns

            # Handle button click logic after the loop
            if button_clicked_this_frame != -1:
                if button_clicked_this_frame == self.selected_joint_index:
                    # Clicked the already selected button -> deselect
                    self.selected_joint_index = -1
                    logger.debug(f"Joint {button_clicked_this_frame} deselected.")
                else:
                    # Clicked a new button -> select it
                    self.selected_joint_index = button_clicked_this_frame
                    logger.debug(f"Joint {self.selected_joint_index} selected.")

            # Add a button to explicitly clear selection
            if psim.Button("Clear Selection"):
                if self.selected_joint_index != -1:
                    logger.debug("Clearing joint selection via button.")
                    self.selected_joint_index = -1
                    joint_selection_changed = True
            psim.Unindent()

        # --- End of UI Definition ---

        # --- Update Model and Polyscope View ---
        model_needs_update = not self.polyscope_initialized or phi_angles_changed

        if model_needs_update:
            logger.debug(
                f"Updating model. Initialized: {self.polyscope_initialized}, Phis Changed: {phi_angles_changed}",
            )

            # 1. Convert current slider angles to phis tensor
            phis_cos_sin_batch = self._angles_deg_to_phis_cos_sin(
                self.current_angles_deg,
            )

            # 2. Re-run the hybrik model
            try:
                smpl_output = self.model.model.hybrik(
                    betas=self.betas_batch[..., :10],
                    pose_skeleton=self.pose_skeleton_batch,
                    phis=phis_cos_sin_batch,
                    transl=self.transl_batch,
                    global_orient=self.global_orient_batch,
                    leaf_thetas=self.leaf_thetas_batch,
                    return_verts=True,
                )

            except Exception as e:
                logger.error(f"Error during hybrik execution: {e}", exc_info=True)
                psim.TextColored(
                    (1.0, 0.0, 0.0, 1.0),
                    "ERROR during model execution! Check logs.",
                )
                return  # Stop if model fails

            # 3. Get updated vertices and joints
            vertices_new_tensor = smpl_output.vertices[0]
            joints_new_tensor = smpl_output.joints[0]

            # Ensure correct number of joints
            num_joints_to_use = min(joints_new_tensor.shape[0], self.num_hybrik_joints)
            if joints_new_tensor.shape[0] < self.num_hybrik_joints:
                logger.warning(
                    f"Hybrik output {joints_new_tensor.shape[0]} joints, but expected {self.num_hybrik_joints}. Using {num_joints_to_use}.",
                )
            joints_new_tensor = joints_new_tensor[:num_joints_to_use, :]

            # 4. Prepare data for Polyscope
            vertices_ps = self._transform_coords_for_polyscope(vertices_new_tensor)
            self.current_joints_ps = self._transform_coords_for_polyscope(
                joints_new_tensor,
            )  # Store for highlight

            # 5. Update Polyscope structures (Mesh and Skeleton)
            try:
                if not self.polyscope_initialized:
                    logger.info("Registering Polyscope structures for the first time.")
                    self.ps_mesh = ps.register_surface_mesh(
                        "SMPL Mesh",
                        vertices_ps,
                        self.faces,
                        smooth_shade=True,
                    )
                    self.ps_skeleton = ps.register_curve_network(
                        "SMPL Skeleton",
                        self.current_joints_ps,
                        self.skeleton_edges,
                    )
                    self.ps_skeleton.set_radius(
                        self.skeleton_node_radius,
                        relative=False,
                    )
                    self.polyscope_initialized = True
                else:
                    if self.ps_mesh:
                        self.ps_mesh.update_vertex_positions(vertices_ps)
                    if self.ps_skeleton:
                        self.ps_skeleton.update_node_positions(self.current_joints_ps)

            except Exception as e:
                logger.error(
                    f"Error updating Polyscope mesh/skeleton structures: {e}",
                    exc_info=True,
                )
                psim.TextColored(
                    (1.0, 0.0, 0.0, 1.0),
                    "ERROR updating Polyscope geometry! Check logs.",
                )
                # Don't proceed to highlight update if main geometry failed
                return

        # --- Update Joint Highlight Visualization ---
        # Update if selection changed OR if the model was updated (joint positions changed)
        highlight_needs_update = joint_selection_changed or model_needs_update

        if highlight_needs_update and self.current_joints_ps is not None:
            try:
                # If a joint is selected
                if 0 <= self.selected_joint_index < self.current_joints_ps.shape[0]:
                    selected_joint_pos = self.current_joints_ps[
                        self.selected_joint_index : self.selected_joint_index + 1,
                        :,
                    ]  # Keep it as [[x, y, z]]

                    # Register or update the highlight point cloud
                    highlight_radius = (
                        self.skeleton_node_radius * HIGHLIGHT_RADIUS_FACTOR
                    )
                    if self.ps_selected_joint is None:
                        self.ps_selected_joint = ps.register_point_cloud(
                            "Selected Joint",
                            selected_joint_pos,
                            radius=highlight_radius,
                            color=HIGHLIGHT_COLOR,
                            enabled=True,
                        )
                        logger.debug(
                            f"Registered highlight point cloud for joint {self.selected_joint_index}.",
                        )
                    else:
                        self.ps_selected_joint.update_point_positions(
                            selected_joint_pos,
                        )
                        self.ps_selected_joint.set_enabled(True)  # Ensure it's visible
                        # Optional: update radius/color if they could change, but they are constant here
                        # self.ps_selected_joint.set_radius(highlight_radius)
                        # self.ps_selected_joint.set_color(HIGHLIGHT_COLOR)
                        logger.debug(
                            f"Updated highlight point cloud position for joint {self.selected_joint_index}.",
                        )

                # If no joint is selected (or selection is invalid)
                else:
                    # If the highlight structure exists, disable it
                    if self.ps_selected_joint is not None:
                        self.ps_selected_joint.set_enabled(False)
                        logger.debug("Disabled highlight point cloud.")

            except Exception as e:
                logger.error(
                    f"Error updating Polyscope highlight structure: {e}",
                    exc_info=True,
                )
                psim.TextColored(
                    (1.0, 0.0, 0.0, 1.0),
                    "ERROR updating joint highlight! Check logs.",
                )

    def show(self):
        """Initialize Polyscope, set up UI callback, and run the main event loop."""
        logger.info("Initializing Polyscope...")
        ps.init()

        ps.set_up_dir(
            "z_up" if self.coordinate_transform else "y_up",
        )  # Match view to data
        ps.set_ground_plane_mode("shadow_only")

        # --- Set the user callback ---
        # Polyscope will now call _update_visualization_callback every frame.
        # This function will handle both UI drawing and geometry updates.
        ps.set_user_callback(self._update_visualization_callback)
        logger.info(
            "User callback set. Initial computation will happen on first frame.",
        )

        # --- Run Polyscope ---
        # The initial model computation and registration will happen automatically
        # during the first execution of the callback inside ps.show().
        logger.info("Starting Polyscope viewer...")
        ps.show()

        # Cleanup after window is closed
        ps.clear_user_callback()  # Good practice
        logger.info("Polyscope window closed.")
        # ps.clear_structures() # Optional: uncomment if needed


# --- Example Usage ---
if __name__ == "__main__":
    logger.info("Setting up dummy data and model...")
    # Example Usage Requires a dummy or real model and data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load or create the smpl_aadecomp_model
    # Replace with your actual model loading
    smpl_family_model_basedir = Path("assets/smpl_based_model")
    gender = "neutral"
    smpl_aadecomp_model = (
        SmplFamilyMetaModelZoo[SmplFamilyMetaModelName]
        .load(smpl_family_model_basedir, gender=gender, num_joints=24)
        .to(device)
    )

    # 2. Prepare fixed inputs (using dummy data)
    num_input_joints = 24  # Or 24, depending on your pose_skeleton source
    batch_size = 1  # Viewer works best with batch size 1
    pose_skeleton_data = (
        torch.randn(num_input_joints, 3) * 0.5
    )  # Example joint positions
    betas_data = torch.zeros(10)  # Example shape parameters
    transl_data = torch.tensor(
        [0.0, 0.0, 2.0],
    )  # Example translation to move model back

    # Initial phis (23 pairs of [cos, sin]) - start at zero rotation (angle=0 => cos=1, sin=0)
    initial_phis_data = torch.zeros(23, 2)
    initial_phis_data[:, 0] = 1.0  # Set cos(0) = 1
    initial_phis_data[:, 1] = 0.0  # Set sin(0) = 0

    logger.info("Creating InteractiveSMPLViewer instance...")
    try:
        viewer = InteractiveSMPLViewer(
            smpl_aadecomp_model=smpl_aadecomp_model,
            pose_skeleton=pose_skeleton_data,
            betas=betas_data,
            transl=transl_data,
            initial_phis=initial_phis_data,
            global_orient=None,
            device=device,
            num_hybrik_joints=24,  # Standard for SMPL output from hybrik
            coordinate_transform=True,
        )

        # 3. Run the viewer
        viewer.show()

    except Exception:
        logger.info("\n--- An error occurred during viewer setup or execution ---")
        import traceback

        traceback.print_exc()
        logger.info("----------------------------------------------------------")

    logger.info("Viewer execution finished.")
