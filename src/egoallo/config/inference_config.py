from __future__ import annotations

import dataclasses
from pathlib import Path

from egoallo.guidance_optimizer_jax import GuidanceMode


@dataclasses.dataclass
class InferenceConfig:
    traj_root: Path = Path("./egoallo_example_trajectories/coffeemachine/")
    """Search directory for trajectories. This should generally be laid out as something like:

    traj_dir/
        video.vrs
        egoallo_outputs/
            {date}_{start_index}-{end_index}.npz
            ...
        ...
    """
    output_dir: Path = Path("./egoallo_example_trajectories/coffeemachine/")
    """Output directory for the results. It can be separated from traj_root."""

    checkpoint_dir: Path = Path("experiments/nov8_v1/v1/checkpoint-10000")
    smplh_model_path: Path = Path("assets/smpl_based_model/smplh/SMPLH_NEUTRAL.pkl")

    glasses_x_angle_offset: float = 0.0
    """Rotate the CPF poses by some X angle."""
    start_index: int = 0
    """Index within the downsampled trajectory to start inference at."""
    traj_length: int = 128
    """How many timesteps to estimate body motion for."""
    num_samples: int = 1
    """Number of samples to take."""
    guidance_mode: GuidanceMode = "no_hands"
    """Which guidance mode to use."""
    guidance_inner: bool = False
    """Whether to apply guidance optimizer between denoising steps."""
    guidance_post: bool = False
    """Whether to apply guidance optimizer after diffusion sampling."""
    save_traj: bool = True
    """Whether to save the output trajectory."""
    visualize_traj: bool = False
    """Whether to visualize the trajectory after sampling."""
    use_ipdb: bool = False
    """Whether to use ipdb instead of pdb."""
