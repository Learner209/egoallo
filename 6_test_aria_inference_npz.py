from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch
import viser
from egoallo import fncsmpl
from egoallo import network
from egoallo.transforms import SO3
from egoallo.vis_helpers import visualize_traj_and_hand_detections


@dataclasses.dataclass
class Args:
    npz_path: Path = Path(
        "./egoallo_example_trajectories/coffeemachine/egoallo_outputs/20240929-011937_10-522.npz",
    )
    """Path to the input trajectory."""
    smplh_npz_path: Path = Path("./data/smplh/neutral/model.npz")
    """Path to the SMPLH model."""
    visualize_traj: bool = True
    """Whether to visualize the trajectory after sampling."""


def main(
    args: Args,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> None:
    # Load data and models
    traj_data = np.load(args.npz_path)
    body_model = fncsmpl.SmplhModel.load(args.smplh_npz_path).to(device)

    Ts_world_cpf = torch.from_numpy(traj_data["Ts_world_cpf"]).to(device)
    body_quats = torch.from_numpy(traj_data["body_quats"]).to(device)
    left_hand_quats = torch.from_numpy(traj_data["left_hand_quats"]).to(device)
    right_hand_quats = torch.from_numpy(traj_data["right_hand_quats"]).to(device)
    contacts = torch.from_numpy(traj_data["contacts"]).to(device)
    betas = torch.from_numpy(traj_data["betas"]).to(device)

    # Convert quats to rotation matrices
    body_rotmats = SO3(body_quats).as_matrix()
    hand_rotmats = SO3(
        torch.cat([left_hand_quats, right_hand_quats], dim=-2),
    ).as_matrix()

    # Create EgoDenoiseTraj instance
    traj = network.AbsoluteDenoiseTraj(
        betas=betas,
        body_rotmats=body_rotmats,
        contacts=contacts,
        hand_rotmats=hand_rotmats,
    )

    # Visualize
    if args.visualize_traj:
        server = viser.ViserServer()
        server.gui.configure_theme(dark_mode=True)
        loop_cb = visualize_traj_and_hand_detections(
            server,
            Ts_world_cpf,
            traj,
            body_model,
            hamer_detections=None,
            aria_detections=None,
            points_data=None,
            splat_path=None,
            floor_z=0.0,
        )
        while True:
            loop_cb()


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
