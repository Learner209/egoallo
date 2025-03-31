from __future__ import annotations


import sys
from pathlib import Path
from typing import Optional
from typing import TYPE_CHECKING
from typing import Tuple
from dataclasses import dataclass
from egoallo.setup_logger import setup_logger

from egoallo.type_stubs import SmplFamilyModelTypeLiteral

if TYPE_CHECKING:
    from egoallo.type_stubs import DenoiseTrajType

logger = setup_logger(output=None, name=__name__)
sys.path.append(str(Path(__file__).parent.parent.parent.parent))


@dataclass
class RendererConfig:
    """Configuration for the renderer."""

    resolution: Tuple[int, int] = (1280, 720)
    fps: float = 30.0
    fov: float = 75.0
    use_blending: bool = True


class SMPLBaseViewer:
    """
    SMPL model viewer with scene support.

    This class provides functionality to render SMPL body models in a 3D scene
    with proper lighting and camera setup.
    """

    def __init__(
        self,
        config: Optional[RendererConfig] = None,
        smpl_family_model_basedir: Path | None = None,
        smpl_family_meta_model_name: SmplFamilyModelTypeLiteral = "SmplhModel",
    ):
        """
        Initialize the SMPL viewer.

        Args:
            config: Renderer configuration for resolution, FPS, and FOV
            scene_path: Optional path to scene mesh file
        """
        self.config = config or RendererConfig()
        self.smpl_family_model_basedir = smpl_family_model_basedir
        self.smpl_family_meta_model_name = smpl_family_meta_model_name

    def render_sequence(
        self,
        traj: "DenoiseTrajType",
        output_path: str = "output.mp4",
        online_render: bool = False,
    ):
        pass
