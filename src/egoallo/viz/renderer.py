from cloudrender.libegl import EGLContext
from OpenGL import GL as gl
import logging
from dataclasses import dataclass
from typing import Tuple

logger = logging.getLogger(__name__)


@dataclass
class RendererConfig:
    """Configuration for the renderer."""

    resolution: Tuple[int, int] = (1280, 720)
    fps: float = 30.0
    fov: float = 75.0


class BaseRenderer:
    """Base class for OpenGL/EGL rendering setup."""

    def __init__(self, config: RendererConfig = RendererConfig()):
        self.config = config
        self.context = None
        self._setup_context()
        self._setup_buffers()
        self._configure_gl()

    def _setup_context(self):
        """Initialize EGL context."""
        logger.info("Initializing EGL and OpenGL")
        self.context = EGLContext()
        if not self.context.initialize(*self.config.resolution):
            raise RuntimeError("Failed to initialize EGL context")

    def _setup_buffers(self):
        """Set up OpenGL frame and render buffers."""
        self._main_cb, self._main_db = gl.glGenRenderbuffers(2)

        # Color buffer
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._main_cb)
        gl.glRenderbufferStorage(
            gl.GL_RENDERBUFFER, gl.GL_RGBA, *self.config.resolution
        )

        # Depth buffer
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._main_db)
        gl.glRenderbufferStorage(
            gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24, *self.config.resolution
        )

        # Frame buffer
        self._main_fb = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self._main_fb)
        gl.glFramebufferRenderbuffer(
            gl.GL_DRAW_FRAMEBUFFER,
            gl.GL_COLOR_ATTACHMENT0,
            gl.GL_RENDERBUFFER,
            self._main_cb,
        )
        gl.glFramebufferRenderbuffer(
            gl.GL_DRAW_FRAMEBUFFER,
            gl.GL_DEPTH_ATTACHMENT,
            gl.GL_RENDERBUFFER,
            self._main_db,
        )

        gl.glDrawBuffers([gl.GL_COLOR_ATTACHMENT0])

    def _configure_gl(self):
        """Configure OpenGL settings."""
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glClearColor(1.0, 1.0, 1.0, 0)
        gl.glViewport(0, 0, *self.config.resolution)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDepthFunc(gl.GL_LESS)
        gl.glDepthRange(0.0, 1.0)
