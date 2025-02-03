import numpy as np


def blend_with_background(image: np.ndarray, background_color: tuple) -> np.ndarray:
    """Blend RGBA image with solid background color."""
    if image.shape[2] != 4:
        return image

    alpha = image[:, :, 3:4] / 255.0
    background = np.ones_like(image[:, :, :3]) * np.array(background_color) * 255
    blended = image[:, :, :3] * alpha + background * (1 - alpha)
    return blended.astype(np.uint8)
