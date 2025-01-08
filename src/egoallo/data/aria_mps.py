from pathlib import Path
from typing import Literal

import numpy as np
from projectaria_tools.core import mps
from projectaria_tools.core.mps.utils import filter_points_from_confidence
from typing import Optional, Tuple
import tyro
from egoallo.utils.setup_logger import setup_logger
from third_party.cloudrender.cloudrender.render.pointcloud import Pointcloud

logger = setup_logger(output=None, name=__name__)


def load_point_cloud_and_find_ground(
    points_path: Path,
    return_points: Literal["all", "filtered", "less_filtered"] = "less_filtered", 
    cache_files: bool = True,
) -> Tuple[Pointcloud.PointcloudContainer, np.ndarray, float]:
    """Load an Aria MPS point cloud and find the ground plane.

    Args:
        points_path: Path to the point cloud file
        return_points: Which set of points to return ("all", "filtered", or "less_filtered")
        cache_files: Whether to cache filtered points to disk for faster future loading
    """
    filtered_points_npz_cache_path = points_path.parent / "_cached_filtered_points.npz"
    less_filtered_points_npz_cache_path = points_path.parent / "_cached_less_filtered_points.npz"

    # Check if we should use cached files
    use_cache = cache_files and (
        filtered_points_npz_cache_path.exists()
        and less_filtered_points_npz_cache_path.exists()
    )

    if use_cache:
        if return_points == "all":
            points_data = mps.read_global_point_cloud(str(points_path))  # type: ignore
        else:
            points_data = None

        logger.debug("Loading pre-filtered points from %s", filtered_points_npz_cache_path)
        filtered_points_data = np.load(filtered_points_npz_cache_path)["points"]
        logger.debug("Loading pre-filtered points from %s", less_filtered_points_npz_cache_path)
        less_filtered_points_data = np.load(less_filtered_points_npz_cache_path)["points"]
    else:
        points_data = mps.read_global_point_cloud(str(points_path))  # type: ignore

        logger.debug("Loading + filtering points")
        assert points_path.exists()
        filtered_points_data = filter_points_from_confidence(
            points_data,
            threshold_invdep=0.0002,
            threshold_dep=0.01,
        )
        less_filtered_points_data = filter_points_from_confidence(
            points_data,
            threshold_invdep=0.001,
            threshold_dep=0.05,
        )
        filtered_points_data = np.array([x.position_world for x in filtered_points_data])  # type: ignore
        less_filtered_points_data = np.array([x.position_world for x in less_filtered_points_data])

        # Only save cache files if caching is enabled
        if cache_files:
            filtered_points_npz_cache_path.parent.mkdir(exist_ok=True, parents=True)
            np.savez_compressed(filtered_points_npz_cache_path, points=filtered_points_data)
            logger.debug("Cached filtered points to %s", filtered_points_npz_cache_path)
            np.savez_compressed(less_filtered_points_npz_cache_path, points=less_filtered_points_data)
            logger.debug("Cached less filtered points to %s", less_filtered_points_npz_cache_path)

    assert filtered_points_data.shape == (filtered_points_data.shape[0], 3)

    # RANSAC floor plane.
    # We consider points in the lowest 10% of the point cloud.
    filtered_zs = filtered_points_data[:, 2]
    zs = filtered_zs

    # Median-based outlier dropping, this doesn't work very well.
    # d = np.abs(zs - np.median(zs))
    # mdev = np.median(d)
    # zs = zs[d / mdev < 2.0]

    done = False
    best_z = 0.0
    while not done:
        # Slightly silly outlier dropping that just... works better.
        zs = np.sort(zs)[len(zs) // 10_000 : -len(zs) // 10_000]

        # Get bottom 10% or 15%.
        alpha = 0.1 if filtered_points_data.shape[0] < 10_000 else 0.15
        min_z = np.min(zs)
        max_z = np.max(zs)

        zs = zs[zs <= min_z + (max_z - min_z) * alpha]

        best_inliers = 0
        best_z = 0.0
        for i in range(10_000):
            z = np.random.choice(zs)
            inliers_bool = np.abs(zs - z) < 0.01
            inliers = np.sum(inliers_bool)
            if inliers > best_inliers:
                best_z = z
                best_inliers = inliers

        looser_inliers = np.sum(np.abs(filtered_zs - best_z) <= 0.075)
        if looser_inliers <= 3:
            # If we found a really small group... seems like noise. Let's remove the inlier points and re-compute.
            filtered_zs = filtered_zs[np.abs(filtered_zs - best_z) >= 0.01]
            zs = filtered_zs
        else:
            done = True

    # Re-fit plane to inliers.
    floor_z = float(np.median(zs[np.abs(zs - best_z) < 0.01]))
    
    # Select points based on return_points parameter
    if return_points == "filtered":
        vertices = filtered_points_data
    elif return_points == "less_filtered":
        vertices = less_filtered_points_data
    else:
        assert points_data is not None
        vertices = np.array([x.position_world for x in points_data])
    # Create colors based on z-values using percentiles to be robust to outliers
    z_values = vertices[:, 2]
    z_5th = np.percentile(z_values, 5)
    z_95th = np.percentile(z_values, 95)
    z_normalized = np.clip((z_values - z_5th) / (z_95th - z_5th), 0, 1)
    
    # Create a colormap that maps z-values to RGB colors
    colors = np.zeros((len(vertices), 4), dtype=np.uint8)
    
    # Red channel - increases with height
    colors[:, 0] = (255 * z_normalized).astype(np.uint8)
    # Green channel - inverse of height  
    colors[:, 1] = (255 * (1 - z_normalized)).astype(np.uint8)
    # Blue channel - varies sinusoidally with height
    colors[:, 2] = (128 + 127 * np.sin(z_normalized * 4 * np.pi)).astype(np.uint8)
    # Alpha channel - full opacity
    colors[:, 3] = 255
    
    # Create PointcloudContainer
    pc_container = Pointcloud.PointcloudContainer(vertices=vertices, colors=colors)
    
    return pc_container, vertices, floor_z

if __name__ == "__main__":
    tyro.cli(load_point_cloud_and_find_ground)