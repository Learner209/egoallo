from pathlib import Path
from typing import Optional
from typing import Tuple

import torch
from egoallo.type_stubs import Device
from egoallo.type_stubs import PathLike
from torch import Tensor


def get_device(device: Optional[Device] = None) -> torch.device:
    """Get torch device, defaulting to CUDA if available."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_path(path: PathLike) -> Path:
    """Convert path-like object to Path."""
    return Path(path) if not isinstance(path, Path) else path


def procrustes_align(
    points_y: Tensor,
    points_x: Tensor,
    fix_scale: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Perform Procrustes alignment between two point sets.

    Args:
        points_y: Target points (..., N, 3)
        points_x: Source points (..., N, 3)
        fix_scale: Whether to fix scale to 1

    Returns:
        Tuple of (scale, rotation, translation)
    """
    *dims, N, _ = points_y.shape
    device = points_y.device
    dtype = points_y.dtype
    N_tensor = torch.tensor(N, device=device, dtype=dtype)

    # Center points
    my = points_y.mean(dim=-2)
    mx = points_x.mean(dim=-2)
    y0 = points_y - my[..., None, :]
    x0 = points_x - mx[..., None, :]

    # Correlation
    C = torch.matmul(y0.transpose(-1, -2), x0) / N_tensor
    U, D, Vh = torch.linalg.svd(C, full_matrices=False)

    # Handle reflection
    S = torch.eye(3, device=device, dtype=dtype).expand(*dims, 3, 3)
    det = torch.det(U) * torch.det(Vh)
    S[..., -1, -1] = torch.where(det < 0, -1.0, 1.0)

    R = torch.matmul(U, torch.matmul(S, Vh))

    if fix_scale:
        s = torch.ones(*dims, 1, device=device, dtype=dtype)
    else:
        var = torch.sum(x0**2, dim=(-1, -2), keepdim=True) / N_tensor
        s = (
            torch.sum(D * S.diagonal(dim1=-2, dim2=-1), dim=-1, keepdim=True)
            / var[..., 0]
        )

    t = my - s * torch.matmul(R, mx[..., None])[..., 0]

    return s, R, t
