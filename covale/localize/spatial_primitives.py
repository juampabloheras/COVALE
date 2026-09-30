from typing import Literal

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt

from covale.registry.masks import AtlasMask

Direction = Literal[
    "anterior",
    "posterior",
    "superior",
    "inferior",
    "left",
    "right",
]

_DIRECTION_AXIS = {
    "left": (0, False),
    "right": (0, True),
    "posterior": (1, False),
    "anterior": (1, True),
    "inferior": (2, False),
    "superior": (2, True),
}


def _foreground_world(mask: AtlasMask) -> tuple[np.ndarray, np.ndarray]:
    indices = np.argwhere(mask.data)
    if not len(indices):
        raise ValueError("Spatial primitives require a non-empty mask.")
    affine_indices = np.pad(
        indices,
        ((0, 0), (0, 3 - indices.shape[1])),
    )
    return indices, nib.affines.apply_affine(mask.affine, affine_indices)


def directional_part(
    mask: AtlasMask,
    direction: Direction,
    fraction: float,
) -> AtlasMask:
    """Keep a directional fraction of a mask's RAS-space extent."""
    if not 0 < fraction <= 1:
        raise ValueError("Directional fraction must be greater than 0 and at most 1.")
    axis, positive = _DIRECTION_AXIS[direction]
    indices, points = _foreground_world(mask)
    coordinates = points[:, axis]
    lower = float(coordinates.min())
    upper = float(coordinates.max())
    threshold = (
        upper - fraction * (upper - lower)
        if positive
        else lower + fraction * (upper - lower)
    )
    selected = coordinates >= threshold if positive else coordinates <= threshold
    data = np.zeros(mask.data.shape, dtype=bool)
    data[tuple(indices[selected].T)] = True
    return AtlasMask(data, mask.affine)


def clip_plane(
    mask: AtlasMask,
    normal: tuple[float, float, float],
    offset_mm: float,
    side: Literal["positive", "negative"] = "positive",
) -> AtlasMask:
    """Clip a mask against a plane defined in RAS millimeter coordinates."""
    normal_array = np.asarray(normal, dtype=float)
    magnitude = float(np.linalg.norm(normal_array))
    if magnitude == 0:
        raise ValueError("Plane normal must not be zero.")
    normal_array /= magnitude
    indices, points = _foreground_world(mask)
    signed_distance = points @ normal_array - offset_mm
    selected = signed_distance >= 0 if side == "positive" else signed_distance <= 0
    data = np.zeros(mask.data.shape, dtype=bool)
    data[tuple(indices[selected].T)] = True
    return AtlasMask(data, mask.affine)


def morphology(
    mask: AtlasMask,
    operation: Literal["dilate", "erode"],
    distance_mm: float,
) -> AtlasMask:
    """Dilate or erode a mask using Euclidean distance in millimeters."""
    if not 0 < distance_mm <= 50:
        raise ValueError(
            "Morphology distance must be greater than 0 and at most 50 mm."
        )
    voxel_sizes = nib.affines.voxel_sizes(mask.affine)[: mask.data.ndim]
    if operation == "dilate":
        distances = distance_transform_edt(
            np.logical_not(mask.data),
            sampling=voxel_sizes,
        )
        data = distances <= distance_mm
    else:
        distances = distance_transform_edt(mask.data, sampling=voxel_sizes)
        data = np.logical_and(mask.data, distances > distance_mm)
    return AtlasMask(data, mask.affine)


def geometry(mask: AtlasMask) -> dict[str, object]:
    """Summarize a mask's physical geometry for agent plane selection."""
    _, points = _foreground_world(mask)
    return {
        "centroid_ras_mm": points.mean(axis=0).tolist(),
        "bounds_ras_mm": {
            "minimum": points.min(axis=0).tolist(),
            "maximum": points.max(axis=0).tolist(),
        },
        "voxel_count": int(np.count_nonzero(mask.data)),
        "voxel_sizes_mm": nib.affines.voxel_sizes(mask.affine).tolist(),
    }
