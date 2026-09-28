from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class AtlasMask:
    data: np.ndarray
    affine: np.ndarray

    def __post_init__(self) -> None:
        data = np.asarray(self.data, dtype=bool)
        affine = np.asarray(self.affine, dtype=float)
        if affine.shape != (4, 4):
            raise ValueError("Mask affine must have shape (4, 4).")
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "affine", affine)


def load_mask(path: str | Path) -> AtlasMask:
    image = nib.load(str(path))
    return AtlasMask(np.asarray(image.dataobj, dtype=bool), image.affine)


def validate_compatibility(masks: tuple[AtlasMask, ...]) -> None:
    if not masks:
        raise ValueError("At least one mask is required.")

    reference = masks[0]
    for mask in masks[1:]:
        if mask.data.shape != reference.data.shape:
            raise ValueError("Masks must have the same shape.")
        if not np.allclose(mask.affine, reference.affine):
            raise ValueError("Masks must have compatible affine matrices.")


def union(*masks: AtlasMask) -> AtlasMask:
    validate_compatibility(masks)
    data = np.logical_or.reduce([mask.data for mask in masks])
    return AtlasMask(data, masks[0].affine)


def intersection(*masks: AtlasMask) -> AtlasMask:
    validate_compatibility(masks)
    data = np.logical_and.reduce([mask.data for mask in masks])
    return AtlasMask(data, masks[0].affine)


def difference(mask: AtlasMask, *subtract: AtlasMask) -> AtlasMask:
    masks = (mask, *subtract)
    validate_compatibility(masks)
    if not subtract:
        return AtlasMask(mask.data.copy(), mask.affine)
    removed = np.logical_or.reduce([other.data for other in subtract])
    return AtlasMask(np.logical_and(mask.data, np.logical_not(removed)), mask.affine)
