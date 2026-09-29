import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np

from covale.masks import AtlasMask
from covale.registry.models import LabelSource, MaskSource, RegistryV2


def _inside(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Atlas mask path must remain inside atlas_registry.")
    if not path.is_file():
        raise ValueError(f"Registry asset does not exist: {relative_path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_image(
    path: Path,
    *,
    require_3d: bool = False,
) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    image = nib.load(str(path))
    if require_3d and len(image.shape) != 3:
        raise ValueError(f"Registry volumes must be 3D: {path}")
    affine = np.asarray(image.affine, dtype=float)
    if not np.isfinite(affine).all():
        raise ValueError(f"Registry volume affine must be finite: {path}")
    if abs(float(np.linalg.det(affine[:3, :3]))) < 1e-12:
        raise ValueError(f"Registry volume affine must be invertible: {path}")
    data = np.asarray(image.dataobj)
    if not np.isfinite(data).all():
        raise ValueError(f"Registry volume contains nonfinite voxels: {path}")
    return image, data


def load_label_volume(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image, data = _load_image(path, require_3d=True)
    if np.any(data < 0) or not np.equal(data, np.rint(data)).all():
        raise ValueError(f"Atlas '{path}' must contain nonnegative integer labels.")
    return data.astype(np.int64, copy=False), np.asarray(image.affine)


def _load_label_volume(
    registry: RegistryV2,
    root: Path,
    atlas_id: str,
    cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    if atlas_id in cache:
        return cache[atlas_id]
    atlas = registry.atlases[atlas_id]
    path = _inside(root, atlas.volume)
    if atlas.sha256 is not None and sha256_file(path) != atlas.sha256:
        raise ValueError(f"Atlas checksum does not match registry: {atlas_id}")
    loaded = load_label_volume(path)
    cache[atlas_id] = loaded
    return loaded


def load_region_mask(
    registry: RegistryV2,
    root: Path,
    region_id: str,
    *,
    atlas_cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> AtlasMask:
    try:
        region = registry.regions[region_id]
    except KeyError as error:
        raise ValueError(f"Unknown registry region ID: {region_id}") from error

    source = region.source
    if isinstance(source, MaskSource):
        path = _inside(root, source.path)
        image, data = _load_image(path)
        mask = data.astype(bool)
        affine = image.affine
    elif isinstance(source, LabelSource):
        data, affine = _load_label_volume(
            registry,
            root,
            source.atlas,
            atlas_cache,
        )
        available = set(int(value) for value in np.unique(data))
        missing = sorted(set(source.values) - available)
        if missing:
            raise ValueError(
                f"Region '{region_id}' references missing atlas labels: {missing}"
            )
        mask = np.isin(data, source.values)
    else:
        raise TypeError(f"Unsupported region source: {type(source).__name__}")

    if not np.any(mask):
        raise ValueError(f"Registry region is empty: {region_id}")
    return AtlasMask(mask, affine)
