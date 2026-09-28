import numpy as np
import pytest

from covale.masks import AtlasMask, difference, intersection, union

AFFINE = np.eye(4)


def mask(values: list[bool], affine: np.ndarray = AFFINE) -> AtlasMask:
    return AtlasMask(np.array(values), affine)


def test_union() -> None:
    result = union(mask([True, False]), mask([False, True]))
    np.testing.assert_array_equal(result.data, [True, True])


def test_intersection() -> None:
    result = intersection(mask([True, True]), mask([False, True]))
    np.testing.assert_array_equal(result.data, [False, True])


def test_difference() -> None:
    result = difference(
        mask([True, True, True]),
        mask([False, True, False]),
        mask([False, False, True]),
    )
    np.testing.assert_array_equal(result.data, [True, False, False])


def test_incompatible_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        union(mask([True]), mask([True, False]))


def test_incompatible_affines() -> None:
    shifted = AFFINE.copy()
    shifted[0, 3] = 1
    with pytest.raises(ValueError, match="affine"):
        intersection(mask([True]), mask([True], shifted))
