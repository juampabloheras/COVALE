import numpy as np
import pytest

from covale import dice


def test_identity() -> None:
    mask = np.array([True, False, True])
    assert dice(mask, mask) == 1.0


def test_symmetry() -> None:
    a = np.array([True, True, False])
    b = np.array([False, True, True])
    assert dice(a, b) == dice(b, a)


def test_disjoint_masks() -> None:
    assert dice([True, False], [False, True]) == 0.0


def test_partial_overlap() -> None:
    assert dice([True, True, False], [False, True, True]) == 0.5


def test_empty_masks() -> None:
    assert dice(np.zeros(3), np.zeros(3)) == 1.0


def test_incompatible_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        dice(np.zeros(2), np.zeros(3))
