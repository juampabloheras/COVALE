import numpy as np


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)

    if a.shape != b.shape:
        raise ValueError("Masks must have the same shape.")

    denominator = int(a.sum() + b.sum())
    if denominator == 0:
        return 1.0

    overlap = np.logical_and(a, b).sum()
    return float(2 * overlap / denominator)
