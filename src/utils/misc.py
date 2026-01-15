import itertools
from ..utils.array_backend import np


def cartesian(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def cartesian_transpose(*arrays):
    broadcastable = np.ix_(*arrays)
    broadcasted = np.broadcast_arrays(*broadcastable)
    rows, cols = np.prod(broadcasted[0].shape), len(broadcasted)
    dtype = np.result_type(*arrays)

    out = np.empty(rows * cols, dtype=dtype)
    start, end = 0, rows
    for a in broadcasted:
        out[start:end] = a.reshape(-1)
        start, end = end, end + rows
    return out.reshape(cols, rows).T


def cartesian_transpose_pp(arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty((la, *map(len, arrays)), dtype=dtype)
    idx = slice(None), *itertools.repeat(None, la)
    for i, a in enumerate(arrays):
        arr[i, ...] = a[idx[:la - i]]
    return arr.reshape(la, -1).T


def cartesian_dot(arr1, arr2):
    """
    Compute the cartesian dot product between two arrays.

    Args:
        arr1: Array of shape (m, n) - first set of vectors
        arr2: Array of shape (k, n) - second set of vectors

    Returns:
        Array of shape (m, k) containing dot products between all pairs
        of vectors from arr1 and arr2
    """
    # Use broadcasting to compute all pairwise dot products efficiently
    # arr1[:, None, :] has shape (m, 1, n)
    # arr2[None, :, :] has shape (1, k, n)
    # The multiplication gives (m, k, n), then sum along last axis gives (m, k)
    return np.sum(arr1[:, None, :] * arr2[None, :, :], axis=-1)


def cartesian_pairs(A, B):
    """
    Generate the cartesian product of two arrays with potentially different inner shapes,
    returning aligned pairs suitable for vectorized functions f(x, m).

    Given:
    - A with shape (m, ...)
    - B with shape (k, ...)

    Returns two arrays:
    - A_rep with shape (m*k, ...) where each element of A is repeated k times
    - B_tile with shape (m*k, ...) where B is tiled m times

    This avoids attempting to stack heterogeneous shapes into a single array.
    """
    if A.ndim == 0 or B.ndim == 0:
        raise ValueError(
            "Inputs must be at least 1D (have leading batch dimension)")

    m = A.shape[0]
    k = B.shape[0]

    # Repeat A along a new axis of size k, then reshape to (m*k, ...)
    A_rep = np.repeat(A[:, None, ...], k, axis=1).reshape(m * k, *A.shape[1:])

    # Tile B along a new leading group of size m, then reshape to (m*k, ...)
    B_tile = np.tile(B[None, ...], (m, *([1] * (B.ndim))))
    B_tile = B_tile.reshape(m * k, *B.shape[1:])

    return A_rep, B_tile
