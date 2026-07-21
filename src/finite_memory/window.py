"""Finite-memory information windows for tabular Q-learning.

Paper notation (N >= 1):
    h_t^N = { y_{[t-N, t]}, u_{[t-N, t-1]} }
with |D^N| = |Y|^{N+1} |U|^N when Y, U are finite.

For N = 0 the window is just the current observation y_t.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class WindowSpec:
    """Dimensions of a finite observation/action alphabet and memory length."""

    n_y: int
    n_u: int
    N: int

    def __post_init__(self) -> None:
        if self.n_y < 1 or self.n_u < 1:
            raise ValueError("n_y and n_u must be >= 1")
        if self.N < 0:
            raise ValueError("memory N must be >= 0")

    @property
    def n_obs_slots(self) -> int:
        return self.N + 1

    @property
    def n_action_slots(self) -> int:
        return self.N

    @property
    def cardinality(self) -> int:
        """Number of distinct windows |D^N|."""
        if self.N == 0:
            return int(self.n_y)
        return int(self.n_y ** (self.N + 1) * self.n_u ** self.N)


class FiniteHistoryWindow:
    """Sliding window of quantized observation / action indices."""

    def __init__(self, spec: WindowSpec):
        self.spec = spec
        self.y_hist = np.full(spec.n_obs_slots, -1, dtype=np.int64)
        self.u_hist = np.full(max(spec.n_action_slots, 1), -1, dtype=np.int64)[: spec.n_action_slots]
        self._filled = 0  # number of observations seen so far

    def reset(self, y0: int) -> None:
        """Initialize with a single observation (and pad older slots with y0 / u=0)."""
        y0 = int(y0)
        if not (0 <= y0 < self.spec.n_y):
            raise ValueError(f"y0={y0} out of range [0, {self.spec.n_y})")
        self.y_hist[:] = y0
        if self.spec.N > 0:
            self.u_hist[:] = 0
        self._filled = 1

    def push(self, y_next: int, u: int) -> None:
        """Append action u then observation y_next (one environment step)."""
        y_next = int(y_next)
        u = int(u)
        if not (0 <= y_next < self.spec.n_y):
            raise ValueError(f"y_next={y_next} out of range [0, {self.spec.n_y})")
        if not (0 <= u < self.spec.n_u):
            raise ValueError(f"u={u} out of range [0, {self.spec.n_u})")

        if self.spec.N == 0:
            self.y_hist[0] = y_next
            self._filled = 1
            return

        # Drop oldest y and u; append new u then new y.
        self.y_hist[:-1] = self.y_hist[1:]
        self.y_hist[-1] = y_next
        self.u_hist[:-1] = self.u_hist[1:]
        self.u_hist[-1] = u
        self._filled = min(self._filled + 1, self.spec.n_obs_slots)

    @property
    def filled(self) -> int:
        return int(self._filled)

    def encode(self) -> int:
        """Map the current window to a unique integer in [0, |D^N|)."""
        return encode_window(self.y_hist, self.u_hist, self.spec)

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return self.y_hist.copy(), self.u_hist.copy()


def encode_window(y_hist: Sequence[int], u_hist: Sequence[int], spec: WindowSpec) -> int:
    """Encode observation/action index arrays into a flat window index."""
    y_hist = np.asarray(y_hist, dtype=np.int64).ravel()
    u_hist = np.asarray(u_hist, dtype=np.int64).ravel()
    if y_hist.shape[0] != spec.n_obs_slots:
        raise ValueError(f"expected {spec.n_obs_slots} observations, got {y_hist.shape[0]}")
    if u_hist.shape[0] != spec.n_action_slots:
        raise ValueError(f"expected {spec.n_action_slots} actions, got {u_hist.shape[0]}")

    if spec.N == 0:
        return int(y_hist[0])

    idx = 0
    for y in y_hist:
        idx = idx * spec.n_y + int(y)
    for u in u_hist:
        idx = idx * spec.n_u + int(u)
    return int(idx)


def decode_window(index: int, spec: WindowSpec) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of encode_window."""
    index = int(index)
    if not (0 <= index < spec.cardinality):
        raise ValueError(f"index={index} out of range [0, {spec.cardinality})")

    if spec.N == 0:
        return np.array([index], dtype=np.int64), np.array([], dtype=np.int64)

    u_hist = np.empty(spec.N, dtype=np.int64)
    for k in range(spec.N - 1, -1, -1):
        u_hist[k] = index % spec.n_u
        index //= spec.n_u
    y_hist = np.empty(spec.N + 1, dtype=np.int64)
    for k in range(spec.N, -1, -1):
        y_hist[k] = index % spec.n_y
        index //= spec.n_y
    return y_hist, u_hist
