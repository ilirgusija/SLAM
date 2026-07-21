"""Unit tests for finite-history window encode/decode."""

import numpy as np
import pytest

from src.finite_memory.window import (
    FiniteHistoryWindow,
    WindowSpec,
    decode_window,
    encode_window,
)


def test_window_spec_cardinality_N0():
    spec = WindowSpec(n_y=5, n_u=3, N=0)
    assert spec.cardinality == 5
    assert spec.n_obs_slots == 1
    assert spec.n_action_slots == 0


def test_window_spec_cardinality_N1():
    spec = WindowSpec(n_y=4, n_u=3, N=1)
    # |Y|^2 * |U|^1
    assert spec.cardinality == 4 * 4 * 3


def test_encode_decode_roundtrip_N0():
    spec = WindowSpec(n_y=7, n_u=2, N=0)
    for y in range(spec.n_y):
        idx = encode_window([y], [], spec)
        y_hist, u_hist = decode_window(idx, spec)
        assert y_hist.tolist() == [y]
        assert u_hist.tolist() == []


def test_encode_decode_roundtrip_N2():
    spec = WindowSpec(n_y=3, n_u=2, N=2)
    rng = np.random.default_rng(0)
    for _ in range(50):
        y_hist = rng.integers(0, spec.n_y, size=spec.n_obs_slots)
        u_hist = rng.integers(0, spec.n_u, size=spec.n_action_slots)
        idx = encode_window(y_hist, u_hist, spec)
        y2, u2 = decode_window(idx, spec)
        np.testing.assert_array_equal(y2, y_hist)
        np.testing.assert_array_equal(u2, u_hist)


def test_finite_history_window_push_and_encode():
    spec = WindowSpec(n_y=4, n_u=3, N=1)
    w = FiniteHistoryWindow(spec)
    w.reset(1)
    assert w.encode() == encode_window([1, 1], [0], spec)
    w.push(y_next=2, u=1)
    assert w.y_hist.tolist() == [1, 2]
    assert w.u_hist.tolist() == [1]
    assert w.encode() == encode_window([1, 2], [1], spec)


def test_encode_rejects_out_of_range():
    spec = WindowSpec(n_y=2, n_u=2, N=0)
    w = FiniteHistoryWindow(spec)
    with pytest.raises(ValueError):
        w.reset(5)
