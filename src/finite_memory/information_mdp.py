"""Information MDP and its fixed-prior approximate finite-window model.

Exact model state: I_t = (b_{t-N}^-, h_t^N)
Approximate model state: hat{I}_t = (b^*, h_t^N) with effective state h_t^N only.

Stage costs are the shared belief costs (classes.costs.stage_cost) evaluated on
Psi(...), identical in form to BeliefMDP_n.rho_n.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ..belief_quantized.belief_mdp_n import BeliefMDP_n_Mapping
from ..classes.costs import stage_cost
from .window import FiniteHistoryWindow, WindowSpec, decode_window, encode_window


def _as_numpy(x: Any) -> np.ndarray:
    return x.get() if hasattr(x, "get") else np.asarray(x)


def _Y_n(mdp: BeliefMDP_n_Mapping) -> np.ndarray:
    return _as_numpy(mdp.Y_n)


def _U_n(mdp: BeliefMDP_n_Mapping) -> np.ndarray:
    return _as_numpy(mdp.AQ.U)


def _window_spec(mdp: BeliefMDP_n_Mapping, memory_N: int) -> WindowSpec:
    return WindowSpec(
        n_y=int(_Y_n(mdp).shape[0]),
        n_u=int(mdp.AQ.n_u),
        N=int(memory_N),
    )


def _uniform_prior(mdp: BeliefMDP_n_Mapping) -> np.ndarray:
    n = int(mdp.len_M)
    return np.full(n, 1.0 / n, dtype=float)


def _filter_update(
    mdp: BeliefMDP_n_Mapping,
    belief: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Bayesian map update via BeliefMDP_n_Mapping.F."""
    b = np.asarray(belief, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    out = mdp.F(b, np.asarray(x, dtype=float), y)
    b_new = out[0] if isinstance(out, list) else out
    return _as_numpy(b_new).astype(float).ravel()


def _observation_index(mdp: BeliefMDP_n_Mapping, y: np.ndarray) -> int:
    y = np.asarray(y, dtype=float).ravel()
    return int(mdp._find_observation_indices(y[np.newaxis, :])[0])


class InformationMDP:
    """Exact information MDP with evolving predictor b^- and finite window h.

    For known-pose mapping, Psi applies the shared map filter F along the window.
    """

    def __init__(
        self,
        mdp: BeliefMDP_n_Mapping,
        memory_N: int,
        *,
        lam: float = 1.0,
        exploration_type: Literal["information gain", "wasserstein distance"] = "information gain",
    ):
        self.mdp = mdp
        self.spec = _window_spec(mdp, memory_N)
        self.lam = float(lam)
        self.exploration_type = exploration_type
        self.mdp.exploration_type = exploration_type
        self.b_minus = _uniform_prior(mdp)
        self.window = FiniteHistoryWindow(self.spec)
        x_dim = int(_as_numpy(mdp.SQ.X_n).shape[1])
        self.pose_hist = np.zeros((self.spec.n_obs_slots, x_dim), dtype=float)

    def reset(self, b_minus: np.ndarray | None, y0_idx: int, x0: np.ndarray) -> int:
        self.b_minus = (
            _uniform_prior(self.mdp)
            if b_minus is None
            else np.asarray(b_minus, dtype=float).ravel()
        )
        self.window.reset(y0_idx)
        x0 = np.asarray(x0, dtype=float).ravel()
        self.pose_hist[:] = x0
        return self.window.encode()

    def psi(
        self,
        b_prior: np.ndarray,
        y_hist: np.ndarray,
        pose_hist: np.ndarray,
    ) -> np.ndarray:
        """Psi(b_prior, h) via successive shared filter updates along the window."""
        b = np.asarray(b_prior, dtype=float).ravel()
        y_hist = np.asarray(y_hist, dtype=np.int64).ravel()
        pose_hist = np.asarray(pose_hist, dtype=float)
        if pose_hist.ndim == 1:
            pose_hist = pose_hist[np.newaxis, :]
        Y = _Y_n(self.mdp)
        for k, y_idx in enumerate(y_hist):
            b = _filter_update(self.mdp, b, pose_hist[k], Y[int(y_idx)])
        return b

    def belief(self) -> np.ndarray:
        return self.psi(self.b_minus, self.window.y_hist, self.pose_hist)

    def stage_cost(self, u_idx: int, belief: np.ndarray | None = None) -> float:
        b = self.belief() if belief is None else belief
        u = _U_n(self.mdp)[int(u_idx)]
        return stage_cost(self.mdp, b, u, lam=self.lam)

    def advance_predictor(self) -> None:
        """Update b^- after a filled window by absorbing the oldest observation."""
        if self.spec.N == 0:
            return
        y_old = int(self.window.y_hist[0])
        x_old = self.pose_hist[0]
        self.b_minus = _filter_update(
            self.mdp, self.b_minus, x_old, _Y_n(self.mdp)[y_old]
        )

    def push(self, y_idx: int, u_idx: int, x_next: np.ndarray) -> int:
        """Push one step into the window and return the new encoded state."""
        if self.spec.N > 0 and self.window.filled >= self.spec.n_obs_slots:
            self.advance_predictor()
        self.window.push(y_idx, u_idx)
        x_next = np.asarray(x_next, dtype=float).ravel()
        if self.spec.N == 0:
            self.pose_hist[0] = x_next
        else:
            self.pose_hist[:-1] = self.pose_hist[1:]
            self.pose_hist[-1] = x_next
        return self.window.encode()


class ApproximateInformationMDP:
    """Fixed-prior approximate information MDP: state is the finite window h only.

    Cost: hat{rho}(h, u) = rho(Psi(b^*, h), u) with shared rho from classes.costs.
    """

    def __init__(
        self,
        mdp: BeliefMDP_n_Mapping,
        memory_N: int,
        b_star: np.ndarray | None = None,
        *,
        lam: float = 1.0,
        exploration_type: Literal["information gain", "wasserstein distance"] = "information gain",
    ):
        self.mdp = mdp
        self.spec = _window_spec(mdp, memory_N)
        self.b_star = (
            _uniform_prior(mdp)
            if b_star is None
            else np.asarray(b_star, dtype=float).ravel()
        )
        self.lam = float(lam)
        self.exploration_type = exploration_type
        self.mdp.exploration_type = exploration_type

    @property
    def n_states(self) -> int:
        return self.spec.cardinality

    @property
    def n_actions(self) -> int:
        return self.spec.n_u

    def psi(self, y_hist: np.ndarray, pose_hist: np.ndarray) -> np.ndarray:
        b = np.asarray(self.b_star, dtype=float).ravel()
        y_hist = np.asarray(y_hist, dtype=np.int64).ravel()
        pose_hist = np.asarray(pose_hist, dtype=float)
        if pose_hist.ndim == 1:
            pose_hist = pose_hist[np.newaxis, :]
        Y = _Y_n(self.mdp)
        for k, y_idx in enumerate(y_hist):
            b = _filter_update(self.mdp, b, pose_hist[k], Y[int(y_idx)])
        return b

    def psi_from_index(self, h_idx: int, pose_hist: np.ndarray) -> np.ndarray:
        y_hist, _ = decode_window(h_idx, self.spec)
        return self.psi(y_hist, pose_hist)

    def stage_cost(self, h_idx: int, u_idx: int, pose_hist: np.ndarray) -> float:
        belief = self.psi_from_index(h_idx, pose_hist)
        u = _U_n(self.mdp)[int(u_idx)]
        return stage_cost(self.mdp, belief, u, lam=self.lam)

    def encode(self, y_hist: np.ndarray, u_hist: np.ndarray) -> int:
        return encode_window(y_hist, u_hist, self.spec)

    def decode(self, h_idx: int) -> tuple[np.ndarray, np.ndarray]:
        return decode_window(h_idx, self.spec)
