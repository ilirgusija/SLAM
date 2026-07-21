"""Finite-memory Q-learning for the approximate information MDP.

Implements the paper's Ass. Q visit-count stepsize and tabular update
    Q <- (1-a) Q + a (C + beta min_a' Q(h', a'))

Also provides MappingRollout: the generative known-pose mapping simulator that
samples quantized (y, u) windows for training.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from .information_mdp import (
    ApproximateInformationMDP,
    _U_n,
    _as_numpy,
    _observation_index,
)
from .window import FiniteHistoryWindow


def _smoothstep01(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, 0.0, 1.0)
    return 3.0 * z**2 - 2.0 * z**3


class MappingRollout:
    """Known-pose active mapping rollout for finite-memory Q-learning.

    Hidden map is static; pose is known to the filter. Continuous measurements
    and actions are quantized to (Y_n, U_n) before entering the information window.
    """

    def __init__(
        self,
        approx_mdp: ApproximateInformationMDP,
        *,
        seed: int | None = None,
    ):
        self.approx = approx_mdp
        self.mdp = approx_mdp.mdp
        self.spec = approx_mdp.spec
        self.rng = np.random.default_rng(seed)

        self.sensor = self.mdp.sensor
        self.dt = float(self.mdp.motion_model.dt)
        bounds = _as_numpy(self.mdp.state_bounds)
        self.lower = np.asarray(bounds[:, 0], dtype=float)
        self.upper = np.asarray(bounds[:, 1], dtype=float)
        self.cov_x = _as_numpy(self.mdp.cov_x).astype(float)
        self.state_dim = int(self.cov_x.shape[0])

        self.window = FiniteHistoryWindow(self.spec)
        self.pose_hist = np.zeros((self.spec.n_obs_slots, self.state_dim), dtype=float)
        self.x = np.zeros(self.state_dim, dtype=float)
        self.true_map = None
        self.true_map_idx = None

    @property
    def all_maps(self) -> np.ndarray:
        return _as_numpy(self.mdp.all_maps_3d)

    def _detection_probability(self, ranges: np.ndarray) -> np.ndarray:
        eps = float(self.sensor.epsilon)
        r0 = float(self.sensor.r0)
        r1 = float(self.sensor.r1)
        r_max = float(self.sensor.r_max)
        eta_in = np.where(
            ranges <= eps,
            0.0,
            np.where(ranges >= r0, 1.0, _smoothstep01((ranges - eps) / (r0 - eps))),
        )
        eta_out = np.where(
            ranges <= r1,
            0.0,
            np.where(ranges >= r_max, 1.0, _smoothstep01((ranges - r1) / (r_max - r1))),
        )
        return np.clip(eta_in * (1.0 - eta_out), 0.0, 1.0)

    def _sample_truncated_range(self, mu: np.ndarray) -> np.ndarray:
        eps = float(self.sensor.epsilon)
        r_max = float(self.sensor.r_max)
        sigma_r = float(self.sensor.sigma_r)
        samples = mu + self.rng.normal(0.0, sigma_r, size=mu.shape)
        invalid = (samples < eps) | (samples > r_max)
        while np.any(invalid):
            samples[invalid] = mu[invalid] + self.rng.normal(
                0.0, sigma_r, size=int(np.sum(invalid))
            )
            invalid = (samples < eps) | (samples > r_max)
        return samples

    def observe(self, x: np.ndarray, true_map: np.ndarray) -> np.ndarray:
        """Sample a continuous range-bearing observation, flattened."""
        x = np.asarray(x, dtype=float).ravel()
        true_map = np.asarray(true_map, dtype=float)
        if true_map.ndim == 1:
            true_map = true_map.reshape(-1, 2)
        delta = true_map - x[:2]
        ranges = np.sqrt((delta**2).sum(axis=1))
        bearings = np.arctan2(delta[:, 1], delta[:, 0])
        L = true_map.shape[0]
        p_det = self._detection_probability(ranges)
        detected = self.rng.random(L) < p_det
        z = np.full((L, 2), np.nan)
        if np.any(detected):
            z[detected, 0] = self._sample_truncated_range(ranges[detected])
            phi = bearings[detected] + self.rng.normal(
                0.0, float(self.sensor.sigma_phi), size=int(np.sum(detected))
            )
            z[detected, 1] = np.arctan2(np.sin(phi), np.cos(phi))
        return z.ravel()

    def _step_pose(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        w = self.rng.multivariate_normal(np.zeros(self.state_dim), self.cov_x)
        return np.clip(x + u * self.dt + w, self.lower, self.upper)

    def reset(
        self,
        true_map: np.ndarray | int | None = None,
        x0: np.ndarray | None = None,
    ) -> tuple[int, dict[str, Any]]:
        maps = self.all_maps
        if true_map is None:
            self.true_map_idx = int(self.rng.integers(0, int(self.mdp.len_M)))
            self.true_map = maps[self.true_map_idx]
        elif isinstance(true_map, (int, np.integer)):
            self.true_map_idx = int(true_map)
            self.true_map = maps[self.true_map_idx]
        else:
            self.true_map = np.asarray(true_map, dtype=float)
            self.true_map_idx = None

        if x0 is None:
            self.x = np.zeros(self.state_dim, dtype=float)
            self.x[:2] = 5.0
        else:
            self.x = np.asarray(x0, dtype=float).ravel().copy()

        y_cont = self.observe(self.x, self.true_map)
        y_idx = _observation_index(self.mdp, y_cont)
        self.window.reset(y_idx)
        self.pose_hist[:] = self.x
        h_idx = self.window.encode()
        info = {
            "y_idx": y_idx,
            "x": self.x.copy(),
            "belief": self.approx.psi(self.window.y_hist, self.pose_hist),
        }
        return h_idx, info

    def step(self, u_idx: int) -> tuple[int, float, int, dict[str, Any]]:
        """Take action u_idx; return (h, cost, h', info)."""
        u_idx = int(u_idx)
        h_idx = self.window.encode()
        u = _U_n(self.mdp)[u_idx]

        cost = self.approx.stage_cost(h_idx, u_idx, self.pose_hist)

        x_next = self._step_pose(self.x, u)
        y_cont = self.observe(x_next, self.true_map)
        y_idx = _observation_index(self.mdp, y_cont)

        self.window.push(y_idx, u_idx)
        if self.spec.N == 0:
            self.pose_hist[0] = x_next
        else:
            self.pose_hist[:-1] = self.pose_hist[1:]
            self.pose_hist[-1] = x_next
        self.x = x_next

        h_next = self.window.encode()
        belief = self.approx.psi(self.window.y_hist, self.pose_hist)
        info = {
            "y_idx": y_idx,
            "x": self.x.copy(),
            "belief": belief,
            "u": u.copy(),
        }
        return h_idx, float(cost), h_next, info


class FiniteMemoryQLearning:
    """Tabular Q-learning on finite windows (hat h, u)."""

    def __init__(
        self,
        approx_mdp: ApproximateInformationMDP,
        rollout: MappingRollout,
        *,
        beta: float = 0.95,
        epsilon: float = 0.2,
        exploration: Literal["epsilon_greedy", "random"] = "epsilon_greedy",
        seed: int | None = None,
    ):
        if not (0.0 < beta < 1.0):
            raise ValueError("beta must lie in (0, 1)")
        self.approx = approx_mdp
        self.rollout = rollout
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.exploration = exploration
        self.rng = np.random.default_rng(seed)

        self.n_states = approx_mdp.n_states
        self.n_actions = approx_mdp.n_actions
        self.Q = np.zeros((self.n_states, self.n_actions), dtype=float)
        self.visits = np.zeros((self.n_states, self.n_actions), dtype=np.int64)
        self.episode_returns: list[float] = []
        self.td_errors: list[float] = []

    def alpha(self, h_idx: int, u_idx: int) -> float:
        """Visit-count stepsize 1 / (1 + #visits)."""
        return 1.0 / (1.0 + float(self.visits[h_idx, u_idx]))

    def select_action(self, h_idx: int) -> int:
        if self.exploration == "random" or self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmin(self.Q[h_idx]))

    def greedy_action(self, h_idx: int) -> int:
        return int(np.argmin(self.Q[h_idx]))

    def update(self, h_idx: int, u_idx: int, cost: float, h_next: int) -> float:
        """One Q-learning update; returns absolute TD error."""
        a = self.alpha(h_idx, u_idx)
        target = float(cost) + self.beta * float(np.min(self.Q[h_next]))
        td = target - float(self.Q[h_idx, u_idx])
        self.Q[h_idx, u_idx] = (1.0 - a) * self.Q[h_idx, u_idx] + a * target
        self.visits[h_idx, u_idx] += 1
        self.td_errors.append(abs(td))
        return abs(td)

    def train_episode(
        self,
        horizon: int,
        *,
        true_map: np.ndarray | int | None = None,
        x0: np.ndarray | None = None,
    ) -> dict:
        h, _ = self.rollout.reset(true_map=true_map, x0=x0)
        total_cost = 0.0
        for _ in range(int(horizon)):
            u = self.select_action(h)
            h_cur, cost, h_next, _ = self.rollout.step(u)
            self.update(h_cur, u, cost, h_next)
            total_cost += float(cost)
            h = h_next
        self.episode_returns.append(total_cost)
        return {"return": total_cost, "final_h": h}

    def train(
        self,
        n_episodes: int,
        horizon: int,
        *,
        true_maps: list | None = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """Run multiple episodes; optionally cycle through provided true maps."""
        for ep in range(int(n_episodes)):
            true_map = None
            if true_maps is not None and len(true_maps) > 0:
                true_map = true_maps[ep % len(true_maps)]
            info = self.train_episode(horizon, true_map=true_map)
            if verbose and (ep % max(1, n_episodes // 10) == 0 or ep == n_episodes - 1):
                print(f"episode {ep}: return={info['return']:.4f}")
        return self.Q

    def greedy_policy(self) -> np.ndarray:
        """Return argmin_u Q(h, u) for each window index."""
        return np.argmin(self.Q, axis=1).astype(np.int64)

    def rollout_greedy(
        self,
        horizon: int,
        *,
        true_map: np.ndarray | int | None = None,
        x0: np.ndarray | None = None,
        epsilon: float = 0.0,
    ) -> dict:
        """Evaluate the current greedy (or epsilon-greedy) policy without learning."""
        old_eps = self.epsilon
        old_expl = self.exploration
        self.epsilon = float(epsilon)
        self.exploration = "epsilon_greedy"
        h, info0 = self.rollout.reset(true_map=true_map, x0=x0)
        costs = []
        entropies = [
            float(-np.sum(info0["belief"] * np.log(np.maximum(info0["belief"], 1e-300))))
        ]
        for _ in range(int(horizon)):
            u = self.select_action(h) if epsilon > 0 else self.greedy_action(h)
            _, cost, h, info = self.rollout.step(u)
            costs.append(float(cost))
            b = info["belief"]
            entropies.append(float(-np.sum(b * np.log(np.maximum(b, 1e-300)))))
        self.epsilon = old_eps
        self.exploration = old_expl
        return {
            "costs": np.asarray(costs),
            "entropies": np.asarray(entropies),
            "total_cost": float(np.sum(costs)),
        }
