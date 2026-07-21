"""Unit tests for shared costs and finite-memory Q-learning arithmetic."""

import numpy as np

from src.classes.costs import control_effort, compose_rho, shannon_entropy_rows, stage_cost
from src.finite_memory.q_learning import FiniteMemoryQLearning


def shannon_entropy(b: np.ndarray) -> float:
    b = np.atleast_2d(np.asarray(b, dtype=float))
    return float(shannon_entropy_rows(b)[0])


class _DummyCostMDP:
    """Minimal stand-in exposing the shared BeliefMDP cost API."""

    def r_exploration(self, B):
        B = np.atleast_2d(np.asarray(B, dtype=float))
        return shannon_entropy_rows(B)

    def c_effort(self, U):
        return control_effort(U)


def test_shannon_entropy_uniform():
    b = np.full(4, 0.25)
    H = shannon_entropy(b)
    np.testing.assert_allclose(H, np.log(4), rtol=1e-10)


def test_shannon_entropy_dirac():
    b = np.array([1.0, 0.0, 0.0])
    assert shannon_entropy(b) == 0.0


def test_compose_rho_and_stage_cost_agree():
    mdp = _DummyCostMDP()
    b = np.full(3, 1.0 / 3.0)
    u = np.array([1.0, 2.0])
    lam = 10.0
    got = stage_cost(mdp, b, u, lam=lam)
    expected = float(compose_rho(mdp.r_exploration(b), mdp.c_effort(u), lam=lam)[0, 0])
    np.testing.assert_allclose(got, expected)


class _DummyApprox:
    def __init__(self, n_states=4, n_actions=2):
        self.n_states = n_states
        self.n_actions = n_actions


class _DummyRollout:
    """Minimal rollout stub that yields a scripted trajectory for Q updates."""

    def __init__(self):
        self.t = 0
        self.traj = [
            (0, 1.0, 1),
            (1, 0.5, 2),
            (2, 0.0, 0),
        ]

    def reset(self, true_map=None, x0=None):
        self.t = 0
        return 0, {"belief": np.array([1.0, 0.0])}

    def step(self, u_idx):
        h, cost, h_next = self.traj[self.t % len(self.traj)]
        self.t += 1
        return h, cost, h_next, {"belief": np.array([0.5, 0.5])}


def test_q_learning_update_arithmetic():
    approx = _DummyApprox(n_states=3, n_actions=2)
    rollout = _DummyRollout()
    ql = FiniteMemoryQLearning(approx, rollout, beta=0.5, epsilon=0.0, seed=0)
    td = ql.update(0, 0, cost=2.0, h_next=1)
    assert ql.visits[0, 0] == 1
    np.testing.assert_allclose(ql.Q[0, 0], 2.0)
    np.testing.assert_allclose(td, 2.0)

    td2 = ql.update(0, 0, cost=0.0, h_next=1)
    np.testing.assert_allclose(ql.Q[0, 0], 1.0)
    assert ql.visits[0, 0] == 2
    assert td2 >= 0.0


def test_q_learning_train_episode_runs():
    approx = _DummyApprox(n_states=3, n_actions=2)
    rollout = _DummyRollout()
    ql = FiniteMemoryQLearning(approx, rollout, beta=0.9, epsilon=1.0, seed=1)
    info = ql.train_episode(horizon=3)
    assert "return" in info
    assert ql.visits.sum() == 3
