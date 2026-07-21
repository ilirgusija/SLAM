"""Integration tests for mapping information MDP (requires Q_n build; may be slow)."""

import os

import numpy as np
import pytest

# Prefer CPU for CI / login-node smoke tests.
os.environ.setdefault("USE_CUPY", "false")


def _tiny_mapping_mdp(
    *,
    pose_n: int = 2,
    map_n: int = 2,
    obs_n: int = 2,
    action_n: int = 2,
    num_landmarks: int = 1,
):
    """Minimal LandmarkMap + BeliefMDP_n_Mapping for smoke tests."""
    from src.belief_quantized.belief_mdp_n import BeliefMDP_n_Mapping
    from src.classes.mapping import LandmarkMap
    from src.classes.model import RangeBearingSensor, SingleIntegratorModel

    landmark_map = LandmarkMap(
        x_min=0.0,
        x_max=10.0,
        y_min=0.0,
        y_max=10.0,
        n=map_n,
        num_landmarks=num_landmarks,
    )
    motion_model = SingleIntegratorModel(i_x=5.0, i_y=5.0, dt=1.0, max_v=4.0)
    epsilon, r_max = 0.1, 4.0
    sensor = RangeBearingSensor(
        r_max=r_max,
        epsilon=epsilon,
        sigma_r=0.5,
        sigma_phi=0.3,
        r0=epsilon + 0.1 * (r_max - epsilon),
        r1=epsilon + 0.9 * (r_max - epsilon),
    )
    cov_y = np.diag(np.tile([0.5**2, 0.3**2], num_landmarks))
    mdp = BeliefMDP_n_Mapping(
        n=pose_n,
        motion_model=motion_model,
        measurement_model=sensor,
        obstacles=[],
        _map=landmark_map,
        sigma_w=0.1,
        cov_y=cov_y,
        exploration_type="information gain",
        obs_n=obs_n,
        action_n=action_n,
    )
    assert mdp.Q_n is not None
    assert mdp.all_maps_3d is not None
    return mdp


@pytest.fixture(scope="module")
def mapping_stack():
    from src.finite_memory.information_mdp import ApproximateInformationMDP
    from src.finite_memory.q_learning import FiniteMemoryQLearning, MappingRollout

    mdp = _tiny_mapping_mdp()
    approx = ApproximateInformationMDP(mdp, memory_N=0, lam=1.0)
    rollout = MappingRollout(approx, seed=0)
    ql = FiniteMemoryQLearning(approx, rollout, beta=0.9, epsilon=0.5, seed=0)
    return mdp, approx, rollout, ql


def test_alphabets_finite(mapping_stack):
    mdp, approx, _, _ = mapping_stack
    n_y = int(mdp.Y_n.shape[0])
    n_u = int(mdp.AQ.n_u)
    assert n_y >= 1
    assert n_u >= 1
    assert mdp.len_M == 4  # map_n=2, L=1 -> (2^2)^1 = 4
    assert approx.n_states == n_y  # N=0
    assert approx.n_actions == n_u


def test_psi_is_probability(mapping_stack):
    mdp, approx, rollout, _ = mapping_stack
    h, info = rollout.reset(true_map=0)
    b = info["belief"]
    assert b.shape == (mdp.len_M,)
    np.testing.assert_allclose(b.sum(), 1.0, atol=1e-8)
    assert np.all(b >= -1e-12)


def test_filter_matches_mdp_F(mapping_stack):
    from src.finite_memory.information_mdp import _filter_update, _uniform_prior

    mdp, _, rollout, _ = mapping_stack
    rollout.reset(true_map=0)
    b0 = _uniform_prior(mdp)
    x = rollout.x.copy()
    y_cont = rollout.observe(x, rollout.true_map)
    b_fast = _filter_update(mdp, b0, x, y_cont)
    b_ref = mdp.F(b0, x, y_cont)[0]
    b_ref = b_ref.get() if hasattr(b_ref, "get") else np.asarray(b_ref)
    np.testing.assert_allclose(b_fast, b_ref, atol=1e-6, rtol=1e-5)


def test_rollout_step_and_q_train_smoke(mapping_stack):
    _, approx, rollout, ql = mapping_stack
    h0, _ = rollout.reset(true_map=0)
    h, cost, h_next, info = rollout.step(0)
    assert h == h0
    assert np.isfinite(cost)
    assert 0 <= h_next < approx.n_states
    assert "belief" in info

    ql.train(n_episodes=3, horizon=5, true_maps=[0, 1], verbose=False)
    assert len(ql.episode_returns) == 3
    assert ql.visits.sum() == 3 * 5
    policy = ql.greedy_policy()
    assert policy.shape == (approx.n_states,)
