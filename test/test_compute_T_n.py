"""
Test _compute_T_n functionality in BeliefMDP_n class.
Tests transition matrix computation correctness and properties.
"""
import numpy as np
import pytest
import shutil
import time
from pathlib import Path

from src.utils.map import load_obstacles_config
from src.classes.mapping import LidarGridMapVec
from src.classes.model import SingleIntegratorModel, LIDAR
from src.belief_quantized.belief_mdp_n import BeliefMDP_n_SLAM as BeliefMDP_n


@pytest.fixture
def belief_mdp_setup():
    """Setup common components for BeliefMDP_n tests."""
    all_obstacles, area = load_obstacles_config(environment='toy2')
    motion_model = SingleIntegratorModel(i_x=5.0, i_y=5.0, dt=1, max_v=5)
    sensor = LIDAR(fov=360, r_max=5, B=8)
    grid_map = LidarGridMapVec(
        x_min=area[0], x_max=area[1],
        y_min=area[2], y_max=area[3],
        quantization_level=3,
    )

    return {
        'obstacles': all_obstacles,
        'motion_model': motion_model,
        'sensor': sensor,
        'grid_map': grid_map
    }


@pytest.fixture
def clean_cache():
    """Optionally clean cache directory before tests; preserve after to speed reruns.

    Set env KEEP_T_CACHE=1 to skip pre-test cleanup. By default we keep cache after tests.
    """
    import os
    project_root = Path(__file__).parent.parent
    cache_dir = project_root / "cache/T_mat"

    # Clean before only if not preserving cache
    keep_cache = os.environ.get("KEEP_T_CACHE", "1") == "1"
    if not keep_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)

    yield

    # Always preserve cache after tests to speed up subsequent runs
    # To force cleanup after, set KEEP_T_CACHE=0 and manually remove if needed


def test_compute_T_n_basic(belief_mdp_setup, clean_cache):
    """Test basic _compute_T_n functionality."""
    setup = belief_mdp_setup
    n = 3  # Small quantization for fast testing

    beliefmdp = BeliefMDP_n(
        n=n,
        motion_model=setup['motion_model'],
        measurement_model=setup['sensor'],
        obstacles=setup['obstacles'],
        _map=setup['grid_map'],
        sigma_w=1.0,
        sigma_v=1.0
    )

    # Compute T_mat manually via _compute_T_n
    T_mat_computed = beliefmdp._compute_T_n()

    # Verify shape
    expected_shape = (beliefmdp.SQ.m_n, beliefmdp.SQ.m_n, beliefmdp.AQ.n_u)
    assert T_mat_computed.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {T_mat_computed.shape}"

    # Verify it matches the cached version
    assert np.allclose(T_mat_computed, beliefmdp.T_mat), \
        "Computed T_mat should match cached version"

    print(f"✓ T_mat shape verified: {T_mat_computed.shape}")


def test_compute_T_n_probability_properties(belief_mdp_setup, clean_cache):
    """Test that _compute_T_n produces valid probability distributions."""
    setup = belief_mdp_setup
    n = 3

    beliefmdp = BeliefMDP_n(
        n=n,
        motion_model=setup['motion_model'],
        measurement_model=setup['sensor'],
        obstacles=setup['obstacles'],
        _map=setup['grid_map']
    )

    T_mat = beliefmdp._compute_T_n()

    # Each feasible column should sum to 1; infeasible (x_i,u_k) columns may be zero
    for k in range(beliefmdp.AQ.n_u):
        col_sums = T_mat[:, :, k].sum(axis=0)
        feasible = beliefmdp.K_mask[:, k]
        assert np.allclose(col_sums[feasible], 1.0, atol=1e-10), \
            f"Action {k}: feasible columns should sum to 1"
        assert np.all(col_sums[~feasible] == 0.0), \
            f"Action {k}: infeasible columns should be zero"

    # All entries should be non-negative
    assert np.all(T_mat >= 0), "All transition probabilities should be non-negative"

    # Check for reasonable sparsity (should not be too sparse or too dense)
    non_zero_ratio = np.count_nonzero(T_mat) / T_mat.size
    assert 0.01 < non_zero_ratio < 0.5, \
        f"Unexpected sparsity: {non_zero_ratio:.4f} non-zero elements"

    print(f"✓ Probability properties verified")
    print(f"  Non-zero ratio: {non_zero_ratio:.4f}")


def test_compute_T_n_deterministic(belief_mdp_setup, clean_cache):
    """Test that _compute_T_n produces deterministic results."""
    setup = belief_mdp_setup
    n = 3

    beliefmdp1 = BeliefMDP_n(
        n=n,
        motion_model=setup['motion_model'],
        measurement_model=setup['sensor'],
        obstacles=setup['obstacles'],
        _map=setup['grid_map']
    )

    beliefmdp2 = BeliefMDP_n(
        n=n,
        motion_model=setup['motion_model'],
        measurement_model=setup['sensor'],
        obstacles=setup['obstacles'],
        _map=setup['grid_map']
    )

    T_mat1 = beliefmdp1._compute_T_n()
    T_mat2 = beliefmdp2._compute_T_n()

    assert np.allclose(T_mat1, T_mat2), \
        "_compute_T_n should produce identical results across instances"

    print("✓ Deterministic computation verified")


def test_compute_T_n_transition_symmetry(belief_mdp_setup, clean_cache):
    """Test that transitions have reasonable properties (not too concentrated)."""
    setup = belief_mdp_setup
    n = 3

    beliefmdp = BeliefMDP_n(
        n=n,
        motion_model=setup['motion_model'],
        measurement_model=setup['sensor'],
        obstacles=setup['obstacles'],
        _map=setup['grid_map']
    )

    T_mat = beliefmdp._compute_T_n()

    # For small actions, transitions should be nearly diagonal
    # For larger actions, transitions should spread out
    for k in range(beliefmdp.AQ.n_u):
        u = beliefmdp.AQ.U[k]
        u_mag = np.linalg.norm(u)

        # Compute average transition distance for this action
        transition_dists = []
        for i in range(beliefmdp.SQ.m_n):
            for j in range(beliefmdp.SQ.m_n):
                if T_mat[j, i, k] > 1e-6:
                    x_i = beliefmdp.SQ.X_n[i]
                    x_j = beliefmdp.SQ.X_n[j]
                    dist = np.linalg.norm(x_j - x_i)
                    transition_dists.append(dist * T_mat[j, i, k])

        avg_dist = np.mean(transition_dists) if transition_dists else 0
        print(f"Action {k} (||u||={u_mag:.2f}): average transition distance = {avg_dist:.3f}")

        # For non-zero actions, transitions should spread out
        if u_mag > 0.1:
            assert avg_dist > 0.01, \
                f"Action {k} should cause non-trivial transitions"


def test_compute_T_n_edge_cases(belief_mdp_setup, clean_cache):
    """Test _compute_T_n with edge cases."""
    setup = belief_mdp_setup
    n = 3

    beliefmdp = BeliefMDP_n(
        n=n,
        motion_model=setup['motion_model'],
        measurement_model=setup['sensor'],
        obstacles=setup['obstacles'],
        _map=setup['grid_map'],
        sigma_w=0.01,  # Very small noise
        sigma_v=0.01
    )

    T_mat = beliefmdp._compute_T_n()

    # Even with small noise, transitions should be smooth
    assert np.all(np.isfinite(T_mat)), "All transitions should be finite"
    assert np.all(T_mat >= 0), "All transitions should be non-negative"

    # No feasible column should be all zeros (each feasible state-action should have transitions)
    for k in range(beliefmdp.AQ.n_u):
        for i in range(beliefmdp.SQ.m_n):
            if beliefmdp.K_mask[i, k]:
                col_sum = T_mat[:, i, k].sum()
                assert col_sum > 1e-10, \
                    f"State {i} under action {k} should have valid transitions"

    print("✓ Edge cases handled correctly")


def test_compute_T_n_performance(belief_mdp_setup, clean_cache):
    """Test _compute_T_n performance and timing."""
    setup = belief_mdp_setup
    n_values = [3, 5]  # Small values for quick tests

    for n in n_values:
        beliefmdp = BeliefMDP_n(
            n=n,
            motion_model=setup['motion_model'],
            measurement_model=setup['sensor'],
            obstacles=setup['obstacles'],
            _map=setup['grid_map']
        )

        print(f"\nTesting n={n} (state space size: {beliefmdp.SQ.m_n}):")

        start_time = time.time()
        T_mat = beliefmdp._compute_T_n()
        elapsed = time.time() - start_time

        print(f"  Computation time: {elapsed:.2f} seconds")
        print(f"  T_mat size: {T_mat.nbytes / (1024**2):.2f} MB")

        # Should complete in reasonable time
        assert elapsed < 60, f"Computation took too long: {elapsed:.2f}s"

        # Verify shape scales as expected
        expected_size = beliefmdp.SQ.m_n ** 2 * beliefmdp.AQ.n_u
        actual_size = T_mat.size
        assert actual_size == expected_size, \
            f"T_mat size mismatch: expected {expected_size}, got {actual_size}"


def test_compute_T_n_with_different_parameters(belief_mdp_setup, clean_cache):
    """Test _compute_T_n with different noise parameters."""
    setup = belief_mdp_setup
    n = 3

    # Test with different sigma_w values
    sigma_w_values = [0.1, 0.5, 1.0, 2.0]

    T_matrices = {}
    for sigma_w in sigma_w_values:
        beliefmdp = BeliefMDP_n(
            n=n,
            motion_model=setup['motion_model'],
            measurement_model=setup['sensor'],
            obstacles=setup['obstacles'],
            _map=setup['grid_map'],
            sigma_w=sigma_w
        )

        T_mat = beliefmdp._compute_T_n()
        T_matrices[sigma_w] = T_mat

        print(f"sigma_w={sigma_w}: T_mat shape={T_mat.shape}")

    # Verify that larger noise increases entropy (more spread-out distributions)
    for i in range(len(sigma_w_values) - 1):
        sigma_w1, sigma_w2 = sigma_w_values[i], sigma_w_values[i + 1]
        T1, T2 = T_matrices[sigma_w1], T_matrices[sigma_w2]

        def mean_entropy(T):
            # compute entropy per (i,k) column over j, average over feasible columns
            Hs = []
            for k in range(beliefmdp.AQ.n_u):
                for i in range(beliefmdp.SQ.m_n):
                    if beliefmdp.K_mask[i, k]:
                        p = T[:, i, k]
                        p = p[p > 0]
                        Hs.append(-np.sum(p * np.log(p)))
            return np.mean(Hs) if Hs else 0.0

        H1 = mean_entropy(T1)
        H2 = mean_entropy(T2)

        assert H1 <= H2 + 1e-9, \
            f"Entropy should not decrease with larger σ_w (H1={H1:.6f}, H2={H2:.6f})"

    print("✓ Different noise parameters produce expected behavior")


def test_compute_T_n_state_reachability(belief_mdp_setup, clean_cache):
    """Test that transition matrix reflects physical reachability constraints."""
    setup = belief_mdp_setup
    n = 3

    beliefmdp = BeliefMDP_n(
        n=n,
        motion_model=setup['motion_model'],
        measurement_model=setup['sensor'],
        obstacles=setup['obstacles'],
        _map=setup['grid_map']
    )

    T_mat = beliefmdp._compute_T_n()

    # For a zero action, transitions should be nearly diagonal (self-loops)
    zero_action_idx = None
    for k in range(beliefmdp.AQ.n_u):
        if np.linalg.norm(beliefmdp.AQ.U[k]) < 0.01:
            zero_action_idx = k
            break

    if zero_action_idx is not None:
        T_zero = T_mat[:, :, zero_action_idx]

        # Diagonal should dominate for zero action
        diagonal_sum = np.trace(T_zero)
        total_sum = np.sum(T_zero)
        diagonal_ratio = diagonal_sum / total_sum if total_sum > 0 else 0

        assert diagonal_ratio > 0.5, \
            f"Zero action should produce mostly self-transitions (diagonal ratio: {diagonal_ratio:.3f})"

        print(f"✓ Zero action produces {diagonal_ratio:.1%} self-transitions")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
