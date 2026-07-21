"""
Test T_mat caching functionality in BeliefMDP_n class.
"""

import numpy as np
import pytest
import shutil
import time
from pathlib import Path

from src.utils.map import load_obstacles_config
from src.classes.mapping import LidarGridMapVec
from src.classes.model import DoubleIntegratorModel, LIDAR
from src.belief_quantized.belief_mdp_n import BeliefMDP_n_SLAM as BeliefMDP_n


@pytest.fixture
def belief_mdp_setup():
    """Setup common components for BeliefMDP_n tests."""
    all_obstacles, area = load_obstacles_config(environment='toy2')
    motion_model = DoubleIntegratorModel(p_x=5.0, p_y=5.0, v_x=0.0, v_y=0.0, dt=1.0, max_a=5.0)
    sensor = LIDAR(fov=360, r_max=10.0, B=8)
    lidar_map = LidarGridMapVec(
        x_min=area[0], x_max=area[1], y_min=area[2], y_max=area[3], quantization_level=3
    )

    return {
        'obstacles': all_obstacles,
        'motion_model': motion_model,
        'sensor': sensor,
        'lidar_map': lidar_map
    }


@pytest.fixture
def clean_cache():
    """Clean cache directory before and after tests."""
    # Cache directory is in project root
    project_root = Path(__file__).parent.parent
    cache_dir = project_root / "cache/T_mat"

    # Clean before test
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    yield

    # Clean after test
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def test_t_mat_caching_basic(belief_mdp_setup, clean_cache):
    """Test basic T_mat caching functionality."""
    setup = belief_mdp_setup
    n = 3  # Small quantization for fast testing

    # First instantiation - should compute T_mat
    beliefmdp1 = BeliefMDP_n(
        n=n,
        motion_model=setup['motion_model'],
        measurement_model=setup['sensor'],
        obstacles=setup['obstacles'],
        _map=setup['lidar_map']
    )

    # Verify T_mat has correct shape
    expected_shape = (beliefmdp1.SQ.m_n, beliefmdp1.SQ.m_n, beliefmdp1.AQ.n_u)
    assert beliefmdp1.T_mat.shape == expected_shape

    # Save the computed T_mat for comparison
    T_mat_original = beliefmdp1.T_mat.copy()

    # Second instantiation - should load from cache
    beliefmdp2 = BeliefMDP_n(
        n=n,
        motion_model=setup['motion_model'],
        measurement_model=setup['sensor'],
        obstacles=setup['obstacles'],
        _map=setup['lidar_map']
    )

    # Verify they're the same
    assert np.allclose(T_mat_original, beliefmdp2.T_mat), "Cached T_mat should match original"
    assert beliefmdp2.T_mat.shape == expected_shape


def test_t_n_method_returns_full_matrix(belief_mdp_setup, clean_cache):
    """Test that T_n() method returns the full T_mat."""
    setup = belief_mdp_setup
    n = 3

    beliefmdp = BeliefMDP_n(
        n=n,
        motion_model=setup['motion_model'],
        measurement_model=setup['sensor'],
        obstacles=setup['obstacles'],
        _map=setup['lidar_map']
    )

    # Test the T_n() method returns the full T_mat
    T_n_result = beliefmdp.T_n()
    assert np.array_equal(T_n_result, beliefmdp.T_mat), "T_n() should return the full T_mat"


def test_different_parameters_create_different_cache(belief_mdp_setup, clean_cache):
    """Test that different parameters create different cache files."""
    setup = belief_mdp_setup

    # Create with n=3
    beliefmdp1 = BeliefMDP_n(
        n=3,
        motion_model=setup['motion_model'],
        measurement_model=setup['sensor'],
        obstacles=setup['obstacles'],
        _map=setup['lidar_map']
    )

    # Create with n=4 - should compute new T_mat
    beliefmdp2 = BeliefMDP_n(
        n=4,
        motion_model=setup['motion_model'],
        measurement_model=setup['sensor'],
        obstacles=setup['obstacles'],
        _map=setup['lidar_map']
    )

    # Verify shapes are different due to different n
    assert beliefmdp1.T_mat.shape != beliefmdp2.T_mat.shape, "Different n should create different T_mat shapes"


def test_cache_file_metadata_validation(belief_mdp_setup, clean_cache):
    """Test that cache files correctly validate metadata."""
    setup = belief_mdp_setup
    n = 3

    # Create first instance
    beliefmdp1 = BeliefMDP_n(
        n=n,
        motion_model=setup['motion_model'],
        measurement_model=setup['sensor'],
        obstacles=setup['obstacles'],
        _map=setup['lidar_map']
    )

    # Verify cache file exists
    cache_path = beliefmdp1._get_cache_path()
    assert cache_path.exists(), "Cache file should exist after first instantiation"

    # Load and verify metadata
    data = np.load(cache_path)

    # Check that all expected metadata fields are present
    expected_fields = ['T_mat', 'n', 'state_bounds', 'max_val', 'map_shape',
                       'sigma_w', 'sigma_v', 'm_n', 'n_u', 'model', 'state_dim', 'kernel_version']
    for field in expected_fields:
        assert field in data, f"Metadata field '{field}' should be in cache file"

    # Verify metadata values
    assert data['n'] == n
    assert data['max_val'] == setup['motion_model'].max_a
    assert data['m_n'] == beliefmdp1.SQ.m_n
    assert data['n_u'] == beliefmdp1.AQ.n_u
    assert data['model'].item() == 'DoubleIntegratorModel'
    assert data['state_dim'] == 4


def test_t_mat_probability_properties(belief_mdp_setup, clean_cache):
    """Test that T_mat maintains probability properties."""
    setup = belief_mdp_setup
    n = 3

    beliefmdp = BeliefMDP_n(
        n=n,
        motion_model=setup['motion_model'],
        measurement_model=setup['sensor'],
        obstacles=setup['obstacles'],
        _map=setup['lidar_map']
    )

    T_mat = beliefmdp.T_mat

    # Each feasible column should sum to 1; infeasible columns should be 0
    for k in range(beliefmdp.AQ.n_u):
        col_sums = T_mat[:, :, k].sum(axis=0)
        feasible = beliefmdp.K_mask[:, k]
        assert np.allclose(col_sums[feasible], 1.0, atol=1e-10), f"Action {k}: feasible columns should sum to 1"
        assert np.allclose(col_sums[~feasible], 0.0, atol=1e-12), f"Action {k}: infeasible columns should be zero"

    # All entries should be non-negative
    assert np.all(T_mat >= 0), "All transition probabilities should be non-negative"


def test_cache_path_generation(belief_mdp_setup, clean_cache):
    """Test that cache paths are generated correctly with metadata."""
    setup = belief_mdp_setup

    beliefmdp1 = BeliefMDP_n(
        n=3,
        motion_model=setup['motion_model'],
        measurement_model=setup['sensor'],
        obstacles=setup['obstacles'],
        _map=setup['lidar_map']
    )

    beliefmdp2 = BeliefMDP_n(
        n=4,
        motion_model=setup['motion_model'],
        measurement_model=setup['sensor'],
        obstacles=setup['obstacles'],
        _map=setup['lidar_map']
    )

    path1 = beliefmdp1._get_cache_path()
    path2 = beliefmdp2._get_cache_path()

    # Different parameters should create different cache paths
    assert path1 != path2, "Different parameters should create different cache paths"

    # Both paths should be in the cache directory
    assert "cache/T_mat" in str(path1)
    assert "cache/T_mat" in str(path2)

    # Filenames should contain readable parameter info
    assert "n3" in path1.name
    assert "n4" in path2.name


def test_t_mat_computation_timing(belief_mdp_setup, clean_cache):
    """Test T_mat computation timing for first computation."""
    setup = belief_mdp_setup
    n = 3  # Small quantization for fast testing

    # Time the initial computation
    start_time = time.time()
    beliefmdp1 = BeliefMDP_n(
        n=n,
        motion_model=setup['motion_model'],
        measurement_model=setup['sensor'],
        obstacles=setup['obstacles'],
        _map=setup['lidar_map']
    )
    first_compute_time = time.time() - start_time

    print(f"\nFirst computation time: {first_compute_time:.2f} seconds")

    # Verify T_mat has correct shape
    expected_shape = (beliefmdp1.SQ.m_n, beliefmdp1.SQ.m_n, beliefmdp1.AQ.n_u)
    assert beliefmdp1.T_mat.shape == expected_shape

    # Verify computation was done (should have printed "Computing T_mat for the first time...")
    # Check cache file was created
    cache_path = beliefmdp1._get_cache_path()
    assert cache_path.exists(), "Cache file should exist after computation"


def test_t_mat_cache_load_timing(belief_mdp_setup, clean_cache):
    """Test T_mat cache loading timing."""
    setup = belief_mdp_setup
    n = 3

    # First instantiation (computation)
    print(f"\nCreating BeliefMDP_n for first time (computation)...")
    start_time = time.time()
    beliefmdp1 = BeliefMDP_n(
        n=n,
        motion_model=setup['motion_model'],
        measurement_model=setup['sensor'],
        obstacles=setup['obstacles'],
        _map=setup['lidar_map']
    )
    first_time = time.time() - start_time

    # Save the T_mat for comparison
    T_mat_computed = beliefmdp1.T_mat.copy()

    # Second instantiation (should load from cache)
    print(f"Creating BeliefMDP_n for second time (cache load)...")
    start_time = time.time()
    beliefmdp2 = BeliefMDP_n(
        n=n,
        motion_model=setup['motion_model'],
        measurement_model=setup['sensor'],
        obstacles=setup['obstacles'],
        _map=setup['lidar_map']
    )
    second_time = time.time() - start_time

    print(f"\nComputation time: {first_time:.2f} seconds")
    print(f"Cache load time: {second_time:.2f} seconds")
    print(f"Speedup: {first_time / second_time:.1f}x")

    # Cache load should be much faster
    assert second_time < first_time, "Cache load should be faster than computation"

    # Results should be identical
    assert np.allclose(T_mat_computed, beliefmdp2.T_mat), "Cached T_mat should match computed T_mat"

    # Verify cache load was actually used (speedup should be significant)
    assert (first_time /
            second_time) > 2.0, f"Cache should provide at least 2x speedup, got {first_time / second_time:.1f}x"


def test_t_mat_generation_consistency(belief_mdp_setup, clean_cache):
    """Test that T_mat generation produces consistent and valid transition matrices."""
    setup = belief_mdp_setup
    n = 3

    beliefmdp = BeliefMDP_n(
        n=n,
        motion_model=setup['motion_model'],
        measurement_model=setup['sensor'],
        obstacles=setup['obstacles'],
        _map=setup['lidar_map']
    )

    T_mat = beliefmdp.T_mat

    print(f"\nT_mat shape: {T_mat.shape}")
    print(f"State space size: {beliefmdp.SQ.m_n}")
    print(f"Action space size: {beliefmdp.AQ.n_u}")

    # Verify shape
    expected_shape = (beliefmdp.SQ.m_n, beliefmdp.SQ.m_n, beliefmdp.AQ.n_u)
    assert T_mat.shape == expected_shape, f"Expected shape {expected_shape}, got {T_mat.shape}"

    # Each column should sum to 1 (probability distribution over next states)
    for k in range(beliefmdp.AQ.n_u):
        col_sums = T_mat[:, :, k].sum(axis=0)
        assert np.allclose(col_sums, 1.0, atol=1e-10), f"Action {k}: columns should sum to 1"

    # All entries should be non-negative
    assert np.all(T_mat >= 0), "All transition probabilities should be non-negative"

    # Check for any zero columns (states with no valid transitions)
    for k in range(beliefmdp.AQ.n_u):
        for i in range(beliefmdp.SQ.m_n):
            col_sum = T_mat[:, i, k].sum()
            assert col_sum > 0, f"State {i} under action {k} should have at least one valid transition"

    print("✓ T_mat validation passed")


def test_t_mat_properties_across_different_n(belief_mdp_setup, clean_cache):
    """Test T_mat properties across different quantization levels."""
    setup = belief_mdp_setup
    n_values = [3, 4]  # Test multiple quantization levels

    times = {}
    shapes = {}

    for n in n_values:
        print(f"\nTesting with n={n}...")
        start_time = time.time()

        beliefmdp = BeliefMDP_n(
            n=n,
            motion_model=setup['motion_model'],
            measurement_model=setup['sensor'],
            obstacles=setup['obstacles'],
            _map=setup['lidar_map']
        )

        elapsed = time.time() - start_time
        times[n] = elapsed
        shapes[n] = beliefmdp.T_mat.shape

        print(f"  Shape: {shapes[n]}")
        print(f"  Time: {elapsed:.2f} seconds")

        # Verify probability properties (feasible vs infeasible)
        T_mat = beliefmdp.T_mat
        for k in range(beliefmdp.AQ.n_u):
            col_sums = T_mat[:, :, k].sum(axis=0)
            feasible = beliefmdp.K_mask[:, k]
            assert np.allclose(col_sums[feasible], 1.0,
                               atol=1e-10), f"n={n}, action {k}: feasible columns should sum to 1"
            assert np.allclose(col_sums[~feasible], 0.0,
                               atol=1e-12), f"n={n}, action {k}: infeasible columns should be zero"

        assert np.all(T_mat >= 0), f"n={n}: All transition probabilities should be non-negative"

    print(f"\nComputation times by n:")
    for n, t in times.items():
        print(f"  n={n}: {t:.2f}s, shape={shapes[n]}")

    # Verify shapes increase with n
    for i in range(len(n_values) - 1):
        n1, n2 = n_values[i], n_values[i + 1]
        assert shapes[n2][0] > shapes[n1][0], f"State space should grow: {shapes[n2][0]} > {shapes[n1][0]}"
        assert shapes[n2][2] > shapes[n1][2], f"Action space should grow: {shapes[n2][2]} > {shapes[n1][2]}"
