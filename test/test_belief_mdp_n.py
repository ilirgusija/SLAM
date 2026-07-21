from src.utils.metrics import tvd, W1_m, W1_state_simple
from src.belief_quantized.belief_mdp_n import BeliefMDP_n_SLAM as BeliefMDP_n
from src.classes.model import SingleIntegratorModel, LIDAR
from src.classes.mapping import LidarGridMapVec
from src.utils.map import load_obstacles_config
import sys
import matplotlib.pyplot as plt
import pytest
import os
import time
import numpy as np

# Ensure Numba is enabled by default in CI; individual tests will toggle
os.environ.setdefault("SLAM_USE_NUMBA", "1")


def test_belief_quantizer_codebook_equivalence():
    from src.classes.quantizer import BeliefQuantizer
    from math import comb

    M = 6
    N_n = 4
    cardinality = comb(M + N_n - 1, N_n - 1)

    # Reference (numpy+itertools) implementation
    bq_ref = BeliefQuantizer(M, N_n)
    codebook_ref = bq_ref._generate_codebook_fast()

    # Numba-clean static function
    codebook_numba = BeliefQuantizer._generate_codebook_numba_clean(cardinality, M, N_n)

    # Same shape and same elements (order may differ; sort rows for comparison)
    assert codebook_ref.shape == codebook_numba.shape
    ref_sorted = np.sort(codebook_ref, axis=1)
    numba_sorted = np.sort(codebook_numba, axis=1)
    # Sort rows lexicographically
    ref_rows = np.array(sorted(ref_sorted.tolist()))
    numba_rows = np.array(sorted(numba_sorted.tolist()))
    np.testing.assert_allclose(ref_rows, numba_rows, atol=0, rtol=0)


def test_belief_quantizer_codebook_perf_smoke():
    from src.classes.quantizer import BeliefQuantizer
    from math import comb

    M = 7
    N_n = 5
    cardinality = comb(M + N_n - 1, N_n - 1)

    # Warm-up JIT if enabled
    _ = BeliefQuantizer._generate_codebook_numba_clean(cardinality, M, N_n)

    # Time both versions
    t0 = time.time()
    codebook_ref = BeliefQuantizer(M, N_n)._generate_codebook_fast()
    t1 = time.time()
    codebook_numba = BeliefQuantizer._generate_codebook_numba_clean(cardinality, M, N_n)
    t2 = time.time()

    # Sanity: same cardinality
    assert codebook_ref.shape == codebook_numba.shape

    # Print for visibility in test logs
    print(f"codebook_fast: {t1 - t0:.4f}s, numba_clean: {t2 - t1:.4f}s")

try:
    from tqdm.auto import tqdm  # better auto-detection for terminals/notebooks
except Exception:
    tqdm = None



def _init_belief_mdp_n(n_m: int = 3) -> BeliefMDP_n:
    all_obstacles, area = load_obstacles_config(environment='toy2')
    motion_model = SingleIntegratorModel(i_x=5.0, i_y=5.0, dt=0.1, max_v=5)
    sensor = LIDAR(fov=360, r_max=5, B=8)
    # LidarGridMapVec uses quantization_level parameter
    quantization_level = n_m
    grid_map = LidarGridMapVec(
        x_min=area[0], x_max=area[1],
        y_min=area[2], y_max=area[3],
        quantization_level=quantization_level,
    )
    belief_mdp_n = BeliefMDP_n(
        n=11,
        motion_model=motion_model,
        measurement_model=sensor,
        obstacles=all_obstacles,
        _map=grid_map,
        sigma_w=1,
        sigma_v=1,
    )
    belief_mdp_n.map.seed_from_obstacles(all_obstacles)
    return belief_mdp_n


def _ground_truth_indices(belief_mdp_n: BeliefMDP_n):
    X_n = belief_mdp_n.SQ.get_quantized_points()
    H, W = belief_mdp_n.map.occupancy_map.height, belief_mdp_n.map.occupancy_map.width

    # Ground-truth state at [5,5]
    target_state = np.array([5.0, 5.0])
    x_dists = np.linalg.norm(X_n - target_state, axis=1)
    X_ind = int(np.argmin(x_dists))

    # Choose a deterministic true map (same as previous tests)
    true_map = np.array([[1, 0, 1],
                         [0, 0, 0],
                         [0, 0, 0]], dtype=np.uint8)
    M_ind = int(belief_mdp_n.map_to_bits(true_map))

    return X_ind, M_ind, X_n, H, W


def validate_basic_properties():
    """
    Combined sanity checks: shape consistency, probability measure validity,
    and finite normalization across typical inputs.
    """
    bmdp = _init_belief_mdp_n()
    X_ind, M_ind, X_n, H, W = _ground_truth_indices(bmdp)
    m_n = len(X_n)
    M_size = 2 ** (H * W)

    # Random belief over joint space
    pi = np.random.dirichlet(np.ones(m_n * M_size)).reshape(m_n, M_size)
    pi = pi / pi.sum()

    # Action and observations
    u = np.array([0.0, 0.0])
    y = np.ones((1, bmdp.sensor.B)) * bmdp.sensor.r_max / 2

    # T shape - now using T_mat directly
    u_idx = bmdp.AQ.get_quantized_index(u)
    T_n = bmdp.T_mat[:, :, u_idx]  # Extract transition matrix for action u
    assert T_n.shape == (m_n, m_n)

    # Integral shape
    integral = T_n @ pi
    assert integral.shape == (m_n, M_size)

    # Map space shape
    M = bmdp.generate_space_of_maps()
    assert M.shape == (M_size, H * W)

    # Roundtrip consistency: bits <-> map for several samples
    # Also confirm reshape orientation alignment with bits_to_map
    for test_bits in [0, 1, min(3, M_size - 1), M_size // 2, M_size - 1]:
        m2d = bmdp.bits_to_map(int(test_bits), (H, W))
        rt_bits = int(bmdp.map_to_bits(m2d))
        assert rt_bits == int(test_bits), (
            f"Roundtrip mismatch: bits {test_bits} -> map -> bits {rt_bits}")
        # Compare one-hot expectation via @M then reshape to bits_to_map
        onehot = np.zeros(M_size, dtype=np.float64)
        onehot[int(test_bits)] = 1.0
        E_flat = onehot @ M
        E_grid = E_flat.reshape(H, W)
        # Expect equal to m2d or possibly vertically flipped depending on display origin
        if not np.array_equal(E_grid, m2d):
            # Try vertical flip to account for imshow origin
            assert np.array_equal(np.flipud(E_grid), m2d) or np.array_equal(E_grid, np.flipud(m2d)), (
                "Map expectation reshape does not match bits_to_map orientation.")

    # Q-vectorized aggregation shape
    Q_matrix = bmdp.Q_vectorized(y, X_n, M[0].reshape(H, W))
    assert Q_matrix.shape[0] == m_n

    # F returns valid probabilities
    post_out = bmdp.F(pi, u, y)
    if isinstance(post_out, (list, tuple)):
        post_arr = np.asarray(post_out)
    else:
        post_arr = np.asarray(post_out)
    # Accept either (m_n, M_size) or (k, m_n, M_size) with k>=1
    assert post_arr.ndim in (2, 3)
    if post_arr.ndim == 3:
        assert post_arr.shape[1:] == (m_n, M_size)
        pi_new = post_arr[0]
    else:
        assert post_arr.shape == (m_n, M_size)
        pi_new = post_arr
    assert np.all(pi_new >= 0)
    s = float(np.sum(pi_new))
    assert np.isfinite(s) and abs(s - 1.0) < 1e-6

    # Robust normalization sanity checks (minimal, inline)
    small_probs = np.array([1e-300, 2e-300, 1e-300, 3e-300])
    robust_norm = bmdp._robust_normalize(small_probs)
    assert np.isfinite(robust_norm).all()
    assert abs(robust_norm.sum() - 1.0) < 1e-12

    zero_probs = np.zeros(4)
    robust_zeros = bmdp._robust_normalize(zero_probs)
    assert np.isfinite(robust_zeros).all()
    assert abs(robust_zeros.sum() - 1.0) < 1e-12
    assert np.allclose(robust_zeros, 0.25)


def run_belief_updates(priors: dict, N: int = 15, compute_metrics: bool = True, prior_filter: list | None = None):
    """
    Run belief updates for each prior independently. Returns a results dict.
    Each prior gets a fresh BeliefMDP_n starting at [5,5].
    """
    results = {}

    # Precompute environment artifacts used across priors
    tmp_bmdp = _init_belief_mdp_n()
    X_ind, M_ind, X_n, H, W = _ground_truth_indices(tmp_bmdp)
    M = tmp_bmdp.generate_space_of_maps()
    all_obstacles, _ = load_obstacles_config(environment='toy2')
    all_obstacle_segments = []
    for obs_i in all_obstacles:
        all_obstacle_segments += obs_i.update()

    # Action used for repeated updates
    u = np.array([-1.0, -1.0])

    # Construct true joint (for metrics)
    m_n = len(X_n)
    M_size = 2 ** (H * W)
    true_joint = np.zeros((m_n, M_size), dtype=np.float64)
    true_joint[X_ind, M_ind] = 1.0

    # Optionally filter which priors to run
    priors_to_run = priors
    prior_filter = ["state_known_map_uniform"]
    if prior_filter is not None:
        priors_to_run = {k: v for k, v in priors.items() if k in prior_filter}

    iterator_priors = priors_to_run.items()
    if tqdm is None:
        raise RuntimeError("tqdm is not installed. Please install tqdm to use this function.")
    if tqdm is not None:
        iterator_priors = tqdm(
            list(priors_to_run.items()),
            total=len(priors_to_run),
            desc="Priors",
            unit="prior",
            dynamic_ncols=True,
            leave=False,
            disable=not sys.stdout.isatty(),
        )

    for prior_item in iterator_priors:
        if isinstance(prior_item, tuple) and len(prior_item) == 2:
            prior_name, pi = prior_item
        else:
            prior_name, pi = prior_item  # fallback, should not happen
        bmdp = _init_belief_mdp_n()
        bmdp.motion_model.x = np.array([5.0, 5.0])

        pi = np.asarray(pi, dtype=np.float64)
        pi = pi / pi.sum()

        map_prior = pi.sum(axis=0)
        state_prior = pi.sum(axis=1)
        if map_prior.sum() > 1:
            map_prior = map_prior / map_prior.sum()
        if state_prior.sum() > 1:
            state_prior = state_prior / state_prior.sum()

        true_pos = bmdp.motion_model.x
        iter_updates = range(N)
        if tqdm is not None:
            iter_updates = tqdm(
                iter_updates,
                total=N,
                leave=False,
                desc=f"{prior_name} updates",
                unit="it",
                dynamic_ncols=True,
                disable=not sys.stdout.isatty(),
            )
        for _ in iter_updates:
            true_pos = bmdp.motion_model.update(u)
            y = bmdp.sensor.get_laser_ref(all_obstacle_segments, true_pos)
            if y.ndim == 1:
                y = y[np.newaxis, :]
            pi = bmdp.F(pi, u, y)[0]

        posterior = pi
        posterior_map = posterior.sum(axis=0)
        posterior_state = posterior.sum(axis=1)
        posterior_map = posterior_map / posterior_map.sum()
        posterior_state = posterior_state / posterior_state.sum()

        result = {
            'prior_name': prior_name,
            'belief_mdp_n': bmdp,
            'prior_belief': pi,
            'posterior_belief': posterior,
            'state_prior': state_prior,
            'posterior_state': posterior_state,
            'map_prior': map_prior,
            'posterior_map': posterior_map,
            'true_joint': true_joint,
            'state_true': true_joint.sum(axis=1),
            'map_true': true_joint.sum(axis=0),
            'X_n': X_n,
            'M': M,
            'final_position': true_pos,
        }

        if compute_metrics:
            tv_prior_true = tvd(pi, true_joint)
            tv_post_true = tvd(posterior, true_joint)
            tv_prior_post = tvd(pi, posterior)
            # State-space W1 using simple sorter-based heuristic
            w1_state_prior_true = W1_state_simple(state_prior, result['state_true'], X_n)
            w1_state_prior_post = W1_state_simple(state_prior, posterior_state, X_n)
            w1_state_post_true = W1_state_simple(posterior_state, result['state_true'], X_n)
            # Map-space W1 using Hamming metric
            w1_map_prior_true = W1_m(map_prior, result['map_true'], M)
            w1_map_prior_post = W1_m(map_prior, posterior_map, M)
            w1_map_post_true = W1_m(posterior_map, result['map_true'], M)

            result['metrics'] = {
                'tv_prior_true': tv_prior_true,
                'tv_post_true': tv_post_true,
                'tv_prior_post': tv_prior_post,
                'w1_state_prior_true': w1_state_prior_true,
                'w1_map_prior_true': w1_map_prior_true,
                'w1_state_post_true': w1_state_post_true,
                'w1_map_post_true': w1_map_post_true,
                'w1_state_prior_post': w1_state_prior_post,
                'w1_map_prior_post': w1_map_prior_post,
            }

        results[prior_name] = result

    return results


def visualize_results(results: dict, output_dir: str = "output"):
    first = next(iter(results.values()))
    bmdp = first['belief_mdp_n']
    H, W = bmdp.map.occupancy_map.height, bmdp.map.occupancy_map.width
    x_min, y_min = bmdp.map.occupancy_map.left_lower
    x_max, y_max = bmdp.map.occupancy_map.right_upper
    extent = [x_min, x_max, y_min, y_max]

    sq = bmdp.state_quantizer
    nQ = sq.n
    X_n = sq.get_quantized_points()

    def _ab_posterior_ratio_for_cell(cell_i, cell_j):
        """
        Build two maps that differ only at (cell_i, cell_j) and compute
        posterior map odds under one static scan from current true pose.
        Prints P(m=1 at cell)/P(m=0 at cell) using a 2-support prior.
        """
        # Prepare prior: state delta at nearest quantized to current pose
        x_target = bmdp.motion_model.x
        x_dists = np.linalg.norm(X_n - x_target, axis=1)
        x_idx = int(np.argmin(x_dists))
        H, W = bmdp.map.occupancy_map.height, bmdp.map.occupancy_map.width
        M_size = 2 ** (H * W)

        # Two maps: identical except target cell
        m0 = np.zeros((H, W), dtype=np.uint8)
        m1 = np.zeros((H, W), dtype=np.uint8)
        m1[cell_i, cell_j] = 1
        b0 = int(bmdp.map_to_bits(m0))
        b1 = int(bmdp.map_to_bits(m1))

        pi = np.zeros((len(X_n), M_size), dtype=np.float64)
        pi[x_idx, b0] = 0.5
        pi[x_idx, b1] = 0.5

        # One observation from current obstacles
        all_obstacles, _ = load_obstacles_config(environment='toy2')
        m_segs = []
        for obs in all_obstacles:
            m_segs += obs.update()
        u = np.array([0.0, 0.0])
        y = bmdp.sensor.get_laser_ref(m_segs, x_target)
        if y.ndim == 1:
            y = y[np.newaxis, :]

        post = bmdp.F(pi, u, y)[0]
        post_map = post.sum(axis=0)
        # Odds of occupied vs free at that cell given the two-map support
        num = float(post_map[b1])
        den = float(post_map[b0]) if post_map[b0] > 0 else np.finfo(float).eps
        ratio = num / den
        print(f"[AB-odds] cell=({cell_i},{cell_j}) P(1)/P(0)={ratio:.3e} \n  P1={num:.3e}, P0={den:.3e}")
        return ratio

    def state_to_quant_grid(prob_vec):
        grid = np.zeros((nQ, nQ), dtype=np.float64)
        qj = np.floor((X_n[:, 0] - sq.x_min) / sq.delta_x).astype(int)
        qi = np.floor((X_n[:, 1] - sq.y_min) / sq.delta_y).astype(int)
        qj = np.clip(qj, 0, nQ - 1)
        qi = np.clip(qi, 0, nQ - 1)
        for k, p in enumerate(prob_vec):
            i_plot = (nQ - 1) - qi[k]
            grid[i_plot, qj[k]] += p
        return grid

    for prior_name, data in results.items():
        state_prior = data['state_prior']
        posterior_state = data['posterior_state']
        map_prior = data['map_prior']
        posterior_map = data['posterior_map']
        M = data['M']

        prior_state_grid = state_to_quant_grid(state_prior)
        post_state_grid = state_to_quant_grid(posterior_state)

        prior_E_m = (map_prior @ M).reshape(H, W)
        post_E_m = (posterior_map @ M).reshape(H, W)

        # ΔE[m] diagnostic and NE vs SW metric
        delta_E = post_E_m - prior_E_m
        # Split into quadrants
        mid_i = H // 2
        mid_j = W // 2
        sw = delta_E[:mid_i, :mid_j]
        ne = delta_E[mid_i:, mid_j:]
        mean_delta_sw = float(np.mean(sw)) if sw.size > 0 else 0.0
        mean_delta_ne = float(np.mean(ne)) if ne.size > 0 else 0.0
        print(f"[ΔE[m]] {prior_name}: mean NE={mean_delta_ne:.3e}, SW={mean_delta_sw:.3e}")

        # A/B posterior odds checks on SW and NE representative cells
        try:
            _ab_posterior_ratio_for_cell(0, 0)              # SW if origin='upper'
            _ab_posterior_ratio_for_cell(H - 1, W - 1)      # NE if origin='upper'
        except Exception as ab_e:
            print(f"[AB-odds] diagnostic failed: {ab_e}")

        plt.close('all')
        fig_s, axes_s = plt.subplots(1, 2, figsize=(10, 4))
        fig_s.suptitle(f"State marginals: {prior_name}")
        imS0 = axes_s[0].imshow(prior_state_grid, cmap='viridis', origin='upper', extent=extent)
        axes_s[0].set_title('State marginal (prior)')
        plt.colorbar(imS0, ax=axes_s[0], fraction=0.046, pad=0.04)
        imS1 = axes_s[1].imshow(post_state_grid, cmap='viridis', origin='upper', extent=extent)
        axes_s[1].set_title('State marginal (posterior)')
        plt.colorbar(imS1, ax=axes_s[1], fraction=0.046, pad=0.04)
        for ax in axes_s.ravel():
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xticks(np.linspace(x_min, x_max, nQ + 1))
            ax.set_yticks(np.linspace(y_min, y_max, nQ + 1))
        out_path_s = f"{output_dir}/belief_state_debug_{prior_name}.png"
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(out_path_s, dpi=150)
        plt.close(fig_s)

        fig_m, axes_m = plt.subplots(1, 3, figsize=(14, 4))
        fig_m.suptitle(f"Map marginals: {prior_name}")
        imM0 = axes_m[0].imshow(prior_E_m, cmap='gray_r', vmin=0.0, vmax=1.0, origin='upper', extent=extent)
        axes_m[0].set_title('E[m] (prior)')
        plt.colorbar(imM0, ax=axes_m[0], fraction=0.046, pad=0.04)
        imM1 = axes_m[1].imshow(post_E_m, cmap='gray_r', vmin=0.0, vmax=1.0, origin='upper', extent=extent)
        axes_m[1].set_title('E[m] (posterior)')
        plt.colorbar(imM1, ax=axes_m[1], fraction=0.046, pad=0.04)
        imM2 = axes_m[2].imshow(delta_E, cmap='bwr', vmin=-1.0, vmax=1.0, origin='upper', extent=extent)
        axes_m[2].set_title('ΔE[m] (post - prior)')
        plt.colorbar(imM2, ax=axes_m[2], fraction=0.046, pad=0.04)
        for ax in axes_m.ravel():
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xticks(np.linspace(x_min, x_max, W + 1))
            ax.set_yticks(np.linspace(y_min, y_max, H + 1))
        out_path_m = f"{output_dir}/belief_map_debug_{prior_name}.png"
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(out_path_m, dpi=150)
        plt.close(fig_m)


def _mixed_strength_priors(bmdp: BeliefMDP_n):
    X_ind, M_ind, X_n, H, W = _ground_truth_indices(bmdp)
    m_n = len(X_n)
    M_size = 2 ** (H * W)

    # state_known_map_uniform
    state_prior_known = np.zeros(m_n, dtype=np.float64)
    state_prior_known[X_ind] = 1.0
    map_prior_uniform = np.full(M_size, 1.0 / M_size, dtype=np.float64)
    pi_state_known_map_uniform = np.outer(state_prior_known, map_prior_uniform)

    # state_uniform_map_known
    state_prior_uniform = np.full(m_n, 1.0 / m_n, dtype=np.float64)
    map_prior_known = np.zeros(M_size, dtype=np.float64)
    map_prior_known[M_ind] = 1.0
    pi_state_uniform_map_known = np.outer(state_prior_uniform, map_prior_known)

    return {
        'state_known_map_uniform': pi_state_known_map_uniform,
        'state_uniform_map_known': pi_state_uniform_map_known,
    }


# ---------------------------- H Function Tests ---------------------------- #

def test_H_concentrated():
    """
    Validate H function against empirical sampling for concentrated belief.
    Tests that H(Y|π,u) matches empirical density from Monte Carlo sampling.
    """
    print("\n=== Testing H function against empirical sampling ===")

    bmdp = _init_belief_mdp_n()
    X_ind, M_ind, X_n, H, W = _ground_truth_indices(bmdp)
    m_n = len(X_n)
    M_size = 2 ** (H * W)
    B = bmdp.sensor.B

    # Use the same ground truth setup as other tests
    x_star = X_n[X_ind]  # Target state at index X_ind
    m_star = bmdp.bits_to_map(M_ind, (H, W))  # Target map at index M_ind

    # Create concentrated belief (Dirac delta) using existing prior function pattern
    π = np.zeros((m_n, M_size), dtype=np.float64)
    π[X_ind, M_ind] = 1.0

    # Pick action (small movement)
    u = np.array([0, 2.0])  # forward velocity, no turn

    print(f"Target state: {x_star} (index {X_ind})")
    print(f"Target map index: {M_ind}")
    print(f"Action: {u}")

    # GROUND TRUTH: Monte Carlo sampling
    n_samples = 2000
    print(f"Generating {n_samples} Monte Carlo samples...")

    y_samples = []
    for _ in range(n_samples):
        # Simulate motion with process noise
        w = np.random.multivariate_normal(np.zeros(2), bmdp.cov_x)
        x_next = bmdp.motion_model.f(x_star, u, w)

        # Get ideal observation
        y_ideal = bmdp.sensor.g_bar(x_next[np.newaxis, :], m_star, bmdp.map)[0]

        # Add measurement noise
        v = np.random.multivariate_normal(np.zeros(B), bmdp.cov_y)
        y_noisy = bmdp.sensor.g(x_next[np.newaxis, :], m_star, v[np.newaxis, :], bmdp.map)[0]
        y_samples.append(y_noisy)

    y_samples = np.array(y_samples)  # (n_samples, B)

    # PREDICTION: Compute H for same observations
    Y_test = y_samples
    print(f"Computing H predictions for {len(Y_test)} observations...")
    H_pred = bmdp.H(Y_test, π, u)  # (n_samples,)

    # COMPARE: Use marginal density approach (more reliable than KDE)
    print("Computing marginal densities for comparison...")

    # Use first beam for marginal comparison
    beam_idx = 0
    y_beam_samples = y_samples[:, beam_idx]

    # Create bins for histogram
    bins = np.linspace(y_beam_samples.min(), y_beam_samples.max(), 50)
    hist_emp, _ = np.histogram(y_beam_samples, bins=bins, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Compute H at these bin centers (for single-beam observations)
    Y_bins = np.tile(bin_centers[:, np.newaxis], (1, B))  # replicate across beams
    # Set other beams to mean value
    Y_bins[:, 1:] = np.mean(y_samples[:, 1:], axis=0)

    H_at_bins = bmdp.H(Y_bins, π, u)
    H_at_bins_norm = H_at_bins / np.sum(H_at_bins)

    # Compare marginal distributions
    hist_emp_norm = hist_emp / np.sum(hist_emp)
    corr_marginal = np.corrcoef(hist_emp_norm, H_at_bins_norm)[0, 1]
    print(f"Marginal correlation (beam {beam_idx}): {corr_marginal:.4f}")

    # Check that correlation is reasonable (should be > 0.7 for good agreement)
    assert corr_marginal > 0.7, f"Marginal correlation too low: {corr_marginal:.4f} < 0.7"

    # Visual comparison
    plt.close('all')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Scatter plot: predicted vs empirical (marginal)
    axes[0, 0].scatter(hist_emp_norm, H_at_bins_norm, alpha=0.7, s=20)
    axes[0, 0].plot([0, max(hist_emp_norm)], [0, max(hist_emp_norm)], 'r--', alpha=0.7)
    axes[0, 0].set_xlabel('Empirical marginal density')
    axes[0, 0].set_ylabel('Predicted H (marginal)')
    axes[0, 0].set_title(f'Marginal H Validation (corr={corr_marginal:.3f})')
    axes[0, 0].grid(True, alpha=0.3)

    # Histogram comparison
    axes[0, 1].hist(hist_emp_norm, bins=20, alpha=0.6, label='Empirical marginal', density=True)
    axes[0, 1].hist(H_at_bins_norm, bins=20, alpha=0.6, label='Predicted H marginal', density=True)
    axes[0, 1].set_xlabel('Normalized probability')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Marginal distribution comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Sample observations (first beam)
    axes[1, 0].hist(y_beam_samples, bins=50, alpha=0.6, density=True)
    axes[1, 0].set_xlabel('Range measurement (beam 0)')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Sample observations (beam 0)')
    axes[1, 0].grid(True, alpha=0.3)

    # H values over samples
    sample_indices = np.arange(len(H_pred))
    axes[1, 1].plot(sample_indices[:200], H_pred[:200], 'b-', alpha=0.7, label='Predicted H')
    axes[1, 1].set_xlabel('Sample index')
    axes[1, 1].set_ylabel('Probability density')
    axes[1, 1].set_title('H values over first 200 samples')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = 'output/H_validation.png'
    plt.savefig(out_path, dpi=150)
    print(f'Saved H validation plot to: {out_path}')


def test_H_shape_consistency():
    """Test that H returns correct shapes for various inputs."""
    print("\n=== Testing H shape consistency ===")

    bmdp = _init_belief_mdp_n()
    X_ind, M_ind, X_n, H, W = _ground_truth_indices(bmdp)
    m_n = len(X_n)
    M_size = 2 ** (H * W)
    B = bmdp.sensor.B

    # Use existing prior function to create uniform belief
    priors = _mixed_strength_priors(bmdp)
    π_uniform = priors['state_uniform_map_known']  # This gives us a proper belief structure

    # Test different observation array sizes
    test_sizes = [1, 5, 10]
    u = np.array([0.1, 0.0])

    for n_obs in test_sizes:
        Y = np.random.uniform(0, bmdp.sensor.r_max, (n_obs, B))

        H_result = bmdp.H(Y, π_uniform, u)

        print(f"Y shape: {Y.shape} -> H shape: {H_result.shape}")
        assert H_result.shape == (n_obs,), f"Expected H shape ({n_obs},), got {H_result.shape}"
        assert np.all(H_result >= 0), "H should return non-negative values"
        assert np.all(np.isfinite(H_result)), "H should return finite values"


def test_H_edge_cases():
    """Test H function with edge cases."""
    print("\n=== Testing H edge cases ===")

    bmdp = _init_belief_mdp_n()
    X_ind, M_ind, X_n, H, W = _ground_truth_indices(bmdp)
    m_n = len(X_n)
    M_size = 2 ** (H * W)
    B = bmdp.sensor.B

    # Test with zero belief (should handle gracefully)
    π_zero = np.zeros((m_n, M_size), dtype=np.float64)
    Y = np.random.uniform(0, bmdp.sensor.r_max, (3, B))
    u = np.array([0.0, 0.0])

    H_result = bmdp.H(Y, π_zero, u)
    print(f"Zero belief case - H shape: {H_result.shape}, values: {H_result}")
    assert H_result.shape == (3,)
    assert np.all(H_result >= 0)

    # Test with very small observations using existing prior
    Y_small = np.ones((2, B)) * 1e-6
    priors = _mixed_strength_priors(bmdp)
    π_uniform = priors['state_uniform_map_known']

    H_result_small = bmdp.H(Y_small, π_uniform, u)
    print(f"Small observations case - H shape: {H_result_small.shape}, values: {H_result_small}")
    assert H_result_small.shape == (2,)
    assert np.all(H_result_small >= 0)
    assert np.all(np.isfinite(H_result_small))

    # Test with maximum range observations
    Y_max = np.ones((2, B)) * bmdp.sensor.r_max
    H_result_max = bmdp.H(Y_max, π_uniform, u)
    print(f"Max range case - H shape: {H_result_max.shape}, values: {H_result_max}")
    assert H_result_max.shape == (2,)
    assert np.all(H_result_max >= 0)
    assert np.all(np.isfinite(H_result_max))


@pytest.mark.parametrize("prior_type", ["state_known_map_uniform", "state_uniform_map_known"])
def test_H_with_different_priors(prior_type):
    """Test H function with different prior types using existing prior functions."""
    print(f"\n=== Testing H with {prior_type} prior ===")

    bmdp = _init_belief_mdp_n()
    priors = _mixed_strength_priors(bmdp)
    π = priors[prior_type]

    X_ind, M_ind, X_n, H, W = _ground_truth_indices(bmdp)
    B = bmdp.sensor.B

    # Generate test observations
    Y = np.random.uniform(0, bmdp.sensor.r_max, (5, B))
    u = np.array([0.1, 0.0])

    H_result = bmdp.H(Y, π, u)

    print(f"Prior type: {prior_type}")
    print(f"Y shape: {Y.shape} -> H shape: {H_result.shape}")
    assert H_result.shape == (5,)
    assert np.all(H_result >= 0), "H should return non-negative values"
    assert np.all(np.isfinite(H_result)), "H should return finite values"


# ---------------------------- Component Validation Tests ---------------------------- #

def test_T_vectorized_vs_motion_model_f():
    """Test T_vectorized consistency with motion_model.f() using multivariate noise."""
    print("\n=== Testing T_vectorized vs motion_model.f() ===")

    bmdp = _init_belief_mdp_n()
    X_ind, M_ind, X_n, H, W = _ground_truth_indices(bmdp)

    # Pick a single state and action for focused testing
    x_test = X_n[X_ind]
    u = np.array([0.5, 0.2])

    # Generate samples using motion_model.f()
    n_samples = 1000
    x_samples = []
    for _ in range(n_samples):
        w = np.random.multivariate_normal(np.zeros(2), bmdp.cov_x)
        x_next = bmdp.motion_model.f(x_test, u, w)
        x_samples.append(x_next)
    x_samples = np.array(x_samples)

    # Get T_mat prediction for this state
    u_idx = bmdp.AQ.get_quantized_index(u)
    T_matrix = bmdp.T_mat[:, :, u_idx]  # Extract transition matrix for action u
    T_row = T_matrix[X_ind, :]  # Transition probabilities from x_test

    # Find the most likely next states according to T_vectorized
    likely_indices = np.argsort(T_row)[-10:]  # Top 10 most likely states

    print(f"Target state: {x_test}")
    print(f"Action: {u}")
    print(f"Sample mean & std: {np.mean(x_samples, axis=0)}, {np.std(x_samples, axis=0)}")
    print(f"True mean & std: {x_test}, {bmdp.σ_w}")


    # Check if sample mean is close to predicted mean from T_mat
    predicted_mean = np.sum(X_n * T_row[:, np.newaxis], axis=0)
    print(f"T_mat predicted mean: {predicted_mean}")

    # Compute correlation between sample distribution and T_mat
    # Create histogram of samples
    bins = np.linspace(X_n[:, 0].min(), X_n[:, 0].max(), 20)
    hist_x, _ = np.histogram(x_samples[:, 0], bins=bins, density=True)

    # Get T_mat probabilities for these bins
    bin_centers = (bins[:-1] + bins[1:]) / 2
    T_probs_x = np.zeros(len(bin_centers))
    for i, bin_center in enumerate(bin_centers):
        # Find closest quantized state
        closest_idx = np.argmin(np.abs(X_n[:, 0] - bin_center))
        T_probs_x[i] = T_row[closest_idx]

    # Normalize for comparison
    hist_x_norm = hist_x / np.sum(hist_x)
    T_probs_x_norm = T_probs_x / np.sum(T_probs_x)

    corr_x = np.corrcoef(hist_x_norm, T_probs_x_norm)[0, 1]
    print(f"X-component correlation: {corr_x:.4f}")

    # Similar for Y component
    bins_y = np.linspace(X_n[:, 1].min(), X_n[:, 1].max(), 20)
    hist_y, _ = np.histogram(x_samples[:, 1], bins=bins_y, density=True)
    bin_centers_y = (bins_y[:-1] + bins_y[1:]) / 2
    T_probs_y = np.zeros(len(bin_centers_y))
    for i, bin_center in enumerate(bin_centers_y):
        closest_idx = np.argmin(np.abs(X_n[:, 1] - bin_center))
        T_probs_y[i] = T_row[closest_idx]

    hist_y_norm = hist_y / np.sum(hist_y)
    T_probs_y_norm = T_probs_y / np.sum(T_probs_y)
    corr_y = np.corrcoef(hist_y_norm, T_probs_y_norm)[0, 1]
    print(f"Y-component correlation: {corr_y:.4f}")

    # Assert reasonable correlation
    assert corr_x > 0.5, f"X-component correlation too low: {corr_x:.4f}"
    assert corr_y > 0.5, f"Y-component correlation too low: {corr_y:.4f}"
    print(f"T_mat correlation: {corr_x:.4f}, {corr_y:.4f}")


def test_Q_vectorized_vs_lidar_g():
    """Test Q_vectorized consistency with LIDAR.g() using multivariate noise."""
    print("\n=== Testing Q_vectorized vs LIDAR.g() ===")

    bmdp = _init_belief_mdp_n()
    X_ind, M_ind, X_n, H, W = _ground_truth_indices(bmdp)
    B = bmdp.sensor.B

    # Pick a single state and map
    x_test = X_n[X_ind]
    m_test = bmdp.bits_to_map(M_ind, (H, W))

    # Generate samples using LIDAR.g()
    n_samples = 1000
    y_samples = []
    for _ in range(n_samples):
        v = np.random.multivariate_normal(np.zeros(B), bmdp.cov_y)
        y_noisy = bmdp.sensor.g(x_test, m_test, v[np.newaxis, :], bmdp.map)[0]
        y_samples.append(y_noisy)
    y_samples = np.array(y_samples)

    y_ideal = bmdp.sensor.g_bar(x_test, m_test, bmdp.map)[0]
    # Get Q_vectorized prediction for this state
    # Create test observations around the sample mean
    y_mean = np.mean(y_samples, axis=0)
    Q_matrix = bmdp.Q_vectorized(y_mean, X_n, m_test)
    Q_val = Q_matrix[X_ind, 0]  # Q value for our test state

    print(f"Target state: {x_test}")
    print(f"Sample mean & std: {y_mean}, {np.std(y_samples, axis=0)}")
    print(f"True mean & std: {y_ideal}, {bmdp.σ_v}")
    print(f"Q_vectorized value: {Q_val:.6f}")

    # Manual likelihood computation for comparison (without bin volume)
    y_ideal = bmdp.sensor.g_bar(x_test[np.newaxis, :], m_test, bmdp.map)[0]
    manual_ll_density = np.exp(-np.sum((y_mean - y_ideal)**2) / (2 * bmdp.σ_v**2))

    # Proper quantization includes bin volume: Q_n ≈ q(y) * Δ^B
    bin_volume = bmdp.observation_quantizer.delta ** B
    manual_ll_proper = manual_ll_density * bin_volume

    print(f"Manual likelihood (density): {manual_ll_density:.6f}")
    print(f"Manual likelihood (with bin volume): {manual_ll_proper:.6f}")
    print(f"Q_vectorized value: {Q_val:.6f}")
    print(f"Ratio Q/manual (proper): {Q_val/manual_ll_proper:.4f}")

    # They should be proportional (within reasonable tolerance)
    assert abs(Q_val - manual_ll_proper) / manual_ll_proper < 0.1, \
        f"Q_vectorized and manual likelihood differ too much: {Q_val:.6f} vs {manual_ll_proper:.6f}"


def test_H_component_consistency():
    """Test H function with known components to isolate issues."""
    print("\n=== Testing H component consistency ===")

    bmdp = _init_belief_mdp_n()
    X_ind, M_ind, X_n, H, W = _ground_truth_indices(bmdp)
    m_n = len(X_n)
    M_size = 2 ** (H * W)
    B = bmdp.sensor.B

    # Create a very simple test case: single state, single map
    π = np.zeros((m_n, M_size), dtype=np.float64)
    π[X_ind, M_ind] = 1.0  # Concentrated belief

    # Test with a single observation
    x_star = X_n[X_ind]
    m_star = bmdp.bits_to_map(M_ind, (H, W))

    # Get ideal observation
    y_ideal = bmdp.sensor.g_bar(x_star, m_star, bmdp.map)

    # Test H with ideal observation (should have high probability)
    u = np.array([0.0, 0.0])  # No motion
    H_ideal = bmdp.H(y_ideal, π, u)

    # Test H with noisy observation (should have lower probability)
    v = np.random.multivariate_normal(np.zeros(B), bmdp.cov_y)
    y_noisy = bmdp.sensor.g(x_star, m_star, v, bmdp.map)
    H_noisy = bmdp.H(y_noisy, π, u)

    print(f"Ideal observation H: {H_ideal:.6f}")
    print(f"Noisy observation H: {H_noisy:.6f}")
    print(f"Ratio ideal/noisy: {H_ideal/H_noisy:.2f}")

    # Ideal observation should have higher probability than noisy
    assert H_ideal > H_noisy, f"Ideal observation should have higher probability than noisy: {H_ideal:.6f} vs {H_noisy:.6f}"


def test_H_monte_carlo_debug():
    """Debug the Monte Carlo vs H integration step by step."""
    print("\n=== Debugging H Monte Carlo integration ===")

    bmdp = _init_belief_mdp_n()
    X_ind, M_ind, X_n, H, W = _ground_truth_indices(bmdp)
    m_n = len(X_n)
    M_size = 2 ** (H * W)
    B = bmdp.sensor.B

    # Use concentrated belief
    π = np.zeros((m_n, M_size), dtype=np.float64)
    π[X_ind, M_ind] = 1.0

    x_star = X_n[X_ind]
    m_star = bmdp.bits_to_map(M_ind, (H, W))
    u = np.array([0, 2.0])

    print(f"Target state: {x_star}")
    print(f"Action: {u}")

    # Step 1: Check motion prediction consistency
    print("\n--- Step 1: Motion prediction ---")
    n_samples = 500
    x_samples = []
    for _ in range(n_samples):
        w = np.random.multivariate_normal(np.zeros(2), bmdp.cov_x)
        x_next = bmdp.motion_model.f(x_star, u, w)
        x_samples.append(x_next)
    x_samples = np.array(x_samples)

    print(f"Motion samples mean: {np.mean(x_samples, axis=0)}")
    print(f"Motion samples std: {np.std(x_samples, axis=0)}")

    # Step 2: Check observation generation consistency
    print("\n--- Step 2: Observation generation ---")
    y_samples = []
    for x_next in x_samples:
        y_ideal = bmdp.sensor.g_bar(x_next[np.newaxis, :], m_star, bmdp.map)[0]
        v = np.random.multivariate_normal(np.zeros(B), bmdp.cov_y)
        y_noisy = bmdp.sensor.g(x_next[np.newaxis, :], m_star, v[np.newaxis, :], bmdp.map)[0]
        y_samples.append(y_noisy)
    y_samples = np.array(y_samples)

    print(f"Observation samples mean: {np.mean(y_samples, axis=0)}")
    print(f"Observation samples std: {np.std(y_samples, axis=0)}")

    # Step 3: Check H function with a subset of observations
    print("\n--- Step 3: H function evaluation ---")
    Y_test = y_samples[:100]  # Use subset for faster computation
    H_pred = bmdp.H(Y_test, π, u)

    print(f"H predictions mean: {np.mean(H_pred):.6f}")
    print(f"H predictions std: {np.std(H_pred):.6f}")
    print(f"H predictions range: [{np.min(H_pred):.6f}, {np.max(H_pred):.6f}]")

    # Step 4: Manual likelihood computation for comparison
    print("\n--- Step 4: Manual likelihood comparison ---")
    manual_likelihoods = []
    for i, y_test in enumerate(Y_test):
        # For each observation, compute likelihood manually
        # This should match what H does internally

        # Get T_mat
        u_idx = bmdp.AQ.get_quantized_index(u)
        T_n = bmdp.T_mat[:, :, u_idx]  # Extract transition matrix for action u

        # Apply transition to belief
        integral = T_n @ π  # (m_n, M_size)

        # Get Q values for this observation
        Q_vals = bmdp.Q_vectorized(y_test[np.newaxis, :], X_n, m_star)  # (m_n, 1)

        # Compute likelihood: sum over states and maps
        # integral: (m_n, M_size), Q_vals[:, 0]: (m_n,)
        # We need to broadcast Q_vals across all maps since Q is only state-dependent for this single map
        likelihood = np.sum(integral * Q_vals[:, 0:1])  # Broadcast (m_n,) to (m_n, 1)
        manual_likelihoods.append(likelihood)

        if i < 5:  # Print first few for debugging
            print(f"Sample {i}: H={H_pred[i]:.6f}, Manual={likelihood:.6f}, Ratio={H_pred[i]/likelihood:.4f}")

    manual_likelihoods = np.array(manual_likelihoods)

    # Compare correlations
    corr = np.corrcoef(H_pred, manual_likelihoods)[0, 1]
    print(f"\nH vs Manual correlation: {corr:.4f}")

    # Check if they're proportional
    ratios = H_pred / manual_likelihoods
    print(f"Ratio statistics: mean={np.mean(ratios):.4f}, std={np.std(ratios):.4f}")

    assert corr > 0.99, f"H and manual likelihoods should be highly correlated: {corr:.4f}"
    assert np.std(ratios) < 0.01, f"H and manual likelihoods should be proportional: ratio std={np.std(ratios):.4f}"


# ---------------------------- F Function tests ---------------------------- #


def test_validate_basic_properties():
    validate_basic_properties()


def test_T_mat_structure():
    """Test that T_mat has correct structure and can be accessed via T_n()."""
    bmdp = _init_belief_mdp_n()

    # Test T_n() returns full T_mat
    T_n_result = bmdp.T_n()
    assert T_n_result.shape == (bmdp.SQ.m_n, bmdp.SQ.m_n, bmdp.AQ.n_u)

    # Verify it's the same as T_mat
    assert np.array_equal(T_n_result, bmdp.T_mat)

    # Test probability properties for all actions
    for k in range(bmdp.AQ.n_u):
        T_for_action = bmdp.T_mat[:, :, k]
        col_sums = T_for_action.sum(axis=0)
        assert np.allclose(col_sums, 1.0, atol=1e-10), f"Action {k}: columns should sum to 1"

    print("✓ T_mat structure validation passed")


def test_T_mat_caching():
    """Test that T_mat is cached and reused correctly."""
    import shutil
    from pathlib import Path
    import time

    # Clean cache before test
    project_root = Path(__file__).parent.parent
    cache_dir = project_root / "cache/T_mat"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    # First instantiation - should compute
    start_time = time.time()
    bmdp1 = _init_belief_mdp_n()
    first_time = time.time() - start_time
    print(f"First computation time: {first_time:.2f}s")

    # Verify cache file exists
    assert cache_dir.exists(), "Cache directory should exist"
    cache_files = list(cache_dir.glob("T_mat*.npz"))
    assert len(cache_files) > 0, "Cache file should be created"

    # Second instantiation - should load from cache
    start_time = time.time()
    bmdp2 = _init_belief_mdp_n()
    second_time = time.time() - start_time
    print(f"Cache load time: {second_time:.2f}s")
    print(f"Speedup: {first_time / second_time:.1f}x")

    # Verify results are identical
    assert np.allclose(bmdp1.T_mat, bmdp2.T_mat)

    # Cache load should be faster
    assert second_time < first_time

    # Clean up
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


@pytest.mark.parametrize("N", [10])
def test_belief_updates_metrics(N):
    bmdp = _init_belief_mdp_n()
    priors = _mixed_strength_priors(bmdp)
    results = run_belief_updates(priors, N=N, compute_metrics=True)
    # ensure metrics exist and are finite; also ensure posteriors are valid
    for name, data in results.items():
        assert 'metrics' in data
        post = data['posterior_belief']
        assert np.isfinite(post).all()
        assert abs(post.sum() - 1.0) < 1e-6
        for k, v in data['metrics'].items():
            assert np.isfinite(v)


@pytest.mark.parametrize("N", [10])
def test_belief_updates_plots(N):
    print("Initializing belief MDP")
    bmdp = _init_belief_mdp_n()
    print("Initializing priors")
    priors = _mixed_strength_priors(bmdp)
    print(f"Running {N} updates for {len(priors)} priors")
    results = run_belief_updates(priors, N=N, compute_metrics=False)
    visualize_results(results, output_dir="output")
