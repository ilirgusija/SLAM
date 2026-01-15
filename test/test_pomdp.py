import time
import numpy as np
import matplotlib.pyplot as plt
from src.utils.map import load_obstacles_config
from src.classes.quantizer import SquareLatticeQuantizer
from src.classes.mapping import OCCUPIED, LidarGridMapVec
from src.classes.model import SingleIntegratorModel, LIDAR
from src.classes.pomdp import POMDP

# 1. Initialization
def test_pomdp_initialization_3x3():
    """Test POMDP initialization with basic components"""
    print("=== Testing POMDP Initialization ===")

    n_m = 3

    # Load obstacles and area
    all_obstacles, area = load_obstacles_config(environment='toy2')
    print(f"Loaded {len(all_obstacles)} obstacles")
    print(f"Area: {area}")

    # Create motion model
    motion_model = SingleIntegratorModel(
        i_x=5.0, i_y=5.0, dt=0.1, max_v=5)
    print(f"Motion model: {motion_model.x}")

    # Create sensor
    sensor = LIDAR(fov=360, r_max=5, B=6)
    print(f"Sensor: {sensor.B} beams, max range {sensor.r_max}")

    # Create map (separate from state quantization)
    map = LidarGridMapVec(
        x_min=area[0], x_max=area[1],
        y_min=area[2], y_max=area[3],
        quantization_level=n_m
    )
    print(f"Map: {map.occupancy_map.width}x{map.occupancy_map.height} cells")
    print(f"Map size: {map.occupancy_map.size} cells")

    # Create POMDP
    pomdp = POMDP(
        motion_model=motion_model,
        measurement_model=sensor,
        obstacles=all_obstacles,
        _map=map,
        sigma_w=0.3,
        sigma_v=0.3
    )

    print(f"POMDP created successfully")
    print(f"Process noise σ_w: {pomdp.σ_w}")
    print(f"Observation noise σ_v: {pomdp.σ_v}")

    # Seed occupancy grid from true obstacles so OGM reflects arena
    pomdp.map.seed_from_obstacles(all_obstacles)

    return pomdp

def test_pomdp_initialization_4x4():
    """Test POMDP initialization with basic components"""
    print("=== Testing POMDP Initialization ===")

    n_m = 4

    # Load obstacles and area
    all_obstacles, area = load_obstacles_config(environment='toy3')
    print(f"Loaded {len(all_obstacles)} obstacles")
    print(f"Area: {area}")

    # Create motion model
    motion_model = SingleIntegratorModel(
        i_x=5.0, i_y=5.0, dt=0.1, max_v=5)
    print(f"Motion model: {motion_model.x}")

    # Create sensor
    sensor = LIDAR(fov=360, r_max=5, B=6)
    print(f"Sensor: {sensor.B} beams, max range {sensor.r_max}")

    # Create map (separate from state quantization)
    map = LidarGridMapVec(
        x_min=area[0], x_max=area[1],
        y_min=area[2], y_max=area[3],
        quantization_level=n_m
    )
    print(f"Map: {map.occupancy_map.width}x{map.occupancy_map.height} cells")
    print(f"Map size: {map.occupancy_map.size} cells")

    # Create POMDP
    pomdp = POMDP(
        motion_model=motion_model,
        measurement_model=sensor,
        obstacles=all_obstacles,
        _map=map,
        sigma_w=0.3,
        sigma_v=0.3
    )

    print(f"POMDP created successfully")
    print(f"Process noise σ_w: {pomdp.σ_w}")
    print(f"Observation noise σ_v: {pomdp.σ_v}")

    # Seed occupancy grid from true obstacles so OGM reflects arena
    pomdp.map.seed_from_obstacles(all_obstacles)

    return pomdp

# 2. Transition model - REMOVED obsolete T_vectorized tests
def test_visualize_transition_probabilities(pomdp):
    """Visualize transition probabilities P(x_{t+1} | x_t, u) as heatmap over state space."""
    print("\n=== Visualizing Transition Probabilities ===")

    # Create a quantized state space for visualization
    from src.classes.quantizer import SquareLatticeQuantizer
    X_n = SquareLatticeQuantizer(0, 5, 0, 5, n=11).get_quantized_points()  # 11x11 = 121 states

    # Pick initial state (center)
    x_initial = np.array([2.5, 2.5])
    x_idx = np.argmin(np.linalg.norm(X_n - x_initial, axis=1))
    x_start = X_n[x_idx]

    # Define control action (northeast movement)
    u = np.array([1.0, 1.0])

    print(f"Initial state: {x_start}")
    print(f"Control action: {u}")

    # Compute transition probabilities using POMDP T method
    # We need to use the T method with Borel sets for each state
    transition_probs = np.zeros(len(X_n))

    # For each possible next state, compute P(x_next | x_start, u)
    for i, x_next in enumerate(X_n):
        # Create a small Borel set around x_next
        delta = 0.1  # Small region around each state
        B = np.array([
            [x_next[0] - delta, x_next[0] + delta],  # x bounds
            [x_next[1] - delta, x_next[1] + delta]  # y bounds
        ])

        # Compute T(B | x_start, u)
        prob = pomdp.T(B, x_start[np.newaxis, :], u)[0]
        transition_probs[i] = prob

    # Normalize to ensure it's a probability distribution
    transition_probs = transition_probs / np.sum(transition_probs)

    # Create visualization
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Transition probability heatmap
    n_grid = int(np.sqrt(len(X_n)))
    prob_grid = transition_probs.reshape(n_grid, n_grid)

    im1 = ax1.imshow(prob_grid, cmap='viridis', origin='lower',
                     extent=[X_n[:, 0].min(), X_n[:, 0].max(),
                             X_n[:, 1].min(), X_n[:, 1].max()])
    ax1.scatter([x_start[0]], [x_start[1]], c='red', s=100, marker='*',
                label='Initial state', edgecolors='black', linewidth=2)
    ax1.set_title('Transition Probabilities P(x_{t+1} | x_t, u)')
    ax1.set_xlabel('X position')
    ax1.set_ylabel('Y position')
    ax1.legend()
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Right plot: Predicted next state (mean)
    predicted_mean = np.sum(X_n * transition_probs[:, np.newaxis], axis=0)
    ax2.scatter(X_n[:, 0], X_n[:, 1], c=transition_probs, cmap='viridis',
                s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax2.scatter([x_start[0]], [x_start[1]], c='red', s=100, marker='*',
                label='Initial state', edgecolors='black', linewidth=2)
    ax2.scatter([predicted_mean[0]], [predicted_mean[1]], c='orange', s=100,
                marker='x', label='Predicted mean', edgecolors='black', linewidth=2)

    # Draw arrow from initial to predicted
    ax2.arrow(x_start[0], x_start[1], predicted_mean[0] - x_start[0],
              predicted_mean[1] - x_start[1], head_width=0.1, head_length=0.1,
              fc='orange', ec='orange', alpha=0.8)

    ax2.set_title('State Space with Transition Probabilities')
    ax2.set_xlabel('X position')
    ax2.set_ylabel('Y position')
    ax2.legend()
    ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    out_path = 'output/transition_probabilities.png'
    plt.savefig(out_path, dpi=150)
    print(f'Saved transition probability visualization to: {out_path}')

    return transition_probs, predicted_mean


# 3. Observation model
def test_ray_casting(pomdp):
    """Test ray casting functionality"""
    print("\n=== Testing Ray Casting ===")

    # Test single position (as array)
    X = np.array([[1.0, 2.0]])  # Array of positions
    print(f"Testing ray casting at X={X}")

    start_time = time.time()
    ranges = pomdp.ray_casting(X, pomdp.map.occupancy_map.data)
    end_time = time.time()

    print(f"Ray casting time: {end_time - start_time:.4f}s")
    print(f"Ranges shape: {ranges.shape}")
    print(f"Number of beams: {len(ranges[0])}")
    print(f"Sample ranges: {ranges[0][:10]}")

    return ranges

def test_observation_model(pomdp):
    """Test observation model Q(y | x, m) using current POMDP interface"""
    print("\n=== Testing Observation Model Q(y | x, m) ===")

    # Create a small set of states for testing
    X_n = SquareLatticeQuantizer(0, 3, 0, 3, n=10).get_quantized_points()
    m = pomdp.map.occupancy_map.data

    # Create a more realistic test observation by taking an actual observation from one of the states
    # This ensures we have a realistic observation that should have reasonable likelihoods
    x_test = X_n[4]  # Use middle state
    y_test = pomdp.ray_casting(x_test[np.newaxis, :], m)[0]  # Get real observation from this state

    print(f"Testing observation model for {len(X_n)} states")
    print(f"States: {X_n}")
    print(f"Test observation (from state {x_test}): {y_test}")

    start_time = time.time()
    Q_likelihoods = pomdp.Q(y_test, X_n, m)
    end_time = time.time()

    print(f"Observation model calculation time: {end_time - start_time:.4f}s")
    print(f"Likelihoods shape: {Q_likelihoods.shape}")
    print(f"Likelihoods: {Q_likelihoods}")

    # Verify properties
    assert np.all(Q_likelihoods >= 0), "Likelihoods should be non-negative"
    assert np.all(np.isfinite(Q_likelihoods)), "Likelihoods should be finite"

    # Check that likelihoods are reasonable (not all zero or all same)
    assert np.any(Q_likelihoods > 0), "At least some likelihoods should be positive"
    assert np.std(Q_likelihoods) > 1e-10, "Likelihoods should vary across states"

    return Q_likelihoods

def test_ogm_to_segments_conversion():
    """Validate converting an occupancy grid to obstacle segments.

    We build a tiny map with known occupied cells forming a 2x3 block and
    assert that `_get_obstacles_from_map` returns the rectangle boundary
    segments with correct centroid and dimensions.
    """

    # Map spanning 0..4 in both axes -> 5x5 grid with quantization_level=5
    map_vec = LidarGridMapVec(0, 4, 0, 4, quantization_level=5)

    # Dummy motion/sensor (not used here)
    motion_model = SingleIntegratorModel(i_x=0.0, i_y=0.0, dt=0.1, max_v=1.0)
    sensor = LIDAR(fov=360, r_max=10, B=36)

    pomdp = POMDP(motion_model=motion_model,
                  measurement_model=sensor,
                  obstacles=[],
                  _map=map_vec,
                  sigma_w=0.1,
                  sigma_v=0.1)

    # Build occupancy grid m (H, W) with a 2x3 block of ones at rows 1..2, cols 1..3
    H, W = 5, 5
    m = np.zeros((H, W), dtype=float)
    m[1:3, 1:4] = 1.0  # component: i in {1,2}, j in {1,2,3}

    # Run conversion
    segments = pomdp.sensor.get_obstacles_from_map(m, pomdp.map)

    # Expect a single rectangle with dx=3, dy=2, centroid at (2, 2.4)
    dx = 2.4
    dy = 1.6
    centroid = np.array([2, 2.4])  # geometric center of occupied rectangle at cell edges

    # Compute expected axis-aligned rectangle segments
    BL = (centroid[0] - dx / 2, centroid[1] - dy / 2)
    BR = (centroid[0] + dx / 2, centroid[1] - dy / 2)
    TL = (centroid[0] - dx / 2, centroid[1] + dy / 2)
    TR = (centroid[0] + dx / 2, centroid[1] + dy / 2)

    expected = [
        (BL[0], BL[1], BR[0], BR[1]),  # bottom
        (TL[0], TL[1], TR[0], TR[1]),  # top
        (BL[0], BL[1], TL[0], TL[1]),  # left
        (BR[0], BR[1], TR[0], TR[1]),  # right
    ]

    # Helper to compare segments irrespective of order with tolerance
    def sort_key(seg):
        return tuple(np.round(seg, 6))

    got_sorted = sorted([tuple(s) for s in segments], key=sort_key)
    exp_sorted = sorted([tuple(s) for s in expected], key=sort_key)

    assert len(got_sorted) == 4, f"Expected 4 segments, got {len(segments)}"
    for g, e in zip(got_sorted, exp_sorted):
        assert np.allclose(g, e, atol=1e-6), f"Segment mismatch: got {g}, expected {e}"

    return segments


def test_visualize_ogm_to_segments():
    """Visualize the synthetic OGM and the recovered obstacle segments in proper coordinate systems."""
    # Recreate the same synthetic setup as in test_ogm_to_segments_conversion
    map_vec = LidarGridMapVec(0, 4, 0, 4, quantization_level=5)
    motion_model = SingleIntegratorModel(i_x=0.0, i_y=0.0, dt=0.1, max_v=1.0)
    sensor = LIDAR(fov=360, r_max=10, B=36)
    pomdp = POMDP(motion_model=motion_model,
                  measurement_model=sensor,
                  obstacles=[],
                  _map=map_vec,
                  sigma_w=0.1,
                  sigma_v=0.1)

    H, W = map_vec.occupancy_map.height, map_vec.occupancy_map.width
    print(f"H: {H}, W: {W}")
    m = np.zeros((H, W), dtype=float)
    m[1:3, 1:4] = OCCUPIED
    print(f"m: {m}")
    segments = pomdp.sensor.get_obstacles_from_map(m, pomdp.map)

    # Create a proper visualization with two subplots
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: OGM in grid coordinates (M space)
    ax1.set_title('Occupancy Grid Map (M space)\nGrid coordinates')
    ax1.imshow(m, cmap='gray_r', origin='lower', alpha=0.8)
    ax1.set_xlabel('Grid column index')
    ax1.set_ylabel('Grid row index')
    ax1.set_xticks(range(W))
    ax1.set_yticks(range(H))
    ax1.grid(True, alpha=0.3)

    # Highlight the occupied cells
    occupied_cells = np.argwhere(m == OCCUPIED)
    for cell in occupied_cells:
        ax1.add_patch(plt.Rectangle((cell[1] - 0.5, cell[0] - 0.5), 1, 1,
                                    fill=False, edgecolor='red', linewidth=2))

    # Right plot: Obstacle segments in world coordinates (X space)
    ax2.set_title('Recovered Obstacle Segments (X space)\nWorld coordinates')

    # Draw the map boundaries
    map_bounds = map_vec.occupancy_map
    ax2.add_patch(plt.Rectangle((map_bounds.left_lower[0], map_bounds.left_lower[1]),
                                map_bounds.right_upper[0] - map_bounds.left_lower[0],
                                map_bounds.right_upper[1] - map_bounds.left_lower[1],
                                fill=False, edgecolor='black', linewidth=1, linestyle='--'))

    # Draw grid lines to show cell boundaries
    resolution = map_bounds.resolution
    for i in range(H + 1):
        y = map_bounds.left_lower[1] + i * resolution
        ax2.axhline(y=y, color='lightgray', linewidth=0.5, alpha=0.5)
    for j in range(W + 1):
        x = map_bounds.left_lower[0] + j * resolution
        ax2.axvline(x=x, color='lightgray', linewidth=0.5, alpha=0.5)

    # Draw recovered obstacle segments (red)
    for (x1, y1, x2, y2) in segments:
        ax2.plot([x1, x2], [y1, y2], 'r-', linewidth=3,
                 label='Obstacle segments' if (x1, y1, x2, y2) == segments[0] else "")

    ax2.set_xlim([map_bounds.left_lower[0], map_bounds.right_upper[0]])
    ax2.set_ylim([map_bounds.left_lower[1], map_bounds.right_upper[1]])
    ax2.set_xlabel('World X coordinate (m)')
    ax2.set_ylabel('World Y coordinate (m)')
    ax2.set_aspect('equal', adjustable='box')
    ax2.legend()

    plt.tight_layout()
    out_path = 'output/ogm_segments_validation.png'
    plt.savefig(out_path, dpi=150)
    print('Saved OGM-to-segments visualization to:', out_path)
    return out_path


def test_observation_model_validation(pomdp):
    """Validate observation model Q(y | x, m) with manual computation."""
    print("\n=== Validating Observation Model Q(y | x, m) ===")

    # Small set: 2 states
    X_n = np.array([[0.0, 0.0], [1.0, 1.0]])
    m = pomdp.map.occupancy_map.data

    # Create test observation
    y_test = np.ones(pomdp.sensor.B) * pomdp.sensor.r_max / 2

    # Get POMDP Q values
    Q_pomdp = pomdp.Q(y_test, X_n, m)

    # Manual computation using ray_casting and multivariate normal
    y_star = pomdp.ray_casting(X_n, m)  # (2, B)
    var = pomdp.σ_v ** 2
    cov = np.eye(pomdp.sensor.B) * var

    Q_manual = np.zeros(len(X_n))
    for i, y_star_i in enumerate(y_star):
        # Manual multivariate normal PDF
        diff = y_test - y_star_i
        Q_manual[i] = np.exp(-0.5 * np.dot(diff, np.linalg.solve(cov, diff))) / \
            np.sqrt((2 * np.pi)**pomdp.sensor.B * np.linalg.det(cov))

    print("Q_pomdp:", Q_pomdp)
    print("Q_manual:", Q_manual)

    # Should match closely
    assert np.allclose(Q_pomdp, Q_manual, atol=1e-10, rtol=1e-5), \
        f"POMDP Q and manual computation differ: {Q_pomdp} vs {Q_manual}"

    return Q_pomdp, Q_manual


def demonstrate_observation_model(pomdp):
    """Demonstrate observation model behavior with different states and observations."""
    print("\n=== Demonstrating Observation Model Behavior ===")

    # Create a small grid of states
    X_n = SquareLatticeQuantizer(0, 2, 0, 2, n=3).get_quantized_points()  # 9 states
    m = pomdp.map.occupancy_map.data

    print(f"Testing with {len(X_n)} states")
    print(f"States: {X_n}")

    # Test with different observation types
    observations = {
        'mid_range': np.ones(pomdp.sensor.B) * pomdp.sensor.r_max / 2,
        'close_range': np.ones(pomdp.sensor.B) * pomdp.sensor.r_max / 4,
        'max_range': np.ones(pomdp.sensor.B) * pomdp.sensor.r_max,
    }

    for obs_name, y_test in observations.items():
        print(f"\n--- {obs_name} observation ---")
        Q_likelihoods = pomdp.Q(y_test, X_n, m)
        print(f"Likelihoods: {Q_likelihoods}")
        print(f"Max likelihood state: {X_n[np.argmax(Q_likelihoods)]}")
        print(f"Likelihood range: [{np.min(Q_likelihoods):.6f}, {np.max(Q_likelihoods):.6f}]")

    return observations


def test_observation_noise_sensitivity(pomdp):
    """Test how observation model responds to different noise levels σ_v."""
    print("\n=== Observation Model Noise Sensitivity ===")

    X_n = SquareLatticeQuantizer(0, 2, 0, 2, n=3).get_quantized_points()
    m = pomdp.map.occupancy_map.data

    # Create a test observation
    y_test = np.ones(pomdp.sensor.B) * pomdp.sensor.r_max / 2

    sigmas = [0.01, 0.05, 0.1, 0.2, 0.5]
    results = []

    for s in sigmas:
        old_sigma = pomdp.σ_v
        pomdp.σ_v = s

        Q_likelihoods = pomdp.Q(y_test, X_n, m)

        # Compute entropy-like measure (higher = more spread)
        entropy = -np.sum(Q_likelihoods * np.log(Q_likelihoods + 1e-16))
        max_likelihood = np.max(Q_likelihoods)
        likelihood_std = np.std(Q_likelihoods)

        results.append({
            'sigma': s,
            'entropy': entropy,
            'max_likelihood': max_likelihood,
            'likelihood_std': likelihood_std
        })

        print(f"σ_v={s:.3f} -> entropy={entropy:.4f}, max_lik={max_likelihood:.6f}, std={likelihood_std:.6f}")

        pomdp.σ_v = old_sigma

    return results


# 4. Cost functions
def test_cost_functions(pomdp):
    """Test cost function calculations"""
    print("\n=== Testing Cost Functions ===")

    # Test effort cost
    u = np.array([[0.5, 0.3]])  # 2D array
    effort_cost = pomdp.c_effort(u)
    print(f"Effort cost for u={u}: {effort_cost}")

    # Test virtual force field cost
    x = np.array([1.0, 2.0])
    m = pomdp.map.occupancy_map.data
    u_1d = np.array([0.5, 0.3])  # 1D for c_vff
    vff_cost = pomdp.c_vff(x, m, u_1d)
    print(f"Virtual force field cost for x={x}, u={u_1d}: {vff_cost}")

    # Test total cost
    total_cost = pomdp.c(x, m, u_1d)
    print(f"Total cost: {total_cost}")

    return total_cost


def test_c_vff_manual_validation():
    """Manually validate c_vff on a tiny OGM with a single occupied cell.

    Setup:
      - Map: 5x5 with resolution 1.0, occupied cell at (i=2,j=2)
      - Position x near the occupied cell; compute F_r analytically from definition
      - Choose u to test movement towards and away from obstacle; compare with c_vff
    """
    # Tiny map
    map_vec = LidarGridMapVec(0, 4, 0, 4, quantization_level=4)
    motion_model = SingleIntegratorModel(i_x=0.0, i_y=0.0, dt=0.1, max_v=1.0)
    sensor = LIDAR(fov=360, r_max=10, B=36)
    pomdp = POMDP(motion_model=motion_model,
                  measurement_model=sensor,
                  obstacles=[],
                  _map=map_vec,
                  sigma_w=0.1,
                  sigma_v=0.1)

    # Occupancy grid with a single occupied cell at (i=2, j=2)
    m = np.zeros((5, 5), dtype=float)
    m[2, 2] = 1.0

    # Robot position and control
    x = np.array([2.5, 1.0])  # directly below the cell center (2.5, 2.5)
    u_towards = np.array([0.0, 1.0])  # moving up, towards obstacle
    u_away = np.array([0.0, -1.0])    # moving down, away from obstacle

    # Manual F_r computation (should match POMDP.F_r)
    # Only one contributing cell at center c = (2.5, 2.5)
    c = np.array([2.5, 2.5])
    F_cr = 100.0
    eps = 1e-9
    diff = c - x
    d = np.linalg.norm(diff)
    inv_d3 = 1.0 / max(d, eps)**3
    F_manual = F_cr * diff * inv_d3  # (2,)

    # c_vff = max(0, <F_r, u> / ||F_r||*||u||)
    def manual_c_vff(F, u):
        num = float(np.dot(F, u))
        den = float(np.linalg.norm(F) * np.linalg.norm(u) + 1e-6)
        val = num / den
        return max(0.0, val)

    expected_towards = manual_c_vff(F_manual, u_towards)
    expected_away = manual_c_vff(F_manual, u_away)

    got_towards = pomdp.c_vff(x, m, u_towards)
    got_away = pomdp.c_vff(x, m, u_away)

    # Diagnostics
    print("\n=== c_vff manual validation ===")
    print(f"Occupied cell center c: {c}")
    print(f"Robot position x: {x}")
    print(f"F_manual: {F_manual}")
    print(f"u_towards: {u_towards}, expected: {expected_towards}, got: {got_towards}")
    print(f"u_away:    {u_away}, expected: {expected_away}, got: {got_away}")

    assert np.allclose(got_towards, expected_towards, atol=1e-12)
    assert np.allclose(got_away, expected_away, atol=1e-12)

    # Sanity: moving orthogonal should be ~0
    u_ortho = np.array([1.0, 0.0])
    expected_ortho = manual_c_vff(F_manual, u_ortho)
    got_ortho = pomdp.c_vff(x, m, u_ortho)
    print(f"u_ortho:   {u_ortho}, expected: {expected_ortho}, got: {got_ortho}")
    assert np.allclose(got_ortho, expected_ortho, atol=1e-12)

    return got_towards, got_away, got_ortho

# 5. Map gen
def test_generate_space_of_maps_3x3():
    """Generate map IDs for a 3x3 grid and verify count + conversion."""

    # Build a 3x3 grid: x,y in [0,2] with resolution 1.0 -> width=3, height=3

    map_vec = LidarGridMapVec(0, 2, 0, 2, quantization_level=3)
    motion_model = SingleIntegratorModel(i_x=0.0, i_y=0.0, dt=0.1, max_v=1.0)
    sensor = LIDAR(fov=360, r_max=10, B=36)
    pomdp = POMDP(motion_model=motion_model,
                  measurement_model=sensor,
                  obstacles=[],
                  _map=map_vec,
                  sigma_w=0.1,
                  sigma_v=0.1)

    # Compact bitset IDs (uint16 sufficient for 9 bits)
    ids = pomdp.generate_space_of_map_ids(as_numpy=True)
    H = pomdp.map.occupancy_map.height
    W = pomdp.map.occupancy_map.width
    total_cells = H * W
    expected_num = 2 ** total_cells

    assert ids.shape == (expected_num,)

    # Memory diagnostics
    # Memory diagnostics for IDs
    theoretical_bits = total_cells * expected_num  # if storing bit-packed arrays
    ids_bytes = ids.nbytes
    print(f"3x3 -> map IDs count: {expected_num}")
    print(f"Bit-packed arrays theoretical: {theoretical_bits} bits (~{theoretical_bits/8:.1f} bytes)")
    print(f"ID array size: {ids_bytes} bytes (dtype={ids.dtype})")

    # For 9 bits, minimal dtype is uint16 -> total ~ 512 * 2 = 1024 bytes
    assert ids.dtype.itemsize in (2, 4, 8)
    assert ids_bytes == expected_num * ids.dtype.itemsize

    # Spot-check converting a few IDs back to array maps
    for b in [0, 1, 255, 511]:
        m = pomdp.bits_to_map(int(b), (H, W))
        print("\n--- Spot check ---")
        print(f"ID (bits): {b}")
        print("Decoded map (H x W):\n", m)
        # re-encode and compare
        b2 = pomdp.map_to_bits(m)
        print(f"Re-encoded ID: {b2}")
        assert int(b) == int(b2)

def test_generate_space_of_maps_4x4():
    """Generate map IDs for a 4x4 grid and verify count, memory, and conversions."""

    # 4x4 grid: x,y in [0,3] with resolution 1.0 -> width=4, height=4
    map_vec = LidarGridMapVec(0, 3, 0, 3, quantization_level=4)
    motion_model = SingleIntegratorModel(i_x=0.0, i_y=0.0, dt=0.1, max_v=1.0)
    sensor = LIDAR(fov=360, r_max=10, B=36)
    pomdp = POMDP(motion_model=motion_model,
                  measurement_model=sensor,
                  obstacles=[],
                  _map=map_vec,
                  sigma_w=0.1,
                  sigma_v=0.1)

    ids = pomdp.generate_space_of_map_ids(as_numpy=True)
    H = pomdp.map.occupancy_map.height
    W = pomdp.map.occupancy_map.width
    total_cells = H * W
    expected_num = 2 ** total_cells

    assert (H, W) == (4, 4)
    assert ids.shape == (expected_num,)

    # Memory diagnostics for IDs vs theoretical bit-packed arrays
    ids_bytes = ids.nbytes
    theoretical_bits = total_cells * expected_num
    print(f"4x4 -> map IDs count: {expected_num}")
    print(f"Bit-packed arrays theoretical: {theoretical_bits} bits (~{theoretical_bits/8:.1f} bytes)")
    print(f"ID array size: {ids_bytes} bytes (dtype={ids.dtype})")
    assert ids.dtype.itemsize in (2, 4, 8)
    assert ids_bytes == expected_num * ids.dtype.itemsize

    # Spot-check round-trips
    for b in [0, 1, 32768, 65535]:
        m = pomdp.bits_to_map(int(b), (H, W))
        print("\n--- 4x4 Spot check ---")
        print(f"ID (bits): {b}")
        print("Decoded map (H x W):\n", m)
        b2 = pomdp.map_to_bits(m)
        print(f"Re-encoded ID: {b2}")
        assert int(b) == int(b2)

# 6. New Visualizations
def test_visualize_observation_likelihood(pomdp):
    """Visualize observation likelihood Q(y | x, m) for various states."""
    print("\n=== Visualizing Observation Likelihood ===")

    # Create a quantized state space
    from src.classes.quantizer import SquareLatticeQuantizer
    X_n = SquareLatticeQuantizer(0, 5, 0, 5, n=11).get_quantized_points()  # 11x11 = 121 states

    # Pick a true state and generate observation from it
    x_true = np.array([2.5, 2.5])
    x_true_idx = np.argmin(np.linalg.norm(X_n - x_true, axis=1))
    x_true_actual = X_n[x_true_idx]

    # Get true observation from this state
    m = pomdp.map.occupancy_map.data
    y_true = pomdp.ray_casting(x_true_actual[np.newaxis, :], m)[0]

    print(f"True state: {x_true_actual}")
    print(f"True observation: {y_true[:5]}... (showing first 5 beams)")

    # Compute likelihood Q(y_true | x, m) for all states
    likelihoods = pomdp.Q(y_true, X_n, m)

    # Create visualization
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Likelihood heatmap
    n_grid = int(np.sqrt(len(X_n)))
    likelihood_grid = likelihoods.reshape(n_grid, n_grid)

    im1 = ax1.imshow(likelihood_grid, cmap='plasma', origin='lower',
                     extent=[X_n[:, 0].min(), X_n[:, 0].max(),
                             X_n[:, 1].min(), X_n[:, 1].max()])
    ax1.scatter([x_true_actual[0]], [x_true_actual[1]], c='red', s=100, marker='*',
                label='True state', edgecolors='black', linewidth=2)
    ax1.set_title('Observation Likelihood Q(y | x, m)')
    ax1.set_xlabel('X position')
    ax1.set_ylabel('Y position')
    ax1.legend()
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Right plot: Likelihood as scatter plot
    scatter = ax2.scatter(X_n[:, 0], X_n[:, 1], c=likelihoods, cmap='plasma',
                          s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
    ax2.scatter([x_true_actual[0]], [x_true_actual[1]], c='red', s=100, marker='*',
                label='True state', edgecolors='black', linewidth=2)

    # Highlight states with high likelihood
    high_likelihood_mask = likelihoods > np.percentile(likelihoods, 90)
    if np.any(high_likelihood_mask):
        ax2.scatter(X_n[high_likelihood_mask, 0], X_n[high_likelihood_mask, 1],
                    c='yellow', s=80, marker='s', alpha=0.6,
                    label='Top 10% likelihood', edgecolors='black', linewidth=1)

    ax2.set_title('State Space with Observation Likelihoods')
    ax2.set_xlabel('X position')
    ax2.set_ylabel('Y position')
    ax2.legend()
    ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    out_path = 'output/observation_likelihood.png'
    plt.savefig(out_path, dpi=150)
    print(f'Saved observation likelihood visualization to: {out_path}')

    # Print statistics
    max_likelihood_idx = np.argmax(likelihoods)
    max_likelihood_state = X_n[max_likelihood_idx]
    print(f"Maximum likelihood state: {max_likelihood_state}")
    print(f"Maximum likelihood value: {likelihoods[max_likelihood_idx]:.6f}")
    print(f"True state likelihood: {likelihoods[x_true_idx]:.6f}")

    return likelihoods, max_likelihood_state

# 7. Performance analysis
def test_performance_analysis(pomdp):
    """Analyze performance bottlenecks using current T method"""
    print("\n=== Performance Analysis ===")

    # Test with different state set sizes
    state_sizes = [4, 9, 16, 25]
    times = []

    for n_states in state_sizes:
        # Create grid of states
        n_per_dim = int(np.sqrt(n_states))
        x_vals = np.linspace(0, 5, n_per_dim)
        y_vals = np.linspace(0, 5, n_per_dim)
        X_n = np.array([[x, y] for x in x_vals for y in y_vals])

        u = np.array([0.5, 0.3])

        # Test transition probability calculation for a single Borel set
        B = np.array([[0.0, 5.0], [0.0, 5.0]])  # Full state space

        start_time = time.time()
        T_probs = pomdp.T(B, X_n, u)
        end_time = time.time()

        elapsed = end_time - start_time
        times.append(elapsed)

        print(f"States: {n_states}, Time: {elapsed:.4f}s")

    # Plot performance
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(state_sizes, times, 'bo-')
    plt.xlabel('Number of States')
    plt.ylabel('Time (s)')
    plt.title('Transition Probability Calculation Time')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(state_sizes, np.array(times) / np.array(state_sizes), 'ro-')
    plt.xlabel('Number of States')
    plt.ylabel('Time per State (s)')
    plt.title('Time per State')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('output/pomdp_performance.png')
    plt.show()

    return times


def test_visualize_single_pose_lidar(pomdp):
    """Plot the robot, its FoV, and lidar hits for a single pose using existing plotting."""
    print("\n=== Visualizing Single-Pose LIDAR Snapshot ===")
    # Lazy import to avoid circulars
    from script import run_simulation as rs
    from src.classes.mapping import LidarGridMap

    # Ensure plot_step expects this attribute name
    if not hasattr(pomdp.sensor, 'max_range'):
        pomdp.sensor.max_range = pomdp.sensor.r_max

    # Prepare a background grid for plotting
    all_obstacles, area = load_obstacles_config(environment='toy2')
    gridmap = LidarGridMap(*area, resolution=pomdp.map.occupancy_map.resolution)

    # Adapter so plot_step can call get_float_data on cells
    class _Cell:
        def __init__(self, v):
            self._v = float(v)

        def get_float_data(self):
            return self._v

    class _OccAdapter:
        def __init__(self, occ):
            self._occ = occ
            self.width = occ.width
            self.height = occ.height
            self.resolution = occ.resolution

        def get_value_from_xy_index(self, x_ind, y_ind):
            val = self._occ.get_value_from_xy_index(x_ind, y_ind)
            if val is None:
                return None
            return _Cell(val)

    gridmap.occupancy_map = _OccAdapter(gridmap.occupancy_map)

    # Choose a pose near the center
    cx = 0.5 * (area[0] + area[1])
    cy = 0.5 * (area[2] + area[3])
    robot_pose = np.array([cx, cy, 0.0])

    # Build obstacle segments from current occupancy grid via sensor helpers
    m = pomdp.map.occupancy_map.data
    segments = pomdp.sensor.get_obstacles_from_map(m, pomdp.map)

    # Get lidar reflections for this single pose
    dist_theta = pomdp.sensor.get_laser_ref(
        segments, robot_poses=robot_pose[:2][np.newaxis, :])[0]
    angles = pomdp.sensor.angles
    laser_data_xy = np.vstack(
        [dist_theta * np.cos(angles), dist_theta * np.sin(angles)]).T + robot_pose[:2]

    # Configure output naming expected by plot_step
    rs.out_fn = 'pomdp_lidar_demo'
    ofn = 'output/'

    # Plot and save a single frame
    rs.plot_step(0, gridmap, pomdp.sensor, robot_pose, laser_data_xy,
                 dist_theta, area, ofn, np.array([robot_pose]))
    print("Saved plot to:", ofn + rs.out_fn + '_frame_0.png')
    return ofn + rs.out_fn + '_frame_0.png'


def test_visualize_arena_and_hits(pomdp):
    """Visualize arena obstacles and overlay immediate lidar observations as red points."""
    print("\n=== Visualizing Arena and Immediate LIDAR Hits ===")
    # Reuse utilities from simulation
    from script.run_simulation import connect_segments

    # Load environment obstacles and area
    all_obstacles, area = load_obstacles_config(environment='toy2')

    # Build true arena segments directly from obstacle geometry (static snapshot)
    all_obstacle_segments = []
    for obs in all_obstacles:
        all_obstacle_segments += list(obs._Obstacle__get_points(obs.centroid))

    # Dense points for background arena visualization from segments
    arena_points = connect_segments(
        np.array(all_obstacle_segments), resolution=0.05)

    # Choose a robot pose near the center
    robot_pose = pomdp.motion_model.x

    # Lidar reflections for this pose
    dist_theta = pomdp.sensor.get_laser_ref(
        all_obstacle_segments, robot_pose[np.newaxis, :])[0]
    angles = pomdp.sensor.angles
    endpoints = np.vstack([dist_theta * np.cos(angles),
                          dist_theta * np.sin(angles)]).T + robot_pose
    is_hit = dist_theta < pomdp.sensor.r_max

    # Plot
    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title('Arena with LIDAR Hits')
    # Arena background
    ax.scatter(arena_points[:, 0], arena_points[:, 1],
               marker='.', c='y', edgecolor='none', alpha=0.3, s=4)
    # Beams
    for i in range(len(endpoints)):
        ax.plot([robot_pose[0], endpoints[i, 0]], [robot_pose[1],
                endpoints[i, 1]], c='b', alpha=0.15, linewidth=0.8)
    # Hits
    ax.scatter(endpoints[is_hit, 0], endpoints[is_hit, 1],
               marker='o', c='r', edgecolor='none', s=10)
    # Robot
    ax.scatter([robot_pose[0]], [robot_pose[1]], marker='*', c='k', s=50)

    ax.set_xlim([area[0], area[1]])
    ax.set_ylim([area[2], area[3]])
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    out_path = 'output/arena_hits.png'
    plt.savefig(out_path, dpi=150)
    print('Saved arena+hits plot to:', out_path)
    return out_path


def test_visualize_ogm_and_raycasting_hits_3x3(pomdp):
    """Validate ray_casting by plotting OGM with y_star hits, alongside true arena + hits."""
    print("\n=== Validating ray_casting: OGM + y_star vs Arena + Hits ===")
    from script.run_simulation import connect_segments

    # Env and robot pose
    all_obstacles, area = load_obstacles_config(environment='toy2')
    robot_pose = pomdp.motion_model.x  # initial position as requested

    # --- Left subplot: OGM + y_star ---
    # OGM grid from vector map (GridMapNP stores data as (width,height)) -> transpose for imshow (H,W)
    grid_data = np.array([[0, 1, 0],
                          [0, 0, 1],
                          [0, 0, 0]], np.float32)

    # y_star from ray_casting on current OGM
    y_star = pomdp.ray_casting(
        robot_pose[np.newaxis, :], grid_data)[0]
    angles = pomdp.sensor.angles
    endpoints_y = np.vstack(
        [y_star * np.cos(angles), y_star * np.sin(angles)]).T + robot_pose
    is_hit_y = y_star < pomdp.sensor.r_max

    # --- Right subplot: True arena + hits from obstacle geometry ---
    all_obstacle_segments = []
    for obs in all_obstacles:
        all_obstacle_segments += list(obs._Obstacle__get_points(obs.centroid))
    arena_points = connect_segments(
        np.array(all_obstacle_segments), resolution=0.05)
    dist_theta = pomdp.sensor.get_laser_ref(
        all_obstacle_segments, robot_pose[np.newaxis, :])[0]
    endpoints_true = np.vstack(
        [dist_theta * np.cos(angles), dist_theta * np.sin(angles)]).T + robot_pose
    is_hit_true = dist_theta < pomdp.sensor.r_max

    # Plot figure with two subplots
    plt.close('all')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: OGM + y_star + obstacles from OGM
    ax = axes[0]
    ax.set_title('OGM + y_star (ray_casting)')
    # Use origin='upper' so row 0 appears at the top, matching matrix convention
    ax.imshow(grid_data, cmap='gray_r', origin='upper',
              extent=[area[0], area[1], area[2], area[3]], alpha=0.8)
    # Overlay obstacle boundaries derived from the OGM used in ray_casting
    ogm_segments = pomdp.sensor.get_obstacles_from_map(grid_data, pomdp.map)
    ogm_points = connect_segments(np.array(ogm_segments), resolution=0.05)
    ax.scatter(ogm_points[:, 0], ogm_points[:, 1],
               marker='.', c='y', edgecolor='none', alpha=0.35, s=4)
    # beams (faint blue)
    for i in range(len(endpoints_y)):
        ax.plot([robot_pose[0], endpoints_y[i, 0]], [robot_pose[1], endpoints_y[i, 1]],
                c='b', alpha=0.1, linewidth=0.8)
    # hits (red)
    ax.scatter(endpoints_y[is_hit_y, 0],
               endpoints_y[is_hit_y, 1], c='r', s=10, edgecolor='none')
    # robot
    ax.scatter([robot_pose[0]], [robot_pose[1]], marker='*', c='k', s=50)
    ax.set_xlim([area[0], area[1]])
    ax.set_ylim([area[2], area[3]])
    ax.set_aspect('equal', adjustable='box')

    # Right: True arena + hits
    ax = axes[1]
    ax.set_title('True Arena + LIDAR Hits')
    ax.scatter(arena_points[:, 0], arena_points[:, 1],
               marker='.', c='y', edgecolor='none', alpha=0.3, s=4)
    for i in range(len(endpoints_true)):
        ax.plot([robot_pose[0], endpoints_true[i, 0]], [robot_pose[1], endpoints_true[i, 1]],
                c='b', alpha=0.15, linewidth=0.8)
    ax.scatter(endpoints_true[is_hit_true, 0], endpoints_true[is_hit_true, 1],
               marker='o', c='r', edgecolor='none', s=10)
    ax.scatter([robot_pose[0]], [robot_pose[1]], marker='*', c='k', s=50)
    ax.set_xlim([area[0], area[1]])
    ax.set_ylim([area[2], area[3]])
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    out_path = 'output/raycast_validation.png'
    plt.savefig(out_path, dpi=150)
    print('Saved ray_casting validation figure to:', out_path)
    return out_path

def test_visualize_ogm_and_raycasting_hits_4x4(pomdp):
    """Validate ray_casting by plotting OGM with y_star hits, alongside true arena + hits."""
    print("\n=== Validating ray_casting: OGM + y_star vs Arena + Hits ===")
    from script.run_simulation import connect_segments

    # Env and robot pose
    all_obstacles, area = load_obstacles_config(environment='toy3')
    robot_pose = pomdp.motion_model.x  # initial position as requested

    # --- Left subplot: OGM + y_star ---
    # OGM grid from vector map (GridMapNP stores data as (width,height)) -> transpose for imshow (H,W)
    grid_data = np.array([[0, 0, 1, 0],
                          [0, 0, 0, 1],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]], np.float32)

    # y_star from ray_casting on current OGM
    y_star = pomdp.ray_casting(
        robot_pose[np.newaxis, :], grid_data)[0]
    angles = pomdp.sensor.angles
    endpoints_y = np.vstack(
        [y_star * np.cos(angles), y_star * np.sin(angles)]).T + robot_pose
    is_hit_y = y_star < pomdp.sensor.r_max

    # --- Right subplot: True arena + hits from obstacle geometry ---
    all_obstacle_segments = []
    for obs in all_obstacles:
        all_obstacle_segments += list(obs._Obstacle__get_points(obs.centroid))
    arena_points = connect_segments(
        np.array(all_obstacle_segments), resolution=0.05)
    dist_theta = pomdp.sensor.get_laser_ref(
        all_obstacle_segments, robot_pose[np.newaxis, :])[0]
    endpoints_true = np.vstack(
        [dist_theta * np.cos(angles), dist_theta * np.sin(angles)]).T + robot_pose
    is_hit_true = dist_theta < pomdp.sensor.r_max

    # Plot figure with two subplots
    plt.close('all')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: OGM + y_star + obstacles from OGM
    ax = axes[0]
    ax.set_title('OGM + y_star (ray_casting)')
    # Use origin='upper' so row 0 appears at the top, matching matrix convention
    ax.imshow(grid_data, cmap='gray_r', origin='upper',
              extent=[area[0], area[1], area[2], area[3]], alpha=0.8)
    # Overlay obstacle boundaries derived from the OGM used in ray_casting
    ogm_segments = pomdp.sensor.get_obstacles_from_map(grid_data, pomdp.map)
    ogm_points = connect_segments(np.array(ogm_segments), resolution=0.05)
    ax.scatter(ogm_points[:, 0], ogm_points[:, 1],
               marker='.', c='y', edgecolor='none', alpha=0.35, s=4)
    # beams (faint blue)
    for i in range(len(endpoints_y)):
        ax.plot([robot_pose[0], endpoints_y[i, 0]], [robot_pose[1], endpoints_y[i, 1]],
                c='b', alpha=0.1, linewidth=0.8)
    # hits (red)
    ax.scatter(endpoints_y[is_hit_y, 0],
               endpoints_y[is_hit_y, 1], c='r', s=10, edgecolor='none')
    # robot
    ax.scatter([robot_pose[0]], [robot_pose[1]], marker='*', c='k', s=50)
    ax.set_xlim([area[0], area[1]])
    ax.set_ylim([area[2], area[3]])
    ax.set_aspect('equal', adjustable='box')

    # Right: True arena + hits
    ax = axes[1]
    ax.set_title('True Arena + LIDAR Hits')
    ax.scatter(arena_points[:, 0], arena_points[:, 1],
               marker='.', c='y', edgecolor='none', alpha=0.3, s=4)
    for i in range(len(endpoints_true)):
        ax.plot([robot_pose[0], endpoints_true[i, 0]], [robot_pose[1], endpoints_true[i, 1]],
                c='b', alpha=0.15, linewidth=0.8)
    ax.scatter(endpoints_true[is_hit_true, 0], endpoints_true[is_hit_true, 1],
               marker='o', c='r', edgecolor='none', s=10)
    ax.scatter([robot_pose[0]], [robot_pose[1]], marker='*', c='k', s=50)
    ax.set_xlim([area[0], area[1]])
    ax.set_ylim([area[2], area[3]])
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    out_path = 'output/raycast_validation_4x4.png'
    plt.savefig(out_path, dpi=150)
    print('Saved ray_casting validation figure to:', out_path)
    return out_path


def main():
    """Run all POMDP tests"""
    print("Starting POMDP Test Suite")
    print("=" * 50)

    # # 1. Test initialization
    print("\n=== 1. POMDP Initialization ===")
    pomdp3x3 = test_pomdp_initialization_3x3()
    pomdp4x4 = test_pomdp_initialization_4x4()

    # 2. Observation model tests
    print("\n=== 2. Observation Model Tests ===")
    test_ray_casting(pomdp3x3)
    test_observation_model(pomdp3x3)
    test_observation_model_validation(pomdp3x3)
    demonstrate_observation_model(pomdp3x3)
    test_observation_noise_sensitivity(pomdp3x3)

    # 3. OGM conversion tests
    print("\n=== 3. OGM Conversion Tests ===")
    test_ogm_to_segments_conversion()
    test_visualize_ogm_to_segments()

    # 4. Cost function tests
    print("\n=== 4. Cost Function Tests ===")
    test_cost_functions(pomdp3x3)
    test_c_vff_manual_validation()

    # 5. Map generation tests
    print("\n=== 5. Map Generation Tests ===")
    test_generate_space_of_maps_3x3()
    test_generate_space_of_maps_4x4()

    # # 6. Visualization tests
    print("\n=== 6. Visualization Tests ===")
    test_visualize_single_pose_lidar(pomdp3x3)
    test_visualize_arena_and_hits(pomdp3x3)
    test_visualize_ogm_and_raycasting_hits_3x3(pomdp3x3)
    test_visualize_ogm_and_raycasting_hits_4x4(pomdp4x4)

    # 7. New transition and observation visualizations
    print("\n=== 7. New Transition and Observation Visualizations ===")
    test_visualize_transition_probabilities(pomdp3x3)
    test_visualize_observation_likelihood(pomdp3x3)

    # 8. Performance analysis
    print("\n=== 8. Performance Analysis ===")
    test_performance_analysis(pomdp3x3)

    print("\n" + "=" * 50)
    print("POMDP Test Suite Complete!")


if __name__ == "__main__":
    main()
