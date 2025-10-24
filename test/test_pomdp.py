import time
from src.utils.map import load_obstacles_config
from src.classes.quantizer import SquareLatticeQuantizer, ObservationQuantizer
from src.classes.mapping import LidarGridMapVec
from src.classes.model import VelocityIntegratorModel, LIDAR
from src.classes.pomdp import POMDP
import matplotlib.pyplot as plt
import numpy as np

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
    motion_model = VelocityIntegratorModel(
        i_x=5.0, i_y=5.0, dt=0.1, max_v=5)
    print(f"Motion model: {motion_model.x}")

    # Create sensor
    sensor = LIDAR(fov=360, r_max=5, B=6)
    print(f"Sensor: {sensor.B} beams, max range {sensor.r_max}")

    # Create map (separate from state quantization)
    map = LidarGridMapVec(
        x_min=area[0], x_max=area[1],
        y_min=area[2], y_max=area[3],
        # since n = (x_max-x_min+1)/resolution
        resolution=(area[1] - area[0] + 1) / (n_m)
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
    motion_model = VelocityIntegratorModel(
        i_x=5.0, i_y=5.0, dt=0.1, max_v=5)
    print(f"Motion model: {motion_model.x}")

    # Create sensor
    sensor = LIDAR(fov=360, r_max=5, B=6)
    print(f"Sensor: {sensor.B} beams, max range {sensor.r_max}")

    # Create map (separate from state quantization)
    map = LidarGridMapVec(
        x_min=area[0], x_max=area[1],
        y_min=area[2], y_max=area[3],
        # since n = (x_max-x_min+1)/resolution
        resolution=(area[1] - area[0] + 1) / (n_m)
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

# 2. Transition model
def test_vectorized_transition(pomdp):
    """Test cartesian transition matrix calculation"""
    print("\n=== Testing Cartesian Transition Matrix ===")

    # Create a small set of states for testing
    X_n = SquareLatticeQuantizer(0, 3, 0, 3, n=10).get_quantized_points()
    u = np.array([0.5, 0.3])

    print(f"Testing transition matrix for {len(X_n)} states")
    print(f"States: {X_n}")
    print(f"Action: {u}")

    start_time = time.time()
    T_matrix = pomdp.T_vectorized(X_n, u)
    end_time = time.time()

    print(f"Transition matrix calculation time: {end_time - start_time:.4f}s")
    print(f"Transition matrix shape: {T_matrix.shape}")
    print(f"Transition matrix:\n{T_matrix}")

    # Verify properties
    row_sums = np.sum(T_matrix, axis=1)
    print(f"Row sums (should be close to 1): {row_sums}")

    # Verify properties
    col_sums = np.sum(T_matrix, axis=0)
    print(f"Column sums (should not be close to 1): {col_sums}")

    return T_matrix


def test_T_vectorized_indexing_validation(pomdp):
    """Validate that row i corresponds to T(\cdot | X_n[i]) and column j to next state X_n[j]."""
    print("\n=== Validating T_vectorized Indexing (rows: current i, cols: next j) ===")

    # Tiny grid of states and a fixed action
    X_n = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    u = np.array([0.5, 0.3])

    # Compute vectorized transition matrix
    T_vec = pomdp.T_vectorized(X_n, u)

    # Manually compute expected matrix using the definition with a loop baseline
    dt = pomdp.motion_model.dt
    sigma = pomdp.σ_w
    var = sigma ** 2
    norm_const = 1.0 / (np.sqrt(2 * np.pi) * sigma)

    predicted_states = X_n + u * dt  # shape (m,2)
    m = X_n.shape[0]
    T_loop = np.zeros((m, m))
    for i in range(m):  # current state index
        for j in range(m):  # next state index
            diff = X_n[j] - predicted_states[i]
            sq = np.dot(diff, diff)
            T_loop[i, j] = norm_const * np.exp(-sq / (2 * var))
        # row-normalize to compare fairly
        row_sum = T_loop[i].sum()
        if row_sum > 0:
            T_loop[i] /= row_sum

    print("T_vec:\n", T_vec)
    print("T_loop baseline:\n", T_loop)

    # The two should match closely if indexing is T(j | i)
    assert np.allclose(
        T_vec, T_loop, atol=1e-10), "Row/column indexing mismatch in T_vectorized"

    # Additionally check that the most likely next state per row aligns with nearest neighbor to predicted state
    nn_indices = np.argmin(
        ((X_n[None, :, :] - predicted_states[:, None, :]) ** 2).sum(axis=-1), axis=1)
    argmax_cols = np.argmax(T_vec, axis=1)
    print("Nearest-neighbor indices per row:", nn_indices)
    print("Argmax columns per row:", argmax_cols)
    assert np.array_equal(
        argmax_cols, nn_indices), "Highest probability column should be nearest to predicted state per row"

    return T_vec, T_loop


def test_compare_T_vectorized_versions(pomdp):
    """Compare T_vectorized (new) vs T_vectorized_v1 (old) on identical inputs."""
    print("\n=== Comparing T_vectorized vs T_vectorized_v1 ===")

    # Use a modest grid to keep runtime reasonable but non-trivial
    X_n = SquareLatticeQuantizer(0, 3, 0, 3, n=9).get_quantized_points()
    u = np.array([0.7, -0.2])

    T_new = pomdp.T_vectorized(X_n, u)
    T_old = pomdp.T_vectorized_v1(X_n, u)

    print("Shapes:", T_new.shape, T_old.shape)
    # Row sums should be ~1 for both
    print("Row sums (new) sample:", np.round(T_new.sum(axis=1)[:5], 6))
    print("Row sums (old) sample:", np.round(T_old.sum(axis=1)[:5], 6))

    # Numerical agreement: allow tiny tolerance due to different stabilization strategies
    assert T_new.shape == T_old.shape
    assert np.allclose(T_new, T_old, atol=1e-8, rtol=1e-5), "New and old T diverge beyond tolerance"

    # Argmax column per row (most likely next state) should match exactly
    argmax_new = np.argmax(T_new, axis=1)
    argmax_old = np.argmax(T_old, axis=1)
    assert np.array_equal(argmax_new, argmax_old), "Argmax next-state columns differ between versions"

    return T_new, T_old


def demonstrate_T_vectorized_broadcasting(pomdp):
    """Demonstrate the broadcasting steps used to build T_vectorized with a tiny example."""
    print("\n=== Demonstrating T_vectorized Broadcasting Shapes ===")

    # X_n = np.array([
    #     [0.0, 0.0],
    #     [1.0, 0.0],
    #     [0.0, 1.0],
    # ])
    X_n = SquareLatticeQuantizer(0, 3, 0, 3, n=4).get_quantized_points()
    u = np.array([3, 3])

    predicted_states = pomdp.motion_model.simulate(X_n, u)
    predicted_expanded = predicted_states[:, np.newaxis, :]
    X_n_expanded = X_n[np.newaxis, :, :]
    squared_diff_full = (X_n_expanded - predicted_expanded) ** 2
    squared_diff = squared_diff_full.sum(axis=-1)

    print("X_n shape:", X_n.shape)
    print("X_n:\n", X_n)
    print("u shape:", u.shape)
    print("predicted_states shape:", predicted_states.shape)
    print("predicted_states:\n", predicted_states)
    print("predicted_expanded shape:", predicted_expanded.shape)
    print("X_n_expanded shape:", X_n_expanded.shape)
    print("squared_diff_full shape (m,m,2):", squared_diff_full.shape)
    print("squared_diff shape (m,m):", squared_diff.shape)

    # Use same numerical stabilization as implementation to avoid underflow/NaNs
    var = pomdp.σ_w ** 2
    min_per_row = np.min(squared_diff, axis=1, keepdims=True)
    stabilized = np.exp(-(squared_diff - min_per_row) / (2 * var))
    row_sums = stabilized.sum(axis=1, keepdims=True)
    T_demo = np.divide(stabilized, row_sums, out=np.zeros_like(
        stabilized), where=row_sums > 0)

    print("Demonstration T (row-normalized):\n", T_demo)

    return {
        "X_n": X_n,
        "predicted_states": predicted_states,
        "predicted_expanded_shape": predicted_expanded.shape,
        "X_n_expanded_shape": X_n_expanded.shape,
        "squared_diff_shape": squared_diff.shape,
        "T_demo": T_demo,
    }


def test_T_vectorized_entropy_vs_sigma(pomdp):
    """Sweep σ_w and report average row entropy to show determinism vs spread."""
    print("\n=== T_vectorized Row Entropy vs σ_w ===")
    X_n = SquareLatticeQuantizer(0, 3, 0, 3, n=4).get_quantized_points()
    u = np.array([0.4, 0.4])

    def row_entropy(T):
        eps = 1e-16
        P = np.clip(T, eps, 1.0)
        H_rows = -(P * np.log(P)).sum(axis=1)
        return H_rows.mean()

    sigmas = [0.005, 0.02, 0.05, 0.1, 0.2]
    entropies = []
    for s in sigmas:
        # Temporarily set sigma and compute T
        old = pomdp.σ_w
        pomdp.σ_w = s
        T = pomdp.T_vectorized(X_n, u)
        pomdp.σ_w = old
        ent = row_entropy(T)
        entropies.append(ent)
        print(f"σ_w={s:.3f} -> avg row entropy={ent:.6f}")

    return sigmas, entropies


def test_T_vectorized_outside_state_space(pomdp):
    """Probe behavior when predicted state lies outside the convex hull of X_n."""
    print("\n=== T_vectorized Outside State Space Behavior ===")
    # Very small grid and large action to push outside
    X_n = SquareLatticeQuantizer(0, 1, 0, 1, n=2).get_quantized_points()
    u = np.array([5.0, 5.0])

    old = pomdp.σ_w
    pomdp.σ_w = 0.05
    T = pomdp.T_vectorized(X_n, u)
    pomdp.σ_w = old

    print("X_n:\n", X_n)
    print("T (rows sum to 1):\n", T)
    print("row sums:", T.sum(axis=1))
    # Ensure no NaNs and rows still normalize
    assert not np.isnan(
        T).any(), "T contains NaNs; stabilization/normalization failed"
    assert np.allclose(
        T.sum(axis=1), 1.0), "Rows should sum to 1 even outside state space"

    return T


def test_performance_compare_T_versions(pomdp):
    """Benchmark T_vectorized (new) vs T_vectorized_v1 (old) across sizes."""
    print("\n=== Performance: T_vectorized vs T_vectorized_v1 ===")
    sizes = [4, 9, 16, 25, 36, 81, 16 * 16]
    timings_new = []
    timings_old = []

    for n in sizes:
        q = int(np.sqrt(n))
        X_n = SquareLatticeQuantizer(0, q - 1, 0, q - 1, n=n).get_quantized_points()
        u = np.array([0.4, -0.1])

        t0 = time.time()
        _ = pomdp.T_vectorized(X_n, u)
        t1 = time.time()
        _ = pomdp.T_vectorized_v1(X_n, u)
        t2 = time.time()

        timings_new.append(t1 - t0)
        timings_old.append(t2 - t1)
        print(
            f"states={n:>3} | new={timings_new[-1]:.6f}s | old={timings_old[-1]:.6f}s | speedup={(timings_old[-1]/max(timings_new[-1],1e-12)):.2f}x")

    # Plot if desired
    plt.close('all')
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, timings_new, 'g-o', label='T_vectorized (new)')
    plt.plot(sizes, timings_old, 'r-o', label='T_vectorized_v1 (old)')
    plt.xlabel('Number of states')
    plt.ylabel('Time (s)')
    plt.title('Transition kernel performance comparison')
    plt.grid(True)
    plt.legend()
    out_path = 'output/T_performance_compare.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print('Saved performance comparison to:', out_path)

    return sizes, timings_new, timings_old


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

def test_vectorized_observation(pomdp):
    """Test cartesian observation matrix calculation"""
    print("\n=== Testing Cartesian Transition Matrix ===")

    # Create a small set of states for testing
    X_n = SquareLatticeQuantizer(0, 3, 0, 3, n=10).get_quantized_points()
    n_y = 4
    B = pomdp.sensor.B
    Y = ObservationQuantizer(y_max=12, n=n_y, B=B).get_quantized_points()
    print(f"Y shape: {Y.shape}")
    assert Y.shape == (n_y**B, B), f"Y shape should be (n_y**B, B), but is {Y.shape}"
    m = pomdp.map.occupancy_map.data

    print(f"Testing observation matrix for {len(X_n)} states")
    print(f"States: {X_n}")
    # print(f"Observations: {Y}")

    start_time = time.time()
    Q_matrix = pomdp.Q_vectorized(Y, X_n, m)
    end_time = time.time()

    print(f"Observation matrix calculation time: {end_time - start_time:.4f}s")
    print(f"Observation matrix shape: {Q_matrix.shape}")
    print(f"Observation matrix:\n{Q_matrix}")

    # Verify properties
    row_sums = np.sum(Q_matrix, axis=1)
    print(f"Row sums (should be close to 1): {row_sums}")

    # Verify properties
    col_sums = np.sum(Q_matrix, axis=0)
    print(f"Column sums (should not be close to 1): {col_sums}")

    return Q_matrix

def test_ogm_to_segments_conversion():
    """Validate converting an occupancy grid to obstacle segments.

    We build a tiny map with known occupied cells forming a 2x3 block and
    assert that `_get_obstacles_from_map` returns the rectangle boundary
    segments with correct centroid and dimensions.
    """
    # Create a minimal POMDP with map resolution = 1.0 for easy reasoning
    from src.utils.map import load_obstacles_config
    from src.classes.mapping import LidarGridMapVec
    from src.classes.model import VelocityIntegratorModel, LIDAR
    from src.classes.pomdp import POMDP

    # Map spanning 0..4 in both axes -> 5x5 grid at resolution 1.0
    map_vec = LidarGridMapVec(0, 4, 0, 4, resolution=1.0)

    # Dummy motion/sensor (not used here)
    motion_model = VelocityIntegratorModel(i_x=0.0, i_y=0.0, dt=0.1, max_v=1.0)
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
    segments = pomdp._get_obstacles_from_map(m)

    # Expect a single rectangle with dx=3, dy=2, centroid at (2.5, 2.0)
    dx = 3.0
    dy = 2.0
    centroid = np.array([2.5, 2.0])  # geometric center of occupied rectangle at cell edges

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
    """Visualize the synthetic OGM and the recovered obstacle segments."""
    # Recreate the same synthetic setup as in test_ogm_to_segments_conversion
    from src.classes.mapping import LidarGridMapVec
    from src.classes.model import VelocityIntegratorModel, LIDAR
    from src.classes.pomdp import POMDP

    map_vec = LidarGridMapVec(0, 4, 0, 4, resolution=1.0)
    motion_model = VelocityIntegratorModel(i_x=0.0, i_y=0.0, dt=0.1, max_v=1.0)
    sensor = LIDAR(fov=360, r_max=10, B=36)
    pomdp = POMDP(motion_model=motion_model,
                  measurement_model=sensor,
                  obstacles=[],
                  _map=map_vec,
                  sigma_w=0.1,
                  sigma_v=0.1)

    H, W = 5, 5
    m = np.zeros((H, W), dtype=float)
    m[1:3, 1:4] = 1.0

    segments = pomdp._get_obstacles_from_map(m)

    # Plot OGM and overlay segments
    plt.close('all')
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title('OGM with recovered obstacle segments')
    ax.imshow(m, cmap='gray_r', origin='lower', extent=[0, 5, 0, 5], alpha=0.5)

    # Draw recovered rectangle segments (red)
    for (x1, y1, x2, y2) in segments:
        ax.plot([x1, x2], [y1, y2], 'r-', linewidth=2)


    ax.set_xlim([0, 5])
    ax.set_ylim([0, 5])
    ax.set_aspect('equal', adjustable='box')
    # Legend (proxy artists)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    occupied_proxy = Patch(facecolor='dimgray', edgecolor='none', alpha=0.5, label='Occupied cells (dark)')
    segment_proxy = Line2D([0], [0], color='red', lw=2, label='Recovered segments')
    ax.legend(handles=[occupied_proxy, segment_proxy], loc='upper right')
    plt.tight_layout()
    out_path = 'output/ogm_segments_validation.png'
    plt.savefig(out_path, dpi=150)
    print('Saved OGM-to-segments visualization to:', out_path)
    return out_path


def test_Q_vectorized_indexing_validation(pomdp):
    """Validate Q_vectorized vs loop baseline: rows=states i, cols=observations k."""
    print("\n=== Validating Q_vectorized Indexing (rows: state i, cols: obs k) ===")
    # Small set: 2 states, 3 observations
    X_n = np.array([[0.0, 0.0], [1.0, 1.0]])
    # Create fake observations by perturbing ideal rays
    y_star = pomdp.ray_casting(X_n, pomdp.map.occupancy_map.data)  # (2,B)
    # Pick 3 observations: exact y_star[0], exact y_star[1], and a mid average
    Y = np.vstack([
        y_star[0],
        y_star[1],
        0.5 * (y_star[0] + y_star[1])
    ])  # (3,B)

    Q_vec = pomdp.Q_vectorized(Y, X_n, pomdp.map.occupancy_map.data)  # (2,3)

    # Loop baseline with stabilized weights
    var = pomdp.σ_v ** 2
    m = X_n.shape[0]
    y_len = Y.shape[0]
    Q_loop = np.zeros((m, y_len))
    for i in range(m):
        d2_row = np.array([np.sum((Y[k] - y_star[i])**2)
                          for k in range(y_len)])
        min_i = d2_row.min()
        w = np.exp(-(d2_row - min_i) / (2 * var))
        s = w.sum()
        Q_loop[i, :] = w / s if s > 0 else 0.0

    print("Q_vec:\n", Q_vec)
    print("Q_loop baseline:\n", Q_loop)
    assert np.allclose(Q_vec, Q_loop, atol=1e-10)
    return Q_vec, Q_loop


def demonstrate_Q_vectorized_broadcasting(pomdp):
    """Show shapes for Q_vectorized broadcasting and stabilized normalization."""
    print("\n=== Demonstrating Q_vectorized Broadcasting Shapes ===")
    X_n = SquareLatticeQuantizer(
        0, 1, 0, 1, n=2).get_quantized_points()  # (4,2)
    y_star = pomdp.ray_casting(X_n, pomdp.map.occupancy_map.data)
    Y = np.vstack([
        y_star[0],
        y_star[1],
        0.5 * (y_star[0] + y_star[1])
    ])

    y_star_expanded = y_star[:, np.newaxis, :]  # (m,1,B)
    Y_expanded = Y[np.newaxis, :, :]  # (1,y_len,B)
    diff = Y_expanded - y_star_expanded
    squared_diff = np.sum(diff**2, axis=-1)

    print("X_n shape:", X_n.shape)
    print("y_star shape:", y_star.shape)
    print("Y shape:", Y.shape)
    print("y_star_expanded shape:", y_star_expanded.shape)
    print("Y_expanded shape:", Y_expanded.shape)
    print("diff shape:", diff.shape)
    print("squared_diff shape:", squared_diff.shape)

    var = pomdp.σ_v ** 2
    min_per_row = np.min(squared_diff, axis=1, keepdims=True)
    stabilized = np.exp(-(squared_diff - min_per_row) / (2 * var))
    row_sums = stabilized.sum(axis=1, keepdims=True)
    Q_demo = np.divide(stabilized, row_sums, out=np.zeros_like(
        stabilized), where=row_sums > 0)
    print("Q_demo (row-normalized):\n", Q_demo)
    return Q_demo


def test_Q_entropy_vs_sigma(pomdp):
    """Sweep σ_v and report avg row entropy of Q."""
    print("\n=== Q_vectorized Row Entropy vs σ_v ===")
    X_n = SquareLatticeQuantizer(0, 1, 0, 1, n=2).get_quantized_points()
    y_star = pomdp.ray_casting(X_n, pomdp.map.occupancy_map.data)
    Y = np.vstack([
        y_star[0],
        y_star[1],
        0.5 * (y_star[0] + y_star[1])
    ])

    def row_entropy(T):
        eps = 1e-16
        P = np.clip(T, eps, 1.0)
        H_rows = -(P * np.log(P)).sum(axis=1)
        return H_rows.mean()

    sigmas = [0.01, 0.05, 0.1, 0.2]
    entropies = []
    for s in sigmas:
        old = pomdp.σ_v
        pomdp.σ_v = s
        Q = pomdp.Q_vectorized(Y, X_n, pomdp.map.occupancy_map.data)
        pomdp.σ_v = old
        ent = row_entropy(Q)
        entropies.append(ent)
        print(f"σ_v={s:.3f} -> avg row entropy={ent:.6f}")
    return sigmas, entropies


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
    from src.classes.mapping import LidarGridMapVec
    from src.classes.model import VelocityIntegratorModel, LIDAR
    from src.classes.pomdp import POMDP

    # Tiny map
    map_vec = LidarGridMapVec(0, 4, 0, 4, resolution=1.0)
    motion_model = VelocityIntegratorModel(i_x=0.0, i_y=0.0, dt=0.1, max_v=1.0)
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
    from src.classes.mapping import LidarGridMapVec
    from src.classes.model import VelocityIntegratorModel, LIDAR
    from src.classes.pomdp import POMDP

    # Build a 3x3 grid: x,y in [0,2] with resolution 1.0 -> width=3, height=3
    map_vec = LidarGridMapVec(0, 2, 0, 2, resolution=1.0)
    motion_model = VelocityIntegratorModel(i_x=0.0, i_y=0.0, dt=0.1, max_v=1.0)
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
    from src.classes.mapping import LidarGridMapVec
    from src.classes.model import VelocityIntegratorModel, LIDAR
    from src.classes.pomdp import POMDP

    # 4x4 grid: x,y in [0,3] with resolution 1.0 -> width=4, height=4
    map_vec = LidarGridMapVec(0, 3, 0, 3, resolution=1.0)
    motion_model = VelocityIntegratorModel(i_x=0.0, i_y=0.0, dt=0.1, max_v=1.0)
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

# 6. Performance analysis
def test_performance_analysis(pomdp):
    """Analyze performance bottlenecks"""
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

        start_time = time.time()
        T_matrix = pomdp.T_vectorized(X_n, u)
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
    plt.title('Transition Matrix Calculation Time')
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
    gridmap = LidarGridMap(*area, resolution=pomdp.map.xy_resolution)

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

    # Build obstacle segments from current occupancy grid via POMDP helpers
    m = pomdp.map.occupancy_map.data
    components = pomdp._find_connected_components(m)
    segments = []
    for comp in components:
        segments.extend(pomdp._create_obstacle_from_component(comp))

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
    ogm_segments = pomdp._get_obstacles_from_map(grid_data)
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
    ogm_segments = pomdp._get_obstacles_from_map(grid_data)
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

    # 1. Test initialization
    pomdp = test_pomdp_initialization_3x3()

    # 2. Test transition matrix functionality
    # test_vectorized_transition(pomdp)
    # test_T_vectorized_indexing_validation(pomdp)
    # demonstrate_T_vectorized_broadcasting(pomdp)
    # test_T_vectorized_entropy_vs_sigma(pomdp)
    # test_T_vectorized_outside_state_space(pomdp)

    # 2b. Compare new vs old T implementations
    # test_compare_T_vectorized_versions(pomdp)
    # test_performance_compare_T_versions(pomdp)

    # 3. Observation model
    # test_ray_casting(pomdp)
    # test_vectorized_observation(pomdp)
    # test_ogm_to_segments_conversion()
    # test_visualize_ogm_to_segments()
    # test_Q_vectorized_indexing_validation(pomdp)
    # demonstrate_Q_vectorized_broadcasting(pomdp)
    # test_Q_entropy_vs_sigma(pomdp)

    # 4. Cost functions
    # test_cost_functions(pomdp)
    # test_c_vff_manual_validation()

    # 5. Map generation
    # test_generate_space_of_maps_3x3()
    # test_generate_space_of_maps_4x4()

    # 6. Visualization snapshot
    # test_visualize_single_pose_lidar(pomdp)
    # test_visualize_arena_and_hits(pomdp)
    # test_visualize_ogm_and_raycasting_hits_4x4(pomdp)

    # # Performance analysis
    # test_performance_analysis(pomdp)

    print("\n" + "=" * 50)
    print("POMDP Test Suite Complete!")


if __name__ == "__main__":
    main()
