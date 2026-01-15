# Use CuPy backend for GPU acceleration (falls back to NumPy if not available)
# Note: T matrix computation stays on CPU (sunk cost), GPU acceleration focused on H, F, η_n
from ..utils.array_backend import np
from .model import SingleIntegratorModel, DoubleIntegratorModel, LIDAR
from .obstacle import Obstacle
from .mapping import LidarGridMapVec
from scipy.stats import multivariate_normal as mvn
import numpy as numpy_cpu
# Optional tqdm import for progress bars
try:
    from tqdm import tqdm
    import tqdm as tqdm_module
    _tqdm_available = True

    def _is_tqdm_active():
        """Check if there's an active tqdm instance (for nested progress bars)."""
        try:
            # Check if there are any active tqdm instances
            return len(tqdm_module.tqdm._instances) > 0
        except (AttributeError, TypeError):
            return False
except ImportError:
    _tqdm_available = False
    # Fallback: create a no-op tqdm-like object

    def tqdm(iterable, *args, **kwargs):
        return iterable

    def _is_tqdm_active():
        return False


class BasePOMDP:
    """
    Base POMDP class with minimal shared functionality across all POMDP problem types.

    Provides only common initialization and noise parameters.
    Subclasses should implement problem-specific functionality:
    - Transition kernels (T)
    - Observation models (Q)
    - Cost functions (c)
    - Ray casting methods

    POMDP is defined as a six tuple (X, U, Y, S, Q, c)
    """

    def __init__(self,
                 motion_model: SingleIntegratorModel | DoubleIntegratorModel,
                 measurement_model: LIDAR,
                 obstacles: list[Obstacle],
                 _map: LidarGridMapVec,
                 sigma_w: float = 0.01,
                 sigma_v: float = 0.01,
                 force_constant: float = 100
                 ):
        self.motion_model = motion_model
        self.sensor = measurement_model
        self.obstacles = obstacles
        self.map = _map

        if isinstance(motion_model, SingleIntegratorModel):
            self.σ_w = sigma_w * np.sqrt(motion_model.dt)
            self.σ_v = sigma_v
            self.cov_x = np.eye(2) * sigma_w * sigma_w * motion_model.dt
            self.cov_y = np.eye(measurement_model.B) * sigma_v * sigma_v
        elif isinstance(motion_model, DoubleIntegratorModel):
            self.σ_w = sigma_w
            self.σ_v = sigma_v
            self.cov_x = motion_model.Q_t * sigma_w
            self.cov_y = np.eye(measurement_model.B) * sigma_v * sigma_v

        self._map_cache = {}
        self._integration_cache = {}
        self.force_constant = force_constant

    @staticmethod
    def c_effort(u):
        """
        Cost function for the effort of the control action u.
        """
        return np.linalg.norm(u, ord=2)


class Localization_POMDP(BasePOMDP):
    """
    Localization-specific POMDP class handling localization only.

    State space: x only (robot pose)
    - m is known/constant (not part of state)

    Belief space: π(x) - shape (m_n,)
    - Only over robot pose
    - Map is fixed/known, represented as obstacle segments from YAML config
    - No occupancy grid maps - works directly with obstacle segments
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.known_map_obstacle_segments = None

    ### 1. Stochastic Kernels #################################
    def T(self, B: np.ndarray, X: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Transition probabilities T(B | X, u) for Borel set B given states X and action u.

        T(B | x, u) = ∫_B N(x'; f_bar(x,u), Σ_w) dx'
        where f_bar(x,u) is deterministic next state and Σ_w is process noise covariance

        Args:
            B: Borel set represented as bounds array:
               - For 2D states: shape (2, 2) with [[x_min, x_max], [y_min, y_max]]
               - For 4D states: shape (4, 2) with [[px_min, px_max], [py_min, py_max], [vx_min, vx_max], [vy_min, vy_max]]
            X: current states (n, d) where d=2 for SingleIntegrator, d=4 for DoubleIntegrator
            u: action (2,)
        Returns:
            Probability masses over B for each state in X (n,)
        """
        if X.ndim == 1:
            X = X[np.newaxis, :]

        state_dim = X.shape[1]
        B_dim = B.shape[0]

        if state_dim != B_dim:
            raise ValueError(f"State dimension {state_dim} does not match Borel set dimension {B_dim}")

        X_cpu = numpy_cpu.asarray(X)
        B_cpu = numpy_cpu.asarray(B)
        u_cpu = numpy_cpu.asarray(u)
        cov_cpu = numpy_cpu.asarray(self.cov_x)

        mu = numpy_cpu.asarray(self.motion_model.f_bar(X_cpu, u_cpu))
        mins = B_cpu[:, 0]
        maxs = B_cpu[:, 1]

        probs = numpy_cpu.zeros(len(X_cpu))
        for i, mu_i in enumerate(mu):
            probs[i] = mvn.cdf(x=maxs, mean=mu_i, cov=cov_cpu, lower_limit=mins)

        return np.asarray(probs)

    ###########################################################
    ### 2. Cost Functions #####################################
    def c(self, x, u):
        """
        Cost function for localization: focuses on pose estimation accuracy and collision avoidance.
        Map is known, so cost depends only on pose x and action u.
        """
        return self.c_effort(u) + self.c_collision(x, u)

    def c_collision(self, x, u, ε=1e-6):
        """
        Cost function for collision avoidance in localization.
        """
        pass

    def c_vectorized_batch(self, X_batch: np.ndarray, U_batch: np.ndarray) -> np.ndarray:
        """
        Batched cost computation: c(x, u) for all (x, u) pairs.

        Args:
            X_batch: states (n_x, state_dim) or (state_dim,)
            U_batch: actions (n_u, 2) or (2,)
        Returns:
            c_batch: (n_x, n_u) where c_batch[i, j] = c(X_batch[i], U_batch[j])
        """
        if X_batch.ndim == 1:
            X_batch = X_batch[np.newaxis, :]
        if U_batch.ndim == 1:
            U_batch = U_batch[np.newaxis, :]

        effort = np.linalg.norm(U_batch, axis=1)[np.newaxis, :]
        collision = np.zeros((X_batch.shape[0], U_batch.shape[0]), dtype=np.float32)
        return effort + collision

    ##########################################################
    ### 3. Ray Casting #######################################
    def ray_casting(self, X):
        """
        Ray casting for localization - uses pre-computed obstacle segments from known map.

        Args:
            X: Array of robot positions (m_n, 2)

        Returns:
            y_star: Array of distances (m_n, B)
        """
        if self.known_map_obstacle_segments is None:
            raise ValueError("Known map obstacle segments not set. Call set_known_map() first.")
        return self.sensor.g_bar_localization(X, self.known_map_obstacle_segments)

    def set_known_map(self, obstacle_segments):
        """
        Set the known map for localization using obstacle segments.

        For localization, we work directly with obstacle segments loaded from YAML config,
        not occupancy grid maps. Obstacle segments are tuples of (x1, y1, x2, y2).

        Args:
            obstacle_segments: List of obstacle segments, where each segment is a tuple (x1, y1, x2, y2)
                               Can also be a list of Obstacle objects, which will be converted to segments
        """
        from .obstacle import Obstacle

        # If given Obstacle objects, extract segments from them
        if obstacle_segments and isinstance(obstacle_segments[0], Obstacle):
            all_segments = []
            for obs in obstacle_segments:
                # Get segments at current centroid position
                segments = obs._Obstacle__get_points(obs.centroid)
                all_segments.extend(list(segments))
            self.known_map_obstacle_segments = all_segments
        else:
            # Assume it's already a list of segments
            self.known_map_obstacle_segments = obstacle_segments

    def Q_vectorized_batch(self, Y_batch: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Batched Q computation: Q(y | x) for all (y, x) pairs using known map.

        Args:
            Y_batch: observations (n_obs, B)
            X: states (m_n, state_dim)
        Returns:
            Q_batch: (n_obs, m_n) where Q_batch[k, i] = Q(Y_batch[k] | X[i])
        """
        if self.known_map_obstacle_segments is None:
            raise ValueError("Known map obstacle segments not set. Call set_known_map() first.")

        X_pos = X[:, :2] if X.shape[1] > 2 else X
        y_star = self.ray_casting(X_pos)

        y_diff = Y_batch[:, np.newaxis, :] - y_star[np.newaxis, :, :]
        log_const = -0.5 * self.sensor.B * np.log(2 * np.pi) - self.sensor.B * np.log(self.σ_v)
        squared_diff = np.sum(y_diff ** 2, axis=2)
        log_Q = log_const - 0.5 * squared_diff / (np.square(self.σ_v))

        return np.exp(log_Q)


class Mapping_POMDP(BasePOMDP):
    """
    Mapping-specific POMDP class handling mapping only.

    State space: m only (occupancy grid map)
    - x is known/constant (robot pose is tracked separately)

    Belief space: π(m) - shape (len_M,)
    - Only over map
    - Pose is fixed/known
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.known_pose = None

    ###########################################################
    ### 1. Cost Functions #####################################
    def c(self, m, u):
        """
        Cost function for mapping: focuses on map exploration and information gain.
        Pose is known, so cost depends only on map m and action u.
        """
        return self.c_effort(u) + self.c_vff(self.known_pose, m, u)

    def F_r(self, x, m):
        """Repulsive force from obstacles using occupancy grid map."""
        x_pos = x[:2] if len(x) > 2 else x
        m_pos = self.map.occupancy_map.get_occupied_map_positions(m)

        diff = m_pos - x_pos
        d = np.linalg.norm(diff, axis=1)
        within = d <= 10.0

        if not np.any(within):
            return np.zeros(2, dtype=np.float32)

        diff = diff[within]
        d = d[within]
        inv_d3 = 1.0 / np.maximum(d, 1e-6)**3
        forces = (self.force_constant * diff) * inv_d3[:, None]

        return forces.sum(axis=0)

    def c_vff(self, x, m, u, ε=1e-6):
        """Collision avoidance cost for mapping."""
        F_r = self.F_r(x, m)
        dot = np.dot(F_r, u)
        norms = np.linalg.norm(F_r) * np.linalg.norm(u) + ε
        return np.maximum(0, dot / norms)

    def F_r_vectorized_batch(self, X_batch: np.ndarray, M_batch: np.ndarray) -> np.ndarray:
        """
        Batched repulsive force computation: F_r(x, m) for all (x, m) pairs.

        Args:
            X_batch: positions (n_x, 2) or (2,)
            M_batch: maps (n_m, H, W) or (H, W)
        Returns:
            F_r_batch: (n_x, n_m, 2) where F_r_batch[i, j] = F_r(X_batch[i], M_batch[j])
        """
        if X_batch.ndim == 1:
            X_batch = X_batch[np.newaxis, :]
        if M_batch.ndim == 2:
            M_batch = M_batch[np.newaxis, :, :]

        n_x, n_m = X_batch.shape[0], M_batch.shape[0]
        m_positions, valid_mask = self.map.occupancy_map.get_occupied_map_positions_batched(M_batch)

        if m_positions.shape[1] == 0 or not np.any(valid_mask):
            return np.zeros((n_x, n_m, 2), dtype=np.float32)

        # Broadcast: (n_x, 1, 1, 2) - (1, n_m, max_occupied, 2) -> (n_x, n_m, max_occupied, 2)
        diff = X_batch[:, np.newaxis, np.newaxis, :] - m_positions[np.newaxis, :, :, :]
        d = np.linalg.norm(diff, axis=3)

        # Mask: valid cells within max_D range
        mask = (d <= 10.0) & valid_mask[np.newaxis, :, :]
        inv_d3 = (1.0 / np.maximum(d, 1e-6)**3) * mask

        # Repulsive force: sum over occupied cells
        forces = (-self.force_constant * diff) * inv_d3[:, :, :, np.newaxis]
        return np.sum(forces, axis=2)

    def c_vectorized_batch(self, M_batch: np.ndarray, U_batch: np.ndarray, X_known: np.ndarray = None) -> np.ndarray:
        """
        Batched cost computation: c(m, u) for all (m, u) pairs.

        Args:
            M_batch: maps (n_m, H, W) or (H, W)
            U_batch: actions (n_u, 2) or (2,)
            X_known: known pose (2,) or (state_dim,)
        Returns:
            c_batch: (n_m, n_u) where c_batch[i, j] = c(M_batch[i], U_batch[j])
        """
        if X_known is None:
            if self.known_pose is None:
                raise ValueError("X_known must be provided or self.known_pose must be set")
            X_known = self.known_pose

        if M_batch.ndim == 2:
            M_batch = M_batch[np.newaxis, :, :]
        if U_batch.ndim == 1:
            U_batch = U_batch[np.newaxis, :]

        x_pos = X_known[:2] if len(X_known) > 2 else X_known
        x_pos_batch = np.broadcast_to(x_pos[np.newaxis, :], (M_batch.shape[0], 2))

        # Effort cost: ||u||_2
        effort = np.linalg.norm(U_batch, axis=1)[np.newaxis, :]

        # VFF cost: repulsive force alignment
        F_r = self.F_r_vectorized_batch(x_pos_batch, M_batch)[:, 0, :]
        dot_prod = np.dot(F_r, U_batch.T)
        norms = np.linalg.norm(F_r, axis=1, keepdims=True) * np.linalg.norm(U_batch, axis=1)[np.newaxis, :]
        vff = np.maximum(0, dot_prod / (norms + 1e-6))

        return effort + vff

    ##########################################################
    ### 2. Ray Casting #######################################
    def ray_casting_batched(self, X, M):
        """
        Batched ray casting for multiple maps simultaneously.

        Args:
            X: robot positions (m_n, 2)
            M: occupancy grids (len_M, H, W)
        Returns:
            y_star_all: distances (m_n, len_M, B)
        """
        return self.sensor.g_bar_batched(X, M, self.map)

    ##########################################################
    ### 3. Observation Models ###############################
    def Q_vectorized_batch(self, Y_batch: np.ndarray) -> np.ndarray:
        """
        Batched Q computation: Q(y | x_known, m) for all (y, m) pairs.

        Args:
            Y_batch: observations (n_obs, B)
        Returns:
            Q_batch: (n_obs, len_M) where Q_batch[k, j] = Q(Y_batch[k] | x_known, maps[j])
        """
        return np.exp(self.Q_log_vectorized_batch(Y_batch))

    def Q_log_vectorized_batch(self, Y_batch: np.ndarray) -> np.ndarray:
        """
        Batched log Q computation: log Q(y | x_known, m) for all (y, m) pairs.

        Args:
            Y_batch: observations (n_obs, B)
        Returns:
            log_Q_batch: (n_obs, len_M) where log_Q_batch[k, j] = log Q(Y_batch[k] | x_known, maps[j])
        """
        if self.known_pose is None:
            raise ValueError("Known pose not set. Set self.known_pose first.")
        if not hasattr(self, 'all_maps_3d') or self.all_maps_3d is None:
            raise ValueError(
                "Q_log_vectorized_batch requires all_maps_3d to be precomputed. "
                f"Map space too large (H*W={getattr(self, 'map_H', '?')}*{getattr(self, 'map_W', '?')} > 16)."
            )

        x_pos = self.known_pose[:2] if len(self.known_pose) > 2 else self.known_pose
        y_star = self.ray_casting_batched(x_pos[np.newaxis, :], self.all_maps_3d)[0]

        y_diff = Y_batch[:, np.newaxis, :] - y_star[np.newaxis, :, :]
        log_const = -0.5 * self.sensor.B * np.log(2 * np.pi) - self.sensor.B * np.log(self.σ_v)
        squared_diff = np.sum(y_diff ** 2, axis=2)
        return log_const - 0.5 * squared_diff / (np.square(self.σ_v))

    ##########################################################
    ### 3. Map Space Utilities ##############################
    def generate_space_of_maps(self, method='bit_iteration', callback=None, show_progress=False, desc=None):
        """
        Generate maps from the space M = {0,1}^{HW} efficiently.

        Args:
            method: Method to use ('bit_iteration', 'full_space', 'symmetry_reduced')
            callback: Optional callback function to process each map
            show_progress: If True, show progress bar (requires tqdm)
            desc: Description for progress bar (default: "Processing maps")

        Returns:
            If callback is None: Array of shape (num_maps, H*W)
            If callback is provided: None (maps are processed via callback)
        """
        H = self.map.occupancy_map.height
        W = self.map.occupancy_map.width
        total_cells = H * W
        total_maps = 2 ** total_cells

        if method == 'bit_iteration':
            if callback is None:
                # Return all maps (use with caution for large spaces)
                maps = []
                iterator = range(total_maps)
                if show_progress and _tqdm_available:
                    iterator = tqdm(iterator, desc=desc or "Generating maps", leave=False, unit="map")
                for map_bits in iterator:
                    map_array = self.bits_to_map(map_bits, (H, W))
                    maps.append(map_array.flatten())
                return np.array(maps)
            else:
                # Process maps via callback (memory efficient)
                iterator = range(total_maps)
                if show_progress and _tqdm_available:
                    # Check if we're already in a tqdm context (nested)
                    is_nested = _is_tqdm_active()
                    if is_nested:
                        # When nested, disable progress bar (it conflicts with outer bar and tqdm.write calls)
                        # Instead, we'll rely on the callback to print periodic updates via tqdm.write()
                        iterator = iterator  # Keep as plain range
                    else:
                        # Standalone progress bar
                        iterator = tqdm(iterator, desc=desc or "Processing maps", leave=False, unit="map")
                for map_bits in iterator:
                    map_array = self.bits_to_map(map_bits, (H, W))
                    callback(map_bits, map_array)
                return None

        elif method == 'full_space':
            # For small spaces, return everything
            if total_cells <= 16:  # 2^16 = 65536 maps
                return self.generate_space_of_maps(method='bit_iteration', callback=callback,
                                                   show_progress=show_progress, desc=desc)
            else:
                raise ValueError(
                    f"Map space too large ({total_maps} maps). Use callback-based methods.")

        else:
            raise ValueError(f"Unknown method: {method}")

    def generate_all_maps_3d(self, show_progress=False):
        """
        Generate all maps as a 3D array for vectorized operations.

        Returns:
            maps_3d: Array of shape (num_maps, H, W) containing all possible maps
            map_bits: Array of shape (num_maps,) containing the bit representation for each map
        """
        H = self.map.occupancy_map.height
        W = self.map.occupancy_map.width
        total_cells = H * W
        total_maps = 2 ** total_cells

        maps_3d = []
        map_bits_list = []

        iterator = range(total_maps)
        if show_progress and _tqdm_available:
            iterator = tqdm(iterator, desc="Generating all maps", leave=False, unit="map")

        for map_bits in iterator:
            map_array = self.bits_to_map(map_bits, (H, W))
            maps_3d.append(map_array)
            map_bits_list.append(map_bits)

        # Stack into 3D array: (num_maps, H, W)
        maps_3d_array = np.stack(maps_3d, axis=0)
        map_bits_array = np.array(map_bits_list, dtype=np.int64)

        return maps_3d_array, map_bits_array

    def map_to_bits(self, m: np.ndarray) -> int:
        """
        Encode an occupancy grid m \in {0,1}^{H\times W} into an integer by row-major bits.

        Bit k corresponds to m.flat[k] (row-major order), with least-significant bit = index 0.
        """
        flat = np.asarray(m, dtype=np.uint8).ravel(order='C')
        bits = 0
        for idx, v in enumerate(flat):
            if v:
                bits |= (1 << idx)
        return bits

    def bits_to_map(self, bits: int, shape: tuple[int, int]) -> np.ndarray:
        """
        Decode integer bits into an occupancy grid of given shape (H,W), row-major.
        """
        H, W = shape
        total = H * W
        out = np.zeros(total, dtype=np.uint8)
        for k in range(total):
            out[k] = (bits >> k) & 1
        return out.reshape((H, W), order='C')

    def _symmetry_transforms(self, m: np.ndarray) -> list[np.ndarray]:
        """
        Generate the 8 dihedral symmetries (D4) of the grid: rotations and flips.
        """
        mats = []
        # Rotations: 0, 90, 180, 270
        for k in range(4):
            r = np.rot90(m, k=k)
            mats.append(r)
            mats.append(np.fliplr(r))
        return mats

    def _bitset_min_dtype(self, total_bits: int):
        """Choose the smallest unsigned integer dtype that can hold total_bits bits.
        Returns a NumPy dtype or None if >64 bits (use Python int)."""
        if total_bits <= 8:
            return np.uint8
        if total_bits <= 16:
            return np.uint16
        if total_bits <= 32:
            return np.uint32
        if total_bits <= 64:
            return np.uint64
        return None

    def generate_space_of_map_ids(self, as_numpy: bool = True, dtype=None, callback=None):
        """
        Generate the map space as compact bitset IDs rather than arrays.

        - If callback is provided: iterate over all IDs and call callback(map_bits) for each.
        - If as_numpy and total_bits<=64: return a NumPy array of chosen dtype with values [0..2^{HW}-1].
        - Otherwise: return a Python list of ints.
        """
        H = self.map.occupancy_map.height
        W = self.map.occupancy_map.width
        total_bits = H * W
        total_maps = 1 << total_bits

        # Select dtype if not given
        if dtype is None:
            dtype = self._bitset_min_dtype(total_bits)

        if callback is not None:
            # Stream through all IDs without storing
            for b in range(total_maps):
                callback(b)
            return None

        if as_numpy and dtype is not None:
            # Use vectorized range with chosen dtype
            return np.arange(total_maps, dtype=dtype)

        # Fallback: Python list of ints (works for any size, but memory heavy)
        return list(range(total_maps))


class SLAM_POMDP(Localization_POMDP, Mapping_POMDP):
    """
    SLAM-specific POMDP class handling Simultaneous Localization and Mapping.

    Uses multiple inheritance from Localization_POMDP and Mapping_POMDP to combine:
    - Transition kernel T() from Localization_POMDP
    - Map space utilities from Mapping_POMDP
    - Ray casting methods from Mapping_POMDP (ray_casting_batched for occupancy grids)

    Method Resolution Order (MRO): SLAM_POMDP -> Localization_POMDP -> Mapping_POMDP -> BasePOMDP
    - Methods defined in SLAM_POMDP override parent methods
    - For ray_casting_batched, we explicitly use Mapping_POMDP's version (inherited)
    - For Q_vectorized_batch, we override both parents with SLAM-specific version

    State space: Joint (x, m) where:
    - x: Robot pose (2D or 4D)
    - m: Occupancy grid map

    Belief space: π(x, m) - joint belief over pose and map.

    Extends both Localization_POMDP and Mapping_POMDP with SLAM-specific functionality:
    - Observation models Q(y | x, m) that depend on both pose and map
    - Combined cost functions for SLAM
    """

    def __init__(self, *args, **kwargs):
        # Call both parent __init__ methods
        # Note: Both parents inherit from BasePOMDP, but BasePOMDP.__init__ is idempotent
        Localization_POMDP.__init__(self, *args, **kwargs)
        Mapping_POMDP.__init__(self, *args, **kwargs)

    ###########################################################
    ### Cost Functions #####################################
    def c_vectorized_batch(self, X_batch: np.ndarray, M_batch: np.ndarray, U_batch: np.ndarray) -> np.ndarray:
        """
        Batched cost computation: c(x, m, u) for all (x, m, u) combinations.

        Args:
            X_batch: states (n_x, state_dim) or (state_dim,)
            M_batch: maps (n_m, H, W) or (H, W)
            U_batch: actions (n_u, 2) or (2,)
        Returns:
            c_batch: (n_x, n_m, n_u) where c_batch[i, j, k] = c(X_batch[i], M_batch[j], U_batch[k])
        """
        if X_batch.ndim == 1:
            X_batch = X_batch[np.newaxis, :]
        if M_batch.ndim == 2:
            M_batch = M_batch[np.newaxis, :, :]
        if U_batch.ndim == 1:
            U_batch = U_batch[np.newaxis, :]

        n_x, n_m, n_u = X_batch.shape[0], M_batch.shape[0], U_batch.shape[0]
        X_pos = X_batch[:, :2] if X_batch.shape[1] > 2 else X_batch

        # Effort cost
        effort = np.linalg.norm(U_batch, axis=1)[np.newaxis, np.newaxis, :]

        # VFF cost: compute F_r for all (x, m) pairs
        X_pos_expanded = np.repeat(X_pos[:, np.newaxis, :], n_m, axis=1)
        M_expanded = np.repeat(M_batch[np.newaxis, :, :, :], n_x, axis=0)
        F_r = self.F_r_vectorized_batch(X_pos_expanded.reshape(-1, 2),
                                        M_expanded.reshape(-1, *M_batch.shape[1:]))[:, 0, :]
        F_r = F_r.reshape(n_x, n_m, 2)

        # Dot products and norms
        dot_prod = np.sum(F_r[:, :, np.newaxis, :] * U_batch[np.newaxis, np.newaxis, :, :], axis=3)
        norms = np.linalg.norm(F_r, axis=2, keepdims=True) * np.linalg.norm(U_batch, axis=1)[np.newaxis, np.newaxis, :]
        vff = np.maximum(0, dot_prod / (norms + 1e-6))

        return effort + vff

    ##########################################################
    ### Observation Models ###################################
    def Q_vectorized_batch(self, Y_batch: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Batched Q computation: Q(y | x, m) for all (y, x, m) pairs.

        Overrides parent implementations to handle both pose and map.

        Args:
            Y_batch: observations (n_obs, B)
            X: states (m_n, state_dim)
        Returns:
            Q_batch: (n_obs, m_n, len_M) where Q_batch[k, i, j] = Q(Y_batch[k] | X[i], maps[j])
        """
        return np.exp(self.Q_log_vectorized_batch(Y_batch, X))

    def Q_log_vectorized_batch(self, Y_batch: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Batched log Q computation: log Q(y | x, m) for all (y, x, m) pairs.

        Args:
            Y_batch: observations (n_obs, B)
            X: states (m_n, state_dim)
        Returns:
            log_Q_batch: (n_obs, m_n, len_M) where log_Q_batch[k, i, j] = log Q(Y_batch[k] | X[i], maps[j])
        """
        if not hasattr(self, 'all_maps_3d') or self.all_maps_3d is None:
            raise ValueError(
                "Q_log_vectorized_batch requires all_maps_3d to be precomputed. "
                f"Map space too large (H*W={getattr(self, 'map_H', '?')}*{getattr(self, 'map_W', '?')} > 16)."
            )

        X_pos = X[:, :2] if X.shape[1] > 2 else X
        y_star = self.ray_casting_batched(X_pos, self.all_maps_3d)

        y_diff = Y_batch[:, np.newaxis, np.newaxis, :] - y_star[np.newaxis, :, :, :]
        log_const = -0.5 * self.sensor.B * np.log(2 * np.pi) - self.sensor.B * np.log(self.σ_v)
        squared_diff = np.sum(y_diff ** 2, axis=3)
        return log_const - 0.5 * squared_diff / (np.square(self.σ_v))


# Backward compatibility: Maintain POMDP as alias for SLAM_POMDP
# Old code using POMDP will still work
POMDP = SLAM_POMDP

__all__ = ['POMDP', 'SLAM_POMDP', 'Localization_POMDP', 'Mapping_POMDP']
