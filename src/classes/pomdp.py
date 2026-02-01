# Use CuPy backend for GPU acceleration (falls back to NumPy if not available)
# Note: T matrix computation stays on CPU (sunk cost), GPU acceleration focused on H, F, η_n
from ..utils.array_backend import np, is_cupy
from .model import SingleIntegratorModel, DoubleIntegratorModel, LIDAR, RangeBearingSensor
from .obstacle import Obstacle
from .mapping import BaseMap, LidarGridMapVec, OrderedLandmarkMap
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm as univariate_norm
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
                 measurement_model,  # LIDAR or RangeBearingSensor
                 obstacles: list[Obstacle],
                 _map: BaseMap,
                 sigma_w: float = 0.01,
                 sigma_v: float = 0.01,
                 force_constant: float = 100
                 ):
        self.motion_model = motion_model
        self.sensor = measurement_model
        self.obstacles = obstacles
        self.map = _map

        # Determine sensor type and observation space dimension
        self._is_lidar = isinstance(measurement_model, LIDAR)
        self._is_range_bearing = isinstance(measurement_model, RangeBearingSensor)

        if self._is_lidar:
            # LIDAR: observation space dimension is B (number of beams)
            obs_dim = measurement_model.B
        else:
            # RangeBearingSensor: observation dimension depends on map at use time
            # cov_y is not set here since it is not used with this sensor
            obs_dim = None

        if isinstance(motion_model, SingleIntegratorModel):
            self.σ_w = sigma_w * np.sqrt(motion_model.dt)
            self.σ_v = sigma_v
            self.cov_x = np.eye(2) * sigma_w * sigma_w * motion_model.dt
            # Only set cov_y if using LIDAR (fixed dimension)
            self.cov_y = np.eye(obs_dim) * sigma_v * sigma_v if self._is_lidar else None
        elif isinstance(motion_model, DoubleIntegratorModel):
            self.σ_w = sigma_w
            self.σ_v = sigma_v
            self.cov_x = motion_model.Q_t * sigma_w
            # Only set cov_y if using LIDAR (fixed dimension)
            self.cov_y = np.eye(obs_dim) * sigma_v * sigma_v if self._is_lidar else None

        self._map_cache = {}
        self._integration_cache = {}
        self.force_constant = force_constant

    def get_observation_dimension(self, map_representation=None):
        """
        Get observation space dimension for the current sensor and map.

        Args:
            map_representation: Optional map representation (for RangeBearingSensor with LandmarkMap)

        Returns:
            int: Observation space dimension
        """
        if self._is_lidar:
            return self.sensor.B
        elif self._is_range_bearing:
            from .mapping import LandmarkMap
            if isinstance(self.map, LandmarkMap):
                if map_representation is not None:
                    # Return dimension based on number of candidate landmarks
                    # Each landmark has 2 observations (range, bearing)
                    return 2 * self.map.num_candidate_landmarks
                else:
                    # Default: return dimension for all candidate landmarks
                    return 2 * self.map.num_candidate_landmarks
            else:
                raise ValueError("RangeBearingSensor requires LandmarkMap")
        else:
            raise ValueError(f"Unsupported sensor type: {type(self.sensor)}")

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

        X_cpu = X.get() if hasattr(X, 'get') else numpy_cpu.asarray(X)
        B_cpu = B.get() if hasattr(B, 'get') else numpy_cpu.asarray(B)
        u_cpu = u.get() if hasattr(u, 'get') else numpy_cpu.asarray(u)
        cov_x_raw = self.cov_x
        cov_cpu = cov_x_raw.get() if hasattr(cov_x_raw, 'get') else numpy_cpu.asarray(cov_x_raw)

        # f_bar uses backend np (could be CuPy), so convert inputs to backend arrays
        # then convert result back to NumPy for mvn.cdf
        if is_cupy:
            X_backend = np.asarray(X_cpu)
            u_backend = np.asarray(u_cpu)
            mu_raw = self.motion_model.f_bar(X_backend, u_backend)
            mu = mu_raw.get() if hasattr(mu_raw, 'get') else numpy_cpu.asarray(mu_raw)
        else:
            mu_raw = self.motion_model.f_bar(X_cpu, u_cpu)
            mu = mu_raw.get() if hasattr(mu_raw, 'get') else numpy_cpu.asarray(mu_raw)
        mins = B_cpu[:, 0]
        maxs = B_cpu[:, 1]

        probs = numpy_cpu.zeros(len(X_cpu))
        for i, mu_i in enumerate(mu):
            probs[i] = mvn.cdf(x=maxs, mean=mu_i, cov=cov_cpu, lower_limit=mins)

        return np.asarray(probs)

    ###########################################################
    ### 2. Cost Functions #####################################
    def c_collision(self, x, u, ε=1e-6):
        """
        Cost function for collision avoidance in localization.
        """
        pass

    def c(self, X_batch: np.ndarray, U_batch: np.ndarray) -> np.ndarray:
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
    ### 4. Ray Casting #######################################
    def ray_casting(self, X):
        """
        Ray casting for localization - uses pre-computed obstacle segments from known map.

        Supports both LIDAR and RangeBearingSensor. For RangeBearingSensor, just calls sensor.g_bar().

        Args:
            X: Array of robot states
               - For LIDAR: (m_n, 2) positions
               - For RangeBearingSensor: (m_n, state_dim) where state_dim can be 2, 3, or 4

        Returns:
            y_star: Array of observations
               - For LIDAR: (m_n, B) distances
               - For RangeBearingSensor: (m_n, num_landmarks, 2) [range, bearing]
        """
        if self._is_lidar:
            if self.known_map_obstacle_segments is None:
                raise ValueError("Known map obstacle segments not set. Call set_known_map() first.")
            return self.sensor.g_bar_localization(X, self.known_map_obstacle_segments)
        elif self._is_range_bearing:
            # For RangeBearingSensor, just call g_bar directly with landmark positions
            # Sensor handles state dimension internally (2D/3D/4D -> appropriate bearing computation)
            from .mapping import LandmarkMap

            # Get landmark positions
            if isinstance(self.map, LandmarkMap) and hasattr(self, 'known_map_representation'):
                landmark_positions = self.map.get_landmark_positions(self.known_map_representation)
            elif self.known_map_obstacle_segments is not None:
                # Fallback: convert obstacle segments to landmark positions (centroids)
                landmark_positions = []
                for seg in self.known_map_obstacle_segments:
                    if isinstance(seg, tuple) and len(seg) == 4:
                        x1, y1, x2, y2 = seg
                        landmark_positions.append([(x1 + x2) / 2, (y1 + y2) / 2])
                landmark_positions = np.array(landmark_positions) if landmark_positions else np.empty((0, 2))
            else:
                raise ValueError(
                    "For RangeBearingSensor, either set known_map_representation or known_map_obstacle_segments")

            # Call sensor.g_bar directly - it handles state dimension and bearing computation
            return self.sensor.g_bar(X, landmark_positions)
        else:
            raise ValueError(f"Unsupported sensor type: {type(self.sensor)}")

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

    def c(self, M_batch: np.ndarray, U_batch: np.ndarray, X_known: np.ndarray = None) -> np.ndarray:
        """
        Batched cost computation: c(m, u) for all (m, u) pairs.

        Args:
            M_batch: maps (n_m, H, W) or (H, W)
            U_batch: actions (n_u, 2) or (2,)
            X_known: known pose (2,) or (state_dim,)
        Returns:
            c_batch: (n_m, n_u) where c_batch[i, j] = c(M_batch[i], U_batch[j])
        """

        if U_batch.ndim == 1:
            U_batch = U_batch[np.newaxis, :]

        # Effort cost: ||u||_2
        effort = np.linalg.norm(U_batch, axis=1)[np.newaxis, :]

        return effort

    ##########################################################
    ### 2. Ray Casting #######################################
    def ray_casting_batched(self, X, M):
        """
        Batched ray casting for multiple maps simultaneously.

        Supports both LIDAR (with occupancy grids) and RangeBearingSensor (with landmark maps).
        Both sensors handle batching internally - just call g_bar().

        Args:
            X: robot states
               - For LIDAR: (m_n, 2) positions
               - For RangeBearingSensor: (m_n, state_dim) where state_dim can be:
                 * 2: (p_x, p_y) - bearings relative to x-axis
                 * 3: (p_x, p_y, θ) - bearings relative to robot heading
                 * 4: (p_x, p_y, v_x, v_y) - bearings relative to x-axis
            M: maps - either:
               - occupancy grids (len_M, H, W) for LIDAR
               - map representations (len_M, num_candidate_landmarks) for LandmarkMap
        Returns:
            y_star_all: observations
               - For LIDAR: (m_n, len_M, B) distances
               - For RangeBearingSensor: (m_n, len_M, num_candidate_landmarks, 2) [range, bearing]
        """
        if isinstance(self.sensor, RangeBearingSensor):
            # For ordered landmark maps, M is already (len_M, l, 2)
            if isinstance(self.map, OrderedLandmarkMap):
                return self.sensor.g_bar(X, M)

            # For subset-encoded LandmarkMap, convert binary vectors to positions
            from .mapping import LandmarkMap
            if isinstance(self.map, LandmarkMap):
                all_candidate_positions = self.map.landmark_positions  # (num_candidate_landmarks, 2)
                num_candidates = all_candidate_positions.shape[0]
                landmark_positions_list = []
                for m_repr in M:
                    active_mask = m_repr.flatten() > 0
                    landmark_pos = np.full((num_candidates, 2), np.nan)
                    landmark_pos[active_mask] = all_candidate_positions[active_mask]
                    landmark_positions_list.append(landmark_pos)

                landmark_positions = np.stack(landmark_positions_list, axis=0)
                return self.sensor.g_bar(X, landmark_positions)

            raise ValueError("RangeBearingSensor requires LandmarkMap or OrderedLandmarkMap")
        else:
            # For LIDAR, call g_bar with map object
            return self.sensor.g_bar(X, M, self.map)


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
        # Get map dimensions - works for both occupancy grids and landmark maps
        map_shape = self.map.map_shape
        if len(map_shape) == 2:
            H, W = map_shape
        else:
            # For landmark maps, we still need H and W for bit iteration
            # Use a default or raise error - this method is occupancy-grid specific
            if hasattr(self.map, 'occupancy_map'):
                H = self.map.occupancy_map.height
                W = self.map.occupancy_map.width
            else:
                raise ValueError("generate_space_of_maps with bit_iteration requires occupancy grid maps")
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
                    map_array = self.map.id_to_map(map_bits, (H, W))
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
                    map_array = self.map.id_to_map(map_bits, (H, W))
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
            maps_3d: Array of shape (num_maps, ...) containing all possible maps
            map_bits: Array of shape (num_maps,) containing the bit representation for each map
        """
        # Use the map's generate_all_maps method
        maps_array, map_bits_array = self.map.generate_all_maps(show_progress=show_progress)
        return maps_array, map_bits_array

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
        # Get map dimensions - works for both occupancy grids and landmark maps
        map_shape = self.map.map_shape
        if len(map_shape) == 2:
            H, W = map_shape
        else:
            # For landmark maps, we still need H and W for bit iteration
            # Use a default or raise error - this method is occupancy-grid specific
            if hasattr(self.map, 'occupancy_map'):
                H = self.map.occupancy_map.height
                W = self.map.occupancy_map.width
            else:
                raise ValueError("generate_space_of_maps with bit_iteration requires occupancy grid maps")
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
    - Ray casting methods from Mapping_POMDP (supports both LIDAR and RangeBearingSensor)

    Method Resolution Order (MRO): SLAM_POMDP -> Localization_POMDP -> Mapping_POMDP -> BasePOMDP
    - Methods defined in SLAM_POMDP override parent methods
    - For ray_casting_batched, we explicitly use Mapping_POMDP's version (inherited)
    - For Q_vectorized_batch, we override both parents with SLAM-specific version

    State space: Joint (x, m) where:
    - x: Robot pose (2D or 4D)
    - m: Map (occupancy grid or landmark map)

    Belief space: π(x, m) - joint belief over pose and map.

    Supports multiple sensor and map combinations:
    - LIDAR sensor with LidarGridMapVec (occupancy grids)
    - RangeBearingSensor with LandmarkMap (landmark-based maps)

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
    def c_effort(self, U_batch: np.ndarray) -> np.ndarray:
        """
        Batched cost computation: c_effort(u) for all (u) pairs.

        Args:
            U_batch: actions (n_u, 2) or (2,)
        Returns:
            c_effort_batch: (n_u,) where c_effort_batch[i] = c_effort(U_batch[i])
        """
        return np.linalg.norm(U_batch, axis=1)
    ##########################################################


# Backward compatibility: Maintain POMDP as alias for SLAM_POMDP
# Old code using POMDP will still work
POMDP = SLAM_POMDP

__all__ = ['POMDP', 'SLAM_POMDP', 'Localization_POMDP', 'Mapping_POMDP']
