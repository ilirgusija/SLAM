# Use CuPy backend for GPU acceleration (falls back to NumPy if not available)
# Note: T matrix computation stays on CPU (sunk cost), GPU acceleration focused on H, F, η_n
from ..utils.array_backend import np, is_cupy
from .model import SingleIntegratorModel, DoubleIntegratorModel, LIDAR, RangeBearingSensor
from .obstacle import Obstacle
from .mapping import BaseMap, LidarGridMapVec, LandmarkMap, OrderedLandmarkMap
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm as univariate_norm
import numpy as numpy_cpu
import os
# Use CuPy's ndimage if available and USE_CUPY is enabled, otherwise scipy's
use_cupy = os.getenv("USE_CUPY", "true").lower() in ("true", "1", "yes")
if use_cupy:
    try:
        from cupyx.scipy import ndimage
    except ImportError:
        from scipy import ndimage
else:
    from scipy import ndimage
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
                 cov_y: np.ndarray | None = None,
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
            # RangeBearingSensor: observation dimension is 2 * num_candidate_landmarks for LandmarkMap
            obs_dim = self.get_observation_dimension() if isinstance(_map, LandmarkMap) else None

        if isinstance(motion_model, SingleIntegratorModel):
            self.σ_w = sigma_w * np.sqrt(motion_model.dt)
            self.σ_v = sigma_v
            self.cov_x = np.eye(2) * sigma_w * sigma_w * motion_model.dt
        elif isinstance(motion_model, DoubleIntegratorModel):
            self.σ_w = sigma_w
            self.σ_v = sigma_v
            self.cov_x = motion_model.Q_t * sigma_w
        # Build observation covariance once and keep it explicit.
        # - LIDAR: default diag(sigma_v^2) unless cov_y is supplied.
        # - RangeBearing: default block-diagonal with per-landmark [sigma_r^2, sigma_phi^2].
        import numpy as _np
        cov_y_np = None
        if cov_y is not None:
            cov_y_np = _np.asarray(cov_y, dtype=float)
        elif self._is_lidar and obs_dim is not None:
            cov_y_np = _np.eye(obs_dim, dtype=float) * (sigma_v ** 2)
        elif self._is_range_bearing and obs_dim is not None:
            sr2 = float(measurement_model.sigma_r) ** 2
            sp2 = float(measurement_model.sigma_phi) ** 2
            diag = _np.empty(obs_dim, dtype=float)
            diag[0::2] = sr2
            diag[1::2] = sp2
            cov_y_np = _np.diag(diag)

        if cov_y_np is not None:
            if cov_y_np.ndim != 2 or cov_y_np.shape[0] != cov_y_np.shape[1]:
                raise ValueError(f"cov_y must be a square matrix, got shape={cov_y_np.shape}")
            if obs_dim is not None and cov_y_np.shape[0] == 2 and self._is_range_bearing:
                # Convenience: allow per-landmark 2x2 covariance and tile for all landmarks.
                n_landmarks = obs_dim // 2
                cov_y_np = _np.kron(_np.eye(n_landmarks, dtype=float), cov_y_np)
            if obs_dim is not None and cov_y_np.shape != (obs_dim, obs_dim):
                raise ValueError(
                    f"cov_y shape mismatch: expected {(obs_dim, obs_dim)}, got {cov_y_np.shape}"
                )
            self.cov_y = np.asarray(cov_y_np)
            diag = _np.diag(cov_y_np)
            self.σ_v = float(_np.sqrt(max(_np.mean(diag), 1e-16)))
        else:
            self.cov_y = None

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

    def ray_casting(self, X, M):
        """
        Unified ray casting method that handles both single map (localization) and batched maps (mapping/SLAM).

        Supports both LIDAR and RangeBearingSensor. Always requires a map representation M to be provided.

        Args:
            X: Array of robot states
               - For LIDAR: (m_n, 2) positions
               - For RangeBearingSensor: (m_n, state_dim) where state_dim can be 2, 3, or 4
            M: Map representation(s) - always required
               - For LIDAR: (H, W) single occupancy grid or (len_M, H, W) batched occupancy grids
               - For RangeBearingSensor: (num_candidate_landmarks,) single map or (len_M, num_candidate_landmarks)  
                 batched, or (num_landmarks, 2) single landmark positions or (len_M, num_landmarks, 2) batched for OrderedLandmarkMap
        Returns:
            y_star: Array of observations
               - For LIDAR with single map: (m_n, B) distances
               - For LIDAR with batched maps: (m_n, len_M, B) distances
               - For RangeBearingSensor with single map: (m_n, num_landmarks, 2) [range, bearing]
               - For RangeBearingSensor with batched maps: (m_n, len_M, num_candidate_landmarks, 2) [range, bearing]
        """
        if isinstance(self.sensor, RangeBearingSensor):
            # For ordered landmark maps, M is already (len_M, l, 2) or (l, 2)
            if isinstance(self.map, OrderedLandmarkMap):
                return self.sensor.g_bar(X, M)

            # For subset-encoded LandmarkMap, handle different M formats
            if isinstance(self.map, LandmarkMap):
                # Check if M is already landmark positions (from fallback conversion)
                # M could be: (num_landmarks, 2) or (len_M, num_landmarks, 2)
                if M.ndim >= 2 and M.shape[-1] == 2:
                    # Already landmark positions, pass directly
                    return self.sensor.g_bar(X, M)

                # Otherwise, M is a binary vector: (num_candidate_landmarks,) or (len_M, num_candidate_landmarks)
                all_candidate_positions = self.map.landmark_positions  # (num_candidate_landmarks, 2)
                num_candidates = all_candidate_positions.shape[0]

                # Handle both single and batched maps
                if M.ndim == 1:
                    # Single map: (num_candidate_landmarks,)
                    active_mask = M.flatten() > 0
                    landmark_pos = np.full((num_candidates, 2), np.nan)
                    landmark_pos[active_mask] = all_candidate_positions[active_mask]
                    return self.sensor.g_bar(X, landmark_pos)
                else:
                    # Batched maps: (len_M, num_candidate_landmarks)
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
            # For LIDAR, preprocess maps into obstacle segments and pass to LIDAR
            if M is None:
                raise ValueError("LIDAR ray_casting requires a map or obstacle segments.")

            # Accept precomputed obstacle segments directly
            if isinstance(M, tuple) and len(M) == 2:
                obstacle_segments, segment_map_indices = M
                return self.sensor.g_bar_from_segments(X, obstacle_segments, segment_map_indices)

            if isinstance(M, list) or isinstance(M, tuple):
                if len(M) == 0 or (len(M) > 0 and isinstance(M[0], (tuple, list)) and len(M[0]) == 4):
                    return self.sensor.g_bar_from_segments(X, M)

            # Otherwise, M is an occupancy grid or batch of grids
            M = np.asarray(M)
            if M.ndim == 2:
                obstacle_segments = self.get_obstacles_from_map(M)
                return self.sensor.g_bar_from_segments(X, obstacle_segments)
            if M.ndim == 3:
                obstacle_segments, segment_map_indices = self.get_obstacles_from_map_batched(M)
                return self.sensor.g_bar_from_segments(X, obstacle_segments, segment_map_indices)

            raise ValueError(f"Unsupported map format for LIDAR ray_casting: shape={getattr(M, 'shape', None)}")

    def get_obstacles_from_map(self, m):
        """
        Convert a single map representation to obstacle segments with caching.
        """
        cache = self._map_cache.setdefault("obstacle_segments", {})
        map_id = None
        if hasattr(self.map, "map_to_id"):
            try:
                map_id = int(self.map.map_to_id(m))
            except Exception:
                map_id = None

        if map_id is not None and map_id in cache:
            return cache[map_id]

        segments = self._compute_obstacle_segments_from_map(m)
        if map_id is not None:
            cache[map_id] = segments
        return segments

    def get_obstacles_from_map_batched(self, M):
        """
        Convert a batch of maps into obstacle segments and map indices.
        """
        all_segments = []
        segment_map_indices = []
        for map_idx, m in enumerate(M):
            segments = self.get_obstacles_from_map(m)
            if segments:
                all_segments.extend(segments)
                segment_map_indices.extend([map_idx] * len(segments))

        if segment_map_indices:
            segment_map_indices = np.array(segment_map_indices, dtype=np.int32)
        else:
            segment_map_indices = np.array([], dtype=np.int32)
        return all_segments, segment_map_indices

    def _compute_obstacle_segments_from_map(self, m):
        """
        Convert map to obstacle segments (LIDAR).
        """
        # Try to use the map's get_obstacle_segments method first
        if hasattr(self.map, "get_obstacle_segments"):
            segments = self.map.get_obstacle_segments(m)
            if segments:
                return segments

        # Fallback for occupancy grids
        if hasattr(self.map, "occupancy_map"):
            connected_components = self._find_connected_components(m)
            all_obstacle_segments = []
            for component in connected_components:
                if len(component) > 0:
                    obstacle_segments = self._create_obstacle_from_component(
                        component, m.shape, self.map
                    )
                    all_obstacle_segments.extend(obstacle_segments)
            return all_obstacle_segments

        return []

    def _find_connected_components(self, m):
        """
        Find connected components of occupied cells (value = 1) in the occupancy grid.
        """
        import numpy as _numpy

        ndimage_module_name = getattr(ndimage, '__name__', '')
        is_cupy_ndimage = 'cupyx' in ndimage_module_name

        if is_cupy_ndimage:
            import cupy as _cupy
            if isinstance(m, _numpy.ndarray):
                m = _cupy.asarray(m)
            elif not hasattr(m, 'device'):
                m = np.array(m)
                if not hasattr(m, 'device'):
                    m = _cupy.asarray(_numpy.asarray(m))
        else:
            try:
                import cupy as _cupy
                if isinstance(m, _cupy.ndarray):
                    m = m.get()
                else:
                    m = _numpy.asarray(m)
            except ImportError:
                m = _numpy.asarray(m)

        result = ndimage.label(m)
        if isinstance(result, tuple):
            labeled, num_features = result
        else:
            labeled = result
            num_features = int(labeled.max())

        components = []
        for label in range(1, num_features + 1):
            component = np.argwhere(labeled == label)
            if hasattr(component, 'get'):
                component = component.get()
            components.append([tuple(coord) for coord in component])

        return components

    def _create_obstacle_from_component(self, component, grid_shape, map_obj: LidarGridMapVec):
        """
        Create obstacle segments from a connected component of cells.
        """
        if not component:
            return []

        H, W = grid_shape
        left_lower = map_obj.occupancy_map.left_lower
        right_upper = map_obj.occupancy_map.right_upper

        cell_w = map_obj.occupancy_map.resolution
        cell_h = map_obj.occupancy_map.resolution

        i_coords = [cell[0] for cell in component]
        j_coords = [cell[1] for cell in component]

        min_i, max_i = min(i_coords), max(i_coords)
        min_j, max_j = min(j_coords), max(j_coords)
        num_cells_i = max_i - min_i + 1
        num_cells_j = max_j - min_j + 1

        x_min = left_lower[0] + min_i * cell_w
        y_min = left_lower[1] + min_j * cell_h
        x_max = x_min + num_cells_i * cell_w
        y_max = y_min + num_cells_j * cell_h

        segments = [
            (x_min, y_min, x_max, y_min),
            (x_max, y_min, x_max, y_max),
            (x_max, y_max, x_min, y_max),
            (x_min, y_max, x_min, y_min),
        ]
        return segments

    def c_effort(self, U: np.ndarray) -> np.ndarray:
        """
        Batched cost computation: c_effort(u) for all (u) pairs.

        Args:
            U_batch: actions (n_u, 2) or (2,)
        Returns:
            c_effort_batch: (n_u,) where c_effort_batch[i] = c_effort(U_batch[i])
        """
        from .costs import control_effort
        return control_effort(U)

    def _transition_gaussian_mass(self, B: np.ndarray, X: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Shared transition kernel computation: T(B | X, u) = ∫_B N(x'; f_bar(x,u), Σ_w) dx'

        Args:
            B: Borel set bounds - shape (2, 2) for 2D or (4, 2) for 4D
            X: current states (n, d)
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
        self.known_map_representation = None

        # Convert obstacles to map representation at initialization
        # Extract segments from obstacles (which are already set by super().__init__)
        if self.obstacles:
            self.set_known_map(self.obstacles)

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
        return self._transition_gaussian_mass(B, X, u)

    ###########################################################
    ### 2. Cost Functions #####################################
    def c_collision(self, x, u, ε=1e-6):
        """
        Cost function for collision avoidance in localization.
        """
        pass

    def c(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """
        Batched cost computation: c(u) for all (u) pairs.

        Args:
            X: states (n_x, 2) or (2,)
            U actions (n_u, 2) or (2,)
        Returns:
            c: (n_u,) where c[i] = c(X[i], U[i])
        """
        effort = self.c_effort(U)
        return effort + self.c_collision(X, U)

    ##########################################################
    ### 4. Ray Casting #######################################
    def ray_casting(self, X):
        """
        Ray casting for localization - uses pre-computed map representation from known map.

        Supports both LIDAR and RangeBearingSensor. Map representation is set at initialization.

        Args:
            X: Array of robot states
               - For LIDAR: (m_n, 2) positions
               - For RangeBearingSensor: (m_n, state_dim) where state_dim can be 2, 3, or 4

        Returns:
            y_star: Array of observations
               - For LIDAR: (m_n, B) distances
               - For RangeBearingSensor: (m_n, num_landmarks, 2) [range, bearing]
        """
        if self.known_map_representation is None:
            raise ValueError("Known map representation not set. This should be set during initialization.")

        # Use the pre-computed map representation
        return super().ray_casting(X, M=self.known_map_representation)

    def set_known_map(self, obstacle_segments):
        """
        Set the known map for localization using obstacle segments.
        Converts obstacle segments to proper map representation for unified ray_casting interface.

        Args:
            obstacle_segments: List of obstacle segments, where each segment is a tuple (x1, y1, x2, y2)
                               Can also be a list of Obstacle objects, which will be converted to segments
        """

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

        # Convert segments to proper map representation
        if self._is_lidar:
            # For LIDAR, keep obstacle segments directly to avoid repeated preprocessing
            self.known_map_representation = self.known_map_obstacle_segments
        elif self._is_range_bearing:
            # For RangeBearingSensor, convert segments to landmark positions
            from .mapping import LandmarkMap
            if isinstance(self.map, LandmarkMap):
                # Convert obstacle segments to landmark positions (centroids)
                # This creates a (num_landmarks, 2) array of landmark positions
                landmark_positions = []
                for seg in self.known_map_obstacle_segments:
                    if isinstance(seg, tuple) and len(seg) == 4:
                        x1, y1, x2, y2 = seg
                        landmark_positions.append([(x1 + x2) / 2, (y1 + y2) / 2])

                if landmark_positions:
                    self.known_map_representation = np.array(landmark_positions)  # (num_landmarks, 2)
                else:
                    self.known_map_representation = np.empty((0, 2))
            else:
                raise ValueError(f"RangeBearingSensor requires LandmarkMap, got {type(self.map)}")


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

    def T(self, B: np.ndarray, X: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Transition probabilities T(B | X, u) for Borel set B given states X and action u.
        Same as Localization_POMDP.T: pose dynamics are identical regardless of mapping vs localization.

        T(B | x, u) = ∫_B N(x'; f_bar(x,u), Σ_w) dx'

        Args:
            B: Borel set bounds - shape (2, 2) for 2D or (4, 2) for 4D
            X: current states (n, d)
            u: action (2,)
        Returns:
            Probability masses over B for each state in X (n,)
        """
        return self._transition_gaussian_mass(B, X, u)

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

    def ray_casting(self, X, M):
        """SLAM ray casting over joint pose-map inputs."""
        return BasePOMDP.ray_casting(self, X, M)


# Backward compatibility: Maintain POMDP as alias for SLAM_POMDP
# Old code using POMDP will still work
POMDP = SLAM_POMDP

__all__ = ['POMDP', 'SLAM_POMDP', 'Localization_POMDP', 'Mapping_POMDP']
