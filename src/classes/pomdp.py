import numpy as np
# import cupy as np
from scipy import ndimage
from .model import VelocityIntegratorModel, LIDAR
from .obstacle import Obstacle
from .mapping import LidarGridMapVec
from ..utils.misc import cartesian, cartesian_dot


class POMDP:
    """
    Base POMDP class handling core POMDP functionality:
    - State transitions
    - Observation models
    - Basic belief updates
    POMDP is defined as a six tuple (X, U, Y, T, Q, c)
    """

    def __init__(self,
                 motion_model: VelocityIntegratorModel,
                 measurement_model: LIDAR,
                 obstacles: list[Obstacle],
                 _map: LidarGridMapVec,
                 sigma_w: float = 0.01,
                 sigma_v: float = 0.01,
                 force_constant: float = 100):
        self.motion_model = motion_model
        self.sensor = measurement_model
        self.obstacles = obstacles
        self.map = _map

        # Probability parameters
        self.σ_w = sigma_w  # process noise
        self.σ_v = sigma_v  # observation noise

        # Caching for map space integration
        self._map_cache = {}  # Cache for computed map values
        self._integration_cache = {}  # Cache for integration results
        self._component_cache = {}  # Cache for connected components
        self.force_constant = force_constant

    ### 1. Stochastic Kernels #################################
    def T_vectorized(self, X_n, u):  # ✅
        """
        Vectorized approach using broadcasting for cartesian product.
        More memory efficient for large state spaces.

        Args:
            X_n: Array of quantized states (m_n, state_dim)
            u: Control action

        Returns:
            T_matrix: Transition probability matrix of shape (m_n, m_n)
        """
        # Compute predicted next states for all current states
        predicted_states = self.motion_model.simulate(
            X_n, u)  # shape: (m_n, state_dim)

        # Reshape for broadcasting: (m_n, 1, state_dim) - (1, m_n, state_dim)
        # (m_n, 1, state_dim)
        predicted_expanded = predicted_states[:, np.newaxis, :]
        X_n_expanded = X_n[np.newaxis, :, :]  # (1, m_n, state_dim)

        # Compute squared differences for all pairs
        # (m_n, m_n, state_dim)
        squared_diff = np.square(X_n_expanded - predicted_expanded)

        # Sum across state dimensions if state is multi-dimensional
        if squared_diff.shape[-1] > 1:
            squared_diff = np.sum(squared_diff, axis=-1)  # (m_n, m_n)
        else:
            squared_diff = squared_diff.squeeze(-1)  # (m_n, m_n)

        # Compute transition probabilities with numerical stabilization
        # Subtract per-row minimum squared distance before exponentiation
        # LogSumExp applied here!
        variance = np.square(self.σ_w)
        min_per_row = np.min(squared_diff, axis=1, keepdims=True)  # (m_n,1)
        stabilized = np.exp(-(squared_diff - min_per_row) / (2 * variance))

        # Normalize each row (common factor cancels out)
        row_sums = stabilized.sum(axis=1, keepdims=True)
        T_matrix = np.divide(stabilized, row_sums, out=np.zeros_like(
            stabilized), where=row_sums > 0)

        return T_matrix

    def Q_vectorized(self, Y, X_n, m):  # ✅ stabilized
        """
        Vectorized observation probability matrix Q(Y | X_n, m).
        Computes observation probabilities for all state-observation pairs in parallel.
        """
        m_n = len(X_n)
        # Y expected shape: (y_len, B)
        if Y.ndim == 1:
            Y = Y[np.newaxis, :]
        y_len = Y.shape[0]

        y_star = self.ray_casting(X_n, m)
        y_star_expanded = y_star[:, np.newaxis, :]  # (m_n, 1, B)
        Y_expanded = Y[np.newaxis, :, :]  # (1, y_len, B)
        diff = Y_expanded - y_star_expanded  # -> (m_n, y_len, B)
        assert diff.shape == (m_n, y_len, self.sensor.B)

        variance = np.square(self.σ_v)
        squared_diff = np.sum(np.square(diff), axis=-1)  # (m_n, y_len)
        # LogSumExp-style stabilization across observations per state (row-wise)
        min_per_row = np.min(squared_diff, axis=1, keepdims=True)  # (m_n,1)
        stabilized = np.exp(-(squared_diff - min_per_row) /
                            (2 * variance))  # (m_n, y_len)

        # Normalize each row safely
        row_sums = stabilized.sum(axis=1, keepdims=True)
        Q_matrix = np.divide(stabilized, row_sums, out=np.zeros_like(
            stabilized), where=row_sums > 0)

        return Q_matrix

    ###########################################################
    ### 2. Cost Functions #####################################
    def c(self, x, m, u):
        """
        Cost function for the POMDP.
        """
        return self.c_effort(u) + self.c_vff(x, m, u)

    @staticmethod
    def c_effort(u):  # ✅
        """
        Cost function for the effort of the control action u.
        """
        return np.linalg.norm(u, ord=2, axis=1)

    def F_r(self, x, m):  # ✅
        F_cr = self.force_constant  # force constant
        # max_D = self.motion_model.dt * self.motion_model.max_v  # (meters)
        max_D = 10.0  # (meters)

        # Convert occupancy grid cells to their center positions (N, 2)
        m_pos = self.map.occupancy_map.get_occupied_map_positions(m)  # (H*W, 2)

        # Vector from each cell center to x and corresponding distances (N, 2), (N,)
        diff = m_pos - x
        d = np.linalg.norm(diff, ord=2, axis=1)

        # Mask to keep only cells within max range
        within = d <= max_D
        if not np.any(within):
            return np.zeros(2)

        diff = diff[within]
        d = d[within]

        # Avoid divide-by-zero at x coinciding with a cell center
        eps = 1e-6
        inv_d3 = 1.0 / np.maximum(d, eps)**3  # (N,)

        # Force contribution: F_cr * (x - m_pos) / d^3  -> (N,2)
        forces = (F_cr * diff) * inv_d3[:, None]

        # Sum vector force over contributing cells -> (2,)
        return forces.sum(axis=0)

    def c_vff(self, x, m, u, ε=1e-6):  # ✅
        """
        Cost function for the collision of the robot with the obstacle.
        """
        F_r = self.F_r(x, m)  # (2, )
        print(f"got F_r: {F_r}")
        print(f"got u: {u}")
        fraction = np.divide(
            np.dot(F_r, u),
            (np.linalg.norm(F_r, ord=2) * np.linalg.norm(u, ord=2) + ε)
        )
        return np.maximum(0, fraction)

    def c_vff_vectorized(self, X, M, U):  # 🚧
        """
        Cost function for the collision of the robot with the obstacle.
        """
        ε = 1e-6
        F_r = np.zeros((len(X) * len(M), 2))
        for i, x in enumerate(X):
            for j, m in enumerate(M):
                F_r[i + j] = self.F_r(x, m)
        F_r_norm = np.linalg.norm(F_r, ord=2, axis=1)
        U_norm = np.linalg.norm(U, ord=2, axis=1)

        # TODO: find way to efficiently compute the cartesian dot product of F_r and u
        F_r_times_U = cartesian_dot(F_r, U)  # (N_n, U_n)

        F_r_times_U_norm = cartesian(F_r_norm, U_norm)  # (N_n,U_n)

        fraction = np.sum(F_r_times_U, axis=1) \
            / (F_r_times_U_norm + ε)
        return max(0, fraction)

##########################################################
### 3. Stochatic Kernel Helper Functions #################
    def ray_casting(self, X, m):  # ✅
        """
        Ray casting to determine distance from robot position x to observed obstacles in map m.
        :param x: (1, 2) array of robot position
        :param m: (H, W) array of map
        :return y_star: (1, B) array of distances to observed obstacles from robot position x
        """

        all_obstacle_segments = self._get_obstacles_from_map(m)

        y_star = self.sensor.get_laser_ref(all_obstacle_segments, X)

        return y_star

    def _find_connected_components(self, m):  # ✅
        """
        Find connected components of occupied cells (value = 1) in the occupancy grid.
        Uses 4-connectivity (cells sharing a side are connected) with scipy.ndimage.label.
        Includes caching for performance optimization.

        :param m: (H, W) array of occupancy grid
        :return: List of connected components, where each component is a list of (i, j) coordinates
        """
        # Create hash for caching
        map_hash = hash(m.tobytes())

        # Check cache first
        if map_hash in self._component_cache:
            return self._component_cache[map_hash]

        # Create working copy for in-place labeling
        working_copy = m.copy()

        # In-place labeling using scipy.ndimage.label
        num_features = ndimage.label(m, output=working_copy)

        # Extract components
        components = []
        for label in range(1, num_features + 1):
            component = np.argwhere(working_copy == label)
            components.append([tuple(coord) for coord in component])

        # Cache the result
        self._component_cache[map_hash] = components

        return components

    # OPTIMIZE vectorize this
    def _create_obstacle_from_component(self, component):  # ✅
        """
        Create an Obstacle object from a connected component of cells.

        :param component: List of (i, j) coordinates representing connected cells
        :return: List of line segments representing the obstacle boundary
        """
        if not component:
            return []

        # Get cell dimensions from map resolution
        cell_size = self.map.occupancy_map.resolution

        # Calculate bounding box
        i_coords = [cell[0] for cell in component]
        j_coords = [cell[1] for cell in component]

        min_i, max_i = min(i_coords), max(i_coords)
        min_j, max_j = min(j_coords), max(j_coords)

        # Calculate dimensions in number of cells
        num_cells_i = max_i - min_i + 1
        num_cells_j = max_j - min_j + 1

        # Calculate physical dimensions
        dx = num_cells_j * cell_size  # width (j direction)
        dy = num_cells_i * cell_size  # height (i direction)

        # Calculate centroid at cell centers (account for 0.5 offset)
        res = self.map.occupancy_map.resolution
        left_lower = self.map.occupancy_map.left_lower
        if num_cells_i % 2 == 1 and num_cells_j % 2 == 1:
            # Odd number of cells in both directions - use center cell index
            center_i = min_i + num_cells_i // 2
            center_j = min_j + num_cells_j // 2
        else:
            # Even number of cells - use average of center cell indices
            center_i = (min_i + max_i) / 2.0
            center_j = (min_j + max_j) / 2.0
        # Map (i,j) -> (x,y) using cell center convention: (j+0.5, i+0.5)
        centroid = left_lower + np.array([center_j + 0.5, center_i + 0.5]) * res

        # Create obstacle and get its line segments
        obstacle = Obstacle(centroid, dx=dx, dy=dy, angle=0)
        return obstacle._Obstacle__get_points(centroid)


    def _get_obstacles_from_map(self, m):
        # Find connected components of occupied cells
        connected_components = self._find_connected_components(m)

        all_obstacle_segments = []

        # Create obstacles for each connected component
        for component in connected_components:
            if len(component) > 0:
                obstacle_segments = self._create_obstacle_from_component(
                    component)
                all_obstacle_segments.extend(obstacle_segments)
        return all_obstacle_segments


    def generate_space_of_maps(self, method='bit_iteration', callback=None):
        """
        Generate maps from the space M = {0,1}^{HW} efficiently.

        Args:
            method: Method to use ('bit_iteration', 'full_space', 'symmetry_reduced')
            callback: Optional callback function to process each map

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
                for map_bits in range(total_maps):
                    map_array = self.bits_to_map(map_bits, (H, W))
                    maps.append(map_array.flatten())
                return np.array(maps)
            else:
                # Process maps via callback (memory efficient)
                for map_bits in range(total_maps):
                    map_array = self.bits_to_map(map_bits, (H, W))
                    callback(map_bits, map_array)
                return None

        elif method == 'symmetry_reduced':
            if callback is None:
                # Return only unique maps up to symmetry
                maps = []
                seen_canonicals = set()

                for map_bits in range(total_maps):
                    canonical = self.get_canonical_map_representative(
                        map_bits, (H, W))
                    if canonical not in seen_canonicals:
                        seen_canonicals.add(canonical)
                        map_array = self.bits_to_map(map_bits, (H, W))
                        maps.append(map_array.flatten())

                return np.array(maps)
            else:
                # Process unique maps via callback
                self.generate_unique_maps(callback)
                return None

        elif method == 'full_space':
            # For small spaces, return everything
            if total_cells <= 16:  # 2^16 = 65536 maps
                return self.generate_space_of_maps(method='bit_iteration', callback=callback)
            else:
                raise ValueError(
                    f"Map space too large ({total_maps} maps). Use callback-based methods.")

        else:
            raise ValueError(f"Unknown method: {method}")
##########################################################
    ### 4. Map Space Utilities (bit encoding + symmetries) ###

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

    def get_canonical_map_representative(self, bits: int, shape: tuple[int, int]) -> int:
        """
        Return the minimal integer encoding among all D4 symmetries of the map decoded from bits.
        """
        m = self.bits_to_map(bits, shape)
        candidate_vals = []
        for t in self._symmetry_transforms(m):
            candidate_vals.append(self.map_to_bits(t))
        return min(candidate_vals)

    def generate_unique_maps(self, callback=None):
        """
        Iterate over maps, yielding each unique map up to D4 symmetry.
        If callback is provided: callback(map_bits, map_array) is invoked for each unique map.
        Returns list of flattened maps if callback is None.
        """
        H = self.map.occupancy_map.height
        W = self.map.occupancy_map.width
        total_cells = H * W
        total_maps = 1 << total_cells

        seen = set()
        if callback is None:
            out = []
            for b in range(total_maps):
                canon = self.get_canonical_map_representative(b, (H, W))
                if canon in seen:
                    continue
                seen.add(canon)
                out.append(self.bits_to_map(b, (H, W)).flatten())
            return np.array(out)
        else:
            for b in range(total_maps):
                canon = self.get_canonical_map_representative(b, (H, W))
                if canon in seen:
                    continue
                seen.add(canon)
                callback(b, self.bits_to_map(b, (H, W)))
            return None

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
