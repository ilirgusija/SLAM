import numpy as np
# import cupy as np
from scipy import ndimage
from .model import VelocityIntegratorModel, LIDAR
from .obstacle import Obstacle
from .mapping import LidarGridMapVec
from utils.misc import cartesian, cartesian_dot
from utils.map import bresenham_vec


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
                 sigma_v: float = 0.01):
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

    ### 1. Stochastic Kernels #################################
    def T(self, x, u):
        """
        Transition probability for robot position x_1 given x and u.
        Returns a function that computes T(x_1 | x, u) for given x_1.
        """
        def inner(x_1):
            return 1/(np.sqrt(2 * np.pi) * self.σ_w) * \
                np.exp(-np.square(x_1 - self.motion_model.simulate(x, u)) /
                       (2 * self.σ_w ** 2))
        return inner

    def T(self, x_star, x, u):
        """
        Transition probability for robot position x_1 given x and u.
        Returns a function that computes T(x_1 | x, u) for given x_1.
        """
        return 1/(np.sqrt(2 * np.pi) * self.σ_w) * \
            np.exp(-np.square(x_star - self.motion_model.simulate(x, u)) /
                   (2 * self.σ_w ** 2))

    def T_cartesian(self, X_n, u):  # ✅
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

        # Compute transition probabilities
        T_matrix = 1/(np.sqrt(2 * np.pi) * self.σ_w) * \
            np.exp(-squared_diff / (2 * np.square(self.σ_w)))

        # Normalize each row
        row_sums = T_matrix.sum(axis=1, keepdims=True)
        T_matrix = np.where(row_sums > 0, T_matrix / row_sums, T_matrix)

        return T_matrix  # TODO: check if conditional is indexed by row or column

    def Q(self, y, X_n, m):  # ✅
        """
        Observation channel for robot observation y given position x, and map m.
        """
        m_n = len(X_n)
        obs = np.array((m_n, self.sensor.B))

        obs = 1/(np.sqrt(2 * np.pi) * self.σ_v) * \
            np.exp(-np.square(y - self.ray_casting(X_n, m)) /
                   (2 * self.σ_v ** 2))
        return obs

    def Q_vectorized(self, Y, X_n, m):  # TODO
        """
        Vectorized observation probability matrix Q(Y | X_n, m).
        Computes observation probabilities for all state-observation pairs in parallel.
        """
        m_n = len(X_n)
        y_len = len(Y)

        y_star = self.ray_casting(X_n, m)
        y_star_expanded = y_star[:, np.newaxis, :]  # (m_n, 1, 1)
        Y_expanded = Y[np.newaxis, :, :]  # (1, r_len, B)
        diff = Y_expanded - y_star_expanded  # (1, y_len, m_n, H*W)
        assert diff.shape == (1, y_len, m_n, m_h_w)

        Q_matrix = (1/(np.sqrt(2 * np.pi) * self.σ_v) *
                    np.exp(-np.sum(np.square(diff), axis=-1) / (2 * np.square(self.σ_v)))).squeeze(0)
        assert Q_matrix.shape == (y_len, m_n, m_h_w)

        return Q_matrix

    ###########################################################
    ### 2. Cost Functions #####################################
    def c(self, x, m, u):
        """
        Cost function for the POMDP.
        """
        return self.c_effort(u) + self.c_collision(x, m, u)

    @staticmethod
    def c_effort(u):  # ✅
        """
        Cost function for the effort of the control action u.
        """
        return np.linalg.norm(u, ord=2, axis=1)

    def F_r(self, x, m):  # ✅
        F_cr = 100  # force constant
        max_D = self.motion_model.dt * self.motion_model.max_v  # (meters)

        # Convert occupancy grid cells to their center positions (N, 2)
        m_pos = self.map.occupancy_map.get_occupied_map_positions(
            m)  # (H*W, 2)

        # Vector from each cell center to x and corresponding distances (N, 2), (N,)
        diff = x - m_pos
        d = np.linalg.norm(diff, ord=2, axis=1)

        # Mask to keep only cells within max range
        within = d <= max_D
        if not np.any(within):
            return np.zeros(2)

        diff = diff[within]
        d = d[within]

        # Avoid divide-by-zero at x coinciding with a cell center
        eps = 1e-9
        inv_d3 = 1.0 / np.maximum(d, eps)**3  # (N,)

        # Force contribution: F_cr * (x - m_pos) / d^3  -> (N,2)
        forces = (F_cr * diff) * inv_d3[:, None]

        # Sum vector force over contributing cells -> (2,)
        return forces.sum(axis=0)

    def c_vff(self, x, m, u):  # 🚧
        """
        Cost function for the collision of the robot with the obstacle.
        """
        ε = 1e-6
        F_r = self.F_r(x, m)  # (N_n,2)
        fraction = np.sum(F_r * u, axis=1) \
            / (np.linalg.norm(F_r, ord=2, axis=1) + ε)
        return max(0, fraction)

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

        # Find connected components of occupied cells
        connected_components = self._find_connected_components(m)

        all_obstacle_segments = []

        # Create obstacles for each connected component
        for component in connected_components:
            if len(component) > 0:
                obstacle_segments = self._create_obstacle_from_component(
                    component)
                all_obstacle_segments.extend(obstacle_segments)

        _, y_star = self.sensor.get_laser_ref(all_obstacle_segments, X)

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
    def _create_obstacle_from_component(self, component):  # 🚧
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

        # Calculate centroid
        if num_cells_i % 2 == 1 and num_cells_j % 2 == 1:
            # Odd number of cells in both directions - use center cell
            center_i = min_i + num_cells_i // 2
            center_j = min_j + num_cells_j // 2
            centroid = self.map.occupancy_map.get_position_from_map_index(
                center_j, center_i)
        else:
            # Even number of cells - use average of center cells
            center_i = (min_i + max_i) / 2.0
            center_j = (min_j + max_j) / 2.0
            centroid = self.map.occupancy_map.get_position_from_map_index(
                center_j, center_i)

        # Create obstacle and get its line segments
        obstacle = Obstacle(centroid, dx=dx, dy=dy, angle=0)
        return obstacle._Obstacle__get_points(centroid)

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
