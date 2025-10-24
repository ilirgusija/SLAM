import numpy as np
# import cupy as np
from scipy import ndimage
from scipy.stats import multivariate_normal as mvn
# from scipy.stats.mvn import mvnun  # Not available in current scipy
from .model import VelocityIntegratorModel, LIDAR
from .obstacle import Obstacle
from .mapping import LidarGridMapVec
from ..utils.misc import cartesian, cartesian_dot
from scipy.spatial.distance import cdist
import openturns as ot

def computeRectangularDomainProbability(lower, upper, means, cov_matrix):
    """
    Compute the probability of a rectangular solid
    under a multinormal distribution.

    """
    # Convert to numpy arrays and ensure proper shape
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    means = np.asarray(means)

    # Center the bounds of the rectangular solid on the mean
    lower = lower - means
    upper = upper - means

    # The same covariance matrix for all rectangular solids.
    cov_matrix = ot.CovarianceMatrix(cov_matrix)

    # This way, we only need to define one multivariate normal distribution.
    # That is the trick that allows vectorization.
    dimension = len(lower)  # For 1D case
    multinormal = ot.Normal([0.0] * dimension, cov_matrix)

    # The probability of the rectangular solid is a weighted sum
    # of the CDF of the vertices (with weights equal to 1 or -1).
    # The following block computes the CDFs and applies the correct weight.
    full_reverse_binary = np.array(list(bin(2**dimension)[:1:-1]), dtype=int)
    prob = 0.0
    for i in range(2**dimension):
        reverse_binary = np.array(list(bin(i)[:1:-1]), dtype=int)
        reverse_binary = np.append(reverse_binary,
                                   np.zeros(len(full_reverse_binary) -
                                            len(reverse_binary) -
                                            1)).astype(int)
        point = np.zeros(dimension)
        for num, digit in enumerate(reverse_binary):
            if digit:
                point[num] = upper[num]
            else:
                point[num] = lower[num]
        cdf = multinormal.computeCDF(point.tolist())
        if (reverse_binary.sum() % 2) == (dimension % 2):
            prob += cdf
        else:
            prob -= cdf

    return prob

class POMDP:
    """
    Base POMDP class handling core POMDP functionality:
    - State transitions
    - Observation models
    - Basic belief updates
    POMDP is defined as a six tuple (X, U, Y, S, Q, c)
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
        self.σ_w = sigma_w * np.sqrt(motion_model.dt)  # process noise
        self.σ_v = sigma_v  # observation noise

        self.cov_x = np.eye(2) * sigma_w * sigma_w * motion_model.dt
        self.cov_y = np.eye(measurement_model.B) * sigma_v * sigma_v

        # Caching for map space integration (kept for potential future use)
        self._map_cache = {}
        self._integration_cache = {}
        self.force_constant = force_constant

    ### 1. Stochastic Kernels #################################
    def T(self, B: np.ndarray, X: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Transition probabilities T(B | X, u) for Borel set B given states X and action u.

        T(B | x, u) = ∫_B N(x'; f_bar(x,u), Σ_w) dx'
        where f_bar(x,u) is deterministic next state and Σ_w = σ_w² * I

        Args:
            B: Borel set represented as bounds [[x_min, x_max], [y_min, y_max]]
            X: current states (n, 2)
            u: action (2,)
        Returns:
            Probability masses over B for each state in X (n,)
        """
        # Deterministic next state means for all states
        mu = self.motion_model.f_bar(X, u)  # (n, 2)

        # Covariance matrix
        cov = self.cov_x  # (2, 2)

        # Bounds for rectangular region B
        x_min, x_max = B[0, 0], B[0, 1]
        y_min, y_max = B[1, 0], B[1, 1]

        # Compute probability masses for all states efficiently
        probs = np.zeros(len(X))

        mins = np.array([x_min, y_min])
        maxs = np.array([x_max, y_max])

        for i, mu_i in enumerate(mu):
            # Evaluate CDF at all corners at once
            prob = mvn.cdf(x=maxs, mean=mu_i, cov=cov, lower_limit=mins)
            probs[i] = prob

        return probs

    def T_ot(self, B: np.ndarray, X: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Transition probabilities T(B | X, u) for Borel set B given states X and action u.

        T(B | x, u) = ∫_B N(x'; f_bar(x,u), Σ_w) dx'
        where f_bar(x,u) is deterministic next state and Σ_w = σ_w² * I

        Args:
            B: Borel set represented as bounds [[x_min, x_max], [y_min, y_max]]
            X: current states (n, 2)
            u: action (2,)
        Returns:
            Probability masses over B for each state in X (n,)
        """
        # Deterministic next state means for all states
        mu = self.motion_model.f_bar(X, u)  # (n, 2)

        # Covariance matrix
        cov = self.cov_x  # (2, 2)

        # Bounds for rectangular region B
        x_min, x_max = B[0, 0], B[0, 1]
        y_min, y_max = B[1, 0], B[1, 1]

        # Compute probability masses for all states efficiently
        probs = np.zeros(len(X))

        mins = np.array([x_min, y_min])
        maxs = np.array([x_max, y_max])

        for i, mu_i in enumerate(mu):
            # Evaluate CDF at all corners at once
            prob = computeRectangularDomainProbability(lower=mins, upper=maxs, means=mu_i, cov_matrix=cov)
            probs[i] = prob

        return probs

    def Q(self, y: np.ndarray, X: np.ndarray, m: np.ndarray) -> np.ndarray:
        """
        Observation likelihoods Q(y | X, m) for observation y given states X and map m.

        Q(y | x, m) = N(y; g_bar(x,m), Σ_v)
        where g_bar(x,m) is ideal observation and Σ_v = σ_v² * I

        Args:
            y: observation (B,)
            X: states (n, 2)
            m: occupancy grid (H, W)
        Returns:
            Likelihood array (n,) - one value per state in X
        """
        # Ideal observations for all states
        y_star = self.ray_casting(X, m)  # (n, B)

        # Covariance matrix
        cov = self.cov_y  # (B, B)

        # Compute likelihoods for all states
        likelihoods = np.zeros(len(X))
        for i, y_star_i in enumerate(y_star):
            # Create multivariate normal distribution for this state
            # Evaluate density at y
            likelihoods[i] = mvn.pdf(x=y, mean=y_star_i, cov=cov)

        return likelihoods

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
        return np.linalg.norm(u, ord=2)

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
            return np.zeros(2, dtype=np.float32)

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
        fraction = np.divide(
            np.dot(F_r, u),
            (np.linalg.norm(F_r, ord=2) * np.linalg.norm(u, ord=2) + ε)
        )
        return np.maximum(0, fraction)

    ##########################################################
    ### 3. Stochatic Kernel Helper Functions #################
    def ray_casting(self, X, m):  # ✅
        """
        Ray casting to determine distance from robot position x to observed obstacles in map m.
        Delegates to LIDAR.g_bar() method.

        Args:
            X: Array of robot positions (m_n, 2)
            m: Occupancy grid (H, W)

        Returns:
            y_star: Array of distances (m_n, B)
        """
        return self.sensor.g_bar(X, m, self.map)


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
