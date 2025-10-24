import numpy as np
# import cupy as np
from .mapping import LidarGridMapVec
from .model import LIDAR, VelocityIntegratorModel
from .obstacle import Obstacle
from .quantizer import StateQuantizer, ActionQuantizer, ObservationQuantizer, BeliefQuantizer
from .pomdp import POMDP
import scipy.integrate as integrate
from tqdm import tqdm


class BeliefMDP_n(POMDP):
    """
    Belief-MDP class that handles belief state transitions and updates.
    Extends POMDP with belief-specific functionality.
    Belief-MDP is defined as a four tuple (Π_n, U_n, η, c_tilde)
    """

    def __init__(
        self,
        n: int,
        motion_model: VelocityIntegratorModel,
        measurement_model: LIDAR,
        obstacles: list[Obstacle],
        _map: LidarGridMapVec,
        sigma_w: float = 0.01,
        sigma_v: float = 0.01
    ):

        super().__init__(motion_model, measurement_model, obstacles, _map, sigma_w, sigma_v)
        self.n = n


        self.SQ = StateQuantizer(
            self.map.occupancy_map.left_lower[0],
            self.map.occupancy_map.right_upper[0],
            self.map.occupancy_map.left_lower[1],
            self.map.occupancy_map.right_upper[1],
            n
        )
        self.AQ = ActionQuantizer(self.motion_model.max_v, n)

        self.H, self.W = self.map.occupancy_map.height, self.map.occupancy_map.width
        self.len_M = 2**(self.H * self.W)

        self.T_mat = np.zeros((self.SQ.m_n, self.SQ.m_n, self.AQ.n_u))
        for i in tqdm(range(self.AQ.n_u), desc="Computing T_mat"):
            self.T_mat[i, :, :] = self.T_n(self.AQ.U[i])


### 1. Filter Update Equation ######################


    def F(self, π, u, y):
        """
        Filter update equation for belief state given single observation y.
        Returns updated belief π' over (x, m) given observation y.

        π'(x,m) ∝ Q(y|x,m) * Σ_{x'} T_n(x|x',u) * π(x',m)
        """
        u_idx = self.AQ.get_quantized_index(u)  # TODO: Implement this
        Tn_mat = self.T_mat[u_idx]
        assert Tn_mat.shape == (self.SQ.m_n, self.SQ.m_n)  # x_{t+1} x x_{t}

        # Predicted belief: Σ_{x'} T_n(x|x',u) * π(x',m)
        integral = Tn_mat @ π  # (m_n, m_n) @ (m_n, 2^HW) = (m_n, 2^HW)

        # Compute observation likelihoods for all (x,m) pairs using callback
        numerator = np.zeros_like(integral)

        def process_map_f(map_bits, m_2d):
            if map_bits <= self.len_M:
                # Q(y|x,m) for all x given this map m
                likelihoods = self.Q(y, self.SQ.X_n, m_2d)  # (m_n,)

                # π'(x,m) ∝ Q(y|x,m) * predicted_belief(x,m)
                numerator[:, map_bits] = likelihoods * integral[:, map_bits]

        # Use callback approach to get 2D maps directly
        self.generate_space_of_maps(callback=process_map_f)

        # Normalize to get probability distribution
        total = np.sum(numerator)
        if total > 0:
            π_new = numerator / total
        else:
            raise ValueError("Total probability is 0")

        return π_new

    def H(self, y, π, u):
        """
        Observation probability: H(y, π, u) = P(y | π, u)

        H(y|π,u) = Σ_{x∈X_n} Σ_{m∈M} Q(y|x,m) * [Σ_{x'} T_n(x|x',u) * π(x',m)]
        """
        u_idx = self.AQ.get_quantized_index(u)  # TODO: Implement this
        Tn_mat = self.T_mat[u_idx]

        # Predicted belief: Σ_{x'} T_n(x|x',u) * π(x',m)
        integral = Tn_mat @ π  # (m_n, m_n) @ (m_n, 2^HW) = (m_n, 2^HW)

        # Compute H(y|π,u) = Σ_{x,m} Q(y|x,m) * predicted_belief(x,m) using callback
        total_prob = 0.0

        def process_map_h(map_bits, m_2d):
            if map_bits < self.len_M:
                # Q(y|x,m) for all x given this map m
                likelihoods = self.Q(y, self.SQ.X_n, m_2d)  # (m_n,)

                # Add to total: Σ_x Q(y|x,m) * predicted_belief(x,m)
                nonlocal total_prob
                total_prob += np.sum(likelihoods * integral[:, map_bits])

        # Use callback approach to get 2D maps directly
        self.generate_space_of_maps(callback=process_map_h)

        assert total_prob >= 0, "negative probability"
        return float(total_prob)

    def T_n(self, u: np.ndarray) -> np.ndarray:
        """
        Quantized transition kernel T_n over X_n for given control u.

        T_n(x_j^n | x_i^n, u) = T(B_j^n | x_i^n, u)
        where B_j^n is the Voronoi cell around x_j^n.

        Args:
            u: action (2,)
        Returns:
            (m_n, m_n) transition matrix: T_n[j,i] = T_n(x_j^n | x_i^n, u)
        """
        # Get precomputed bounds from quantizer
        bounds = self.SQ.get_bounds()  # (m_n, 2, 2)

        Tn = np.zeros((self.SQ.m_n, self.SQ.m_n))

        for j in range(self.SQ.m_n):
            # T_n(x_j^n | x_i^n, u) = T(B_j^n | x_i^n, u) for all x_i^n
            Tn[j, :] = self.T(bounds[j], self.SQ.X_n, u)

        # Normalize columns to ensure probability measure
        col_sums = Tn.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums == 0.0, 1.0, col_sums)
        Tn = Tn / col_sums

        return Tn

    def c_n(self, x: np.ndarray, m: np.ndarray, u: np.ndarray) -> float:
        """
        Quantized one-stage cost c_n(x_i^n, m, u) = c(x_i^n, m, u).

        With Dirac weighting at centroids, this simplifies to evaluating
        the continuous cost at the quantized state.

        Args:
            x: single quantized state (2,)
            m: map (H, W)  
            u: action (2,)
        Returns:
            Cost value
        """
        return float(self.c(x, m, u))

###################################################
### 2. Cost Function ##############################

    def c_tilde_n(self, π, u):  # ✅
        """
        Expected cost function for belief-MDP: c_tilde(π, u) = E[c_n(x,m,u) | π]

        c_tilde(π, u) = Σ_{x∈X_n} Σ_{m∈M} c_n(x, m, u) * π(x,m)
        """
        assert π.shape == (self.SQ.m_n, self.len_M)

        # Compute expected cost using c_n for each (x,m) pair
        expected_cost = 0.0
        for i, x in enumerate(self.SQ.X_n):
            for j in range(self.len_M):
                # Cost for this (x,m) pair using quantized cost
                cost = self.c_n(x, self.bits_to_map(j, self.H, self.W), u)

                # Weight by belief
                expected_cost += cost * π[i, j]

        return float(expected_cost)

###################################################
### 3. Robust Numerical Methods ###################

    def _robust_normalize(self, probabilities):
        """
        Robust normalization using LogSumExp to prevent underflow.

        For a probability distribution p, compute p / sum(p) using:
        log(p_normalized) = log(p) - logsumexp(log(p))

        Args:
            probabilities: Array of non-negative values to normalize

        Returns:
            Normalized probabilities that sum to 1
        """
        # Handle edge cases
        if np.all(probabilities == 0):
            # If all probabilities are zero, return uniform distribution
            return np.ones_like(probabilities) / probabilities.size

        # Convert to log space for numerical stability
        # Add small epsilon to avoid log(0)
        epsilon = 1e-300
        log_probs = np.log(probabilities + epsilon)

        # Compute logsumexp for normalization
        log_sum = self._logsumexp(log_probs)

        # Normalize in log space and convert back
        log_normalized = log_probs - log_sum
        normalized = np.exp(log_normalized)

        # Ensure sum is exactly 1 (within numerical precision)
        normalized = normalized / np.sum(normalized)

        return normalized

    def _robust_sum(self, probabilities):
        """
        Robust summation using LogSumExp to prevent underflow.

        Args:
            probabilities: Array of non-negative values to sum

        Returns:
            Sum of all probabilities
        """
        if np.all(probabilities == 0):
            return 0.0

        # Convert to log space
        epsilon = 1e-300
        log_probs = np.log(probabilities + epsilon)

        # Use logsumexp for robust summation
        log_sum = self._logsumexp(log_probs)

        return np.exp(log_sum)

    def _logsumexp(self, log_probs):
        """
        Compute log(sum(exp(log_probs))) using the log-sum-exp trick.

        This prevents overflow/underflow when computing sum of exponentials
        of large numbers by subtracting the maximum value.

        Args:
            log_probs: Array of log probabilities

        Returns:
            log(sum(exp(log_probs)))
        """
        # Find the maximum log probability
        max_log_prob = np.max(log_probs)

        # Compute log(sum(exp(log_probs - max_log_prob))) + max_log_prob
        # This is mathematically equivalent to log(sum(exp(log_probs)))
        # but numerically stable
        log_sum = max_log_prob + np.log(np.sum(np.exp(log_probs - max_log_prob)))

        return log_sum

###################################################
