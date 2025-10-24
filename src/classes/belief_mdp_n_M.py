import math
import numpy as np
from scipy.spatial.distance import cdist
# import cupy as np
from .belief_mdp_n import BeliefMDP_n
from .mapping import LidarGridMapVec
from .model import LIDAR, VelocityIntegratorModel
from .obstacle import Obstacle
from .quantizer import BeliefQuantizer


class BeliefMDP_n_M(BeliefMDP_n):
    """
    Belief-MDP_n class that handles finite approximation of belief space.
    Extends BeliefMDP with quantization and finite model approximations.
    Belief-MDP_n is defined as a four tuple (Π_n^(M), U_n, p_n, c_n)
    """

    def __init__(
        self,
        M: int,
        β: float,
        n: int,
        motion_model: VelocityIntegratorModel,
        measurement_model: LIDAR,
        obstacles: list[Obstacle],
        _map: LidarGridMapVec,
        sigma_w: float = 0.01,
        sigma_v: float = 0.01
    ):
        super().__init__(n, motion_model, measurement_model, obstacles, _map, sigma_w, sigma_v)

        self.M = M  # controls size and density of belief space
        self.β = β  # discount factor
        N_n = self.SQ.m_n + self.len_M

        # Initialize belief quantizer
        self.BQ = BeliefQuantizer(M, N_n)

    def p_n_M(self, u):
        """
        Quantized transition probability to all beliefs in Π_n_M given belief π and action u.

        p_n_M(π_j^M | π_i^M, u) = η_n(B_j^M | π_i^M, u)
        where B_j^M is the Voronoi cell around π_j^M.

        Args:
            π: Current belief (N_n,)
            u: Action (2,)
        Returns:
            Transition probabilities to all quantized beliefs (M,)
        """

        # Compute transition probabilities
        cardinality = self.BQ.cardinality  # Actual size of belief space
        p_n_M = np.zeros((cardinality, cardinality))

        for i in range(cardinality):
            for j in range(cardinality):
                # p_n_M(π_j^M | π, u) = η_n(B_j^M | π, u)
                p_n_M[i, j] = self.η_n(j, self.BQ.Π_n_M[i], u)

        return p_n_M

    def p_n_M_parallel(self, actions, n_samples=50000, n_jobs=-1, save_path=None):
        """
        parallel version using joblib
        """
        from joblib import Parallel, delayed

        n_actions = len(actions)

        def compute_row(u_idx, i):
            """compute entire row P[u_idx, i, :]"""
            u = actions[u_idx]
            π_i = self.BQ.Π_n_M[i]
            cardinality = self.BQ.cardinality
            row = np.zeros(cardinality)
            for j in range(cardinality):
                row[j] = self.η_n(j, π_i, u, n_samples,
                                  seed=u_idx * cardinality + i)  # deterministic seed
            return (u_idx, i, row)

        # generate all (u_idx, i) pairs
        cardinality = self.BQ.cardinality
        tasks = [(u_idx, i) for u_idx in range(n_actions)
                 for i in range(cardinality)]

        # parallel computation
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(compute_row)(u_idx, i) for u_idx, i in tasks
        )

        # reconstruct matrix
        cardinality = self.BQ.cardinality
        P = np.zeros((n_actions, cardinality, cardinality))
        for u_idx, i, row in results:
            P[u_idx, i, :] = row

        # normalize
        for u_idx in range(n_actions):
            for i in range(cardinality):
                row_sum = np.sum(P[u_idx, i, :])
                if row_sum > 0:
                    P[u_idx, i, :] /= row_sum

        if save_path:
            np.save(save_path, P)

        return P

    def η_n(self, cell_idx: int, π: np.ndarray, u: np.ndarray, tol=1e-8, n_samples: int = 50000, seed: int = None):  # ✅
        """
        ∫ 𝟙_{F(π,u,y)∈B_{cell_idx}} H(dy|π,u) via monte carlo

        args:
            cell_idx: target voronoi cell index j
            π: current belief
            u: action
            n_samples: number of observation samples
        Returns:
            probability of next belief being in cell with index cell_idx
        """
        if seed is not None:
            np.random.seed(seed)

        count = 0

        for _ in range(n_samples):
            # sample observation y ~ H(·|π,u)
            y = self.sample_observation(π, u)

            # compute next belief F(π,u,y)
            π_new = self.F(π, u, y)

            # check if π_new ∈ B_j
            if self.BQ.is_in_voronoi_cell(π_new, cell_idx):
                count += 1

        return count / n_samples  # probability of next belief being in cell with index cell_idx

    def η_n_optimized(self, cell_idx: int, π: np.ndarray, u: np.ndarray, n_samples: int = 50000, seed: int = None):
        """
        Optimized version of η_n that squashes function calls to reduce overhead.

        This version eliminates the call stack:
        p_n_M → η_n → sample_observation → H → F → T_n → T → Q

        Instead, it inlines the critical path for maximum performance.
        """
        if seed is not None:
            np.random.seed(seed)

        count = 0

        # Precompute frequently used values
        u_idx = self.AQ.get_quantized_index(u)
        Tn_mat = self.T_mat[u_idx]  # (m_n, m_n)

        # Precompute predicted belief: Tn_mat @ π
        integral = Tn_mat @ π  # (m_n, 2^HW)

        # Precompute sensor parameters
        r_max = self.sensor.r_max
        B = self.sensor.B
        n_bins = 20

        for _ in range(n_samples):
            # INLINED: sample_observation_forward (fastest path)
            # Sample (x', m) from predicted belief
            flat_belief = integral.flatten()
            flat_belief /= np.sum(flat_belief)
            idx = np.random.choice(len(flat_belief), p=flat_belief)

            x_idx = idx // self.len_M
            m_idx = idx % self.len_M
            x_prime = self.SQ.X_n[x_idx]
            m_2d = self.bits_to_map(m_idx, self.H, self.W)

            # INLINED: sample_from_sensor_model
            # Sample y from sensor model (deterministic + noise)
            v = np.random.multivariate_normal(np.zeros(B), self.cov_y)
            y = self.sensor.g(x_prime, m_2d, v)  # (B,)

            # INLINED: F function (filter update)
            # Compute observation likelihoods for all (x,m) pairs
            numerator = np.zeros_like(integral)

            # Vectorized Q computation
            y_star = self.ray_casting(self.SQ.X_n, m_2d)  # (m_n, B)
            likelihoods = np.zeros(self.SQ.m_n)
            for i, y_star_i in enumerate(y_star):
                likelihoods[i] = mvn.pdf(x=y, mean=y_star_i, cov=self.cov_y)

            # Apply likelihoods to predicted belief
            numerator[:, m_idx] = likelihoods * integral[:, m_idx]

            # Normalize
            total = np.sum(numerator)
            if total > 0:
                π_new = numerator / total
            else:
                π_new = np.ones_like(numerator) / (self.SQ.m_n * self.len_M)

            # INLINED: is_in_voronoi_cell check
            if self.BQ.is_in_voronoi_cell(π_new, cell_idx):
                count += 1

        return count / n_samples

    def sample_observation(self, π: np.ndarray, u: np.ndarray):
        """
        sample y ~ H(·|π,u)

        for continuous observation space [0, r_max]^B, discretize first
        """
        r_max = self.sensor.r_max
        B = self.sensor.B

        # discretize observation space
        # each sensor reading in [0, r_max], discretize to n_bins values
        n_bins = 20  # adjust based on accuracy/speed tradeoff

        # METHOD 1: sample each sensor reading independently (approximate)
        y = np.zeros(B)
        for b in range(B):
            # discretize [0, r_max] for this sensor
            y_possible = np.linspace(0, r_max, n_bins)

            # compute H(y_b | π, u) for each possible value
            # this is expensive but unavoidable
            probs = np.zeros(n_bins)
            for i, y_val in enumerate(y_possible):
                # construct full observation with this value for sensor b
                y_test = np.full(B, r_max)  # default to max range
                y_test[b] = y_val
                probs[i] = self.H(y_test, π, u)

            # normalize and sample
            if np.sum(probs) > 0:
                probs /= np.sum(probs)
                y[b] = np.random.choice(y_possible, p=probs)
            else:
                y[b] = r_max  # default if all probs zero

        return y

    def sample_observation_forward(self, π, u):
        """forward sampling version (faster)"""

        # predicted belief
        u_idx = self.AQ.get_quantized_index(u)  # TODO: Implement this
        Tn_mat = self.T_mat[u_idx]
        predicted_belief = Tn_mat @ π

        # sample (x', m)
        flat_belief = predicted_belief.flatten()
        flat_belief /= np.sum(flat_belief)
        idx = np.random.choice(len(flat_belief), p=flat_belief)

        x_idx = idx // self.len_M
        m_idx = idx % self.len_M

        x_prime = self.SQ.X_n[x_idx]
        m_2d = self.bits_to_map(m_idx, self.H, self.W)

        # sample y from sensor model
        y = self.sample_from_sensor_model(x_prime, m_2d)

        return y

    def sample_from_sensor_model(self, x, m):
        """
        sample y ~ Q(y|x,m)

        assuming your Q is deterministic raycasting + noise:
        1. compute expected ranges via raycasting
        2. add noise
        """
        B = self.sensor.B
        r_max = self.sensor.r_max

        # compute true ranges via raycasting (your Q function does this)
        # but we need the expected values, not just likelihood

        v = np.random.multivariate_normal(np.zeros(B), self.cov_y)
        y = self.sensor.g(x, m, v)  # (B,) array

        return y

    def c_n_M(self, π, u):
        """
        Quantized cost function for belief-MDP_n_M.

        c_n_M(π^M, u) = c_tilde_n(π^M, u)
        where π^M is a quantized belief.

        Args:
            π: Quantized belief (N_n,) - flattened belief vector
            u: Action (2,)
        Returns:
            Cost value
        """
        # Reshape the flattened belief vector to the expected 2D format
        # π is (N_n,) where N_n = m_n + 2^(H*W)
        # We need to split it into state and map components
        m_n = self.SQ.size
        H, W = self.map.occupancy_map.height, self.map.occupancy_map.width
        len_M = 2**(H * W)

        # Extract state and map components
        π_states = π[:m_n]  # First m_n elements
        π_maps = π[m_n:]    # Remaining elements

        # Reshape to 2D format expected by c_tilde_n
        π_2d = np.zeros((m_n, len_M))
        for i in range(m_n):
            for j in range(len_M):
                # Map the flattened index to 2D coordinates
                flat_idx = i * len_M + j
                if flat_idx < len(π):
                    π_2d[i, j] = π[flat_idx]

        # Use the quantized cost from the parent class
        return self.c_tilde_n(π_2d, u)
