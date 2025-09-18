import math
import numpy as np
# import cupy as np
from classes.belief_mdp_n import BeliefMDP_n
from classes.mapping import LidarGridMapVec
from classes.model import LIDAR, VelocityIntegratorModel
from classes.obstacle import Obstacle
from .quantizer import reznik_algorithm


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
        self.N_n = self.state_quantizer.size + \
            (2 ** self.map.occupancy_map.size)

        # Initialize reznik algorithm based on import and fix the second and third argument
        # effectively making it a function of just z
        self.belief_quantizer = lambda z: reznik_algorithm(z, self.M, self.N_n)

    def p_n(self, π, u):  # ✅
        """
        Quantized transition probability to all beliefs in Π_n given belief π and action u.
        """
        # assuming ν is dirac on the quantized measures
        belief = self.η(π, u)  # shape (Π_size, 1)
        p_n = self.belief_quantizer(belief)
        return p_n

    def c_n(self, π, u):  # ✅
        """
        Cost function for belief-MDP_n with belief π, action u.
        """
        c_n = self.c_tilde(self.belief_quantizer(π), u)
        return c_n

    def η(self, π, u, tol=1e-8):  # ✅
        """
        Transition probability to all beliefs in Π_n given belief π and action u.
        Returns a vector of size Π_size.
        """
        Π_n = self.get_codebook()
        Π_size = len(Π_n.flatten())

        # shape ( |R|^B, B, 1) for each permutation of possible measurements (i.e. |R| the size of the discretized range set) from our B beams
        # could set quantization approach to decay exponentially as we get farther, reflecting the low probability of far detections
        Y = self.observation_quantizer.get_quantized_points()

        # find the index of the closest F_vals to π_star using np.isclose
        idxs = np.array(len(Y), dtype=list)
        for i, π_star in enumerate(Π_n):
            for j, y in enumerate(Y):
                if np.isclose(self.F(π, u, y), π_star, atol=tol):
                    idxs[i].append(j)

        # compute integral
        η = np.array(Π_size)
        for j in range(Π_size):
            η[j] = np.array([np.sum([self.H(Y[i], π, u)
                            for i in idx]) for idx in idxs])
        return η  # shape (Π_size, 1)

    # TODO: make this iterable, as it stands this is way too big to return an entire array (or is it?)
    def get_codebook(self):
        """
        Extract codebook by densely sampling simplex and collecting outputs
        """
        # sample densely on (k-1)-simplex
        card = math.comb(self.M+self.N_n-1, self.N_n-1)  # eq 60
        Π_n = np.zeros((card, self.N_n))
        for i in range(card):
            Π_n[i] = np.random.dirichlet(np.ones(self.N_n), size=1)[0]
        quantized = [self.belief_quantizer(π_i) for π_i in Π_n]

        # extract unique reproduction points
        codebook = np.unique(np.array(quantized), axis=0)
        assert codebook.shape[0] == card, "Codebook size mismatch"
        return codebook

    # Vectorized methods
    def p_n_vectorized(self, Π, u):
        """
        Transition probability for belief-MDP_n with belief pi_1, pi_0 and action u_0.
        """
        # assuming ν is dirac on the quantized measures
        p_n = self.belief_quantizer(self.η_vectorized(Π, u))
        return p_n

    def c_n_vectorized(self, Π, U_n):
        """
        Saldi, 2019, Asymptotic Optimality of Finite Model Approximations for Partially Observed Markov Decision Processes With Discounted Cost
        Cost function for belief-MDP_n with belief π, action u.
        This implementation assumes the state space is discrete.
        c_n(π,u)=\int_{B_i^{(n)}} \\tilde{c}(π,u) \\ν_i^{(self.M)}(dπ)
        where \\ν_i^{(self.M)} is the weighting measure for the i-th belief. 
        """
        c_n = self.c_tilde(
            Π, U_n)  # assuming ν is dirac on the quantized measures
        return c_n

    def η_vectorized(self, Π, u, tol=1e-8):
        """
        Filter transition probability for belief-MDP given belief pi_0 and action u_0.
        Returns a vector of size N_n.
        """
        Y = self.observation_quantizer.get_quantized_points()
        F_vals = self.F_vectorized(Π, u, Y)  # (1) run -> p_n -> η -> F

        # compare each F(π, u, y) ≈ Π (broadcasted)
        matches = np.all(np.isclose(F_vals, Π, atol=tol), axis=1)  # shape (k,)

        if not np.any(matches):
            raise ValueError("No observation matched Π")

        if matches.sum() > 1:
            raise ValueError("Multiple observations matched Π")

        idx = np.argmax(matches)  # take first match (you assume deterministic)
        y_star = Y[idx]

        return self.H_vectorized(y_star, Π, u)  # (2) run -> p_n -> η -> H
