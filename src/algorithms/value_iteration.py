import numpy as np
# import cupy as np
from ..classes.belief_mdp_n_M import BeliefMDP_n_M


class ValueIteration():
    """
    Value Iteration solver for the finite belief-MDP approximation.
    Extends BeliefMDP_n with value iteration specific functionality.
    """

    def __init__(self, MDP_n: BeliefMDP_n_M, epsilon=1e-6):
        self.MDP_n = MDP_n
        self.epsilon = epsilon  # convergence threshold

        # initialize value function to zeros
        self.V = np.zeros(self.MDP_n.N_n)
        # initialize old value function to zeros
        self.V_old = np.zeros(self.MDP_n.N_n)

    def run_vectorized(self):
        """
        Run value iteration algorithm.
        """
        Π_n = self.MDP_n.get_codebook()
        U_n = self.MDP_n.action_quantizer.get_quantized_points()

        while self.is_not_converged():
            self.V_old = self.V.copy()

            q_values = self.MDP_n.c_n_vectorized(Π_n, U_n) + self.MDP_n.β * \
                np.sum(self.MDP_n.p_n_vectorized(
                    Π_n, U_n) * self.V)  # (1) -> p_n
            assert q_values.shape == (Π_n.shape[0], U_n.shape[0])
            self.V = np.min(q_values, axis=1)

        return self.V

    def run(self): # ✅
        """
        Run value iteration algorithm.
        """
        Π_n = self.MDP_n.get_codebook()
        U_n = self.MDP_n.action_quantizer.get_quantized_points()

        while self.is_not_converged():
            self.V_old = self.V.copy()

            for i, π_t in enumerate(Π_n):
                q_values = [
                    self.MDP_n.c_n(π_t, u_t) + 
                    self.MDP_n.β * self.V @ self.MDP_n.p_n(π_t, u_t).T
                    for u_t in U_n
                ]
                self.V[i] = np.min(q_values)

        return self.V

    def is_not_converged(self):
        """
        Check if value iteration has converged.
        """
        if np.max(np.abs(self.V - self.V_old)) < self.epsilon:
            return False
        return True
