import numpy as np
# import cupy as np
from classes.mapping import LidarGridMapVec
from classes.model import LIDAR, VelocityIntegratorModel
from classes.obstacle import Obstacle
from .quantizer import StateQuantizer, ActionQuantizer, ObservationQuantizer
from .pomdp import POMDP
from utils.misc import cartesian, cartesian_pairs


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
        self.state_quantizer = StateQuantizer(
            self.map.occupancy_map.left_lower[0],
            self.map.occupancy_map.right_upper[0],
            self.map.occupancy_map.left_lower[1],
            self.map.occupancy_map.right_upper[1],
            self.n
        )
        self.action_quantizer = ActionQuantizer(
            self.motion_model.max_v, self.n)
        self.observation_quantizer = ObservationQuantizer(
            self.sensor.r_max,
            self.n
        )

### 1. Filter Update Equation ######################
    def F(self, π, u, y):  # ✅
        """
        Filter update equation for belief state.
        """
        X_n = self.state_quantizer.get_quantized_points()
        m_n = len(X_n)
        M = self.generate_space_of_maps()

        # Compute transition matrix for all states given action u
        T_n = self.T_cartesian(X_n, u)
        assert T_n.shape == (len(X_n), len(X_n), 1)  # x_{t+1} x x_{t}

        # first sum the belief over the space of maps, then apply transition matrix for each conditional
        integral = T_n @ π  # (m_n, m_n) @ (m_n, 2^HW) = (m_n, 2^HW)

        Q_vals = np.zeros((m_n, len(M)))
        for i, m in enumerate(M):  # TODO: Vectorize self.Q to handle full map
            # insert value for each column, where self.Q returns an array of len(X_n)
            Q_vals[:, i] = self.Q(y, X_n, m)  # shape (m_n, 2^HW))
        numerator = Q_vals * integral  # shape (m_n, 2^HW)
        denominator = np.sum(np.sum(numerator))

        return numerator / denominator

    def F_vectorized(self, π, u, Y):  # TODO
        """
        Filter update equation for belief state.
        """
        X_n = self.state_quantizer.get_quantized_points()
        # M = self.

        # Compute transition matrix for all states given action u
        T_n = self.T_cartesian(X_n, u)
        assert T_n.shape == (len(X_n), 2)

        # first sum the belief over the space of maps, then apply transition matrix for each conditional
        integral = T_n @ π.T  # (N_n,N_n) @ (N_n, 2^HW) = (N_n, 2^HW)

        Q_vals = self.Q_vectorized(Y, X_n, M)  # shape (N_n, 2^HW)
        # Q(y|x', m_t)   shape (N_n, 2^HW)
        # T(dx'| x_t, u) shape (N_n, N_n)
        # π(dx_t, dm_t)  shape (N_n, 2^HW)

        # Q_vals @ T.T shape (N_n, 2^HW)
        # above @ π.T shape (N_n, 2^HW)

        numerator = Q_vals * integral
        denominator = np.sum(Q_vals * integral)

        return numerator / denominator

    def H(self, y, π, u):  # ✅
        """
        Belief of observation y given belief pi and action u.
        """
        X_n = self.state_quantizer.get_quantized_points()
        m_n = len(X_n)
        M = self.generate_space_of_maps()

        # Compute transition matrix for all states given action u
        T_n = self.T_cartesian(X_n, u)
        assert T_n.shape == (len(X_n), len(X_n), 1)  # x_{t+1} x x_{t}

        # first sum the belief over the space of maps, then apply transition matrix for each conditional
        integral = T_n @ π  # (m_n, m_n) @ (m_n, 2^HW) = (m_n, 2^HW)

        H, W = self.map.occupancy_map.height, self.map.occupancy_map.width
        Q_vals = np.zeros((m_n, 2 ** (H * W)))
        for i, m in enumerate(M):  # TODO: Vectorize self.Q to handle full map
            # insert value for each column, where self.Q returns an array of len(X_n)
            Q_vals[:, i] = self.Q(y, X_n, m)  # shape (m_n, 2^HW))
        denominator = np.sum(np.sum(Q_vals * integral))

        return denominator

    def H_vectorized(self, Y, Π, u):  # TODO
        """
        Belief of observation y given belief pi and action u.
        """
        X_n = self.state_quantizer.get_quantized_points()

        # Get transition matrix for all states
        T_n = self.T_cartesian(X_n, u)  # shape: (N_n, N_n)

        # Compute observation probabilities for all states
        Q_vals = self.Q_vectorized(Y, X_n, self.true_map)  # shape (N_n, |Y_n|)

        # Vectorized computation: sum over states
        # h = sum_x sum_m Q(y|x,m) * T(x|x,u) * π(x)
        # Since we're using the transition matrix, this becomes:
        # h = sum_x Q(y|x,m) * (T_matrix @ π)[x]

        return Q_vals @ (T_n @ np.sum(Π, axis=1))

###################################################
### 2. Cost Function ##############################
    def c_tilde(self, π, u):  # ✅
        """
        Cost function for belief-MDP with belief π, action u and cost function self.c. 
        This implementation assumes the state space is discrete.
        tilde{c}(π,u))=\int_{X_n} c(x,m,u) π(dx,dm)
        """
        X_n = self.state_quantizer.get_quantized_points()
        H, W = self.map.occupancy_map.height, self.map.occupancy_map.width
        m_n = len(X_n)
        M = self.generate_space_of_maps()
        # π is belief over (x, m): expected shape (m_n, |M|)
        assert π.shape == (m_n, 2 ** (H * W))

        # Evaluate costs per pair with a lightweight loop to respect current c(x,m,u)
        pair_costs = np.array((m_n, len(M)))
        for i, x in enumerate(X_n):
            for j, m in enumerate(M):
                pair_costs[i, j] = self.c(x, m, u)

        # Weighted sum: sum_{x,m} c(x,m,u) * π(x,m)
        return ((pair_costs * π).sum()).sum()
###################################################
