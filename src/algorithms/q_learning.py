from ..utils.map import load_obstacles_config
from ..classes.mapping import LidarGridMapVec, GridMapNP, OCCUPIED
from ..classes.model import SingleIntegratorModel, LIDAR
from ..utils.array_backend import np


class QLearning:
    def __init__(self, state_space, action_space, x_0, discount_factor=0.9, resolution=0.2, env='toy1'):
        self.state_space = state_space
        self.action_space = action_space
        self.beta = discount_factor
        self.Q = np.zeros((len(state_space), len(action_space)))
        self.x_history = [x_0]
        self.u_history = []
        # Initialize true map from config accounting for resolution
        self.model = SingleIntegratorModel(x_0[0], x_0[1], 0.1, 3)
        self.sensor = LIDAR(fov=360, max_range=12, n_reflections=360)
        self.gridmap = LidarGridMapVec(*area, resolution=resolution)
        all_obstacles, area = load_obstacles_config(environment=env)
        self.true_map = self.initialize_true_map(all_obstacles, area)

    def initialize_true_map(self, all_obstacles, area):
        x_min, x_max, y_min, y_max = area
        true_map = GridMapNP(x_min, x_max, y_min, y_max,
                             resolution=self.gridmap.resolution, init_val=0.0)
        for obs in all_obstacles:
            # Get rectangle corners in world coordinates
            dx_cos = obs.dx * np.cos(obs.angle)
            dx_sin = obs.dx * np.sin(obs.angle)
            dy_sin = obs.dy * np.sin(obs.angle)
            dy_cos = obs.dy * np.cos(obs.angle)
            cx, cy = obs.centroid
            # Four corners (clockwise)
            corners = np.array([
                [cx + 0.5 * (dx_cos + dy_sin), cy + 0.5 * (dx_sin - dy_cos)],  # BR
                [cx + 0.5 * (dx_cos - dy_sin), cy + 0.5 * (dx_sin + dy_cos)],  # TR
                [cx - 0.5 * (dx_cos - dy_sin), cy - 0.5 * (dx_sin + dy_cos)],  # BL
                [cx - 0.5 * (dx_cos + dy_sin), cy - 0.5 * (dx_sin - dy_cos)],  # TL
            ])
            # Fill grid cells inside the polygon
            # (Adapted from GridMap.set_value_from_polygon)
            pol_x = corners[:, 0]
            pol_y = corners[:, 1]
            # Make sure polygon is closed
            if pol_x[0] != pol_x[-1] or pol_y[0] != pol_y[-1]:
                pol_x = np.append(pol_x, pol_x[0])
                pol_y = np.append(pol_y, pol_y[0])
            for x_ind in range(self.true_map.width):
                for y_ind in range(self.true_map.height):
                    # Get cell center in world coordinates
                    x_pos = self.true_map.left_lower[0] + x_ind * \
                        self.true_map.resolution + self.true_map.resolution / 2.0
                    y_pos = self.true_map.left_lower[1] + y_ind * \
                        self.true_map.resolution + self.true_map.resolution / 2.0
                    # Point-in-polygon test (ray casting)
                    n_point = len(pol_x) - 1
                    inside = False
                    for i1 in range(n_point):
                        i2 = (i1 + 1) % (n_point + 1)
                        if pol_x[i1] >= pol_x[i2]:
                            min_x, max_x = pol_x[i2], pol_x[i1]
                        else:
                            min_x, max_x = pol_x[i1], pol_x[i2]
                        if not min_x <= x_pos < max_x:
                            continue
                        tmp1 = (pol_y[i2] - pol_y[i1]) / (pol_x[i2] -
                                                          pol_x[i1]) if (pol_x[i2] - pol_x[i1]) != 0 else 0
                        if (pol_y[i1] + tmp1 * (x_pos - pol_x[i1]) - y_pos) > 0.0:
                            inside = not inside
                    if inside:
                        self.true_map.data[x_ind, y_ind] = OCCUPIED
        return true_map

    def alpha_t(self, t, x, u):
        return 1 / (1 + np.sum((self.x_history[:t] == x) & (self.u_history[:t] == u)))

    def select_action(self, state):
        return np.random.choice(self.action_space)

    def weighted_cost(self, x, u):
        # --- Weights (tune as needed) ---
        w_estimation = 1.0
        w_coverage = 10.0
        w_obstacle = 10.0
        w_control = 0.1

        # --- 1. Estimation error: count uncertain cells (close to 0.5) ---
        est_cost = self.estimation_cost(x, u)

        # --- 2. Map coverage: count cells far from 0.5 (well-explored) ---
        cov_cost = self.coverage_cost(x, u)

        # --- 3. Obstacle avoidance: penalize being near occupied cells ---
        obs_cost = self.obstacle_cost(x, u)

        # --- 4. Control effort: quadratic penalty on action ---
        if hasattr(u, '__getitem__') and len(u) == 2:
            v, omega = u
        else:
            v, omega = u, 0
        ctrl_cost = v**2 + omega**2

        # --- Weighted sum ---
        total = (w_estimation * est_cost +
                 w_coverage * cov_cost +
                 w_obstacle * obs_cost +
                 w_control * ctrl_cost)
        return total

    def coverage_cost(self, x, u):
        return -np.mean(np.abs(self.gridmap.occupancy_map.data - 0.5))

    def estimation_cost(self, x, u):
        grid = self.gridmap.occupancy_map.data  # 2D numpy array
        est_cost = np.mean(np.abs(grid - self.true_map.data))
        return est_cost

    def obstacle_cost(self, x, u):
        # --- 3. Obstacle avoidance: penalize being near occupied cells ---
        x_pos, y_pos = x[0], x[1]
        x_ind, y_ind = self.gridmap.occupancy_map.get_xy_index_from_xy_pos(
            np.array([x_pos, y_pos]))
        window = grid[x_ind - 16:x_ind + 16, y_ind - 16:y_ind + 16]
        F_cr = 10.0  # constant
        indices = np.indices(window.shape)
        d_ij = np.sqrt(indices[0]**2 + indices[1]**2)
        F_ij = (F_cr * window / (d_ij**2)) * (1 / d_ij *
                                              np.linalg.norm(indices - [x_ind, y_ind], axis=0))
        F_r = np.sum(F_ij)
        cost_obs = np.max(0, np.dot(F_r, u) /
                          (np.linalg.norm(F_r) * np.linalg.norm(u) + 1e-6))
        return cost_obs

    def c(self, x, u):
        return self._qlearning_cost(x, u)

    def update_q_table(self, x, u, t):
        alpha_t = self.alpha_t(t, x, u)

        x_next = self.model.update(u)
        u_next = np.argmin(self.Q[x_next, :])
        self.x_history.append(x_next)
        self.u_history.append(u_next)

        self.Q[x, u] += alpha_t * \
            (self.c(x, u) + self.beta * self.Q[x_next, u_next] - self.Q[x, u])

    def run_algorithm(self):
        t = 0
        while True:
            x = self.x_history[t]
            u = self.u_history[t]
            self.update_q_table(x, u, t)
            t += 1
            if t > 100000:
                break
