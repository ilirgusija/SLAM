import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from classes.model import CarModel, LIDAR
from utils.angle import rot_mat_2d
import time


class OccupancyGrid:
    """Occupancy grid representation of the environment"""

    def __init__(self, width, height, resolution):
        """
        Initialize an occupancy grid
        Args:
            width (float): Width of the environment in meters
            height (float): Height of the environment in meters
            resolution (float): Grid resolution in meters/cell
        """
        self.width = width
        self.height = height
        self.resolution = resolution

        # Calculate grid dimensions
        self.grid_width = int(np.ceil(width / resolution))
        self.grid_height = int(np.ceil(height / resolution))

        # Initialize grid with 0.5 (unknown) probability
        self.grid = 0.5 * np.ones((self.grid_height, self.grid_width))

        # Map origin (bottom-left corner in world coordinates)
        self.origin_x = 0
        self.origin_y = 0

    def update_grid(self, points):
        """
        Update the occupancy grid with new measurement points
        Args:
            points: Nx3 array of (x, y, label) points, where label is 0 (free) or 1 (occupied)
        """
        for x, y, label in points:
            # Convert world coordinates to grid indices
            grid_x = int((x - self.origin_x) / self.resolution)
            grid_y = int((y - self.origin_y) / self.resolution)

            # Check if cell is within grid bounds
            if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                # Update cell using log-odds update
                if label == 1:  # Occupied
                    self.grid[grid_y, grid_x] = min(
                        0.95, self.grid[grid_y, grid_x] * 1.2)
                else:  # Free
                    self.grid[grid_y, grid_x] = max(
                        0.05, self.grid[grid_y, grid_x] * 0.8)

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices"""
        grid_x = int((x - self.origin_x) / self.resolution)
        grid_y = int((y - self.origin_y) / self.resolution)
        return grid_x, grid_y

    def grid_to_world(self, grid_x, grid_y):
        """Convert grid indices to world coordinates (cell center)"""
        x = self.origin_x + (grid_x + 0.5) * self.resolution
        y = self.origin_y + (grid_y + 0.5) * self.resolution
        return x, y

    def is_occupied(self, x, y, threshold=0.6):
        """Check if a cell at world coordinates (x,y) is occupied"""
        grid_x, grid_y = self.world_to_grid(x, y)
        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
            return self.grid[grid_y, grid_x] > threshold
        return False

    def plot(self, ax):
        """
        Plot the occupancy grid with custom colormap for different states
        - White: Free space (probability < 0.3)
        - Black: Occupied space (probability > 0.7)
        - Gray: Unobserved/uncertain space (probability around 0.5)
        """
        # Create custom colormap
        colors = [(1, 1, 1),    # White for free space
                  (0.7, 0.7, 0.7),  # Gray for unknown
                  (0, 0, 0)]     # Black for occupied
        n_bins = 100  # Number of color bins
        cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

        # Display grid using matplotlib imshow
        im = ax.imshow(self.grid, cmap=cmap, origin='lower',
                       extent=[self.origin_x, self.origin_x + self.width,
                               self.origin_y, self.origin_y + self.height],
                       vmin=0, vmax=1)

        # Add colorbar
        plt.colorbar(im, ax=ax, label='Occupancy Probability')

        # Set title and labels
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')

        return im


class StateQuantizer:
    """Quantizes the continuous state space into discrete cells for value iteration"""

    def __init__(self, x_min, x_max, y_min, y_max, theta_min, theta_max,
                 nx, ny, ntheta):
        """
        Initialize state quantizer
        Args:
            x_min, x_max: X coordinate bounds
            y_min, y_max: Y coordinate bounds
            theta_min, theta_max: Orientation bounds
            nx, ny, ntheta: Number of quantization cells in each dimension
        """
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.theta_min, self.theta_max = theta_min, theta_max

        self.nx, self.ny, self.ntheta = nx, ny, ntheta

        # Calculate cell sizes
        self.dx = (x_max - x_min) / nx
        self.dy = (y_max - y_min) / ny
        self.dtheta = (theta_max - theta_min) / ntheta

        # Create meshgrid for visualization
        self.x_centers = np.linspace(x_min + self.dx/2, x_max - self.dx/2, nx)
        self.y_centers = np.linspace(y_min + self.dy/2, y_max - self.dy/2, ny)
        self.theta_centers = np.linspace(theta_min + self.dtheta/2,
                                         theta_max - self.dtheta/2, ntheta)

    def continuous_to_discrete(self, x, y, theta):
        """Convert continuous state to discrete cell indices"""
        # Normalize angle to [theta_min, theta_max]
        while theta < self.theta_min:
            theta += 2*np.pi
        while theta >= self.theta_max:
            theta -= 2*np.pi

        # Calculate indices
        ix = min(max(int((x - self.x_min) / self.dx), 0), self.nx - 1)
        iy = min(max(int((y - self.y_min) / self.dy), 0), self.ny - 1)
        itheta = min(max(int((theta - self.theta_min) / self.dtheta), 0),
                     self.ntheta - 1)

        return (ix, iy, itheta)

    def discrete_to_continuous(self, ix, iy, itheta):
        """Convert discrete cell indices to continuous state (cell center)"""
        x = self.x_min + (ix + 0.5) * self.dx
        y = self.y_min + (iy + 0.5) * self.dy
        theta = self.theta_min + (itheta + 0.5) * self.dtheta

        return (x, y, theta)

    def get_all_states(self):
        """Get all discrete states as a list of (x,y,theta) tuples"""
        states = []
        for ix in range(self.nx):
            for iy in range(self.ny):
                for itheta in range(self.ntheta):
                    states.append((ix, iy, itheta))
        return states


class ActionSet:
    """Defines the set of discrete actions for value iteration"""

    def __init__(self, v_options, omega_options):
        """
        Initialize action set
        Args:
            v_options: List of possible linear velocities
            omega_options: List of possible angular velocities
        """
        self.v_options = v_options
        self.omega_options = omega_options

        # Create all action combinations
        self.actions = []
        for v in v_options:
            for omega in omega_options:
                self.actions.append((v, omega))

    def get_actions(self):
        """Get all available actions"""
        return self.actions


class CostFunction:
    """Cost function for SLAM value iteration"""

    def __init__(self, occupancy_grid, w_estimation=1.0, w_coverage=10.0,
                 w_obstacle=10.0, w_control=0.1):
        """
        Initialize cost function with weights
        Args:
            occupancy_grid: OccupancyGrid object
            w_estimation: Weight for estimation error term
            w_coverage: Weight for map coverage term
            w_obstacle: Weight for obstacle avoidance term
            w_control: Weight for control effort term
        """
        self.grid = occupancy_grid
        self.w_estimation = w_estimation
        self.w_coverage = w_coverage
        self.w_obstacle = w_obstacle
        self.w_control = w_control

        # Safety distance for obstacle avoidance (in meters)
        self.safety_distance = 0.5

    def estimation_error(self, robot_state, map_state):
        """
        Calculate estimation error cost term
        For simulation purposes, we approximate this with uncertainty in the map
        """
        # Currently using a simplified measure - counts grid cells with
        # uncertain values (close to 0.5)
        uncertain_count = np.sum(np.abs(self.grid.grid - 0.5) < 0.2)
        return uncertain_count / (self.grid.grid_width * self.grid.grid_height)

    def map_coverage(self):
        """
        Calculate map coverage cost term
        Measures the difference of map values from the initial 0.5
        """
        # Count cells that differ significantly from the initial 0.5 value
        coverage = np.sum(np.abs(self.grid.grid - 0.5) > 0.2)
        # Normalize by grid size and negate (since we want to maximize coverage)
        return -coverage / (self.grid.grid_width * self.grid.grid_height)

    def obstacle_avoidance(self, x, y):
        """
        Calculate obstacle avoidance cost
        Returns high cost for positions close to obstacles
        """
        # Check grid cells within safety distance
        safety_cells = int(
            np.ceil(self.safety_distance / self.grid.resolution))
        cost = 0

        # For efficiency, only check cells within a square around the robot
        grid_x, grid_y = self.grid.world_to_grid(x, y)

        for i in range(max(0, grid_x - safety_cells),
                       min(self.grid.grid_width, grid_x + safety_cells + 1)):
            for j in range(max(0, grid_y - safety_cells),
                           min(self.grid.grid_height, grid_y + safety_cells + 1)):
                # Calculate distance to this cell
                cell_x, cell_y = self.grid.grid_to_world(i, j)
                dist = np.sqrt((x - cell_x)**2 + (y - cell_y)**2)

                if dist <= self.safety_distance:
                    # Weight by occupancy probability and inverse distance
                    weight = max(0, 1 - dist/self.safety_distance)
                    occupied_prob = self.grid.grid[j, i]
                    cost += weight * occupied_prob

        return cost

    def control_effort(self, v, omega):
        """Calculate control effort cost"""
        # Simple quadratic cost on control inputs
        return v**2 + omega**2

    def total_cost(self, robot_state, map_state, action):
        """
        Calculate total weighted cost
        Args:
            robot_state: (x, y, theta) robot state
            map_state: Current map state (currently not used directly)
            action: (v, omega) control inputs
        """
        x, y, theta = robot_state
        v, omega = action

        # Calculate individual cost terms
        est_cost = self.estimation_error(robot_state, map_state)
        cov_cost = self.map_coverage()
        obs_cost = self.obstacle_avoidance(x, y)
        ctrl_cost = self.control_effort(v, omega)

        # Weighted sum
        total = (self.w_estimation * est_cost +
                 self.w_coverage * cov_cost +
                 self.w_obstacle * obs_cost +
                 self.w_control * ctrl_cost)

        return total


class ValueIteration:
    """Value iteration algorithm for SLAM"""

    def __init__(self, quantizer, action_set, occupancy_grid, vehicle_model,
                 lidar_model, segments, discount_factor=0.9, max_iter=100,
                 convergence_threshold=0.01):
        """
        Initialize value iteration
        Args:
            quantizer: StateQuantizer object
            action_set: ActionSet object
            occupancy_grid: OccupancyGrid object
            vehicle_model: VehicleSimulator object
            lidar_model: LIDAR object
            segments: Environment segments for LIDAR simulation
            discount_factor: Discount factor for future rewards
            max_iter: Maximum number of iterations
            convergence_threshold: Convergence criterion
        """
        self.quantizer = quantizer
        self.action_set = action_set
        self.grid = occupancy_grid
        self.vehicle = vehicle_model
        self.lidar = lidar_model
        self.segments = segments
        self.gamma = discount_factor
        self.max_iter = max_iter
        self.convergence_threshold = convergence_threshold

        # Initialize cost function
        self.cost_function = CostFunction(occupancy_grid)

        # Value function (3D array indexed by quantized state)
        self.V = np.zeros((quantizer.nx, quantizer.ny, quantizer.ntheta))

        # Policy (3D array of action indices)
        self.policy = np.zeros((quantizer.nx, quantizer.ny, quantizer.ntheta),
                               dtype=int)

        # For visualization
        self.value_history = []
        self.policy_history = []

        # Create a deep copy of the vehicle for simulation
        self.sim_vehicle = CarModel(
            i_x=vehicle_model.x,
            i_y=vehicle_model.y,
            i_theta=vehicle_model.theta,
            i_omega=vehicle_model.omega,
            dt=vehicle_model.dt,
            i_v=vehicle_model.v,
            max_v=vehicle_model.max_v,
            w=vehicle_model.W,
            L=vehicle_model.L
        )

    def get_next_state(self, state, action, dt):
        """
        Predict next state given current state and action
        Returns both the continuous next state and its quantized version
        """
        x, y, theta = state
        v, omega = action

        # Set vehicle to current state (using sim_vehicle to avoid modifying the main vehicle)
        self.sim_vehicle.x = x
        self.sim_vehicle.y = y
        self.sim_vehicle.theta = theta

        # Apply action for one time step
        self.sim_vehicle.update(v, omega)

        # Get next state
        next_state = (self.sim_vehicle.x, self.sim_vehicle.y,
                      self.sim_vehicle.theta)

        # Get quantized next state
        discrete_next_state = self.quantizer.continuous_to_discrete(
            *next_state)

        return next_state, discrete_next_state

    def update_map(self, robot_state):
        """Update occupancy grid based on LIDAR measurements from robot_state"""
        x, y, theta = robot_state
        robot_pose = np.array([x, y, theta])

        # Get LIDAR measurements
        angles, dist_theta = self.lidar.get_laser_ref(
            self.segments, robot_pose)

        # Get occupancy points
        points = self.lidar.get_filled_txy(dist_theta, angles, robot_pose)

        # Update grid
        self.grid.update_grid(points)

    def run_iteration(self):
        """Run one iteration of value iteration"""
        new_V = np.zeros_like(self.V)
        new_policy = np.zeros_like(self.policy)

        # Get all possible actions
        actions = self.action_set.get_actions()

        # Loop over all states
        for ix in range(self.quantizer.nx):
            for iy in range(self.quantizer.ny):
                for itheta in range(self.quantizer.ntheta):
                    # Get continuous state from discrete indices
                    state = self.quantizer.discrete_to_continuous(
                        ix, iy, itheta)

                    # Find best action for this state
                    best_value = float('inf')  # We're minimizing cost
                    best_action_idx = 0

                    for idx, action in enumerate(actions):
                        # Calculate immediate cost
                        cost = self.cost_function.total_cost(
                            state, None, action)

                        # Predict next state
                        next_state, discrete_next_state = self.get_next_state(
                            state, action, self.vehicle.dt)

                        # Add discounted future cost
                        nx, ny, ntheta = discrete_next_state
                        expected_future_cost = self.gamma * \
                            self.V[nx, ny, ntheta]

                        # Total value (cost)
                        value = cost + expected_future_cost

                        if value < best_value:
                            best_value = value
                            best_action_idx = idx

                    # Update value function and policy
                    new_V[ix, iy, itheta] = best_value
                    new_policy[ix, iy, itheta] = best_action_idx

        # Calculate change in value function
        delta = np.max(np.abs(new_V - self.V))

        # Update value function and policy
        self.V = new_V
        self.policy = new_policy

        # Store for visualization
        self.value_history.append(new_V.copy())
        self.policy_history.append(new_policy.copy())

        return delta

    def run(self, visualize=True):
        """Run value iteration algorithm until convergence"""
        converged = False
        iter_count = 0

        # For visualization
        if visualize:
            plt.ion()  # Turn on interactive mode
            # Create figure with GridSpec to properly manage colorbar space
            fig = plt.figure(figsize=(15, 7))
            gs = plt.GridSpec(1, 4, figure=fig)
            ax1 = fig.add_subplot(gs[0, :2])  # First two columns
            ax2 = fig.add_subplot(gs[0, 2:])  # Last two columns

        # Main iteration loop
        while not converged and iter_count < self.max_iter:
            # Run one iteration
            delta = self.run_iteration()

            # Execute current best policy for visualization
            if visualize:
                # Get current best action
                v, omega = self.get_optimal_action(
                    self.vehicle.x, self.vehicle.y, self.vehicle.theta)

                # Update vehicle state
                self.vehicle.update(v, omega)

                # Update map based on new position
                self.update_map(
                    (self.vehicle.x, self.vehicle.y, self.vehicle.theta))

            iter_count += 1

            # Check for convergence
            if delta < self.convergence_threshold:
                converged = True

            # Print progress
            if iter_count % 5 == 0:
                print(f"Iteration {iter_count}, delta = {delta:.6f}")

            # Update visualization
            if visualize:
                # Clear everything
                ax1.clear()
                ax2.clear()

                # Clear all colorbars by removing all axes except our main plots
                for ax in fig.axes:
                    if ax not in [ax1, ax2]:
                        ax.remove()

                # Plot occupancy grid
                self.grid.plot(ax1)
                ax1.set_title(f'Occupancy Grid - Iteration {iter_count}')

                # Plot robot position and orientation
                robot_x, robot_y, robot_theta = self.vehicle.x, self.vehicle.y, self.vehicle.theta
                ax1.plot(robot_x, robot_y, 'ro', markersize=10, label='Robot')

                # Draw arrow for robot orientation
                arrow_length = 0.5
                ax1.arrow(robot_x, robot_y,
                          arrow_length*np.cos(robot_theta),
                          arrow_length*np.sin(robot_theta),
                          head_width=0.2, head_length=0.3, fc='r', ec='r')

                # Plot value function for current orientation
                discrete_theta = self.quantizer.continuous_to_discrete(
                    0, 0, robot_theta)[2]
                value_slice = self.V[:, :, discrete_theta]

                # Plot value function
                im = ax2.imshow(value_slice.T, origin='lower',
                                extent=[self.quantizer.x_min, self.quantizer.x_max,
                                        self.quantizer.y_min, self.quantizer.y_max],
                                cmap='RdYlBu_r')

                ax2.set_title(
                    f'Value Function Slice (θ={robot_theta:.2f} rad)')
                ax2.set_xlabel('X (m)')
                ax2.set_ylabel('Y (m)')

                # Add colorbar for value function
                plt.colorbar(im, ax=ax2, label='Value')

                # Add grid
                ax1.grid(True)
                ax2.grid(True)

                # Adjust layout to prevent overlap
                plt.tight_layout()

                # Update display
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.1)

        if visualize:
            plt.ioff()  # Turn off interactive mode
            plt.close(fig)

        print(f"Value iteration {'converged' if converged else 'did not converge'} "
              f"after {iter_count} iterations")

        return converged

    def get_optimal_action(self, x, y, theta):
        """Get optimal action for a given continuous state"""
        # Convert to discrete state
        ix, iy, itheta = self.quantizer.continuous_to_discrete(x, y, theta)

        # Get action index from policy
        action_idx = self.policy[ix, iy, itheta]

        # Return the actual action
        return self.action_set.actions[action_idx]

    def simulate_trajectory(self, start_state, n_steps):
        """
        Simulate trajectory following the optimal policy
        Args:
            start_state: (x, y, theta) starting state
            n_steps: Number of steps to simulate
        """
        # Initialize vehicle
        self.vehicle.x, self.vehicle.y, self.vehicle.theta = start_state

        # Storage for trajectory
        x_hist = [self.vehicle.x]
        y_hist = [self.vehicle.y]
        theta_hist = [self.vehicle.theta]

        # Simulate
        for _ in range(n_steps):
            # Get optimal action
            v, omega = self.get_optimal_action(self.vehicle.x, self.vehicle.y,
                                               self.vehicle.theta)

            # Update vehicle state
            self.vehicle.update(v, omega)

            # Store current state
            x_hist.append(self.vehicle.x)
            y_hist.append(self.vehicle.y)
            theta_hist.append(self.vehicle.theta)

            # Update map
            self.update_map(
                (self.vehicle.x, self.vehicle.y, self.vehicle.theta))

        return np.array(x_hist), np.array(y_hist), np.array(theta_hist)

# Example usage


def main():
    # Define environment (obstacle segments as line segments)
    segments = [
        # Outer boundaries of a rectangular environment
        [0, 0, 10, 0],         # Bottom wall
        [10, 0, 10, 10],       # Right wall
        [10, 10, 0, 10],       # Top wall
        [0, 10, 0, 0],         # Left wall

        # Add some internal obstacles for more interesting exploration
        [3, 3, 7, 3],          # Horizontal wall
        [5, 3, 5, 7],          # Vertical wall
    ]

    # Initialize robot and sensor models
    dt = 0.1
    vehicle = CarModel(
        i_x=1.0, i_y=1.0, i_theta=0.0,  # Start position
        i_omega=0.0, dt=dt, i_v=0.0, max_v=1.0,
        w=0.5, L=1.0  # Vehicle dimensions
    )

    # Use fewer reflections to reduce computational load
    lidar = LIDAR(fov=2*np.pi, max_range=5.0, n_reflections=36)

    # Initialize occupancy grid
    grid = OccupancyGrid(width=10.0, height=10.0,
                         resolution=0.2)  # Finer resolution

    # Initialize state quantizer
    quantizer = StateQuantizer(
        x_min=0.0, x_max=10.0,
        y_min=0.0, y_max=10.0,
        theta_min=0.0, theta_max=2*np.pi,
        nx=20, ny=20, ntheta=8  # More states for better policy
    )

    # Initialize action set with more options
    action_set = ActionSet(
        v_options=[0.0, 0.3, 0.6],  # Three velocity options
        omega_options=[-0.5, -0.2, 0.0, 0.2, 0.5]  # Five steering options
    )

    # Initialize value iteration with adjusted weights
    vi = ValueIteration(
        quantizer=quantizer,
        action_set=action_set,
        occupancy_grid=grid,
        vehicle_model=vehicle,
        lidar_model=lidar,
        segments=segments,
        discount_factor=0.95,  # Higher discount factor for longer-term planning
        max_iter=50,  # More iterations
        convergence_threshold=0.1
    )

    try:
        # Initial map update based on LIDAR
        vi.update_map((vehicle.x, vehicle.y, vehicle.theta))

        # Run value iteration
        print("Running value iteration with policy execution...")
        vi.run(visualize=True)

        print("Value iteration completed successfully!")
        print("Computing final trajectory...")

        # Reset vehicle and map for final trajectory
        vehicle.x, vehicle.y, vehicle.theta = 1.0, 1.0, 0.0
        grid = OccupancyGrid(width=10.0, height=10.0, resolution=0.2)
        vi.grid = grid

        # Simulate longer trajectory
        x_hist, y_hist, theta_hist = vi.simulate_trajectory(
            start_state=(1.0, 1.0, 0.0),
            n_steps=100  # Longer trajectory
        )

        # Create final visualization
        plt.figure(figsize=(12, 8))
        ax = plt.gca()

        # Plot occupancy grid with custom colormap
        colors = [(1, 1, 1), (0.7, 0.7, 0.7), (0, 0, 0)]
        cmap = LinearSegmentedColormap.from_list("custom", colors, N=100)
        im = ax.imshow(vi.grid.grid, cmap=cmap, origin='lower',
                       extent=[0, 10, 0, 10],
                       vmin=0, vmax=1)
        plt.colorbar(im, label='Occupancy Probability')

        # Plot trajectory
        ax.plot(x_hist, y_hist, 'b-', linewidth=2, label='Trajectory')
        ax.plot(x_hist[0], y_hist[0], 'go', markersize=10, label='Start')
        ax.plot(x_hist[-1], y_hist[-1], 'ro', markersize=10, label='End')

        # Draw robot orientations along trajectory
        plot_interval = max(1, len(x_hist) // 10)
        for i in range(0, len(x_hist), plot_interval):
            arrow_length = 0.3
            ax.arrow(x_hist[i], y_hist[i],
                     arrow_length*np.cos(theta_hist[i]),
                     arrow_length*np.sin(theta_hist[i]),
                     head_width=0.1, head_length=0.2, fc='b', ec='b', alpha=0.5)

        # Add legend and grid
        ax.legend(loc='upper right')
        ax.grid(True)
        ax.set_aspect('equal')
        ax.set_title('Final Trajectory and Occupancy Grid')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')

        # Save and show the result
        plt.savefig('value_iteration_result.png', dpi=300, bbox_inches='tight')
        print("Results saved to value_iteration_result.png")
        plt.show()

    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
