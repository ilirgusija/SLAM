import numpy as np
import matplotlib.pyplot as plt
from classes.model import CarModel


def test_circular_motion():
    """Test vehicle following a circular trajectory"""
    # Initialize vehicle
    dt = 0.1  # time step
    vehicle = CarModel(
        i_x=0.0,        # initial x
        i_y=0.0,        # initial y
        i_theta=0.0,    # initial orientation
        i_omega=0.0,    # initial angular velocity
        dt=dt,          # time step
        i_v=1.0,        # initial velocity
        max_v=2.0,      # maximum velocity
        w=0.5,          # vehicle width
        L=1.0           # vehicle length
    )

    # Simulation parameters
    t_max = 10.0  # maximum simulation time
    n_steps = int(t_max/dt)

    # Storage for trajectory
    x_hist = np.zeros(n_steps)
    y_hist = np.zeros(n_steps)
    theta_hist = np.zeros(n_steps)

    # Constant velocity and angular velocity for circular motion
    v = 1.0  # linear velocity
    omega = 0.5  # angular velocity

    # Simulate
    for i in range(n_steps):
        # Store current state
        x_hist[i] = vehicle.x
        y_hist[i] = vehicle.y
        theta_hist[i] = vehicle.theta

        # Update vehicle state
        vehicle.update(v, omega)

    # Plot results
    plt.figure(figsize=(10, 10))
    plt.plot(x_hist, y_hist, 'b-', label='Vehicle trajectory')
    plt.plot(x_hist[0], y_hist[0], 'go', label='Start')
    plt.plot(x_hist[-1], y_hist[-1], 'ro', label='End')

    # Plot vehicle orientation at intervals
    plot_interval = n_steps // 10
    for i in range(0, n_steps, plot_interval):
        plt.arrow(x_hist[i], y_hist[i],
                  0.2*np.cos(theta_hist[i]), 0.2*np.sin(theta_hist[i]),
                  head_width=0.05)

    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title('Vehicle Circular Motion Test')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.savefig('motion_model_test.png')
    plt.close()


def test_straight_line_motion():
    """Test vehicle moving in a straight line"""
    # Initialize vehicle
    dt = 0.1
    vehicle = CarModel(
        i_x=0.0, i_y=0.0, i_theta=np.pi/4,  # 45-degree initial orientation
        i_omega=0.0, dt=dt, i_v=1.0, max_v=2.0,
        w=0.5, L=1.0
    )

    # Simulation parameters
    t_max = 5.0
    n_steps = int(t_max/dt)

    # Storage
    x_hist = np.zeros(n_steps)
    y_hist = np.zeros(n_steps)
    theta_hist = np.zeros(n_steps)

    # Constant velocity, zero angular velocity
    v = 1.0
    omega = 0.0

    # Simulate
    for i in range(n_steps):
        x_hist[i] = vehicle.x
        y_hist[i] = vehicle.y
        theta_hist[i] = vehicle.theta
        vehicle.update(v, omega)

    # Plot results
    plt.figure(figsize=(10, 10))
    plt.plot(x_hist, y_hist, 'b-', label='Vehicle trajectory')
    plt.plot(x_hist[0], y_hist[0], 'go', label='Start')
    plt.plot(x_hist[-1], y_hist[-1], 'ro', label='End')

    # Plot vehicle orientation at intervals
    plot_interval = n_steps // 5
    for i in range(0, n_steps, plot_interval):
        plt.arrow(x_hist[i], y_hist[i],
                  0.2*np.cos(theta_hist[i]), 0.2*np.sin(theta_hist[i]),
                  head_width=0.05)

    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title('Vehicle Straight Line Motion Test')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.savefig('straight_line_test.png')
    plt.close()


if __name__ == "__main__":
    test_circular_motion()
    test_straight_line_motion()
