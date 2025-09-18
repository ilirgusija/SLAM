"""
This simulator can be used to generate laser reflections in dynamic environments.
Ransalu Senanayake
06-Aug-2018

Step 1: Specify the output folder and file name
Step 2: Specify the environemnt (use an existing or create your own)
Step 3: Specify the lidar parameters such as distance, angle, etc,
Step 4: Draw the robot's path by clicking on various locations on the gui (close to exit) or hard code the pose/s
Output: .csv or carmen file and images

Note: Output file type .csv: column1=time, column2=longitude, column3=latitude, column4=occupied/free
"""
import numpy as np
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from PIL import Image
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from classes.mapping import LidarGridMap
from classes.model import LIDAR, VelocityIntegratorModel
from utils.map import load_obstacles_config

def get_way_points_gui(all_obstacles, area, vehicle_poses=None):
    """
    :param environment: yaml config file
    :param vehicle_poses: vehicle poses
    :return:
    """
    class mouse_events:
        def __init__(self, fig, line):
            self.path_start = False  # If true, capture data
            self.fig = fig
            self.line = line
            self.xs = list(line.get_xdata())
            self.ys = list(line.get_ydata())
            self.orientation = []

        def connect(self):
            self.a = self.fig.canvas.mpl_connect(
                'button_press_event', self.__on_press)
            self.b = self.fig.canvas.mpl_connect(
                'motion_notify_event', self.__on_motion)

        def __on_press(self, event):
            print('You pressed', event.button, event.xdata, event.ydata)
            self.path_start = not self.path_start

        def __on_motion(self, event):
            if self.path_start is True:
                if len(self.orientation) == 0:
                    self.orientation.append(0)
                else:
                    self.orientation.append(
                        np.pi/2 + np.arctan2((self.ys[-1] - event.ydata), (self.xs[-1] - event.xdata)))
                self.xs.append(event.xdata)
                self.ys.append(event.ydata)
                self.line.set_data(self.xs, self.ys)
                self.line.figure.canvas.draw()

    # update obstacles
    all_obstacle_segments = []
    for obs_i in all_obstacles:
        all_obstacle_segments += obs_i.update()

    connected_components = connect_segments(all_obstacle_segments)

    # plot
    pl.close('all')
    fig = pl.figure()  # figsize=(10, 5))  # (9,5)
    ax = fig.add_subplot(111)
    pl.title('Generate waypoints: 1) Click to start. 2) Move the mouse. \n3) Click to stop. 4) lose the gui to exit')
    ax.scatter(connected_components[:, 0], connected_components[:, 1],
               marker='.', c='y', edgecolor='none', alpha=0.2)  # obstacles
    if vehicle_poses is not None:
        pl.plot(vehicle_poses[:, 0], vehicle_poses[:, 1], 'o--', c='m')
    pl.xlim(area[:2])
    pl.ylim(area[2:])

    line, = ax.plot([], [])
    mouse = mouse_events(fig, line)
    mouse.connect()

    pl.show()

    return np.hstack((np.array(mouse.xs)[:, None], np.array(mouse.ys)[:, None], np.array(mouse.orientation)[:, None]))[1:]

####### PLOTTING FUNCTIONS #####################################################

def connect_segments(segments, resolution=0.01):
    """
    This is used to generate lines that make up the cone of measurement of our 
    LIDAR scanner. It takes the location of the sensor and the end point where
    the sensor hits and interpolates those two to generate a dense point
    representation of the line connecting the two. This is used strictly for
    plotting purposes.
    :param segments: start and end points of all segments as ((x1,y1,x1',y1'), (x2,y2,x2',y2'), (x3,y3,x3',y3'), (...))
           step_size : resolution for plotting
    :return: stack of all connected line segments as (X, Y)
    """

    for i, seg_i in enumerate(segments):
        if seg_i[1] == seg_i[3]:  # horizontal segment
            x = np.arange(min(seg_i[0], seg_i[2]), max(
                seg_i[0], seg_i[2]), resolution)
            y = seg_i[1]*np.ones(len(x))
        elif seg_i[0] == seg_i[2]:  # vertical segment
            y = np.arange(min(seg_i[1], seg_i[3]), max(
                seg_i[1], seg_i[3]), resolution)
            x = seg_i[0]*np.ones(len(y))
        else:  # gradient exists
            m = (seg_i[3] - seg_i[1])/(seg_i[2] - seg_i[0])
            c = seg_i[1] - m*seg_i[0]
            x = np.arange(min(seg_i[0], seg_i[2]), max(
                seg_i[0], seg_i[2]), resolution)
            y = m*x + c

        obs = np.vstack((x, y)).T
        connected_segments = obs if i == 0 else np.vstack(
            (connected_segments, obs))

    return connected_segments


def create_movie(image_folder, output_filename, fps=10):
    """Create a movie using Matplotlib's animation module."""

    # Get the list of images in order
    images = [img for img in os.listdir(image_folder)
              if img.endswith(".png") and img.startswith(out_fn)]
    images.sort(key=lambda x: int(x.split('_frame_')[1].split('.png')[0]))

    # Load the images
    frames = [np.array(Image.open(os.path.join(image_folder, img)))
              for img in images]

    # Create the figure and axes
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.axis('off')

    # Create the animation
    ims = [[ax.imshow(frame, animated=True)] for frame in frames]
    ani = animation.ArtistAnimation(fig, ims, interval=1000/fps, blit=True)

    # Save the animation
    ani.save(output_filename, writer='ffmpeg')
    plt.close()
    print(f"Movie saved as {output_filename}")


def plot_step(t, gridmap, sensor, robot_pos, laser_data_xy, dist_theta, area, ofn, robot_poses):
    # Define color map for grid visualization
    # 0.0 = free space (white), 0.5 = unknown (grey), 1.0 = obstacle (red)
    grid_cmap = mcolors.ListedColormap(['white', 'grey', 'red'])
    grid_bounds = [0.0, 0.25, 0.75, 1.0]
    grid_norm = mcolors.BoundaryNorm(grid_bounds, grid_cmap.N)

    # plot
    pl.close('all')
    fig = pl.figure()  # figsize=(9,5)
    ax = fig.add_subplot(111)

    # Extract grid data as a 2D array from the GridMap
    # We'll create a 2D array that matches the GridMap's dimensions and fill it with the GridMap values
    grid_width = gridmap.occupancy_map.width
    grid_height = gridmap.occupancy_map.height
    grid_data = np.zeros((grid_height, grid_width))

    # Efficiently fill the grid_data array with values from the GridMap
    for y_ind in range(grid_height):
        for x_ind in range(grid_width):
            val = gridmap.occupancy_map.get_value_from_xy_index(x_ind, y_ind)
            if val is not None:
                grid_data[y_ind, x_ind] = val.get_float_data()
            else:
                grid_data[y_ind, x_ind] = 0.5  # Unknown if out of bounds

    # Plot the occupancy grid as background using imshow
    # Extent is [left, right, bottom, top] in data coordinates
    img = ax.imshow(grid_data, cmap=grid_cmap, norm=grid_norm,
                    extent=[area[0], area[1], area[2], area[3]],
                    origin='lower', alpha=0.7, zorder=0)
    # print(grid_data)

    # Plot laser beams over the grid
    for i in range(len(laser_data_xy)):
        ax.plot(np.asarray([robot_pos[0], laser_data_xy[i, 0]]),
                np.asarray([robot_pos[1], laser_data_xy[i, 1]]),
                c='b', zorder=1, alpha=0.2)

        # Add endpoints only for actual hits (not max range readings)
        if dist_theta[i] < sensor.max_range:
            ax.scatter(laser_data_xy[i, 0], laser_data_xy[i, 1],
                       marker='o', c='r', zorder=2, edgecolor='none')

    # Plot global origin and robot position
    ax.scatter([0], [0], marker='*', c='k', s=20,
               alpha=1.0, zorder=3, edgecolor='k')
    ax.scatter(robot_pos[0], robot_pos[1],
               marker=(3, 0, robot_pos[2] / np.pi * 180),
               c='k', s=300, alpha=1.0, zorder=3, edgecolor='k')

    # Plot robot path
    ax.plot(robot_poses[:, 0], robot_poses[:, 1], 'k--', zorder=2)

    # Add a colorbar legend
    cbar = fig.colorbar(img, ticks=[0.125, 0.5, 0.875])
    cbar.ax.set_yticklabels(['Free', 'Unknown', 'Obstacle'])

    # Set plot limits and title
    ax.set_xlim([area[0], area[1]])
    ax.set_ylim([area[2], area[3]])
    ax.set_title(f'Occupancy Grid Map - Frame {t}')

    pl.tight_layout()
    pl.savefig(ofn + out_fn + '_frame_{}.png'.format(t))

################################################################################


def run_occupancy_grid_demo(env='toy1', out_fn='toy1_setting1', save_all_data_as_npz=False,
                            n_reflections=360, fov=180, max_laser_distance=12, resolution=0.02):
    """
    :param env: name of the yaml file inside the config folder
    :param out_fn: name of the output folder - create this folder inside the output folder
    :param save_all_data_as_npz: True or False
    :param n_reflections: number of lidar beams in the 2D plane
    :param fov: lidar field of view in degrees
    :param max_laser_distance: maximum lidar distance in meters
    :papram unoccupied_points_per_meter: density of zeros between the robot and laser hit
    """

    # Step 1: output folder and file name pre
    ofn = '../output/occupancy_grid/' + out_fn + '/'

    # Step 2: set up the environment
    #         run the robot - click on various locations on the gui and then close the gui to exit
    all_obstacles, area = load_obstacles_config(environment=env)
    robot_poses = get_way_points_gui(all_obstacles, area)

    # Convert fov from degrees to radians
    fov = fov*np.pi/180

    # Initialize our sensor and map objects
    sensor = LIDAR(max_range=max_laser_distance,
                   fov=fov, n_reflections=n_reflections)
    gridmap = LidarGridMap(*area, resolution=resolution)

    # Loop through each robot pose
    for t in range(len(robot_poses)):
        print('time = {}...'.format(t))

        # update obstacles
        all_obstacle_segments = []
        for obs_i in all_obstacles:
            all_obstacle_segments += obs_i.update()

        # get robot pose
        robot_pos = robot_poses[t, :]

        # Take in observations
        angles, dist_theta = sensor.get_laser_ref(all_obstacle_segments, robot_pos)

        # (x,y) of laser reflections
        laser_data_xy = np.vstack([dist_theta * np.cos(angles), dist_theta *
                                   np.sin(angles)]).T + robot_pos[:2]

        # Create mask for actual obstacle detections (not max range)
        is_obstacle = dist_theta < sensor.max_range

        # Update the gridmap
        ox, oy = laser_data_xy[:, 0], laser_data_xy[:, 1]
        gridmap.update_grid_map(
            robot_pos[0], robot_pos[1], ox, oy, is_obstacle)

        # plot the step
        plot_step(t, gridmap, sensor, robot_pos, laser_data_xy,
                  dist_theta, area, ofn, robot_poses)

    print('Images printed in ' + ofn)
    print(ofn + out_fn + '_robot_poses.npz' + ' created!')
    if save_all_data_as_npz is True:
        print('All data files saved in' + ofn)
    create_movie(ofn, ofn + out_fn + '_simulation.mp4', fps=10)


def run_trajectory_following(waypoints, dt=0.1, max_v=2.0):
    """
    Generate a trajectory by following waypoints using simple control law.

    Args:
        waypoints: List of (x,y) coordinates to visit in sequence
        dt: Time step for simulation
        max_v: Maximum velocity allowed
    Returns:
        trajectory: Nx3 array of (x, y, theta) poses
    """
    # Initialize robot model at first waypoint
    robot = VelocityIntegratorModel(
        waypoints[0][0], waypoints[0][1], dt, max_v)

    # Storage for trajectory with orientation
    trajectory = []
    prev_pos = robot.get_state()
    curr_pos = prev_pos

    # Add initial position with estimated orientation
    if len(waypoints) > 1:
        init_dir = np.array(waypoints[1]) - np.array(waypoints[0])
        init_theta = np.arctan2(init_dir[1], init_dir[0])
        trajectory.append((robot.x[0], robot.x[1], init_theta))
    else:
        trajectory.append((robot.x[0], robot.x[1], 0.0))

    # For each waypoint
    for target in waypoints[1:]:
        target = np.array(target)

        while True:
            # Current state
            prev_pos = curr_pos
            curr_pos = robot.get_state()

            # Compute orientation from movement direction
            if not np.array_equal(prev_pos, curr_pos):
                movement = curr_pos - prev_pos
                theta = np.arctan2(movement[1], movement[0])
            else:
                # Keep previous orientation if not moving
                theta = trajectory[-1][2]

            # Error to target
            error = target - curr_pos
            dist_to_target = np.linalg.norm(error)

            # Break if we're close enough to target
            if dist_to_target < 0.1:
                break

            # Simple proportional control for velocity
            desired_v = error / dist_to_target * max_v
            if dist_to_target < 1.0:
                desired_v *= dist_to_target  # Slow down near target

            # Update robot state
            robot.update(desired_v[0], desired_v[1])

            # Store trajectory point with orientation
            trajectory.append((curr_pos[0], curr_pos[1], theta))

    return np.array(trajectory)


def run_trajectory_occupancy_grid(trajectory, env='toy1', out_fn='trajectory_following',
                                  n_reflections=360, fov=360, max_laser_distance=12, resolution=0.02):
    """
    Generate occupancy grid map from a pre-computed trajectory.

    Args:
        trajectory: Nx3 array of (x, y, theta) poses
        env: Name of environment config file
        out_fn: Output filename prefix
        n_reflections: Number of lidar beams
        fov: Field of view in degrees
        max_laser_distance: Maximum lidar range
    """
    # Step 1: output folder and file name
    ofn = 'output/occupancy_grid/' + out_fn + '/'

    # Step 2: set up the environment
    all_obstacles, area = load_obstacles_config(environment=env)

    # Convert fov from degrees to radians
    fov = fov*np.pi/180

    # Initialize sensor and map objects
    sensor = LIDAR(max_range=max_laser_distance,
                   fov=fov, n_reflections=n_reflections)
    gridmap = LidarGridMap(*area, resolution=resolution)

    # Loop through each pose in the trajectory
    for t in range(len(trajectory)):
        print('time = {}...'.format(t))

        # Update obstacles
        all_obstacle_segments = []
        for obs_i in all_obstacles:
            all_obstacle_segments += obs_i.update()

        # Get robot pose from trajectory
        robot_pos = trajectory[t]

        # Take in observations
        angles, dist_theta = sensor.get_laser_ref(
            all_obstacle_segments, robot_pos)

        # (x,y) of laser reflections
        laser_data_xy = np.vstack([dist_theta * np.cos(angles),
                                  dist_theta * np.sin(angles)]).T + robot_pos[:2]

        # Create mask for actual obstacle detections
        is_obstacle = dist_theta < sensor.max_range

        # Update the gridmap
        ox, oy = laser_data_xy[:, 0], laser_data_xy[:, 1]
        gridmap.update_grid_map(
            robot_pos[0], robot_pos[1], ox, oy, is_obstacle)

        # Plot the step
        plot_step(t, gridmap, sensor, robot_pos, laser_data_xy,
                  dist_theta, area, ofn, trajectory)

    print('Images printed in ' + ofn)
    create_movie(ofn, ofn + out_fn + '_simulation.mp4', fps=10)


def occupancy_grid_from_trajectory(trajectory, env, n_reflections, fov, max_laser_distance, resolution, method='sequential'):
    """
    Generalized function to run occupancy grid mapping from a trajectory using a specified update method.
    Args:
        trajectory: Nx3 array of (x, y, theta) poses
        env: Name of environment config file
        n_reflections: Number of lidar beams
        fov: Field of view in degrees
        max_laser_distance: Maximum lidar range
        resolution: Grid resolution
        method: Update method to use ('sequential', or 'vectorized')
    Returns:
        gridmap: The resulting LidarGridMap
        total_time: Total time taken for the mapping
    """
    import time
    all_obstacles, area = load_obstacles_config(environment=env)
    fov_rad = fov * np.pi / 180
    sensor = LIDAR(max_range=max_laser_distance,
                   fov=fov_rad, n_reflections=n_reflections)
    gridmap = LidarGridMap(*area, resolution=resolution)

    start_time = time.time()
    for t in range(len(trajectory)):
        # Update obstacles
        all_obstacle_segments = []
        for obs_i in all_obstacles:
            all_obstacle_segments += obs_i.update()
        robot_pos = trajectory[t]
        angles, dist_theta = sensor.get_laser_ref(
            all_obstacle_segments, robot_pos)
        obstacles = np.vstack([dist_theta * np.cos(angles), dist_theta * np.sin(angles)]).T + robot_pos[:2]
        is_obstacle = dist_theta < sensor.max_range

        if method == 'vectorized':
            gridmap.update_grid_map_vec(robot_pos, obstacles, is_obstacle)
        else:  # sequential
            gridmap.update_grid_map(robot_pos, obstacles, is_obstacle)

    total_time = time.time() - start_time
    return gridmap, total_time


def run_performance_comparison_trajectory(env='toy1', out_fn='trajectory_test', n_reflections=360, fov=360, max_laser_distance=12, resolution=0.02, batch_sizes=None):
    """
    Compare sequential, parallel, and vectorized occupancy grid updates using a real trajectory.
    """
    import multiprocessing as mp
    from math import ceil

    # Generate trajectory
    waypoints = [
        (90.0, 5.0),
        (20.0, 5.0),
        (20.0, 48.0),
        (90.0, 48.0),
        (90.0, 5.0)
    ]
    trajectory = run_trajectory_following(waypoints)

    # Sequential
    print("\nRunning sequential occupancy grid mapping...")
    gridmap_seq, time_seq = occupancy_grid_from_trajectory(
        trajectory, env, n_reflections, fov, max_laser_distance, resolution,
        method='sequential')
    print(f"Sequential total time: {time_seq:.4f} seconds")

    # Vectorized
    print("\nRunning vectorized occupancy grid mapping...")
    gridmap_vec, time_vec = occupancy_grid_from_trajectory(
        trajectory, env, n_reflections, fov, max_laser_distance, resolution,
        method='vectorized')
    speedup_vec = time_seq / time_vec
    print(f"Vectorized total time: {time_vec:.4f} seconds")
    print(f"Vectorized speedup: {speedup_vec:.2f}x")

    # Determine batch sizes for parallel version
    if batch_sizes is None:
        cpu_count = mp.cpu_count()
        batch_sizes = [
            ceil(n_reflections / (cpu_count * 4)),
            ceil(n_reflections / (cpu_count * 2)),
            ceil(n_reflections / cpu_count),
            n_reflections
        ]

    # Parallel with different batch sizes
    parallel_results = []
    for batch_size in batch_sizes:
        print(
            f"\nRunning parallel occupancy grid mapping (batch_size={batch_size})...")
        gridmap_par, time_par = occupancy_grid_from_trajectory(
            trajectory, env, n_reflections, fov, max_laser_distance, resolution,
            method='parallel')
        speedup = time_seq / time_par
        parallel_results.append((batch_size, time_par, speedup))
        print(
            f"Parallel total time: {time_par:.4f} seconds | Speedup: {speedup:.2f}x")

    # Find best parallel configuration
    best_parallel = min(parallel_results, key=lambda x: x[1])
    print(f"\nBest parallel configuration:")
    print(f"Batch size: {best_parallel[0]}")
    print(f"Time: {best_parallel[1]:.4f} seconds")
    print(f"Speedup: {best_parallel[2]:.2f}x")

    # Compare best methods
    print("\nComparison Summary:")
    print(f"Sequential: {time_seq:.4f} seconds (baseline)")
    print(f"Vectorized: {time_vec:.4f} seconds ({speedup_vec:.2f}x speedup)")
    print(
        f"Best Parallel: {best_parallel[1]:.4f} seconds ({best_parallel[2]:.2f}x speedup)")

    return {
        'sequential': time_seq,
        'vectorized': (time_vec, speedup_vec),
        'parallel': parallel_results
    }


if __name__ == "__main__":
    # file configuration
    env = 'toy1'  # name of the yaml file inside the config folder
    out_fn = 'toy1_setting1'
    save_all_data_as_npz = False

    # robot configuration
    fov = 360  # lidar field of view in degrees
    n_reflections = fov*2  # number of lidar beams in the 2D plane
    max_laser_distance = 12  # maximum lidar distance in meters
    resolution = 0.2  # resolution of the occupancy grid

    # Example usage of trajectory following
    waypoints = [
        (90.0, 5.0),      # Start
        (20.0, 5.0),      # First waypoint
        (20.0, 48.0),     # Second waypoint
        (90.0, 48.0),     # Third waypoint
        (90.0, 5.0)       # Goal
    ]

    # Choose which demo to run
    DEMO_TYPE = "performance"  # "trajectory", "GUI", or "performance"

    if DEMO_TYPE == "GUI":
        run_occupancy_grid_demo(env=env, out_fn=out_fn,
                                save_all_data_as_npz=save_all_data_as_npz,
                                n_reflections=n_reflections, fov=fov,
                                max_laser_distance=max_laser_distance, resolution=resolution)
    elif DEMO_TYPE == "trajectory":
        # Generate trajectory first
        trajectory = run_trajectory_following(waypoints)

        # Then create occupancy grid using the trajectory
        run_trajectory_occupancy_grid(trajectory, env=env, out_fn="trajectory_test",
                                      n_reflections=n_reflections, fov=fov,
                                      max_laser_distance=max_laser_distance, resolution=resolution)
    else:
        # Run performance comparison using trajectory
        results = run_performance_comparison_trajectory(
            env=env,
            out_fn="trajectory_test",
            n_reflections=n_reflections,
            fov=fov,
            max_laser_distance=max_laser_distance,
            resolution=resolution
        )
