import numpy as np
import matplotlib.pyplot as plt
from ..utils.angle import rot_mat_2d


class CarModel:
    def __init__(self, i_x, i_y, i_theta, i_omega, dt, i_v, max_v, w, L):
        self.x = i_x
        self.y = i_y
        self.theta = i_theta
        self.omega = i_omega
        self.v = i_v
        self.max_v = max_v
        self.W = w
        self.L = L
        self.A = np.array([[1.0,  0,   0],
                           [0,  1.0,  0],
                           [0,   0,  1.0]])
        self.dt = dt
        self._calc_vehicle_contour()

    def getB(self):
        return np.array([[np.cos(self.theta)*self.dt, 0],
                         [np.sin(self.theta)*self.dt, 0],
                         [0, self.dt]])

    def update(self, v, omega):
        """
        Update vehicle state using discrete-time approximation of SE(2) dynamics
        Args:
            v (float): Linear velocity
            omega (float): Angular velocity
        """
        # Apply velocity limits
        if v >= self.max_v:
            v = self.max_v
        elif v <= -self.max_v:
            v = -self.max_v

        if abs(omega) < 1e-6:  # Moving in straight line
            # Use exact solution for straight line motion
            self.x += v * np.cos(self.theta) * self.dt
            self.y += v * np.sin(self.theta) * self.dt
        else:  # Circular motion
            # Use exact solution for circular motion
            self.x += v/omega * \
                (np.sin(self.theta + omega*self.dt) - np.sin(self.theta))
            self.y += v/omega * \
                (-np.cos(self.theta + omega*self.dt) + np.cos(self.theta))

        # Update orientation
        self.theta += omega * self.dt
        # Normalize angle to [-π, π]
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))

    def plot(self):
        plt.plot(self.x, self.y, ".b")

        # convert global coordinate
        gx, gy = self.calc_global_contour()
        plt.plot(gx, gy, "--b")

    def _calc_vehicle_contour(self):
        """
        This method initializes the vehicle contour by defining points that
        correspond to each corner of the vehicle. Then it connects these
        corners using the interpolate function.
        """
        self.vc_x = []
        self.vc_y = []

        # Front right corner
        self.vc_x.append(self.L / 2.0)
        self.vc_y.append(self.W / 2.0)

        # Front left corner
        self.vc_x.append(self.L / 2.0)
        self.vc_y.append(-self.W / 2.0)

        # Rear left corner
        self.vc_x.append(-self.L / 2.0)
        self.vc_y.append(-self.W / 2.0)

        # Rear right corner
        self.vc_x.append(-self.L / 2.0)
        self.vc_y.append(self.W / 2.0)

        # Back to front right (closes the shape)
        self.vc_x.append(self.L / 2.0)
        self.vc_y.append(self.W / 2.0)

        self.vc_x, self.vc_y = self._interpolate(self.vc_x, self.vc_y)

    def calc_global_contour(self):
        """
        This method transforms the vehicle's local contour into global
        coordinates.
        """
        gxy = np.stack([self.vc_x, self.vc_y]).T @ rot_mat_2d(self.theta)
        gx = gxy[:, 0] + self.x
        gy = gxy[:, 1] + self.y

        return gx, gy

    @staticmethod
    def _interpolate(x, y):
        rx, ry = [], []
        d_theta = 0.05
        for i in range(len(x) - 1):
            rx.extend([(1.0 - theta) * x[i] + theta * x[i + 1]
                       for theta in np.arange(0.0, 1.0, d_theta)])
            ry.extend([(1.0 - theta) * y[i] + theta * y[i + 1]
                       for theta in np.arange(0.0, 1.0, d_theta)])

        rx.extend([(1.0 - theta) * x[len(x) - 1] + theta * x[1]
                   for theta in np.arange(0.0, 1.0, d_theta)])
        ry.extend([(1.0 - theta) * y[len(y) - 1] + theta * y[1]
                   for theta in np.arange(0.0, 1.0, d_theta)])

        return rx, ry


class VelocityIntegratorModel:
    def __init__(self, i_x, i_y, dt, max_v):
        self.x = np.array([i_x, i_y], dtype=float)
        self.dt = dt
        self.max_v = max_v

    def update(self, v):
        speed = np.linalg.norm(v)

        # clip to max velocity
        if speed > self.max_v:
            v = v / speed * self.max_v

        self.x += v * self.dt
        return self.x

    def simulate(self, x, u):
        speed = np.linalg.norm(u)
        if speed > self.max_v:
            u = u / speed * self.max_v
        # x is Nx2 and u is 1x2, so x + u * self.dt is Nx2
        x_star = x + u * self.dt
        return x_star

    def plot(self):
        plt.plot(self.x[0], self.x[1], ".b")

    def get_state(self):
        return self.x


class VIM_with_Noise(VelocityIntegratorModel):
    def __init__(self, i_x, i_y, dt, max_v, noise_std):
        super().__init__(i_x, i_y, dt, max_v)
        self.noise_std = noise_std

    def update(self, v):
        x = super().update(v)
        x += np.random.normal(0, self.noise_std, size=x.shape)
        return x

    def simulate(self, v):
        x = super().simulate(v)
        x += np.random.normal(0, self.noise_std, size=x.shape)
        return x


class LIDAR:
    def __init__(self, fov=2*np.pi, r_max=100, B=360):
        """
        :param fov: sight of the robot - typically 2*pi or pi
        :param B: number of beams - resolution=fov/B
        :param r_max: max distance the robot can see. If no obstacle, laser end point = max_dist
        """
        self.r_max = r_max
        self.fov = fov
        self.B = B
        self.resolution = fov/B
        self.angles = np.linspace(0, self.fov*180/np.pi-1, self.B)*np.pi/180

    # OPTIMIZE vectorize this
    def get_intersection(self, a1, a2, b1, b2):
        """
        :param a1: (x1,y1) line segment 1 - starting position
        :param a2: (x1',y1') line segment 1 - ending position
        :param b1: (x2,y2) line segment 2 - starting position
        :param b2: (x2',y2') line segment 2 - ending position
        :return: point of intersection, if intersect; None, if do not intersect
        #adopted from https://github.com/LinguList/TreBor/blob/master/polygon.py
        """
        def perp(a):
            b = np.empty_like(a)
            b[0] = -a[1]
            b[1] = a[0]
            return b

        da = a2-a1
        db = b2-b1
        dp = a1-b1
        dap = perp(da)
        denom = np.dot(dap, db)
        num = np.dot(dap, dp)

        # TODO: check logic here
        # Check for zero denominator (parallel lines)
        if abs(denom) < 1e-10:
            return None

        intersct = np.array((num/denom)*db + b1)

        delta = 1e-3
        condx_a = min(a1[0], a2[0])-delta <= intersct[0] and max(a1[0],
                                                                 a2[0])+delta >= intersct[0]  # within line segment a1_x-a2_x
        condx_b = min(b1[0], b2[0])-delta <= intersct[0] and max(b1[0],
                                                                 b2[0])+delta >= intersct[0]  # within line segment b1_x-b2_x
        condy_a = min(a1[1], a2[1])-delta <= intersct[1] and max(a1[1],
                                                                 a2[1])+delta >= intersct[1]  # within line segment a1_y-b1_y
        condy_b = min(b1[1], b2[1])-delta <= intersct[1] and max(b1[1],
                                                                 b2[1])+delta >= intersct[1]  # within line segment a2_y-b2_y
        if not (condx_a and condy_a and condx_b and condy_b):
            intersct = None  # line segments do not intercept i.e. interception is away from from the line segments

        return intersct

    # OPTIMIZE: vectorize this
    def get_laser_ref(self, segments, robot_poses=np.array([[0.0, 0.0]])):
        """
        :param segments: start and end points of all segments as ((x1,y1,x1',y1'),
                                (x2,y2,x2',y2'), (x3,y3,x3',y3'), (...))
               robot_pose: robot's pose in the global coordinate system
        :return: m_nxB array indicating the laser end point
        """
        if robot_poses.ndim == 1:
            robot_poses = robot_poses[np.newaxis, :]

        # set all laser reflections to r_max
        dist_theta = self.r_max*np.ones((robot_poses.shape[0], self.B))

        for seg_i in segments:
            # starting and ending points of each segment
            xy_i_start, xy_i_end = np.array(seg_i[:2]), np.array(seg_i[2:])

            for i, xy_robot in enumerate(robot_poses):
                for j, theta in enumerate(self.angles):
                    # max possible distance
                    xy_ij_max = xy_robot + np.array([self.r_max*np.cos(theta),
                                                    self.r_max*np.sin(theta)])
                    intersection = self.get_intersection(
                        xy_i_start, xy_i_end, xy_robot, xy_ij_max)

                    # if the line segments intersect
                    if intersection is not None:
                        r = np.sqrt(
                            np.sum((intersection-xy_robot)**2))  # radius

                        if r < dist_theta[i, j]:
                            dist_theta[i, j] = r

        return dist_theta

    def get_filled_txy(self, dist_theta, angles, robot_pos,
                       unoccupied_points_per_meter=0.1, margin=0.1):
        """
        :param dist_theta: lidar hit distance
        :param angles: angles of each sensor
        :param robot_pos: robot pose
        :param unoccupied_points_per_meter: in-fill density
        :param margin: in-fill density of free points
        :return: (points, labels) - where 0 label is for free points and 1 
            label is for hits
        """

        laser_data_xy = np.vstack([dist_theta * np.cos(angles), dist_theta *
                                   np.sin(angles)]).T + robot_pos[:2]

        for i, _ in enumerate(angles):
            dist = dist_theta[i]
            laser_endpoint = laser_data_xy[i, :]

            # parametric filling
            para = np.sort(np.random.random(np.int16(dist * unoccupied_points_per_meter))
                           * (1 - 2 * margin) + margin)[:, np.newaxis]  # TODO: Uniform[0.05, 0.95]
            # y = <x0, y0, z0> + para <x, y, z>; para \in [0, 1]
            points_scan_i = robot_pos[:2] + para * \
                (laser_endpoint - robot_pos[:2])

            if i == 0:  # first data point
                if dist >= self.r_max:  # there's no laser reflection
                    points = points_scan_i
                    labels = np.zeros((points_scan_i.shape[0], 1))
                else:  # append the arrays with laser end-point
                    points = np.vstack((points_scan_i, laser_endpoint))
                    labels = np.vstack(
                        (np.zeros((points_scan_i.shape[0], 1)), np.array([1])[:, np.newaxis]))
            else:
                if dist >= self.r_max:  # there's no laser reflection
                    points = np.vstack((points, points_scan_i))
                    labels = np.vstack(
                        (labels, np.zeros((points_scan_i.shape[0], 1))))
                else:  # append the arrays with laser end-point
                    points = np.vstack(
                        (points, np.vstack((points_scan_i, laser_endpoint))))
                    labels = np.vstack((labels, np.vstack(
                        (np.zeros((points_scan_i.shape[0], 1)), np.array([1])[:, np.newaxis]))))

        return np.hstack((points, labels))


class LIDAR_with_Noise(LIDAR):
    def __init__(self, fov, r_max, B, noise_std):
        super().__init__(fov, r_max, B)
        self.noise_std = noise_std

    def get_laser_ref(self, segments, robot_pose=np.array([0.0, 0.0])):
        angles, dist_theta = super().get_laser_ref(segments, robot_pose)
        dist_theta += np.random.normal(0,
                                       self.noise_std, size=dist_theta.shape)
        return angles, dist_theta
