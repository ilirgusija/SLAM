# Use CuPy backend (drop-in replacement for NumPy)
from ..utils.array_backend import np
import matplotlib.pyplot as plt
# Use CuPy's ndimage if available, otherwise scipy's
try:
    from cupyx.scipy import ndimage
except ImportError:
    from scipy import ndimage
from ..utils.angle import rot_mat_2d
from .obstacle import Obstacle
from .mapping import LidarGridMapVec


class SingleIntegratorModel:
    def __init__(self, i_x, i_y, dt, max_v):
        self.x = np.array([i_x, i_y], dtype=float)
        self.dt = dt
        self.max_v = max_v
        self._state_bounds = None

    def set_state_bounds(self, bounds):
        bounds = np.asarray(bounds, dtype=float)
        if bounds.shape != (2, 2):
            raise ValueError("SingleIntegratorModel bounds must be shape (2, 2)")
        self._state_bounds = bounds
        self.lower = self._state_bounds[:, 0]
        self.upper = self._state_bounds[:, 1]

    def _project_state(self, x):
        if self._state_bounds is None:
            return x

        return np.clip(x, self.lower, self.upper)

    def _nominal_step(self, x, u):
        return x + u * self.dt

    def update(self, v):
        self.x = self._project_state(self._nominal_step(self.x, v))
        return self.x

    def f_bar(self, x, u):
        x = np.asarray(x, dtype=float)
        x_star = self._nominal_step(x, u)
        return self._project_state(x_star)

    def f(self, x, u, w):
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        x_nom = self._nominal_step(x, u)
        return self._project_state(x_nom + w)

class DoubleIntegratorModel:
    def __init__(self, p_x, p_y, v_x, v_y, dt, max_a):
        self.x = np.array([p_x, p_y, v_x, v_y], dtype=float)
        self.dt = dt
        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.B = np.array([[0.5 * dt**2, 0],
                           [0, 0.5 * dt**2],
                           [dt, 0],
                           [0, dt]])
        self.Q_t = np.array([[(dt**3) / 3, 0, (dt**2) / 2, 0],
                             [0, (dt**3) / 3, 0, (dt**2) / 2],
                             [(dt**2) / 2, 0, dt, 0],
                             [0, (dt**2) / 2, 0, dt]])
        self.max_a = max_a
        self._state_bounds = None

    def set_state_bounds(self, bounds):
        bounds = np.asarray(bounds, dtype=float)
        if bounds.shape != (4, 2):
            raise ValueError("DoubleIntegratorModel bounds must be shape (4, 2)")
        self._state_bounds = bounds
        self.lower = self._state_bounds[:, 0]
        self.upper = self._state_bounds[:, 1]

    def _project_state(self, x):
        if self._state_bounds is None:
            return x

        return np.clip(x, self.lower, self.upper)

    def _nominal_step(self, x, u):
        return x @ self.A.T + self.B @ u

    def f_bar(self, x, u):
        x = np.asarray(x, dtype=float)
        x_nom = self._nominal_step(x, u)
        return self._project_state(x_nom)

    def f(self, x, u, w):
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        x_nom = self._nominal_step(x, u)
        return self._project_state(x_nom + w)

class LIDAR:
    def __init__(self, fov=2 * np.pi, r_max=100, B=360):
        """
        :param fov: sensor field of view. Accepts radians (default) or degrees (> 2π).
        :param B: number of beams - evenly spaced, starting at 0 rad (along +x), CCW.
        :param r_max: max distance the robot can see. If no obstacle, laser end point = max_dist
        """
        self.r_max = r_max
        # Normalize FOV to radians if user passed degrees (common when using 360)
        fov_rad = np.deg2rad(fov) if fov > 2 * np.pi + 1e-6 else float(fov)
        self.fov = fov_rad
        self.B = int(B)
        self.resolution = self.fov / self.B
        # Evenly spaced angles in [0, fov), endpoint excluded to avoid duplicate at fov
        self.angles = np.linspace(0.0, self.fov, self.B, endpoint=False)

    def get_intersection(self, a1, a2, b1, b2):
        """
        Vectorized line segment intersection computation.

        :param a1: (x1,y1) or (N, 2) line segment 1 - starting position
        :param a2: (x1',y1') or (N, 2) line segment 1 - ending position
        :param b1: (x2,y2) or (N, 2) line segment 2 - starting position
        :param b2: (x2',y2') or (N, 2) line segment 2 - ending position
        :return: point of intersection(s), if intersect; None or array of Nones if do not intersect
        #adopted from https://github.com/LinguList/TreBor/blob/master/polygon.py
        """
        # Convert to arrays and ensure 2D
        a1 = np.asarray(a1)
        a2 = np.asarray(a2)
        b1 = np.asarray(b1)
        b2 = np.asarray(b2)

        single = a1.ndim == 1
        if single:
            a1 = a1[np.newaxis, :]
            a2 = a2[np.newaxis, :]
            b1 = b1[np.newaxis, :]
            b2 = b2[np.newaxis, :]

        # Vectorized perpendicular: perp([x, y]) = [-y, x]
        da = a2 - a1  # (N, 2)
        db = b2 - b1  # (N, 2)
        dp = a1 - b1  # (N, 2)

        dap = np.stack([-da[:, 1], da[:, 0]], axis=1)  # (N, 2)
        denom = np.sum(dap * db, axis=1)  # (N,)
        num = np.sum(dap * dp, axis=1)  # (N,)

        # Check for parallel lines (zero denominator)
        parallel_mask = np.abs(denom) < 1e-10

        # Compute intersections (avoid division by zero)
        t = np.where(parallel_mask, np.nan, num / denom)  # (N,)
        intersections = b1 + t[:, np.newaxis] * db  # (N, 2)

        # Check if intersections are within line segments
        delta = 1e-3
        a_min_x = np.minimum(a1[:, 0], a2[:, 0])  # (N,)
        a_max_x = np.maximum(a1[:, 0], a2[:, 0])
        a_min_y = np.minimum(a1[:, 1], a2[:, 1])
        a_max_y = np.maximum(a1[:, 1], a2[:, 1])

        b_min_x = np.minimum(b1[:, 0], b2[:, 0])
        b_max_x = np.maximum(b1[:, 0], b2[:, 0])
        b_min_y = np.minimum(b1[:, 1], b2[:, 1])
        b_max_y = np.maximum(b1[:, 1], b2[:, 1])

        condx_a = (a_min_x - delta <= intersections[:, 0]) & (intersections[:, 0] <= a_max_x + delta)
        condx_b = (b_min_x - delta <= intersections[:, 0]) & (intersections[:, 0] <= b_max_x + delta)
        condy_a = (a_min_y - delta <= intersections[:, 1]) & (intersections[:, 1] <= a_max_y + delta)
        condy_b = (b_min_y - delta <= intersections[:, 1]) & (intersections[:, 1] <= b_max_y + delta)

        valid_mask = ~parallel_mask & condx_a & condy_a & condx_b & condy_b

        # Set invalid intersections to NaN
        intersections[~valid_mask] = np.nan

        if single:
            if valid_mask[0]:
                return intersections[0]
            else:
                return None
        else:
            # Return array with NaN for invalid intersections
            return intersections

    def get_laser_ref_old(self, segments, robot_poses=None):
        """
        Original version with loop over segments (kept for comparison/fallback).
        Vectorized laser reflection computation for multiple robot poses and angles.

        :param segments: start and end points of all segments as ((x1,y1,x1',y1'),
                                (x2,y2,x2',y2'), (x3,y3,x3',y3'), (...))
               robot_poses: robot's pose(s) in the global coordinate system (N, 2) or (2,)
        :return: (N, B) array indicating the distance traveled by each laser beam
        """
        if robot_poses is None:
            robot_poses = np.array([[0.0, 0.0]])
        if robot_poses.ndim == 1:
            robot_poses = robot_poses[np.newaxis, :]

        N = robot_poses.shape[0]

        # Initialize all laser reflections to r_max
        dist_theta = self.r_max * np.ones((N, self.B))

        # Precompute ray endpoints for all poses and angles
        # robot_poses: (N, 2), angles: (B,)
        # ray_dirs: (N, B, 2) = cos/sin for each pose-angle combination
        cos_angles = np.cos(self.angles)  # (B,)
        sin_angles = np.sin(self.angles)  # (B,)
        ray_dirs = np.stack([cos_angles, sin_angles], axis=1)  # (B, 2)

        # Expand for broadcasting: (N, 1, 2) + (1, B, 2) * r_max = (N, B, 2)
        robot_poses_expanded = robot_poses[:, np.newaxis, :]  # (N, 1, 2)
        ray_dirs_expanded = ray_dirs[np.newaxis, :, :]  # (1, B, 2)
        ray_endpoints = robot_poses_expanded + self.r_max * ray_dirs_expanded  # (N, B, 2)

        # Process each segment
        for seg_i in segments:
            xy_i_start = np.array(seg_i[:2])  # (2,)
            xy_i_end = np.array(seg_i[2:])  # (2,)

            # Expand segment endpoints for broadcasting: reshape to (1, 2) for proper broadcasting
            seg_start = xy_i_start[np.newaxis, :]  # (1, 2)
            seg_end = xy_i_end[np.newaxis, :]  # (1, 2)

            # Expand robot poses and ray endpoints: (N, B, 2)
            # Broadcast robot_poses_expanded from (N, 1, 2) to (N, B, 2)
            robot_expanded = np.broadcast_to(robot_poses_expanded, (N, self.B, 2))  # (N, B, 2)
            ray_expanded = ray_endpoints  # (N, B, 2)

            # Reshape for vectorized intersection: flatten (N, B) combinations
            robot_flat = robot_expanded.reshape(-1, 2)  # (N*B, 2)
            ray_flat = ray_expanded.reshape(-1, 2)  # (N*B, 2)
            seg_start_flat = np.broadcast_to(seg_start, (N * self.B, 2))  # (N*B, 2)
            seg_end_flat = np.broadcast_to(seg_end, (N * self.B, 2))  # (N*B, 2)

            # Compute intersections for all (pose, angle, segment) combinations
            intersections = self.get_intersection(seg_start_flat, seg_end_flat,
                                                  robot_flat, ray_flat)  # (N*B, 2) or None

            if intersections is not None:
                # Check for valid intersections (not NaN)
                valid = ~np.isnan(intersections).any(axis=1)  # (N*B,)

                if valid.any():
                    # Compute distances for valid intersections
                    distances = np.sqrt(np.sum((intersections - robot_flat)**2, axis=1))  # (N*B,)

                    # Reshape back to (N, B) and update minimum distances
                    distances_2d = distances.reshape(N, self.B)  # (N, B)
                    valid_2d = valid.reshape(N, self.B)  # (N, B)

                    # Update dist_theta where intersections are valid and closer than current
                    update_mask = valid_2d & (distances_2d < dist_theta)
                    dist_theta[update_mask] = distances_2d[update_mask]

        return dist_theta

    def get_laser_ref(self, segments, robot_poses=None):
        """
        Vectorized laser reflection computation for multiple robot poses and angles.
        Fully vectorized version that processes all segments at once.

        :param segments: start and end points of all segments as ((x1,y1,x1',y1'),
                                (x2,y2,x2',y2'), (x3,y3,x3',y3'), (...))
               robot_poses: robot's pose(s) in the global coordinate system (N, 2) or (2,)
        :return: (N, B) array indicating the distance traveled by each laser beam
        """
        if robot_poses is None:
            robot_poses = np.array([[0.0, 0.0]])
        if robot_poses.ndim == 1:
            robot_poses = robot_poses[np.newaxis, :]

        N = robot_poses.shape[0]
        S = len(segments)  # Number of segments

        # Initialize all laser reflections to r_max
        dist_theta = self.r_max * np.ones((N, self.B))

        # Early return if no segments (no obstacles) - all rays hit max range
        if S == 0:
            return dist_theta

        # Precompute ray endpoints for all poses and angles
        # robot_poses: (N, 2), angles: (B,)
        # ray_dirs: (N, B, 2) = cos/sin for each pose-angle combination
        cos_angles = np.cos(self.angles)  # (B,)
        sin_angles = np.sin(self.angles)  # (B,)
        ray_dirs = np.stack([cos_angles, sin_angles], axis=1)  # (B, 2)

        # Expand for broadcasting: (N, 1, 2) + (1, B, 2) * r_max = (N, B, 2)
        robot_poses_expanded = robot_poses[:, np.newaxis, :]  # (N, 1, 2)
        ray_dirs_expanded = ray_dirs[np.newaxis, :, :]  # (1, B, 2)
        ray_endpoints = robot_poses_expanded + self.r_max * ray_dirs_expanded  # (N, B, 2)

        # Convert all segments to arrays upfront: (S, 4) -> (S, 2, 2)
        # Now we know S > 0, so this will create a proper 2D array
        segments_array = np.array(segments)  # (S, 4)
        seg_starts = segments_array[:, :2]  # (S, 2) - start points
        seg_ends = segments_array[:, 2:]  # (S, 2) - end points

        # Expand segments to (S, N, B, 2) for all combinations
        # seg_starts: (S, 1, 1, 2) -> (S, N, B, 2)
        seg_starts_expanded = seg_starts[:, np.newaxis, np.newaxis, :]  # (S, 1, 1, 2)
        seg_ends_expanded = seg_ends[:, np.newaxis, np.newaxis, :]  # (S, 1, 1, 2)

        # Expand robot poses: (N, 1, 2) -> reshape to (1, N, 1, 2) -> broadcast to (S, N, B, 2)
        robot_expanded = robot_poses_expanded.reshape(1, N, 1, 2)  # (1, N, 1, 2)
        # ray_endpoints is already (N, B, 2), expand to (1, N, B, 2)
        ray_expanded = ray_endpoints[np.newaxis, :, :, :]  # (1, N, B, 2)

        # Broadcast to (S, N, B, 2) for all combinations
        seg_starts_all = np.broadcast_to(seg_starts_expanded, (S, N, self.B, 2))  # (S, N, B, 2)
        seg_ends_all = np.broadcast_to(seg_ends_expanded, (S, N, self.B, 2))  # (S, N, B, 2)
        robot_all = np.broadcast_to(robot_expanded, (S, N, self.B, 2))  # (S, N, B, 2)
        ray_all = np.broadcast_to(ray_expanded, (S, N, self.B, 2))  # (S, N, B, 2)

        # Flatten to (S*N*B, 2) for vectorized intersection computation
        seg_starts_flat = seg_starts_all.reshape(-1, 2)  # (S*N*B, 2)
        seg_ends_flat = seg_ends_all.reshape(-1, 2)  # (S*N*B, 2)
        robot_flat = robot_all.reshape(-1, 2)  # (S*N*B, 2)
        ray_flat = ray_all.reshape(-1, 2)  # (S*N*B, 2)

        # Compute all intersections at once: (S*N*B, 2)
        intersections = self.get_intersection(seg_starts_flat, seg_ends_flat,
                                              robot_flat, ray_flat)  # (S*N*B, 2)

        if intersections is not None:
            # Check for valid intersections (not NaN)
            valid = ~np.isnan(intersections).any(axis=1)  # (S*N*B,)

            if valid.any():
                # Compute distances for valid intersections
                distances = np.sqrt(np.sum((intersections - robot_flat)**2, axis=1))  # (S*N*B,)

                # Reshape to (S, N, B) to find minimum across segments
                distances_3d = distances.reshape(S, N, self.B)  # (S, N, B)
                valid_3d = valid.reshape(S, N, self.B)  # (S, N, B)

                # Set invalid intersections to infinity so they don't affect minimum
                distances_3d[~valid_3d] = np.inf

                # Find minimum distance across all segments for each (pose, angle) pair
                min_distances = np.min(distances_3d, axis=0)  # (N, B)

                # Update dist_theta where we found valid intersections closer than r_max
                update_mask = (min_distances < dist_theta) & (min_distances < np.inf)
                dist_theta[update_mask] = min_distances[update_mask]

        return dist_theta

    def get_laser_ref_per_segment(self, segments, robot_poses=None):
        """
        Vectorized laser reflection that returns per-segment distances.
        Useful for batched processing where we need to group segments by map.

        Returns:
            distances_3d: (S, N, B) array of distances for each segment
            valid_3d: (S, N, B) boolean array indicating valid intersections
        """
        if robot_poses is None:
            robot_poses = np.array([[0.0, 0.0]])
        if robot_poses.ndim == 1:
            robot_poses = robot_poses[np.newaxis, :]

        N = robot_poses.shape[0]
        S = len(segments)

        if S == 0:
            return np.full((0, N, self.B), np.inf), np.zeros((0, N, self.B), dtype=bool)

        # Same computation as get_laser_ref but return per-segment results
        cos_angles = np.cos(self.angles)
        sin_angles = np.sin(self.angles)
        ray_dirs = np.stack([cos_angles, sin_angles], axis=1)

        robot_poses_expanded = robot_poses[:, np.newaxis, :]
        ray_dirs_expanded = ray_dirs[np.newaxis, :, :]
        ray_endpoints = robot_poses_expanded + self.r_max * ray_dirs_expanded

        segments_array = np.array(segments)
        seg_starts = segments_array[:, :2]
        seg_ends = segments_array[:, 2:]

        seg_starts_expanded = seg_starts[:, np.newaxis, np.newaxis, :]
        seg_ends_expanded = seg_ends[:, np.newaxis, np.newaxis, :]
        robot_expanded = robot_poses_expanded.reshape(1, N, 1, 2)
        ray_expanded = ray_endpoints[np.newaxis, :, :, :]

        seg_starts_all = np.broadcast_to(seg_starts_expanded, (S, N, self.B, 2))
        seg_ends_all = np.broadcast_to(seg_ends_expanded, (S, N, self.B, 2))
        robot_all = np.broadcast_to(robot_expanded, (S, N, self.B, 2))
        ray_all = np.broadcast_to(ray_expanded, (S, N, self.B, 2))

        seg_starts_flat = seg_starts_all.reshape(-1, 2)
        seg_ends_flat = seg_ends_all.reshape(-1, 2)
        robot_flat = robot_all.reshape(-1, 2)
        ray_flat = ray_all.reshape(-1, 2)

        intersections = self.get_intersection(seg_starts_flat, seg_ends_flat, robot_flat, ray_flat)

        if intersections is not None:
            valid = ~np.isnan(intersections).any(axis=1)
            distances = np.sqrt(np.sum((intersections - robot_flat)**2, axis=1))
            distances_3d = distances.reshape(S, N, self.B)
            valid_3d = valid.reshape(S, N, self.B)
            distances_3d[~valid_3d] = np.inf
            return distances_3d, valid_3d
        else:
            return np.full((S, N, self.B), np.inf), np.zeros((S, N, self.B), dtype=bool)

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

    def g_bar(self, X, m, map_obj: LidarGridMapVec):
        """
        Deterministic observation model - ray casting to determine distances to obstacles.

        Args:
            X: Array of robot positions (m_n, 2)
            m: Occupancy grid (H, W)
            map_obj: Map object with occupancy_map attribute containing left_lower, right_upper

        Returns:
            y_star: Array of distances (m_n, B)
        """
        X = X[np.newaxis, :] if X.ndim == 1 else X
        all_obstacle_segments = self.get_obstacles_from_map(m, map_obj)
        y_star = self.get_laser_ref(all_obstacle_segments, X)
        return y_star if y_star.ndim == 2 else y_star[np.newaxis, :]

    def g_bar_localization(self, X, obstacle_segments):
        """
        Deterministic observation model for localization - ray casting using pre-computed obstacle segments.

        For localization, the map is known and obstacle segments are pre-computed.
        This avoids recomputing obstacle segments from the map on every call.

        Args:
            X: Array of robot positions (m_n, 2)
            obstacle_segments: Pre-computed list of obstacle segments from the known map

        Returns:
            y_star: Array of distances (m_n, B)
        """
        X = X[np.newaxis, :] if X.ndim == 1 else X
        y_star = self.get_laser_ref(obstacle_segments, X)
        return y_star if y_star.ndim == 2 else y_star[np.newaxis, :]

    def get_laser_ref_batched(self, segments_list, segment_map_indices, robot_poses):
        """
        Batched laser reflection computation that processes segments from multiple maps.
        FULLY VECTORIZED - NO LOOPS. Processes all segments together, then groups by map.

        Args:
            segments_list: List of all segments from all maps
            segment_map_indices: Array (total_segments,) indicating which map each segment belongs to
            robot_poses: Robot poses (m_n, 2)

        Returns:
            y_star_all: (m_n, len_M, B) array with distances for each map
        """
        if robot_poses.ndim == 1:
            robot_poses = robot_poses[np.newaxis, :]

        m_n = robot_poses.shape[0]
        len_M = int(segment_map_indices.max() + 1) if len(segment_map_indices) > 0 else 0
        B = self.B

        # Initialize output to r_max
        y_star_all = np.full((m_n, len_M, B), self.r_max, dtype=np.float64)

        if len(segments_list) == 0:
            return y_star_all

        # Convert to GPU array
        segment_map_indices = np.asarray(segment_map_indices)

        # Process ALL segments together - fully vectorized, no loops
        distances_all, _ = self.get_laser_ref_per_segment(segments_list, robot_poses)
        # distances_all: (S, m_n, B) where S is total number of segments

        # Group by map and take minimum per map using FULLY VECTORIZED operations - NO LOOPS
        # Sort segments by map index to enable vectorized grouping
        sort_indices = np.argsort(segment_map_indices)
        sorted_distances = distances_all[sort_indices, :, :]  # (S, m_n, B)
        sorted_map_indices = segment_map_indices[sort_indices]  # (S,)

        # Find boundaries between maps using vectorized operations
        # Convert Python bool to backend array type (CuPy if using GPU)
        first_true = np.array([True], dtype=bool)
        map_diffs = sorted_map_indices[1:] != sorted_map_indices[:-1]
        map_changes = np.concatenate((first_true, map_diffs))
        segment_boundaries = np.where(map_changes)[0]  # Indices where map changes
        # Convert Python int to backend array type
        end_idx = np.array([len(sorted_map_indices)], dtype=np.int64)
        segment_boundaries = np.concatenate((segment_boundaries, end_idx))  # Add end

        # Compute minimum per map using vectorized reduceat-style operations
        # Process all maps: extract segments per map and reduce
        unique_maps, map_start_indices = np.unique(sorted_map_indices, return_index=True)
        end_idx_array = np.array([len(sorted_map_indices)], dtype=map_start_indices.dtype)
        map_end_indices = np.concatenate((map_start_indices[1:], end_idx_array))

        # Vectorized reduction: compute minimum per map
        # Reshape to (S, m_n*B) for easier manipulation
        sorted_distances_2d = sorted_distances.reshape(len(sorted_distances), m_n * B)  # (S, m_n*B)

        # TODO: Replace with np.maximum.reduceat once CuPy adds support for reduceat:
        # negated_distances = -sorted_distances_2d  # (S, m_n*B)
        # max_negated = np.maximum.reduceat(negated_distances, map_start_indices, axis=0)  # (len_M, m_n*B)
        # min_distances_2d = -max_negated  # (len_M, m_n*B) - this is minimum per map
        # min_distances_2d = np.minimum(min_distances_2d, self.r_max)  # Clamp to r_max
        # CuPy issue: https://github.com/cupy/cupy/issues (reduceat not yet implemented for maximum/minimum)
        # Workaround: Use vectorized slicing + np.min per map
        # Note: np.min() on slices is fully vectorized on GPU - the loop is only O(len_M) for organization

        # Preallocate output array
        min_distances_2d = np.full((len(unique_maps), m_n * B), self.r_max, dtype=np.float64)

        # Process each map: extract segments and compute minimum
        # The computation (np.min) is fully vectorized - loop is only for organizing variable-length groups
        for i, (start_idx, end_idx) in enumerate(zip(map_start_indices, map_end_indices)):
            # Extract this map's segments: (num_segments_map, m_n*B)
            map_segments_2d = sorted_distances_2d[start_idx:end_idx, :]
            # Compute minimum across segments: (m_n*B,) - fully vectorized GPU operation
            min_distances_2d[i, :] = np.min(map_segments_2d, axis=0)

        # Clamp to r_max
        min_distances_2d = np.minimum(min_distances_2d, self.r_max)  # (len_M, m_n*B)

        # Reshape back and transpose to (m_n, len_M, B)
        map_results = min_distances_2d.reshape(len(unique_maps), m_n, B)  # (len_M, m_n, B)
        y_star_all = map_results.transpose(1, 0, 2)  # (m_n, len_M, B)

        # Reorder to match original map order (unique_maps might not be 0..len_M-1)
        # Use advanced indexing to assign to correct map positions
        y_star_all_reordered = np.full((m_n, len_M, B), self.r_max, dtype=np.float64)
        y_star_all_reordered[:, unique_maps, :] = y_star_all

        return y_star_all_reordered

    def g_bar_batched(self, X, M, map_obj: LidarGridMapVec):
        """
        Batched deterministic observation model - ray casting for multiple maps simultaneously.

        Processes all maps using vectorized operations. Uses list comprehension for
        obstacle extraction (necessary due to connected components algorithm), but all
        ray casting operations are fully vectorized on GPU.

        Args:
            X: Array of robot positions (m_n, 2)
            M: Array of occupancy grids (len_M, H, W)
            map_obj: Map object with occupancy_map attribute

        Returns:
            y_star_all: Array of distances (m_n, len_M, B)
        """
        X = X[np.newaxis, :] if X.ndim == 1 else X
        len_M = M.shape[0]

        # Extract obstacles for all maps - list comprehension is necessary here
        # because connected components labeling is inherently per-map
        # But this keeps data on GPU (M[map_idx] is a slice, stays on GPU)
        map_segments_list = [self.get_obstacles_from_map(M[map_idx], map_obj) for map_idx in range(len_M)]

        # Flatten segments with map indices - use list operations but convert to GPU arrays
        all_segments = []
        segment_map_indices_list = []

        # Build segment list and indices - this is necessary for grouping
        for map_idx, segments in enumerate(map_segments_list):
            all_segments.extend(segments)
            segment_map_indices_list.extend([map_idx] * len(segments))

        # Convert to GPU array for vectorized operations
        segment_map_indices = np.array(
            segment_map_indices_list) if segment_map_indices_list else np.array([], dtype=np.int32)

        # Use batched laser reflection computation
        return self.get_laser_ref_batched(all_segments, segment_map_indices, X)

    def g(self, X, m, v, map_obj: LidarGridMapVec):
        """
        Stochastic observation model: g(x, m, v) = g_bar(x, m) + v

        Args:
            X: Array of robot positions (m_n, 2)
            m: Occupancy grid (H, W)
            v: Observation noise (m_n, B)
            map_obj: Map object with occupancy_map attribute

        Returns:
            y_noisy: Noisy observations (m_n, B)
        """
        v = v[np.newaxis, :] if X.ndim == 1 else v
        y_ideal = self.g_bar(X, m, map_obj)
        y_noisy = y_ideal + v
        return y_noisy

    def get_obstacles_from_map(self, m, map_obj: LidarGridMapVec):
        """
        Convert occupancy grid to obstacle segments.

        Args:
            m: Occupancy grid (H, W)
            map_obj: Map object with occupancy_map attribute

        Returns:
            List of obstacle segments
        """
        # Find connected components of occupied cells
        connected_components = self._find_connected_components(m)

        all_obstacle_segments = []

        # Create obstacles for each connected component
        for component in connected_components:
            if len(component) > 0:
                obstacle_segments = self._create_obstacle_from_component(
                    component, m.shape, map_obj)
                all_obstacle_segments.extend(obstacle_segments)
        return all_obstacle_segments

    def _find_connected_components(self, m):
        """
        Find connected components of occupied cells (value = 1) in the occupancy grid.
        Uses 4-connectivity (cells sharing a side are connected) with ndimage.label (CuPy required).

        Args:
            m: (H, W) array of occupancy grid

        Returns:
            List of connected components, where each component is a list of (i, j) coordinates

        Raises:
            RuntimeError: If CuPy operations fail. CuPy is required for this operation.
        """
        # Convert to array if needed
        m = np.asarray(m)

        # Label connected components (CuPy's ndimage.label)
        # Returns (labeled_array, num_features)
        result = ndimage.label(m)
        if isinstance(result, tuple):
            labeled, num_features = result
        else:
            # Fallback: if it returns just the array, compute num_features
            labeled = result
            num_features = int(labeled.max())

        # Extract components
        components = []
        for label in range(1, num_features + 1):
            component = np.argwhere(labeled == label)
            # Convert to list of tuples (handles both CPU and GPU arrays)
            if hasattr(component, 'get'):
                component = component.get()
            components.append([tuple(coord) for coord in component])

        return components



    def _create_obstacle_from_component(self, component, grid_shape, map_obj: LidarGridMapVec):
        """
        Create obstacle segments from a connected component of cells.

        Args:
            component: List of (i, j) coordinates representing connected cells
            grid_shape: Shape of the grid (H, W)
            map_obj: Map object with occupancy_map attribute

        Returns:
            List of line segments representing the obstacle boundary
        """
        if not component:
            return []

        # Use the actual resolution from the map object
        H, W = grid_shape
        left_lower = map_obj.occupancy_map.left_lower
        right_upper = map_obj.occupancy_map.right_upper

        cell_w = map_obj.occupancy_map.resolution
        cell_h = map_obj.occupancy_map.resolution

        # Calculate bounding box
        i_coords = [cell[0] for cell in component]
        j_coords = [cell[1] for cell in component]

        min_i, max_i = min(i_coords), max(i_coords)
        min_j, max_j = min(j_coords), max(j_coords)
        # Calculate dimensions in number of cells
        num_cells_i = max_i - min_i + 1
        num_cells_j = max_j - min_j + 1

        # Calculate physical dimensions
        dx = num_cells_j * cell_w  # width (columns)
        dy = num_cells_i * cell_h  # height (rows)

        # Calculate centroid at cell centers (account for 0.5 offset)
        # IMPORTANT: Array row 0 is the TOP of the map. World y increases upward from bottom.
        # Convert row index i to world row by flipping: i_world = H - 1 - i
        res_x = cell_w
        res_y = cell_h

        center_i = (min_i + max_i + 1) / 2.0
        center_j = (min_j + max_j + 1) / 2.0

        # Map (i,j) -> (x,y) using cell edge convention, flipping row index
        i_world = H - center_i
        j_world = center_j
        centroid = left_lower + np.array([j_world * res_x,
                                          i_world * res_y])

        # Create obstacle and get its line segments
        obstacle = Obstacle(centroid, dx=dx, dy=dy, angle=0)
        return obstacle._Obstacle__get_points(centroid)  # tuple: (bottom, top, left, right)


class LIDAR_with_Noise(LIDAR):
    def __init__(self, fov, r_max, B, noise_std):
        super().__init__(fov, r_max, B)
        self.noise_std = noise_std

    def get_laser_ref(self, segments, robot_pose=None):
        if robot_pose is None:
            robot_pose = np.array([0.0, 0.0])
        angles, dist_theta = super().get_laser_ref(segments, robot_pose)
        dist_theta += np.random.normal(0,
                                       self.noise_std, size=dist_theta.shape)
        return angles, dist_theta
