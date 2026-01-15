import math
from collections import deque
import matplotlib.pyplot as plt
from ..utils.array_backend import np

from ..utils.map import bresenham_vec

EXTEND_AREA = 1.0
FREE = 0.0
OCCUPIED = 1.0


class GridMap:
    def __init__(self, x_min, x_max, y_min, y_max, resolution=1.0,
                 init_val=0.5):
        """__init__
        :param x_min, x_max, y_min, y_max: positions of extremities of map [m]
        :param resolution: number of levels of quantization per meter [m]
        :param init_val: initial value for all grid cells
        """
        self.left_lower_x = x_min
        self.left_lower_y = y_min
        self.resolution = resolution
        self.center_x = (x_max - x_min) / 2
        self.center_y = (y_max - y_min) / 2

        self.width = int((x_max - x_min + 1) * 1 / self.resolution)
        self.height = int((y_max - y_min + 1) * 1 / self.resolution)

        self.n_data = self.width * self.height
        self.data = [init_val] * self.n_data
        self.data_type = type(init_val)

    # Getters##################################################################
    def get_value_from_xy_index(self, x_ind, y_ind):
        """get_value_from_xy_index

        when the index is out of grid map area, return None

        :param x_ind: x index
        :param y_ind: y index
        """

        grid_ind = self.calc_grid_index_from_xy_index(x_ind, y_ind)

        if 0 <= grid_ind < self.n_data:
            return self.data[grid_ind]
        else:
            return None

    def get_map_index_from_pos(self, pos):
        """get_map_index_from_pos

        :param pos: x,y position [m]
        """
        x_ind = self.calc_xy_index_from_position(
            pos[0], self.left_lower_x, self.width)
        y_ind = self.calc_xy_index_from_position(
            pos[1], self.left_lower_y, self.height)

        return x_ind, y_ind

    # Setters##################################################################
    def set_value_from_pos(self, pos, val):
        """set_value_from_pos

        return bool flag, which means setting value is succeeded or not

        :param pos: x, y position [m]
        :param val: grid value
        """

        x_ind, y_ind = self.get_map_index_from_pos(pos)

        if (not x_ind) or (not y_ind):

            # print("returning false on set_value_from_pos")
            # raise ValueError("Not GOOD!")
            return False  # NG

        # print("not returning false on set_value_from_pos")
        flag = self.set_value_from_xy_index(x_ind, y_ind, val)

        return flag

    def set_value_from_xy_index(self, x_ind, y_ind, val):
        """set_value_from_xy_index

        return bool flag, which means setting value is succeeded or not

        :param x_ind: x index
        :param y_ind: y index
        :param val: grid value
        """

        if (x_ind is None) or (y_ind is None):
            raise ValueError("returning false from set_value_from_xy_index")
            # return False, False

        # print("not returning false from set_value_from_xy_index")
        grid_ind = int(y_ind * self.width + x_ind)

        if 0 <= grid_ind < self.n_data and isinstance(val, self.data_type):
            self.data[grid_ind] = val
            return True  # OK
        else:
            print(f"grid_ind: {grid_ind} vs self.n_data: {self.n_data}")
            print(f"val: {val} vs self.data_type: {self.data_type}")
            raise ValueError(f"y_ind: {y_ind} or x_ind: {x_ind} out of bounds")
            # return False  # NG

    def set_value_from_polygon(self, pol_x, pol_y, val, inside=True):
        """set_value_from_polygon

        Setting value inside or outside polygon

        :param pol_x: x position list for a polygon
        :param pol_y: y position list for a polygon
        :param val: grid value
        :param inside: setting data inside or outside
        """

        # making ring polygon
        if (pol_x[0] != pol_x[-1]) or (pol_y[0] != pol_y[-1]):
            np.append(pol_x, pol_x[0])
            np.append(pol_y, pol_y[0])

        # setting value for all grid
        for x_ind in range(self.width):
            for y_ind in range(self.height):
                x_pos, y_pos = self.calc_grid_central_xy_position_from_xy_index(
                    x_ind, y_ind)

                flag = self.check_inside_polygon(x_pos, y_pos, pol_x, pol_y)

                if flag is inside:
                    self.set_value_from_xy_index(x_ind, y_ind, val)

    # Calculators##############################################################
    def calc_grid_index_from_xy_index(self, x_ind, y_ind):
        grid_ind = int(y_ind * self.width + x_ind)
        return grid_ind

    def calc_xy_index_from_grid_index(self, grid_ind):
        y_ind, x_ind = divmod(grid_ind, self.width)
        return x_ind, y_ind

    def calc_grid_index_from_xy_pos(self, pos):
        """get_map_index_from_pos

        :param pos: x, y position [m]
        """
        x_ind = self.calc_xy_index_from_position(
            pos[0], self.left_lower_x, self.width)
        y_ind = self.calc_xy_index_from_position(
            pos[1], self.left_lower_y, self.height)

        return self.calc_grid_index_from_xy_index(x_ind, y_ind)

    def calc_grid_central_xy_position_from_grid_index(self, grid_ind):
        x_ind, y_ind = self.calc_xy_index_from_grid_index(grid_ind)
        return self.calc_grid_central_xy_position_from_xy_index(x_ind, y_ind)

    def calc_grid_central_xy_position_from_xy_index(self, x_ind, y_ind):
        x_pos = self.calc_grid_central_xy_position_from_index(
            x_ind, self.left_lower_x)
        y_pos = self.calc_grid_central_xy_position_from_index(
            y_ind, self.left_lower_y)

        return x_pos, y_pos

    def calc_grid_central_xy_position_from_index(self, index, lower_pos):
        return lower_pos + index * self.resolution + self.resolution / 2.0

    def calc_xy_index_from_position(self, pos, lower_pos, max_index):
        ind = int(np.floor((pos - lower_pos) / self.resolution))
        if 0 <= ind <= max_index:
            return ind
        else:
            return None

    def check_occupied_from_xy_index(self, x_ind, y_ind, occupied_val):
        val = self.get_value_from_xy_index(x_ind, y_ind)

        if val is None or val >= occupied_val:
            return True
        else:
            return False

    def expand_grid(self, occupied_val=1.0):
        x_inds, y_inds, values = [], [], []

        for ix in range(self.width):
            for iy in range(self.height):
                if self.check_occupied_from_xy_index(ix, iy, occupied_val):
                    x_inds.append(ix)
                    y_inds.append(iy)
                    values.append(self.get_value_from_xy_index(ix, iy))

        for (ix, iy, value) in zip(x_inds, y_inds, values):
            self.set_value_from_xy_index(ix + 1, iy, val=value)
            self.set_value_from_xy_index(ix, iy + 1, val=value)
            self.set_value_from_xy_index(ix + 1, iy + 1, val=value)
            self.set_value_from_xy_index(ix - 1, iy, val=value)
            self.set_value_from_xy_index(ix, iy - 1, val=value)
            self.set_value_from_xy_index(ix - 1, iy - 1, val=value)

    @staticmethod
    def check_inside_polygon(iox, ioy, x, y):
        n_point = len(x) - 1
        inside = False
        for i1 in range(n_point):
            i2 = (i1 + 1) % (n_point + 1)

            if x[i1] >= x[i2]:
                min_x, max_x = x[i2], x[i1]
            else:
                min_x, max_x = x[i1], x[i2]
            if not min_x <= iox < max_x:
                continue

            tmp1 = (y[i2] - y[i1]) / (x[i2] - x[i1])
            if (y[i1] + tmp1 * (iox - x[i1]) - ioy) > 0.0:
                inside = not inside

        return inside

    def print_grid_map_info(self):
        print("width:", self.width)
        print("height:", self.height)
        print("resolution:", self.resolution)
        print("center_x:", self.center_x)
        print("center_y:", self.center_y)
        print("left_lower_x:", self.left_lower_x)
        print("left_lower_y:", self.left_lower_y)
        print("n_data:", self.n_data)

    def plot_grid_map(self, ax=None):
        float_data_array = np.array([d.get_float_data() for d in self.data])
        grid_data = np.reshape(float_data_array, (self.height, self.width))
        if not ax:
            fig, ax = plt.subplots()
        heat_map = ax.pcolor(grid_data, cmap="Blues", vmin=0.0, vmax=1.0)
        plt.axis("equal")

        return heat_map


class GridMapNP:
    def __init__(self, x_min, x_max, y_min, y_max, quantization_level=5,
                 init_val=0.5):
        """__init__
        :param x_min, x_max, y_min, y_max: positions of extremities of map [m]
        :param quantization_level: number of grid cells per dimension (creates quantization_level x quantization_level grid)
        :param init_val: initial value for all grid cells
        """
        self.left_lower = np.array([x_min, y_min])
        self.right_upper = np.array([x_max, y_max])
        self.quantization_level = quantization_level
        self.resolution = (x_max - x_min) / quantization_level  # Calculate actual cell size
        self.center = np.array([(x_max - x_min) / 2, (y_max - y_min) / 2])

        self.width = quantization_level
        self.height = quantization_level

        self.data = np.full((self.width, self.height),
                            init_val, dtype=type(init_val))
        self.size = self.width * self.height
        self.data_type = type(init_val)

    # Getters##################################################################
    def get_value_from_xy_index(self, x_ind, y_ind):
        if x_ind is None or y_ind is None:
            return None
        if 0 <= x_ind < self.width and 0 <= y_ind < self.height:
            return self.data[x_ind, y_ind]
        else:
            return None

    def get_values_from_xy_indices(self, x_inds, y_inds):
        if (x_inds is None) or (y_inds is None):
            raise ValueError("returning false from get_values_from_xy_indices")
        valid = (x_inds >= 0) & (x_inds < self.width) & (
            y_inds >= 0) & (y_inds < self.height)
        if not np.all(valid):
            print(f"x_inds: {x_inds} or y_inds: {y_inds} out of bounds")
            raise ValueError(
                f"x_inds: {x_inds} or y_inds: {y_inds} out of bounds")
        return self.data[x_inds, y_inds]

    def get_map_index_from_pos(self, pos):
        x_ind, y_ind = np.floor(
            (pos - self.left_lower) / self.resolution, dtype=int)
        if 0 <= x_ind < self.width and 0 <= y_ind < self.height:
            return x_ind, y_ind
        else:
            raise ValueError(f'Index {x_ind}, {y_ind} out of bounds!')

    def get_map_indices_from_positions(self, positions):
        x_inds, y_inds = np.floor(
            (positions - self.left_lower) / self.resolution).astype(int)
        valid = (x_inds >= 0) & (x_inds < self.width) & (
            y_inds >= 0) & (y_inds < self.height)
        if np.all(valid):
            return x_inds, y_inds
        else:
            print(f"x_inds: {x_inds} or y_inds: {y_inds} out of bounds")
            raise ValueError(
                f"x_inds: {x_inds} or y_inds: {y_inds} out of bounds")

    def get_position_from_map_index(self, x_ind, y_ind):
        return self.left_lower + np.array([x_ind, y_ind]) * self.resolution

    def get_map_positions(self, grid):
        """
        Takes in a HxW occupancy map as an argument and returns a (H*W, 2) array of positions
        corresponding to each grid cell (x=j, y=i) at cell centers in world coordinates.
        """
        H, W = grid.shape
        ii, jj = np.indices((H, W))  # ii: rows (y/i), jj: cols (x/j)
        xy_indices = np.stack((jj.ravel(), ii.ravel()),
                              axis=1)  # (N, 2) as (x, y)
        return self.left_lower + (xy_indices + 0.5) * self.resolution

    def get_occupied_map_positions(self, grid):
        """
        Takes in an (H, W) occupancy grid and returns (K, 2) positions of nonzero cells.
        """
        ij = np.argwhere(grid != 0)         # (K, 2) with (i, j) = (row, col)
        xy = ij[:, [1, 0]]                  # reorder to (x=j, y=i)
        return self.left_lower + (xy + 0.5) * self.resolution

    def get_occupied_map_positions_batched(self, grids):
        """
        Fully vectorized batched version that processes multiple maps at once.

        Uses advanced indexing and sorting to eliminate Python loops.

        Args:
            grids: Array of shape (n_m, H, W) - multiple occupancy grids
        Returns:
            positions: Array of shape (n_m, max_occupied, 2) - padded positions
            valid_mask: Array of shape (n_m, max_occupied) - boolean mask indicating valid entries
        """
        if grids.ndim == 2:
            grids = grids[np.newaxis, :, :]  # (1, H, W)

        n_m, H, W = grids.shape

        # Vectorized approach: find all occupied cells across all maps
        # Create coordinate grids: ii: rows (y/i), jj: cols (x/j)
        ii, jj = np.indices((H, W))  # (H, W), (H, W)

        # Broadcast to all maps: (1, H, W) -> (n_m, H, W)
        ii_broadcast = np.broadcast_to(ii[np.newaxis, :, :], (n_m, H, W))  # (n_m, H, W)
        jj_broadcast = np.broadcast_to(jj[np.newaxis, :, :], (n_m, H, W))  # (n_m, H, W)

        # Find occupied cells: (n_m, H, W) boolean
        occupied = grids != 0  # (n_m, H, W)

        # Count occupied cells per map: (n_m,)
        counts_per_map = np.sum(occupied, axis=(1, 2))  # (n_m,)
        max_occupied = int(np.max(counts_per_map))

        if max_occupied == 0:
            # No occupied cells in any map
            return np.zeros((n_m, 1, 2), dtype=np.float32), np.zeros((n_m, 1), dtype=bool)

        # Flatten for easier indexing: (n_m, H*W)
        occupied_flat = occupied.reshape(n_m, H * W)  # (n_m, H*W)
        ii_flat = ii_broadcast.reshape(n_m, H * W)  # (n_m, H*W)
        jj_flat = jj_broadcast.reshape(n_m, H * W)  # (n_m, H*W)

        # Create map indices for grouping: (n_m, H*W)
        map_indices = np.broadcast_to(np.arange(n_m)[:, np.newaxis], (n_m, H * W))  # (n_m, H*W)

        # Flatten everything: (n_m * H * W,)
        occupied_all = occupied_flat.flatten()  # (n_m * H * W,)
        ii_all = ii_flat.flatten()  # (n_m * H * W,)
        jj_all = jj_flat.flatten()  # (n_m * H * W,)
        map_indices_all = map_indices.flatten()  # (n_m * H * W,)

        # Get indices of all occupied cells across all maps
        occupied_idx = np.where(occupied_all)[0]  # (total_occupied,)

        if len(occupied_idx) == 0:
            return np.zeros((n_m, 1, 2), dtype=np.float32), np.zeros((n_m, 1), dtype=bool)

        # Extract coordinates for occupied cells: (total_occupied,)
        i_occupied = ii_all[occupied_idx]  # (total_occupied,)
        j_occupied = jj_all[occupied_idx]  # (total_occupied,)
        map_idx_occupied = map_indices_all[occupied_idx]  # (total_occupied,)

        # Convert to world coordinates: (total_occupied, 2)
        xy = np.stack([j_occupied, i_occupied], axis=1)  # (total_occupied, 2)
        world_positions = self.left_lower + (xy + 0.5) * self.resolution  # (total_occupied, 2)

        # Now we need to group by map and pad to max_occupied
        # Sort by map index to group together
        sort_idx = np.argsort(map_idx_occupied)
        sorted_map_idx = map_idx_occupied[sort_idx]  # (total_occupied,)
        sorted_positions = world_positions[sort_idx]  # (total_occupied, 2)

        # Find boundaries between maps using vectorized operations
        # First entry of each map is where map_idx changes
        if len(sorted_map_idx) > 1:
            map_changes = np.concatenate((np.array([True]), sorted_map_idx[1:] != sorted_map_idx[:-1]))
        else:
            map_changes = np.array([True])

        # Get start and end indices for each map using vectorized operations
        # Find where each map starts in the sorted array
        change_indices = np.where(map_changes)[0]  # (n_m_actual,) - indices where maps start
        change_map_ids = sorted_map_idx[change_indices]  # (n_m_actual,) - which map starts at each index

        # Create mapping from map_id to start index: (n_m,)
        map_start_indices = np.full(n_m, len(sorted_positions), dtype=np.int32)  # Initialize to end
        if len(change_indices) > 0:
            map_start_indices[change_map_ids] = change_indices

        # Get end indices: start of next map, or end of array
        # Sort change_indices to get sequential order
        if len(change_indices) > 0:
            sorted_change_idx = np.argsort(change_map_ids)
            sorted_change_positions = change_indices[sorted_change_idx]
            sorted_change_maps = change_map_ids[sorted_change_idx]

            # End index is start of next map (or end of array)
            map_end_indices = np.full(n_m, len(sorted_positions), dtype=np.int32)
            for i in range(len(sorted_change_positions) - 1):
                map_idx = sorted_change_maps[i]
                map_end_indices[map_idx] = sorted_change_positions[i + 1]
            # Last map ends at end of array (already set)
        else:
            map_end_indices = np.full(n_m, len(sorted_positions), dtype=np.int32)

        # Create output arrays: (n_m, max_occupied, 2)
        positions = np.zeros((n_m, max_occupied, 2), dtype=np.float32)
        valid_mask = np.zeros((n_m, max_occupied), dtype=bool)

        # Use advanced indexing to fill all maps at once
        # Create a flat index array that maps from sorted_positions to output array
        # This is tricky because each map has different length, so we need per-map assignment
        # The loop here is O(n_m) which is typically small, but we can't easily vectorize
        # variable-length assignments to different rows
        for j in range(n_m):
            start_idx = map_start_indices[j]
            end_idx = map_end_indices[j]
            K_j = end_idx - start_idx

            if K_j > 0 and K_j <= max_occupied:
                positions[j, :K_j, :] = sorted_positions[start_idx:end_idx]
                valid_mask[j, :K_j] = True

        return positions, valid_mask

    # Setters##################################################################
    def set_value_from_pos(self, pos, val):
        x_ind, y_ind = self.get_map_index_from_pos(pos)
        if (not x_ind) or (not y_ind):
            raise ValueError("Not GOOD!")
            # return False
        flag = self.set_value_from_xy_index(x_ind, y_ind, val)
        return flag

    def set_value_from_xy_index(self, x_ind, y_ind, val):
        if (x_ind is None) or (y_ind is None):
            raise ValueError("returning false from set_value_from_xy_index")
        if 0 <= x_ind < self.width and 0 <= y_ind < self.height and isinstance(val, self.data_type):
            self.data[x_ind, y_ind] = val
            return True
        else:
            print(f'x_ind: {x_ind}, y_ind: {y_ind} out of bounds')
            print(f'val: {val} vs self.data_type: {self.data_type}')
            raise ValueError(f'y_ind: {y_ind} or x_ind: {x_ind} out of bounds')

    def set_values_from_xy_indices(self, inds, val):
        if (inds is None):
            raise ValueError("returning false from set_values_from_xy_indices")
        valid = (inds[0] >= 0) & (inds[0] < self.width) & (
            inds[1] >= 0) & (inds[1] < self.height)
        if not np.all(valid):
            print(f"inds: {inds} out of bounds")
            print(f"val: {val} vs self.data_type: {self.data_type}")
            raise ValueError(
                f"inds: {inds} out of bounds")
        if not isinstance(val, self.data_type):
            raise ValueError(
                f"val: {val} is not a valid data type for self.data_type: {self.data_type}")
        # TODO: check if this is a valid way to access array element
        self.data[inds] = val
        return True

    # Rest of the methods are the same as in the original GridMap class
    def check_occupied_from_xy_index(self, x_ind, y_ind, occupied_val):
        val = self.get_value_from_xy_index(x_ind, y_ind)
        if val is None or val >= occupied_val:
            return True
        else:
            return False

    def expand_grid(self, occupied_val=1.0):
        x_inds, y_inds, values = [], [], []
        for ix in range(self.width):
            for iy in range(self.height):
                if self.check_occupied_from_xy_index(ix, iy, occupied_val):
                    x_inds.append(ix)
                    y_inds.append(iy)
                    values.append(self.get_value_from_xy_index(ix, iy))
        for (ix, iy, value) in zip(x_inds, y_inds, values):
            self.set_value_from_xy_index(ix + 1, iy, val=value)
            self.set_value_from_xy_index(ix, iy + 1, val=value)
            self.set_value_from_xy_index(ix + 1, iy + 1, val=value)
            self.set_value_from_xy_index(ix - 1, iy, val=value)
            self.set_value_from_xy_index(ix, iy - 1, val=value)
            self.set_value_from_xy_index(ix - 1, iy - 1, val=value)

    def print_grid_map_info(self):
        print("width:", self.width)
        print("height:", self.height)
        print("resolution:", self.resolution)
        print("center_x:", self.center_x)
        print("center_y:", self.center_y)
        print("left_lower_x:", self.left_lower_x)
        print("left_lower_y:", self.left_lower_y)

    def plot_grid_map(self, ax=None):
        grid_data = self.data
        if not ax:
            fig, ax = plt.subplots()
        heat_map = ax.pcolor(grid_data, cmap="Blues", vmin=0.0, vmax=1.0)
        plt.axis("equal")
        return heat_map


class LidarGridMap:
    def __init__(self, x_min, x_max, y_min, y_max, resolution=0.02):
        self.xy_resolution = resolution
        self.occupancy_map = GridMap(
            x_min, x_max, y_min, y_max, resolution=self.xy_resolution)

    @staticmethod
    def bresenham(start, end):
        """
        Implementation of Bresenham's line drawing algorithm
        See en.wikipedia.org/wiki/Bresenham's_line_algorithm
        Bresenham's Line Algorithm
        Produces a np.array from start and end (original from roguebasin.com)
        >>> points1 = bresenham((4, 4), (6, 10))
        >>> print(points1)
        np.array([[4,4], [4,5], [5,6], [5,7], [5,8], [6,9], [6,10]])
        """
        # setup initial conditions
        d = end - start
        is_steep = abs(d[1]) > abs(d[0])  # determine how steep the line is
        if is_steep:  # rotate line
            start[0], start[1] = start[1], start[0]
            end[0], end[1] = end[1], end[0]
        # swap start and end points if necessary and store swap state
        swapped = False
        if start[0] > end[0]:
            start, end = end, start
            swapped = True
        d = end - start  # recalculate differentials
        error = int(d[0] / 2.0)  # calculate error
        y_step = 1 if start[1] < end[1] else -1
        # iterate over bounding box generating points between start and end
        y = start[1]
        points = []
        for x in range(start[0], end[0] + 1):
            coord = [y, x] if is_steep else (x, y)
            points.append(coord)
            error -= abs(d[1])
            if error < 0:
                y += y_step
                error += d[0]
        if swapped:  # reverse the list if the coordinates were swapped
            points.reverse()
        points = np.array(points)
        return points

    @staticmethod
    def atan_zero_to_twopi(y, x):
        angle = math.atan2(y, x)
        if angle < 0.0:
            angle += math.pi * 2.0
        return angle

    def init_flood_fill(self, center_point, obstacle_points, xy_points, min_coord,
                        xy_resolution):
        """
        center_point: center point
        obstacle_points: detected obstacles points (x,y)
        xy_points: (x,y) point pairs
        """
        center_x, center_y = center_point
        prev_ix, prev_iy = center_x - 1, center_y
        ox, oy = obstacle_points
        xw, yw = xy_points
        min_x, min_y = min_coord
        occupancy_map = (np.ones((xw, yw))) * 0.5
        for (x, y) in zip(ox, oy):
            # x coordinate of the the occupied area
            ix = int(round((x - min_x) / xy_resolution))
            # y coordinate of the the occupied area
            iy = int(round((y - min_y) / xy_resolution))
            free_area = self.bresenham((prev_ix, prev_iy), (ix, iy))
            for fa in free_area:
                occupancy_map[fa[0]][fa[1]] = 0  # free area 0.0
            prev_ix = ix
            prev_iy = iy
        return occupancy_map

    @staticmethod
    def flood_fill(center_point, occupancy_map):
        """
        center_point: starting point (x,y) of fill
        occupancy_map: occupancy map generated from Bresenham ray-tracing
        """
        # Fill empty areas with queue method
        sx, sy = occupancy_map.shape
        fringe = deque()
        fringe.appendleft(center_point)
        while fringe:
            n = fringe.pop()
            nx, ny = n
            # West
            if nx > 0:
                if occupancy_map[nx - 1, ny] == 0.5:
                    occupancy_map[nx - 1, ny] = 0.0
                    fringe.appendleft((nx - 1, ny))
            # East
            if nx < sx - 1:
                if occupancy_map[nx + 1, ny] == 0.5:
                    occupancy_map[nx + 1, ny] = 0.0
                    fringe.appendleft((nx + 1, ny))
            # North
            if ny > 0:
                if occupancy_map[nx, ny - 1] == 0.5:
                    occupancy_map[nx, ny - 1] = 0.0
                    fringe.appendleft((nx, ny - 1))
            # South
            if ny < sy - 1:
                if occupancy_map[nx, ny + 1] == 0.5:
                    occupancy_map[nx, ny + 1] = 0.0
                    fringe.appendleft((nx, ny + 1))

    def update_grid_map(self, pos, ox, oy, is_obstacle):
        """
        The breshenham boolean tells if it's computed with bresenham ray casting
        (True) or with flood fill (False)
        """
        center_x, center_y = self.occupancy_map.get_map_index_from_pos(pos)
        # occupancy grid computed with bresenham ray casting
        for i, (x, y) in enumerate(zip(ox, oy)):
            # Get index of the endpoint
            ix, iy = self.occupancy_map.get_map_index_from_pos(np.array[x, y])
            if ix is None or iy is None:
                continue

            # Line from the lidar to the endpoint
            laser_beam = self.bresenham((center_x, center_y), (ix, iy))

            # Mark cells along beam as free
            for (x_free, y_free) in laser_beam:
                if self.occupancy_map.get_value_from_xy_index(x_free, y_free) != OCCUPIED:
                    self.occupancy_map.set_value_from_xy_index(
                        x_free, y_free, FREE)

            # Only mark endpoint as occupied if it's an actual obstacle
            if is_obstacle[i]:
                self.occupancy_map.set_value_from_xy_index(ix, iy, OCCUPIED)


class LidarGridMapVec:
    def __init__(self, x_min, x_max, y_min, y_max, quantization_level=5):
        self.quantization_level = quantization_level
        self.occupancy_map = GridMapNP(
            x_min, x_max, y_min, y_max, quantization_level=self.quantization_level)

    def seed_from_obstacles(self, obstacles):
        """Rasterize obstacle segments into the occupancy grid as OCCUPIED (1.0)."""
        # Collect points sampled along each obstacle segment
        sampled_points = []
        step = max(self.occupancy_map.resolution / 2.0, 1e-3)
        for obs in obstacles:
            # Use obstacle geometry at its centroid
            segments = obs._Obstacle__get_points(obs.centroid)
            for (x1, y1, x2, y2) in segments:
                dx = x2 - x1
                dy = y2 - y1
                length = (dx**2 + dy**2) ** 0.5
                n = max(2, int(length / step))
                ts = np.linspace(0.0, 1.0, n)
                xs = x1 + ts * dx
                ys = y1 + ts * dy
                sampled_points.append(np.stack([xs, ys], axis=1))

        if len(sampled_points) == 0:
            return

        points = np.vstack(sampled_points)

        # Convert to grid indices with bounds checking
        left_lower = self.occupancy_map.left_lower
        res = self.occupancy_map.resolution
        x_inds = np.floor((points[:, 0] - left_lower[0]) / res).astype(int)
        y_inds = np.floor((points[:, 1] - left_lower[1]) / res).astype(int)
        valid = (x_inds >= 0) & (x_inds < self.occupancy_map.width) & \
                (y_inds >= 0) & (y_inds < self.occupancy_map.height)
        if not np.any(valid):
            return
        inds = (x_inds[valid], y_inds[valid])
        self.occupancy_map.set_values_from_xy_indices(inds, OCCUPIED)

    def update_grid_map_vec(self, pos, detections, is_obstacle):
        """
        The breshenham boolean tells if it's computed with bresenham ray casting
        (True) or with flood fill (False)
        """
        start = self.occupancy_map.get_map_index_from_pos(pos)
        ends = self.occupancy_map.get_map_indices_from_positions(detections)
        # Get index of the endpoint
        # check if indices contains None
        if any(index is None for index in ends):
            raise ValueError(f"ends: {ends} contains None")
        else:
            # Only set the values of the detections that are obstacles
            self.occupancy_map.set_values_from_xy_indices(
                ends[is_obstacle], OCCUPIED)

        # Line from the lidar to the endpoint
        beams = bresenham_vec(start, ends)

        # Mark cells along beam as free
        for laser_beam in beams:
            vals = self.occupancy_map.get_values_from_xy_indices(
                laser_beam, FREE)
            free_mask = vals != OCCUPIED
            self.occupancy_map.set_values_from_xy_indices(
                laser_beam[free_mask], FREE)
