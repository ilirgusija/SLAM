import math
from collections import deque
import numpy as _numpy
import matplotlib.pyplot as plt
from ..utils.array_backend import np

from ..utils.map import bresenham_vec
from abc import ABC, abstractmethod
from typing import Optional, Tuple


EXTEND_AREA = 1.0
FREE = 0.0
OCCUPIED = 1.0
"""
Abstract base class for map representations.

This module provides an abstract interface for different map types (occupancy grids,
landmark-based maps, etc.) to be used interchangeably in POMDP and Belief-MDP classes.
"""


class BaseMap(ABC):
    """
    Abstract base class for map representations.

    Subclasses must implement:
    - Map space generation (for belief-MDP)
    - Map distance/metric (d_M)
    - Map representation conversion (for ray casting, belief states)
    - Properties: len_M (number of possible maps), map_shape (for grids)
    """

    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float):
        """
        Initialize map with bounds.

        Args:
            x_min, x_max, y_min, y_max: Bounds of the map in world coordinates
        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        # Use host NumPy for tiny bounds so map construction never touches GPU (avoids CUDARuntimeError when context is bad)
        self.left_lower = _numpy.array([x_min, y_min], dtype=_numpy.float64)
        self.right_upper = _numpy.array([x_max, y_max], dtype=_numpy.float64)

    @property
    @abstractmethod
    def len_M(self) -> int:
        """
        Number of possible maps in the map space M.

        Returns:
            int: Total number of possible maps
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def map_shape(self) -> Tuple[int, ...]:
        """
        Shape of individual map representation.

        For occupancy grids: (H, W)
        For landmark maps: (num_landmarks, 2) or similar

        Returns:
            Tuple[int, ...]: Shape of a single map
        """
        raise NotImplementedError

    @abstractmethod
    def d_M(self, m1: np.ndarray, m2: np.ndarray) -> float:
        """
        Metric on the map space.

        Computes distance between two maps.

        Args:
            m1: First map representation
            m2: Second map representation

        Returns:
            float: Distance between m1 and m2
        """
        raise NotImplementedError

    @abstractmethod
    def generate_all_maps(self, show_progress: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Generate all possible maps in the map space.

        Args:
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (maps_array, map_ids):
            - maps_array: Array of all maps, shape (len_M, ...) where ... is map_shape
            - map_ids: Optional array of map identifiers (e.g., bit representations for grids)
        """
        raise NotImplementedError

    @abstractmethod
    def map_to_id(self, m: np.ndarray) -> int:
        """
        Convert a map representation to a unique identifier.

        Args:
            m: Map representation

        Returns:
            int: Unique identifier for the map
        """
        raise NotImplementedError

    @abstractmethod
    def id_to_map(self, map_id: int) -> np.ndarray:
        """
        Convert a map identifier to a map representation.

        Args:
            map_id: Unique identifier for the map

        Returns:
            np.ndarray: Map representation
        """
        raise NotImplementedError

    @abstractmethod
    def get_obstacle_segments(self, m: np.ndarray) -> list:
        """
        Extract obstacle segments from a map for ray casting.

        Args:
            m: Map representation

        Returns:
            list: List of obstacle segments, where each segment is (x1, y1, x2, y2)
        """
        raise NotImplementedError

    def get_obstacle_segments_batched(self, M: np.ndarray) -> list:
        """
        Extract obstacle segments from multiple maps for batched ray casting.

        Default implementation calls get_obstacle_segments for each map.
        Subclasses can override for more efficient batched operations.

        Args:
            M: Array of maps, shape (n_maps, ...)

        Returns:
            list: List of obstacle segments for all maps
        """
        all_segments = []
        for i in range(M.shape[0]):
            segments = self.get_obstacle_segments(M[i])
            all_segments.extend(segments)
        return all_segments

    # Compatibility properties for occupancy grids
    @property
    def height(self) -> Optional[int]:
        """
        Height of the map (for occupancy grids).

        Returns:
            int if applicable, None otherwise
        """
        if hasattr(self, 'occupancy_map'):
            return getattr(self.occupancy_map, 'height', None)
        return None

    @property
    def width(self) -> Optional[int]:
        """
        Width of the map (for occupancy grids).

        Returns:
            int if applicable, None otherwise
        """
        if hasattr(self, 'occupancy_map'):
            return getattr(self.occupancy_map, 'width', None)
        return None

class LandmarkMap(BaseMap):
    """
    Landmark-based map representation with spatial quantization.

    Map space M = W^l where:
    - W = workspace grid: n×n cell centres over [x_min,x_max]×[y_min,y_max]
    - l = number of landmarks (L)
    So |M| = (n²)^l. Each map is an (L, 2) array of landmark positions (cell centres).

    For workspace [0,10]×[0,10] and n=2: W = {(2.5,2.5), (2.5,7.5), (7.5,2.5), (7.5,7.5)}.
    For n=2, l=2: |M| = 4^2 = 16.
    """

    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        n: int = 2,
        *,
        num_landmarks: int | None = None,
        landmark_positions: np.ndarray | None = None,
    ):
        """
        Args:
            x_min, x_max, y_min, y_max: Workspace bounds. W is the n×n grid over this box.
            n: Quantization level. W has K = n² cell centres.
            num_landmarks: Number of landmarks (l). Use this for coherent M=W^l; then
                landmark_positions is set to the first l grid centres (for compat).
            landmark_positions: Optional (L, 2) array. If provided, L = shape[0] and
                overrides num_landmarks. Kept for backward compat; map space is still
                built from the grid (cell_centers), not these values.
        """
        super().__init__(x_min, x_max, y_min, y_max)
        self.n = int(n)
        self.n_cells = self.n * self.n  # K = n²

        # Build W = K cell centres in row-major order (row = y-axis, col = x-axis).
        # Cell (i, j): centre_x = x_min + (j+0.5)*dx, centre_y = y_min + (i+0.5)*dy
        dx = (x_max - x_min) / self.n
        dy = (y_max - y_min) / self.n
        self.cell_centers = np.array(
            [[x_min + (j + 0.5) * dx, y_min + (i + 0.5) * dy]
             for i in range(self.n) for j in range(self.n)],
            dtype=np.float64,
        )  # (K, 2) = workspace grid W

        if landmark_positions is not None:
            self.landmark_positions = np.asarray(landmark_positions, dtype=_numpy.float64)  # (L, 2)
            self.num_candidate_landmarks = self.landmark_positions.shape[0]  # L (number of landmarks)
        elif num_landmarks is not None:
            self.num_candidate_landmarks = int(num_landmarks)  # L
            # Nominal positions for compat: first L grid centres (cycle if L > K)
            idx = np.arange(self.num_candidate_landmarks, dtype=np.intp) % self.n_cells
            self.landmark_positions = np.asarray(self.cell_centers[idx], dtype=np.float64)  # (L, 2)
        else:
            raise ValueError("LandmarkMap requires either num_landmarks or landmark_positions")

        self._len_M = self.n_cells ** self.num_candidate_landmarks  # |M| = K^L = (n²)^L

    @property
    def len_M(self) -> int:
        """Number of possible maps: (n²)^L."""
        return self._len_M

    def get_cache_fingerprint(self) -> tuple:
        """Return a hashable tuple identifying the workspace grid (for cache keys).
        Ensures caches for Q_n, all_maps, p_n_M are invalidated when the grid changes.
        """
        return (
            float(self.x_min), float(self.x_max),
            float(self.y_min), float(self.y_max),
            int(self.n), int(self.n_cells), int(self.num_candidate_landmarks),
        )

    @property
    def map_shape(self) -> Tuple[int, ...]:
        """Shape of a single map: (L, 2) landmark positions."""
        return (self.num_candidate_landmarks, 2)

    def d_M(self, m1: np.ndarray, m2: np.ndarray, p: float = 2.0) -> float:
        """
        Lp norm of per-landmark Euclidean distances.

        Args:
            m1, m2: (L, 2) landmark-position arrays.
            p: Lp exponent (use float('inf') for L∞).
        """
        per_lm = np.linalg.norm(
            np.asarray(m1).reshape(-1, 2) - np.asarray(m2).reshape(-1, 2),
            axis=1,
        )  # (L,) per-landmark distances
        if p == float('inf'):
            return float(np.max(per_lm))
        return float(np.linalg.norm(per_lm, ord=p))

    def generate_all_maps(self, show_progress: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Generate all (n²)^L landmark maps.

        Returns:
            maps_array: (len_M, L, 2) array of landmark positions.
            map_ids:    (len_M,) integer ids in [0, len_M).
        """
        L = self.num_candidate_landmarks
        K = self.n_cells
        # All K^L combinations of L cell indices, MSB = landmark 0.
        coords = [np.arange(K) for _ in range(L)]
        mesh = np.meshgrid(*coords, indexing='ij')    # each (K,…,K) with L dims
        idx_grid = np.stack([g.ravel() for g in mesh], axis=1)  # (len_M, L)
        maps_array = self.cell_centers[idx_grid]           # (len_M, L, 2)
        map_ids = np.arange(self.len_M, dtype=np.int64)
        return np.asarray(maps_array), map_ids

    def map_to_id(self, m: np.ndarray) -> int:
        """
        Encode an (L, 2) position array to its base-K integer id.

        Each landmark position is snapped to the nearest cell centre.

        Args:
            m: (L, 2) array of landmark positions, or (L,) array of cell indices.

        Returns:
            int: Unique identifier in [0, len_M).
        """
        m_arr = np.asarray(m)
        if m_arr.ndim == 1 and m_arr.size == self.num_candidate_landmarks:
            cell_indices = m_arr.astype(int)
        else:
            m_2d = m_arr.reshape(-1, 2)  # (L, 2)
            diffs = m_2d[:, None, :] - self.cell_centers[None, :, :]  # (L, K, 2)
            cell_indices = np.argmin(np.linalg.norm(diffs, axis=2), axis=1)  # (L,)
        K = self.n_cells
        map_id = 0
        for idx in cell_indices:
            map_id = map_id * K + int(idx)
        return map_id

    def id_to_map(self, map_id: int) -> np.ndarray:
        """
        Decode a base-K integer into an (L, 2) array of landmark positions.

        Args:
            map_id: Integer in [0, len_M).

        Returns:
            np.ndarray: (L, 2) array of cell-centre coordinates.
        """
        L = self.num_candidate_landmarks
        K = self.n_cells
        indices = []
        remaining = int(map_id)
        for _ in range(L):
            indices.append(remaining % K)
            remaining //= K
        indices = indices[::-1]  # MSB = landmark 0
        return np.asarray(self.cell_centers[indices])  # (L, 2)

    def get_obstacle_segments(self, m: np.ndarray) -> list:
        """
        Return tiny square segments around each landmark position (for ray casting).

        Args:
            m: (L, 2) landmark positions.

        Returns:
            list of (x1, y1, x2, y2) tuples.
        """
        half = 0.05
        segments = []
        for pos in np.asarray(m).reshape(-1, 2):
            x, y = float(pos[0]), float(pos[1])
            segments += [
                (x - half, y + half, x + half, y + half),
                (x + half, y + half, x + half, y - half),
                (x + half, y - half, x - half, y - half),
                (x - half, y - half, x - half, y + half),
            ]
        return segments

    def get_landmark_positions(self, m: np.ndarray) -> np.ndarray:
        """Return the (L, 2) landmark positions from a map array."""
        return np.asarray(m).reshape(-1, 2)


class OrderedLandmarkMap(BaseMap):
    """
    Ordered landmark map representation with known data association.

    Maps are represented as ordered tuples of landmark positions drawn from a
    discrete candidate set, so M = X_n^l where l is the number of landmarks.
    """

    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        landmark_positions: np.ndarray,
        num_landmarks: int
    ):
        super().__init__(x_min, x_max, y_min, y_max)
        self.landmark_positions = np.asarray(landmark_positions)  # (N, 2)
        self.num_candidate_landmarks = self.landmark_positions.shape[0]
        self.num_landmarks = int(num_landmarks)
        self.max_landmarks = self.num_landmarks
        self._len_M = self.num_candidate_landmarks ** self.num_landmarks

    @property
    def len_M(self) -> int:
        return self._len_M

    @property
    def map_shape(self) -> Tuple[int, ...]:
        return (self.num_landmarks, 2)

    def d_M(self, m1: np.ndarray, m2: np.ndarray, p: float = 2.0) -> float:
        m1_flat = m1.reshape(-1) if m1.ndim > 1 else m1
        m2_flat = m2.reshape(-1) if m2.ndim > 1 else m2
        diff = m1_flat - m2_flat
        if p == float('inf'):
            return float(np.max(np.abs(diff)))
        return float(np.linalg.norm(diff, ord=p))

    def generate_all_maps(self, show_progress: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        try:
            from tqdm import tqdm
            _tqdm_available = True
        except ImportError:
            _tqdm_available = False

        N = self.num_candidate_landmarks
        l = self.num_landmarks

        # Generate indices in the same order as map_to_id() expects
        # map_to_id computes: id = idx[0]*N^(l-1) + idx[1]*N^(l-2) + ... + idx[l-1]*N^0
        # So idx[0] varies slowest, idx[l-1] varies fastest
        # np.meshgrid with indexing='ij' makes first index vary slowest, which is what we want
        coords = [np.arange(N) for _ in range(l)]
        mesh = np.meshgrid(*coords, indexing='ij')  # idx[0] varies slowest, idx[l-1] varies fastest
        idx_grid = np.stack([m.ravel() for m in mesh], axis=1)  # (len_M, l)

        if show_progress and _tqdm_available:
            idx_iter = tqdm(idx_grid, desc="Generating ordered landmark maps", leave=False, unit="map")
            maps_list = [self.landmark_positions[idx] for idx in idx_iter]
            maps_array = np.stack(maps_list, axis=0)
        else:
            maps_array = self.landmark_positions[idx_grid]  # (len_M, l, 2)

        map_ids = np.arange(self.len_M, dtype=np.int64)
        return maps_array, map_ids

    def map_to_id(self, m: np.ndarray) -> int:
        if m.ndim == 1 and m.size == self.num_landmarks:
            indices = m.astype(int)
        else:
            # Map provided as positions: find nearest candidate indices
            diffs = m[:, np.newaxis, :] - self.landmark_positions[np.newaxis, :, :]
            dist = np.linalg.norm(diffs, axis=2)
            indices = np.argmin(dist, axis=1)

        N = self.num_candidate_landmarks
        map_id = 0
        for idx in indices:
            map_id = map_id * N + int(idx)
        return int(map_id)

    def id_to_map(self, map_id: int) -> np.ndarray:
        N = self.num_candidate_landmarks
        l = self.num_landmarks
        indices = []
        remaining = int(map_id)
        for _ in range(l):
            indices.append(remaining % N)
            remaining //= N
        indices = indices[::-1]
        return self.landmark_positions[np.array(indices, dtype=int)]

    def get_landmark_positions(self, m: np.ndarray) -> np.ndarray:
        if m.ndim == 2 and m.shape[1] == 2:
            return m
        if m.ndim == 1 and m.size == self.num_landmarks:
            return self.landmark_positions[m.astype(int)]
        return m

    def get_obstacle_segments(self) -> list:
        """Landmark maps have no obstacle segments."""
        return []


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


class LidarGridMapVec(BaseMap):
    def __init__(self, x_min, x_max, y_min, y_max, quantization_level=5):
        super().__init__(x_min, x_max, y_min, y_max)
        self.quantization_level = quantization_level
        self.occupancy_map = GridMapNP(
            x_min, x_max, y_min, y_max, quantization_level=self.quantization_level)

    @property
    def len_M(self) -> int:
        """Number of possible maps: 2^(H*W) for occupancy grids."""
        return 2 ** (self.occupancy_map.height * self.occupancy_map.width)

    @property
    def map_shape(self) -> tuple:
        """Shape of individual map: (H, W) for occupancy grids."""
        return (self.occupancy_map.height, self.occupancy_map.width)

    def d_M(self, m1: np.ndarray, m2: np.ndarray, p: float = 2.0) -> float:
        """
        Lp-Hausdorff metric on the map space (Baddeley 1992).

        For a finite metric space (X, d), the Lp-Hausdorff distance between two subsets
        A and B of X is defined as:
            d_Lp_Haus(A, B) = (Σ_{x∈X} |d(x, A) - d(x, B)|^p)^(1/p)

        where d(x, A) = min_{y∈A} d(x, y) is the point-set distance.

        For discretized occupancy grids:
        - X is the set of all grid cells (H*W cells)
        - A and B are the unions of occupied cell squares in m1 and m2
        - d(x, A) is the minimum distance from point x to any occupied cell surface
        - Distance is computed in world coordinates

        Args:
            m1: First occupancy grid (H, W) or flattened
            m2: Second occupancy grid (H, W) or flattened
            p: Lp norm parameter (default: 2.0 for L2-Hausdorff)
                Use p=float('inf') for standard Hausdorff metric

        Returns:
            float: Lp-Hausdorff distance between m1 and m2
        """
        # Reshape to 2D if needed
        if m1.ndim == 1:
            m1 = m1.reshape(self.map_shape)
        if m2.ndim == 1:
            m2 = m2.reshape(self.map_shape)

        H, W = self.map_shape

        # Occupancy masks for both maps (1 indicates occupied, 0 indicates free)
        occupied1 = m1 != 0  # (H, W) boolean
        occupied2 = m2 != 0  # (H, W) boolean

        # If both maps are empty or identical, return 0
        if np.array_equal(occupied1, occupied2):
            return 0.0

        # Use grid-cell centers as query points for d(x, A) over the workspace
        # Grid indexing: i is row (y), j is column (x)
        # World position: left_lower + (j + 0.5, i + 0.5) * resolution
        ii, jj = np.indices((H, W))  # (H, W) each
        cell_positions = np.stack([
            self.occupancy_map.left_lower[0] + (jj + 0.5) * self.occupancy_map.resolution,
            self.occupancy_map.left_lower[1] + (ii + 0.5) * self.occupancy_map.resolution
        ], axis=-1)  # (H, W, 2)

        # Flatten to (H*W, 2)
        cell_positions_flat = cell_positions.reshape(-1, 2)  # (H*W, 2)

        # Handle edge cases
        if not np.any(occupied1):
            # m1 is empty, d(x, m1) = inf for all x (or use a large distance)
            # For Lp-Hausdorff, we use a large finite distance
            if not np.any(occupied2):
                return 0.0
            # Compute distances from all cells to m2
            # d(x, m1) = large_value, d(x, m2) = min distance to m2
            distances_to_m2 = self._compute_point_to_occupied_distances(
                cell_positions_flat, occupied2)  # (H*W,)
            large_distance = np.max(distances_to_m2) + 1.0  # Use max distance + buffer
            if p == float('inf'):
                return large_distance
            else:
                diff = large_distance - distances_to_m2  # (H*W,)
                return float(np.power(np.sum(np.power(np.abs(diff), p)), 1.0 / p))

        if not np.any(occupied2):
            # m2 is empty, symmetric to above
            distances_to_m1 = self._compute_point_to_occupied_distances(
                cell_positions_flat, occupied1)  # (H*W,)
            large_distance = np.max(distances_to_m1) + 1.0
            if p == float('inf'):
                return large_distance
            else:
                diff = distances_to_m1 - large_distance  # (H*W,)
                return float(np.power(np.sum(np.power(np.abs(diff), p)), 1.0 / p))

        # Compute point-set distances for all cells
        distances_to_m1 = self._compute_point_to_occupied_distances(
            cell_positions_flat, occupied1)  # (H*W,)
        distances_to_m2 = self._compute_point_to_occupied_distances(
            cell_positions_flat, occupied2)  # (H*W,)

        # Compute Lp-Hausdorff distance
        diff = distances_to_m1 - distances_to_m2  # (H*W,)

        if p == float('inf'):
            # Standard Hausdorff metric: max over all cells
            return float(np.max(np.abs(diff)))
        else:
            # Lp-Hausdorff: (Σ |diff|^p)^(1/p)
            return float(np.power(np.sum(np.power(np.abs(diff), p)), 1.0 / p))

    def _compute_point_set_distances(self, points: np.ndarray, set_points: np.ndarray) -> np.ndarray:
        """
        Compute point-set distances: d(x, A) = min_{y∈A} d(x, y) for all x in points.

        Uses vectorized computation for efficiency.

        Args:
            points: Array of shape (N, 2) - points to compute distances for
            set_points: Array of shape (M, 2) - points in the set A

        Returns:
            Array of shape (N,) - minimum distance from each point to the set
        """
        if len(set_points) == 0:
            # Empty set - return large distances
            return np.full(len(points), np.inf, dtype=np.float64)

        # Compute pairwise distances: (N, M)
        # Using broadcasting: points[:, None, :] - set_points[None, :, :] -> (N, M, 2)
        diff = points[:, np.newaxis, :] - set_points[np.newaxis, :, :]  # (N, M, 2)
        distances = np.linalg.norm(diff, axis=2)  # (N, M) - Euclidean distance

        # Take minimum over set_points (axis=1)
        min_distances = np.min(distances, axis=1)  # (N,)

        return min_distances

    def _compute_point_to_occupied_distances(self, points: np.ndarray, occupied: np.ndarray) -> np.ndarray:
        """
        Compute minimum distances from points to occupied cell surfaces.
        """
        if not np.any(occupied):
            return np.full(points.shape[0], np.inf, dtype=np.float64)

        inds = np.argwhere(occupied)  # (K, 2) rows (y), cols (x)
        ii = inds[:, 0]
        jj = inds[:, 1]
        res = self.occupancy_map.resolution
        left_lower = self.occupancy_map.left_lower

        rect_min = np.stack([
            left_lower[0] + jj * res,
            left_lower[1] + ii * res
        ], axis=1)  # (K, 2)
        rect_max = rect_min + res

        px = points[:, 0][:, np.newaxis]  # (N, 1)
        py = points[:, 1][:, np.newaxis]  # (N, 1)

        dx = np.maximum(rect_min[:, 0][np.newaxis, :] - px, 0.0) + \
            np.maximum(px - rect_max[:, 0][np.newaxis, :], 0.0)
        dy = np.maximum(rect_min[:, 1][np.newaxis, :] - py, 0.0) + \
            np.maximum(py - rect_max[:, 1][np.newaxis, :], 0.0)

        dist = np.sqrt(dx * dx + dy * dy)  # (N, K)
        return np.min(dist, axis=1)

    def generate_all_maps(self, show_progress: bool = False) -> tuple:
        """
        Generate all possible occupancy grid maps.

        Args:
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (maps_3d, map_bits):
            - maps_3d: Array of shape (len_M, H, W)
            - map_bits: Array of shape (len_M,) with bit representations
        """
        try:
            from tqdm import tqdm
            _tqdm_available = True
        except ImportError:
            _tqdm_available = False

        H, W = self.map_shape
        total_cells = H * W
        total_maps = 2 ** total_cells

        maps_3d = []
        map_bits_list = []

        iterator = range(total_maps)
        if show_progress and _tqdm_available:
            iterator = tqdm(iterator, desc="Generating all maps", leave=False, unit="map")

        for map_bits in iterator:
            map_array = self.bits_to_map(map_bits, (H, W))
            maps_3d.append(map_array)
            map_bits_list.append(map_bits)

        maps_3d_array = np.stack(maps_3d, axis=0)
        map_bits_array = np.array(map_bits_list, dtype=np.int64)

        return maps_3d_array, map_bits_array

    def map_to_id(self, m: np.ndarray) -> int:
        """
        Encode an occupancy grid m into an integer by row-major bits.

        Args:
            m: Occupancy grid (H, W)

        Returns:
            int: Bit representation of the map
        """
        return self.map_to_bits(m)

    def id_to_map(self, map_id: int) -> np.ndarray:
        """
        Decode integer bits into an occupancy grid.

        Args:
            map_id: Bit representation of the map

        Returns:
            np.ndarray: Occupancy grid (H, W)
        """
        return self.bits_to_map(map_id, self.map_shape)

    def map_to_bits(self, m: np.ndarray) -> int:
        """
        Encode an occupancy grid m \in {0,1}^{H\times W} into an integer by row-major bits.

        Bit k corresponds to m.flat[k] (row-major order), with least-significant bit = index 0.
        """
        flat = np.asarray(m, dtype=np.uint8).ravel(order='C')
        bits = 0
        for idx, v in enumerate(flat):
            if v:
                bits |= (1 << idx)
        return bits

    def bits_to_map(self, bits: int, shape: tuple = None) -> np.ndarray:
        """
        Decode integer bits into an occupancy grid of given shape (H,W), row-major.

        Args:
            bits: Bit representation
            shape: Optional shape (H, W). If None, uses self.map_shape
        """
        if shape is None:
            shape = self.map_shape
        H, W = shape
        total = H * W
        out = np.zeros(total, dtype=np.uint8)
        for k in range(total):
            out[k] = (bits >> k) & 1
        return out.reshape((H, W), order='C')

    def get_obstacle_segments(self, m: np.ndarray) -> list:
        """
        Extract obstacle segments from an occupancy grid for ray casting.

        This method delegates to the sensor model's get_obstacles_from_map method
        which handles connected components extraction. For direct use, call
        LIDAR.get_obstacles_from_map(m, self) instead.

        Args:
            m: Occupancy grid (H, W)

        Returns:
            list: List of obstacle segments (x1, y1, x2, y2)
        """
        # The actual implementation requires the sensor model for connected components
        # This is a placeholder - the sensor model's get_obstacles_from_map should be used
        # For now, return empty list - the sensor will handle it
        return []

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
