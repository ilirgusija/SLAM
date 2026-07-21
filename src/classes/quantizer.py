# Use CuPy backend (drop-in replacement for NumPy)
import sys
from pathlib import Path
from itertools import combinations_with_replacement
from math import comb
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from scipy.spatial import Voronoi, voronoi_plot_2d
from ..utils.array_backend import np, is_cupy
# KDTree: Use cupyx.scipy.spatial.KDTree when CuPy is available, fall back to scipy.spatial.KDTree otherwise
_use_cupy_kdtree = False
_KDTreeClass = None  # Store the actual KDTree class

if is_cupy:
    # Try multiple import strategies for CuPy KDTree
    # Strategy 1: Direct import from cupyx.scipy.spatial
    try:
        from cupyx.scipy.spatial import KDTree as _CuPyKDTree
        _KDTreeClass = _CuPyKDTree
        _use_cupy_kdtree = True
        print("✓ Using cupyx.scipy.spatial.KDTree")
    except ImportError as e1:
        # Strategy 2: Check if it's in a submodule
        try:
            import cupyx.scipy.spatial.distance as cssd
            if hasattr(cssd, 'KDTree'):
                _KDTreeClass = cssd.KDTree
                _use_cupy_kdtree = True
                print("✓ Using cupyx.scipy.spatial.distance.KDTree")
        except (ImportError, AttributeError):
            pass

        # Strategy 3: Check what's actually available
        if _KDTreeClass is None:
            try:
                import cupyx.scipy.spatial as css
                available = [x for x in dir(css) if not x.startswith('_')]
                print(f"⚠ CuPy KDTree not available. Available in cupyx.scipy.spatial: {available}")
                print(f"⚠ Falling back to scipy.spatial.KDTree")
            except Exception:
                print("⚠ CuPy KDTree not available, falling back to scipy.spatial.KDTree")

# Fall back to scipy's KDTree if CuPy KDTree is not available
if _KDTreeClass is None:
    from scipy.spatial import KDTree as _scipy_KDTree
    _KDTreeClass = _scipy_KDTree
    _use_cupy_kdtree = False
    if not is_cupy:
        print("ℹ Using scipy.spatial.KDTree (NumPy backend)")

# Create a wrapper class that handles array conversion automatically
# This avoids needing to convert arrays on every query
class _KDTreeWrapper:
    """Wrapper for KDTree that automatically handles array conversion."""

    def __init__(self, points):
        import numpy as _numpy
        # Convert points to appropriate format once during construction
        if _use_cupy_kdtree:
            # CuPy KDTree - use points as-is
            self._kdtree = _KDTreeClass(points)
        else:
            # scipy KDTree - convert to NumPy
            if hasattr(points, 'get'):  # CuPy array
                points = _numpy.asarray(points.get())
            else:
                points = _numpy.asarray(points)
            self._kdtree = _KDTreeClass(points)
        self._use_cupy = _use_cupy_kdtree

    def query(self, x, k=1, p=2, distance_upper_bound=np.inf):
        """Query the KDTree, automatically converting input arrays."""
        import numpy as _numpy
        # Convert query point if needed
        if not self._use_cupy:
            if hasattr(x, 'get'):  # CuPy array
                x = _numpy.asarray(x.get())
            else:
                x = _numpy.asarray(x)
        return self._kdtree.query(x, k=k, p=p, distance_upper_bound=distance_upper_bound)

    @property
    def data(self):
        """Access the underlying data."""
        return self._kdtree.data

    def __getattr__(self, name):
        """Delegate all other attributes to the underlying KDTree."""
        return getattr(self._kdtree, name)

# Use the wrapper class instead of raw KDTree
# This way all array conversion happens automatically
KDTree = _KDTreeWrapper

class SquareLatticeQuantizer:
    def __init__(self, x_min, x_max, y_min, y_max, n):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.n = n
        self.delta_x = (x_max - x_min) / n  # quantization resolution
        self.delta_y = (y_max - y_min) / n  # quantization resolution

    def get_quantized_points(self):
        x_vals = np.arange(self.x_min + self.delta_x / 2, self.x_max + self.delta_x / 2, self.delta_x)
        y_vals = np.arange(self.y_min + self.delta_y / 2, self.y_max + self.delta_y / 2, self.delta_y)
        xv, yv = np.meshgrid(x_vals, y_vals, indexing='xy')
        points = np.stack([xv.ravel(), yv.ravel()], axis=1)
        return points

    def get_quantized_index(self, x: np.ndarray):
        """
        Get quantized index of a point using KDTree for efficient nearest neighbor search.

        Args:
            x: point vector (2,)
        Returns:
            Index of the nearest quantized point in self.X_n
        """
        if not hasattr(self, '_kdtree'):
            self.build_index()

        # Find nearest neighbor using KDTree
        _, idx = self._kdtree.query(np.asarray(x))
        return int(idx)

    def build_index(self):
        self._points = self.get_quantized_points()
        self._kdtree = KDTree(self._points)
        self._bounds = self._compute_bounds()

    @property
    def points(self):
        """Get quantized points from KDTree data."""
        if not hasattr(self, '_kdtree'):
            self.build_index()
        return self._kdtree.data

    @property
    def n_points(self):
        """Get number of quantized points."""
        if not hasattr(self, '_kdtree'):
            self.build_index()
        return len(self._kdtree.data)

    def _compute_bounds(self):
        """Compute rectangular bounds for each quantized point (Voronoi cells)."""
        points = self._points
        bounds = []

        for point in points:
            # For square lattice: B = [x - Δx/2, x + Δx/2] × [y - Δy/2, y + Δy/2]
            bound = np.array([
                [point[0] - self.delta_x / 2, point[0] + self.delta_x / 2],  # x bounds
                [point[1] - self.delta_y / 2, point[1] + self.delta_y / 2]   # y bounds
            ])
            bounds.append(bound)

        return np.array(bounds)  # (n_points, 2, 2)

    def get_bounds(self):
        """Get precomputed bounds for all quantized points."""
        if not hasattr(self, '_bounds'):
            self.build_index()
        return self._bounds

    def find_nearest(self, x: np.ndarray):
        """Find nearest quantized point using KDTree"""
        if not hasattr(self, '_kdtree'):
            self.build_index()
        dist, idx = self._kdtree.query(np.asarray(x))
        return self._points[int(idx)], int(idx)

    def quantize(self, x):
        pos = np.array(x)
        x, y = pos[0], pos[1]
        qx = self.delta_x * np.round(x / self.delta_x)
        qy = self.delta_y * np.round(y / self.delta_y)
        return np.array([qx, qy])

    def quantization_error(self, points):
        """compute mse between original and quantized points"""
        points = np.array(points)
        quantized = self.quantize(points)
        return np.mean(np.sum((points - quantized)**2, axis=1))

    def plot_quantization(self):
        points = self.get_quantized_points()

        # plot original state space as bounding box filled with color
        plt.fill_between(
            [self.x_min, self.x_max],
            [self.y_min, self.y_min],
            [self.y_max, self.y_max],
            color='lightgray',
            alpha=0.5)
        plt.scatter(points[:, 0], points[:, 1], label='quantized')
        # plot voronoi cells
        plt.legend()
        plt.show()

        vor = Voronoi(points)
        voronoi_plot_2d(vor)
        plt.show()

class HypercubeQuantizer:
    """General d-dimensional hypercube quantizer for state spaces"""

    def __init__(self, bounds: list[tuple[float, float]], n: int):
        """
        Initialize hypercube quantizer

        Args:
            bounds: List of (min, max) tuples for each dimension, e.g., [(x_min, x_max), (y_min, y_max), ...]
            n: Number of quantization cells per dimension
        """
        self.bounds = bounds
        self.n = n
        self.dim = len(bounds)
        self.deltas = [(bmax - bmin) / n for bmin, bmax in bounds]
        self.mins = [bmin for bmin, _ in bounds]
        self.maxs = [bmax for _, bmax in bounds]

    def get_quantized_points(self):
        """Generate quantized points using meshgrid for d dimensions"""
        # Create coordinate arrays for each dimension
        coords = []
        for i, (bmin, bmax) in enumerate(self.bounds):
            delta = self.deltas[i]
            vals = np.arange(bmin + delta / 2, bmax + delta / 2, delta)
            coords.append(vals)

        # Create meshgrid for all dimensions
        mesh = np.meshgrid(*coords, indexing='ij')
        # Stack and reshape to (n^dim, dim)
        points = np.stack([m.ravel() for m in mesh], axis=1)
        return points

    def get_quantized_index(self, x: np.ndarray):
        """Get quantized index using KDTree"""
        if not hasattr(self, '_kdtree'):
            self.build_index()
        _, idx = self._kdtree.query(np.asarray(x))
        return int(idx)

    def build_index(self):
        """Build KDTree and compute bounds"""
        self._points = self.get_quantized_points()
        self._kdtree = KDTree(self._points)
        self._bounds = self._compute_bounds()

    @property
    def points(self):
        """Get quantized points"""
        if not hasattr(self, '_kdtree'):
            self.build_index()
        return self._kdtree.data

    @property
    def n_points(self):
        """Get number of quantized points"""
        if not hasattr(self, '_kdtree'):
            self.build_index()
        return len(self._kdtree.data)

    def _compute_bounds(self):
        """Compute hypercube bounds for each quantized point"""
        bounds = []
        for point in self._points:
            # Each dimension: [center - delta/2, center + delta/2]
            bound = np.array([
                [p - d / 2, p + d / 2] for p, d in zip(point, self.deltas)
            ])
            bounds.append(bound)
        return np.array(bounds)  # (n_points, dim, 2)

    def get_bounds(self):
        """Get precomputed bounds"""
        if not hasattr(self, '_bounds'):
            self.build_index()
        return self._bounds


class ObservationQuantizer:
    """
    Thin wrapper around HypercubeQuantizer for observation spaces.

    This mirrors the pattern of StateQuantizer / ActionQuantizer but for Y.
    It exposes:
        - Y_n: grid points in observation space (m_y, dim)
        - get_bounds(): axis-aligned hyper-rectangular cells (m_y, dim, 2)
    """

    def __init__(self, bounds: list[tuple[float, float]], n: int):
        """
        Args:
            bounds: List of (min, max) per observation dimension.
            n: Number of quantization cells per dimension.
        """
        self._quantizer = HypercubeQuantizer(bounds, n)

    @property
    def Y_n(self):
        """Observation grid points, shape (m_y, dim)."""
        return self._quantizer.points

    @property
    def m_y(self) -> int:
        """Number of observation points |Y_n|."""
        return self._quantizer.n_points

    def get_bounds(self):
        """
        Axis-aligned hyper-rectangular bounds for each observation point.

        Shape: (m_y, dim, 2)
        """
        return self._quantizer.get_bounds()

    def find_nearest(self, x: np.ndarray):
        """Find nearest quantized point using KDTree"""
        if not hasattr(self, '_kdtree'):
            self.build_index()
        dist, idx = self._kdtree.query(np.asarray(x))
        return self._points[int(idx)], int(idx)

    def quantize(self, x):
        """Quantize a point to nearest grid point"""
        x = np.asarray(x)
        quantized = []
        for i, (bmin, bmax) in enumerate(self.bounds):
            delta = self.deltas[i]
            q = delta * np.round((x[i] - bmin) / delta) + bmin
            quantized.append(q)
        return np.array(quantized)

class StateQuantizer:
    """Quantizes the continuous state space into discrete cells - supports 2D or 4D states"""

    def __init__(self, bounds: list[tuple[float, float]] | tuple[float, float, float, float], n: int):
        """
        Initialize state quantizer

        Args:
            bounds: For 2D: (x_min, x_max, y_min, y_max) tuple
                    For 4D: list of 4 (min, max) tuples: [(x_min, x_max), (y_min, y_max), (vx_min, vx_max), (vy_min, vy_max)]
            n: Number of quantization cells per dimension
        """
        self.n = n

        # Handle both 2D (backward compat) and 4D cases
        if isinstance(bounds, tuple) and len(bounds) == 4:
            # 2D case: (x_min, x_max, y_min, y_max)
            x_min, x_max, y_min, y_max = bounds
            self._quantizer = SquareLatticeQuantizer(x_min, x_max, y_min, y_max, n)
            self.dim = 2
        elif isinstance(bounds, list) and len(bounds) == 4:
            # 4D case: hybrid quantizer (square lattice for position, polar for velocity)

            # Extract max_vel from velocity bounds (assume symmetric)
            vx_min, vx_max = bounds[2]
            vy_min, vy_max = bounds[3]
            max_vel = max(abs(vx_min), abs(vx_max), abs(vy_min), abs(vy_max))
            pos_bounds = bounds[:2]  # [(x_min, x_max), (y_min, y_max)]
            self._quantizer = HybridStateQuantizer(pos_bounds, max_vel, n)
            self.dim = 4
        elif isinstance(bounds, list) and len(bounds) >= 2:
            # Other d-dimensional case (fallback to hypercube)
            self._quantizer = HypercubeQuantizer(bounds, n)
            self.dim = len(bounds)
        else:
            raise ValueError(f"Invalid bounds format: {bounds}")

        self._quantizer.build_index()

    @property
    def X_n(self):
        """Get quantized state points"""
        return self._quantizer.points

    @property
    def m_n(self):
        """Get number of quantized state points"""
        return self._quantizer.n_points

    @property
    def size(self):
        """Get number of quantized state points (for backward compatibility)"""
        return self._quantizer.n_points

    def get_bounds(self):
        """Get precomputed bounds for all quantized points"""
        return self._quantizer.get_bounds()

    def get_quantized_index(self, x: np.ndarray):
        """Get quantized index of a state"""
        return self._quantizer.get_quantized_index(x)

    def find_nearest(self, x: np.ndarray):
        """Find nearest quantized state"""
        return self._quantizer.find_nearest(x)

    def quantize(self, x):
        """Quantize a state to nearest grid point"""
        return self._quantizer.quantize(x)

    # Backward compatibility properties for 2D case
    @property
    def x_min(self):
        if self.dim == 2:
            return self._quantizer.x_min
        return self._quantizer.mins[0]

    @property
    def x_max(self):
        if self.dim == 2:
            return self._quantizer.x_max
        return self._quantizer.maxs[0]

    @property
    def y_min(self):
        if self.dim == 2:
            return self._quantizer.y_min
        return self._quantizer.mins[1]

    @property
    def y_max(self):
        if self.dim == 2:
            return self._quantizer.y_max
        return self._quantizer.maxs[1]

class HybridStateQuantizer:
    """Quantizes 4D state space: square lattice for position, square lattice + circle mapping for velocity."""

    def __init__(self, pos_bounds: list[tuple[float, float]], max_vel: float, n: int):
        """
        Initialize hybrid state quantizer for 4D states [x, y, vx, vy].

        Args:
            pos_bounds: List of 2 (min, max) tuples for position: [(x_min, x_max), (y_min, y_max)]
            max_vel: Maximum velocity magnitude (radius of velocity space)
            n: Number of quantization levels for both position and velocity
        """
        if n < 1:
            raise ValueError("n must be >= 1")

        self.n = int(n)
        self.max_vel = float(max_vel)

        # Position quantizer: square lattice
        x_min, x_max = pos_bounds[0]
        y_min, y_max = pos_bounds[1]
        self._pos_quantizer = SquareLatticeQuantizer(x_min, x_max, y_min, y_max, n)
        self._pos_quantizer.build_index()

        # Velocity quantizer: square lattice + circle mapping (like ActionQuantizer)
        self._build_vel_index()

        # Build combined 4D state space
        self._build_combined_index()

    def _map_to_circle(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Map points from square to circle using Lipschitz mapping.

        f(x,y) = (x*sqrt(1-(y**2)/2), y*sqrt(1-(x**2)/2))

        Args:
            x: x-coordinates (scalar or array)
            y: y-coordinates (scalar or array)

        Returns:
            Mapped points as (N, 2) array
        """
        x = np.asarray(x)
        y = np.asarray(y)

        # Handle scalar inputs
        if x.ndim == 0:
            x = np.array([x])
            y = np.array([y])

        # Apply mapping: f(x,y) = (x*sqrt(1-(y**2)/2), y*sqrt(1-(x**2)/2))
        # Clip arguments to sqrt to avoid numerical issues (shouldn't happen for |y| <= max_vel)
        u = x * np.sqrt(np.clip(1 - (y**2) / (2 * self.max_vel**2), 0, 1))
        v = y * np.sqrt(np.clip(1 - (x**2) / (2 * self.max_vel**2), 0, 1))

        return np.stack([u, v], axis=-1)

    def _build_vel_index(self):
        """Build velocity quantizer using square lattice + circle mapping.

        Uses a square lattice quantizer on [-max_vel, max_vel]^2 and maps points to the circle
        using the Lipschitz mapping: f(x,y) = (x*sqrt(1-(y**2)/2), y*sqrt(1-(x**2)/2))
        This avoids duplicate points at the origin and reuses the lattice quantizer code.
        """
        # Create square lattice quantizer on [-max_vel, max_vel]^2
        vel_lattice = SquareLatticeQuantizer(
            x_min=-self.max_vel, x_max=self.max_vel,
            y_min=-self.max_vel, y_max=self.max_vel,
            n=self.n
        )

        # Get quantized points from square lattice
        square_points = vel_lattice.get_quantized_points()  # (n*n, 2)

        # Map square points to circle
        self._vel_points = self._map_to_circle(square_points[:, 0], square_points[:, 1])

        # Filter out any points that exceed max_vel (shouldn't happen, but safety check)
        magnitudes = np.linalg.norm(self._vel_points, axis=1)
        valid_mask = magnitudes <= self.max_vel + 1e-10  # Small tolerance for numerical errors
        self._vel_points = self._vel_points[valid_mask]

        # Build KD-tree for nearest neighbor queries
        self._vel_kdtree = KDTree(self._vel_points)

    def _build_combined_index(self):
        """Build combined 4D state space: cartesian product of position and velocity."""
        pos_points = self._pos_quantizer.points  # (n_pos, 2)
        vel_points = self._vel_points  # (n_vel, 2)

        # Create all combinations: (n_pos * n_vel, 4)
        n_pos = len(pos_points)
        n_vel = len(vel_points)

        points = []
        for pos in pos_points:
            for vel in vel_points:
                points.append([pos[0], pos[1], vel[0], vel[1]])

        self._points = np.array(points, dtype=float)
        # Build KD-tree
        self._kdtree = KDTree(self._points)

        # Store bounds for each dimension
        self.mins = np.array([
            self._pos_quantizer.x_min,
            self._pos_quantizer.y_min,
            -self.max_vel,
            -self.max_vel
        ])
        self.maxs = np.array([
            self._pos_quantizer.x_max,
            self._pos_quantizer.y_max,
            self.max_vel,
            self.max_vel
        ])

    @property
    def points(self):
        """Get quantized 4D state points."""
        return self._points

    @property
    def n_points(self):
        """Get number of quantized state points."""
        return len(self._points)

    def build_index(self):
        """Build/rebuild indices."""
        self._pos_quantizer.build_index()
        self._build_vel_index()
        self._build_combined_index()

    def get_quantized_index(self, x: np.ndarray) -> int:
        """Get quantized index of a 4D state using KDTree."""
        if not hasattr(self, '_kdtree'):
            self.build_index()
        _, idx = self._kdtree.query(np.asarray(x, dtype=float))
        return int(idx)

    def find_nearest(self, x: np.ndarray):
        """Find nearest quantized state using KDTree."""
        if not hasattr(self, '_kdtree'):
            self.build_index()
        dist, idx = self._kdtree.query(np.asarray(x, dtype=float))
        return self._points[int(idx)], int(idx)

    def quantize(self, x):
        """Quantize a 4D state to nearest grid point."""
        idx = self.get_quantized_index(x)
        return self._points[idx]

    def get_bounds(self):
        """Get precomputed bounds for all quantized points."""
        # For each 4D point, compute hypercube bounds
        bounds = []
        pos_delta_x = (self._pos_quantizer.x_max - self._pos_quantizer.x_min) / self.n
        pos_delta_y = (self._pos_quantizer.y_max - self._pos_quantizer.y_min) / self.n
        # Velocity spacing: square lattice on [-max_vel, max_vel]^2 has spacing 2*max_vel/n
        # After mapping to circle, approximate spacing is similar
        vel_delta = 2 * self.max_vel / self.n

        for point in self._points:
            # Position bounds (from square lattice)
            bound = np.array([
                [point[0] - pos_delta_x / 2, point[0] + pos_delta_x / 2],  # x
                [point[1] - pos_delta_y / 2, point[1] + pos_delta_y / 2],  # y
                [point[2] - vel_delta / 2, point[2] + vel_delta / 2],      # vx (approximate)
                [point[3] - vel_delta / 2, point[3] + vel_delta / 2]       # vy (approximate)
            ])
            bounds.append(bound)

        return np.array(bounds)  # (n_points, 4, 2)

class ActionQuantizer:
    """Quantizes continuous 2D accelerations into discrete levels within an L2 ball.

    Uses a square lattice quantizer on [-max_acc, max_acc]^2 and maps points to the circle
    using the Lipschitz mapping: f(x,y) = (x*sqrt(1-(y**2)/2), y*sqrt(1-(x**2)/2))
    This avoids duplicate points at the origin and reuses the lattice quantizer code.
    """

    def __init__(self, max_acc: float, n: int):
        """
        Initialize square-lattice-based action quantizer with circle mapping.

        Args:
            max_acc: Maximum acceleration magnitude (radius of action space).
            n: Number of quantization levels per dimension in the square lattice.
        """
        if n < 1:
            raise ValueError("n must be >= 1")

        self.max_acc = float(max_acc)
        self.n = int(n)

        # Create square lattice quantizer on [-max_acc, max_acc]^2
        self._lattice = SquareLatticeQuantizer(
            x_min=-max_acc, x_max=max_acc,
            y_min=-max_acc, y_max=max_acc,
            n=n
        )

        self._build_index()

    def _map_to_circle(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Map points from square to circle using Lipschitz mapping.

        f(x,y) = (x*sqrt(1-(y**2)/2), y*sqrt(1-(x**2)/2))

        Args:
            x: x-coordinates (scalar or array)
            y: y-coordinates (scalar or array)

        Returns:
            Mapped points as (N, 2) array
        """
        x = np.asarray(x)
        y = np.asarray(y)

        # Handle scalar inputs
        if x.ndim == 0:
            x = np.array([x])
            y = np.array([y])

        # Apply mapping: f(x,y) = (x*sqrt(1-(y**2)/2), y*sqrt(1-(x**2)/2))
        # Clip arguments to sqrt to avoid numerical issues (shouldn't happen for |y| <= max_acc)
        u = x * np.sqrt(np.clip(1 - (y**2) / (2 * self.max_acc**2), 0, 1))
        v = y * np.sqrt(np.clip(1 - (x**2) / (2 * self.max_acc**2), 0, 1))

        return np.stack([u, v], axis=-1)

    def _build_index(self):
        """Build action quantizer by mapping square lattice points to circle."""
        # Get quantized points from square lattice
        square_points = self._lattice.get_quantized_points()  # (n*n, 2)

        # Map square points to circle
        self._points = self._map_to_circle(square_points[:, 0], square_points[:, 1])

        # Filter out any points that exceed max_acc (shouldn't happen, but safety check)
        magnitudes = np.linalg.norm(self._points, axis=1)
        valid_mask = magnitudes <= self.max_acc + 1e-10  # Small tolerance for numerical errors
        self._points = self._points[valid_mask]

        # Build KD-tree for nearest neighbor queries
        self._kdtree = KDTree(self._points)

    def rebuild(self):
        """Rebuild KD-tree (useful if parameters are changed externally)."""
        self._build_index()

    @property
    def U(self) -> np.ndarray:
        """Return quantized action points."""
        return self._points

    @property
    def n_u(self) -> int:
        """Number of quantized action points."""
        return self._points.shape[0]

    def build_index(self):
        """Compatibility method (delegates to rebuild)."""
        self.rebuild()

    def get_quantized_index(self, u: np.ndarray) -> int:
        """Return index of nearest quantized action using KDTree."""
        if not hasattr(self, '_kdtree'):
            self._build_index()
        _, idx = self._kdtree.query(np.asarray(u, dtype=float))
        return int(idx)

    def find_nearest(self, u: np.ndarray):
        """Return nearest quantized action and its index using KDTree."""
        if not hasattr(self, '_kdtree'):
            self._build_index()
        dist, idx = self._kdtree.query(np.asarray(u, dtype=float))
        return self._points[int(idx)], int(idx), float(dist)

    def quantize(self, u: np.ndarray) -> np.ndarray:
        """Quantize an action vector to nearest grid point."""
        idx = self.get_quantized_index(u)
        return self._points[idx]

class UniformQuantizer:
    def __init__(self, min_val, max_val, n):
        self.min_val = min_val
        self.max_val = max_val
        self.n = n
        self.delta = (max_val - min_val) / n

    def get_quantized_points(self):
        return np.linspace(start=self.min_val, stop=self.max_val, num=self.n)

class BeliefQuantizer:
    """
    Quantizer for belief space Π_n using Reznik algorithm.
    Mirrors StateQuantizer functionality but for belief spaces.
    """

    def __init__(self, M: int, N_n: int, cache_dir: str = "cache/belief_quantizer", force_regenerate: bool = False):
        """
        Initialize belief quantizer.

        Args:
            M: Parameter controlling quantization (used in Reznik algorithm)
            N_n: Dimension of belief space (size of state space + map space)
            cache_dir: Directory to store/load cached codebooks
            force_regenerate: If True, always regenerate codebook even if cached version exists
        """
        self.M = M  # Parameter controlling quantization
        self.N_n = N_n  # Dimension of belief space
        self.cardinality = comb(self.M + self.N_n - 1, self.N_n - 1)  # Actual size of belief space
        self.cache_dir = Path(cache_dir)

        # Try to load from cache first, unless force_regenerate is True
        if not force_regenerate and self._load_from_cache():
            print(f"  ✓ Loaded codebook from cache: {self.cache_file}")
        else:
            print(f"  Generating new codebook: M={self.M}, N_n={self.N_n}, cardinality={self.cardinality:,}")
            self.Π_n_M = self._generate_codebook()
            self._save_to_cache()
        # Build fast membership index
        self._build_codebook_index()

    @property
    def cache_file(self) -> Path:
        """Get the cache file path for this quantizer configuration."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"belief_quantizer_M{self.M}_N{self.N_n}.npz"

    def _load_from_cache(self) -> bool:
        """
        Try to load codebook from cache.

        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            if not self.cache_file.exists():
                return False

            # Load the cached data
            data = np.load(self.cache_file)

            # Verify the cached data matches our parameters
            cached_M = int(data['M'])
            cached_N_n = int(data['N_n'])
            cached_cardinality = int(data['cardinality'])

            if cached_M != self.M or cached_N_n != self.N_n or cached_cardinality != self.cardinality:
                print(f"  ⚠ Cached data mismatch: expected M={self.M}, N_n={self.N_n}, cardinality={self.cardinality}")
                print(f"  ⚠ Found M={cached_M}, N_n={cached_N_n}, cardinality={cached_cardinality}")
                return False

            # Load the codebook
            self.Π_n_M = data['codebook']
            print(f"  ✓ Cache validation passed")
            return True

        except Exception as e:
            print(f"  ⚠ Failed to load from cache: {e}")
            return False

    def _save_to_cache(self) -> None:
        """Save codebook to cache."""
        try:
            # Save the codebook and metadata
            np.savez_compressed(
                self.cache_file,
                codebook=self.Π_n_M,
                M=self.M,
                N_n=self.N_n,
                cardinality=self.cardinality
            )
            print(f"  ✓ Saved codebook to cache: {self.cache_file}")

            # Print cache file size
            file_size_mb = self.cache_file.stat().st_size / (1024 * 1024)
            print(f"  ✓ Cache file size: {file_size_mb:.1f} MB")

        except Exception as e:
            print(f"  ⚠ Failed to save to cache: {e}")

    def clear_cache(self) -> None:
        """Remove cached codebook file."""
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
                print(f"  ✓ Cleared cache: {self.cache_file}")
            else:
                print(f"  ℹ No cache file to clear: {self.cache_file}")
        except Exception as e:
            print(f"  ⚠ Failed to clear cache: {e}")

    @classmethod
    def list_cached_quantizers(cls, cache_dir: str = "cache/belief_quantizer") -> list:
        """
        List all cached quantizer configurations.

        Args:
            cache_dir: Directory to search for cached quantizers

        Returns:
            List of (M, N_n, cardinality, file_size_mb) tuples
        """
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            return []

        cached_quantizers = []
        for cache_file in cache_path.glob("belief_quantizer_M*_N*.npz"):
            try:
                data = np.load(cache_file)
                M = int(data['M'])
                N_n = int(data['N_n'])
                cardinality = int(data['cardinality'])
                file_size_mb = cache_file.stat().st_size / (1024 * 1024)
                cached_quantizers.append((M, N_n, cardinality, file_size_mb))
            except Exception as e:
                print(f"  ⚠ Failed to read cache file {cache_file}: {e}")

        return sorted(cached_quantizers)

    @classmethod
    def clear_all_cache(cls, cache_dir: str = "cache/belief_quantizer") -> None:
        """
        Clear all cached quantizer files.

        Args:
            cache_dir: Directory containing cached quantizers
        """
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            print(f"  ℹ No cache directory to clear: {cache_dir}")
            return

        cleared_count = 0
        total_size_mb = 0

        for cache_file in cache_path.glob("belief_quantizer_M*_N*.npz"):
            try:
                file_size_mb = cache_file.stat().st_size / (1024 * 1024)
                cache_file.unlink()
                cleared_count += 1
                total_size_mb += file_size_mb
            except Exception as e:
                print(f"  ⚠ Failed to clear cache file {cache_file}: {e}")

        print(f"  ✓ Cleared {cleared_count} cache files")
        print(f"  ✓ Freed {total_size_mb:.1f} MB of disk space")

    @classmethod
    def cache_stats(cls, cache_dir: str = "cache/belief_quantizer") -> dict:
        """
        Get statistics about cached quantizers.

        Args:
            cache_dir: Directory containing cached quantizers

        Returns:
            Dictionary with cache statistics
        """
        cached_quantizers = cls.list_cached_quantizers(cache_dir)

        if not cached_quantizers:
            return {
                'count': 0,
                'total_size_mb': 0,
                'total_cardinality': 0,
                'largest_file_mb': 0,
                'largest_cardinality': 0
            }

        total_size_mb = sum(size_mb for _, _, _, size_mb in cached_quantizers)
        total_cardinality = sum(cardinality for _, _, cardinality, _ in cached_quantizers)
        largest_file_mb = max(size_mb for _, _, _, size_mb in cached_quantizers)
        largest_cardinality = max(cardinality for _, _, cardinality, _ in cached_quantizers)

        return {
            'count': len(cached_quantizers),
            'total_size_mb': total_size_mb,
            'total_cardinality': total_cardinality,
            'largest_file_mb': largest_file_mb,
            'largest_cardinality': largest_cardinality,
            'quantizers': cached_quantizers
        }

    def _build_codebook_index(self):
        """Create integer and key views to enable O(log K) membership checks."""
        # Integer representation of codebook probabilities
        # Codebook is already generated in lexicographic order, so no need to sort
        codebook_int = np.rint(self.Π_n_M * self.M).astype(np.int64)
        # Ensure contiguous before view
        codebook_int_c = np.ascontiguousarray(codebook_int)
        self._codebook_int = codebook_int_c

        # Structured dtype to leverage lexicographic ordering and searchsorted
        # Note: CuPy doesn't support structured dtypes, so convert to NumPy for this operation
        import numpy as _numpy
        if is_cupy:
            # Convert CuPy array to NumPy for structured dtype operations
            codebook_int_np = _numpy.asarray(codebook_int_c.get())
        else:
            codebook_int_np = _numpy.asarray(codebook_int_c)
        dtype = _numpy.dtype([(f'f{i}', _numpy.int64) for i in range(self.N_n)])
        codebook_int_np_contig = _numpy.ascontiguousarray(codebook_int_np)
        self._codebook_key_dtype = dtype
        self._codebook_keys = codebook_int_np_contig.view(dtype).reshape(-1)

    @staticmethod
    def reznik_algorithm(z: np.ndarray, N_n: int, M: int):
        """
        Reznik algorithm for belief space quantization.

        Args:
            z: Belief vector (N_n,)
        Returns:
            Quantized belief vector (N_n,)
        """
        assert z.shape[0] == N_n
        assert M > 0
        s = np.sum(z)
        assert np.abs(s - 1.0) <= 1e-10

        k_prime = np.zeros(N_n)
        k_prime = np.floor(M * z + 0.5)
        # Work in integer space for exact combinatorics
        k_prime_i = k_prime.astype(np.int64)
        M_prime = 0
        for i in range(N_n):
            M_prime += int(k_prime_i[i])
        Delta = M_prime - int(M)

        if Delta == 0:
            # Return exact multiples of 1/M as float64
            return k_prime_i.astype(np.float64) / M

        delta = np.zeros(N_n)
        delta = k_prime - M * z

        # sort delta in increasing order
        sorted_indices = np.argsort(delta)

        k_i = np.zeros(N_n, dtype=np.int64)
        if Delta > 0:
            # Decrease the largest deltas: last Delta elements in sorted order
            for pos in range(N_n):
                idx = sorted_indices[pos]
                if pos >= N_n - Delta:
                    k_i[idx] = k_prime_i[idx] - 1
                else:
                    k_i[idx] = k_prime_i[idx]
        else:
            # Increase the smallest deltas: first -Delta elements in sorted order
            inc = -Delta
            for pos in range(N_n):
                idx = sorted_indices[pos]
                if pos < inc:
                    k_i[idx] = k_prime_i[idx] + 1
                else:
                    k_i[idx] = k_prime_i[idx]

        # Ensure the result sums to 1
        result = k_i.astype(np.float64) / M
        # Normalize to ensure sum = 1 (no-op if arithmetic exact)
        result = result / np.sum(result)
        # Match codebook dtype to avoid tiny numerical mismatches in comparisons
        return result.astype(np.float64, copy=False)

    def _generate_codebook(self):
        """
        fastest version using numpy + itertools.
        leverages C-level iteration.
        """

        print(f"  Generating codebook: M={self.M}, N_n={self.N_n}, cardinality={self.cardinality:,}")

        # preallocate
        t0 = time.time()
        codebook = np.zeros((self.cardinality, self.N_n), dtype=np.float32)
        t1 = time.time()
        print(f"  Array allocation: {t1-t0:.6f}s")

        # this is the key optimization: bincount is C-level
        t0 = time.time()
        combo_iter = combinations_with_replacement(range(self.N_n), self.M)
        t1 = time.time()
        print(f"  Iterator creation: {t1-t0:.6f}s")

        print(f"  Starting combination iteration...")
        t0 = time.time()

        # Use tqdm for progress tracking
        for i, combo in enumerate(tqdm(combo_iter, total=self.cardinality, desc="Generating combinations", miniters=1000000)):
            codebook[i] = np.bincount(np.asarray(combo), minlength=self.N_n).astype(np.float32)

            # Print progress every 1000000 iterations for debugging
            if i % 1000000 == 0 and i > 0:
                elapsed = time.time() - t0
                rate = i / elapsed
                print(
                    f"    Progress: {i:,}/{self.cardinality:,} ({i/self.cardinality*100:.1f}%) - Rate: {rate:.0f} combos/sec")

        t1 = time.time()
        print(f"  Combination iteration: {t1-t0:.6f}s")

        t0 = time.time()
        codebook /= self.M  # vectorized division
        t1 = time.time()
        print(f"  Normalization: {t1-t0:.6f}s")

        print(f"  ✓ Codebook generation complete!")
        return codebook

    def is_in_voronoi_cell(self, belief: np.ndarray, cell_idx: int, tol=1e-8):
        """
        check if belief ∈ voronoi cell via polytope constraints
        """
        A, b = self.voronoi_cells[cell_idx]
        return np.all(A @ belief <= b + tol)

    def find_nearest(self, belief: np.ndarray):
        """
        Find nearest quantized belief to given belief.

        Args:
            belief: Belief vector (N_n,)
        Returns:
            Nearest quantized belief and index
        """

        # Use L2 distance for belief space
        distances = np.linalg.norm(self.Π_n_M - belief, axis=1)
        idx = np.argmin(distances)

        return self.Π_n_M[idx], idx

    def quantize(self, belief: np.ndarray):
        """
        Quantize a belief vector.

        Args:
            belief: Belief vector (N_n,)
        Returns:
            Quantized belief vector (N_n,)
        """
        return self.reznik_algorithm(belief, self.N_n, self.M)

    # Codebook look up
    def is_in_codebook_distance(self, q: np.ndarray, atol: float = 1e-10):
        """Check if q is in codebook using distance-based method (original approach).

        Returns:
            (is_in_codebook, min_distance, index_of_nearest)
        """
        distances = np.linalg.norm(self.Π_n_M - q.astype(self.Π_n_M.dtype, copy=False), axis=1)
        min_distance = np.min(distances)
        idx = int(np.argmin(distances))
        return (min_distance < atol, float(min_distance), idx)

    def is_in_codebook(self, q: np.ndarray) -> int:
        """Check if q is in codebook by exact integer match and return index.

        Uses linear search over the codebook. Returns the index if found, -1 if not found.
        The return value is truthy when found (index >= 0), falsy when not found (-1).

        Returns:
            Index of q in codebook if found, else -1
        """
        q_int = np.rint(q * self.M).astype(np.int64)
        s = int(np.sum(q_int))
        if s != int(self.M):
            return -1

        # Check all codebook entries for exact integer match
        for idx, codebook_int_row in enumerate(self._codebook_int):
            if np.array_equal(codebook_int_row, q_int):
                return idx
        return -1

    def find_codebook_index(self, q: np.ndarray) -> int:
        """Linear search: return index of q in codebook if present, else -1."""
        q_int = np.rint(q * self.M).astype(np.int64)
        if int(np.sum(q_int)) != int(self.M):
            return -1

        # Linear search for exact match
        for idx, codebook_int_row in enumerate(self._codebook_int):
            if np.array_equal(codebook_int_row, q_int):
                return idx
        return -1

    def debug_nearest(self, q: np.ndarray, atol: float = 1e-8):
        """Diagnose mismatches: return nearest codebook vector and diffs.

        Returns a dict with keys: in_codebook, in_codebook_linear, idx, q_int, q_int_sum, expected_sum, nearest_int, nearest, distance, diff_int
        """
        in_cb_idx = self.is_in_codebook(q)
        in_cb_dist, min_distance, nearest_idx_dist = self.is_in_codebook_distance(q)

        q_int = np.rint(q * self.M).astype(np.int64)

        result = {
            'in_codebook_idx': in_cb_idx,
            'in_codebook': (in_cb_idx >= 0),
            'in_codebook_distance': in_cb_dist,
            'q_int': q_int.copy(),
            'q_int_sum': int(np.sum(q_int)),
            'expected_sum': int(self.M),
        }

        if in_cb_idx >= 0:
            idx = in_cb_idx
            result['idx'] = idx
            result['nearest_int'] = self._codebook_int[idx].copy()
            result['nearest'] = (self._codebook_int[idx].astype(np.float64) / self.M).astype(self.Π_n_M.dtype)
            result['distance'] = 0.0
            result['diff_int'] = np.zeros_like(q_int)
        else:
            # Not found: compute nearest by distance
            distances = np.linalg.norm(self.Π_n_M - q.astype(self.Π_n_M.dtype, copy=False), axis=1)
            idx = int(np.argmin(distances))
            nearest = self.Π_n_M[idx]
            nearest_int = np.rint(nearest * self.M).astype(np.int64)
            result['idx'] = idx
            result['nearest'] = nearest
            result['nearest_int'] = nearest_int.copy()
            result['distance'] = float(distances[idx])
            result['diff_int'] = (nearest_int - q_int)
            result['allclose'] = bool(np.allclose(nearest, q, atol=atol))
        return result
