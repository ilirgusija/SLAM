import math
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d, KDTree
import matplotlib.pyplot as plt

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
        if not hasattr(self, '_kdtree'):
            self.build_index()
        dist, idx = self._kdtree.query(np.asarray(x))
        return self._points[idx], int(idx)

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

class StateQuantizer(SquareLatticeQuantizer):
    """Quantizes the continuous state space into discrete cells"""

    def __init__(self, x_min, x_max, y_min, y_max, n):
        """
        Initialize state quantizer
        Args:
            x_min, x_max: X coordinate bounds
            y_min, y_max: Y coordinate bounds
            n: Number of quantization cells in each dimension
        """
        super().__init__(x_min, x_max, y_min, y_max, n)
        self.build_index()  # Initialize KDTree

    @property
    def X_n(self):
        """Get quantized state points from KDTree."""
        return self.points

    @property
    def m_n(self):
        """Get number of quantized state points."""
        return self.n_points

    @property
    def size(self):
        """Get number of quantized state points (for backward compatibility)."""
        return self.n_points

class ActionQuantizer(SquareLatticeQuantizer):
    """Quantizes continuous 2D velocities into discrete levels using KD-tree for efficient nearest neighbor search"""

    def __init__(self, max_vel, n):
        """
        Initialize velocity quantizer with KD-tree for efficient nearest neighbor search

        Args:
            max_vel: Maximum velocity magnitude
            n: Number of quantization levels for each dimension
        """
        super().__init__(-max_vel, max_vel, -max_vel, max_vel, n)
        self.build_index()

    @property
    def U(self):
        """Get quantized action points from KDTree."""
        return self.points

    @property
    def n_u(self):
        """Get number of quantized action points."""
        return self.n_points

    def get_quantized_index(self, u: np.ndarray):
        """
        Get quantized index of a velocity using KDTree for efficient nearest neighbor search.

        Args:
            u: velocity vector (2,)
        Returns:
            Index of the nearest quantized velocity in self.U
        """
        if not hasattr(self, '_kdtree'):
            self.build_index()

        # Find nearest neighbor using KDTree
        _, idx = self._kdtree.query(np.asarray(u))
        return int(idx)

class UniformQuantizer:
    def __init__(self, min_val, max_val, n):
        self.min_val = min_val
        self.max_val = max_val
        self.n = n
        self.delta = (max_val - min_val) / n

    def get_quantized_points(self):
        return np.linspace(start=self.min_val, stop=self.max_val, num=self.n)

class ObservationQuantizer(UniformQuantizer):
    """Quantizes the continuous observation space into discrete cells"""

    def __init__(self, y_max, n, B):
        """
        Initialize observation quantizer

        Args:
            y_max: Maximum observation value
            n: Number of quantization cells
            B: Number of beams (dimensions)
        """
        super().__init__(0, y_max, n)
        self.B = B

    def get_quantized_points(self):
        """
        Returns all permutations of quantized points for B beams.
        Shape: (n^B, B) where n is number of quantization levels per beam
        """
        # Get base quantized values (1D array of length n)
        base_values = super().get_quantized_points()

        # Generate all permutations using meshgrid
        # This creates all possible combinations of the n values across B beams
        mesh = np.meshgrid(*[base_values for _ in range(self.B)], indexing='ij')

        # Stack and reshape to get shape (n^B, B)
        all_permutations = np.stack(mesh, axis=-1).reshape(-1, self.B)

        return all_permutations

    def build_index(self):
        self._points = self.get_quantized_points()
        self._kdtree = KDTree(self._points)

    def find_nearest(self, y: np.ndarray):
        if not hasattr(self, '_kdtree'):
            self.build_index()
        dist, idx = self._kdtree.query(np.asarray(y))
        return self._points[idx], int(idx)

class BeliefQuantizer:
    """
    Quantizer for belief space Π_n using Reznik algorithm.
    Mirrors StateQuantizer functionality but for belief spaces.
    """

    def __init__(self, M: int, N_n: int):
        """
        Initialize belief quantizer.

        Args:
            M: Parameter controlling quantization (used in Reznik algorithm)
            N_n: Dimension of belief space (size of state space + map space)
        """
        self.M = M  # Parameter controlling quantization
        self.N_n = N_n  # Dimension of belief space
        self.cardinality = math.comb(self.M + self.N_n - 1, self.N_n - 1)  # Actual size of belief space
        self.Π_n_M = self._generate_codebook()
        self.voronoi_cells = self._compute_bounds()

    def reznik_algorithm(self, z: np.ndarray):
        """
        Reznik algorithm for belief space quantization.

        Args:
            z: Belief vector (N_n,)
        Returns:
            Quantized belief vector (N_n,)
        """
        assert len(z) == self.N_n
        assert np.isclose(np.sum(z), 1.0, atol=1e-10), f"Sum is {np.sum(z)}, expected 1.0"
        assert self.M > 0

        k_prime = np.zeros(self.N_n)
        k_prime = np.floor(self.M * z + 0.5)
        M_prime = np.sum(k_prime)
        Delta = M_prime - self.M

        if Delta == 0:
            return k_prime / self.M

        delta = np.zeros(self.N_n)
        delta = k_prime - self.M * z

        # sort delta in increasing order
        sorted_indices = np.argsort(delta)

        k = np.zeros(self.N_n)
        if Delta > 0:
            condition = (sorted_indices + 1) <= self.N_n - Delta - 1
            k = np.where(condition, k_prime, k_prime - 1)
        else:
            condition = (sorted_indices + 1) <= abs(Delta)
            k = np.where(condition, k_prime + 1, k_prime)

        # Ensure the result sums to 1
        result = k / self.M
        result = result / np.sum(result)  # Normalize to ensure sum = 1
        return result

    def _generate_codebook(self):
        """
        Get quantized belief space Π_n_M.
        Generate exactly cardinality number of quantization points.

        Returns:
            Codebook of quantized beliefs (cardinality, N_n)
        """
        # Generate cardinality random beliefs and quantize them
        Π_n = np.random.dirichlet(np.ones(self.N_n), size=self.cardinality)
        # Ensure each belief sums to 1
        for i in range(self.cardinality):
            Π_n[i] = Π_n[i] / np.sum(Π_n[i])

        # Quantize each sample
        quantized = [self.reznik_algorithm(π_i) for π_i in Π_n]
        codebook = np.array(quantized)

        return codebook

    def _compute_bounds(self):
        """
        compute voronoi cell as polytope (A, b) where cell = {x ∈ Π : Ax ≤ b}
        """
        voronoi_cells = []

        for center_idx, q in enumerate(self.Π_n_M):
            A_list = []
            b_list = []

            # voronoi condition: ||p - q||₂ ≤ ||p - q'||₂ for all q' ≠ q
            # equivalent to: 2p^T(q' - q) ≤ ||q'||² - ||q||²
            for i, q_prime in enumerate(self.Π_n_M):
                if i == center_idx:
                    continue

                a = 2 * (q_prime - q)
                b_val = np.linalg.norm(q_prime)**2 - np.linalg.norm(q)**2

                A_list.append(a)
                b_list.append(b_val)

            # simplex constraints: p_i ≥ 0 (encoded as -p_i ≤ 0)
            A_simplex = -np.eye(self.N_n)
            b_simplex = np.zeros(self.N_n)

            # equality constraint Σp_i = 1 becomes two inequalities:
            # Σp_i ≤ 1 and -Σp_i ≤ -1
            A_sum = np.array([np.ones(self.N_n), -np.ones(self.N_n)])
            b_sum = np.array([1.0, -1.0])

            A = np.vstack([np.array(A_list), A_simplex, A_sum])
            b = np.hstack([np.array(b_list), b_simplex, b_sum])

            voronoi_cells.append((A, b))

        return voronoi_cells

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
        return self.reznik_algorithm(belief)



# example usage
if __name__ == "__main__":
    # test error bound
    test_points = np.random.uniform([0, 0], [2, 1], (1000, 2))
    square_quantizer = SquareLatticeQuantizer(0, 3, 0, 1, n=10)
    X_n_square = square_quantizer.get_quantized_points()
    quantized_square = square_quantizer.quantize(test_points)
    errors_square = np.linalg.norm(test_points - quantized_square, axis=1)
    max_error_square = np.max(errors_square)

    print(f"(square) max quantization error: {max_error_square:.6f}")
    print(f"required bound 1/n = {1/10:.6f}")
    print(f"bound satisfied: {max_error_square <= 1/10}")

    square_quantizer.plot_quantization()
