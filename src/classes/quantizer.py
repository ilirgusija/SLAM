import numpy as np
from sklearn.neighbors import KDTree
from multiprocessing import Pool
import itertools
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

class SquareLatticeQuantizer:
    def __init__(self, x_min, x_max, y_min, y_max, n):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.n = n
        self.delta = 1.0 / n  # quantization resolution
        X_n = self.get_quantized_points()
        self.size = len(X_n)

    def get_quantized_points(self):
        # x_vals = np.arange(self.x_min+self.delta/2, self.x_max + self.delta/2, self.delta)
        # y_vals = np.arange(self.y_min+self.delta/2, self.y_max + self.delta/2, self.delta)
        x_vals = np.arange(self.x_min+self.delta/2, self.x_max + self.delta/2, self.delta)
        y_vals = np.arange(self.y_min+self.delta/2, self.y_max + self.delta/2, self.delta)
        xv, yv = np.meshgrid(x_vals, y_vals, indexing='xy')
        points = np.stack([xv.ravel(), yv.ravel()], axis=1)
        return points

    def quantize(self, x):
        x = np.array(x)
        qx = self.delta * np.round(x / self.delta)
        return qx
    
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
    
class UniformQuantizer:
    def __init__(self, min_val, max_val, n):
        self.min_val = min_val
        self.max_val = max_val
        self.n = n
        self.delta = (max_val - min_val) / n
    
    def get_quantized_points(self):
        return np.linspace(self.min_val, self.max_val, self.n)
    
class ObservationQuantizer(UniformQuantizer):
    """Quantizes the continuous observation space into discrete cells"""

    def __init__(self, y_max, n):
        """
        Initialize observation quantizer

        Args:
            y_min, y_max: Y coordinate bounds
            n: Number of quantization cells
        """
        super().__init__(0, y_max, n)

def reznik_algorithm(z: np.ndarray, M: int, N_n: int):
    """
    Reznik algorithm for belief space quantization
    """
    assert len(z) == N_n
    assert np.sum(z) == 1
    assert M > 0
    
    k_prime = np.zeros(N_n)
    k_prime = np.floor(M*z + 0.5)
    M_prime = np.sum(k_prime)
    Delta = M_prime - M
    
    if Delta == 0:
        return k_prime/M
    
    delta = np.zeros(N_n)
    delta = k_prime - M*z
    
    # sort delta in increasing order        
    sorted_indices = np.argsort(delta)
    
    k = np.zeros(N_n)
    if Delta > 0:
        condition = (sorted_indices+1) <= N_n - Delta - 1
        k = np.where(condition, k_prime, k_prime-1)
    else:
        condition = (sorted_indices+1) <= abs(Delta)
        k = np.where(condition, k_prime+1, k_prime)
    return k/M


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