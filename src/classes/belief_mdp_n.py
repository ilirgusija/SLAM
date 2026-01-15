import hashlib
import time
from pathlib import Path
from contextlib import contextmanager
from .mapping import LidarGridMapVec
from .model import LIDAR, SingleIntegratorModel, DoubleIntegratorModel
from .obstacle import Obstacle
from .quantizer import StateQuantizer, ActionQuantizer
from abc import ABC, abstractmethod
from .pomdp import SLAM_POMDP, Localization_POMDP, Mapping_POMDP
try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm
# Use CuPy backend for GPU acceleration (falls back to NumPy if not available)
from ..utils.array_backend import np, random, is_cupy
import numpy as _numpy  # For file I/O only

# Import logsumexp from appropriate backend
if is_cupy:
    from cupyx.scipy.special import logsumexp
else:
    from scipy.special import logsumexp


class BaseBeliefMDP_n(ABC):
    """
    Base Belief-MDP class with common functionality and abstract methods.

    Belief-MDP is defined as a four tuple (Π_n, U_n, η, c_tilde).
    Subclasses must implement problem-specific belief update and cost functions.

    Common functionality:
    - State and action quantizers (SQ, AQ)
    - Transition matrix T_mat (common across all variants)
    - Cache management for T_mat
    """

    def _init_common(
        self,
        n: int,
        motion_model: SingleIntegratorModel | DoubleIntegratorModel,
        _map: LidarGridMapVec
    ):
        """
        Initialize common BeliefMDP_n components (quantizers and T_mat).

        This is called by subclasses after their POMDP parent is initialized.

        Args:
            n: Quantization level
            motion_model: Motion model (SingleIntegrator or DoubleIntegrator)
            _map: Map object
        """
        self.n = n

        def to_float(x):
            return float(x.item() if hasattr(x, 'item') else x)

        if isinstance(motion_model, SingleIntegratorModel):
            ll = _map.occupancy_map.left_lower
            ru = _map.occupancy_map.right_upper
            state_bounds = np.array([
                [to_float(ll[0]), to_float(ru[0])],
                [to_float(ll[1]), to_float(ru[1])]
            ], dtype=float)
            self.SQ = StateQuantizer(
                (to_float(ll[0]), to_float(ru[0]), to_float(ll[1]), to_float(ru[1])),
                n
            )
            self.state_dim = 2
            self.AQ = ActionQuantizer(motion_model.max_v, n)
        elif isinstance(motion_model, DoubleIntegratorModel):
            ll = _map.occupancy_map.left_lower
            ru = _map.occupancy_map.right_upper
            v_max = 5.0
            bounds_list = [
                (to_float(ll[0]), to_float(ru[0])),
                (to_float(ll[1]), to_float(ru[1])),
                (-v_max, v_max),
                (-v_max, v_max)
            ]
            state_bounds = np.array(bounds_list, dtype=float)
            self.SQ = StateQuantizer(bounds_list, n)
            self.state_dim = 4
            self.AQ = ActionQuantizer(motion_model.max_a, n)
        else:
            raise ValueError(f"Unsupported motion model type: {type(motion_model)}")

        self.state_bounds = state_bounds
        if hasattr(motion_model, "set_state_bounds"):
            motion_model.set_state_bounds(self.state_bounds)

        self.map_H, self.map_W = _map.occupancy_map.height, _map.occupancy_map.width
        self.len_M = 2**(self.map_H * self.map_W)

        cache_path = self._get_cache_path()
        if self._load_T_mat(cache_path):
            print(f"Loaded cached T_mat from {cache_path}")
        else:
            compatible_cache = self._find_compatible_cache()
            if compatible_cache and self._load_T_mat(compatible_cache):
                print(f"Loaded cached T_mat from {compatible_cache} (compatible old cache)")
            else:
                print("Computing T_mat for the first time...")
                self.T_mat = self._compute_T_n()
                self._save_T_mat(cache_path)
                print(f"Saved T_mat to {cache_path}")

    def _get_cache_path(self) -> Path:
        """Generate cache file path based on quantization parameters and map properties."""
        cache_dir = Path(__file__).parent.parent.parent / "cache/T_mat"
        cache_dir.mkdir(parents=True, exist_ok=True)

        max_val = self.motion_model.max_v if hasattr(self.motion_model, 'max_v') else self.motion_model.max_a
        metadata = {
            'n': self.n,
            'state_bounds': self.state_bounds.flatten().tolist(),
            'max_val': max_val,
            'dt': float(self.motion_model.dt),
            'map_shape': (self.map_H, self.map_W),
            'sigma_w': float(self.σ_w),
            'model': self.motion_model.__class__.__name__,
            'state_dim': self.state_dim,
            'kernel_version': 'v7-square-lattice-circle-map',
            'action_space': 'square-lattice-circle-map',
            'action_n': getattr(self.AQ, 'n', None)
        }

        metadata_hash = hashlib.md5(str(sorted(metadata.items())).encode()).hexdigest()[:8]
        filename = f"T_mat_n{self.n}_map{self.map_H}x{self.map_W}_max{max_val}_{metadata_hash}.npz"

        return cache_dir / filename

    def _find_compatible_cache(self) -> Path | None:
        """Search for compatible cache files (backward compatibility)."""
        cache_dir = Path(__file__).parent.parent.parent / "cache/T_mat"
        if not cache_dir.exists():
            return None

        max_val = self.motion_model.max_v if hasattr(self.motion_model, 'max_v') else self.motion_model.max_a
        pattern = f"T_mat_n{self.n}_map{self.map_H}x{self.map_W}_max{max_val}_*.npz"

        for cache_file in cache_dir.glob(pattern):
            if self._load_T_mat(cache_file):
                return cache_file

        return None

    def _compute_K_mask(self) -> np.ndarray:
        """Compute feasibility mask: True iff applying u_k at x_i stays within bounds."""
        m_n, n_u = self.SQ.m_n, self.AQ.n_u
        K_mask = np.zeros((m_n, n_u), dtype=bool)
        X = self.SQ.X_n

        if self.state_dim == 2:
            dim_bounds = [(self.SQ.x_min, self.SQ.x_max), (self.SQ.y_min, self.SQ.y_max)]
        else:
            dim_bounds = [(self.SQ._quantizer.mins[d], self.SQ._quantizer.maxs[d]) for d in range(self.state_dim)]

        for k in range(n_u):
            X_next = self.motion_model.f_bar(X, self.AQ.U[k])
            valid = np.ones(m_n, dtype=bool)
            for d in range(self.state_dim):
                d_min, d_max = dim_bounds[d]
                valid = valid & (X_next[:, d] >= d_min) & (X_next[:, d] <= d_max)
            K_mask[:, k] = valid

        return K_mask

    @property
    def K_mask(self) -> np.ndarray:
        """
        Lazy property to compute feasibility mask K_mask over state-action pairs.
        Computed on-demand, not cached to disk.

        Returns:
            Boolean array (m_n, n_u) where True iff applying u_k at x_i stays within X_n bounds.
        """
        return self._compute_K_mask()

    def _get_all_maps_cache_path(self) -> Path:
        """Generate cache file path for all maps."""
        cache_dir = Path(__file__).parent.parent.parent / "cache/all_maps"
        cache_dir.mkdir(parents=True, exist_ok=True)

        filename = f"all_maps_H{self.map_H}_W{self.map_W}.npz"
        return cache_dir / filename

    def _load_or_generate_all_maps(self):
        """Load or generate all maps for vectorized operations."""
        cache_path = self._get_all_maps_cache_path()

        if cache_path.exists():
            try:
                data = _numpy.load(str(cache_path))
                maps_3d = np.asarray(data['maps_3d'])
                map_bits = np.asarray(data['map_bits'])
                if maps_3d.shape == (self.len_M, self.map_H, self.map_W):
                    return maps_3d, map_bits
            except Exception as e:
                print(f"Error loading all_maps cache: {e}")

        print(f"Generating all {self.len_M} maps (this may take a moment)...")
        maps_3d, map_bits = self.generate_all_maps_3d(show_progress=True)

        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            maps_3d_np = maps_3d.get() if hasattr(maps_3d, 'get') else maps_3d
            map_bits_np = map_bits.get() if hasattr(map_bits, 'get') else map_bits
            _numpy.savez_compressed(str(cache_path), maps_3d=maps_3d_np, map_bits=map_bits_np)
            print(f"Saved all_maps cache to {cache_path}")
        except Exception as e:
            print(f"Error saving all_maps cache: {e}")

        return maps_3d, map_bits

    def _save_T_mat(self, cache_path: Path) -> None:
        """Save T_mat to cache file with metadata."""
        state_bounds_save = self.state_bounds.astype(float)
        state_bounds_save = state_bounds_save.get() if hasattr(state_bounds_save, 'get') else state_bounds_save
        T_mat_save = self.T_mat.get() if hasattr(self.T_mat, 'get') else self.T_mat

        max_val = self.motion_model.max_v if hasattr(self.motion_model, 'max_v') else self.motion_model.max_a
        metadata = {
            'n': self.n,
            'state_bounds': state_bounds_save,
            'max_val': max_val,
            'dt': float(self.motion_model.dt),
            'map_shape': _numpy.array([self.map_H, self.map_W]),
            'sigma_w': float(self.σ_w),
            'm_n': self.SQ.m_n,
            'n_u': self.AQ.n_u,
            'model': self.motion_model.__class__.__name__,
            'state_dim': self.state_dim,
            'kernel_version': 'v7-square-lattice-circle-map',
            'action_space': 'square-lattice-circle-map',
            'action_n': getattr(self.AQ, 'n', None)
        }

        _numpy.savez_compressed(cache_path, T_mat=T_mat_save, **metadata)

    def _load_T_mat(self, cache_path: Path) -> bool:
        """Load T_mat from cache file if it exists and matches current parameters."""
        if not cache_path.exists():
            return False

        try:
            data = _numpy.load(str(cache_path))
            state_bounds_np = self.state_bounds.astype(float)
            state_bounds_np = state_bounds_np.get() if hasattr(state_bounds_np, 'get') else state_bounds_np

            max_val = self.motion_model.max_v if hasattr(self.motion_model, 'max_v') else self.motion_model.max_a
            expected_metadata = {
                'n': self.n,
                'state_bounds': state_bounds_np,
                'max_val': max_val,
                'dt': float(self.motion_model.dt),
                'map_shape': _numpy.array([self.map_H, self.map_W]),
                'sigma_w': float(self.σ_w),
                'm_n': self.SQ.m_n,
                'n_u': self.AQ.n_u,
                'model': self.motion_model.__class__.__name__,
                'state_dim': self.state_dim,
                'kernel_version': 'v7-square-lattice-circle-map',
                'action_space': 'square-lattice-circle-map',
                'action_n': getattr(self.AQ, 'n', None)
            }

            for key, expected_value in expected_metadata.items():
                if key not in data:
                    return False
                data_value = data[key]
                if isinstance(expected_value, _numpy.ndarray):
                    if not _numpy.array_equal(data_value, expected_value):
                        return False
                else:
                    if isinstance(data_value, _numpy.ndarray):
                        try:
                            data_value = data_value.item()
                        except ValueError:
                            return False
                    if data_value != expected_value:
                        return False

            self.T_mat = np.asarray(data['T_mat'])
            if self.T_mat.shape != (self.SQ.m_n, self.SQ.m_n, self.AQ.n_u):
                return False

            return True

        except Exception as e:
            print(f"Error loading cache: {e}")
            return False

    def _compute_T_n(self) -> np.ndarray:
        """Compute transition matrix T_mat over all state-action pairs."""
        import numpy as numpy_cpu

        T_mat = numpy_cpu.zeros((self.SQ.m_n, self.SQ.m_n, self.AQ.n_u), dtype=numpy_cpu.float64)
        bounds = numpy_cpu.asarray(self.SQ.get_bounds())
        X_n_cpu = numpy_cpu.asarray(self.SQ.X_n)
        K_mask_cpu = numpy_cpu.asarray(self._compute_K_mask())

        for k in tqdm(range(self.AQ.n_u), desc="Computing T_mat"):
            u = numpy_cpu.asarray(self.AQ.U[k])

            for j in range(self.SQ.m_n):
                transition_probs = self.T(bounds[j], X_n_cpu, u)
                feasible_pairs = K_mask_cpu[:, k].astype(numpy_cpu.float64)
                T_mat[j, :, k] = transition_probs * feasible_pairs

            col_sums = T_mat[:, :, k].sum(axis=0, keepdims=True)
            col_sums = numpy_cpu.where(col_sums == 0.0, 1.0, col_sums)
            T_mat[:, :, k] = T_mat[:, :, k] / col_sums

        return np.asarray(T_mat)


    @abstractmethod
    def F_batch(self, π, *args, Y_batch):
        """
        Batched filter update equation. Must be implemented by subclasses.

        Note: Signature varies by problem type:
        - SLAM/Localization: F_batch(π, u, Y_batch)
        - Mapping: F_batch(π, x_next, Y_batch) - uses known pose instead of action
        """
        raise NotImplementedError

    @abstractmethod
    def F_batch_log(self, π, *args, Y_batch):
        """
        Batched filter update equation in log-space. Must be implemented by subclasses.

        Note: Signature varies by problem type:
        - SLAM/Localization: F_batch_log(π, u, Y_batch)
        - Mapping: F_batch_log(π, x_next, Y_batch) - uses known pose instead of action
        """
        raise NotImplementedError

    @abstractmethod
    def c_tilde_n(self, π, *args):
        """
        Expected cost function for belief-MDP. Must be implemented by subclasses.

        Note: Signature varies by problem type:
        - SLAM: c_tilde_n(π, u)
        - Localization: c_tilde_n(π, u)
        - Mapping: c_tilde_n(π, x_current, u) - may need current pose for cost computation
        """
        raise NotImplementedError

    @abstractmethod
    def sample_observations_batch(self, π, *args, batch_size):
        """
        Sample observations from the model. Must be implemented by subclasses.

        Note: Signature varies by problem type:
        - SLAM/Localization: sample_observations_batch(π, u, batch_size)
        - Mapping: sample_observations_batch(π, x_next, batch_size) - uses known pose instead of action
        """
        raise NotImplementedError

    @abstractmethod
    def η_n(self, π_new: np.ndarray, π: np.ndarray, *args, n_samples: int = 50000,
            seed: int = None, batch_size: int = 500, show_progress: bool = True) -> float:
        """
        Belief transition probability. Must be implemented by subclasses.

        Note: Signature varies by problem type:
        - SLAM/Localization: η_n(π_new, π, u, ...)
        - Mapping: η_n(π_new, π, x_current, u, x_next, ...) - computes joint probability η(π', x_next | b_t, u)
          where b_t = (x_t, π_t) is the augmented state, and x_next is given as input (not sampled).
          Implements: η(π', x_next | b_t, u) = T(x_next | x_t, u_t) * ∫_Y δ_{F(π_t, x_next, y)}(π') * H(dy | π_t, x_next)
        """
        raise NotImplementedError


class BeliefMDP_n_SLAM(SLAM_POMDP, BaseBeliefMDP_n):
    """
    Belief-MDP class for SLAM: handles joint belief over pose and map.

    Belief space: π(x, m) - shape (m_n, len_M)
    Extends SLAM_POMDP with belief-specific functionality.
    """

    def __init__(
        self,
        n: int,
        motion_model: SingleIntegratorModel | DoubleIntegratorModel,
        measurement_model: LIDAR,
        obstacles: list[Obstacle],
        _map: LidarGridMapVec,
        sigma_w: float = 0.01,
        sigma_v: float = 0.01
    ):
        # Initialize POMDP parent first
        SLAM_POMDP.__init__(self, motion_model, measurement_model, obstacles, _map, sigma_w, sigma_v)

        # Initialize common BeliefMDP_n components (quantizers, T_mat)
        BaseBeliefMDP_n._init_common(self, n, motion_model, _map)

        # Load or generate cached space of maps for vectorized operations
        # Only cache if map space is manageable (<= 2^16 = 65536 maps)
        if self.map_H * self.map_W <= 16:
            self.all_maps_3d, self.map_bits_array = self._load_or_generate_all_maps()
        else:
            self.all_maps_3d = None
            self.map_bits_array = None

    ##########################################################
    ### 0. Belief Shape Conversion Utilities ############
    def flatten_belief(self, π_2d: np.ndarray) -> np.ndarray:
        """
        Convert belief from (m_n, len_M) to (N_n,) where N_n = m_n * len_M.

        Flattening order: row-major (C-order), so beliefs are indexed as:
        π_flat[i * len_M + j] = π_2d[i, j]

        Args:
            π_2d: belief (m_n, len_M)
        Returns:
            π_flat: belief (N_n,) where N_n = m_n * len_M
        """
        return π_2d.flatten(order='C')

    def unflatten_belief(self, π_flat: np.ndarray) -> np.ndarray:
        """
        Convert belief from (N_n,) to (m_n, len_M) where N_n = m_n * len_M.

        Inverse of flatten_belief using row-major (C-order) reshape.

        Args:
            π_flat: belief (N_n,)
        Returns:
            π_2d: belief (m_n, len_M)
        """
        return π_flat.reshape((self.SQ.m_n, self.len_M), order='C')

    ##########################################################
    ### 1. Filter Update and Observation Functions ######
    def F_batch(self, π, u, Y_batch):
        """
        Batched filter update: F(π, u, y) for all observations.

        Args:
            π: current belief (m_n, len_M)
            u: action (2,)
            Y_batch: observations (K, B)
        Returns:
            π_new_batch: updated beliefs (K, m_n, len_M)
        """
        if self.all_maps_3d is None:
            raise ValueError(
                "F_batch requires all_maps_3d to be precomputed. "
                f"Map space too large (H*W={self.map_H * self.map_W} > 16)."
            )

        Tn_mat = self.T_mat[:, :, self.AQ.get_quantized_index(u)]
        integral = Tn_mat @ π
        Q_batch = self.Q_vectorized_batch(Y_batch, self.SQ.X_n)

        numerator = Q_batch * np.broadcast_to(integral[np.newaxis, :, :], (Y_batch.shape[0], *integral.shape))
        totals = np.sum(numerator, axis=(1, 2), keepdims=True)
        return numerator / np.maximum(totals, 1e-300)

    def F_batch_log(self, π, u, Y_batch):
        """
        Batched filter update in log-space for numerical stability.

        Args:
            π: current belief (m_n, len_M)
            u: action (2,)
            Y_batch: observations (K, B)
        Returns:
            π_new_batch: updated beliefs (K, m_n, len_M)
        """
        if self.all_maps_3d is None:
            raise ValueError(
                "F_batch_log requires all_maps_3d to be precomputed. "
                f"Map space too large (H*W={self.map_H * self.map_W} > 16)."
            )

        Tn_mat = self.T_mat[:, :, self.AQ.get_quantized_index(u)]
        integral = Tn_mat @ π
        log_integral = np.log(np.maximum(integral, 1e-300))
        log_Q_batch = self.Q_log_vectorized_batch(Y_batch, self.SQ.X_n)

        log_numerator = log_Q_batch + \
            np.broadcast_to(log_integral[np.newaxis, :, :], (Y_batch.shape[0], *log_integral.shape))
        log_numerator_2d = log_numerator.reshape(Y_batch.shape[0], -1)
        log_totals = logsumexp(log_numerator_2d, axis=1, keepdims=True)

        return np.exp(log_numerator_2d - log_totals).reshape(Y_batch.shape[0], *integral.shape)

    def H_vectorized_batch(self, Y_batch: np.ndarray, π: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Batched H computation: H(y | π, u) for all observations.

        Args:
            Y_batch: observations (n_obs, B)
            π: belief (m_n, len_M)
            u: action (2,)
        Returns:
            H_batch: (n_obs,) with H values for each observation
        """
        if self.all_maps_3d is None:
            raise ValueError(
                "H_vectorized_batch requires all_maps_3d to be precomputed. "
                f"Map space too large (H*W={self.map_H * self.map_W} > 16)."
            )

        Tn_mat = self.T_mat[:, :, self.AQ.get_quantized_index(u)]
        integral = Tn_mat @ π
        Q_batch = self.Q_vectorized_batch(Y_batch, self.SQ.X_n)

        contributions = Q_batch * np.broadcast_to(integral[np.newaxis, :, :], (Y_batch.shape[0], *integral.shape))
        return np.sum(contributions, axis=(1, 2))

    ###################################################
    ### 2. Belief State Transitions η_n ###############
    def sample_from_posterior_batch(self, π, u, batch_size):
        """
        Batch sample from predicted belief over (x', m).

        Args:
            π: current belief (m_n, 2^(H*W))
            u: action (2,)
            batch_size: number of samples to draw
        Returns:
            x_prime_batch: next states (batch_size, state_dim)
            m_indices_batch: map indices (batch_size,)
        """
        Tn_mat = self.T_mat[:, :, self.AQ.get_quantized_index(u)]
        flat_belief = (Tn_mat @ π).flatten()
        flat_belief = flat_belief / np.sum(flat_belief)

        indices = random.choice(len(flat_belief), size=batch_size, p=flat_belief)
        x_indices = indices // self.len_M
        m_indices = indices % self.len_M

        return self.SQ.X_n[x_indices], m_indices

    def sample_observation_from_model_batch(self, x_batch, m_indices):
        """
        Batch sample observations y ~ Q(y|x,m) = N(y; g_bar(x,m), Σ_v).

        Args:
            x_batch: robot states (batch_size, state_dim)
            m_indices: map indices (batch_size,)
        Returns:
            y_batch: observations (batch_size, B)
        """
        y_star_all = self.ray_casting_batched(x_batch[:, :2], self.all_maps_3d)
        y_ideal = y_star_all[np.arange(x_batch.shape[0]), m_indices, :]

        v_batch = random.multivariate_normal(
            np.zeros(self.sensor.B),
            self.cov_y,
            size=x_batch.shape[0]
        )

        return y_ideal + v_batch

    def sample_observations_batch(self, π, u, batch_size):
        """Combined batch sampling: sample observations from belief π and action u."""
        x_prime_batch, m_indices = self.sample_from_posterior_batch(π, u, batch_size)
        return self.sample_observation_from_model_batch(x_prime_batch, m_indices)

    def η_n(self, π_new: np.ndarray, π: np.ndarray, u: np.ndarray, n_samples: int = 50000, seed: int = None, batch_size: int = 500, show_progress: bool = True) -> float:
        """
        Compute transition probability η_n(π' | π, u) via Monte Carlo integration.

        η_n(π' | π, u) = ∫ 𝟙_{F(π,u,y) ≈ π'} H(dy | π, u)

        Uses forward sampling: sample (x', m) ~ Tn_mat @ π, then sample y ~ Q(y|x',m).
        Now uses batched operations for GPU acceleration.

        Args:
            π_new: target belief, either (m_n, len_M) or (N_n,) where N_n = m_n * len_M
            π: current belief, either (m_n, len_M) or (N_n,)
            u: action (2,)
            n_samples: number of Monte Carlo samples
            seed: random seed for reproducibility
            batch_size: batch size for GPU-accelerated sampling (default: 500)
            show_progress: whether to show progress bar (default: True)
        Returns:
            float: probability of transitioning to π' from π under action u
        """
        if seed is not None:
            random.seed(seed)

        # Normalize beliefs to 2D format if needed
        π_new_2d = π_new if π_new.ndim == 2 else self.unflatten_belief(π_new)
        π_2d = π if π.ndim == 2 else self.unflatten_belief(π)

        count = 0
        n_batches = (n_samples + batch_size - 1) // batch_size

        # Sequential processing - simple and reliable
        # Multi-GPU removed due to complexity with shared GPU state in BeliefMDP_n object
        batch_iter = tqdm(range(n_batches), desc=f"η_n MC sampling (n={n_samples}, batch={batch_size})",
                          unit="batch", disable=not show_progress) if show_progress else range(n_batches)
        for batch_idx in batch_iter:
            # Determine actual batch size for last batch
            current_batch_size = min(batch_size, n_samples - batch_idx * batch_size)

            # Batch sample observations
            Y_batch = self.sample_observations_batch(π_2d, u, current_batch_size)  # (current_batch_size, B)

            # Batch update beliefs using F_batch_log (log-space for numerical stability)
            π_sampled_batch = self.F_batch_log(π_2d, u, Y_batch)  # (current_batch_size, m_n, len_M)

            # Compute L2 distances for all beliefs in batch
            # Broadcast π_new_2d: (m_n, len_M) -> (1, m_n, len_M) -> (current_batch_size, m_n, len_M)
            π_new_broadcast = np.broadcast_to(π_new_2d[np.newaxis, :, :], (current_batch_size, *π_new_2d.shape))
            distances = np.linalg.norm(π_sampled_batch - π_new_broadcast, axis=(1, 2))  # (current_batch_size,)

            # Count matches
            count += int(np.sum(distances.get() < 1e-3) if hasattr(distances, 'get') else np.sum(distances < 1e-3))

        return count / n_samples

    ###################################################
    ### 3. Cost Function ##############################
    def c_n(self, x: np.ndarray, m: np.ndarray, u: np.ndarray) -> float:
        """
        Quantized one-stage cost c_n(x_i^n, m, u) = c(x_i^n, m, u).

        With Dirac weighting at centroids, this simplifies to evaluating
        the continuous cost at the quantized state.

        Args:
            x: single quantized state (2,)
            m: map (H, W)  
            u: action (2,)
        Returns:
            Cost value
        """
        return float(self.c(x, m, u))

    def c_tilde_n(self, π, u):
        """
        Expected cost function for belief-MDP: c_tilde(π, u) = E[c_n(x,m,u) | π]

        c_tilde(π, u) = Σ_{x∈X_n} Σ_{m∈M} c_n(x, m, u) * π(x,m)
        """
        assert π.shape == (self.SQ.m_n, self.len_M)

        # Compute expected cost using c_n for each (x,m) pair
        expected_cost = 0.0
        for i, x in enumerate(self.SQ.X_n):
            for j in range(self.len_M):
                # Cost for this (x,m) pair using quantized cost
                cost = self.c_n(x, self.bits_to_map(j, (self.map_H, self.map_W)), u)

                # Weight by belief
                expected_cost += cost * π[i, j]

        return float(expected_cost)

    def c_tilde_n_vectorized_batch(self, Π_batch: np.ndarray, U_batch: np.ndarray) -> np.ndarray:
        """
        Batched expected cost: c_tilde(π, u) for all (π, u) pairs.

        Args:
            Π_batch: beliefs (n_π, m_n, len_M) or (m_n, len_M)
            U_batch: actions (n_u, 2) or (2,)
        Returns:
            c_tilde_batch: (n_π, n_u) where c_tilde_batch[i, j] = c_tilde(Π_batch[i], U_batch[j])
        """
        if self.all_maps_3d is None:
            raise ValueError(
                "c_tilde_n_vectorized_batch requires all_maps_3d to be precomputed. "
                f"Map space too large (H*W={self.map_H * self.map_W} > 16)."
            )

        if Π_batch.ndim == 2:
            Π_batch = Π_batch[np.newaxis, :, :]
        if U_batch.ndim == 1:
            U_batch = U_batch[np.newaxis, :]

        c_batch = self.c_vectorized_batch(self.SQ.X_n, self.all_maps_3d, U_batch)
        weighted_costs = c_batch[np.newaxis, :, :, :] * Π_batch[:, :, :, np.newaxis]
        return np.sum(weighted_costs, axis=(1, 2))


class BeliefMDP_n_Localization(Localization_POMDP, BaseBeliefMDP_n):
    """
    Belief-MDP class for Localization: handles belief over pose only.

    Belief space: π(x) - shape (m_n,)
    Map is known (not part of belief state).

    Filter update: F(b_t, u_t, y_{t+1})(x) = Q(y|x) * ∫T(x|x_t, u_t)b_t(dx_t) / normalization
    """

    def __init__(
        self,
        n: int,
        motion_model: SingleIntegratorModel | DoubleIntegratorModel,
        measurement_model: LIDAR,
        obstacles: list[Obstacle],
        _map: LidarGridMapVec,
        sigma_w: float = 0.01,
        sigma_v: float = 1
    ):
        # Initialize POMDP parent first
        Localization_POMDP.__init__(self, motion_model, measurement_model, obstacles, _map, sigma_w, sigma_v)

        # Initialize common BeliefMDP_n components (quantizers, T_mat)
        BaseBeliefMDP_n._init_common(self, n, motion_model, _map)


    ##########################################################
    ### 1. Filter Update and Observation Functions ######


    def F_batch(self, π, u, Y_batch):
        """
        Batched filter update: F(π, u, y) for all observations.

        Args:
            π: current belief (m_n,)
            u: action (2,)
            Y_batch: observations (K, B)
        Returns:
            π_new_batch: updated beliefs (K, m_n)
        """
        if self.known_map_obstacle_segments is None:
            raise ValueError("Known map obstacle segments not set. Call set_known_map() first.")

        Tn_mat = self.T_mat[:, :, self.AQ.get_quantized_index(u)]
        integral = Tn_mat @ π
        Q_batch = self.Q_vectorized_batch(Y_batch, self.SQ.X_n)

        numerator = Q_batch * np.broadcast_to(integral[np.newaxis, :], (Y_batch.shape[0], len(integral)))
        totals = np.sum(numerator, axis=1, keepdims=True)
        return numerator / np.maximum(totals, 1e-300)

    def F_batch_log(self, π, u, Y_batch):
        """
        Batched filter update in log-space for numerical stability.

        Args:
            π: current belief (m_n,)
            u: action (2,)
            Y_batch: observations (K, B)
        Returns:
            π_new_batch: updated beliefs (K, m_n)
        """
        if self.known_map_obstacle_segments is None:
            raise ValueError("Known map obstacle segments not set. Call set_known_map() first.")

        Tn_mat = self.T_mat[:, :, self.AQ.get_quantized_index(u)]
        log_integral = np.log(np.maximum(Tn_mat @ π, 1e-300))
        Q_batch = self.Q_vectorized_batch(Y_batch, self.SQ.X_n)
        log_Q_batch = np.log(np.maximum(Q_batch, 1e-300))

        log_numerator = log_Q_batch + \
            np.broadcast_to(log_integral[np.newaxis, :], (Y_batch.shape[0], len(log_integral)))
        log_totals = logsumexp(log_numerator, axis=1, keepdims=True)

        return np.exp(log_numerator - log_totals)

    def H_vectorized_batch(self, Y_batch: np.ndarray, π: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Batched H computation: H(y | π, u) for all observations.

        Args:
            Y_batch: observations (n_obs, B)
            π: belief (m_n,)
            u: action (2,)
        Returns:
            H_batch: (n_obs,) with H values for each observation
        """
        if self.known_map_obstacle_segments is None:
            raise ValueError("Known map obstacle segments not set. Call set_known_map() first.")

        Tn_mat = self.T_mat[:, :, self.AQ.get_quantized_index(u)]
        integral = Tn_mat @ π
        Q_batch = self.Q_vectorized_batch(Y_batch, self.SQ.X_n)

        contributions = Q_batch * np.broadcast_to(integral[np.newaxis, :], (Y_batch.shape[0], len(integral)))
        return np.sum(contributions, axis=1)

    def c_tilde_n(self, π, u):
        """
        Expected cost function for localization belief-MDP: c_tilde(π, u) = E[c_n(x, u) | π]

        c_tilde(π, u) = Σ_{x∈X_n} c_n(x, u) * π(x)
        Note: Map is known, so cost depends only on pose x and action u.
        """
        assert π.shape == (self.SQ.m_n,)

        # Compute expected cost using c_n for each pose
        expected_cost = 0.0
        for i, x in enumerate(self.SQ.X_n):
            # Cost for this pose using localization cost function
            cost = float(self.c(x, u))  # Localization_POMDP.c(x, u) - no map argument

            # Weight by belief
            expected_cost += cost * π[i]

        return float(expected_cost)

    def c_tilde_n_vectorized_batch(self, Π_batch: np.ndarray, U_batch: np.ndarray) -> np.ndarray:
        """
        Batched expected cost: c_tilde(π, u) for all (π, u) pairs.

        Args:
            Π_batch: beliefs (n_π, m_n) or (m_n,)
            U_batch: actions (n_u, 2) or (2,)
        Returns:
            c_tilde_batch: (n_π, n_u) where c_tilde_batch[i, j] = c_tilde(Π_batch[i], U_batch[j])
        """
        if Π_batch.ndim == 1:
            Π_batch = Π_batch[np.newaxis, :]
        if U_batch.ndim == 1:
            U_batch = U_batch[np.newaxis, :]

        c_batch = self.c_vectorized_batch(self.SQ.X_n, U_batch)
        weighted_costs = c_batch[np.newaxis, :, :] * Π_batch[:, :, np.newaxis]
        return np.sum(weighted_costs, axis=1)

    def sample_observations_batch(self, π, u, batch_size):
        """
        Combined batch sampling: sample observations from belief π and action u.

        Args:
            π: current belief (m_n,)
            u: action (2,)
            batch_size: number of observations to sample
        Returns:
            y_batch: observations (batch_size, B)
        """
        Tn_mat = self.T_mat[:, :, self.AQ.get_quantized_index(u)]
        predicted_belief = (Tn_mat @ π) / np.sum(Tn_mat @ π)

        x_indices = random.choice(self.SQ.m_n, size=batch_size, p=predicted_belief)
        x_prime_batch = self.SQ.X_n[x_indices]

        y_ideal_batch = self.ray_casting(x_prime_batch[:, :2])
        v_batch = random.multivariate_normal(np.zeros(self.sensor.B), self.cov_y, size=batch_size)

        return y_ideal_batch + v_batch

    def η_n(self, π_new: np.ndarray, π: np.ndarray, u: np.ndarray, n_samples: int = 50000,
            seed: int = None, batch_size: int = 500, show_progress: bool = True) -> float:
        """
        Compute transition probability η_n(π' | π, u) via Monte Carlo integration for localization.

        η_n(π' | π, u) = ∫ 𝟙_{F(π,u,y) ≈ π'} H(dy | π, u)

        Args:
            π_new: target belief (m_n,)
            π: current belief (m_n,)
            u: action (2,)
            n_samples: number of Monte Carlo samples
            seed: random seed for reproducibility
            batch_size: batch size for GPU-accelerated sampling
            show_progress: whether to show progress bar
        Returns:
            float: probability of transitioning to π' from π under action u
        """
        if seed is not None:
            random.seed(seed)

        # Ensure beliefs are 1D
        π_new_1d = π_new.flatten() if π_new.ndim > 1 else π_new
        π_1d = π.flatten() if π.ndim > 1 else π

        count = 0
        n_batches = (n_samples + batch_size - 1) // batch_size

        batch_iter = tqdm(range(n_batches), desc=f"η_n MC sampling (n={n_samples}, batch={batch_size})",
                          unit="batch", disable=not show_progress) if show_progress else range(n_batches)
        for batch_idx in batch_iter:
            current_batch_size = min(batch_size, n_samples - batch_idx * batch_size)

            # Batch sample observations
            Y_batch = self.sample_observations_batch(π_1d, u, current_batch_size)  # (current_batch_size, B)

            # Batch update beliefs using F_batch_log
            π_sampled_batch = self.F_batch_log(π_1d, u, Y_batch)  # (current_batch_size, m_n)

            # Compute L2 distances for all beliefs in batch
            π_new_broadcast = np.broadcast_to(π_new_1d[np.newaxis, :], (current_batch_size, len(π_new_1d)))
            distances = np.linalg.norm(π_sampled_batch - π_new_broadcast, axis=1)  # (current_batch_size,)

            # Count matches
            count += int(np.sum(distances.get() < 1e-3) if hasattr(distances, 'get') else np.sum(distances < 1e-3))

        return count / n_samples


class BeliefMDP_n_Mapping(Mapping_POMDP, BaseBeliefMDP_n):
    """
    Belief-MDP class for Mapping: handles belief over map only.

    Belief space: π(m) - shape (len_M,)
    Pose is known (not part of belief state).

    Filter update: F(b_t, x_{t+1}, y_{t+1})(m) = Q(y|x_{t+1}, m) * b_t(m) / normalization

    Note: This filter update depends on known pose x_{t+1} rather than action u_t,
    which differs from the typical POMDP structure.
    """

    def __init__(
        self,
        n: int,
        motion_model: SingleIntegratorModel | DoubleIntegratorModel,
        measurement_model: LIDAR,
        obstacles: list[Obstacle],
        _map: LidarGridMapVec,
        sigma_w: float = 0.01,
        sigma_v: float = 0.01
    ):
        # Initialize POMDP parent first
        Mapping_POMDP.__init__(self, motion_model, measurement_model, obstacles, _map, sigma_w, sigma_v)

        # Initialize common BeliefMDP_n components (quantizers, T_mat)
        BaseBeliefMDP_n._init_common(self, n, motion_model, _map)

        # Load or generate cached space of maps for vectorized operations
        if self.map_H * self.map_W <= 16:
            self.all_maps_3d, self.map_bits_array = self._load_or_generate_all_maps()
        else:
            self.all_maps_3d = None
            self.map_bits_array = None

    ##########################################################
    ### 1. Filter Update and Observation Functions ######
    def F_batch(self, π, x_next, Y_batch):
        """
        Batched filter update: F(π, x_next, y) for all observations.

        Args:
            π: current belief (len_M,)
            x_next: known next pose (state_dim,) or (2,)
            Y_batch: observations (K, B)
        Returns:
            π_new_batch: updated beliefs (K, len_M)
        """
        if self.known_pose is None:
            raise ValueError("Known pose not set. Set self.known_pose first.")
        if self.all_maps_3d is None:
            raise ValueError(
                "F_batch requires all_maps_3d to be precomputed. "
                f"Map space too large (H*W={self.map_H * self.map_W} > 16)."
            )

        self.known_pose = x_next
        Q_batch = self.Q_vectorized_batch(Y_batch)

        numerator = Q_batch * np.broadcast_to(π[np.newaxis, :], (Y_batch.shape[0], len(π)))
        totals = np.sum(numerator, axis=1, keepdims=True)
        return numerator / np.maximum(totals, 1e-300)

    def F_batch_log(self, π, x_next, Y_batch):
        """
        Batched filter update in log-space for numerical stability.

        Args:
            π: current belief (len_M,)
            x_next: known next pose (state_dim,) or (2,)
            Y_batch: observations (K, B)
        Returns:
            π_new_batch: updated beliefs (K, len_M)
        """
        if self.known_pose is None:
            raise ValueError("Known pose not set. Set self.known_pose first.")
        if self.all_maps_3d is None:
            raise ValueError(
                "F_batch_log requires all_maps_3d to be precomputed. "
                f"Map space too large (H*W={self.map_H * self.map_W} > 16)."
            )

        self.known_pose = x_next
        log_π = np.log(np.maximum(π, 1e-300))
        Q_batch = self.Q_vectorized_batch(Y_batch)
        log_Q_batch = np.log(np.maximum(Q_batch, 1e-300))

        log_numerator = log_Q_batch + np.broadcast_to(log_π[np.newaxis, :], (Y_batch.shape[0], len(log_π)))
        log_totals = logsumexp(log_numerator, axis=1, keepdims=True)

        return np.exp(log_numerator - log_totals)

    def H_vectorized_batch(self, Y_batch: np.ndarray, π: np.ndarray, x_next: np.ndarray) -> np.ndarray:
        """
        Batched H computation: H(y | π, x_next) for all observations.

        Args:
            Y_batch: observations (n_obs, B)
            π: belief (len_M,)
            x_next: known next pose (state_dim,) or (2,)
        Returns:
            H_batch: (n_obs,) with H values for each observation
        """
        if self.known_pose is None:
            raise ValueError("Known pose not set. Set self.known_pose first.")
        if self.all_maps_3d is None:
            raise ValueError(
                "H_vectorized_batch requires all_maps_3d to be precomputed. "
                f"Map space too large (H*W={self.map_H * self.map_W} > 16)."
            )

        self.known_pose = x_next
        Q_batch = self.Q_vectorized_batch(Y_batch)

        contributions = Q_batch * np.broadcast_to(π[np.newaxis, :], (Y_batch.shape[0], len(π)))
        return np.sum(contributions, axis=1)

    def c_tilde_n(self, π, x_current, u):
        """
        Expected cost function for mapping belief-MDP: c_tilde(π, x_current, u) = E[c_n(m, u) | π]

        c_tilde(π, x_current, u) = Σ_{m∈M} c_n(m, u) * π(m)
        Note: Cost depends on map m and action u. Pose x_current is known but used for cost computation.
        """
        assert π.shape == (self.len_M,)

        # Compute expected cost using c_n for each map
        expected_cost = 0.0
        for j in range(self.len_M):
            # Get map from bits
            m = self.bits_to_map(j, (self.map_H, self.map_W))

            # Cost for this map using mapping cost function
            cost = float(self.c(m, u))  # Mapping_POMDP.c(m, u) - no pose argument

            # Weight by belief
            expected_cost += cost * π[j]

        return float(expected_cost)

    def c_tilde_n_vectorized_batch(self, Π_batch: np.ndarray, X_current_batch: np.ndarray, U_batch: np.ndarray) -> np.ndarray:
        """
        Batched expected cost: c_tilde(π, x_current, u) for all combinations.

        Args:
            Π_batch: beliefs (n_π, len_M) or (len_M,)
            X_current_batch: current poses (n_x, state_dim) or (state_dim,)
            U_batch: actions (n_u, 2) or (2,)
        Returns:
            c_tilde_batch: (n_π, n_x, n_u) where c_tilde_batch[i, j, k] = c_tilde(Π_batch[i], X_current_batch[j], U_batch[k])
        """
        if self.all_maps_3d is None:
            raise ValueError(
                "c_tilde_n_vectorized_batch requires all_maps_3d to be precomputed. "
                f"Map space too large (H*W={self.map_H * self.map_W} > 16)."
            )

        if Π_batch.ndim == 1:
            Π_batch = Π_batch[np.newaxis, :]
        if X_current_batch.ndim == 1:
            X_current_batch = X_current_batch[np.newaxis, :]
        if U_batch.ndim == 1:
            U_batch = U_batch[np.newaxis, :]

        n_x = X_current_batch.shape[0]
        len_M = self.all_maps_3d.shape[0]
        X_pos = X_current_batch[:, :2] if X_current_batch.shape[1] > 2 else X_current_batch

        # Compute F_r for all (x, m) pairs
        # F_r_vectorized_batch expects (n_x, 2) positions and (n_m, H, W) maps
        # and returns (n_x, n_m, 2) where F_r[i, j] = F_r(X[i], M[j])
        F_r = self.F_r_vectorized_batch(X_pos, self.all_maps_3d)  # (n_x, len_M, 2)

        # Compute costs: effort + VFF for all (x, m, u) combinations
        effort = np.linalg.norm(U_batch, axis=1)[np.newaxis, np.newaxis, :]
        dot_prod = np.sum(F_r[:, :, np.newaxis, :] * U_batch[np.newaxis, np.newaxis, :, :], axis=3)
        norms = np.linalg.norm(F_r, axis=2, keepdims=True) * np.linalg.norm(U_batch, axis=1)[np.newaxis, np.newaxis, :]
        vff = np.maximum(0, dot_prod / (norms + 1e-6))
        c_batch = effort + vff

        # Weight by beliefs and sum over maps
        return np.sum(c_batch[np.newaxis, :, :, :] * Π_batch[:, np.newaxis, :, np.newaxis], axis=2)

    def sample_observations_batch(self, π, x_next, batch_size):
        """
        Combined batch sampling: sample observations from belief π and known pose x_next.

        Args:
            π: current belief (len_M,)
            x_next: known next pose (state_dim,) or (2,)
            batch_size: number of observations to sample
        Returns:
            y_batch: observations (batch_size, B)
        """
        if self.all_maps_3d is None:
            raise ValueError(
                "sample_observations_batch requires all_maps_3d to be precomputed. "
                f"Map space too large (H*W={self.map_H * self.map_W} > 16)."
            )

        self.known_pose = x_next
        x_pos = x_next[:2] if len(x_next) > 2 else x_next
        x_pos_batch = np.broadcast_to(x_pos[np.newaxis, :], (batch_size, 2))

        m_indices = random.choice(self.len_M, size=batch_size, p=π / np.sum(π))
        y_star_all = self.ray_casting_batched(x_pos_batch, self.all_maps_3d)

        y_ideal = y_star_all[np.arange(batch_size), m_indices, :]
        v_batch = random.multivariate_normal(np.zeros(self.sensor.B), self.cov_y, size=batch_size)

        return y_ideal + v_batch

    def η_n(self, π_new: np.ndarray, π: np.ndarray, x_current: np.ndarray, u: np.ndarray, x_next: np.ndarray,
            n_samples: int = 50000, seed: int = None, batch_size: int = 500, show_progress: bool = True) -> float:
        """
        Compute joint transition probability η_n(π', x_next | b_t, u) via Monte Carlo integration for mapping.

        Implements the joint probability:
        η(π', x_next | b_t, u) = T(x_next | x_t, u_t) * ∫_Y δ_{F(π_t, x_next, y)}(π') * H(dy | π_t, x_next)

        where b_t = (x_t, π_t) is the augmented state, and x_next is given as input (not sampled).

        Monte Carlo procedure:
        1. Get T(x_next | x_current, u) from T_mat (transition probability)
        2. Sample y ~ H(· | π_t, x_next) (Monte Carlo over observations)
        3. Update belief: π' = F(π_t, x_next, y)
        4. Check if π' ≈ π_new
        5. Return T(x_next | x_current, u) * (fraction of samples where π' ≈ π_new)

        Args:
            π_new: target belief (len_M,)
            π: current belief (len_M,)
            x_current: current state (state_dim,) - part of augmented state b_t = (x_t, π_t)
            u: action (2,)
            x_next: next state (state_dim,) - given as input, not sampled
            n_samples: number of Monte Carlo samples
            seed: random seed for reproducibility
            batch_size: batch size for GPU-accelerated sampling
            show_progress: whether to show progress bar
        Returns:
            float: joint probability of (π', x_next) from augmented state (x_t, π_t) under action u
        """
        if seed is not None:
            random.seed(seed)

        # Ensure beliefs are 1D
        π_new_1d = π_new.flatten() if π_new.ndim > 1 else π_new
        π_1d = π.flatten() if π.ndim > 1 else π

        # Get quantized indices
        x_current_idx = self.SQ.get_quantized_index(x_current)
        x_next_idx = self.SQ.get_quantized_index(x_next)
        u_idx = self.AQ.get_quantized_index(u)

        # Step 1: Get T(x_next | x_current, u) from T_mat
        Tn_mat = self.T_mat[:, :, u_idx]  # (m_n, m_n)
        T_x_next_given_x_current_u = float(Tn_mat[x_next_idx, x_current_idx])  # Scalar probability

        # If transition probability is zero, return 0
        if T_x_next_given_x_current_u < 1e-300:
            return 0.0

        # Step 2: Sample observations y ~ H(· | π_t, x_next) via Monte Carlo
        if self.all_maps_3d is None:
            raise ValueError(
                "η_n requires all_maps_3d to be precomputed. "
                f"Map space too large (H*W={self.map_H * self.map_W} > 16)."
            )

        count = 0
        n_batches = (n_samples + batch_size - 1) // batch_size

        batch_iter = tqdm(range(n_batches), desc=f"η_n MC sampling (n={n_samples}, batch={batch_size})",
                          unit="batch", disable=not show_progress) if show_progress else range(n_batches)
        for batch_idx in batch_iter:
            current_batch_size = min(batch_size, n_samples - batch_idx * batch_size)

            # Sample observations using the existing method
            Y_batch = self.sample_observations_batch(π_1d, x_next, current_batch_size)

            # Update beliefs: π' = F(π_t, x_next, y)
            π_sampled_batch = self.F_batch_log(π_1d, x_next, Y_batch)

            # Check if π' ≈ π_new
            π_new_broadcast = np.broadcast_to(π_new_1d[np.newaxis, :], (current_batch_size, len(π_new_1d)))
            distances = np.linalg.norm(π_sampled_batch - π_new_broadcast, axis=1)

            # Count matches
            count += int(np.sum(distances.get() < 1e-3) if hasattr(distances, 'get') else np.sum(distances < 1e-3))

        # Step 5: Return joint probability = T(x_next | x_current, u) * P(π' ≈ π_new | x_next)
        fraction_matching = count / n_samples
        return T_x_next_given_x_current_u * fraction_matching


__all__ = ['BaseBeliefMDP_n', 'BeliefMDP_n_SLAM', 'BeliefMDP_n_Localization', 'BeliefMDP_n_Mapping']
