import hashlib
from pathlib import Path
from .mapping import BaseMap
from .model import LIDAR, RangeBearingSensor, SingleIntegratorModel, DoubleIntegratorModel
from .obstacle import Obstacle
from .quantizer import StateQuantizer, ActionQuantizer, ObservationQuantizer
from abc import ABC, abstractmethod
from typing import Any, Literal
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
    from cupyx.scipy.spatial.distance import cdist
else:
    from scipy.special import logsumexp
    from scipy.spatial.distance import cdist


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
        _map: BaseMap,
        obs_n: int | None = None,
        action_n: int | None = None
    ):
        """
        Initialize common BeliefMDP_n components (quantizers and T_mat).

        This is called by subclasses after their POMDP parent is initialized.

        Args:
            n: State quantization level
            motion_model: Motion model (SingleIntegrator or DoubleIntegrator)
            _map: Map object
            obs_n: Observation quantization level (defaults to n if None)
            action_n: Action quantization level (defaults to n if None)
        """
        self.n = n
        if obs_n is None:
            obs_n = n
        if action_n is None:
            action_n = n

        def to_float(x):
            return float(x.item() if hasattr(x, 'item') else x)

        if isinstance(motion_model, SingleIntegratorModel):
            ll = _map.left_lower
            ru = _map.right_upper
            state_bounds = np.array([
                [to_float(ll[0]), to_float(ru[0])],
                [to_float(ll[1]), to_float(ru[1])]
            ], dtype=float)
            self.SQ = StateQuantizer(
                (to_float(ll[0]), to_float(ru[0]), to_float(ll[1]), to_float(ru[1])),
                n
            )
            self.state_dim = 2
            self.AQ = ActionQuantizer(motion_model.max_v, action_n)
        elif isinstance(motion_model, DoubleIntegratorModel):
            ll = _map.left_lower
            ru = _map.right_upper
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
            self.AQ = ActionQuantizer(motion_model.max_a, action_n)
        else:
            raise ValueError(f"Unsupported motion model type: {type(motion_model)}")

        self.map_distance_p = float('inf') if len(_map.map_shape) == 2 else 2.0
        self.state_bounds = state_bounds
        if hasattr(motion_model, "set_state_bounds"):
            motion_model.set_state_bounds(self.state_bounds)

        # Get map dimensions - works for both occupancy grids and landmark maps
        map_shape = _map.map_shape
        if len(map_shape) == 2:
            # Occupancy grid: (H, W)
            self.map_H, self.map_W = map_shape
        else:
            # Landmark map or other: use map_shape[0] as primary dimension
            # For compatibility, set map_H and map_W to map_shape[0] and 1
            self.map_H = map_shape[0]
            self.map_W = 1 if len(map_shape) == 1 else map_shape[1]
        self.len_M = _map.len_M

        # ----------------------------
        # Observation space quantizer
        # ----------------------------
        # We quantize each LIDAR beam independently on [0, r_max] with a uniform grid.
        # The full observation alphabet is then the Cartesian product Y_n = {centers}^B.
        # NOTE: |Y_n| = n^B, so the user MUST choose small B and/or n in practice.
        self.obs_n = obs_n  # Observation quantization level (can differ from state/action n)

        if hasattr(self, "sensor") and isinstance(self.sensor, LIDAR):
            # LIDAR: bounds [0, r_max] for each beam
            b_bounds = [(0.0, float(self.sensor.r_max))] * int(self.sensor.B)
            self.YQ = ObservationQuantizer(b_bounds, self.obs_n)
            self.Y_n = self.YQ.Y_n              # (m_y, B)
            self.Y_bounds = self.YQ.get_bounds()  # (m_y, B, 2)
        elif hasattr(self, "sensor") and isinstance(self.sensor, RangeBearingSensor):
            # Range-bearing: per landmark, observation space is (R x Theta) U {⊥}
            num_landmarks = self._get_landmark_count(_map)
            if num_landmarks is None:
                raise ValueError("RangeBearingSensor requires LandmarkMap with num_candidate_landmarks")

            eps = float(self.sensor.epsilon)
            r_max = float(self.sensor.r_max)
            r_vals = _numpy.linspace(eps, r_max, self.obs_n)
            dr = (r_max - eps) / self.obs_n
            r_bounds = _numpy.stack([r_vals - dr / 2.0, r_vals + dr / 2.0], axis=1)
            r_bounds[:, 0] = _numpy.clip(r_bounds[:, 0], eps, r_max)
            r_bounds[:, 1] = _numpy.clip(r_bounds[:, 1], eps, r_max)

            phi_vals = _numpy.linspace(-_numpy.pi, _numpy.pi, self.obs_n, endpoint=False)
            dphi = 2.0 * _numpy.pi / self.obs_n
            phi_bounds = _numpy.stack([phi_vals - dphi / 2.0, phi_vals + dphi / 2.0], axis=1)

            rr, pp = _numpy.meshgrid(r_vals, phi_vals, indexing='ij')
            per_landmark_points = _numpy.stack([rr.reshape(-1), pp.reshape(-1)], axis=1)  # (n^2, 2)
            per_landmark_bounds = _numpy.stack([
                r_bounds.repeat(self.obs_n, axis=0),
                _numpy.tile(phi_bounds, (self.obs_n, 1))
            ], axis=1)  # (n^2, 2, 2)

            # Append sentinel ⊥ as [nan, nan]
            per_landmark_points = _numpy.vstack([per_landmark_points, _numpy.array([[_numpy.nan, _numpy.nan]])])
            per_landmark_bounds = _numpy.vstack([per_landmark_bounds, _numpy.array(
                [[[_numpy.nan, _numpy.nan], [_numpy.nan, _numpy.nan]]])])

            n_points = per_landmark_points.shape[0]
            idx_grid = _numpy.indices((n_points,) * int(num_landmarks)).reshape(int(num_landmarks), -1).T
            self.Y_n = per_landmark_points[idx_grid].reshape(-1, int(num_landmarks) * 2)
            self.Y_bounds = per_landmark_bounds[idx_grid].reshape(-1, int(num_landmarks) * 2, 2)
            self.YQ = None
        else:
            self.YQ = None
            self.Y_n = None
            self.Y_bounds = None

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

        # Load or compute Q_n (observation probability matrix)
        if self.Y_n is not None:
            Q_cache_path = self._get_Q_cache_path()
            if self._load_Q_n(Q_cache_path):
                print(f"Loaded cached Q_n from {Q_cache_path}")
            else:
                print("Computing Q_n for the first time...")
                self.Q_n = self._compute_Q_n()
                self._save_Q_n(Q_cache_path)
                print(f"Saved Q_n to {Q_cache_path}")
        else:
            self.Q_n = None

    def configure_observation_quantization(self, obs_n: int, load_cache: bool = True):
        """
        Reconfigure observation quantization after initialization.

        This allows changing the observation quantization level without recreating
        the entire MDP. Note that Q_n will need to be recomputed if it was already
        computed with the old observation quantization.

        Args:
            obs_n: New number of quantization levels per observation dimension
        """
        # Store old Q_n temporarily in case we need to restore it
        old_Q_n = self.Q_n

        self.obs_n = obs_n

        if hasattr(self, "sensor") and isinstance(self.sensor, LIDAR):
            b_bounds = [(0.0, float(self.sensor.r_max))] * int(self.sensor.B)
            self.YQ = ObservationQuantizer(b_bounds, self.obs_n)
            self.Y_n = self.YQ.Y_n              # (m_y, B)
            self.Y_bounds = self.YQ.get_bounds()  # (m_y, B, 2)
        elif hasattr(self, "sensor") and isinstance(self.sensor, RangeBearingSensor):
            num_landmarks = self._get_landmark_count()
            if num_landmarks is None:
                raise ValueError("RangeBearingSensor requires LandmarkMap with num_candidate_landmarks")

            eps = float(self.sensor.epsilon)
            r_max = float(self.sensor.r_max)
            r_vals = _numpy.linspace(eps, r_max, self.obs_n)
            dr = (r_max - eps) / self.obs_n
            r_bounds = _numpy.stack([r_vals - dr / 2.0, r_vals + dr / 2.0], axis=1)
            r_bounds[:, 0] = _numpy.clip(r_bounds[:, 0], eps, r_max)
            r_bounds[:, 1] = _numpy.clip(r_bounds[:, 1], eps, r_max)

            phi_vals = _numpy.linspace(-_numpy.pi, _numpy.pi, self.obs_n, endpoint=False)
            dphi = 2.0 * _numpy.pi / self.obs_n
            phi_bounds = _numpy.stack([phi_vals - dphi / 2.0, phi_vals + dphi / 2.0], axis=1)

            rr, pp = _numpy.meshgrid(r_vals, phi_vals, indexing='ij')
            per_landmark_points = _numpy.stack([rr.reshape(-1), pp.reshape(-1)], axis=1)
            per_landmark_bounds = _numpy.stack([
                r_bounds.repeat(self.obs_n, axis=0),
                _numpy.tile(phi_bounds, (self.obs_n, 1))
            ], axis=1)

            per_landmark_points = _numpy.vstack([per_landmark_points, _numpy.array([[_numpy.nan, _numpy.nan]])])
            per_landmark_bounds = _numpy.vstack([per_landmark_bounds, _numpy.array(
                [[[_numpy.nan, _numpy.nan], [_numpy.nan, _numpy.nan]]])])

            n_points = per_landmark_points.shape[0]
            idx_grid = _numpy.indices((n_points,) * int(num_landmarks)).reshape(int(num_landmarks), -1).T
            self.Y_n = per_landmark_points[idx_grid].reshape(-1, int(num_landmarks) * 2)
            self.Y_bounds = per_landmark_bounds[idx_grid].reshape(-1, int(num_landmarks) * 2, 2)
            self.YQ = None
        else:
            self.YQ = None
            self.Y_n = None
            self.Y_bounds = None

        # Try to load Q_n with the new obs_n value
        # If it doesn't exist, Q_n will be None and will be recomputed when accessed
        if self.Y_n is not None and load_cache:
            Q_cache_path = self._get_Q_cache_path()
            if self._load_Q_n(Q_cache_path):
                print(f"Loaded cached Q_n with obs_n={obs_n} from {Q_cache_path}")
            else:
                # Clear Q_n since it depends on the observation space
                # It will be recomputed on next access if needed
                self.Q_n = None
        else:
            self.Q_n = None

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
                # Explicitly convert to backend array type (CuPy or NumPy)
                # Use np.array() to ensure conversion happens even if input is already an array
                maps_3d = np.array(data['maps_3d'])
                map_bits = np.array(data['map_bits'])
                # Check if shape matches expected map shape
                expected_shape = (self.len_M,) + self.map.map_shape
                if maps_3d.shape == expected_shape:
                    return maps_3d, map_bits
            except Exception as e:
                print(f"Error loading all_maps cache: {e}")

        print(f"Generating all {self.len_M} maps (this may take a moment)...")
        maps_3d, map_bits = self.map.generate_all_maps(show_progress=True)

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
        bounds_raw = self.SQ.get_bounds()
        bounds = bounds_raw.get() if hasattr(bounds_raw, 'get') else numpy_cpu.asarray(bounds_raw)
        X_n_raw = self.SQ.X_n
        X_n_cpu = X_n_raw.get() if hasattr(X_n_raw, 'get') else numpy_cpu.asarray(X_n_raw)
        K_mask_raw = self._compute_K_mask()
        K_mask_cpu = K_mask_raw.get() if hasattr(K_mask_raw, 'get') else numpy_cpu.asarray(K_mask_raw)

        for k in tqdm(range(self.AQ.n_u), desc="Computing T_mat"):
            u_raw = self.AQ.U[k]
            u = u_raw.get() if hasattr(u_raw, 'get') else numpy_cpu.asarray(u_raw)

            for j in range(self.SQ.m_n):
                transition_probs_raw = self.T(bounds[j], X_n_cpu, u)
                transition_probs = transition_probs_raw.get() if hasattr(
                    transition_probs_raw, 'get') else numpy_cpu.asarray(transition_probs_raw)
                feasible_pairs = K_mask_cpu[:, k].astype(numpy_cpu.float64)
                T_mat[j, :, k] = transition_probs * feasible_pairs

            col_sums = T_mat[:, :, k].sum(axis=0, keepdims=True)
            col_sums = numpy_cpu.where(col_sums == 0.0, 1.0, col_sums)
            T_mat[:, :, k] = T_mat[:, :, k] / col_sums

        return np.asarray(T_mat)

    def _get_Q_cache_path(self) -> Path:
        """Generate cache file path for Q_n based on quantization parameters."""
        cache_dir = Path(__file__).parent.parent.parent / "cache/Q_n"
        cache_dir.mkdir(parents=True, exist_ok=True)

        max_val = self.motion_model.max_v if hasattr(self.motion_model, 'max_v') else self.motion_model.max_a
        metadata = {
            'n': self.n,
            'obs_n': self.obs_n,
            'state_bounds': self.state_bounds.flatten().tolist(),
            'max_val': max_val,
            'dt': float(self.motion_model.dt),
            'map_shape': (self.map_H, self.map_W),
            'sigma_v': float(self.σ_v),
            'sensor_r_max': float(self.sensor.r_max),
            'model': self.motion_model.__class__.__name__,
            'state_dim': self.state_dim,
            'problem_type': self._get_problem_type(),
            'kernel_version': 'v2-gaussian-borel-rb',
        }

        # Add sensor-specific metadata
        if isinstance(self.sensor, LIDAR):
            metadata['sensor_B'] = int(self.sensor.B)
            sensor_suffix = f"B{self.sensor.B}"
        elif isinstance(self.sensor, RangeBearingSensor):
            num_landmarks = self._get_landmark_count()
            if num_landmarks is not None:
                metadata['sensor_num_landmarks'] = int(num_landmarks)
                sensor_suffix = f"L{num_landmarks}"
            else:
                sensor_suffix = "RB"
        else:
            sensor_suffix = "unknown"

        metadata_hash = hashlib.md5(str(sorted(metadata.items())).encode()).hexdigest()[:8]
        filename = f"Q_n_n{self.n}_obs{self.obs_n}_map{self.map_H}x{self.map_W}_{sensor_suffix}_{metadata_hash}.npz"
        return cache_dir / filename

    def _save_Q_n(self, cache_path: Path) -> None:
        """Save Q_n to cache file with metadata."""
        if self.Q_n is None:
            return

        state_bounds_save = self.state_bounds.astype(float)
        state_bounds_save = state_bounds_save.get() if hasattr(state_bounds_save, 'get') else state_bounds_save
        Q_n_save = self.Q_n.get() if hasattr(self.Q_n, 'get') else self.Q_n

        max_val = self.motion_model.max_v if hasattr(self.motion_model, 'max_v') else self.motion_model.max_a
        metadata = {
            'n': self.n,
            'obs_n': self.obs_n,
            'state_bounds': state_bounds_save,
            'max_val': max_val,
            'dt': float(self.motion_model.dt),
            'map_shape': _numpy.array([self.map_H, self.map_W]),
            'sigma_v': float(self.σ_v),
            'sensor_r_max': float(self.sensor.r_max),
            'm_y': self.Y_n.shape[0],
            'm_n': self.SQ.m_n,
            'len_M': self.len_M,
            'model': self.motion_model.__class__.__name__,
            'state_dim': self.state_dim,
            'problem_type': self._get_problem_type(),
            'kernel_version': 'v2-gaussian-borel-rb',
        }

        # Add sensor-specific metadata
        if isinstance(self.sensor, LIDAR):
            metadata['sensor_B'] = int(self.sensor.B)
        elif isinstance(self.sensor, RangeBearingSensor):
            num_landmarks = self._get_landmark_count()
            if num_landmarks is not None:
                metadata['sensor_num_landmarks'] = int(num_landmarks)

        _numpy.savez_compressed(cache_path, Q_n=Q_n_save, **metadata)

    def _load_Q_n(self, cache_path: Path) -> bool:
        """Load Q_n from cache file if it exists and matches current parameters."""
        if not cache_path.exists():
            print(f"  Cache file does not exist: {cache_path}")
            return False

        print(f"  Checking cache file: {cache_path.name}")
        try:
            data = _numpy.load(str(cache_path))
            state_bounds_np = self.state_bounds.astype(float)
            state_bounds_np = state_bounds_np.get() if hasattr(state_bounds_np, 'get') else state_bounds_np

            max_val = self.motion_model.max_v if hasattr(self.motion_model, 'max_v') else self.motion_model.max_a
            expected_metadata = {
                'n': self.n,
                'obs_n': self.obs_n,
                'state_bounds': state_bounds_np,
                'max_val': max_val,
                'dt': float(self.motion_model.dt),
                'map_shape': _numpy.array([self.map_H, self.map_W]),
                'sigma_v': float(self.σ_v),
                'sensor_r_max': float(self.sensor.r_max),
                'm_y': self.Y_n.shape[0],
                'm_n': self.SQ.m_n,
                'len_M': self.len_M,
                'model': self.motion_model.__class__.__name__,
                'state_dim': self.state_dim,
                'problem_type': self._get_problem_type(),
                'kernel_version': 'v2-gaussian-borel-rb',
            }

            # Add sensor-specific metadata
            if isinstance(self.sensor, LIDAR):
                expected_metadata['sensor_B'] = int(self.sensor.B)
            elif isinstance(self.sensor, RangeBearingSensor):
                num_landmarks = self._get_landmark_count()
                if num_landmarks is not None:
                    expected_metadata['sensor_num_landmarks'] = int(num_landmarks)

            for key, expected_value in expected_metadata.items():
                if key not in data:
                    print(f"  Cache mismatch: key '{key}' not found in cache")
                    return False
                data_value = data[key]
                if isinstance(expected_value, _numpy.ndarray):
                    if not _numpy.array_equal(data_value, expected_value):
                        print(f"  Cache mismatch: '{key}' array values differ")
                        print(f"    Expected: {expected_value}")
                        print(f"    Cached: {data_value}")
                        return False
                else:
                    if isinstance(data_value, _numpy.ndarray):
                        try:
                            data_value = data_value.item()
                        except ValueError:
                            print(f"  Cache mismatch: '{key}' cannot be converted to scalar")
                            return False
                    if data_value != expected_value:
                        print(f"  Cache mismatch: '{key}' values differ")
                        print(f"    Expected: {expected_value} (type: {type(expected_value)})")
                        print(f"    Cached: {data_value} (type: {type(data_value)})")
                        return False

            self.Q_n = np.asarray(data['Q_n'])
            # Q_n shape is (m_y, ...) where ... is the shape returned by _get_Q_n_shape()
            expected_shape = (self.Y_n.shape[0],) + self._get_Q_n_shape()
            if self.Q_n.shape != expected_shape:
                print(f"  Cache mismatch: Q_n shape differs")
                print(f"    Expected: {expected_shape}")
                print(f"    Cached: {self.Q_n.shape}")
                return False

            print(f"  ✓ Cache metadata matches, loading Q_n")

            return True

        except Exception as e:
            print(f"Error loading Q_n cache: {e}")
            return False

    @abstractmethod
    def _get_problem_type(self) -> str:
        """Return problem type: 'slam', 'localization', or 'mapping'."""
        raise NotImplementedError


    @abstractmethod
    def _get_Q_n_shape(self) -> tuple:
        """Return expected shape of Q_n for this problem type."""
        raise NotImplementedError

    @abstractmethod
    def _Q_y_star(self):
        """
        Compute ideal observations y_star for all quantized states/maps.
        Must be implemented by subclasses.

        Returns:
            y_star: Array of shape determined by problem type:
                - SLAM: (m_n, len_M, B)
                - Localization: (m_n, B)
                - Mapping: (len_M, B) or (m_n, len_M, B) if pose is quantized
        """
        raise NotImplementedError

    def _compute_Q_n(self) -> np.ndarray:
        """
        Compute observation probability matrix Q_n over all observation cells and states/maps.
        Analogous to _compute_T_n but for observations.

        Q_n[k, ...] = Q(B_y_k | ...) where B_y_k is the k-th observation cell.
        """
        from scipy.stats import norm as univariate_norm

        m_y = self.Y_n.shape[0]
        expected_shape = self._get_Q_n_shape()
        Q_n = _numpy.zeros((m_y,) + expected_shape, dtype=_numpy.float64)

        # Get y_star for all states/maps (computed once)
        y_star = self._Q_y_star()  # Shape depends on problem type
        # Loop over observation cells
        for k in tqdm(range(m_y), desc="Computing Q_n"):
            Y_bounds_k = self.Y_bounds[k]  # (B, 2)
            # Convert CuPy array to NumPy if needed
            B_y_k = Y_bounds_k.get() if hasattr(Y_bounds_k, 'get') else _numpy.asarray(Y_bounds_k)
            mins = B_y_k[:, 0]  # (B,)
            maxs = B_y_k[:, 1]  # (B,)

            if hasattr(self, "sensor") and isinstance(self.sensor, RangeBearingSensor):
                # y_star shape: (..., L, 2)
                y_star_rb = _numpy.asarray(y_star)
                num_landmarks = int(self._get_landmark_count())
                y_star_rb = y_star_rb.reshape(*y_star_rb.shape[:-2], num_landmarks, 2)

                bounds_k = B_y_k.reshape(num_landmarks, 2, 2)
                r_min = bounds_k[:, 0, 0]
                r_max = bounds_k[:, 0, 1]
                phi_min = bounds_k[:, 1, 0]
                phi_max = bounds_k[:, 1, 1]

                r_star = _numpy.asarray(y_star_rb[..., 0])
                phi_star = _numpy.asarray(y_star_rb[..., 1])
                visible_mask = ~_numpy.isnan(r_star)

                sigma_r = float(self.sensor.sigma_r)
                sigma_phi = float(self.sensor.sigma_phi)
                eps = float(self.sensor.epsilon)
                r_max_sensor = float(self.sensor.r_max)

                denom = univariate_norm.cdf((r_max_sensor - r_star) / sigma_r) - \
                    univariate_norm.cdf((eps - r_star) / sigma_r)
                denom = _numpy.where(denom == 0.0, 1.0, denom)
                r_prob = (univariate_norm.cdf((r_max[_numpy.newaxis, _numpy.newaxis, :] - r_star) / sigma_r) -
                          univariate_norm.cdf((r_min[_numpy.newaxis, _numpy.newaxis, :] - r_star) / sigma_r)) / denom

                shifts = _numpy.array([-2.0 * _numpy.pi, 0.0, 2.0 * _numpy.pi])
                phi_min_shifted = phi_min[_numpy.newaxis, _numpy.newaxis, :] + shifts[:, _numpy.newaxis, _numpy.newaxis]
                phi_max_shifted = phi_max[_numpy.newaxis, _numpy.newaxis, :] + shifts[:, _numpy.newaxis, _numpy.newaxis]
                phi_star_expanded = phi_star[_numpy.newaxis, ...]
                b_phi = (phi_max_shifted[:, _numpy.newaxis, _numpy.newaxis, :] - phi_star_expanded) / sigma_phi
                a_phi = (phi_min_shifted[:, _numpy.newaxis, _numpy.newaxis, :] - phi_star_expanded) / sigma_phi
                bearing_prob = _numpy.sum(univariate_norm.cdf(b_phi) - univariate_norm.cdf(a_phi), axis=0)

                per_landmark_prob = _numpy.asarray(r_prob * bearing_prob)
                per_landmark_prob = _numpy.nan_to_num(per_landmark_prob, nan=0.0)

                Y_n_np = self.Y_n.get() if hasattr(self.Y_n, 'get') else _numpy.asarray(self.Y_n)
                sentinel_mask = _numpy.isnan(Y_n_np[k]).reshape(num_landmarks, 2).all(axis=1)
                per_landmark_prob = _numpy.where(
                    visible_mask,
                    per_landmark_prob,
                    sentinel_mask[_numpy.newaxis, :].astype(_numpy.float64)
                )

                Q_n[k] = _numpy.prod(per_landmark_prob, axis=-1)
                Q_n[k] = _numpy.nan_to_num(Q_n[k], nan=0.0)
            else:
                sigma_vec = float(self.σ_v)
                a = (mins[_numpy.newaxis, :] - y_star) / sigma_vec
                b = (maxs[_numpy.newaxis, :] - y_star) / sigma_vec

                cdf_b = univariate_norm.cdf(b)
                cdf_a = univariate_norm.cdf(a)
                per_beam_probs = _numpy.clip(cdf_b - cdf_a, 0.0, 1.0)
                if _numpy.isnan(y_star).any():
                    nan_mask = _numpy.isnan(y_star)
                    per_beam_probs = _numpy.where(nan_mask, 1.0, per_beam_probs)

                Q_n[k] = _numpy.prod(per_beam_probs, axis=-1)

        # Normalize Q_n: for each (state, map) pair, sum over observations should equal 1
        # Q_n shape: (m_y, ...) where ... is (m_n, len_M) for SLAM
        col_sums = Q_n.sum(axis=0, keepdims=True)  # (1, m_n, len_M) or (1, ...)
        # Avoid division by zero - if sum is 0, keep Q_n as is (all zeros)
        col_sums = _numpy.where(col_sums == 0.0, 1.0, col_sums)
        Q_n = Q_n / col_sums

        return np.asarray(Q_n)

    def _compute_Q_batch_for_obs_indices(self, obs_indices: np.ndarray) -> np.ndarray:
        """
        Compute Q_n slices for a batch of observation indices without building full Q_n.

        This is useful when obs_n is large and Q_n would be too big to store.
        Returns Q_batch with shape (K, ...) where ... matches _get_Q_n_shape().
        """
        from scipy.stats import norm as univariate_norm

        obs_indices = _numpy.asarray(obs_indices, dtype=_numpy.int64)
        expected_shape = self._get_Q_n_shape()
        Q_batch = _numpy.zeros((obs_indices.shape[0],) + expected_shape, dtype=_numpy.float64)

        y_star = self._Q_y_star()

        if hasattr(self, "sensor") and isinstance(self.sensor, RangeBearingSensor):
            y_star_rb = _numpy.asarray(y_star)
            num_landmarks = int(self._get_landmark_count())
            y_star_rb = y_star_rb.reshape(*y_star_rb.shape[:-2], num_landmarks, 2)

            r_star = _numpy.asarray(y_star_rb[..., 0])
            phi_star = _numpy.asarray(y_star_rb[..., 1])
            visible_mask = ~_numpy.isnan(r_star)

            sigma_r = float(self.sensor.sigma_r)
            sigma_phi = float(self.sensor.sigma_phi)
            eps = float(self.sensor.epsilon)
            r_max_sensor = float(self.sensor.r_max)

            denom = univariate_norm.cdf((r_max_sensor - r_star) / sigma_r) - \
                univariate_norm.cdf((eps - r_star) / sigma_r)
            denom = _numpy.where(denom == 0.0, 1.0, denom)

            shifts = _numpy.array([-2.0 * _numpy.pi, 0.0, 2.0 * _numpy.pi])

            Y_n_np = self.Y_n.get() if hasattr(self.Y_n, 'get') else _numpy.asarray(self.Y_n)
            for i, k in enumerate(obs_indices):
                bounds_k = self.Y_bounds[k].reshape(num_landmarks, 2, 2)
                r_min = bounds_k[:, 0, 0]
                r_max = bounds_k[:, 0, 1]
                phi_min = bounds_k[:, 1, 0]
                phi_max = bounds_k[:, 1, 1]

                # Broadcast bounds to match r_star/phi_star dimensions
                expand_shape = (1,) * (r_star.ndim - 1) + (num_landmarks,)
                r_min_b = r_min.reshape(expand_shape)
                r_max_b = r_max.reshape(expand_shape)
                phi_min_b = phi_min.reshape(expand_shape)
                phi_max_b = phi_max.reshape(expand_shape)

                r_prob = (univariate_norm.cdf((r_max_b - r_star) / sigma_r) -
                          univariate_norm.cdf((r_min_b - r_star) / sigma_r)) / denom

                phi_min_shifted = phi_min_b.reshape((1,) + phi_min_b.shape) + \
                    shifts[:, _numpy.newaxis, _numpy.newaxis, _numpy.newaxis]
                phi_max_shifted = phi_max_b.reshape((1,) + phi_max_b.shape) + \
                    shifts[:, _numpy.newaxis, _numpy.newaxis, _numpy.newaxis]
                phi_star_expanded = phi_star[_numpy.newaxis, ...]
                b_phi = (phi_max_shifted - phi_star_expanded) / sigma_phi
                a_phi = (phi_min_shifted - phi_star_expanded) / sigma_phi
                bearing_prob = _numpy.sum(univariate_norm.cdf(b_phi) - univariate_norm.cdf(a_phi), axis=0)

                per_landmark_prob = _numpy.asarray(r_prob * bearing_prob)
                per_landmark_prob = _numpy.nan_to_num(per_landmark_prob, nan=0.0)

                sentinel_mask = _numpy.isnan(Y_n_np[k]).reshape(num_landmarks, 2).all(axis=1)
                per_landmark_prob = _numpy.where(
                    visible_mask,
                    per_landmark_prob,
                    sentinel_mask.reshape(expand_shape).astype(_numpy.float64)
                )

                Q_batch[i] = _numpy.prod(per_landmark_prob, axis=-1)
                Q_batch[i] = _numpy.nan_to_num(Q_batch[i], nan=0.0)
            return Q_batch

        for i, k in enumerate(obs_indices):
            B_y_k = self.Y_bounds[k]
            B_y_k = B_y_k.get() if hasattr(B_y_k, 'get') else _numpy.asarray(B_y_k)
            mins = B_y_k[:, 0]
            maxs = B_y_k[:, 1]
            sigma_vec = float(self.σ_v)
            a = (mins[_numpy.newaxis, :] - y_star) / sigma_vec
            b = (maxs[_numpy.newaxis, :] - y_star) / sigma_vec
            cdf_b = univariate_norm.cdf(b)
            cdf_a = univariate_norm.cdf(a)
            per_beam_probs = _numpy.clip(cdf_b - cdf_a, 0.0, 1.0)
            if _numpy.isnan(y_star).any():
                nan_mask = _numpy.isnan(y_star)
                per_beam_probs = _numpy.where(nan_mask, 1.0, per_beam_probs)
            Q_batch[i] = _numpy.prod(per_beam_probs, axis=-1)

        return np.asarray(Q_batch)


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
    def ρ_n(self, Π_batch: np.ndarray, *args: Any) -> np.ndarray:
        """
        Expected cost function for belief-MDP. Must be implemented by subclasses.

        Note: Signature varies by problem type:
        - SLAM: c_tilde_n(π, u)
        - Localization: c_tilde_n(π, u)
        - Mapping: c_tilde_n(π, x_current, u) - may need current pose for cost computation
        """
        raise NotImplementedError


    @abstractmethod
    def η_n(self, π_new: np.ndarray, π: np.ndarray, *args) -> float:
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
        measurement_model: LIDAR | RangeBearingSensor,
        obstacles: list[Obstacle],
        _map: BaseMap,
        sigma_w: float = 0.01,
        sigma_v: float = 0.01,
        exploration_type: Literal['information gain', 'wasserstein distance'] = 'information gain',
        obs_n: int | None = None,
        action_n: int | None = None
    ):
        # Initialize POMDP parent first
        SLAM_POMDP.__init__(self, motion_model, measurement_model, obstacles, _map, sigma_w, sigma_v)

        # Initialize common BeliefMDP_n components (quantizers, T_mat)
        # Override _init_common to initialize all_maps_3d before Q_n computation
        # If obs_n is None, use n (backward compatibility)
        self._init_common_slam(n, motion_model, _map, obs_n=obs_n if obs_n is not None else n, action_n=action_n)

        self.exploration_type = exploration_type

    def _get_problem_type(self) -> str:
        return 'slam'

    def _init_common_slam(self, n: int, motion_model, _map, obs_n: int | None = None, action_n: int | None = None):
        """
        SLAM-specific initialization that sets up all_maps_3d before Q_n computation.
        This is needed because _Q_y_star() requires all_maps_3d.
        
        Args:
            n: State quantization level
            motion_model: Motion model instance
            _map: Map instance
            obs_n: Observation quantization level (defaults to n if None)
            action_n: Action quantization level (defaults to n if None)
        """
        self.n = n
        if obs_n is None:
            obs_n = n
        if action_n is None:
            action_n = n

        def to_float(x):
            return float(x.item() if hasattr(x, 'item') else x)

        # State space quantizer setup (from BaseBeliefMDP_n._init_common)
        if isinstance(motion_model, SingleIntegratorModel):
            ll = _map.left_lower
            ru = _map.right_upper
            state_bounds = np.array([
                [to_float(ll[0]), to_float(ru[0])],
                [to_float(ll[1]), to_float(ru[1])]
            ], dtype=float)
            self.SQ = StateQuantizer(
                (to_float(ll[0]), to_float(ru[0]), to_float(ll[1]), to_float(ru[1])),
                n
            )
            self.state_dim = 2
            self.AQ = ActionQuantizer(motion_model.max_v, action_n)
        elif isinstance(motion_model, DoubleIntegratorModel):
            ll = _map.left_lower
            ru = _map.right_upper
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
            self.AQ = ActionQuantizer(motion_model.max_a, action_n)
        else:
            raise ValueError(f"Unsupported motion model type: {type(motion_model)}")

        self.state_bounds = state_bounds
        if hasattr(motion_model, "set_state_bounds"):
            motion_model.set_state_bounds(self.state_bounds)

        # Get map dimensions - works for both occupancy grids and landmark maps
        map_shape = _map.map_shape
        self.map_distance_p = float('inf') if len(map_shape) == 2 else 2.0
        if len(map_shape) == 2:
            # Occupancy grid: (H, W)
            self.map_H, self.map_W = map_shape
        else:
            # Landmark map or other: use map_shape[0] as primary dimension
            # For compatibility, set map_H and map_W to map_shape[0] and 1
            self.map_H = map_shape[0]
            self.map_W = 1 if len(map_shape) == 1 else map_shape[1]
        self.len_M = _map.len_M

        # Initialize all_maps_3d BEFORE Q_n computation (needed by _Q_y_star)
        # Load or generate cached space of maps for vectorized operations
        # Only cache if map space is manageable (<= 2^16 = 65536 maps)
        # For occupancy grids: check H*W, for landmark maps: check len_M directly
        if hasattr(self.map, 'occupancy_map'):
            # Occupancy grid: check H*W
            if self.map_H * self.map_W <= 16:
                self.all_maps_3d, self.map_bits_array = self._load_or_generate_all_maps()
            else:
                self.all_maps_3d = None
                self.map_bits_array = None
        else:
            # Landmark map or other: check len_M directly
            if self.len_M <= 65536:
                self.all_maps_3d, self.map_bits_array = self._load_or_generate_all_maps()
            else:
                self.all_maps_3d = None
                self.map_bits_array = None

        # ----------------------------
        # Observation space quantizer
        # ----------------------------
        # We quantize each LIDAR beam independently on [0, r_max] with a uniform grid.
        # The full observation alphabet is then the Cartesian product Y_n = {centers}^B.
        # NOTE: |Y_n| = n^B, so the user MUST choose small B and/or n in practice.
        self.obs_n = obs_n  # Observation quantization level (can differ from state/action n)

        if hasattr(self, "sensor") and isinstance(self.sensor, LIDAR):
            b_bounds = [(0.0, float(self.sensor.r_max))] * int(self.sensor.B)
            self.YQ = ObservationQuantizer(b_bounds, self.obs_n)
            self.Y_n = self.YQ.Y_n              # (m_y, B)
            self.Y_bounds = self.YQ.get_bounds()  # (m_y, B, 2)
        elif hasattr(self, "sensor") and isinstance(self.sensor, RangeBearingSensor):
            num_landmarks = self._get_landmark_count(_map)
            if num_landmarks is None:
                raise ValueError("RangeBearingSensor requires LandmarkMap with num_candidate_landmarks")

            eps = float(self.sensor.epsilon)
            r_max = float(self.sensor.r_max)
            r_vals = _numpy.linspace(eps, r_max, self.obs_n)
            dr = (r_max - eps) / self.obs_n
            r_bounds = _numpy.stack([r_vals - dr / 2.0, r_vals + dr / 2.0], axis=1)
            r_bounds[:, 0] = _numpy.clip(r_bounds[:, 0], eps, r_max)
            r_bounds[:, 1] = _numpy.clip(r_bounds[:, 1], eps, r_max)

            phi_vals = _numpy.linspace(-_numpy.pi, _numpy.pi, self.obs_n, endpoint=False)
            dphi = 2.0 * _numpy.pi / self.obs_n
            phi_bounds = _numpy.stack([phi_vals - dphi / 2.0, phi_vals + dphi / 2.0], axis=1)

            rr, pp = _numpy.meshgrid(r_vals, phi_vals, indexing='ij')
            per_landmark_points = _numpy.stack([rr.reshape(-1), pp.reshape(-1)], axis=1)
            per_landmark_bounds = _numpy.stack([
                r_bounds.repeat(self.obs_n, axis=0),
                _numpy.tile(phi_bounds, (self.obs_n, 1))
            ], axis=1)

            per_landmark_points = _numpy.vstack([per_landmark_points, _numpy.array([[_numpy.nan, _numpy.nan]])])
            per_landmark_bounds = _numpy.vstack([per_landmark_bounds, _numpy.array(
                [[[_numpy.nan, _numpy.nan], [_numpy.nan, _numpy.nan]]])])

            n_points = per_landmark_points.shape[0]
            idx_grid = _numpy.indices((n_points,) * int(num_landmarks)).reshape(int(num_landmarks), -1).T
            self.Y_n = per_landmark_points[idx_grid].reshape(-1, int(num_landmarks) * 2)
            self.Y_bounds = per_landmark_bounds[idx_grid].reshape(-1, int(num_landmarks) * 2, 2)
            self.YQ = None
        else:
            self.YQ = None
            self.Y_n = None
            self.Y_bounds = None

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

        # Load or compute Q_n (observation probability matrix)
        # Now all_maps_3d is initialized, so _Q_y_star() will work
        if self.Y_n is not None:
            Q_cache_path = self._get_Q_cache_path()
            if self._load_Q_n(Q_cache_path):
                print(f"Loaded cached Q_n from {Q_cache_path}")
            else:
                print("Computing Q_n for the first time...")
                self.Q_n = self._compute_Q_n()
                self._save_Q_n(Q_cache_path)
                print(f"Saved Q_n to {Q_cache_path}")
        else:
            self.Q_n = None

    def _get_Q_n_shape(self) -> tuple:
        return (self.SQ.m_n, self.len_M)

    def _Q_y_star(self):
        """Compute y_star for SLAM: (m_n, len_M, B)."""
        if self.all_maps_3d is None:
            raise ValueError("all_maps_3d must be precomputed for SLAM Q_n")
        X_in = self.SQ.X_n if not isinstance(self.sensor, LIDAR) else (
            self.SQ.X_n[:, :2] if self.SQ.X_n.shape[1] > 2 else self.SQ.X_n
        )
        y_star_result = self.ray_casting_batched(X_in, self.all_maps_3d)
        # Convert CuPy array to NumPy if needed
        y_star = y_star_result.get() if hasattr(y_star_result, 'get') else _numpy.asarray(y_star_result)
        return y_star

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
    def F(self, π, u, y):
        """
        Filter update for a single observation.

        Wrapper around F_batch_log for convenience.

        Args:
            π: current belief (m_n, len_M)
            u: action (2,)
            y: observation (B,) or (1, B)
        Returns:
            π_new: updated belief (m_n, len_M) or list containing it
        """
        # Ensure y is 2D
        y = np.asarray(y)
        if y.ndim == 1:
            y = y[np.newaxis, :]  # (1, B)

        # Use F_batch_log and return first result
        π_new_batch = self.F_batch_log(π, u, y)  # (1, m_n, len_M)
        return [π_new_batch[0]]  # Return as list for compatibility with test code

    def F_batch_log(self, π, u, Y_batch):
        """
        Batched filter update in log-space for numerical stability.

        Reuses H computation: F = (Q * predicted) / H, where H = sum(Q * predicted).

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
        predicted = Tn_mat @ π  # (m_n, len_M)
        log_predicted = np.log(np.maximum(predicted, 1e-300))

        # Find observation indices for Y_batch
        Y_batch = self._prepare_observation_batch(Y_batch)
        obs_indices = self._find_observation_indices(Y_batch)

        if self.Q_n is None:
            Q_batch = self._compute_Q_batch_for_obs_indices(obs_indices)
        else:
            # Q_n shape: (m_y, m_n, len_M)
            # Convert obs_indices to same backend as Q_n for proper indexing
            if hasattr(self.Q_n, 'device'):
                # Q_n is CuPy - convert obs_indices to CuPy
                import cupy as cp
                obs_indices_cp = cp.asarray(obs_indices)
                Q_batch = self.Q_n[obs_indices_cp]  # (K, m_n, len_M)
            else:
                # Q_n is NumPy - use NumPy indices
                Q_batch = self.Q_n[obs_indices]  # (K, m_n, len_M)
        log_Q_batch = np.log(np.maximum(Q_batch, 1e-300))

        # Compute numerator: Q * predicted (in log space)
        log_numerator = log_Q_batch + \
            np.broadcast_to(log_predicted[np.newaxis, :, :], (Y_batch.shape[0], *log_predicted.shape))

        # Compute H (normalization constant) = sum(Q * predicted) over (x', m)
        log_numerator_2d = log_numerator.reshape(Y_batch.shape[0], -1)
        log_H = logsumexp(log_numerator_2d, axis=1, keepdims=True)  # (K, 1)

        # F = numerator / H (in log space: log_F = log_numerator - log_H)
        log_F_2d = log_numerator_2d - log_H
        return np.exp(log_F_2d).reshape(Y_batch.shape[0], *predicted.shape)

    ###################################################
    ### 2. Belief State Transitions η_n ###############
    def _compute_H_y_and_F(self, π: np.ndarray, u: np.ndarray):
        """
        Helper method to compute observation probabilities H_y and updated beliefs F for all observations.
        Reused by η_n and r_information_gain to avoid redundant computation.

        Args:
            π: current belief (m_n, len_M)
            u: action (2,)
        Returns:
            H_y: observation probabilities (m_y,)
            π_all_batch: updated beliefs for all observations (m_y, m_n, len_M)
        """
        if self.Q_n is None:
            raise ValueError("Q_n must be precomputed. Ensure observation quantization is configured.")

        # 1) Predicted belief over (x', m) using T_mat
        Tn_mat = self.T_mat[:, :, self.AQ.get_quantized_index(u)]  # (m_n, m_n)
        predicted = Tn_mat @ π  # (m_n, len_M)
        log_predicted = np.log(np.maximum(predicted, 1e-300))

        # 2) Compute H for ALL observations at once: H_y = sum(Q_n * predicted) over (x', m)
        m_y = int(self.Y_n.shape[0])
        # Q_n shape: (m_y, m_n, len_M), predicted shape: (m_n, len_M)
        log_Q_n = np.log(np.maximum(self.Q_n, 1e-300))  # (m_y, m_n, len_M)
        log_numerator = log_Q_n + log_predicted[np.newaxis, :, :]  # (m_y, m_n, len_M)

        # H_y = sum(Q_n * predicted) over (x', m) - compute in log space for stability
        log_numerator_2d = log_numerator.reshape(m_y, -1)  # (m_y, m_n * len_M)
        log_H_y = logsumexp(log_numerator_2d, axis=1)  # (m_y,)
        H_y = np.exp(log_H_y)  # (m_y,)

        # Normalize H_y to be a proper distribution
        total_H = np.sum(H_y)
        if total_H > 0:
            H_y = H_y / total_H

        # 3) Compute F for ALL observations directly from Q_n and predicted (reusing H_y computation!)
        # F = (Q_n * predicted) / H_y, so log_F = log_numerator - log_H_y
        log_F_2d = log_numerator_2d - log_H_y[:, np.newaxis]  # (m_y, m_n * len_M)
        π_all_batch = np.exp(log_F_2d).reshape(m_y, *predicted.shape)  # (m_y, m_n, len_M)

        return H_y, π_all_batch

    def η_n(self, π_new: np.ndarray, π: np.ndarray, u: np.ndarray) -> float:
        """
        Compute transition probability η_n(π' | π, u) using finite observation quantization.

        Fully vectorized implementation: computes F for ALL observations at once, then
        vectorized distance computation and matching.

        With a finite quantized observation alphabet Y_n, we approximate:
            η_n(π' | π, u) = ∑_{y∈Y_n} 𝟙_{F(π,u,y) ≈ π'} · H({y} | π, u)

        Args:
            π_new: target belief, either (m_n, len_M) or (N_n,) where N_n = m_n * len_M
            π: current belief, either (m_n, len_M) or (N_n,)
            u: action (2,)
        Returns:
            float: probability of transitioning to π' from π under action u
        """
        # Normalize beliefs to 2D format if needed
        π_new_2d = π_new if π_new.ndim == 2 else self.unflatten_belief(π_new)
        π_2d = π if π.ndim == 2 else self.unflatten_belief(π)

        # Use shared helper to compute H_y and F
        H_y, π_all_batch = self._compute_H_y_and_F(π_2d, u)
        m_y = int(self.Y_n.shape[0])

        # 4) Vectorized distance computation: L2 norm between each π_k and π_new
        π_new_broadcast = np.broadcast_to(π_new_2d[np.newaxis, :, :], (m_y, *π_new_2d.shape))  # (m_y, m_n, len_M)
        π_diff = π_all_batch - π_new_broadcast  # (m_y, m_n, len_M)
        distances = np.linalg.norm(π_diff.reshape(m_y, -1), axis=1)  # (m_y,)

        # 5) Vectorized matching: mask observations where distance < threshold
        threshold = 1e-3
        matches = (distances < threshold) & (H_y > 0.0)  # (m_y,)

        # 6) Sum H_y for matching observations
        prob = float(np.sum(H_y[matches]))

        return prob

    ###################################################
    ### 3. Cost Function ##############################
    def r_exploration(self, Π_batch: np.ndarray) -> np.ndarray:
        """
        Exploration reward: action-independent belief cost for a batch of beliefs.
        """
        if self.exploration_type == 'information gain':
            return self.r_information_gain(Π_batch)
        elif self.exploration_type == 'wasserstein distance':
            return self.r_wasserstein_distance(Π_batch)
        else:
            raise ValueError(f"Invalid exploration type: {self.exploration_type}")

    def H(self, π: np.ndarray) -> np.ndarray:
        """
        Compute entropy of belief distribution(s): H(π) = -Σ π * log(π)

        Vectorized to handle batches of beliefs.

        Args:
            π: belief distribution(s), shape (..., m_n, len_M) for SLAM, (..., m_n,) for Localization, (..., len_M,) for Mapping
        Returns:
            entropy: entropy value(s), scalar if single belief, array if batched
        """
        problem_type = self._get_problem_type()
        is_single = (π.ndim == 1) or (problem_type == 'slam' and π.ndim == 2)

        if π.ndim == 1:
            π_flat = π[np.newaxis, :]
        elif π.ndim == 2:
            if problem_type == 'slam':
                π_flat = π.reshape(1, -1)
            else:
                π_flat = π
        else:
            π_flat = π.reshape(π.shape[0], -1)

        # Only consider non-zero probabilities to avoid log(0)
        π_nonzero = np.maximum(π_flat, 1e-300)  # (n_batch, n_states)
        log_π = np.log(π_nonzero)
        H_vals = -np.sum(π_nonzero * log_π, axis=1)  # (n_batch,)

        return H_vals[0] if is_single else H_vals

    def _prepare_observation_batch(self, Y_batch: np.ndarray) -> np.ndarray:
        """
        Normalize observation batch to shape (K, B).
        """
        Y = np.asarray(Y_batch)
        if Y.ndim == 1:
            Y = Y[np.newaxis, :]
        elif Y.ndim == 3:
            Y = Y.reshape(Y.shape[0], -1)
        return Y

    def _get_landmark_count(self, map_obj=None):
        """
        Get number of landmarks used in the measurement model.
        Prefer max_landmarks when available.
        """
        map_obj = map_obj or self.map
        count = getattr(map_obj, 'num_landmarks', None)
        if count is None:
            count = getattr(map_obj, 'max_landmarks', None)
        if count is None:
            count = getattr(map_obj, 'num_candidate_landmarks', None)
        return count

    def _find_observation_indices(self, Y_batch: np.ndarray) -> np.ndarray:
        """
        Find nearest observation indices in Y_n, handling sentinel ⊥ for RangeBearingSensor.
        """
        if hasattr(self, "sensor") and isinstance(self.sensor, RangeBearingSensor):
            Y_batch_np = Y_batch.get() if hasattr(Y_batch, 'get') else _numpy.asarray(Y_batch)
            Y_n_np = self.Y_n.get() if hasattr(self.Y_n, 'get') else _numpy.asarray(self.Y_n)
            Y_n_nan = _numpy.isnan(Y_n_np)
            nan_mask = _numpy.isnan(Y_batch_np)
            obs_indices = _numpy.empty(Y_batch_np.shape[0], dtype=_numpy.int64)
            for i in range(Y_batch_np.shape[0]):
                if _numpy.any(nan_mask[i]):
                    matches = _numpy.all(Y_n_nan == nan_mask[i], axis=1)
                    if _numpy.any(matches):
                        non_nan = ~nan_mask[i]
                        Y_candidates = Y_n_np[matches][:, non_nan]
                        y_target = Y_batch_np[i, non_nan]
                        dist = _numpy.linalg.norm(Y_candidates - y_target, axis=1)
                        obs_indices[i] = int(_numpy.where(matches)[0][_numpy.argmin(dist)])
                    else:
                        dist = _numpy.linalg.norm(_numpy.nan_to_num(Y_n_np) - _numpy.nan_to_num(Y_batch_np[i]), axis=1)
                        dist = _numpy.where(_numpy.isnan(Y_n_np).any(axis=1), _numpy.inf, dist)
                        obs_indices[i] = int(_numpy.argmin(dist))
                else:
                    dist = _numpy.linalg.norm(Y_n_np - Y_batch_np[i], axis=1)
                    dist = _numpy.where(_numpy.isnan(dist), _numpy.inf, dist)
                    obs_indices[i] = int(_numpy.argmin(dist))
            return obs_indices

        return np.argmin(cdist(Y_batch, self.Y_n), axis=1)

    def _compute_H_y_and_F_batched(self, Π_batch: np.ndarray, U_batch: np.ndarray):
        """
        Batched version of _compute_H_y_and_F for all belief-action pairs.

        Args:
            Π_batch: beliefs (n_π, m_n, len_M)
            U_batch: actions (n_u, 2)
        Returns:
            H_y_batch: observation probabilities (n_π, n_u, m_y)
            π_all_batch: updated beliefs (n_π, n_u, m_y, m_n, len_M)
        """
        if self.Q_n is None:
            raise ValueError("Q_n must be precomputed. Ensure observation quantization is configured.")

        n_π, n_u = Π_batch.shape[0], U_batch.shape[0]
        m_y = int(self.Y_n.shape[0])
        m_n, len_M = Π_batch.shape[1], Π_batch.shape[2]

        # Get action indices for all actions
        u_indices = np.array([self.AQ.get_quantized_index(u) for u in U_batch])  # (n_u,)

        # 1) Compute predicted beliefs for all (belief, action) pairs (fully vectorized)
        # T_mat: (m_n, m_n, n_u), Π_batch: (n_π, m_n, len_M)
        # Extract T matrices for all actions: (m_n, m_n, n_u)
        T_mat_selected = self.T_mat[:, :, u_indices]  # (m_n, m_n, n_u)

        # Vectorized matrix multiplication: (m_n, m_n, n_u) @ (n_π, m_n, len_M) -> (n_π, n_u, m_n, len_M)
        # Using einsum: 'ijk,ilm->ijlm' where i=m_n, j=n_u, k=m_n, l=n_π, m=len_M
        # Actually: T_mat_selected: (m_n, m_n, n_u), Π_batch: (n_π, m_n, len_M)
        # We want: for each (π_i, u_j): T_mat[:, :, u_j] @ π_i
        # Using einsum: 'jkl,il->ijk' but we need len_M dimension
        # Better: 'jkl,ilm->ijlm' where j indexes actions, k and l are state dims, i indexes beliefs, m indexes maps
        # T_mat_selected: (m_n, m_n, n_u) -> rearrange to (n_u, m_n, m_n)
        T_mat_rearranged = T_mat_selected.transpose(2, 0, 1)  # (n_u, m_n, m_n)
        # Now: (n_u, m_n, m_n) @ (n_π, m_n, len_M) -> (n_π, n_u, m_n, len_M)
        predicted_batch = np.einsum('jkl,ilm->ijlm', T_mat_rearranged, Π_batch)  # (n_π, n_u, m_n, len_M)

        log_predicted_batch = np.log(np.maximum(predicted_batch, 1e-300))  # (n_π, n_u, m_n, len_M)

        # 2) Compute H_y for all (belief, action, observation) combinations
        # Q_n: (m_y, m_n, len_M), log_predicted_batch: (n_π, n_u, m_n, len_M)
        log_Q_n = np.log(np.maximum(self.Q_n, 1e-300))  # (m_y, m_n, len_M)

        # Broadcast: (1, 1, m_y, m_n, len_M) + (n_π, n_u, 1, m_n, len_M) -> (n_π, n_u, m_y, m_n, len_M)
        log_numerator = log_Q_n[np.newaxis, np.newaxis, :, :, :] + \
            log_predicted_batch[:, :, np.newaxis, :, :]  # (n_π, n_u, m_y, m_n, len_M)

        # H_y = sum(Q_n * predicted) over (x', m) - compute in log space
        log_numerator_flat = log_numerator.reshape(n_π, n_u, m_y, -1)  # (n_π, n_u, m_y, m_n * len_M)
        log_H_y_batch = logsumexp(log_numerator_flat, axis=3)  # (n_π, n_u, m_y)
        H_y_batch = np.exp(log_H_y_batch)  # (n_π, n_u, m_y)

        # Normalize H_y for each (belief, action) pair
        total_H = np.sum(H_y_batch, axis=2, keepdims=True)  # (n_π, n_u, 1)
        H_y_batch = np.where(total_H > 0, H_y_batch / total_H, H_y_batch)  # (n_π, n_u, m_y)

        # 3) Compute F for all combinations
        # F = (Q_n * predicted) / H_y, so log_F = log_numerator - log_H_y
        log_F_flat = log_numerator_flat - log_H_y_batch[:, :, :, np.newaxis]  # (n_π, n_u, m_y, m_n * len_M)
        π_all_batch = np.exp(log_F_flat).reshape(n_π, n_u, m_y, m_n, len_M)  # (n_π, n_u, m_y, m_n, len_M)

        return H_y_batch, π_all_batch

    def r_information_gain(self, Π_batch: np.ndarray) -> np.ndarray:
        """
        Stage-wise entropy reward: r_information_gain(b_t) = H(b_t).

        Args:
            Π_batch: beliefs (n_π, ...) or a single belief
        Returns:
            r_batch: (n_π,) where r_batch[i] = H(Π_batch[i])
        """
        H_π_batch = self.H(Π_batch)
        return np.atleast_1d(H_π_batch)

    def r_wasserstein_distance(self, Π_batch: np.ndarray) -> np.ndarray:
        """
        Wasserstein distance reward for SLAM: compute on map marginals only.
        """
        if Π_batch.ndim == 2:
            Π_batch = Π_batch[np.newaxis, :, :]
        # Marginalize joint belief over pose to get map belief
        b_map_batch = np.sum(Π_batch, axis=1)  # (n_π, len_M)
        map_cost = BeliefMDP_n_Mapping.r_wasserstein_distance(self, b_map_batch)

        # Pose dispersion: E[||x - x_hat||_2] under pose marginal
        b_pose_batch = np.sum(Π_batch, axis=2)  # (n_π, m_n)
        x_support = self.SQ.X_n  # (m_n, state_dim)
        x_hat = b_pose_batch @ x_support  # (n_π, state_dim)
        diffs = x_support[np.newaxis, :, :] - x_hat[:, np.newaxis, :]  # (n_π, m_n, state_dim)
        dists = np.linalg.norm(diffs, axis=2)  # (n_π, m_n)
        pose_cost = np.sum(b_pose_batch * dists, axis=1)  # (n_π,)

        return map_cost + pose_cost

    def ρ_n(self, Π_batch: np.ndarray, U_batch: np.ndarray) -> np.ndarray:
        if Π_batch.ndim == 2:
            Π_batch = Π_batch[np.newaxis, :, :]
        if U_batch.ndim == 1:
            U_batch = U_batch[np.newaxis, :]

        c_effort_batch = self.c_effort(U_batch)  # (n_u,)
        r_exploration_batch = np.atleast_1d(self.r_exploration(Π_batch))  # (n_π,)
        return r_exploration_batch[:, np.newaxis] + c_effort_batch[np.newaxis, :]


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
        measurement_model: LIDAR | RangeBearingSensor,
        obstacles: list[Obstacle],
        _map: BaseMap,
        sigma_w: float = 0.01,
        sigma_v: float = 1,
        obs_n: int | None = None,
        action_n: int | None = None
    ):
        # Initialize POMDP parent first
        Localization_POMDP.__init__(self, motion_model, measurement_model, obstacles, _map, sigma_w, sigma_v)

        # Initialize common BeliefMDP_n components (quantizers, T_mat)
        # If obs_n is None, use n (backward compatibility)
        BaseBeliefMDP_n._init_common(self, n, motion_model, _map, obs_n=obs_n if obs_n is not None else n, action_n=action_n)

    def _get_problem_type(self) -> str:
        return 'localization'

    def _get_Q_n_shape(self) -> tuple:
        return (self.SQ.m_n,)

    def _Q_y_star(self):
        """Compute y_star for Localization: (m_n, B)."""
        if self.known_map_obstacle_segments is None:
            raise ValueError("Known map obstacle segments not set. Call set_known_map() first.")
        X_in = self.SQ.X_n if not isinstance(self.sensor, LIDAR) else (
            self.SQ.X_n[:, :2] if self.SQ.X_n.shape[1] > 2 else self.SQ.X_n
        )
        y_star_result = self.ray_casting(X_in)
        # Convert CuPy array to NumPy if needed
        y_star = y_star_result.get() if hasattr(y_star_result, 'get') else _numpy.asarray(y_star_result)
        return y_star

    ##########################################################
    ### 1. Filter Update and Observation Functions ######


    def F(self, π, u, y):
        """
        Filter update for a single observation.

        Wrapper around F_batch_log for convenience.

        Args:
            π: current belief (m_n,)
            u: action (2,)
            y: observation (B,) or (1, B)
        Returns:
            π_new: updated belief (m_n,) or list containing it
        """
        # Ensure y is 2D
        y = np.asarray(y)
        if y.ndim == 1:
            y = y[np.newaxis, :]  # (1, B)

        # Use F_batch_log and return first result
        π_new_batch = self.F_batch_log(π, u, y)  # (1, m_n)
        return [π_new_batch[0]]  # Return as list for compatibility with test code

    def F_batch_log(self, π, u, Y_batch):
        """
        Batched filter update in log-space for numerical stability.

        Reuses H computation: F = (Q * predicted) / H, where H = sum(Q * predicted).

        Args:
            π: current belief (m_n,)
            u: action (2,)
            Y_batch: observations (K, B)
        Returns:
            π_new_batch: updated beliefs (K, m_n)
        """
        if self.known_map_obstacle_segments is None:
            raise ValueError("Known map obstacle segments not set. Call set_known_map() first.")

        if self.Q_n is None:
            raise ValueError("Q_n must be precomputed. Ensure observation quantization is configured.")
        Tn_mat = self.T_mat[:, :, self.AQ.get_quantized_index(u)]
        predicted = Tn_mat @ π  # (m_n,)
        log_predicted = np.log(np.maximum(predicted, 1e-300))

        # Find observation indices for Y_batch
        Y_batch = self._prepare_observation_batch(Y_batch)
        obs_indices = self._find_observation_indices(Y_batch)

        # Q_n shape: (m_y, m_n)
        Q_batch = self.Q_n[obs_indices]  # (K, m_n)
        log_Q_batch = np.log(np.maximum(Q_batch, 1e-300))

        # Compute numerator: Q * predicted (in log space)
        log_numerator = log_Q_batch + \
            np.broadcast_to(log_predicted[np.newaxis, :], (Y_batch.shape[0], len(log_predicted)))

        # Compute H (normalization constant) = sum(Q * predicted) over poses
        log_H = logsumexp(log_numerator, axis=1, keepdims=True)  # (K, 1)

        # F = numerator / H (in log space: log_F = log_numerator - log_H)
        log_F = log_numerator - log_H
        return np.exp(log_F)

    def c_tilde_n(self, Π_batch: np.ndarray, U_batch: np.ndarray) -> np.ndarray:
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


    def η_n(self, π_new: np.ndarray, π: np.ndarray, u: np.ndarray) -> float:
        """
        Compute transition probability η_n(π' | π, u) using finite observation quantization.

        With a finite quantized observation alphabet Y_n, we approximate

            η_n(π' | π, u) = ∑_{y∈Y_n} 𝟙_{F(π,u,y) ≈ π'} · H({y} | π, u)

        where H({y_k} | π, u) is the probability mass of the k-th observation
        cell, and F is evaluated at the corresponding representative y_k.

        Args:
            π_new: target belief (m_n,)
            π: current belief (m_n,)
            u: action (2,)
        Returns:
            float: probability of transitioning to π' from π under action u
        """
        # Ensure beliefs are 1D
        π_new_1d = π_new.flatten() if π_new.ndim > 1 else π_new
        π_1d = π.flatten() if π.ndim > 1 else π

        # We require a finite observation alphabet Y_n and associated cells
        if not hasattr(self, "Y_n") or self.Y_n is None or self.Y_bounds is None:
            raise ValueError(
                "Localization η_n requires finite observation quantization (Y_n, Y_bounds) "
                "from the underlying POMDP. Ensure obs_quantization_level is configured."
            )

        if self.Q_n is None:
            raise ValueError("Q_n must be precomputed. Ensure observation quantization is configured.")

        # 1) Compute predicted next-state belief over X_n using precomputed T_mat
        Tn_mat = self.T_mat[:, :, self.AQ.get_quantized_index(u)]  # (m_n, m_n)
        predicted = Tn_mat @ π_1d  # (m_n,)
        log_predicted = np.log(np.maximum(predicted, 1e-300))

        # 2) Compute H for ALL observations at once: H_y = sum(Q_n * predicted) over poses
        m_y = int(self.Y_n.shape[0])
        # Q_n shape: (m_y, m_n), predicted shape: (m_n,)
        log_Q_n = np.log(np.maximum(self.Q_n, 1e-300))  # (m_y, m_n)
        log_numerator = log_Q_n + log_predicted[np.newaxis, :]  # (m_y, m_n)

        # H_y = sum(Q_n * predicted) over poses - compute in log space for stability
        log_H_y = logsumexp(log_numerator, axis=1)  # (m_y,)
        H_y = np.exp(log_H_y)  # (m_y,)

        # Normalize H_y
        total_H = np.sum(H_y)
        if total_H > 0:
            H_y = H_y / total_H

        # 3) Compute F for ALL observations directly from Q_n and predicted (reusing H_y computation!)
        # F = (Q_n * predicted) / H_y, so log_F = log_numerator - log_H_y
        log_F = log_numerator - log_H_y[:, np.newaxis]  # (m_y, m_n)
        π_all_batch = np.exp(log_F)  # (m_y, m_n)

        # 4) Vectorized distance computation
        π_new_broadcast = np.broadcast_to(π_new_1d[np.newaxis, :], (m_y, len(π_new_1d)))  # (m_y, m_n)
        π_diff = π_all_batch - π_new_broadcast  # (m_y, m_n)
        distances = np.linalg.norm(π_diff, axis=1)  # (m_y,)

        # 5) Vectorized matching
        threshold = 1e-3
        matches = (distances < threshold) & (H_y > 0.0)  # (m_y,)

        # 6) Sum H_y for matching observations
        prob = float(np.sum(H_y[matches]))

        return prob


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
        measurement_model: LIDAR | RangeBearingSensor,
        obstacles: list[Obstacle],
        _map: BaseMap,
        sigma_w: float = 0.01,
        sigma_v: float = 0.01,
        obs_n: int | None = None,
        action_n: int | None = None
    ):
        # Initialize POMDP parent first
        Mapping_POMDP.__init__(self, motion_model, measurement_model, obstacles, _map, sigma_w, sigma_v)

        # Initialize common BeliefMDP_n components (quantizers, T_mat)
        # If obs_n is None, use n (backward compatibility)
        BaseBeliefMDP_n._init_common(self, n, motion_model, _map, obs_n=obs_n if obs_n is not None else n, action_n=action_n)

        # Load or generate cached space of maps for vectorized operations
        if self.map_H * self.map_W <= 16:
            self.all_maps_3d, self.map_bits_array = self._load_or_generate_all_maps()
        else:
            self.all_maps_3d = None
            self.map_bits_array = None

    def _get_problem_type(self) -> str:
        return 'mapping'

    def _get_map_distance_matrix(self) -> np.ndarray:
        if getattr(self, "_map_distance_matrix", None) is None:
            if self.all_maps_3d is None:
                raise ValueError("all_maps_3d must be precomputed for Wasserstein distance reward")

            maps = self.all_maps_3d
            len_M = maps.shape[0]

            if maps.ndim == 2:
                # Landmark maps: vectorized Lp distances between indicator vectors
                diff = maps[:, np.newaxis, :] - maps[np.newaxis, :, :]
                dist = np.linalg.norm(diff, axis=2, ord=self.map_distance_p)
            else:
                # Occupancy grids or other map types: fall back to pairwise d_M
                dist = np.zeros((len_M, len_M), dtype=float)
                for i in range(len_M):
                    for j in range(i + 1, len_M):
                        d_ij = self.map.d_M(maps[i], maps[j], p=self.map_distance_p)
                        dist[i, j] = d_ij
                        dist[j, i] = d_ij

            self._map_distance_matrix = dist

        return self._map_distance_matrix

    def r_wasserstein_distance(self, Π_batch: np.ndarray) -> np.ndarray:
        """
        Wasserstein distance reward over map beliefs:
        c(b) = E_{m,m'~b}[d_M(m', m)] = b^T D b
        """
        Π_batch = np.atleast_2d(Π_batch)
        D = self._get_map_distance_matrix()  # (len_M, len_M)
        return np.einsum("bi,ij,bj->b", Π_batch, D, Π_batch)

    def _get_Q_n_shape(self) -> tuple:
        # For mapping, Q_n is (m_y, m_n, len_M) to support all quantized poses
        return (self.SQ.m_n, self.len_M)

    def _Q_y_star(self):
        """Compute y_star for Mapping: (m_n, len_M, B) for all quantized poses."""
        if self.all_maps_3d is None:
            raise ValueError("all_maps_3d must be precomputed for Mapping Q_n")
        X_in = self.SQ.X_n if not isinstance(self.sensor, LIDAR) else (
            self.SQ.X_n[:, :2] if self.SQ.X_n.shape[1] > 2 else self.SQ.X_n
        )
        y_star_result = self.ray_casting_batched(X_in, self.all_maps_3d)
        # Convert CuPy array to NumPy if needed
        y_star = y_star_result.get() if hasattr(y_star_result, 'get') else _numpy.asarray(y_star_result)
        return y_star

    ##########################################################
    ### 1. Filter Update and Observation Functions ######
    def F(self, π, x_next, y):
        """
        Filter update for a single observation.

        Wrapper around F_batch_log for convenience.

        Args:
            π: current belief (len_M,)
            x_next: known next pose (state_dim,) or (2,)
            y: observation (B,) or (1, B)
        Returns:
            π_new: updated belief (len_M,) or list containing it
        """
        # Ensure y is 2D
        y = np.asarray(y)
        if y.ndim == 1:
            y = y[np.newaxis, :]  # (1, B)

        # Use F_batch_log and return first result
        π_new_batch = self.F_batch_log(π, x_next, y)  # (1, len_M)
        return [π_new_batch[0]]  # Return as list for compatibility with test code

    def F_batch_log(self, π, x_next, Y_batch):
        """
        Batched filter update in log-space for numerical stability.

        Reuses H computation: F = (Q * π) / H, where H = sum(Q * π).

        Args:
            π: current belief (len_M,)
            x_next: known next pose (state_dim,) or (2,)
            Y_batch: observations (K, B)
        Returns:
            π_new_batch: updated beliefs (K, len_M)
        """
        if self.all_maps_3d is None:
            raise ValueError(
                "F_batch_log requires all_maps_3d to be precomputed. "
                f"Map space too large (H*W={self.map_H * self.map_W} > 16)."
            )

        if self.Q_n is None:
            raise ValueError("Q_n must be precomputed. Ensure observation quantization is configured.")
        self.known_pose = x_next
        x_next_idx = self.SQ.get_quantized_index(x_next)
        log_π = np.log(np.maximum(π, 1e-300))

        # Find observation indices for Y_batch
        Y_batch = self._prepare_observation_batch(Y_batch)
        obs_indices = self._find_observation_indices(Y_batch)

        # Q_n shape: (m_y, m_n, len_M), slice for current pose: (m_y, len_M)
        Q_batch = self.Q_n[:, x_next_idx, :][obs_indices]  # (K, len_M)
        log_Q_batch = np.log(np.maximum(Q_batch, 1e-300))

        # Compute numerator: Q * π (in log space)
        log_numerator = log_Q_batch + np.broadcast_to(log_π[np.newaxis, :], (Y_batch.shape[0], len(log_π)))

        # Compute H (normalization constant) = sum(Q * π) over maps
        log_H = logsumexp(log_numerator, axis=1, keepdims=True)  # (K, 1)

        # F = numerator / H (in log space: log_F = log_numerator - log_H)
        log_F = log_numerator - log_H
        return np.exp(log_F)

    def c_tilde_n(self, Π_batch: np.ndarray, X_current_batch: np.ndarray, U_batch: np.ndarray) -> np.ndarray:
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


    def η_n(self, π_new: np.ndarray, π: np.ndarray, x_current: np.ndarray, u: np.ndarray, x_next: np.ndarray) -> float:
        """
        Compute joint transition probability η_n(π', x_next | b_t, u) using finite observation quantization.

        Implements the joint probability:
        η(π', x_next | b_t, u) = T(x_next | x_t, u_t) * ∑_{y∈Y_n} 𝟙_{F(π_t, x_next, y) ≈ π'} · H({y} | π_t, x_next)

        where b_t = (x_t, π_t) is the augmented state, and x_next is given as input.

        Args:
            π_new: target belief (len_M,)
            π: current belief (len_M,)
            x_current: current state (state_dim,) - part of augmented state b_t = (x_t, π_t)
            u: action (2,)
            x_next: next state (state_dim,) - given as input, not sampled
        Returns:
            float: joint probability of (π', x_next) from augmented state (x_t, π_t) under action u
        """
        if self.Q_n is None:
            raise ValueError("Q_n must be precomputed. Ensure observation quantization is configured.")

        # Ensure beliefs are 1D
        π_new_1d = π_new.flatten() if π_new.ndim > 1 else π_new
        π_1d = π.flatten() if π.ndim > 1 else π

        # Get quantized indices
        x_current_idx = self.SQ.get_quantized_index(x_current)
        x_next_idx = self.SQ.get_quantized_index(x_next)
        u_idx = self.AQ.get_quantized_index(u)

        # Step 1: Get T(x_next | x_current, u) from T_mat
        Tn_mat = self.T_mat[:, :, u_idx]  # (m_n, m_n)
        T_x_next_given_x_current_u = float(Tn_mat[x_next_idx, x_current_idx])

        # If transition probability is zero, return 0
        if T_x_next_given_x_current_u < 1e-300:
            return 0.0

        # Step 2: Compute H for ALL observations at once: H_y = sum(Q_slice * π) over maps
        m_y = int(self.Y_n.shape[0])
        # Q_n shape: (m_y, m_n, len_M), slice for x_next: (m_y, len_M)
        Q_slice = self.Q_n[:, x_next_idx, :]  # (m_y, len_M)
        log_Q_slice = np.log(np.maximum(Q_slice, 1e-300))  # (m_y, len_M)
        log_π = np.log(np.maximum(π_1d, 1e-300))  # (len_M,)
        log_numerator = log_Q_slice + log_π[np.newaxis, :]  # (m_y, len_M)

        # H_y = sum(Q_slice * π) over maps - compute in log space for stability
        log_H_y = logsumexp(log_numerator, axis=1)  # (m_y,)
        H_y = np.exp(log_H_y)  # (m_y,)

        # Normalize H_y
        total_H = np.sum(H_y)
        if total_H > 0:
            H_y = H_y / total_H

        # Step 3: Compute F for ALL observations directly from Q_slice and π (reusing H_y computation!)
        # F = (Q_slice * π) / H_y, so log_F = log_numerator - log_H_y
        log_F = log_numerator - log_H_y[:, np.newaxis]  # (m_y, len_M)
        π_all_batch = np.exp(log_F)  # (m_y, len_M)

        # Step 4: Vectorized distance computation
        π_new_broadcast = np.broadcast_to(π_new_1d[np.newaxis, :], (m_y, len(π_new_1d)))  # (m_y, len_M)
        π_diff = π_all_batch - π_new_broadcast  # (m_y, len_M)
        distances = np.linalg.norm(π_diff, axis=1)  # (m_y,)

        # Step 5: Vectorized matching
        threshold = 1e-3
        matches = (distances < threshold) & (H_y > 0.0)  # (m_y,)

        # Step 6: Sum H_y for matching observations
        prob = float(np.sum(H_y[matches]))

        # Step 7: Return joint probability = T(x_next | x_current, u) * P(π' ≈ π_new | x_next)
        return T_x_next_given_x_current_u * prob


__all__ = ['BaseBeliefMDP_n', 'BeliefMDP_n_SLAM', 'BeliefMDP_n_Localization', 'BeliefMDP_n_Mapping']
