# import cupy as np
import hashlib
from pathlib import Path
from typing import Literal
from .belief_mdp_n import BeliefMDP_n_SLAM, BeliefMDP_n_Localization, BeliefMDP_n_Mapping
from ..classes.mapping import BaseMap, LidarGridMapVec
from ..classes.model import LIDAR, RangeBearingSensor, SingleIntegratorModel, DoubleIntegratorModel
from ..classes.obstacle import Obstacle
from ..classes.quantizer import BeliefQuantizer
from ..utils.array_backend import np, random, is_cupy
import numpy as _numpy  # For file I/O only
try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

# Import sparse matrix support
if is_cupy:
    try:
        from cupyx.scipy.sparse import csr_matrix, coo_matrix
        _sparse_available = True
    except ImportError:
        from scipy.sparse import csr_matrix, coo_matrix
        _sparse_available = True
else:
    from scipy.sparse import csr_matrix, coo_matrix
    _sparse_available = True

class BaseBeliefMDP_n_M:
    """Base class for BeliefMDP_n_M classes with shared caching logic."""

    _class_skip_p_n_M_computation = False

    @property
    def problem_type(self) -> str:
        """Problem type: 'slam', 'localization', or 'mapping'."""
        raise NotImplementedError

    @property
    def N_n(self) -> int:
        """Belief space dimension."""
        raise NotImplementedError

    def _get_base_metadata(self) -> dict:
        """Get base metadata common to all problem types."""
        state_bounds = self.state_bounds.astype(float)
        state_bounds_save = state_bounds.get() if hasattr(state_bounds, 'get') else state_bounds
        max_val = self.motion_model.max_v if hasattr(self.motion_model, 'max_v') else self.motion_model.max_a

        obs_n = getattr(self, 'obs_n', None)
        action_n = getattr(self.AQ, 'n', None) if hasattr(self, 'AQ') else None
        exploration_type = getattr(self, 'exploration_type', None)

        out = {
            'M': self.M,
            'n': self.n,
            'n_map': getattr(self, 'n_map', None),
            'state_bounds': state_bounds_save,
            'max_val': max_val,
            'dt': float(self.motion_model.dt),
            'map_shape': _numpy.array([self.map_H, self.map_W]),
            'sigma_w': float(self.σ_w),
            'sigma_v': float(self.σ_v),
            'obs_noise_diag': (
                _numpy.diag(self.cov_y.get() if hasattr(self.cov_y, 'get') else _numpy.asarray(self.cov_y))
                if getattr(self, 'cov_y', None) is not None else _numpy.array([], dtype=float)
            ),
            'sensor_sigma_r': float(self.sensor.sigma_r) if isinstance(self.sensor, RangeBearingSensor) else None,
            'sensor_sigma_phi': float(self.sensor.sigma_phi) if isinstance(self.sensor, RangeBearingSensor) else None,
            'sensor_epsilon': float(self.sensor.epsilon) if isinstance(self.sensor, RangeBearingSensor) else None,
            'sensor_r_max': float(self.sensor.r_max) if isinstance(self.sensor, RangeBearingSensor) else None,
            'sensor_r0': float(self.sensor.r0) if isinstance(self.sensor, RangeBearingSensor) and hasattr(self.sensor, 'r0') else None,
            'sensor_r1': float(self.sensor.r1) if isinstance(self.sensor, RangeBearingSensor) and hasattr(self.sensor, 'r1') else None,
            'm_n': self.SQ.m_n,
            'n_u': self.AQ.n_u,
            'cardinality': self.BQ.cardinality,
            'N_n': self.N_n,
            'model': self.motion_model.__class__.__name__,
            'state_dim': self.state_dim,
            'problem_type': self.problem_type,
            'obs_n': obs_n,
            'action_n': action_n,
            'exploration_type': exploration_type,
        }
        if hasattr(self.map, 'get_cache_fingerprint'):
            out['map_cache_fingerprint'] = self.map.get_cache_fingerprint()
        return out

    def _get_cache_path(self, cache_name: str, kernel_version: str = None) -> Path:
        """Generate cache file path with metadata hash."""
        cache_dir = Path(__file__).parent.parent.parent / f"cache/{cache_name}"
        cache_dir.mkdir(parents=True, exist_ok=True)

        metadata = self._get_base_metadata()
        if kernel_version:
            metadata['kernel_version'] = kernel_version

        state_bounds_flat = self.state_bounds.flatten().tolist()
        metadata['state_bounds'] = state_bounds_flat

        metadata_str = str(sorted(metadata.items()))
        metadata_hash = hashlib.md5(metadata_str.encode()).hexdigest()[:8]

        max_val = self.motion_model.max_v if hasattr(self.motion_model, 'max_v') else self.motion_model.max_a
        # Extract base name from cache_name (in case it includes path separators)
        cache_base_name = Path(cache_name).name if '/' in cache_name else cache_name
        n_map = getattr(self, 'n_map', None)
        nmap_part = f"_nmap{n_map}" if n_map is not None else ""
        filename = f"{cache_base_name}_M{self.M}_n{self.n}{nmap_part}_map{self.map_H}x{self.map_W}_max{max_val}_{metadata_hash}.npz"

        return cache_dir / filename

    def _verify_metadata(self, data: dict, expected_metadata: dict) -> bool:
        """Verify that cached metadata matches current configuration."""
        for key, expected_value in expected_metadata.items():
            if key not in data:
                return False
            data_value = data[key]
            if key == 'map_cache_fingerprint':
                data_val = tuple(data_value.tolist()) if hasattr(data_value, 'tolist') else tuple(data_value)
                if data_val != expected_value:
                    return False
                continue
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
        return True

    def _save_sparse_p_n_M(self, cache_path: Path, p_n_M: list, kernel_version: str) -> None:
        """Save sparse p_n_M (list of CSR matrices) to cache."""
        from scipy.sparse import csr_matrix as scipy_csr

        metadata = self._get_base_metadata()
        metadata['kernel_version'] = kernel_version

        save_dict = metadata.copy()
        for k, sparse_mat in enumerate(p_n_M):
            try:
                cupy_data = sparse_mat.data.get() if hasattr(sparse_mat.data, 'get') else sparse_mat.data
                cupy_indices = sparse_mat.indices.get() if hasattr(sparse_mat.indices, 'get') else sparse_mat.indices
                cupy_indptr = sparse_mat.indptr.get() if hasattr(sparse_mat.indptr, 'get') else sparse_mat.indptr
                sparse_mat_np = scipy_csr((cupy_data, cupy_indices, cupy_indptr), shape=sparse_mat.shape)
            except Exception:
                dense = sparse_mat.toarray()
                if hasattr(dense, 'get'):
                    dense = dense.get()
                sparse_mat_np = scipy_csr(dense)

            save_dict[f'p_n_M_{k}_data'] = sparse_mat_np.data.astype(_numpy.float16)
            save_dict[f'p_n_M_{k}_indices'] = sparse_mat_np.indices.astype(_numpy.int32)
            save_dict[f'p_n_M_{k}_indptr'] = sparse_mat_np.indptr.astype(_numpy.int32)
            save_dict[f'p_n_M_{k}_shape'] = _numpy.array(sparse_mat_np.shape, dtype=_numpy.int32)

        save_dict['sparse_format'] = True
        save_dict['n_actions'] = len(p_n_M)
        _numpy.savez_compressed(cache_path, **save_dict)

    def _load_sparse_p_n_M(self, cache_path: Path, kernel_version: str) -> bool:
        """Load sparse p_n_M from cache."""
        if not cache_path.exists():
            return False

        try:
            data = _numpy.load(str(cache_path))

            if data.get('kernel_version') != kernel_version:
                return False

            if not data.get('sparse_format', False):
                return False

            state_bounds = self.state_bounds.astype(float)
            state_bounds_np = state_bounds.get() if hasattr(state_bounds, 'get') else state_bounds

            expected_metadata = self._get_base_metadata()
            expected_metadata['state_bounds'] = state_bounds_np
            expected_metadata['map_shape'] = _numpy.array([self.map_H, self.map_W])

            if not self._verify_metadata(data, expected_metadata):
                return False

            n_actions = int(data.get('n_actions', self.AQ.n_u))
            p_n_M_sparse = []

            from scipy.sparse import csr_matrix as scipy_csr
            # CuPy sparse only supports float32/float64; use float32 in memory when CuPy
            load_dtype = _numpy.float32 if is_cupy else _numpy.float16
            for k in range(n_actions):
                data_key = f'p_n_M_{k}_data'
                if data_key not in data:
                    return False

                mat_data = data[data_key].astype(load_dtype)
                mat_indices = data[f'p_n_M_{k}_indices'].astype(_numpy.int32)
                mat_indptr = data[f'p_n_M_{k}_indptr'].astype(_numpy.int32)
                mat_shape = tuple(data[f'p_n_M_{k}_shape'])

                scipy_mat = scipy_csr((mat_data, mat_indices, mat_indptr), shape=mat_shape)
                p_n_M_sparse.append(csr_matrix(scipy_mat) if is_cupy else scipy_mat)

            self.p_n_M = p_n_M_sparse

            for sparse_mat in p_n_M_sparse:
                if sparse_mat.shape != (self.BQ.cardinality, self.BQ.cardinality):
                    return False

            return True
        except Exception as e:
            print(f"Error loading p_n_M cache: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _save_dense_c_n_M(self, cache_path: Path, c_n_M: np.ndarray, kernel_version: str) -> None:
        """Save dense c_n_M to cache."""
        metadata = self._get_base_metadata()
        metadata['kernel_version'] = kernel_version

        if hasattr(c_n_M, 'get'):
            c_n_M_np = c_n_M.get()
        else:
            c_n_M_np = c_n_M

        save_dict = metadata.copy()
        save_dict['c_n_M'] = c_n_M_np.astype(_numpy.float32)
        _numpy.savez_compressed(cache_path, **save_dict)

    def _load_dense_c_n_M(
        self,
        cache_path: Path,
        kernel_version: str,
        expected_shape: tuple,
        target_attr: str = 'c_n_M',
        validate_finite: bool = True,
    ) -> bool:
        """Load dense c_n_M from cache."""
        if not cache_path.exists():
            return False

        try:
            data = _numpy.load(str(cache_path))

            if data.get('kernel_version') != kernel_version:
                return False

            state_bounds = self.state_bounds.astype(float)
            state_bounds_np = state_bounds.get() if hasattr(state_bounds, 'get') else state_bounds

            expected_metadata = self._get_base_metadata()
            expected_metadata['state_bounds'] = state_bounds_np
            expected_metadata['map_shape'] = _numpy.array([self.map_H, self.map_W])

            if not self._verify_metadata(data, expected_metadata):
                return False

            c_n_M = data['c_n_M'].astype(_numpy.float32)
            if c_n_M.shape != expected_shape:
                return False
            if validate_finite and (_numpy.any(_numpy.isnan(c_n_M)) or _numpy.any(_numpy.isinf(c_n_M))):
                raise ValueError(
                    f"{target_attr} cache contains NaN or Inf. Delete cache and recompute."
                )

            if is_cupy:
                import cupy as cp
                c_n_M = cp.asarray(c_n_M)

            setattr(self, target_attr, c_n_M)
            return True
        except Exception as e:
            print(f"Error loading c_n_M cache: {e}")
            import traceback
            traceback.print_exc()
            return False

    def transition_kernel_report(
        self,
        action_indices: list[int] | None = None,
        row_sum_tolerance: float = 1e-3,
        verbose: bool = True,
    ) -> dict:
        """
        Diagnose sparse transition kernels p_n_M.

        Reports per-action and aggregate metrics:
        - row coverage (non-empty rows)
        - row-sum statistics (for stochasticity checks)
        - sparsity / outgoing branching
        - mapping-specific unique next-belief branching

        Args:
            action_indices: Optional subset of action indices to analyze.
            row_sum_tolerance: Allowed |row_sum - 1| for non-empty rows.
            verbose: Print human-readable report if True.
        Returns:
            Dictionary with aggregate and per-action diagnostics.
        """
        if not hasattr(self, 'p_n_M') or self.p_n_M is None:
            raise ValueError("p_n_M is not available. Compute or load transitions before diagnostics.")

        n_actions = len(self.p_n_M)
        if action_indices is None:
            action_indices = list(range(n_actions))
        else:
            action_indices = [int(k) for k in action_indices]
            for k in action_indices:
                if k < 0 or k >= n_actions:
                    raise ValueError(f"Invalid action index {k}; expected in [0, {n_actions-1}]")

        is_mapping = (self.problem_type == 'mapping')
        cardinality = int(self.BQ.cardinality)
        m_n = int(self.SQ.m_n) if hasattr(self, 'SQ') else 1
        expected_rows = cardinality * m_n if is_mapping else cardinality

        per_action = []
        for k in action_indices:
            mat = self.p_n_M[k]
            if mat.shape[0] != expected_rows:
                raise ValueError(
                    f"Action {k}: unexpected row count {mat.shape[0]} (expected {expected_rows})"
                )

            indptr = mat.indptr.get() if hasattr(mat.indptr, 'get') else _numpy.asarray(mat.indptr)
            indices = mat.indices.get() if hasattr(mat.indices, 'get') else _numpy.asarray(mat.indices)

            row_nnz = _numpy.diff(indptr).astype(_numpy.int64)
            nonempty_mask = row_nnz > 0
            nonempty_rows = int(_numpy.count_nonzero(nonempty_mask))
            total_rows = int(mat.shape[0])
            empty_rows = int(total_rows - nonempty_rows)
            row_coverage = float(nonempty_rows / total_rows) if total_rows > 0 else 0.0

            row_sums = mat.sum(axis=1)
            if hasattr(row_sums, 'get'):
                row_sums = row_sums.get()
            row_sums = _numpy.asarray(row_sums).reshape(-1).astype(_numpy.float64, copy=False)
            nz_row_sums = row_sums[nonempty_mask]
            if nz_row_sums.size > 0:
                row_sum_min = float(nz_row_sums.min())
                row_sum_max = float(nz_row_sums.max())
                row_sum_mean = float(nz_row_sums.mean())
                bad_row_count = int(_numpy.count_nonzero(_numpy.abs(nz_row_sums - 1.0) > row_sum_tolerance))
            else:
                row_sum_min = 0.0
                row_sum_max = 0.0
                row_sum_mean = 0.0
                bad_row_count = 0

            avg_nnz_all_rows = float(row_nnz.mean()) if total_rows > 0 else 0.0
            avg_nnz_nonempty_rows = float(row_nnz[nonempty_mask].mean()) if nonempty_rows > 0 else 0.0
            nnz = int(mat.nnz)
            sparsity = float(1.0 - nnz / (mat.shape[0] * mat.shape[1])) if mat.shape[0] * mat.shape[1] > 0 else 1.0

            # Mapping-specific diagnostic:
            # rows are (x_current, belief_i), cols are (x_next, belief_j).
            # Unique next beliefs for one row = number of unique (col % cardinality).
            unique_next_belief_mean = None
            unique_next_belief_min = None
            unique_next_belief_max = None
            if is_mapping and nonempty_rows > 0:
                nonempty_rows_idx = _numpy.where(nonempty_mask)[0]
                unique_counts = _numpy.zeros(nonempty_rows, dtype=_numpy.int32)
                for t, r in enumerate(nonempty_rows_idx):
                    start = int(indptr[r])
                    end = int(indptr[r + 1])
                    belief_cols = _numpy.mod(indices[start:end], cardinality)
                    unique_counts[t] = int(_numpy.unique(belief_cols).size)
                unique_next_belief_mean = float(unique_counts.mean())
                unique_next_belief_min = int(unique_counts.min())
                unique_next_belief_max = int(unique_counts.max())

            item = {
                'action_idx': int(k),
                'shape': (int(mat.shape[0]), int(mat.shape[1])),
                'nnz': nnz,
                'sparsity': sparsity,
                'row_coverage': row_coverage,
                'nonempty_rows': nonempty_rows,
                'empty_rows': empty_rows,
                'avg_nnz_all_rows': avg_nnz_all_rows,
                'avg_nnz_nonempty_rows': avg_nnz_nonempty_rows,
                'row_sum_min_nonempty': row_sum_min,
                'row_sum_max_nonempty': row_sum_max,
                'row_sum_mean_nonempty': row_sum_mean,
                'bad_row_count': bad_row_count,
                'row_sum_tolerance': float(row_sum_tolerance),
                'unique_next_belief_mean': unique_next_belief_mean,
                'unique_next_belief_min': unique_next_belief_min,
                'unique_next_belief_max': unique_next_belief_max,
            }
            per_action.append(item)

        total_rows_all = int(sum(x['shape'][0] for x in per_action))
        total_nonempty = int(sum(x['nonempty_rows'] for x in per_action))
        aggregate = {
            'n_actions_analyzed': int(len(per_action)),
            'row_coverage_mean': float(_numpy.mean([x['row_coverage'] for x in per_action])) if per_action else 0.0,
            'row_coverage_weighted': float(total_nonempty / total_rows_all) if total_rows_all > 0 else 0.0,
            'bad_rows_total': int(sum(x['bad_row_count'] for x in per_action)),
            'nnz_total': int(sum(x['nnz'] for x in per_action)),
        }

        report = {
            'problem_type': self.problem_type,
            'cardinality': cardinality,
            'm_n': m_n,
            'row_sum_tolerance': float(row_sum_tolerance),
            'aggregate': aggregate,
            'per_action': per_action,
        }

        if verbose:
            print("Transition kernel diagnostics")
            print(f"  problem_type={self.problem_type}, actions={len(per_action)}, cardinality={cardinality}, m_n={m_n}")
            print(
                f"  aggregate: coverage_mean={aggregate['row_coverage_mean']:.4f}, "
                f"bad_rows_total={aggregate['bad_rows_total']}, nnz_total={aggregate['nnz_total']}"
            )
            for x in per_action:
                base = (
                    f"  action {x['action_idx']}: coverage={x['row_coverage']:.4f}, "
                    f"nnz={x['nnz']}, avg_nnz_row={x['avg_nnz_all_rows']:.2f}, "
                    f"row_sum_nonempty[min/mean/max]=("
                    f"{x['row_sum_min_nonempty']:.6f}/{x['row_sum_mean_nonempty']:.6f}/{x['row_sum_max_nonempty']:.6f}), "
                    f"bad_rows={x['bad_row_count']}"
                )
                if is_mapping and x['unique_next_belief_mean'] is not None:
                    base += (
                        f", unique_next_beliefs[min/mean/max]="
                        f"({x['unique_next_belief_min']}/{x['unique_next_belief_mean']:.2f}/{x['unique_next_belief_max']})"
                    )
                print(base)

        return report

class BeliefMDP_n_M_SLAM(BeliefMDP_n_SLAM, BaseBeliefMDP_n_M):
    """Belief-MDP_n_M for SLAM: quantized belief space over pose and map."""

    @property
    def problem_type(self) -> str:
        return 'slam'

    @property
    def N_n(self) -> int:
        return self.SQ.m_n * self.len_M

    def __init__(
        self,
        M: int,
        β: float,
        n: int,
        motion_model: SingleIntegratorModel | DoubleIntegratorModel,
        measurement_model: LIDAR | RangeBearingSensor,
        obstacles: list[Obstacle],
        _map: BaseMap,
        sigma_w: float = 0.01,
        sigma_v: float = 0.01,
        cov_y: np.ndarray | None = None,
        exploration_type: Literal['information gain', 'wasserstein distance'] = 'information gain',
        obs_n: int | None = None,
        action_n: int | None = None
    ):
        super().__init__(
            n,
            motion_model,
            measurement_model,
            obstacles,
            _map,
            sigma_w,
            sigma_v,
            cov_y=cov_y,
            exploration_type=exploration_type,
            obs_n=obs_n,
            action_n=action_n
        )
        self.M = M
        self.β = β
        self.BQ = BeliefQuantizer(M, self.N_n)

        skip_computation = self._class_skip_p_n_M_computation or getattr(self, '_skip_p_n_M_computation', False)
        cache_path = self._get_p_n_M_cache_path()
        if self._load_p_n_M(cache_path):
            print(f"Loaded cached p_n_M from {cache_path}")
        elif skip_computation:
            print("⚠ Skipping p_n_M computation (testing mode)")
            self.p_n_M = None
        else:
            print("Computing p_n_M for the first time...")
            print(f"This may take a while: {self.AQ.n_u} actions × {self.BQ.cardinality}² belief transitions")
            # Use None to process all j targets at once for maximum GPU utilization
            # Use i_batch_size=3 to process multiple beliefs simultaneously
            j_batch_size = getattr(self, '_test_j_batch_size', None)
            i_batch_size = getattr(self, '_test_i_batch_size', 5)
            self.p_n_M = self._compute_p_n_M(j_batch_size=j_batch_size, i_batch_size=i_batch_size)
            self._save_p_n_M(cache_path, self.p_n_M)
            print(f"Saved p_n_M to {cache_path}")

        # Load or compute c_n_M (cost matrix) for all belief-action pairs
        c_n_M_cache_path = self._get_c_n_M_cache_path()
        if self._load_c_n_M(c_n_M_cache_path):
            print(f"Loaded cached c_n_M from {c_n_M_cache_path}")
        else:
            print("Computing c_n_M for the first time...")
            print(f"This will compute costs for {self.BQ.cardinality:,} beliefs × {self.AQ.n_u} actions")
            self._c_n_M_matrix = self._compute_c_n_M()
            self._save_c_n_M(c_n_M_cache_path, self._c_n_M_matrix)
            print(f"Saved c_n_M to {c_n_M_cache_path}")

    def _get_p_n_M_cache_path(self) -> Path:
        return BaseBeliefMDP_n_M._get_cache_path(self, 'p_n_M_slam', 'v2-sparse-float16')

    def _save_p_n_M(self, cache_path: Path, p_n_M: list) -> None:
        self._save_sparse_p_n_M(cache_path, p_n_M, 'v2-sparse-float16')

    def _load_p_n_M(self, cache_path: Path) -> bool:
        return self._load_sparse_p_n_M(cache_path, 'v2-sparse-float16')

    def η_n(self, j: int, i: int, u: np.ndarray) -> float:
        """
        Overloaded η_n that takes belief indices instead of belief vectors.

        Now delegates to the finite-observation η_n implementation in BeliefMDP_n_SLAM,
        using quantized beliefs from the codebook B_n_M.

        Args:
            j: Index of target belief in B_n_M (π_j^M)
            i: Index of current belief in B_n_M (π_i^M)
            u: Action (2,)
        Returns:
            float: Probability of transitioning from π_i^M to π_j^M under action u
        """
        # Access beliefs directly from cached codebook
        B_new_flat = self.BQ.Π_n_M[j]  # (N_n,)
        π_flat = self.BQ.Π_n_M[i]      # (N_n,)

        B_new_2d = self.unflatten_belief(B_new_flat)
        π_2d = self.unflatten_belief(π_flat)

        # Delegate to parent (finite-sum) implementation
        return super().η_n(B_new_2d, π_2d, u)

    def η_n_batch(self, j_list: list[int], i: int, u: np.ndarray) -> np.ndarray:
        """
        Fully vectorized batched version of η_n over multiple target belief indices.

        Computes probabilities for all target beliefs simultaneously by:
        1. Computing H_y and π_all_batch (all updated beliefs for all observations) once
        2. Vectorized distance computation between all target beliefs and all updated beliefs
        3. Summing H_y for matching observations for each target belief

        Args:
            j_list: List of target belief indices in B_n_M
            i: Index of current belief in B_n_M
            u: Action (2,)
        Returns:
            np.ndarray: Array of probabilities, shape (len(j_list),)
        """
        if len(j_list) == 0:
            return np.array([])

        # Access current belief from codebook
        π_flat = self.BQ.Π_n_M[i]  # (N_n,)
        π_2d = self.unflatten_belief(π_flat)  # (m_n, len_M)

        # Compute H_y and π_all_batch for ALL observations at once (reused for all targets)
        H_y, π_all_batch = self._compute_H_y_and_F(π_2d, u)
        # H_y: (m_y,), π_all_batch: (m_y, m_n, len_M)
        m_y = int(self.Y_n.shape[0])

        # Pushforward under quantizer: assign each filtered belief to nearest codebook point.
        π_all_flat = π_all_batch.reshape(m_y, -1)  # (m_y, N_n)
        codebook = self.BQ.Π_n_M  # (cardinality, N_n)
        codebook_norm_sq = np.sum(codebook * codebook, axis=1)[np.newaxis, :]  # (1, cardinality)
        sampled_norm_sq = np.sum(π_all_flat * π_all_flat, axis=1, keepdims=True)  # (m_y, 1)
        dots = π_all_flat @ codebook.T  # (m_y, cardinality)
        distances_sq = sampled_norm_sq + codebook_norm_sq - 2.0 * dots
        nearest_idx = np.argmin(distances_sq, axis=1).astype(np.int64)  # (m_y,)

        mass_all = np.bincount(
            nearest_idx,
            weights=H_y.astype(np.float32, copy=False),
            minlength=self.BQ.cardinality,
        ).astype(np.float32)  # (cardinality,)

        j_arr = np.array(j_list, dtype=np.int64)
        return mass_all[j_arr]

    def η_n_batch_i_batch(self, j_list: list[int], i_list: list[int], u: np.ndarray) -> np.ndarray:
        """
        Fully vectorized batched version of η_n over multiple source and target belief indices.

        Processes multiple source beliefs (i) and multiple target beliefs (j) simultaneously.

        Args:
            j_list: List of target belief indices in B_n_M
            i_list: List of source belief indices in B_n_M
            u: Action (2,)
        Returns:
            np.ndarray: Array of probabilities, shape (len(i_list), len(j_list))
                       where result[i_idx, j_idx] = η_n(π_j_list[j_idx] | π_i_list[i_idx], u)
        """
        if len(j_list) == 0 or len(i_list) == 0:
            return np.array([]).reshape(len(i_list), len(j_list))

        # Access all source beliefs from codebook
        n_sources = len(i_list)
        π_sources_flat = np.array([self.BQ.Π_n_M[i] for i in i_list])  # (n_sources, N_n)
        π_sources_2d = np.array([self.unflatten_belief(π_flat) for π_flat in π_sources_flat])  # (n_sources, m_n, len_M)

        # Use batched H_y and F computation for all source beliefs at once
        # Create a batch with single action repeated
        U_batch = u[np.newaxis, :]  # (1, 2)
        H_y_batch, π_all_batch = self._compute_H_y_and_F_batched(π_sources_2d, U_batch)
        # H_y_batch: (n_sources, 1, m_y) -> squeeze to (n_sources, m_y)
        # π_all_batch: (n_sources, 1, m_y, m_n, len_M) -> squeeze to (n_sources, m_y, m_n, len_M)
        H_y_batch = H_y_batch[:, 0, :]  # (n_sources, m_y)
        π_all_batch = π_all_batch[:, 0, :, :, :]  # (n_sources, m_y, m_n, len_M)

        m_y = int(self.Y_n.shape[0])

        # Pushforward under quantizer for each source belief.
        codebook = self.BQ.Π_n_M  # (cardinality, N_n)
        codebook_norm_sq = np.sum(codebook * codebook, axis=1)[np.newaxis, :]  # (1, cardinality)
        π_all_flat = π_all_batch.reshape(n_sources, m_y, -1)  # (n_sources, m_y, N_n)
        probs_all = np.zeros((n_sources, self.BQ.cardinality), dtype=np.float32)

        for src in range(n_sources):
            sampled = π_all_flat[src]  # (m_y, N_n)
            sampled_norm_sq = np.sum(sampled * sampled, axis=1, keepdims=True)  # (m_y, 1)
            dots = sampled @ codebook.T  # (m_y, cardinality)
            distances_sq = sampled_norm_sq + codebook_norm_sq - 2.0 * dots
            nearest_idx = np.argmin(distances_sq, axis=1).astype(np.int64)  # (m_y,)
            probs_all[src] = np.bincount(
                nearest_idx,
                weights=H_y_batch[src].astype(np.float32, copy=False),
                minlength=self.BQ.cardinality,
            ).astype(np.float32)

        j_arr = np.array(j_list, dtype=np.int64)
        return probs_all[:, j_arr]

    def _compute_p_n_M(self, j_batch_size: int = None, i_batch_size: int = 3,
                       show_progress: bool = True, threshold: float = 1e-8) -> list:
        """
        Compute the full transition probability matrix p_n^{(M)} over all belief-action pairs.
        Uses sparse matrices (CSR format) with float16 precision for memory efficiency.

        Args:
            j_batch_size: Number of target beliefs (j) to process in parallel.
                         If None, processes all targets at once (fastest, requires more GPU memory).
                         Default: None (process all at once)
            i_batch_size: Number of source beliefs (i) to process in parallel.
                         Default: 3 (processes 3 beliefs simultaneously)
            show_progress: Whether to show progress bars
            threshold: Minimum probability threshold (values below this are treated as zero)

        Returns:
            List of CSR sparse matrices, one per action. p_n_M[k][i, j] = p_n^{(M)}(π_j^M | π_i^M, u_k)
        """
        cardinality = self.BQ.cardinality
        n_u = self.AQ.n_u

        # Auto-determine optimal batch size if not provided
        if j_batch_size is None:
            # Process all j targets at once for maximum GPU utilization
            j_batch_size = cardinality
            if show_progress:
                tqdm.write(f"  Auto-selected j_batch_size={j_batch_size} (processing all targets at once)")

        # Store as list of sparse matrices (one per action)
        p_n_M_sparse = []

        for k in range(n_u):
            u = self.AQ.U[k]
            if show_progress:
                tqdm.write(f"Computing p_n_M for action {k+1}/{n_u}...")
                tqdm.write(f"  This will compute transitions for {cardinality:,} belief states")
                tqdm.write(f"  Processing {cardinality:,} target beliefs per belief state")
                tqdm.write(
                    f"  Using i_batch_size={i_batch_size} (processing {i_batch_size} source beliefs simultaneously)")
                if j_batch_size < cardinality:
                    tqdm.write(
                        f"  Using j_batch_size={j_batch_size} ({(cardinality + j_batch_size - 1) // j_batch_size} batches per belief)")

            # Build COO matrix incrementally (row, col, data)
            # Use lists to accumulate, then convert to arrays once
            rows_list = []
            cols_list = []
            data_list = []

            # Test: Only process first belief state to verify it works
            test_mode = False
            if test_mode:
                tqdm.write("  TEST MODE: Only processing first belief state")
                cardinality = 1

            # Process belief states with batched η_n for efficiency
            # Batch multiple j targets together to leverage GPU parallelism

            # Progress bar for beliefs (outer loop)
            pbar = tqdm(range(cardinality),
                        desc=f"Action {k+1}/{n_u}",
                        disable=not show_progress,
                        leave=True,
                        mininterval=0.5)

            # Pre-allocate indices lists once
            j_indices = list(range(cardinality))
            i_indices = list(range(cardinality))
            n_j_batches = (cardinality + j_batch_size - 1) // j_batch_size
            n_i_batches = (cardinality + i_batch_size - 1) // i_batch_size

            # Process source beliefs (i) in batches
            for i_batch_idx in range(n_i_batches):
                start_i = i_batch_idx * i_batch_size
                end_i = min(start_i + i_batch_size, cardinality)
                i_batch = i_indices[start_i:end_i]

                # Initialize row data storage for each source belief in this batch
                batch_row_data = {i: [] for i in i_batch}
                batch_row_cols = {i: [] for i in i_batch}

                # Process j targets in batches (or all at once if j_batch_size >= cardinality)
                for j_batch_idx in range(n_j_batches):
                    # Get batch of j indices
                    start_j = j_batch_idx * j_batch_size
                    end_j = min(start_j + j_batch_size, cardinality)
                    j_batch = j_indices[start_j:end_j]

                    # Compute η_n for all (i, j) pairs simultaneously
                    # probs shape: (len(i_batch), len(j_batch))
                    probs = self.η_n_batch_i_batch(j_batch, i_batch, u)

                    # Extract probabilities (convert from GPU if needed for threshold check)
                    if hasattr(probs, 'get'):
                        probs_cpu = probs.get()
                    else:
                        probs_cpu = probs

                    # Store non-zero probabilities for each source belief in the batch
                    for i_idx_in_batch, i in enumerate(i_batch):
                        for j_idx_in_batch, j in enumerate(j_batch):
                            prob = probs_cpu[i_idx_in_batch, j_idx_in_batch]
                            if prob > threshold:
                                batch_row_data[i].append(float(prob))
                                batch_row_cols[i].append(j)

                # Normalize and store rows for all source beliefs in this batch
                for i in i_batch:
                    row_data = batch_row_data[i]
                    row_cols = batch_row_cols[i]

                    # Normalize row to ensure probability measure
                    if row_data:
                        row_data_arr = np.array(row_data, dtype=np.float32)
                        row_sum = float(np.sum(row_data_arr))
                        if row_sum > 0:
                            row_data_arr = row_data_arr / row_sum

                            # Append to lists (keep float32 for sparse ops)
                            rows_list.extend([i] * len(row_cols))
                            cols_list.extend(row_cols)
                            data_list.extend(row_data_arr.tolist())

                # Update progress bar
                if show_progress:
                    pbar.update(len(i_batch))
                    pbar.set_postfix({'processed': f'{end_i}/{cardinality}'})

                # Only sync GPU periodically to reduce overhead
                if is_cupy and i_batch_idx % 10 == 0:
                    import cupy as cp
                    cp.cuda.Stream.null.synchronize()

                # Periodically clear GPU cache to prevent memory issues
                if is_cupy and i_batch_idx % 100 == 0 and i_batch_idx > 0:
                    import cupy as cp
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()

            # Convert to COO matrix, then CSR for efficient row operations
            if rows_list:
                # Convert to backend arrays (use int32 for indices, float32 for data)
                rows_arr = np.array(rows_list, dtype=np.int32)
                cols_arr = np.array(cols_list, dtype=np.int32)
                data_arr = np.array(data_list, dtype=np.float32)

                # Create COO matrix (cupyx supports float32/float64 only)
                coo_mat = coo_matrix((data_arr, (rows_arr, cols_arr)),
                                     shape=(cardinality, cardinality),
                                     dtype=np.float32)

                # Convert to CSR for efficient operations
                csr_mat = coo_mat.tocsr()
            else:
                # Empty matrix - create zero CSR matrix
                csr_mat = csr_matrix((cardinality, cardinality), dtype=np.float32)

            p_n_M_sparse.append(csr_mat)

            # Print statistics
            nnz = csr_mat.nnz
            sparsity = (1.0 - nnz / (cardinality * cardinality)) * 100
            if show_progress:
                tqdm.write(f"  Action {k+1}/{n_u}: {nnz:,} non-zeros ({sparsity:.2f}% sparse)")

        return p_n_M_sparse

    @property
    def c_n_M(self) -> np.ndarray:
        """
        Get the precomputed cost matrix.

        Returns:
            Cost matrix of shape (cardinality, n_u) where c_n_M[i, k] = ρ_n(π_i^M, u_k)
        """
        if not hasattr(self, '_c_n_M_matrix') or self._c_n_M_matrix is None:
            raise ValueError("c_n_M not precomputed. Ensure initialization completed successfully.")
        return self._c_n_M_matrix

    def get_cost(self, i: int, k: int) -> float:
        """
        Get cost for a specific belief-action pair.

        c_n_M(π_i^M, u_k) = ρ_n(π_i^M, u_k) = c_effort(u_k) + r_exploration(π_i^M, u_k)

        Args:
            i: Index of quantized belief in B_n_M
            k: Index of action in U_n
        Returns:
            Cost value (float)
        """
        return float(self.c_n_M[i, k])

    def _get_c_n_M_cache_path(self) -> Path:
        """Generate cache file path for c_n_M based on quantization parameters."""
        return BaseBeliefMDP_n_M._get_cache_path(self, 'SLAM/cost_slam', 'v1-dense-float32')

    def _compute_c_n_M(self, show_progress: bool = True) -> np.ndarray:
        """
        Precompute costs for all belief-action combinations using batched ρ_n.

        Returns:
            c_n_M: Array of shape (cardinality, n_u) where c_n_M[i, k] = ρ_n(π_i^M, u_k)
        """
        cardinality = self.BQ.cardinality
        n_u = self.AQ.n_u
        B_n_M = self.BQ.Π_n_M  # (cardinality, N_n)
        U_n = self.AQ.U  # (n_u, 2)

        if show_progress:
            print(f"  Computing costs for {cardinality:,} beliefs × {n_u} actions...")

        # Convert flattened beliefs to 2D format for ρ_n
        # B_n_M: (cardinality, N_n) where N_n = m_n * len_M
        # Need to reshape to (cardinality, m_n, len_M)
        B_n_M_2d = np.array([self.unflatten_belief(π_flat) for π_flat in B_n_M])  # (cardinality, m_n, len_M)

        # Use batched ρ_n to compute all costs at once
        # ρ_n expects (n_π, m_n, len_M) and (n_u, 2), returns (n_π, n_u)
        c_n_M = self.ρ_n(B_n_M_2d, U_n)  # (cardinality, n_u)

        c_n_M_np = c_n_M.get() if hasattr(c_n_M, 'get') else _numpy.asarray(c_n_M)
        if _numpy.any(_numpy.isnan(c_n_M_np)) or _numpy.any(_numpy.isinf(c_n_M_np)):
            raise ValueError(
                "c_n_M contains NaN or Inf after ρ_n computation. "
                "Check beliefs (codebook Π_n_M) and cost ρ_n upstream."
            )
        return c_n_M

    def _save_c_n_M(self, cache_path: Path, c_n_M: np.ndarray) -> None:
        """Save c_n_M to cache file with metadata."""
        self._save_dense_c_n_M(cache_path, c_n_M, 'v1-dense-float32')

    def _load_c_n_M(self, cache_path: Path) -> bool:
        """Load c_n_M from cache file if it exists and matches current parameters."""
        return self._load_dense_c_n_M(
            cache_path,
            kernel_version='v1-dense-float32',
            expected_shape=(self.BQ.cardinality, self.AQ.n_u),
            target_attr='_c_n_M_matrix',
            validate_finite=True,
        )


class BeliefMDP_n_M_Localization(BeliefMDP_n_Localization, BaseBeliefMDP_n_M):
    """Belief-MDP_n_M for Localization: quantized belief space over poses only."""

    @property
    def problem_type(self) -> str:
        return 'localization'

    @property
    def N_n(self) -> int:
        return self.SQ.m_n

    def __init__(self, M: int, β: float, n: int, motion_model: SingleIntegratorModel | DoubleIntegratorModel,
                 measurement_model: LIDAR | RangeBearingSensor, obstacles: list[Obstacle], _map: BaseMap,
                 sigma_w: float = 0.01, sigma_v: float = 0.01,
                 cov_y: np.ndarray | None = None,
                 obs_n: int | None = None, action_n: int | None = None):
        super().__init__(n, motion_model, measurement_model, obstacles, _map,
                         sigma_w, sigma_v, cov_y=cov_y, obs_n=obs_n, action_n=action_n)
        self.M = M
        self.β = β
        self.BQ = BeliefQuantizer(M, self.N_n)

        skip_computation = self._class_skip_p_n_M_computation or getattr(self, '_skip_p_n_M_computation', False)
        cache_path = self._get_p_n_M_cache_path()
        if self._load_p_n_M(cache_path):
            print(f"Loaded cached p_n_M from {cache_path}")
        elif skip_computation:
            print("⚠ Skipping p_n_M computation (testing mode)")
            self.p_n_M = None
        else:
            print("Computing p_n_M for the first time...")
            print(f"This may take a while: {self.AQ.n_u} actions × {self.BQ.cardinality}² belief transitions")
            j_batch_size = getattr(self, '_test_j_batch_size', None)
            self.p_n_M = self._compute_p_n_M(j_batch_size=j_batch_size)
            self._save_p_n_M(cache_path, self.p_n_M)
            print(f"Saved p_n_M to {cache_path}")

        c_n_M_cache_path = self._get_c_n_M_cache_path()
        if self._load_c_n_M(c_n_M_cache_path):
            print(f"Loaded cached c_n_M from {c_n_M_cache_path}")
        else:
            print("Computing c_n_M for the first time...")
            print(f"This will compute costs for {self.BQ.cardinality} beliefs × {self.AQ.n_u} actions")
            self.c_n_M = self._compute_c_n_M()
            self._save_c_n_M(c_n_M_cache_path, self.c_n_M)
            print(f"Saved c_n_M to {c_n_M_cache_path}")

    def _get_p_n_M_cache_path(self) -> Path:
        return BaseBeliefMDP_n_M._get_cache_path(self, 'p_n_M_localization', 'v3-sparse-float16')

    def _save_p_n_M(self, cache_path: Path, p_n_M: list) -> None:
        self._save_sparse_p_n_M(cache_path, p_n_M, 'v2-sparse-float16')

    def _load_p_n_M(self, cache_path: Path) -> bool:
        return self._load_sparse_p_n_M(cache_path, 'v2-sparse-float16')

    def _get_c_n_M_cache_path(self) -> Path:
        return BaseBeliefMDP_n_M._get_cache_path(self, 'c_n_M_localization', 'v3-dense-float32-info-gain')

    def _save_c_n_M(self, cache_path: Path, c_n_M: np.ndarray) -> None:
        self._save_dense_c_n_M(cache_path, c_n_M, 'v3-dense-float32-info-gain')

    def _load_c_n_M(self, cache_path: Path) -> bool:
        return self._load_dense_c_n_M(
            cache_path,
            kernel_version='v3-dense-float32-info-gain',
            expected_shape=(self.BQ.cardinality, self.AQ.n_u),
            target_attr='c_n_M',
            validate_finite=True,
        )

    def η_n(self, j: int, i: int, u: np.ndarray) -> float:
        """
        Overloaded η_n that takes belief indices instead of belief vectors.

        Now delegates to the finite-observation η_n implementation in BeliefMDP_n_Localization,
        using quantized beliefs from the codebook B_n_M.

        Args:
            j: Index of target belief in B_n_M (π_j^M)
            i: Index of current belief in B_n_M (π_i^M)
            u: Action (2,)
        Returns:
            float: Probability of transitioning from π_i^M to π_j^M under action u
        """
        # Access beliefs directly from cached codebook
        B_new_1d = self.BQ.Π_n_M[j]  # (m_n,)
        π_1d = self.BQ.Π_n_M[i]      # (m_n,)

        # Delegate to parent (finite-sum) implementation
        return super().η_n(B_new_1d, π_1d, u)

    def η_n_batch(self, j_list: list[int], i: int, u: np.ndarray) -> np.ndarray:
        """
        Fully vectorized batched version of η_n over multiple target belief indices for localization.

        Computes probabilities for all target beliefs simultaneously by:
        1. Computing H_y and π_all_batch (all updated beliefs for all observations) once
        2. Vectorized distance computation between all target beliefs and all updated beliefs
        3. Summing H_y for matching observations for each target belief

        Args:
            j_list: List of target belief indices in B_n_M
            i: Index of current belief in B_n_M
            u: Action (2,)
        Returns:
            np.ndarray: Array of probabilities, shape (len(j_list),)
        """
        if len(j_list) == 0:
            return np.array([])

        # Access current belief from codebook
        π_1d = self.BQ.Π_n_M[i]  # (m_n,)

        # For localization, we need to compute H_y and π_all_batch manually
        # (Localization doesn't have _compute_H_y_and_F, but we can replicate the logic)
        if self.Q_n is None:
            raise ValueError("Q_n must be precomputed. Ensure observation quantization is configured.")

        # 1) Predicted belief over poses using T_mat
        Tn_mat = self.T_mat[:, :, self.AQ.get_quantized_index(u)]  # (m_n, m_n)
        predicted = Tn_mat @ π_1d  # (m_n,)
        log_predicted = np.log(np.maximum(predicted, 1e-300))

        # 2) Compute H for ALL observations at once: H_y = sum(Q_n * predicted) over poses
        m_y = int(self.Y_n.shape[0])
        # Q_n shape: (m_y, m_n), predicted shape: (m_n,)
        from scipy.special import logsumexp
        if is_cupy:
            from cupyx.scipy.special import logsumexp
        log_Q_n = np.log(np.maximum(self.Q_n, 1e-300))  # (m_y, m_n)
        log_numerator = log_Q_n + log_predicted[np.newaxis, :]  # (m_y, m_n)

        # H_y = sum(Q_n * predicted) over poses - compute in log space for stability
        log_H_y = logsumexp(log_numerator, axis=1)  # (m_y,)
        H_y = np.exp(log_H_y)  # (m_y,)

        # Normalize H_y
        total_H = np.sum(H_y)
        if total_H > 0:
            H_y = H_y / total_H

        # 3) Compute F for ALL observations directly from Q_n and predicted
        log_F = log_numerator - log_H_y[:, np.newaxis]  # (m_y, m_n)
        π_all_batch = np.exp(log_F)  # (m_y, m_n)

        # Pushforward under quantizer: assign each filtered belief to nearest codebook point.
        codebook = self.BQ.Π_n_M  # (cardinality, m_n)
        codebook_norm_sq = np.sum(codebook * codebook, axis=1)[np.newaxis, :]  # (1, cardinality)
        sampled_norm_sq = np.sum(π_all_batch * π_all_batch, axis=1, keepdims=True)  # (m_y, 1)
        dots = π_all_batch @ codebook.T  # (m_y, cardinality)
        distances_sq = sampled_norm_sq + codebook_norm_sq - 2.0 * dots
        nearest_idx = np.argmin(distances_sq, axis=1).astype(np.int64)  # (m_y,)

        mass_all = np.bincount(
            nearest_idx,
            weights=H_y.astype(np.float32, copy=False),
            minlength=self.BQ.cardinality,
        ).astype(np.float32)  # (cardinality,)

        j_arr = np.array(j_list, dtype=np.int64)
        return mass_all[j_arr]

    def _compute_p_n_M(self, j_batch_size: int = 100,
                       show_progress: bool = True, threshold: float = 1e-8) -> list:
        """Compute the full transition probability matrix p_n^{(M)} over all belief-action pairs for localization."""
        cardinality = self.BQ.cardinality
        n_u = self.AQ.n_u

        p_n_M_sparse = []

        for k in range(n_u):
            u = self.AQ.U[k]
            if show_progress:
                tqdm.write(f"Computing p_n_M for action {k+1}/{n_u}...")
                tqdm.write(f"  This will compute transitions for {cardinality:,} belief states")

            rows_list = []
            cols_list = []
            data_list = []

            test_mode = False
            if test_mode:
                tqdm.write("  TEST MODE: Only processing first belief state")
                cardinality = 1

            # Progress bar for beliefs (outer loop)
            pbar = tqdm(range(cardinality),
                        desc=f"Action {k+1}/{n_u}",
                        disable=not show_progress,
                        leave=True,
                        mininterval=0.5)

            for i in pbar:
                row_data = []
                row_cols = []

                if is_cupy:
                    import cupy as cp
                    cp.cuda.Stream.null.synchronize()

                j_indices = list(range(cardinality))
                n_j_batches = (cardinality + j_batch_size - 1) // j_batch_size

                for j_batch_idx in range(n_j_batches):
                    start_j = j_batch_idx * j_batch_size
                    end_j = min(start_j + j_batch_size, cardinality)
                    j_batch = j_indices[start_j:end_j]

                    probs = self.η_n_batch(j_batch, i, u)

                    if hasattr(probs, 'get'):
                        probs_cpu = probs.get()
                    else:
                        probs_cpu = probs

                    for j_idx, prob in zip(j_batch, probs_cpu):
                        if prob > threshold:
                            row_data.append(float(prob))
                            row_cols.append(j_idx)

                # Update progress bar with current belief's statistics
                if show_progress:
                    pbar.set_postfix({'nonzeros': len(row_data)})

                if is_cupy:
                    import cupy as cp
                    cp.cuda.Stream.null.synchronize()

                if is_cupy and i % 100 == 0 and i > 0:
                    import cupy as cp
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()

                if row_data:
                    row_data_arr = np.array(row_data, dtype=np.float32)
                    row_sum = float(np.sum(row_data_arr))
                    if row_sum > 0:
                        row_data_arr = row_data_arr / row_sum

                    rows_list.extend([i] * len(row_cols))
                    cols_list.extend(row_cols)
                    data_list.extend(row_data_arr.tolist())

            if rows_list:
                rows_arr = np.array(rows_list, dtype=np.int32)
                cols_arr = np.array(cols_list, dtype=np.int32)
                data_arr = np.array(data_list, dtype=np.float32)

                coo_mat = coo_matrix((data_arr, (rows_arr, cols_arr)),
                                     shape=(cardinality, cardinality),
                                     dtype=np.float32)

                csr_mat = coo_mat.tocsr()
            else:
                csr_mat = csr_matrix((cardinality, cardinality), dtype=np.float32)

            p_n_M_sparse.append(csr_mat)

            nnz = csr_mat.nnz
            sparsity = (1.0 - nnz / (cardinality * cardinality)) * 100
            if show_progress:
                tqdm.write(f"  Action {k+1}/{n_u}: {nnz:,} non-zeros ({sparsity:.2f}% sparse)")

        return p_n_M_sparse

    def get_codebook(self) -> np.ndarray:
        """
        Get the quantized belief codebook.

        Returns:
            B_n_M: Array of shape (cardinality, m_n) containing all quantized beliefs
        """
        return self.BQ.Π_n_M

    def _compute_c_n_M(self, show_progress: bool = True) -> np.ndarray:
        """
        Precompute costs for all belief-action pairs.

        Returns:
            c_n_M: Array of shape (cardinality, n_u) where c_n_M[i, k] = c_n_M(π_i^M, u_k)
        """
        cardinality = self.BQ.cardinality
        n_u = self.AQ.n_u
        B_n = self.BQ.Π_n_M  # (cardinality, m_n)
        U_n = self.AQ.U  # (n_u, 2)

        if show_progress:
            print(f"  Computing costs for {cardinality:,} beliefs × {n_u} actions...")

        # Debug: ensure inputs are finite so we can pinpoint NaN source
        B_np = B_n.get() if hasattr(B_n, 'get') else _numpy.asarray(B_n)
        U_np = U_n.get() if hasattr(U_n, 'get') else _numpy.asarray(U_n)
        if _numpy.any(_numpy.isnan(B_np)) or _numpy.any(_numpy.isinf(B_np)):
            raise ValueError(
                "Belief codebook Π_n_M contains NaN or Inf. Regenerate or clear belief quantizer cache."
            )
        if _numpy.any(_numpy.isnan(U_np)) or _numpy.any(_numpy.isinf(U_np)):
            raise ValueError(
                "Action set U_n contains NaN or Inf. Check action quantizer."
            )

        # ρ_n(B, u) = r_exploration(B) + c_effort(u); compute components for clearer errors
        r_exploration_batch = np.atleast_1d(self.r_information_gain(B_n))  # (cardinality,)
        c_effort_batch = self.c_effort(U_n)  # (n_u,)
        r_np = r_exploration_batch.get() if hasattr(r_exploration_batch, 'get') else _numpy.asarray(r_exploration_batch)
        c_np = c_effort_batch.get() if hasattr(c_effort_batch, 'get') else _numpy.asarray(c_effort_batch)
        if _numpy.any(_numpy.isnan(r_np)) or _numpy.any(_numpy.isinf(r_np)):
            raise ValueError(
                "r_exploration (information gain H(B)) contains NaN or Inf. "
                "Check entropy H() and belief codebook Π_n_M."
            )
        if _numpy.any(_numpy.isnan(c_np)) or _numpy.any(_numpy.isinf(c_np)):
            raise ValueError(
                "c_effort(U) contains NaN or Inf. Check action quantizer U_n."
            )

        c_n_M = r_exploration_batch[:, np.newaxis] + c_effort_batch[np.newaxis, :]  # (cardinality, n_u)

        c_n_M_np = c_n_M.get() if hasattr(c_n_M, 'get') else _numpy.asarray(c_n_M)
        if _numpy.any(_numpy.isnan(c_n_M_np)) or _numpy.any(_numpy.isinf(c_n_M_np)):
            raise ValueError(
                "c_n_M contains NaN or Inf after ρ_n computation. "
                "Check beliefs (codebook Π_n_M) and cost ρ_n = r_exploration + c_effort upstream."
            )
        return c_n_M

class BeliefMDP_n_M_Mapping(BeliefMDP_n_Mapping, BaseBeliefMDP_n_M):
    """Belief-MDP_n_M for Mapping: quantized belief space over maps only."""

    @property
    def problem_type(self) -> str:
        return 'mapping'

    @property
    def N_n(self) -> int:
        return self.len_M

    def __init__(self, M: int, β: float, n: int, motion_model: SingleIntegratorModel | DoubleIntegratorModel,
                 measurement_model: LIDAR | RangeBearingSensor, obstacles: list[Obstacle], _map: BaseMap,
                 sigma_w: float = 0.01, sigma_v: float = 0.01,
                 cov_y: np.ndarray | None = None,
                 exploration_type: Literal['information gain', 'wasserstein distance'] = 'information gain',
                 obs_n: int | None = None, action_n: int | None = None,
                 j_batch_size: int = 136, i_batch_size: int = 10,
                 n_gpus: int = 1):
        super().__init__(
            n,
            motion_model,
            measurement_model,
            obstacles,
            _map,
            sigma_w,
            sigma_v,
            cov_y=cov_y,
            exploration_type=exploration_type,
            obs_n=obs_n,
            action_n=action_n
        )
        self.M = M
        self.β = β
        self.BQ = BeliefQuantizer(M, self.N_n)
        self.j_batch_size = j_batch_size if j_batch_size is not None else self.BQ.cardinality
        self.i_batch_size = i_batch_size if i_batch_size is not None else self.BQ.cardinality
        self.n_gpus = n_gpus

        skip_computation = self._class_skip_p_n_M_computation or getattr(self, '_skip_p_n_M_computation', False)
        self._initialize_p_n_M(skip_computation)
        self._initialize_c_n_M()

    def _initialize_p_n_M(self, skip_computation: bool) -> None:
        """Load or compute p_n_M transition cache."""
        cache_path = self._get_p_n_M_cache_path()
        checkpoint_path = self._get_p_n_M_checkpoint_path(cache_path)
        if self._load_p_n_M(cache_path):
            print(f"Loaded cached p_n_M from {cache_path}")
            if checkpoint_path.exists():
                checkpoint_path.unlink()
        elif skip_computation:
            print("⚠ Skipping p_n_M computation (testing mode)")
            self.p_n_M = None
        else:
            print("Computing p_n_M for the first time...")
            print(f"This will compute: {self.AQ.n_u} actions × {self.BQ.cardinality}² belief-to-belief transitions")
            print(f"  Note: Uses quantized-observation η_n_batch (no MC sampling)")
            self.p_n_M = self._compute_p_n_M(j_batch_size=self.j_batch_size, i_batch_size=self.i_batch_size,
                                              n_gpus=self.n_gpus, checkpoint_path=checkpoint_path)
            self._save_p_n_M(cache_path, self.p_n_M)
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            print(f"Saved p_n_M to {cache_path}")

    def _initialize_c_n_M(self) -> None:
        """Load or compute c_n_M cost cache."""
        c_n_M_cache_path = self._get_c_n_M_cache_path()
        if self._load_c_n_M(c_n_M_cache_path):
            print(f"Loaded cached c_n_M from {c_n_M_cache_path}")
        else:
            print("Computing c_n_M for the first time...")
            print(f"This will compute costs for {self.BQ.cardinality:,} beliefs × {self.AQ.n_u} actions")
            self.c_n_M = self._compute_c_n_M()
            self._save_c_n_M(c_n_M_cache_path, self.c_n_M)
            print(f"Saved c_n_M to {c_n_M_cache_path}")

    def _get_p_n_M_cache_path(self) -> Path:
        return BaseBeliefMDP_n_M._get_cache_path(self, 'MAP/p_n_M_mapping', 'v6-sparse-mapping-nearest-detection-taper')

    def _get_p_n_M_checkpoint_path(self, cache_path: Path | None = None) -> Path:
        """Path for resumable partial p_n_M checkpoints."""
        if cache_path is None:
            cache_path = self._get_p_n_M_cache_path()
        return cache_path.with_suffix('.checkpoint.npz')

    def _save_p_n_M_checkpoint(self, checkpoint_path: Path, partial_actions: list) -> None:
        """
        Save partial mapping p_n_M progress to a checkpoint .npz.

        Stores completed action sparse matrices so interrupted runs can resume.
        """
        from scipy.sparse import csr_matrix as scipy_csr

        metadata = self._get_base_metadata()
        metadata['kernel_version'] = 'v1-mapping-round-checkpoint'

        save_dict = metadata.copy()
        save_dict['n_actions_total'] = int(self.AQ.n_u)
        n_states = int(self.SQ.m_n * self.BQ.cardinality)
        save_dict['n_states'] = n_states

        completed = [k for k, mat in enumerate(partial_actions) if mat is not None]
        save_dict['completed_actions'] = _numpy.asarray(completed, dtype=_numpy.int32)

        for k in completed:
            sparse_mat = partial_actions[k]
            try:
                mat_data = sparse_mat.data.get() if hasattr(sparse_mat.data, 'get') else sparse_mat.data
                mat_indices = sparse_mat.indices.get() if hasattr(sparse_mat.indices, 'get') else sparse_mat.indices
                mat_indptr = sparse_mat.indptr.get() if hasattr(sparse_mat.indptr, 'get') else sparse_mat.indptr
                sparse_mat_np = scipy_csr((mat_data, mat_indices, mat_indptr), shape=sparse_mat.shape)
            except Exception:
                dense = sparse_mat.toarray()
                if hasattr(dense, 'get'):
                    dense = dense.get()
                sparse_mat_np = scipy_csr(dense)

            save_dict[f'p_n_M_{k}_data'] = sparse_mat_np.data.astype(_numpy.float32)
            save_dict[f'p_n_M_{k}_indices'] = sparse_mat_np.indices.astype(_numpy.int32)
            save_dict[f'p_n_M_{k}_indptr'] = sparse_mat_np.indptr.astype(_numpy.int32)
            save_dict[f'p_n_M_{k}_shape'] = _numpy.array(sparse_mat_np.shape, dtype=_numpy.int32)

        tmp_path = checkpoint_path.with_name(checkpoint_path.name + '.tmp.npz')
        _numpy.savez_compressed(tmp_path, **save_dict)
        tmp_path.replace(checkpoint_path)

    def _load_p_n_M_checkpoint(self, checkpoint_path: Path) -> list | None:
        """Load partial mapping p_n_M checkpoint if valid; otherwise return None."""
        if not checkpoint_path.exists():
            return None

        try:
            data = _numpy.load(str(checkpoint_path))
            if data.get('kernel_version') != 'v1-mapping-round-checkpoint':
                return None

            state_bounds_np = (self.state_bounds.get() if hasattr(self.state_bounds, 'get') else self.state_bounds).astype(float)
            expected_metadata = self._get_base_metadata()
            expected_metadata['state_bounds'] = state_bounds_np
            expected_metadata['map_shape'] = _numpy.array([self.map_H, self.map_W])
            if not self._verify_metadata(data, expected_metadata):
                return None

            n_actions_total = int(data.get('n_actions_total', -1))
            if n_actions_total != int(self.AQ.n_u):
                return None
            n_states = int(data.get('n_states', -1))
            expected_n_states = int(self.SQ.m_n * self.BQ.cardinality)
            if n_states != expected_n_states:
                return None

            completed = data.get('completed_actions')
            if completed is None:
                return None
            completed = [int(k) for k in _numpy.asarray(completed).tolist()]

            from scipy.sparse import csr_matrix as scipy_csr
            partial = [None] * self.AQ.n_u
            for k in completed:
                mat_data = data[f'p_n_M_{k}_data'].astype(_numpy.float32)
                mat_indices = data[f'p_n_M_{k}_indices'].astype(_numpy.int32)
                mat_indptr = data[f'p_n_M_{k}_indptr'].astype(_numpy.int32)
                mat_shape = tuple(data[f'p_n_M_{k}_shape'])
                if mat_shape != (expected_n_states, expected_n_states):
                    return None
                scipy_mat = scipy_csr((mat_data, mat_indices, mat_indptr), shape=mat_shape)
                partial[k] = csr_matrix(scipy_mat) if is_cupy else scipy_mat

            return partial
        except Exception:
            return None

    def _save_p_n_M(self, cache_path: Path, p_n_M: list) -> None:
        """Save sparse p_n_M (list of CSR matrices) to cache."""
        self._save_sparse_p_n_M(cache_path, p_n_M, 'v6-sparse-mapping-nearest-detection-taper')

    def _load_p_n_M(self, cache_path: Path) -> bool:
        """Load sparse p_n_M from cache (list of CSR, each shape (m_n*cardinality, m_n*cardinality))."""
        if not cache_path.exists():
            print(f"Cache file does not exist: {cache_path}")
            cache_dir = cache_path.parent
            if cache_dir.exists():
                max_val = self.motion_model.max_v if hasattr(self.motion_model, 'max_v') else self.motion_model.max_a
                n_map = getattr(self, 'n_map', None)
                nmap_part = f"_nmap{n_map}" if n_map is not None else ""
                pattern = f"p_n_M_mapping_M{self.M}_n{self.n}{nmap_part}_map{self.map_H}x{self.map_W}_max{max_val}_*.npz"
                for f in sorted(cache_dir.glob(pattern)):
                    print(f"  Trying: {f.name}")
                    if self._try_load_sparse_p_n_M_mapping(f):
                        print(f"  ✓ Loaded: {f.name}")
                        return True
            return False
        return self._try_load_sparse_p_n_M_mapping(cache_path)

    def _try_load_sparse_p_n_M_mapping(self, cache_path: Path) -> bool:
        """Load sparse p_n_M for mapping; expect shape (m_n*cardinality, m_n*cardinality) per action."""
        if not cache_path.exists():
            return False
        try:
            data = _numpy.load(str(cache_path))
            if data.get('kernel_version') != 'v6-sparse-mapping-nearest-detection-taper':
                return False
            if not data.get('sparse_format', False):
                return False
            state_bounds_np = (self.state_bounds.get() if hasattr(self.state_bounds, 'get') else self.state_bounds).astype(float)
            expected_metadata = self._get_base_metadata()
            expected_metadata['state_bounds'] = state_bounds_np
            expected_metadata['map_shape'] = _numpy.array([self.map_H, self.map_W])
            if not self._verify_metadata(data, expected_metadata):
                return False
            n_states = self.SQ.m_n * self.BQ.cardinality
            expected_shape = (n_states, n_states)
            n_actions = int(data.get('n_actions', self.AQ.n_u))
            from scipy.sparse import csr_matrix as scipy_csr
            load_dtype = _numpy.float32 if is_cupy else _numpy.float16
            p_n_M_sparse = []
            for k in range(n_actions):
                mat_data = data[f'p_n_M_{k}_data'].astype(load_dtype)
                mat_indices = data[f'p_n_M_{k}_indices'].astype(_numpy.int32)
                mat_indptr = data[f'p_n_M_{k}_indptr'].astype(_numpy.int32)
                mat_shape = tuple(data[f'p_n_M_{k}_shape'])
                if mat_shape != expected_shape:
                    return False
                scipy_mat = scipy_csr((mat_data, mat_indices, mat_indptr), shape=mat_shape)
                p_n_M_sparse.append(csr_matrix(scipy_mat) if is_cupy else scipy_mat)
            self.p_n_M = p_n_M_sparse
            return True
        except Exception as e:
            return False

    def _get_c_n_M_cache_path(self) -> Path:
        """Generate cache file path for c_n_M based on quantization parameters."""
        return BaseBeliefMDP_n_M._get_cache_path(self, 'MAP/cost_mapping', 'v2-dense-float32-mapping-cost')

    def _save_c_n_M(self, cache_path: Path, c_n_M: np.ndarray) -> None:
        self._save_dense_c_n_M(cache_path, c_n_M, 'v2-dense-float32-mapping-cost')

    def _load_c_n_M(self, cache_path: Path) -> bool:
        return self._load_dense_c_n_M(
            cache_path,
            kernel_version='v2-dense-float32-mapping-cost',
            expected_shape=(self.BQ.cardinality, self.AQ.n_u),
            target_attr='c_n_M',
            validate_finite=True,
        )

    def η_n(self, j: int, i: int, x_current: np.ndarray, u: np.ndarray, x_next: np.ndarray) -> float:
        """
        Overloaded η_n that takes belief indices instead of belief vectors for mapping.

        Now delegates to the finite-observation η_n implementation in BeliefMDP_n_Mapping,
        using quantized beliefs from the codebook B_n_M.

        Args:
            j: Index of target belief in B_n_M (i.e., π_j^M = self.BQ.Π_n_M[j])
            i: Index of current belief in B_n_M (i.e., π_i^M = self.BQ.Π_n_M[i])
            x_current: current state (state_dim,) - part of augmented state b_t = (x_t, π_t)
            u: Action (2,)
            x_next: next state (state_dim,) - given as input, not sampled
        Returns:
            float: Joint probability of (π_j^M, x_next) from augmented state (x_current, π_i^M) under action u
        """
        # Access beliefs directly from cached codebook
        B_new_flat = self.BQ.Π_n_M[j]  # Target belief (len_M,)
        π_flat = self.BQ.Π_n_M[i]      # Current belief (len_M,)

        # For mapping, beliefs are already 1D (no unflatten needed)
        B_new_1d = B_new_flat
        π_1d = π_flat

        # Delegate to parent (finite-sum) implementation
        return super().η_n(B_new_1d, π_1d, x_current, u, x_next)

    def η_n_batch(self, j_list: list[int], i: int, x_current: np.ndarray, u: np.ndarray, x_next: np.ndarray) -> np.ndarray:
        """
        Fully vectorized batched version of η_n for mapping.

        Computes joint probabilities η_n(π_j^M, x_next | π_i^M, x_current, u) for all target beliefs simultaneously
        using discrete observation quantization.

        Args:
            j_list: List of target belief indices in B_n_M
            i: Index of current belief in B_n_M
            x_current: current state (state_dim,) - part of augmented state b_t = (x_t, π_t)
            u: Action (2,)
            x_next: next state (state_dim,) - given as input, not sampled
        Returns:
            np.ndarray: Array of joint probabilities, shape (len(j_list),)
        """
        if len(j_list) == 0:
            return np.array([])

        if self.Q_n is None:
            raise ValueError("Q_n must be precomputed. Ensure observation quantization is configured.")

        # Access current belief from codebook
        π_1d = self.BQ.Π_n_M[i]  # Current belief (len_M,)

        # Get quantized indices
        x_current_idx = self.SQ.get_quantized_index(x_current)
        x_next_idx = self.SQ.get_quantized_index(x_next)
        u_idx = self.AQ.get_quantized_index(u)

        # Step 1: Get T(x_next | x_current, u) from T_mat
        Tn_mat = self.T_mat[:, :, u_idx]  # (m_n, m_n)
        T_x_next_given_x_current_u = float(Tn_mat[x_next_idx, x_current_idx])

        # If transition probability is zero, return zeros
        if T_x_next_given_x_current_u < 1e-300:
            return np.zeros(len(j_list), dtype=np.float32)

        # Step 2: Compute H for ALL observations at once: H_y = sum(Q_slice * π) over maps
        m_y = int(self.Y_n.shape[0])
        # Q_n shape: (m_y, m_n, len_M), slice for x_next: (m_y, len_M)
        Q_slice = self.Q_n[:, x_next_idx, :]  # (m_y, len_M)
        from scipy.special import logsumexp
        if is_cupy:
            from cupyx.scipy.special import logsumexp
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

        # Step 3: Compute F for ALL observations directly from Q_slice and π
        log_F = log_numerator - log_H_y[:, np.newaxis]  # (m_y, len_M)
        π_all_batch = np.exp(log_F)  # (m_y, len_M)

        # Pushforward under quantizer: assign each filtered belief to nearest codebook point.
        codebook = self.BQ.Π_n_M  # (cardinality, len_M)
        codebook_norm_sq = np.sum(codebook * codebook, axis=1)[np.newaxis, :]  # (1, cardinality)
        sampled_norm_sq = np.sum(π_all_batch * π_all_batch, axis=1, keepdims=True)  # (m_y, 1)
        dots = π_all_batch @ codebook.T  # (m_y, cardinality)
        distances_sq = sampled_norm_sq + codebook_norm_sq - 2.0 * dots
        nearest_idx = np.argmin(distances_sq, axis=1).astype(np.int64)  # (m_y,)

        prob_belief_all = np.bincount(
            nearest_idx,
            weights=H_y.astype(np.float32, copy=False),
            minlength=self.BQ.cardinality,
        ).astype(np.float32)  # (cardinality,)
        j_arr = np.array(j_list, dtype=np.int64)
        prob_belief = prob_belief_all[j_arr]

        # Return joint probability = T(x_next | x_current, u) * P(π' ≈ π_j | x_next)
        probabilities = prob_belief * T_x_next_given_x_current_u

        return probabilities.astype(np.float32)

    def η_n_batch_i_batch(
        self,
        j_list: list[int],
        i_list: list[int],
        x_current: np.ndarray,
        u: np.ndarray,
        x_next: np.ndarray,
    ) -> np.ndarray:
        """
        Batched mapping transition over source beliefs i and target beliefs j.
        Legacy interface — delegates to _η_n_fast with index lookups.
        """
        if len(j_list) == 0 or len(i_list) == 0:
            return np.zeros((len(i_list), len(j_list)), dtype=np.float32)
        x_current_idx = self.SQ.get_quantized_index(x_current)
        x_next_idx = self.SQ.get_quantized_index(x_next)
        u_idx = self.AQ.get_quantized_index(u)
        T_val = float(self.T_mat[x_next_idx, x_current_idx, u_idx])
        if T_val < 1e-300:
            return np.zeros((len(i_list), len(j_list)), dtype=np.float32)
        # Precompute what we can
        log_Q_slice = np.log(np.maximum(self.Q_n[:, x_next_idx, :], 1e-300))
        valid_obs = np.any(self.Q_n[:, x_next_idx, :] > 1e-300, axis=1)
        i_arr = np.array(i_list, dtype=np.int64)
        j_arr = np.array(j_list, dtype=np.int64)
        log_π_all = np.log(np.maximum(self.BQ.Π_n_M, 1e-300))
        π_targets_all = self.BQ.Π_n_M
        target_norm_sq = np.sum(π_targets_all * π_targets_all, axis=1)
        return self._η_n_fast(
            i_arr, j_arr, T_val, log_Q_slice, valid_obs,
            log_π_all, π_targets_all, target_norm_sq,
        )

    def _η_n_fast(
        self,
        i_arr: np.ndarray,
        j_arr: np.ndarray,
        T_val: float,
        log_Q_slice: np.ndarray,
        valid_obs_mask: np.ndarray,
        log_π_all: np.ndarray,
        π_targets_all: np.ndarray,
        target_norm_sq_all: np.ndarray,
        j_search_batches: list | None = None,
    ) -> np.ndarray:
        """
        Core η computation with precomputed invariants. All args are already on the active backend.

        Args:
            i_arr: source belief indices (n_i,)
            j_arr: target belief indices (n_j,)
            T_val: scalar T(x_next | x_current, u)
            log_Q_slice: precomputed log Q_n[:, x_next_idx, :] — (m_y, len_M)
            valid_obs_mask: boolean mask of observations with nonzero Q — (m_y,)
            log_π_all: precomputed log of full codebook — (cardinality, len_M)
            π_targets_all: full codebook — (cardinality, len_M)
            target_norm_sq_all: precomputed ||π_j||² for full codebook — (cardinality,)
        """
        if is_cupy:
            from cupyx.scipy.special import logsumexp
        else:
            from scipy.special import logsumexp

        nearest_idx, H_y = self._η_n_fast_assignments(
            i_arr=i_arr,
            log_Q_slice=log_Q_slice,
            valid_obs_mask=valid_obs_mask,
            log_π_all=log_π_all,
            π_targets_all=π_targets_all,
            target_norm_sq_all=target_norm_sq_all,
            j_search_batches=j_search_batches,
        )

        n_i = int(len(i_arr))
        n_j = int(len(j_arr))
        if n_i == 0 or n_j == 0:
            return np.zeros((n_i, n_j), dtype=np.float32)

        # Build requested j-slice via weighted bincount per source belief.
        nearest_idx_cpu = nearest_idx.get() if hasattr(nearest_idx, 'get') else _numpy.asarray(nearest_idx)
        H_y_cpu = H_y.get() if hasattr(H_y, 'get') else _numpy.asarray(H_y)
        j_arr_cpu = j_arr.get() if hasattr(j_arr, 'get') else _numpy.asarray(j_arr)
        cardinality = int(π_targets_all.shape[0])

        probs = _numpy.zeros((n_i, n_j), dtype=_numpy.float32)
        for src in range(n_i):
            mass_all = _numpy.bincount(
                nearest_idx_cpu[src].astype(_numpy.int64, copy=False),
                weights=H_y_cpu[src].astype(_numpy.float32, copy=False),
                minlength=cardinality,
            ).astype(_numpy.float32, copy=False)
            probs[src] = mass_all[j_arr_cpu]

        return (np.asarray(probs) * T_val).astype(np.float32)

    def _η_n_fast_assignments(
        self,
        i_arr: np.ndarray,
        log_Q_slice: np.ndarray,
        valid_obs_mask: np.ndarray,
        log_π_all: np.ndarray,
        π_targets_all: np.ndarray,
        target_norm_sq_all: np.ndarray,
        j_search_batches: list | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute nearest codebook assignment for each filtered belief sample.

        Returns:
            nearest_idx: (n_i, m_y_valid) global codebook indices
            H_y: (n_i, m_y_valid) normalized observation weights
        """
        if is_cupy:
            from cupyx.scipy.special import logsumexp
            import cupy as cp
        else:
            from scipy.special import logsumexp

        n_i = len(i_arr)
        valid_idx = np.where(valid_obs_mask)[0]
        if len(valid_idx) == 0:
            return (
                np.zeros((n_i, 0), dtype=np.int64),
                np.zeros((n_i, 0), dtype=np.float32),
            )

        log_Q_valid = log_Q_slice[valid_idx]  # (m_y_valid, len_M)
        log_π_i = log_π_all[i_arr]            # (n_i, len_M)
        log_numerator = log_Q_valid[np.newaxis, :, :] + log_π_i[:, np.newaxis, :]

        log_H_y = logsumexp(log_numerator, axis=2)
        H_y = np.exp(log_H_y)
        total_H = np.sum(H_y, axis=1, keepdims=True)
        total_H = np.where(total_H > 0, total_H, 1.0)
        H_y = (H_y / total_H).astype(np.float32)

        log_F = log_numerator - log_H_y[:, :, np.newaxis]
        π_filtered = np.exp(log_F)  # (n_i, m_y_valid, len_M)

        sampled_norm_sq = np.sum(π_filtered * π_filtered, axis=2)  # (n_i, m_y_valid)
        m_y_valid = sampled_norm_sq.shape[1]

        # Chunked nearest-neighbor search over full codebook to control memory.
        if j_search_batches is None:
            if is_cupy:
                all_j = cp.arange(int(π_targets_all.shape[0]), dtype=cp.int64)
            else:
                all_j = _numpy.arange(int(π_targets_all.shape[0]), dtype=_numpy.int64)
            j_search_batches = [all_j]

        inf_val = np.array(_numpy.inf, dtype=np.float32)
        best_dist = np.full((n_i, m_y_valid), inf_val, dtype=np.float32)
        best_idx = np.full((n_i, m_y_valid), -1, dtype=np.int64)

        for j_chunk in j_search_batches:
            π_chunk = π_targets_all[j_chunk]          # (c, len_M)
            norm_chunk = target_norm_sq_all[j_chunk]  # (c,)
            dots = np.einsum('iml,jl->imj', π_filtered, π_chunk)  # (n_i, m_y_valid, c)
            distances_sq = (
                sampled_norm_sq[:, :, np.newaxis]
                + norm_chunk[np.newaxis, np.newaxis, :]
                - 2.0 * dots
            ).astype(np.float32)

            local_arg = np.argmin(distances_sq, axis=2).astype(np.int64)  # (n_i, m_y_valid)
            local_best = np.take_along_axis(distances_sq, local_arg[:, :, np.newaxis], axis=2)[:, :, 0]
            chunk_idx = j_chunk[local_arg]

            improve = local_best < best_dist
            best_dist = np.where(improve, local_best, best_dist)
            best_idx = np.where(improve, chunk_idx, best_idx)

        return best_idx, H_y

    def _compute_action_sparse(self, k, nonzero_pairs, precomputed, gpu_id=0,
                                show_progress=True, pbar_position=None):
        """
        Compute sparse matrix entries for a single action on a specific GPU.

        Args:
            k: Action index (for logging only).
            nonzero_pairs: List of (x_current_idx, x_next_idx, T_value) tuples.
            precomputed: Dict with CPU numpy arrays: log_π_all, π_targets_all,
                         target_norm_sq_all, log_Q_n, Q_any_nonzero, j_batch_arrays_cpu,
                         cardinality, m_n, i_batch_size, threshold.
            gpu_id: Which GPU device to run on (0-indexed).
            show_progress: Whether to show a tqdm progress bar.
            pbar_position: tqdm bar position (for multi-GPU, each GPU gets its own line).

        Returns:
            (rows_list, cols_list, data_list, profile) where profile includes timing and nnz stats.
        """
        import time
        cardinality = precomputed['cardinality']
        m_n = precomputed['m_n']
        i_batch_size = precomputed['i_batch_size']
        threshold = precomputed['threshold']
        j_batch_arrays_cpu = precomputed['j_batch_arrays_cpu']
        Q_any_nonzero_cpu = precomputed['Q_any_nonzero']
        n_i_batches = (cardinality + i_batch_size - 1) // i_batch_size
        rows_chunks = []
        cols_chunks = []
        data_chunks = []
        assignment_time_s = 0.0
        assembly_time_s = 0.0
        t_total_start = time.time()
        assembly_src_chunk = int(precomputed.get('assembly_src_chunk', 256))

        if is_cupy:
            import cupy as cp
            device_ctx = cp.cuda.Device(gpu_id)
            device_ctx.__enter__()
            # Transfer precomputed data to this GPU
            local_log_π_all = cp.asarray(precomputed['log_π_all'])
            local_π_targets_all = cp.asarray(precomputed['π_targets_all'])
            local_target_norm_sq_all = cp.asarray(precomputed['target_norm_sq_all'])
            local_log_Q_n = cp.asarray(precomputed['log_Q_n'])
            local_j_batches = [cp.asarray(jb) for jb in j_batch_arrays_cpu]
        else:
            device_ctx = None
            local_log_π_all = precomputed['log_π_all']
            local_π_targets_all = precomputed['π_targets_all']
            local_target_norm_sq_all = precomputed['target_norm_sq_all']
            local_log_Q_n = precomputed['log_Q_n']
            local_j_batches = [_numpy.asarray(jb) for jb in j_batch_arrays_cpu]

        try:
            n_u_total = precomputed.get('n_u', '?')
            n_states = m_n * cardinality
            pbar = tqdm(
                total=n_states,
                desc=f"Action {k+1}/{n_u_total} [GPU {gpu_id}]",
                disable=not show_progress,
                leave=False,
                mininterval=1.0,
                position=pbar_position,
            )
            local_nnz = 0

            for x_current_idx in range(m_n):
                reachable = [(xn, tv) for (xc, xn, tv) in nonzero_pairs if xc == x_current_idx]
                if not reachable:
                    pbar.update(cardinality)
                    continue

                for i_batch_idx in range(n_i_batches):
                    start_i = i_batch_idx * i_batch_size
                    end_i = min(start_i + i_batch_size, cardinality)
                    if is_cupy:
                        i_arr = cp.arange(start_i, end_i, dtype=cp.int64)
                    else:
                        i_arr = _numpy.arange(start_i, end_i, dtype=_numpy.int64)
                    i_arr_cpu = _numpy.arange(start_i, end_i, dtype=_numpy.int64)

                    for x_next_idx, T_val in reachable:
                        log_Q_slice = local_log_Q_n[:, x_next_idx, :]
                        if is_cupy:
                            valid_obs = cp.asarray(Q_any_nonzero_cpu[:, x_next_idx])
                        else:
                            valid_obs = Q_any_nonzero_cpu[:, x_next_idx]

                        t_assign_start = time.time()
                        nearest_idx, H_y = self._η_n_fast_assignments(
                            i_arr=i_arr,
                            log_Q_slice=log_Q_slice,
                            valid_obs_mask=valid_obs,
                            log_π_all=local_log_π_all,
                            π_targets_all=local_π_targets_all,
                            target_norm_sq_all=local_target_norm_sq_all,
                            j_search_batches=local_j_batches,
                        )
                        assignment_time_s += time.time() - t_assign_start

                        t_assemble_start = time.time()
                        nearest_idx_cpu = nearest_idx.get() if hasattr(nearest_idx, 'get') else _numpy.asarray(nearest_idx)
                        H_y_cpu = H_y.get() if hasattr(H_y, 'get') else _numpy.asarray(H_y)
                        n_src, m_obs = int(nearest_idx_cpu.shape[0]), int(nearest_idx_cpu.shape[1])
                        if n_src > 0 and m_obs > 0:
                            t_val32 = _numpy.float32(T_val)
                            for src_start in range(0, n_src, assembly_src_chunk):
                                src_end = min(src_start + assembly_src_chunk, n_src)
                                n_local = src_end - src_start

                                nearest_local = nearest_idx_cpu[src_start:src_end].astype(_numpy.int64, copy=False)
                                weights_local = (H_y_cpu[src_start:src_end].astype(_numpy.float32, copy=False) * t_val32)

                                # Pack (source_rel, belief_idx) into a single integer key.
                                # keys in [0, n_local*cardinality); aggregation is C-level.
                                src_offsets = (
                                    _numpy.arange(n_local, dtype=_numpy.int64)[:, _numpy.newaxis] * cardinality
                                )
                                packed_keys = (src_offsets + nearest_local).reshape(-1)
                                packed_weights = weights_local.reshape(-1)

                                masses_flat = _numpy.bincount(
                                    packed_keys,
                                    weights=packed_weights,
                                    minlength=n_local * cardinality,
                                ).astype(_numpy.float32, copy=False)

                                nz_local = _numpy.flatnonzero(masses_flat > threshold)
                                if nz_local.size == 0:
                                    continue

                                src_rel = nz_local // cardinality
                                belief_idx = nz_local % cardinality
                                rows_block = (
                                    x_current_idx * cardinality
                                    + i_arr_cpu[src_start + src_rel].astype(_numpy.int64, copy=False)
                                )
                                cols_block = x_next_idx * cardinality + belief_idx.astype(_numpy.int64, copy=False)
                                vals_block = masses_flat[nz_local]

                                rows_chunks.append(rows_block)
                                cols_chunks.append(cols_block)
                                data_chunks.append(vals_block)
                                local_nnz += int(vals_block.size)
                        assembly_time_s += time.time() - t_assemble_start

                    pbar.update(end_i - start_i)

                pbar.set_postfix({'nnz': f'{local_nnz:,}'})

                if is_cupy and x_current_idx % 2 == 0 and x_current_idx > 0:
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()

            pbar.close()
        finally:
            if is_cupy:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                device_ctx.__exit__(None, None, None)

        elapsed_total_s = time.time() - t_total_start
        if rows_chunks:
            rows_arr = _numpy.concatenate(rows_chunks).astype(_numpy.int64, copy=False)
            cols_arr = _numpy.concatenate(cols_chunks).astype(_numpy.int64, copy=False)
            data_arr = _numpy.concatenate(data_chunks).astype(_numpy.float32, copy=False)
        else:
            rows_arr = _numpy.empty(0, dtype=_numpy.int64)
            cols_arr = _numpy.empty(0, dtype=_numpy.int64)
            data_arr = _numpy.empty(0, dtype=_numpy.float32)

        profile = {
            'action_idx': int(k),
            'gpu_id': int(gpu_id),
            'elapsed_total_s': float(elapsed_total_s),
            'assignment_time_s': float(assignment_time_s),
            'assembly_time_s': float(assembly_time_s),
            'nnz_entries': int(local_nnz),
        }
        return rows_arr, cols_arr, data_arr, profile

    def _build_csr_from_entries(self, rows_list, cols_list, data_list, n_states, main_gpu=0):
        """Build a normalized CSR sparse matrix from COO entries on the main GPU."""
        if rows_list is not None and len(rows_list) > 0:
            rows_arr = _numpy.asarray(rows_list, dtype=_numpy.int64)
            cols_arr = _numpy.asarray(cols_list, dtype=_numpy.int32)
            data_arr = _numpy.asarray(data_list, dtype=_numpy.float32)
            # Normalize each source row so outgoing mass sums to 1
            row_sums = _numpy.bincount(rows_arr, weights=data_arr, minlength=n_states).astype(_numpy.float32)
            nonzero_mask = row_sums[rows_arr] > 0
            data_arr[nonzero_mask] = data_arr[nonzero_mask] / row_sums[rows_arr][nonzero_mask]
            if is_cupy:
                import cupy as cp
                with cp.cuda.Device(main_gpu):
                    rows_backend = cp.asarray(rows_arr).ravel()
                    cols_backend = cp.asarray(cols_arr).ravel()
                    data_backend = cp.asarray(data_arr).ravel()
            else:
                rows_backend = rows_arr.ravel()
                cols_backend = cols_arr.ravel()
                data_backend = data_arr.ravel()
            coo_mat = coo_matrix((data_backend, (rows_backend, cols_backend)),
                                 shape=(n_states, n_states), dtype=np.float32)
            return coo_mat.tocsr()
        else:
            return csr_matrix((n_states, n_states), dtype=np.float32)

    def _compute_p_n_M(self, j_batch_size: int = None,
                       i_batch_size: int = None, show_progress: bool = True,
                       threshold: float = 1e-8, n_gpus: int = 1,
                       checkpoint_path: Path | None = None) -> list:
        """
        Compute the full transition probability for mapping as a list of sparse matrices (one per action).

        Supports multi-GPU parallelism: when n_gpus > 1, actions are computed in parallel
        across GPU devices using ThreadPoolExecutor (CuPy releases GIL during kernel execution).

        State index: row/col = x_idx * cardinality + π_idx (flat over (m_n, cardinality)).
        p_n_M[k][row, col] = p(x_next, π_next | x_current, π_current, u_k).

        Returns:
            List of CSR matrices, each shape (m_n * cardinality, m_n * cardinality).
        """
        cardinality = self.BQ.cardinality
        n_u = self.AQ.n_u
        m_n = self.SQ.m_n
        n_states = m_n * cardinality
        j_batch_size = j_batch_size or getattr(self, 'j_batch_size', 136)
        i_batch_size = i_batch_size or getattr(self, 'i_batch_size', 10)
        n_j_batches = (cardinality + j_batch_size - 1) // j_batch_size

        # ========== PRECOMPUTATION on CPU (portable across GPUs) ==========
        Π_cpu = self.BQ.Π_n_M.get() if hasattr(self.BQ.Π_n_M, 'get') else _numpy.asarray(self.BQ.Π_n_M)
        Q_n_cpu = self.Q_n.get() if hasattr(self.Q_n, 'get') else _numpy.asarray(self.Q_n)
        T_mat_full_cpu = self.T_mat.get() if hasattr(self.T_mat, 'get') else _numpy.asarray(self.T_mat)

        log_π_all_cpu = _numpy.log(_numpy.maximum(Π_cpu, 1e-300))
        π_targets_cpu = Π_cpu.copy()
        target_norm_sq_cpu = _numpy.sum(π_targets_cpu ** 2, axis=1)
        log_Q_n_cpu = _numpy.log(_numpy.maximum(Q_n_cpu, 1e-300))
        Q_any_nonzero_cpu = _numpy.any(Q_n_cpu > 1e-300, axis=2)

        j_batch_arrays_cpu = []
        for jb in range(n_j_batches):
            start_j = jb * j_batch_size
            end_j = min(start_j + j_batch_size, cardinality)
            j_batch_arrays_cpu.append(_numpy.arange(start_j, end_j, dtype=_numpy.int64))

        precomputed = {
            'log_π_all': log_π_all_cpu,
            'π_targets_all': π_targets_cpu,
            'target_norm_sq_all': target_norm_sq_cpu,
            'log_Q_n': log_Q_n_cpu,
            'Q_any_nonzero': Q_any_nonzero_cpu,
            'j_batch_arrays_cpu': j_batch_arrays_cpu,
            'cardinality': cardinality,
            'm_n': m_n,
            'i_batch_size': i_batch_size,
            'assembly_src_chunk': 256,
            'threshold': threshold,
            'n_u': n_u,
        }

        # ========== Precompute nonzero T pairs per action ==========
        action_nonzero_pairs = []
        for k in range(n_u):
            u_idx = int(self.AQ.get_quantized_index(self.AQ.U[k]))
            T_k = T_mat_full_cpu[:, :, u_idx]
            nonzero_pairs = []
            for xc in range(m_n):
                for xn in range(m_n):
                    t_val = float(T_k[xn, xc])
                    if t_val > 1e-300:
                        nonzero_pairs.append((xc, xn, t_val))
            action_nonzero_pairs.append(nonzero_pairs)

        if show_progress:
            print(f"Computing p_n_M for mapping (quantized η_n_batch, sparse)...")
            print(f"  States: {n_states:,} = m_n={m_n} × cardinality={cardinality}, n_u={n_u}")
            valid_counts = [int(_numpy.sum(Q_any_nonzero_cpu[:, xn])) for xn in range(m_n)]
            m_y_total = int(Q_n_cpu.shape[0])
            print(f"  Obs pruning: {m_y_total} total obs, valid per x_next: "
                  f"min={min(valid_counts)}, max={max(valid_counts)}, mean={sum(valid_counts)/len(valid_counts):.0f}")
            if n_gpus > 1:
                print(f"  Multi-GPU: {n_gpus} devices, {n_u} actions → "
                      f"{(n_u + n_gpus - 1) // n_gpus} rounds")

        # ========== DISPATCH: multi-GPU or single-GPU ==========
        partial_actions = self._load_p_n_M_checkpoint(checkpoint_path) if checkpoint_path is not None else None
        if partial_actions is not None and show_progress:
            done = sum(1 for x in partial_actions if x is not None)
            print(f"  Resuming from checkpoint: {done}/{n_u} actions already computed")

        if is_cupy and n_gpus > 1:
            p_n_M_sparse = self._compute_p_n_M_multi_gpu(
                n_u, n_gpus, n_states, action_nonzero_pairs, precomputed, show_progress, m_n,
                checkpoint_path=checkpoint_path, partial_actions=partial_actions)
        else:
            p_n_M_sparse = self._compute_p_n_M_single_gpu(
                n_u, n_states, action_nonzero_pairs, precomputed, show_progress, m_n,
                checkpoint_path=checkpoint_path, partial_actions=partial_actions)

        return p_n_M_sparse

    def _compute_p_n_M_single_gpu(self, n_u, n_states, action_nonzero_pairs,
                                   precomputed, show_progress, m_n,
                                   checkpoint_path: Path | None = None,
                                   partial_actions: list | None = None):
        """Single-GPU (or CPU) sequential computation of all actions."""
        import time
        p_n_M_sparse = partial_actions if partial_actions is not None else [None] * n_u

        for k in range(n_u):
            if p_n_M_sparse[k] is not None:
                if show_progress:
                    tqdm.write(f"  Action {k+1}/{n_u}: loaded from checkpoint, skipping")
                continue
            nonzero_pairs = action_nonzero_pairs[k]
            n_nonzero = len(nonzero_pairs)
            n_skipped = m_n * m_n - n_nonzero

            if show_progress:
                tqdm.write(f"  Action {k+1}/{n_u}: T-prefilter {n_nonzero}/{m_n*m_n} nonzero "
                           f"({n_skipped} skipped, {n_skipped/(m_n*m_n)*100:.0f}% saved)")

            rows_list, cols_list, data_list, profile = self._compute_action_sparse(
                k, nonzero_pairs, precomputed, gpu_id=0,
                show_progress=show_progress, pbar_position=0)

            t_build_start = time.time()
            csr_mat = self._build_csr_from_entries(rows_list, cols_list, data_list, n_states)
            build_time_s = time.time() - t_build_start
            p_n_M_sparse[k] = csr_mat
            if checkpoint_path is not None:
                self._save_p_n_M_checkpoint(checkpoint_path, p_n_M_sparse)

            if show_progress:
                nnz = csr_mat.nnz
                sparsity = (1.0 - nnz / (n_states * n_states)) * 100
                tqdm.write(f"  Action {k+1}/{n_u}: {nnz:,} non-zeros ({sparsity:.6f}% sparse)")
                total_s = float(profile['elapsed_total_s'] + build_time_s)
                nnz_rate = (nnz / total_s) if total_s > 0 else 0.0
                tqdm.write(
                    "    profile: "
                    f"assign={profile['assignment_time_s']:.2f}s, "
                    f"assemble={profile['assembly_time_s']:.2f}s, "
                    f"csr_build={build_time_s:.2f}s, "
                    f"total={total_s:.2f}s, nnz/sec={nnz_rate:,.0f}"
                )

        return p_n_M_sparse

    def _compute_p_n_M_multi_gpu(self, n_u, n_gpus, n_states, action_nonzero_pairs,
                                  precomputed, show_progress, m_n,
                                  checkpoint_path: Path | None = None,
                                  partial_actions: list | None = None):
        """Multi-GPU parallel computation. Processes actions in rounds of n_gpus."""
        from concurrent.futures import ThreadPoolExecutor
        import time

        p_n_M_sparse = partial_actions if partial_actions is not None else [None] * n_u
        pending_actions = [k for k in range(n_u) if p_n_M_sparse[k] is None]
        if len(pending_actions) == 0:
            return p_n_M_sparse
        n_rounds = (len(pending_actions) + n_gpus - 1) // n_gpus

        for round_idx in range(n_rounds):
            round_start = round_idx * n_gpus
            round_end = min(round_start + n_gpus, len(pending_actions))
            round_actions = pending_actions[round_start:round_end]

            if show_progress:
                action_strs = ', '.join(str(k+1) for k in round_actions)
                tqdm.write(f"  Round {round_idx+1}/{n_rounds}: actions [{action_strs}] "
                           f"on GPUs 0..{len(round_actions)-1}")

            t0 = time.time()

            with ThreadPoolExecutor(max_workers=len(round_actions)) as pool:
                futures = {}
                future_gpu = {}
                for i, k in enumerate(round_actions):
                    gpu_id = i  # Each action in the round gets its own GPU
                    futures[k] = pool.submit(
                        self._compute_action_sparse,
                        k, action_nonzero_pairs[k], precomputed, gpu_id,
                        show_progress=show_progress, pbar_position=i,
                    )
                    future_gpu[k] = gpu_id

                for k in round_actions:
                    try:
                        rows_list, cols_list, data_list, profile = futures[k].result()
                    except Exception as e:
                        gpu_id = future_gpu.get(k, -1)
                        raise RuntimeError(
                            f"Action {k+1}/{n_u} failed on GPU {gpu_id} during round {round_idx+1}/{n_rounds}"
                        ) from e
                    t_build_start = time.time()
                    csr_mat = self._build_csr_from_entries(rows_list, cols_list, data_list, n_states)
                    build_time_s = time.time() - t_build_start
                    p_n_M_sparse[k] = csr_mat

                    if show_progress:
                        nnz = csr_mat.nnz
                        sparsity = (1.0 - nnz / (n_states * n_states)) * 100
                        tqdm.write(f"    Action {k+1}/{n_u}: {nnz:,} non-zeros ({sparsity:.6f}% sparse)")
                        total_s = float(profile['elapsed_total_s'] + build_time_s)
                        nnz_rate = (nnz / total_s) if total_s > 0 else 0.0
                        tqdm.write(
                            "      profile: "
                            f"assign={profile['assignment_time_s']:.2f}s, "
                            f"assemble={profile['assembly_time_s']:.2f}s, "
                            f"csr_build={build_time_s:.2f}s, "
                            f"total={total_s:.2f}s, nnz/sec={nnz_rate:,.0f}"
                        )

            if checkpoint_path is not None:
                self._save_p_n_M_checkpoint(checkpoint_path, p_n_M_sparse)
                if show_progress:
                    completed = sum(1 for x in p_n_M_sparse if x is not None)
                    tqdm.write(
                        f"    Checkpoint saved: {completed}/{n_u} actions -> {checkpoint_path}"
                    )

            if show_progress:
                elapsed = time.time() - t0
                tqdm.write(f"    Round {round_idx+1} done in {elapsed:.1f}s")

        return p_n_M_sparse

    def get_codebook(self) -> np.ndarray:
        """
        Get the quantized belief codebook.

        Returns:
            B_n_M: Array of shape (cardinality, len_M) containing all quantized beliefs
        """
        return self.BQ.Π_n_M

    def _compute_c_n_M(self, show_progress: bool = True) -> np.ndarray:
        """
        Precompute costs for all belief-action combinations.

        ρ_n(B, u) = r_exploration(B) + c_effort(u), with B from codebook and u from action set.

        Returns:
            c_n_M: Array of shape (cardinality, n_u) where c_n_M[i, k] = ρ_n(π_i^M, u_k)
        """
        cardinality = self.BQ.cardinality
        n_u = self.AQ.n_u
        B_n = self.BQ.Π_n_M  # (cardinality, len_M)
        U_n = self.AQ.U  # (n_u, 2)

        if show_progress:
            print(f"  Computing costs for {cardinality:,} beliefs × {n_u} actions...")

        # Staged finite checks for early error localization
        B_np = B_n.get() if hasattr(B_n, 'get') else _numpy.asarray(B_n)
        U_np = U_n.get() if hasattr(U_n, 'get') else _numpy.asarray(U_n)
        if _numpy.any(_numpy.isnan(B_np)) or _numpy.any(_numpy.isinf(B_np)):
            raise ValueError(
                "Belief codebook Π_n_M contains NaN or Inf. Regenerate or clear belief quantizer cache."
            )
        if _numpy.any(_numpy.isnan(U_np)) or _numpy.any(_numpy.isinf(U_np)):
            raise ValueError(
                "Action set U_n contains NaN or Inf. Check action quantizer."
            )

        # Use vectorized batch computation: ρ_n(B_n, U_n) -> (cardinality, n_u)
        c_n_M = self.ρ_n(B_n, U_n)  # (cardinality, n_u)

        c_n_M_np = c_n_M.get() if hasattr(c_n_M, 'get') else _numpy.asarray(c_n_M)
        if _numpy.any(_numpy.isnan(c_n_M_np)) or _numpy.any(_numpy.isinf(c_n_M_np)):
            raise ValueError(
                "c_n_M contains NaN or Inf after ρ_n computation. "
                "Check beliefs (codebook Π_n_M) and cost ρ_n upstream."
            )
        return c_n_M

__all__ = ['BeliefMDP_n_M_SLAM', 'BeliefMDP_n_M_Localization', 'BeliefMDP_n_M_Mapping']
