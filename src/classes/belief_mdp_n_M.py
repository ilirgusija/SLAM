# import cupy as np
import hashlib
from pathlib import Path
from typing import Literal
from .belief_mdp_n import BeliefMDP_n_SLAM, BeliefMDP_n_Localization, BeliefMDP_n_Mapping
from .mapping import BaseMap, LidarGridMapVec
from .model import LIDAR, RangeBearingSensor, SingleIntegratorModel, DoubleIntegratorModel
from .obstacle import Obstacle
from .quantizer import BeliefQuantizer
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

        return {
            'M': self.M,
            'n': self.n,
            'state_bounds': state_bounds_save,
            'max_val': max_val,
            'dt': float(self.motion_model.dt),
            'map_shape': _numpy.array([self.map_H, self.map_W]),
            'sigma_w': float(self.σ_w),
            'sigma_v': float(self.σ_v),
            'm_n': self.SQ.m_n,
            'n_u': self.AQ.n_u,
            'cardinality': self.BQ.cardinality,
            'N_n': self.N_n,
            'model': self.motion_model.__class__.__name__,
            'state_dim': self.state_dim,
            'problem_type': self.problem_type,
        }

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
        filename = f"{cache_base_name}_M{self.M}_n{self.n}_map{self.map_H}x{self.map_W}_max{max_val}_{metadata_hash}.npz"

        return cache_dir / filename

    def _verify_metadata(self, data: dict, expected_metadata: dict) -> bool:
        """Verify that cached metadata matches current configuration."""
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
            for k in range(n_actions):
                data_key = f'p_n_M_{k}_data'
                if data_key not in data:
                    return False

                mat_data = data[data_key].astype(_numpy.float16)
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

    def _load_dense_c_n_M(self, cache_path: Path, kernel_version: str, expected_shape: tuple) -> bool:
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

            if is_cupy:
                import cupy as cp
                c_n_M = cp.asarray(c_n_M)

            self.c_n_M = c_n_M
            return True
        except Exception as e:
            print(f"Error loading c_n_M cache: {e}")
            import traceback
            traceback.print_exc()
            return False

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
        using quantized beliefs from the codebook Π_n_M.

        Args:
            j: Index of target belief in Π_n_M (π_j^M)
            i: Index of current belief in Π_n_M (π_i^M)
            u: Action (2,)
        Returns:
            float: Probability of transitioning from π_i^M to π_j^M under action u
        """
        # Access beliefs directly from cached codebook
        π_new_flat = self.BQ.Π_n_M[j]  # (N_n,)
        π_flat = self.BQ.Π_n_M[i]      # (N_n,)

        π_new_2d = self.unflatten_belief(π_new_flat)
        π_2d = self.unflatten_belief(π_flat)

        # Delegate to parent (finite-sum) implementation
        return super().η_n(π_new_2d, π_2d, u)

    def η_n_batch(self, j_list: list[int], i: int, u: np.ndarray) -> np.ndarray:
        """
        Fully vectorized batched version of η_n over multiple target belief indices.

        Computes probabilities for all target beliefs simultaneously by:
        1. Computing H_y and π_all_batch (all updated beliefs for all observations) once
        2. Vectorized distance computation between all target beliefs and all updated beliefs
        3. Summing H_y for matching observations for each target belief

        Args:
            j_list: List of target belief indices in Π_n_M
            i: Index of current belief in Π_n_M
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

        # Access all target beliefs from codebook
        n_targets = len(j_list)
        π_targets_flat = np.array([self.BQ.Π_n_M[j] for j in j_list])  # (n_targets, N_n)
        π_targets_2d = np.array([self.unflatten_belief(π_flat) for π_flat in π_targets_flat])  # (n_targets, m_n, len_M)

        # Vectorized distance computation:
        # π_all_batch: (m_y, m_n, len_M)
        # π_targets_2d: (n_targets, m_n, len_M)
        # We want distances: (n_targets, m_y)
        π_all_flat = π_all_batch.reshape(m_y, -1)  # (m_y, m_n * len_M)
        π_targets_flat_2d = π_targets_2d.reshape(n_targets, -1)  # (n_targets, m_n * len_M)

        # Compute pairwise distances: (n_targets, m_y)
        # Using broadcasting: (n_targets, 1, m_n*len_M) - (1, m_y, m_n*len_M) -> (n_targets, m_y, m_n*len_M)
        π_diff = π_targets_flat_2d[:, np.newaxis, :] - π_all_flat[np.newaxis, :, :]  # (n_targets, m_y, m_n*len_M)
        distances = np.linalg.norm(π_diff, axis=2)  # (n_targets, m_y)

        # Vectorized matching: mask observations where distance < threshold for each target
        threshold = 1e-3
        matches = (distances < threshold) & (H_y[np.newaxis, :] > 0.0)  # (n_targets, m_y)

        # Sum H_y for matching observations for each target belief
        H_y_broadcast = H_y[np.newaxis, :]  # (1, m_y)
        probs = np.sum(H_y_broadcast * matches, axis=1)  # (n_targets,)

        return probs.astype(np.float32)

    def η_n_batch_i_batch(self, j_list: list[int], i_list: list[int], u: np.ndarray) -> np.ndarray:
        """
        Fully vectorized batched version of η_n over multiple source and target belief indices.

        Processes multiple source beliefs (i) and multiple target beliefs (j) simultaneously.

        Args:
            j_list: List of target belief indices in Π_n_M
            i_list: List of source belief indices in Π_n_M
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

        # Access all target beliefs from codebook
        n_targets = len(j_list)
        π_targets_flat = np.array([self.BQ.Π_n_M[j] for j in j_list])  # (n_targets, N_n)
        π_targets_2d = np.array([self.unflatten_belief(π_flat) for π_flat in π_targets_flat])  # (n_targets, m_n, len_M)

        # Vectorized distance computation for all (source, target, observation) combinations:
        # π_all_batch: (n_sources, m_y, m_n, len_M)
        # π_targets_2d: (n_targets, m_n, len_M)
        # We want distances: (n_sources, n_targets, m_y)
        π_all_flat = π_all_batch.reshape(n_sources, m_y, -1)  # (n_sources, m_y, m_n * len_M)
        π_targets_flat_2d = π_targets_2d.reshape(n_targets, -1)  # (n_targets, m_n * len_M)

        # Compute pairwise distances: (n_sources, n_targets, m_y)
        # Broadcasting: (n_sources, 1, m_y, m_n*len_M) - (1, n_targets, 1, m_n*len_M) -> (n_sources, n_targets, m_y, m_n*len_M)
        π_diff = π_all_flat[:, np.newaxis, :, :] - π_targets_flat_2d[np.newaxis,
                                                                     :, np.newaxis, :]  # (n_sources, n_targets, m_y, m_n*len_M)
        distances = np.linalg.norm(π_diff, axis=3)  # (n_sources, n_targets, m_y)

        # Vectorized matching: mask observations where distance < threshold
        threshold = 1e-3
        matches = (distances < threshold) & (H_y_batch[:, np.newaxis, :] > 0.0)  # (n_sources, n_targets, m_y)

        # Sum H_y for matching observations for each (source, target) pair
        H_y_broadcast = H_y_batch[:, np.newaxis, :]  # (n_sources, 1, m_y)
        probs = np.sum(H_y_broadcast * matches, axis=2)  # (n_sources, n_targets)

        return probs.astype(np.float32)

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
                            row_data_arr = row_data_arr.astype(np.float16)

                            # Append to lists
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
                # Convert to backend arrays (use int32 for indices, float16 for data)
                rows_arr = np.array(rows_list, dtype=np.int32)
                cols_arr = np.array(cols_list, dtype=np.int32)
                data_arr = np.array(data_list, dtype=np.float16)

                # Create COO matrix
                coo_mat = coo_matrix((data_arr, (rows_arr, cols_arr)),
                                     shape=(cardinality, cardinality),
                                     dtype=np.float16)

                # Convert to CSR for efficient operations
                csr_mat = coo_mat.tocsr()
            else:
                # Empty matrix - create zero CSR matrix
                csr_mat = csr_matrix((cardinality, cardinality), dtype=np.float16)

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
            i: Index of quantized belief in Π_n_M
            k: Index of action in U_n
        Returns:
            Cost value (float)
        """
        return float(self.c_n_M[i, k])

    def _get_c_n_M_cache_path(self) -> Path:
        """Generate cache file path for c_n_M based on quantization parameters."""
        cache_dir = Path(__file__).parent.parent.parent / "cache/SLAM/cost_slam"
        cache_dir.mkdir(parents=True, exist_ok=True)

        state_bounds_flat = self.state_bounds.flatten().tolist()

        metadata = {
            'M': self.M,
            'n': self.n,
            'state_bounds': state_bounds_flat,
            'max_val': self.motion_model.max_v if hasattr(self.motion_model, 'max_v') else self.motion_model.max_a,
            'dt': float(self.motion_model.dt),
            'map_shape': (self.map_H, self.map_W),
            'sigma_w': float(self.σ_w),
            'sigma_v': float(self.σ_v),
            'model': self.motion_model.__class__.__name__,
            'state_dim': self.state_dim,
            'cardinality': self.BQ.cardinality,
            'N_n': self.N_n,
            'm_n': self.SQ.m_n,
            'n_u': self.AQ.n_u,
            'exploration_type': self.exploration_type,
            'kernel_version': 'v1-dense-float32',
            'problem_type': 'slam',
        }

        metadata_str = str(sorted(metadata.items()))
        metadata_hash = hashlib.md5(metadata_str.encode()).hexdigest()[:8]

        max_val = self.motion_model.max_v if hasattr(self.motion_model, 'max_v') else self.motion_model.max_a
        filename = f"cost_slam_M{self.M}_n{self.n}_map{self.map_H}x{self.map_W}_max{max_val}_{metadata_hash}.npz"

        return cache_dir / filename

    def _compute_c_n_M(self, show_progress: bool = True) -> np.ndarray:
        """
        Precompute costs for all belief-action combinations using batched ρ_n.

        Returns:
            c_n_M: Array of shape (cardinality, n_u) where c_n_M[i, k] = ρ_n(π_i^M, u_k)
        """
        cardinality = self.BQ.cardinality
        n_u = self.AQ.n_u
        Π_n_M = self.BQ.Π_n_M  # (cardinality, N_n)
        U_n = self.AQ.U  # (n_u, 2)

        if show_progress:
            print(f"  Computing costs for {cardinality:,} beliefs × {n_u} actions...")

        # Convert flattened beliefs to 2D format for ρ_n
        # Π_n_M: (cardinality, N_n) where N_n = m_n * len_M
        # Need to reshape to (cardinality, m_n, len_M)
        Π_n_M_2d = np.array([self.unflatten_belief(π_flat) for π_flat in Π_n_M])  # (cardinality, m_n, len_M)

        # Use batched ρ_n to compute all costs at once
        # ρ_n expects (n_π, m_n, len_M) and (n_u, 2), returns (n_π, n_u)
        c_n_M = self.ρ_n(Π_n_M_2d, U_n)  # (cardinality, n_u)

        return c_n_M

    def _save_c_n_M(self, cache_path: Path, c_n_M: np.ndarray) -> None:
        """Save c_n_M to cache file with metadata."""
        state_bounds = self.state_bounds.astype(float)
        state_bounds_save = state_bounds.get() if hasattr(state_bounds, 'get') else state_bounds

        metadata = {
            'M': self.M,
            'n': self.n,
            'state_bounds': state_bounds_save,
            'max_val': self.motion_model.max_v if hasattr(self.motion_model, 'max_v') else self.motion_model.max_a,
            'dt': float(self.motion_model.dt),
            'map_shape': _numpy.array([self.map_H, self.map_W]),
            'sigma_w': float(self.σ_w),
            'sigma_v': float(self.σ_v),
            'm_n': self.SQ.m_n,
            'n_u': self.AQ.n_u,
            'cardinality': self.BQ.cardinality,
            'N_n': self.N_n,
            'model': self.motion_model.__class__.__name__,
            'state_dim': self.state_dim,
            'exploration_type': self.exploration_type,
            'kernel_version': 'v1-dense-float32',
            'problem_type': 'slam',
        }

        # Convert to numpy if CuPy array
        if hasattr(c_n_M, 'get'):
            c_n_M_np = c_n_M.get()
        else:
            c_n_M_np = c_n_M

        save_dict = metadata.copy()
        save_dict['c_n_M'] = c_n_M_np.astype(_numpy.float32)

        _numpy.savez_compressed(cache_path, **save_dict)

    def _load_c_n_M(self, cache_path: Path) -> bool:
        """Load c_n_M from cache file if it exists and matches current parameters."""
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
                'M': self.M,
                'n': self.n,
                'state_bounds': state_bounds_np,
                'max_val': max_val,
                'dt': float(self.motion_model.dt),
                'map_shape': _numpy.array([self.map_H, self.map_W]),
                'sigma_w': float(self.σ_w),
                'sigma_v': float(self.σ_v),
                'm_n': self.SQ.m_n,
                'n_u': self.AQ.n_u,
                'cardinality': self.BQ.cardinality,
                'N_n': self.N_n,
                'model': self.motion_model.__class__.__name__,
                'state_dim': self.state_dim,
                'exploration_type': self.exploration_type,
                'kernel_version': 'v1-dense-float32',
                'problem_type': 'slam',
            }

            for key, expected_value in expected_metadata.items():
                if key not in data:
                    print(f"  Cache mismatch: key '{key}' not found in cache")
                    return False
                data_value = data[key]
                if isinstance(expected_value, _numpy.ndarray):
                    if not _numpy.array_equal(data_value, expected_value):
                        print(f"  Cache mismatch: '{key}' array values differ")
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

            self._c_n_M_matrix = np.asarray(data['c_n_M'])
            expected_shape = (self.BQ.cardinality, self.AQ.n_u)
            if self._c_n_M_matrix.shape != expected_shape:
                print(f"  Cache mismatch: c_n_M shape differs")
                print(f"    Expected: {expected_shape}")
                print(f"    Cached: {self._c_n_M_matrix.shape}")
                return False

            print(f"  ✓ Cache metadata matches, loading c_n_M")
            return True

        except Exception as e:
            print(f"Error loading c_n_M cache: {e}")
            import traceback
            traceback.print_exc()
            return False


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
                 sigma_w: float = 0.01, sigma_v: float = 1.0):
        super().__init__(n, motion_model, measurement_model, obstacles, _map, sigma_w, sigma_v)
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
            j_batch_size = getattr(self, '_test_j_batch_size', 100)
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
        return self._get_cache_path('p_n_M_localization', 'v2-sparse-float16')

    def _save_p_n_M(self, cache_path: Path, p_n_M: list) -> None:
        self._save_sparse_p_n_M(cache_path, p_n_M, 'v2-sparse-float16')

    def _load_p_n_M(self, cache_path: Path) -> bool:
        return self._load_sparse_p_n_M(cache_path, 'v2-sparse-float16')

    def _get_c_n_M_cache_path(self) -> Path:
        return self._get_cache_path('c_n_M_localization', 'v1-dense-float32')

    def _save_c_n_M(self, cache_path: Path, c_n_M: np.ndarray) -> None:
        self._save_dense_c_n_M(cache_path, c_n_M, 'v1-dense-float32')

    def _load_c_n_M(self, cache_path: Path) -> bool:
        return self._load_dense_c_n_M(cache_path, 'v1-dense-float32', (self.BQ.cardinality, self.AQ.n_u))

    def η_n(self, j: int, i: int, u: np.ndarray) -> float:
        """
        Overloaded η_n that takes belief indices instead of belief vectors.

        Now delegates to the finite-observation η_n implementation in BeliefMDP_n_Localization,
        using quantized beliefs from the codebook Π_n_M.

        Args:
            j: Index of target belief in Π_n_M (π_j^M)
            i: Index of current belief in Π_n_M (π_i^M)
            u: Action (2,)
        Returns:
            float: Probability of transitioning from π_i^M to π_j^M under action u
        """
        # Access beliefs directly from cached codebook
        π_new_1d = self.BQ.Π_n_M[j]  # (m_n,)
        π_1d = self.BQ.Π_n_M[i]      # (m_n,)

        # Delegate to parent (finite-sum) implementation
        return super().η_n(π_new_1d, π_1d, u)

    def η_n_batch(self, j_list: list[int], i: int, u: np.ndarray) -> np.ndarray:
        """
        Fully vectorized batched version of η_n over multiple target belief indices for localization.

        Computes probabilities for all target beliefs simultaneously by:
        1. Computing H_y and π_all_batch (all updated beliefs for all observations) once
        2. Vectorized distance computation between all target beliefs and all updated beliefs
        3. Summing H_y for matching observations for each target belief

        Args:
            j_list: List of target belief indices in Π_n_M
            i: Index of current belief in Π_n_M
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

        # Access all target beliefs from codebook
        n_targets = len(j_list)
        π_targets_1d = np.array([self.BQ.Π_n_M[j] for j in j_list])  # (n_targets, m_n)

        # Vectorized distance computation:
        # π_all_batch: (m_y, m_n)
        # π_targets_1d: (n_targets, m_n)
        # Compute pairwise distances: (n_targets, m_y)
        π_diff = π_targets_1d[:, np.newaxis, :] - π_all_batch[np.newaxis, :, :]  # (n_targets, m_y, m_n)
        distances = np.linalg.norm(π_diff, axis=2)  # (n_targets, m_y)

        # Vectorized matching: mask observations where distance < threshold for each target
        threshold = 1e-3
        matches = (distances < threshold) & (H_y[np.newaxis, :] > 0.0)  # (n_targets, m_y)

        # Sum H_y for matching observations for each target belief
        H_y_broadcast = H_y[np.newaxis, :]  # (1, m_y)
        probs = np.sum(H_y_broadcast * matches, axis=1)  # (n_targets,)

        return probs.astype(np.float32)

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
                        row_data_arr = row_data_arr.astype(np.float16)

                        rows_list.extend([i] * len(row_cols))
                        cols_list.extend(row_cols)
                        data_list.extend(row_data_arr.tolist())

            if rows_list:
                rows_arr = np.array(rows_list, dtype=np.int32)
                cols_arr = np.array(cols_list, dtype=np.int32)
                data_arr = np.array(data_list, dtype=np.float16)

                coo_mat = coo_matrix((data_arr, (rows_arr, cols_arr)),
                                     shape=(cardinality, cardinality),
                                     dtype=np.float16)

                csr_mat = coo_mat.tocsr()
            else:
                csr_mat = csr_matrix((cardinality, cardinality), dtype=np.float16)

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
            Π_n_M: Array of shape (cardinality, m_n) containing all quantized beliefs
        """
        return self.BQ.Π_n_M

    def _get_c_n_M_cache_path(self) -> Path:
        """Generate cache file path for c_n_M based on quantization parameters."""
        cache_dir = Path(__file__).parent.parent.parent / "cache/c_n_M_localization"
        cache_dir.mkdir(parents=True, exist_ok=True)

        state_bounds_flat = self.state_bounds.flatten().tolist()

        metadata = {
            'M': self.M,
            'n': self.n,
            'state_bounds': state_bounds_flat,
            'max_val': self.motion_model.max_v if hasattr(self.motion_model, 'max_v') else self.motion_model.max_a,
            'dt': float(self.motion_model.dt),
            'map_shape': (self.map_H, self.map_W),
            'sigma_w': float(self.σ_w),
            'sigma_v': float(self.σ_v),
            'model': self.motion_model.__class__.__name__,
            'state_dim': self.state_dim,
            'cardinality': self.BQ.cardinality,
            'N_n': self.SQ.m_n,
            'n_u': self.AQ.n_u,
            'kernel_version': 'v1-dense-float32',
            'problem_type': 'localization',
        }

        metadata_str = str(sorted(metadata.items()))
        metadata_hash = hashlib.md5(metadata_str.encode()).hexdigest()[:8]

        max_val = self.motion_model.max_v if hasattr(self.motion_model, 'max_v') else self.motion_model.max_a
        filename = f"c_n_M_localization_M{self.M}_n{self.n}_map{self.map_H}x{self.map_W}_max{max_val}_{metadata_hash}.npz"

        return cache_dir / filename

    def _compute_c_n_M(self, show_progress: bool = True) -> np.ndarray:
        """
        Precompute costs for all belief-action pairs.

        Returns:
            c_n_M: Array of shape (cardinality, n_u) where c_n_M[i, k] = c_n_M(π_i^M, u_k)
        """
        cardinality = self.BQ.cardinality
        n_u = self.AQ.n_u
        Π_n = self.BQ.Π_n_M  # (cardinality, m_n)
        U_n = self.AQ.U  # (n_u, 2)

        if show_progress:
            print(f"  Computing costs for {cardinality:,} beliefs × {n_u} actions...")

        # Use vectorized batch computation
        c_n_M = self.c_tilde_n_vectorized_batch(Π_n, U_n)  # (cardinality, n_u)

        return c_n_M

    def _save_c_n_M(self, cache_path: Path, c_n_M: np.ndarray) -> None:
        """Save c_n_M to cache file with metadata."""
        state_bounds = self.state_bounds.astype(float)
        state_bounds_save = state_bounds.get() if hasattr(state_bounds, 'get') else state_bounds

        metadata = {
            'M': self.M,
            'n': self.n,
            'state_bounds': state_bounds_save,
            'max_val': self.motion_model.max_v if hasattr(self.motion_model, 'max_v') else self.motion_model.max_a,
            'dt': float(self.motion_model.dt),
            'map_shape': _numpy.array([self.map_H, self.map_W]),
            'sigma_w': float(self.σ_w),
            'sigma_v': float(self.σ_v),
            'm_n': self.SQ.m_n,
            'n_u': self.AQ.n_u,
            'cardinality': self.BQ.cardinality,
            'N_n': self.SQ.m_n,
            'model': self.motion_model.__class__.__name__,
            'state_dim': self.state_dim,
            'kernel_version': 'v1-dense-float32',
            'problem_type': 'localization',
        }

        # Convert to numpy if CuPy array
        if hasattr(c_n_M, 'get'):
            c_n_M_np = c_n_M.get()
        else:
            c_n_M_np = c_n_M

        save_dict = metadata.copy()
        save_dict['c_n_M'] = c_n_M_np.astype(_numpy.float32)

        _numpy.savez_compressed(cache_path, **save_dict)

    def _load_c_n_M(self, cache_path: Path) -> bool:
        """Load c_n_M from cache file if it exists and matches current parameters."""
        if not cache_path.exists():
            return False

        try:
            data = _numpy.load(str(cache_path))

            state_bounds = self.state_bounds.astype(float)
            state_bounds_np = state_bounds.get() if hasattr(state_bounds, 'get') else state_bounds

            expected_metadata = {
                'M': self.M,
                'n': self.n,
                'state_bounds': state_bounds_np,
                'max_val': self.motion_model.max_v if hasattr(self.motion_model, 'max_v') else self.motion_model.max_a,
                'dt': float(self.motion_model.dt),
                'map_shape': _numpy.array([self.map_H, self.map_W]),
                'sigma_w': float(self.σ_w),
                'sigma_v': float(self.σ_v),
                'm_n': self.SQ.m_n,
                'n_u': self.AQ.n_u,
                'cardinality': self.BQ.cardinality,
                'N_n': self.SQ.m_n,
                'model': self.motion_model.__class__.__name__,
                'state_dim': self.state_dim,
                'problem_type': 'localization',
            }

            kernel_version = data.get('kernel_version', 'v1-dense-float32')
            if kernel_version != 'v1-dense-float32':
                return False

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

            c_n_M = data['c_n_M'].astype(_numpy.float32)
            expected_shape = (self.BQ.cardinality, self.AQ.n_u)

            if c_n_M.shape != expected_shape:
                return False

            # Convert to backend array (CuPy if available)
            if is_cupy:
                import cupy as cp
                c_n_M = cp.asarray(c_n_M)

            self.c_n_M = c_n_M
            return True

        except Exception as e:
            print(f"Error loading c_n_M cache: {e}")
            import traceback
            traceback.print_exc()
            return False


class BeliefMDP_n_M_Mapping(BeliefMDP_n_Mapping, BaseBeliefMDP_n_M):
    """Belief-MDP_n_M for Mapping: quantized belief space over maps only."""

    @property
    def problem_type(self) -> str:
        return 'mapping'

    @property
    def N_n(self) -> int:
        return self.len_M

    def __init__(self, M: int, β: float, n: int, motion_model: SingleIntegratorModel | DoubleIntegratorModel,
                 measurement_model: LIDAR, obstacles: list[Obstacle], _map: LidarGridMapVec,
                 sigma_w: float = 0.01, sigma_v: float = 0.01, j_batch_size: int = 136, i_batch_size: int = 10):
        super().__init__(n, motion_model, measurement_model, obstacles, _map, sigma_w, sigma_v)
        self.M = M
        self.β = β
        self.j_batch_size = j_batch_size
        self.i_batch_size = i_batch_size
        self.BQ = BeliefQuantizer(M, self.N_n)

        skip_computation = self._class_skip_p_n_M_computation or getattr(self, '_skip_p_n_M_computation', False)
        cache_path = self._get_p_n_M_cache_path()
        print(f"Looking for p_n_M cache at: {cache_path}")
        print(f"Cache file exists: {cache_path.exists()}")
        if self._load_p_n_M(cache_path):
            print(f"Loaded cached p_n_M from {cache_path}")
        elif skip_computation:
            print("⚠ Skipping p_n_M computation (testing mode)")
            self.p_n_M = None
        else:
            print("Computing p_n_M for the first time...")
            print(f"This will compute: {self.AQ.n_u} actions × {self.BQ.cardinality}² belief-to-belief transitions")
            print(f"  Note: For mapping, η_n depends on (x_current, u, x_next), so we marginalize over state transitions")
            print(f"  (sampling {getattr(self, '_test_state_samples', 100)} state pairs per belief-action for averaging)")
            self.p_n_M = self._compute_p_n_M(j_batch_size=self.j_batch_size, i_batch_size=self.i_batch_size)
            self._save_p_n_M(cache_path, self.p_n_M)
            print(f"Saved p_n_M to {cache_path}")

        c_n_M_cache_path = self._get_c_n_M_cache_path()
        if self._load_c_n_M(c_n_M_cache_path):
            print(f"Loaded cached c_n_M from {c_n_M_cache_path}")
        else:
            print("Computing c_n_M for the first time...")
            print(
                f"This will compute costs for {self.BQ.cardinality} beliefs × {self.SQ.m_n} states × {self.AQ.n_u} actions")
            self.c_n_M = self._compute_c_n_M()
            self._save_c_n_M(c_n_M_cache_path, self.c_n_M)
            print(f"Saved c_n_M to {c_n_M_cache_path}")

    def _get_p_n_M_cache_path(self) -> Path:
        return BaseBeliefMDP_n_M._get_cache_path(self, 'MAP/p_n_M_mapping', 'v3-dense-5d-mapping')

    def _save_p_n_M(self, cache_path: Path, p_n_M: np.ndarray) -> None:
        """Save dense 5D p_n_M tensor to cache."""
        metadata = self._get_base_metadata()
        metadata['kernel_version'] = 'v3-dense-5d-mapping'
        metadata['tensor_format'] = 'dense_5d'

        if hasattr(p_n_M, 'get'):
            p_n_M_np = p_n_M.get()
        else:
            p_n_M_np = p_n_M

        save_dict = metadata.copy()
        save_dict['p_n_M'] = p_n_M_np.astype(_numpy.float16)
        _numpy.savez_compressed(cache_path, **save_dict)

    def _load_p_n_M(self, cache_path: Path) -> bool:
        """Load dense 5D p_n_M tensor from cache."""
        if not cache_path.exists():
            print(f"Cache file does not exist: {cache_path}")
            # Try to find any matching cache files and attempt to load them
            cache_dir = cache_path.parent
            if cache_dir.exists():
                max_val = self.motion_model.max_v if hasattr(self.motion_model, 'max_v') else self.motion_model.max_a
                pattern = f"p_n_M_mapping_M{self.M}_n{self.n}_map{self.map_H}x{self.map_W}_max{max_val}_*.npz"
                matching_files = list(cache_dir.glob(pattern))
                if matching_files:
                    print(f"Found {len(matching_files)} potential cache files, trying to load...")
                    for f in matching_files:
                        print(f"  Trying: {f.name}")
                        if self._try_load_p_n_M_file(f):
                            print(f"  ✓ Successfully loaded: {f.name}")
                            return True
                        else:
                            print(f"  ✗ Failed to load: {f.name}")
            return False

        return self._try_load_p_n_M_file(cache_path)

    def _try_load_p_n_M_file(self, cache_path: Path) -> bool:
        """Try to load p_n_M from a specific cache file."""
        if not cache_path.exists():
            return False

        try:
            data = _numpy.load(str(cache_path))

            if data.get('kernel_version') != 'v3-dense-5d-mapping':
                return False

            state_bounds = self.state_bounds.astype(float)
            state_bounds_np = state_bounds.get() if hasattr(state_bounds, 'get') else state_bounds

            expected_metadata = self._get_base_metadata()
            expected_metadata['state_bounds'] = state_bounds_np
            expected_metadata['map_shape'] = _numpy.array([self.map_H, self.map_W])

            if not self._verify_metadata(data, expected_metadata):
                return False

            p_n_M = data['p_n_M'].astype(_numpy.float32)
            expected_shape = (self.SQ.m_n, self.BQ.cardinality, self.SQ.m_n, self.BQ.cardinality, self.AQ.n_u)

            if p_n_M.shape != expected_shape:
                return False

            if is_cupy:
                import cupy as cp
                p_n_M = cp.asarray(p_n_M)

            self.p_n_M = p_n_M
            return True
        except Exception as e:
            return False

    def _save_c_n_M(self, cache_path: Path, c_n_M: np.ndarray) -> None:
        self._save_dense_c_n_M(cache_path, c_n_M, 'v1-dense-float32')

    def _load_c_n_M(self, cache_path: Path) -> bool:
        return self._load_dense_c_n_M(cache_path, 'v1-dense-float32', (self.BQ.cardinality, self.SQ.m_n, self.AQ.n_u))

    def η_n(self, j: int, i: int, x_current: np.ndarray, u: np.ndarray, x_next: np.ndarray) -> float:
        """
        Overloaded η_n that takes belief indices instead of belief vectors for mapping.

        Now delegates to the finite-observation η_n implementation in BeliefMDP_n_Mapping,
        using quantized beliefs from the codebook Π_n_M.

        Args:
            j: Index of target belief in Π_n_M (i.e., π_j^M = self.BQ.Π_n_M[j])
            i: Index of current belief in Π_n_M (i.e., π_i^M = self.BQ.Π_n_M[i])
            x_current: current state (state_dim,) - part of augmented state b_t = (x_t, π_t)
            u: Action (2,)
            x_next: next state (state_dim,) - given as input, not sampled
        Returns:
            float: Joint probability of (π_j^M, x_next) from augmented state (x_current, π_i^M) under action u
        """
        # Access beliefs directly from cached codebook
        π_new_flat = self.BQ.Π_n_M[j]  # Target belief (len_M,)
        π_flat = self.BQ.Π_n_M[i]      # Current belief (len_M,)

        # For mapping, beliefs are already 1D (no unflatten needed)
        π_new_1d = π_new_flat
        π_1d = π_flat

        # Delegate to parent (finite-sum) implementation
        return super().η_n(π_new_1d, π_1d, x_current, u, x_next)

    def η_n_batch(self, j_list: list[int], i: int, x_current: np.ndarray, u: np.ndarray, x_next: np.ndarray) -> np.ndarray:
        """
        Fully vectorized batched version of η_n for mapping.

        Computes joint probabilities η_n(π_j^M, x_next | π_i^M, x_current, u) for all target beliefs simultaneously
        using discrete observation quantization.

        Args:
            j_list: List of target belief indices in Π_n_M
            i: Index of current belief in Π_n_M
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

        # Access all target beliefs from codebook
        n_targets = len(j_list)
        π_targets_1d = np.array([self.BQ.Π_n_M[j] for j in j_list])  # (n_targets, len_M)

        # Vectorized distance computation:
        # π_all_batch: (m_y, len_M)
        # π_targets_1d: (n_targets, len_M)
        # Compute pairwise distances: (n_targets, m_y)
        π_diff = π_targets_1d[:, np.newaxis, :] - π_all_batch[np.newaxis, :, :]  # (n_targets, m_y, len_M)
        distances = np.linalg.norm(π_diff, axis=2)  # (n_targets, m_y)

        # Vectorized matching: mask observations where distance < threshold for each target
        threshold = 1e-3
        matches = (distances < threshold) & (H_y[np.newaxis, :] > 0.0)  # (n_targets, m_y)

        # Sum H_y for matching observations for each target belief
        H_y_broadcast = H_y[np.newaxis, :]  # (1, m_y)
        prob_belief = np.sum(H_y_broadcast * matches, axis=1)  # (n_targets,)

        # Return joint probability = T(x_next | x_current, u) * P(π' ≈ π_j | x_next)
        probabilities = prob_belief * T_x_next_given_x_current_u

        return probabilities.astype(np.float32)

    def P_batch(self, j_list: list[int], i: int, x_next: np.ndarray,
                n_samples: int = 24000, seed: int = None, mc_batch_size: int = 24000, show_progress: bool = True) -> np.ndarray:
        """
        Compute P(π_j | x_next, π_i) without the T term for efficient matrix multiplication.

        This is the MC-integrated probability: P(π_j | x_next, π_i) = ∫_Y δ_{F(π_i, x_next, y)}(π_j) * H(dy | π_i, x_next)

        Args:
            j_list: List of target belief indices in Π_n_M
            i: Index of current belief in Π_n_M
            x_next: next state (state_dim,)
            n_samples: Number of Monte Carlo samples
            seed: Random seed
            mc_batch_size: Batch size for MC sampling
            show_progress: Whether to show progress
        Returns:
            np.ndarray: Array of probabilities P(π_j | x_next, π_i) for each j in j_list, shape (len(j_list),)
        """
        if seed is not None:
            random.seed(seed)

        if len(j_list) == 0:
            return np.array([])

        # Access current belief from codebook
        π_flat = self.BQ.Π_n_M[i]  # Current belief (len_M,)
        π_1d = π_flat

        # Access all target beliefs from codebook
        n_targets = len(j_list)
        π_targets_list = []
        for j in j_list:
            π_flat = self.BQ.Π_n_M[j]
            if hasattr(π_flat, 'get'):
                π_targets_list.append(π_flat)
            else:
                π_targets_list.append(np.asarray(π_flat))
        π_targets_1d = np.stack(π_targets_list)  # (n_targets, len_M)

        # Initialize counts for each target
        counts = np.zeros(n_targets, dtype=np.int32)

        # Process Monte Carlo samples in batches
        n_mc_batches = (n_samples + mc_batch_size - 1) // mc_batch_size
        batch_iter = tqdm(range(n_mc_batches), desc=f"P_batch (n={n_samples}, targets={n_targets}, mc_batch={mc_batch_size})",
                          unit="batch", disable=not show_progress) if show_progress else range(n_mc_batches)

        for batch_idx in batch_iter:
            current_mc_batch_size = min(mc_batch_size, n_samples - batch_idx * mc_batch_size)

            # Batch sample observations (shared across all j targets)
            Y_batch = self.sample_observations_batch(π_1d, x_next, current_mc_batch_size)  # (current_mc_batch_size, B)

            # Batch update beliefs using F_batch_log (computed once, shared across all j)
            π_sampled_batch = self.F_batch_log(π_1d, x_next, Y_batch)  # (current_mc_batch_size, len_M)

            # Compute distances to all target beliefs simultaneously
            π_sampled_expanded = π_sampled_batch[:, np.newaxis, :]  # (current_mc_batch_size, 1, len_M)
            π_targets_expanded = π_targets_1d[np.newaxis, :, :]  # (1, n_targets, len_M)

            # Compute L2 distances: (current_mc_batch_size, n_targets)
            distances = np.linalg.norm(π_sampled_expanded - π_targets_expanded, axis=2)

            # Count matches for each target
            matches = distances < 1e-3  # (current_mc_batch_size, n_targets)

            # Sum matches directly - keep on same backend as counts
            batch_counts = np.sum(matches, axis=0)  # (n_targets,)

            # Ensure batch_counts matches the backend of counts (avoid unnecessary transfers)
            if hasattr(counts, 'get') and not hasattr(batch_counts, 'get'):
                # counts is CuPy, batch_counts is NumPy - convert batch_counts to CuPy
                batch_counts = np.asarray(batch_counts, dtype=np.int32)
            elif not hasattr(counts, 'get') and hasattr(batch_counts, 'get'):
                # counts is NumPy, batch_counts is CuPy - convert batch_counts to NumPy
                # But try to avoid this by keeping counts on GPU if possible
                batch_counts = batch_counts.get().astype(np.int32)
            else:
                # Same backend - just ensure dtype
                batch_counts = batch_counts.astype(np.int32)

            counts += batch_counts

        # Convert counts to probabilities (without T term)
        # Keep everything on GPU - no CPU transfer here
        counts_array = counts.astype(np.float32)
        probabilities = counts_array / n_samples

        # Return GPU array (or NumPy if not using CuPy)
        return probabilities

    def P_batch_i_j(self, i_list: list[int], x_next: np.ndarray,
                    n_samples: int = 24000, seed: int = None, mc_batch_size: int = 24000,
                    show_progress: bool = True) -> np.ndarray:
        """
        Compute P(π_j | x_next, π_i) for multiple i values and ALL j targets simultaneously.

        This batches across i values (current beliefs) and computes distances to ALL j targets
        in parallel. Much more efficient than calling P_batch repeatedly.

        Args:
            i_list: List of current belief indices in Π_n_M
            x_next: next state (state_dim,)
            n_samples: Number of Monte Carlo samples per i
            seed: Random seed
            mc_batch_size: Batch size for MC sampling
            show_progress: Whether to show progress
        Returns:
            np.ndarray: Array of probabilities, shape (len(i_list), cardinality)
                       P_batch_i_j[k, j] = P(π_j | x_next, π_i_list[k])
        """
        if seed is not None:
            random.seed(seed)

        if len(i_list) == 0:
            return np.array([])

        cardinality = self.BQ.cardinality
        n_i = len(i_list)

        # Access all current beliefs from codebook
        π_current_list = []
        for i in i_list:
            π_flat = self.BQ.Π_n_M[i]
            if hasattr(π_flat, 'get'):
                π_current_list.append(π_flat)
            else:
                π_current_list.append(np.asarray(π_flat))
        π_current_batch = np.stack(π_current_list)  # (n_i, len_M)

        # Access ALL target beliefs from codebook (all j targets)
        π_targets_list = []
        for j in range(cardinality):
            π_flat = self.BQ.Π_n_M[j]
            if hasattr(π_flat, 'get'):
                π_targets_list.append(π_flat)
            else:
                π_targets_list.append(np.asarray(π_flat))
        π_targets_all = np.stack(π_targets_list)  # (cardinality, len_M)

        # Initialize counts: (n_i, cardinality) - one count per (i, j) pair
        counts = np.zeros((n_i, cardinality), dtype=np.int32)

        # Process Monte Carlo samples in batches
        n_mc_batches = (n_samples + mc_batch_size - 1) // mc_batch_size
        batch_iter = tqdm(range(n_mc_batches),
                          desc=f"P_batch_i_j (n={n_samples}, i_batch={n_i}, j_all={cardinality}, mc_batch={mc_batch_size})",
                          unit="batch", disable=not show_progress) if show_progress else range(n_mc_batches)

        for batch_idx in batch_iter:
            current_mc_batch_size = min(mc_batch_size, n_samples - batch_idx * mc_batch_size)

            # For each i, sample observations and update beliefs
            # We need to process each i separately since sample_observations_batch takes one π
            # But we can vectorize the distance computation across all i and j

            # Process each i in the batch
            π_sampled_all_i = []  # Will be list of (current_mc_batch_size, len_M) arrays

            for i_idx, i in enumerate(i_list):
                π_1d = π_current_batch[i_idx]

                # Sample observations for this i
                Y_batch = self.sample_observations_batch(π_1d, x_next, current_mc_batch_size)

                # Update beliefs
                π_sampled_batch = self.F_batch_log(π_1d, x_next, Y_batch)  # (current_mc_batch_size, len_M)
                π_sampled_all_i.append(π_sampled_batch)

            # Stack all sampled beliefs: (n_i, current_mc_batch_size, len_M)
            π_sampled_stack = np.stack(π_sampled_all_i)  # (n_i, current_mc_batch_size, len_M)

            # Reshape for broadcasting: (n_i, current_mc_batch_size, 1, len_M)
            π_sampled_expanded = π_sampled_stack[:, :, np.newaxis, :]  # (n_i, current_mc_batch_size, 1, len_M)

            # Reshape targets: (1, 1, cardinality, len_M)
            π_targets_expanded = π_targets_all[np.newaxis, np.newaxis, :, :]  # (1, 1, cardinality, len_M)

            # Compute L2 distances: (n_i, current_mc_batch_size, cardinality)
            distances = np.linalg.norm(π_sampled_expanded - π_targets_expanded, axis=3)

            # Count matches: (n_i, current_mc_batch_size, cardinality)
            matches = distances < 1e-3  # (n_i, current_mc_batch_size, cardinality)

            # Sum across MC samples: (n_i, cardinality)
            batch_counts = np.sum(matches, axis=1)  # (n_i, cardinality)

            # Ensure correct backend
            if hasattr(counts, 'get') and not hasattr(batch_counts, 'get'):
                batch_counts = np.asarray(batch_counts, dtype=np.int32)
            elif not hasattr(counts, 'get') and hasattr(batch_counts, 'get'):
                batch_counts = batch_counts.get().astype(np.int32)
            else:
                batch_counts = batch_counts.astype(np.int32)

            counts += batch_counts

        # Convert counts to probabilities
        counts_array = counts.astype(np.float32)
        probabilities = counts_array / n_samples  # (n_i, cardinality)

        return probabilities

    def _compute_p_n_M(self, n_samples: int = 40000, batch_size: int = 40000, j_batch_size: int = None,
                       i_batch_size: int = None, show_progress: bool = True, threshold: float = 1e-8, ) -> list:
        """
        Compute the full transition probability tensor p_n^{(M)} for mapping.

        Mathematical formulation:
        p_n_M(x_{t+1}, π_{t+1} | x_t, π_t, u_t) = T(x_{t+1} | x_t, u_t) * P(π_{t+1} | x_{t+1}, π_t)

        where the augmented state is b_t = (x_t, π_t).

        Shape: (m_n, cardinality, m_n, cardinality, n_u)
        - p_n_M[x_next_idx, π_next_idx, x_current_idx, π_current_idx, u_idx]

        Optimized implementation:
        1. First compute P(π_j | x_next, π_i) for all (j, i, x_next) via MC - shape (cardinality, cardinality, m_n)
        2. Then multiply with T_mat: p_n_M[x_next, j, x_current, i, u] = T[x_next, x_current, u] * P[j, i, x_next]
        3. Normalize: for each (x_current, π_current, u), sum over (x_next, π_next) should equal 1

        Returns:
            5D numpy array with shape (m_n, cardinality, m_n, cardinality, n_u)
            representing p_n_M[x_next, π_next, x_current, π_current, u]
        """
        # Use instance j_batch_size if not provided
        if j_batch_size is None:
            j_batch_size = getattr(self, 'j_batch_size', 100)

        cardinality = self.BQ.cardinality
        n_u = self.AQ.n_u
        m_n = self.SQ.m_n

        if show_progress:
            print(f"Computing p_n_M for mapping...")
            print(f"  Step 1: Computing P(π_j | x_next, π_i) for all beliefs and states")
            print(
                f"    This will compute {cardinality}×{cardinality}×{m_n} = {cardinality*cardinality*m_n:,} probabilities")

        # Step 1: Compute P(π_j | x_next, π_i) for all (j, i, x_next)
        # Shape: (cardinality, cardinality, m_n) where P_mat[j, i, x_next_idx] = P(π_j | x_next, π_i)
        # Keep everything on GPU - no CPU transfers until save
        P_mat = np.zeros((cardinality, cardinality, m_n), dtype=np.float32)

        # Choose computation strategy based on j_batch_size:
        # - If j_batch_size >= cardinality: compute ALL j targets at once (optimized path)
        # - If j_batch_size < cardinality: batch j targets (memory-efficient path for large cardinality)
        use_all_j_at_once = (j_batch_size >= cardinality)

        if use_all_j_at_once:
            # Optimized path: compute all j targets at once, batch across i
            if i_batch_size is None:
                i_batch_size = getattr(self, 'i_batch_size', 10)  # Process 10 i values in parallel

            if show_progress:
                print(
                    f"  Using optimized path: all {cardinality} j targets at once, batching i (i_batch_size={i_batch_size})")

            # Process in batches over i and x_next
            for x_next_idx in tqdm(range(m_n), desc="Computing P matrix (x_next)", disable=not show_progress):
                x_next = self.SQ.X_n[x_next_idx]

                # Batch across i values
                n_i_batches = (cardinality + i_batch_size - 1) // i_batch_size

                for i_batch_idx in range(n_i_batches):
                    start_i = i_batch_idx * i_batch_size
                    end_i = min(start_i + i_batch_size, cardinality)
                    i_batch = list(range(start_i, end_i))

                    # Compute P(π_j | x_next, π_i) for ALL j targets and batched i values
                    # This processes i_batch_size integrations in parallel, each computing all cardinality j targets
                    P_batch_result = self.P_batch_i_j(i_batch, x_next,
                                                      n_samples=n_samples,
                                                      mc_batch_size=batch_size,
                                                      show_progress=False)

                    # P_batch_result shape: (len(i_batch), cardinality)
                    # Vectorized assignment: P_mat[j, i_batch, x_next_idx] = P_batch_result.T
                    # Transpose to get (cardinality, len(i_batch)) to match P_mat[:, i_batch, x_next_idx]
                    P_mat[:, i_batch, x_next_idx] = P_batch_result.T
        else:
            # Memory-efficient path: batch j targets (for large cardinality)
            if show_progress:
                print(f"  Using batched j path: j_batch_size={j_batch_size} (cardinality={cardinality})")

            # Process in batches over i and x_next
            for i in tqdm(range(cardinality), desc="Computing P matrix", disable=not show_progress):
                for x_next_idx in range(m_n):
                    x_next = self.SQ.X_n[x_next_idx]

                    # Compute P(π_j | x_next, π_i) for batched j targets
                    j_list = list(range(cardinality))
                    n_j_batches = (cardinality + j_batch_size - 1) // j_batch_size

                    # Process all j_batches and accumulate results on GPU
                    for j_batch_idx in range(n_j_batches):
                        start_j = j_batch_idx * j_batch_size
                        end_j = min(start_j + j_batch_size, cardinality)
                        j_batch = j_list[start_j:end_j]

                        # Each integration uses FULL n_samples (not divided across combinations)
                        # j_batch_size integrations run in parallel, each using n_samples MC samples
                        # P_batch returns GPU array - keep it on GPU
                        P_values = self.P_batch(j_batch, i, x_next,
                                                n_samples=n_samples,  # Full sample count per integration
                                                mc_batch_size=batch_size, show_progress=False)

                        # Store results directly on GPU using vectorized assignment (no CPU transfer)
                        # P_values is shape (len(j_batch),) - assign to P_mat slice
                        P_mat[j_batch, i, x_next_idx] = P_values

        if show_progress:
            print(f"  Step 2: Multiplying with T_mat to create 5D tensor")

        # Step 2: Compute p_n_M[x_next, j, x_current, i, u] = T[x_next, x_current, u] * P[j, i, x_next]
        # Shape: (m_n, cardinality, m_n, cardinality, n_u)
        p_n_M = np.zeros((m_n, cardinality, m_n, cardinality, n_u), dtype=np.float32)

        for k in range(n_u):
            u = self.AQ.U[k]
            u_idx = self.AQ.get_quantized_index(u)
            T_mat_u = self.T_mat[:, :, u_idx]  # (m_n, m_n)

            if show_progress:
                print(f"  Processing action {k+1}/{n_u}...")

            # Compute: p_n_M[x_next, j, x_current, i, u] = T[x_next, x_current, u] * P[j, i, x_next]
            # Fully vectorized using broadcasting:
            # - T_mat_u: (m_n, m_n) = (x_next, x_current) -> reshape to (m_n, 1, m_n, 1)
            # - P_mat: (cardinality, cardinality, m_n) = (j, i, x_next) -> transpose and reshape to (m_n, cardinality, 1, cardinality)
            # - Broadcast multiply: (m_n, 1, m_n, 1) * (m_n, cardinality, 1, cardinality) = (m_n, cardinality, m_n, cardinality)

            # Reshape T_mat_u: (m_n, m_n) -> (m_n, 1, m_n, 1) = (x_next, 1, x_current, 1)
            T_broadcast = T_mat_u[:, np.newaxis, :, np.newaxis]  # (m_n, 1, m_n, 1)

            # Reshape P_mat: (cardinality, cardinality, m_n) = (j, i, x_next)
            # -> transpose to (m_n, cardinality, cardinality) = (x_next, j, i)
            # -> reshape to (m_n, cardinality, 1, cardinality) = (x_next, j, 1, i)
            P_broadcast = P_mat.transpose(2, 0, 1)[:, :, np.newaxis, :]  # (m_n, cardinality, 1, cardinality)

            # Broadcast multiply: (m_n, 1, m_n, 1) * (m_n, cardinality, 1, cardinality)
            # Broadcasting rules: align from right, so:
            #   (m_n, 1, m_n, 1) expands to (m_n, cardinality, m_n, 1) then (m_n, cardinality, m_n, cardinality)
            #   (m_n, cardinality, 1, cardinality) expands to (m_n, cardinality, m_n, cardinality)
            # Result: (m_n, cardinality, m_n, cardinality) = (x_next, j, x_current, i)
            p_n_M[:, :, :, :, k] = T_broadcast * P_broadcast

            # Normalize: for each (x_current, π_current, u), sum over (x_next, π_next) should equal 1
            # Vectorized: sum over axes 0 (x_next) and 1 (π_next)
            totals = np.sum(p_n_M[:, :, :, :, k], axis=(0, 1), keepdims=True)  # (1, 1, m_n, cardinality)
            totals = totals.squeeze()  # (m_n, cardinality) = (x_current, i)
            # Avoid division by zero
            mask = totals > 1e-300  # (m_n, cardinality) = (x_current, i)
            # Broadcast mask and totals to (1, 1, m_n, cardinality) to match p_n_M shape (m_n, cardinality, m_n, cardinality)
            p_n_M[:, :, :, :, k] = np.where(mask[np.newaxis, np.newaxis, :, :],
                                            p_n_M[:, :, :, :, k] / totals[np.newaxis, np.newaxis, :, :],
                                            p_n_M[:, :, :, :, k])

            if show_progress:
                nnz = np.count_nonzero(p_n_M[:, :, :, :, k] > threshold)
                total_elements = m_n * cardinality * m_n * cardinality
                sparsity = (1.0 - nnz / total_elements) * 100
                tqdm.write(f"    Action {k+1}/{n_u}: {nnz:,} non-zeros ({sparsity:.2f}% sparse)")

        return p_n_M

    def c_n_M(self, π, x_current, u):
        """
        Quantized cost function for belief-MDP_n_M_Mapping.

        c_n_M(π^M, x_current, u) = c_tilde_n(π^M, x_current, u)
        where π^M is a quantized belief (len_M,).

        Args:
            π: Quantized belief (len_M,) - belief over maps only
            x_current: current state (state_dim,) - known pose
            u: Action (2,)
        Returns:
            Cost value
        """
        # For mapping, belief is already 1D (len_M,)
        return self.c_tilde_n(π, x_current, u)

    def get_codebook(self) -> np.ndarray:
        """
        Get the quantized belief codebook.

        Returns:
            Π_n_M: Array of shape (cardinality, len_M) containing all quantized beliefs
        """
        return self.BQ.Π_n_M

    def _get_c_n_M_cache_path(self) -> Path:
        """Generate cache file path for c_n_M based on quantization parameters."""
        cache_dir = Path(__file__).parent.parent.parent / "cache/MAP/cost_mapping"
        cache_dir.mkdir(parents=True, exist_ok=True)

        state_bounds_flat = self.state_bounds.flatten().tolist()

        metadata = {
            'M': self.M,
            'n': self.n,
            'state_bounds': state_bounds_flat,
            'max_val': self.motion_model.max_v if hasattr(self.motion_model, 'max_v') else self.motion_model.max_a,
            'dt': float(self.motion_model.dt),
            'map_shape': (self.map_H, self.map_W),
            'sigma_w': float(self.σ_w),
            'sigma_v': float(self.σ_v),
            'model': self.motion_model.__class__.__name__,
            'state_dim': self.state_dim,
            'cardinality': self.BQ.cardinality,
            'N_n': self.len_M,
            'm_n': self.SQ.m_n,
            'n_u': self.AQ.n_u,
            'kernel_version': 'v1-dense-float32',
            'problem_type': 'mapping',
        }

        metadata_str = str(sorted(metadata.items()))
        metadata_hash = hashlib.md5(metadata_str.encode()).hexdigest()[:8]

        max_val = self.motion_model.max_v if hasattr(self.motion_model, 'max_v') else self.motion_model.max_a
        filename = f"cost_mapping_M{self.M}_n{self.n}_map{self.map_H}x{self.map_W}_max{max_val}_{metadata_hash}.npz"

        return cache_dir / filename

    def _compute_c_n_M(self, show_progress: bool = True) -> np.ndarray:
        """
        Precompute costs for all belief-state-action combinations.

        Returns:
            c_n_M: Array of shape (cardinality, m_n, n_u) where c_n_M[i, j, k] = c_n_M(π_i^M, x_j, u_k)
        """
        cardinality = self.BQ.cardinality
        m_n = self.SQ.m_n
        n_u = self.AQ.n_u
        Π_n = self.BQ.Π_n_M  # (cardinality, len_M)
        X_current = self.SQ.X_n  # (m_n, state_dim)
        U_n = self.AQ.U  # (n_u, 2)

        if show_progress:
            print(f"  Computing costs for {cardinality:,} beliefs × {m_n} states × {n_u} actions...")

        # Use vectorized batch computation
        c_n_M = self.c_tilde_n_vectorized_batch(Π_n, X_current, U_n)  # (cardinality, m_n, n_u)

        return c_n_M

    def _save_c_n_M(self, cache_path: Path, c_n_M: np.ndarray) -> None:
        """Save c_n_M to cache file with metadata."""
        state_bounds = self.state_bounds.astype(float)
        state_bounds_save = state_bounds.get() if hasattr(state_bounds, 'get') else state_bounds

        metadata = {
            'M': self.M,
            'n': self.n,
            'state_bounds': state_bounds_save,
            'max_val': self.motion_model.max_v if hasattr(self.motion_model, 'max_v') else self.motion_model.max_a,
            'dt': float(self.motion_model.dt),
            'map_shape': _numpy.array([self.map_H, self.map_W]),
            'sigma_w': float(self.σ_w),
            'sigma_v': float(self.σ_v),
            'm_n': self.SQ.m_n,
            'n_u': self.AQ.n_u,
            'cardinality': self.BQ.cardinality,
            'N_n': self.len_M,
            'model': self.motion_model.__class__.__name__,
            'state_dim': self.state_dim,
            'kernel_version': 'v1-dense-float32',
            'problem_type': 'mapping',
        }

        # Convert to numpy if CuPy array
        if hasattr(c_n_M, 'get'):
            c_n_M_np = c_n_M.get()
        else:
            c_n_M_np = c_n_M

        save_dict = metadata.copy()
        save_dict['c_n_M'] = c_n_M_np.astype(_numpy.float32)

        _numpy.savez_compressed(cache_path, **save_dict)

    def _load_c_n_M(self, cache_path: Path) -> bool:
        """Load c_n_M from cache file if it exists and matches current parameters."""
        if not cache_path.exists():
            return False

        try:
            data = _numpy.load(str(cache_path))

            state_bounds = self.state_bounds.astype(float)
            state_bounds_np = state_bounds.get() if hasattr(state_bounds, 'get') else state_bounds

            expected_metadata = {
                'M': self.M,
                'n': self.n,
                'state_bounds': state_bounds_np,
                'max_val': self.motion_model.max_v if hasattr(self.motion_model, 'max_v') else self.motion_model.max_a,
                'dt': float(self.motion_model.dt),
                'map_shape': _numpy.array([self.map_H, self.map_W]),
                'sigma_w': float(self.σ_w),
                'sigma_v': float(self.σ_v),
                'm_n': self.SQ.m_n,
                'n_u': self.AQ.n_u,
                'cardinality': self.BQ.cardinality,
                'N_n': self.len_M,
                'model': self.motion_model.__class__.__name__,
                'state_dim': self.state_dim,
                'problem_type': 'mapping',
            }

            kernel_version = data.get('kernel_version', 'v1-dense-float32')
            if kernel_version != 'v1-dense-float32':
                return False

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

            c_n_M = data['c_n_M'].astype(_numpy.float32)
            expected_shape = (self.BQ.cardinality, self.SQ.m_n, self.AQ.n_u)

            if c_n_M.shape != expected_shape:
                return False

            # Convert to backend array (CuPy if available)
            if is_cupy:
                import cupy as cp
                c_n_M = cp.asarray(c_n_M)

            self.c_n_M = c_n_M
            return True

        except Exception as e:
            print(f"Error loading c_n_M cache: {e}")
            import traceback
            traceback.print_exc()
            return False

__all__ = ['BeliefMDP_n_M_SLAM', 'BeliefMDP_n_M_Localization', 'BeliefMDP_n_M_Mapping']
