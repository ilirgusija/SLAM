# import cupy as np
import hashlib
from pathlib import Path
from .belief_mdp_n import BeliefMDP_n_SLAM, BeliefMDP_n_Localization, BeliefMDP_n_Mapping
from .mapping import LidarGridMapVec
from .model import LIDAR, SingleIntegratorModel, DoubleIntegratorModel
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

    def __init__(self, M: int, β: float, n: int, motion_model: SingleIntegratorModel | DoubleIntegratorModel,
                 measurement_model: LIDAR, obstacles: list[Obstacle], _map: LidarGridMapVec,
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

    def _get_p_n_M_cache_path(self) -> Path:
        return self._get_cache_path('p_n_M_slam', 'v2-sparse-float16')

    def _save_p_n_M(self, cache_path: Path, p_n_M: list) -> None:
        self._save_sparse_p_n_M(cache_path, p_n_M, 'v2-sparse-float16')

    def _load_p_n_M(self, cache_path: Path) -> bool:
        return self._load_sparse_p_n_M(cache_path, 'v2-sparse-float16')

    def η_n(self, j: int, i: int, u: np.ndarray, n_samples: int = 24000, seed: int = None, batch_size: int = 24000, show_progress: bool = True) -> float:
        """
        Overloaded η_n that takes belief indices instead of belief vectors.

        Computes transition probability η_n(π_j^M | π_i^M, u) via Monte Carlo integration.
        Accesses beliefs directly from cached codebook self.BQ.Π_n_M, avoiding unnecessary
        copying of large belief vectors.

        η_n(π_j^M | π_i^M, u) = ∫ 𝟙_{F(π_i^M,u,y) ≈ π_j^M} H(dy | π_i^M, u)

        Uses forward sampling: sample (x', m) ~ Tn_mat @ π_i^M, then sample y ~ Q(y|x',m).
        Uses batched operations for GPU acceleration.

        Args:
            j: Index of target belief in Π_n_M (i.e., π_j^M = self.BQ.Π_n_M[j])
            i: Index of current belief in Π_n_M (i.e., π_i^M = self.BQ.Π_n_M[i])
            u: Action (2,)
            n_samples: Number of Monte Carlo samples (default: 24000)
            seed: Random seed for reproducibility
            batch_size: Batch size for GPU-accelerated sampling (default: 24000, processes all samples at once)
            show_progress: Whether to show progress bar (default: True)
        Returns:
            float: Probability of transitioning from π_i^M to π_j^M under action u
        """
        if seed is not None:
            random.seed(seed)

        # Access beliefs directly from cached codebook (no copying)
        π_new_flat = self.BQ.Π_n_M[j]  # Target belief (N_n,)
        π_flat = self.BQ.Π_n_M[i]      # Current belief (N_n,)

        # Normalize beliefs to 2D format if needed
        π_new_2d = self.unflatten_belief(π_new_flat)
        π_2d = self.unflatten_belief(π_flat)

        count = 0
        n_batches = (n_samples + batch_size - 1) // batch_size

        # Sequential processing - simple and reliable
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

    def η_n_batch(self, j_list: list[int], i: int, u: np.ndarray, n_samples: int = 24000, seed: int = None,
                  mc_batch_size: int = 24000, show_progress: bool = True) -> np.ndarray:
        """
        Batched version of η_n that processes multiple target beliefs in parallel.

        Computes transition probabilities η_n(π_j^M | π_i^M, u) for multiple j targets simultaneously.
        This is much more efficient than calling η_n repeatedly because:
        1. Monte Carlo samples are shared across all j targets
        2. Belief updates F(π, u, y) are computed once
        3. Distance computations to all j targets are vectorized

        η_n(π_j^M | π_i^M, u) = ∫ 𝟙_{F(π_i^M,u,y) ≈ π_j^M} H(dy | π_i^M, u)

        Args:
            j_list: List of target belief indices in Π_n_M
            i: Index of current belief in Π_n_M (i.e., π_i^M = self.BQ.Π_n_M[i])
            u: Action (2,)
            n_samples: Number of Monte Carlo samples (default: 24000)
            seed: Random seed for reproducibility
            mc_batch_size: Batch size for Monte Carlo sampling (default: 24000, processes all at once)
            show_progress: Whether to show progress bar (default: True)
        Returns:
            np.ndarray: Array of probabilities, shape (len(j_list),). prob[k] = η_n(π_j_list[k]^M | π_i^M, u)
        """
        if seed is not None:
            random.seed(seed)

        if len(j_list) == 0:
            return np.array([])

        # Access current belief from codebook
        π_flat = self.BQ.Π_n_M[i]  # Current belief (N_n,)
        π_2d = self.unflatten_belief(π_flat)

        # Access all target beliefs from codebook
        n_targets = len(j_list)
        # Convert each belief to backend array (handle both numpy and cupy)
        π_targets_flat_list = []
        for j in j_list:
            π_flat = self.BQ.Π_n_M[j]
            # Ensure it's a backend array (convert from numpy if needed)
            if hasattr(π_flat, 'get'):  # CuPy array
                π_targets_flat_list.append(π_flat)
            else:  # NumPy array
                π_targets_flat_list.append(np.asarray(π_flat))
        π_targets_flat = np.stack(π_targets_flat_list)  # (n_targets, N_n)

        # Unflatten all beliefs
        π_targets_2d_list = []
        for π_flat in π_targets_flat:
            π_2d = self.unflatten_belief(π_flat)
            π_targets_2d_list.append(π_2d)
        π_targets_2d = np.stack(π_targets_2d_list)  # (n_targets, m_n, len_M)

        # Initialize counts for each target (use backend array)
        counts = np.zeros(n_targets, dtype=np.int32)

        # Process Monte Carlo samples in batches
        n_mc_batches = (n_samples + mc_batch_size - 1) // mc_batch_size
        batch_iter = tqdm(range(n_mc_batches), desc=f"η_n_batch (n={n_samples}, targets={n_targets}, mc_batch={mc_batch_size})",
                          unit="batch", disable=not show_progress) if show_progress else range(n_mc_batches)

        for batch_idx in batch_iter:
            # Determine actual batch size for last batch
            current_mc_batch_size = min(mc_batch_size, n_samples - batch_idx * mc_batch_size)

            # Batch sample observations (shared across all j targets)
            Y_batch = self.sample_observations_batch(π_2d, u, current_mc_batch_size)  # (current_mc_batch_size, B)

            # Batch update beliefs using F_batch_log (computed once, shared across all j)
            π_sampled_batch = self.F_batch_log(π_2d, u, Y_batch)  # (current_mc_batch_size, m_n, len_M)

            # Compute distances to all target beliefs simultaneously
            # π_sampled_batch: (current_mc_batch_size, m_n, len_M)
            # π_targets_2d: (n_targets, m_n, len_M)
            # We want: distances[k, j] = ||π_sampled_batch[k] - π_targets_2d[j]||

            # Broadcast for vectorized distance computation
            # π_sampled_batch: (current_mc_batch_size, 1, m_n, len_M)
            # π_targets_2d: (1, n_targets, m_n, len_M)
            # Result: (current_mc_batch_size, n_targets, m_n, len_M)
            π_sampled_expanded = π_sampled_batch[:, np.newaxis, :, :]  # (current_mc_batch_size, 1, m_n, len_M)
            π_targets_expanded = π_targets_2d[np.newaxis, :, :, :]  # (1, n_targets, m_n, len_M)

            # Compute L2 distances: (current_mc_batch_size, n_targets)
            distances = np.linalg.norm(π_sampled_expanded - π_targets_expanded, axis=(2, 3))

            # Count matches for each target (distance < threshold)
            matches = distances < 1e-3  # (current_mc_batch_size, n_targets)

            # Sum matches across Monte Carlo samples for each target
            # Convert to CPU if needed for accumulation
            if hasattr(matches, 'get'):
                matches_cpu = matches.get()
            else:
                matches_cpu = matches

            # Sum along MC batch dimension to get counts per target
            batch_counts = np.sum(matches_cpu, axis=0)  # (n_targets,)
            counts += batch_counts.astype(np.int32)

        # Convert counts to probabilities (use backend array)
        counts_array = np.array(counts, dtype=np.float32)
        probabilities = counts_array / n_samples

        return probabilities

    def _compute_p_n_M(self, n_samples: int = 24000, batch_size: int = 24000, j_batch_size: int = 100,
                       show_progress: bool = True, threshold: float = 1e-8) -> list:
        """
        Compute the full transition probability matrix p_n^{(M)} over all belief-action pairs.
        Uses sparse matrices (CSR format) with float16 precision for memory efficiency.

        Args:
            n_samples: Number of Monte Carlo samples for η_n computation (default: 24000)
            batch_size: Batch size for Monte Carlo sampling (default: 24000, processes all at once)
            j_batch_size: Number of target beliefs (j) to process in parallel (default: 100)
            show_progress: Whether to show progress bars
            threshold: Minimum probability threshold (values below this are treated as zero)

        Returns:
            List of CSR sparse matrices, one per action. p_n_M[k][i, j] = p_n^{(M)}(π_j^M | π_i^M, u_k)
        """
        cardinality = self.BQ.cardinality
        n_u = self.AQ.n_u

        # Store as list of sparse matrices (one per action)
        p_n_M_sparse = []

        for k in range(n_u):
            u = self.AQ.U[k]
            if show_progress:
                print(f"Computing p_n_M for action {k+1}/{n_u}...")
                print(f"  This will compute transitions for {cardinality:,} belief states")
                print(f"  Each belief state checks {cardinality:,} target beliefs")
                print(f"  Estimated time per belief state: ~{cardinality * 1.0 / 3600:.1f} hours (at ~1s per η_n call)")
                print(f"  Total estimated time: ~{cardinality * cardinality * 1.0 / 3600 / 24:.1f} days per action")
                print(f"  WARNING: This is computationally infeasible! Consider reducing M or using a smarter approach.")

            # Build COO matrix incrementally (row, col, data)
            # Use lists to accumulate, then convert to arrays once
            rows_list = []
            cols_list = []
            data_list = []

            # Test: Only process first belief state to verify it works
            test_mode = False
            if test_mode:
                print("  TEST MODE: Only processing first belief state")
                cardinality = 1

            # Process belief states with batched η_n for efficiency
            # Batch multiple j targets together to leverage GPU parallelism

            for i in range(cardinality):
                row_data = []
                row_cols = []

                # Sync GPU before starting to ensure previous operations are complete
                if is_cupy:
                    import cupy as cp
                    cp.cuda.Stream.null.synchronize()

                # Process j targets in batches
                j_indices = list(range(cardinality))
                n_j_batches = (cardinality + j_batch_size - 1) // j_batch_size

                pbar = tqdm(range(n_j_batches),
                            desc=f"Action {k+1}/{n_u}, Belief {i+1}/{cardinality}",
                            disable=not show_progress,
                            leave=False,
                            mininterval=0.5)

                for j_batch_idx in pbar:
                    # Get batch of j indices
                    start_j = j_batch_idx * j_batch_size
                    end_j = min(start_j + j_batch_size, cardinality)
                    j_batch = j_indices[start_j:end_j]

                    # Compute η_n for all j targets in this batch simultaneously
                    probs = self.η_n_batch(j_batch, i, u, n_samples=n_samples,
                                           mc_batch_size=batch_size, show_progress=False)

                    # Extract probabilities (convert from GPU if needed)
                    if hasattr(probs, 'get'):
                        probs_cpu = probs.get()
                    else:
                        probs_cpu = probs

                    # Store non-zero probabilities
                    for j_idx, prob in zip(j_batch, probs_cpu):
                        if prob > threshold:
                            row_data.append(float(prob))
                            row_cols.append(j_idx)

                    # Update progress bar
                    if show_progress:
                        pbar.set_postfix({'nonzeros': len(row_data), 'j_batch': f'{start_j}-{end_j-1}'})

                # Sync GPU after each belief state to ensure operations complete
                if is_cupy:
                    import cupy as cp
                    cp.cuda.Stream.null.synchronize()

                # Periodically clear GPU cache to prevent memory issues
                if is_cupy and i % 100 == 0 and i > 0:
                    import cupy as cp
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()

                # Print summary after each belief state
                if show_progress:
                    print(f"  Belief {i+1}/{cardinality}: {len(row_data)} non-zero transitions")

                # Normalize row to ensure probability measure
                if row_data:
                    row_data_arr = np.array(row_data, dtype=np.float32)  # Use float32 for intermediate computation
                    row_sum = float(np.sum(row_data_arr))
                    if row_sum > 0:
                        row_data_arr = row_data_arr / row_sum
                        # Convert to float16 for storage
                        row_data_arr = row_data_arr.astype(np.float16)

                        # Append to lists
                        rows_list.extend([i] * len(row_cols))
                        cols_list.extend(row_cols)
                        data_list.extend(row_data_arr.tolist())  # Convert to list for accumulation

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
                print(f"  Action {k}: {nnz:,} non-zeros ({sparsity:.2f}% sparse)")

        return p_n_M_sparse

    def c_n_M(self, π, u):
        """
        Quantized cost function for belief-MDP_n_M.

        c_n_M(π^M, u) = c_tilde_n(π^M, u)
        where π^M is a quantized belief.

        Args:
            π: Quantized belief (N_n,) - flattened belief vector
            u: Action (2,)
        Returns:
            Cost value
        """
        # Reshape the flattened belief vector to the expected 2D format
        # π is (N_n,) where N_n = m_n + 2^(H*W)
        # We need to split it into state and map components
        m_n = self.SQ.size
        H, W = self.map.occupancy_map.height, self.map.occupancy_map.width
        len_M = 2**(H * W)

        # Extract state and map components
        π_states = π[:m_n]  # First m_n elements
        π_maps = π[m_n:]    # Remaining elements

        # Reshape to 2D format expected by c_tilde_n
        π_2d = np.zeros((m_n, len_M))
        for i in range(m_n):
            for j in range(len_M):
                # Map the flattened index to 2D coordinates
                flat_idx = i * len_M + j
                if flat_idx < len(π):
                    π_2d[i, j] = π[flat_idx]

        # Use the quantized cost from the parent class
        return self.c_tilde_n(π_2d, u)


class BeliefMDP_n_M_Localization(BeliefMDP_n_Localization, BaseBeliefMDP_n_M):
    """Belief-MDP_n_M for Localization: quantized belief space over poses only."""

    @property
    def problem_type(self) -> str:
        return 'localization'

    @property
    def N_n(self) -> int:
        return self.SQ.m_n

    def __init__(self, M: int, β: float, n: int, motion_model: SingleIntegratorModel | DoubleIntegratorModel,
                 measurement_model: LIDAR, obstacles: list[Obstacle], _map: LidarGridMapVec,
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

    def η_n(self, j: int, i: int, u: np.ndarray, n_samples: int = 24000, seed: int = None, batch_size: int = 24000, show_progress: bool = True) -> float:
        """
        Overloaded η_n that takes belief indices instead of belief vectors.

        Computes transition probability η_n(π_j^M | π_i^M, u) via Monte Carlo integration for localization.
        Accesses beliefs directly from cached codebook self.BQ.Π_n_M, avoiding unnecessary
        copying of large belief vectors.

        η_n(π_j^M | π_i^M, u) = ∫ 𝟙_{F(π_i^M,u,y) ≈ π_j^M} H(dy | π_i^M, u)

        Args:
            j: Index of target belief in Π_n_M (i.e., π_j^M = self.BQ.Π_n_M[j])
            i: Index of current belief in Π_n_M (i.e., π_i^M = self.BQ.Π_n_M[i])
            u: Action (2,)
            n_samples: Number of Monte Carlo samples (default: 24000)
            seed: Random seed for reproducibility
            batch_size: Batch size for GPU-accelerated sampling (default: 24000, processes all samples at once)
            show_progress: Whether to show progress bar (default: True)
        Returns:
            float: Probability of transitioning from π_i^M to π_j^M under action u
        """
        if seed is not None:
            random.seed(seed)

        # Access beliefs directly from cached codebook (no copying)
        π_new_flat = self.BQ.Π_n_M[j]  # Target belief (m_n,)
        π_flat = self.BQ.Π_n_M[i]      # Current belief (m_n,)

        # For localization, beliefs are already 1D (no unflatten needed)
        π_new_1d = π_new_flat
        π_1d = π_flat

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

    def η_n_batch(self, j_list: list[int], i: int, u: np.ndarray, n_samples: int = 24000, seed: int = None,
                  mc_batch_size: int = 24000, show_progress: bool = True) -> np.ndarray:
        """
        Batched version of η_n that processes multiple target beliefs in parallel for localization.

        Args:
            j_list: List of target belief indices in Π_n_M
            i: Index of current belief in Π_n_M
            u: Action (2,)
            n_samples: Number of Monte Carlo samples (default: 24000)
            seed: Random seed for reproducibility
            mc_batch_size: Batch size for Monte Carlo sampling (default: 24000)
            show_progress: Whether to show progress bar (default: True)
        Returns:
            np.ndarray: Array of probabilities, shape (len(j_list),)
        """
        if seed is not None:
            random.seed(seed)

        if len(j_list) == 0:
            return np.array([])

        # Access current belief from codebook
        π_flat = self.BQ.Π_n_M[i]  # Current belief (m_n,)
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
        π_targets_1d = np.stack(π_targets_list)  # (n_targets, m_n)

        # Initialize counts for each target
        counts = np.zeros(n_targets, dtype=np.int32)

        # Process Monte Carlo samples in batches
        n_mc_batches = (n_samples + mc_batch_size - 1) // mc_batch_size
        batch_iter = tqdm(range(n_mc_batches), desc=f"η_n_batch (n={n_samples}, targets={n_targets}, mc_batch={mc_batch_size})",
                          unit="batch", disable=not show_progress) if show_progress else range(n_mc_batches)

        for batch_idx in batch_iter:
            current_mc_batch_size = min(mc_batch_size, n_samples - batch_idx * mc_batch_size)

            # Batch sample observations (shared across all j targets)
            Y_batch = self.sample_observations_batch(π_1d, u, current_mc_batch_size)  # (current_mc_batch_size, B)

            # Batch update beliefs using F_batch_log (computed once, shared across all j)
            π_sampled_batch = self.F_batch_log(π_1d, u, Y_batch)  # (current_mc_batch_size, m_n)

            # Compute distances to all target beliefs simultaneously
            π_sampled_expanded = π_sampled_batch[:, np.newaxis, :]  # (current_mc_batch_size, 1, m_n)
            π_targets_expanded = π_targets_1d[np.newaxis, :, :]  # (1, n_targets, m_n)

            # Compute L2 distances: (current_mc_batch_size, n_targets)
            distances = np.linalg.norm(π_sampled_expanded - π_targets_expanded, axis=2)

            # Count matches for each target
            matches = distances < 1e-3  # (current_mc_batch_size, n_targets)

            # Sum matches directly (works with both NumPy and CuPy)
            batch_counts = np.sum(matches, axis=0)  # (n_targets,)

            # Ensure batch_counts matches the backend of counts
            # If counts is CuPy and batch_counts is NumPy (or vice versa), convert
            if hasattr(counts, 'get') and not hasattr(batch_counts, 'get'):
                # counts is CuPy, batch_counts is NumPy - convert batch_counts to CuPy
                batch_counts = np.asarray(batch_counts, dtype=np.int32)
            elif not hasattr(counts, 'get') and hasattr(batch_counts, 'get'):
                # counts is NumPy, batch_counts is CuPy - convert batch_counts to NumPy
                batch_counts = batch_counts.get().astype(np.int32)
            else:
                # Same backend - just ensure dtype
                batch_counts = batch_counts.astype(np.int32)

            counts += batch_counts

        counts_array = np.array(counts, dtype=np.float32)
        probabilities = counts_array / n_samples

        return probabilities

    def _compute_p_n_M(self, n_samples: int = 24000, batch_size: int = 24000, j_batch_size: int = 100,
                       show_progress: bool = True, threshold: float = 1e-8) -> list:
        """Compute the full transition probability matrix p_n^{(M)} over all belief-action pairs for localization."""
        cardinality = self.BQ.cardinality
        n_u = self.AQ.n_u

        p_n_M_sparse = []

        for k in range(n_u):
            u = self.AQ.U[k]
            if show_progress:
                print(f"Computing p_n_M for action {k+1}/{n_u}...")
                print(f"  This will compute transitions for {cardinality:,} belief states")

            rows_list = []
            cols_list = []
            data_list = []

            test_mode = False
            if test_mode:
                print("  TEST MODE: Only processing first belief state")
                cardinality = 1

            for i in range(cardinality):
                row_data = []
                row_cols = []

                if is_cupy:
                    import cupy as cp
                    cp.cuda.Stream.null.synchronize()

                j_indices = list(range(cardinality))
                n_j_batches = (cardinality + j_batch_size - 1) // j_batch_size

                pbar = tqdm(range(n_j_batches),
                            desc=f"Action {k+1}/{n_u}, Belief {i+1}/{cardinality}",
                            disable=not show_progress,
                            leave=False,
                            mininterval=0.5)

                for j_batch_idx in pbar:
                    start_j = j_batch_idx * j_batch_size
                    end_j = min(start_j + j_batch_size, cardinality)
                    j_batch = j_indices[start_j:end_j]

                    probs = self.η_n_batch(j_batch, i, u, n_samples=n_samples,
                                           mc_batch_size=batch_size, show_progress=False)

                    if hasattr(probs, 'get'):
                        probs_cpu = probs.get()
                    else:
                        probs_cpu = probs

                    for j_idx, prob in zip(j_batch, probs_cpu):
                        if prob > threshold:
                            row_data.append(float(prob))
                            row_cols.append(j_idx)

                    if show_progress:
                        pbar.set_postfix({'nonzeros': len(row_data), 'j_batch': f'{start_j}-{end_j-1}'})

                if is_cupy:
                    import cupy as cp
                    cp.cuda.Stream.null.synchronize()

                if is_cupy and i % 100 == 0 and i > 0:
                    import cupy as cp
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()

                if show_progress:
                    print(f"  Belief {i+1}/{cardinality}: {len(row_data)} non-zero transitions")

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
                print(f"  Action {k}: {nnz:,} non-zeros ({sparsity:.2f}% sparse)")

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

    def η_n(self, j: int, i: int, x_current: np.ndarray, u: np.ndarray, x_next: np.ndarray,
            n_samples: int = 24000, seed: int = None, batch_size: int = 24000, show_progress: bool = True) -> float:
        """
        Overloaded η_n that takes belief indices instead of belief vectors for mapping.

        Computes joint transition probability η_n(π_j^M, x_next | π_i^M, x_current, u) via Monte Carlo integration.
        Note: For mapping, this computes the joint probability of belief π_j^M and state x_next.

        Args:
            j: Index of target belief in Π_n_M (i.e., π_j^M = self.BQ.Π_n_M[j])
            i: Index of current belief in Π_n_M (i.e., π_i^M = self.BQ.Π_n_M[i])
            x_current: current state (state_dim,) - part of augmented state b_t = (x_t, π_t)
            u: Action (2,)
            x_next: next state (state_dim,) - given as input, not sampled
            n_samples: Number of Monte Carlo samples (default: 24000)
            seed: Random seed for reproducibility
            batch_size: Batch size for GPU-accelerated sampling (default: 24000)
            show_progress: Whether to show progress bar (default: True)
        Returns:
            float: Joint probability of (π_j^M, x_next) from augmented state (x_current, π_i^M) under action u
        """
        if seed is not None:
            random.seed(seed)

        # Access beliefs directly from cached codebook
        π_new_flat = self.BQ.Π_n_M[j]  # Target belief (len_M,)
        π_flat = self.BQ.Π_n_M[i]      # Current belief (len_M,)

        # For mapping, beliefs are already 1D (no unflatten needed)
        π_new_1d = π_new_flat
        π_1d = π_flat

        # Call parent's η_n method with belief vectors
        return super().η_n(π_new_1d, π_1d, x_current, u, x_next, n_samples=n_samples,
                           seed=seed, batch_size=batch_size, show_progress=show_progress)

    def η_n_batch(self, j_list: list[int], i: int, x_current: np.ndarray, u: np.ndarray, x_next: np.ndarray,
                  n_samples: int = 24000, seed: int = None, mc_batch_size: int = 24000, show_progress: bool = True) -> np.ndarray:
        """
        Batched version of η_n that processes multiple target beliefs in parallel for mapping.

        Args:
            j_list: List of target belief indices in Π_n_M
            i: Index of current belief in Π_n_M
            x_current: current state (state_dim,)
            u: Action (2,)
            x_next: next state (state_dim,)
            n_samples: Number of Monte Carlo samples (default: 24000)
            seed: Random seed for reproducibility
            mc_batch_size: Batch size for Monte Carlo sampling (default: 24000)
            show_progress: Whether to show progress bar (default: True)
        Returns:
            np.ndarray: Array of probabilities, shape (len(j_list),)
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
        batch_iter = tqdm(range(n_mc_batches), desc=f"η_n_batch (n={n_samples}, targets={n_targets}, mc_batch={mc_batch_size})",
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

            # Sum matches directly (works with both NumPy and CuPy)
            batch_counts = np.sum(matches, axis=0)  # (n_targets,)

            # Ensure batch_counts matches the backend of counts
            # If counts is CuPy and batch_counts is NumPy (or vice versa), convert
            if hasattr(counts, 'get') and not hasattr(batch_counts, 'get'):
                # counts is CuPy, batch_counts is NumPy - convert batch_counts to CuPy
                batch_counts = np.asarray(batch_counts, dtype=np.int32)
            elif not hasattr(counts, 'get') and hasattr(batch_counts, 'get'):
                # counts is NumPy, batch_counts is CuPy - convert batch_counts to NumPy
                batch_counts = batch_counts.get().astype(np.int32)
            else:
                # Same backend - just ensure dtype
                batch_counts = batch_counts.astype(np.int32)

            counts += batch_counts

        # Get transition probability T(x_next | x_current, u)
        x_current_idx = self.SQ.get_quantized_index(x_current)
        x_next_idx = self.SQ.get_quantized_index(x_next)
        u_idx = self.AQ.get_quantized_index(u)
        Tn_mat = self.T_mat[:, :, u_idx]
        T_x_next_given_x_current_u = float(Tn_mat[x_next_idx, x_current_idx])

        # Convert counts to probabilities and weight by transition probability
        counts_array = np.array(counts, dtype=np.float32)
        probabilities = (counts_array / n_samples) * T_x_next_given_x_current_u

        return probabilities

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
                print(f"    Action {k}: {nnz:,} non-zeros ({sparsity:.2f}% sparse)")

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
