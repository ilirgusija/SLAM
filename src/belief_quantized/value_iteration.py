import hashlib
from pathlib import Path
from ..utils.array_backend import np
from .belief_mdp_n_M import BeliefMDP_n_M_Localization, BeliefMDP_n_M_Mapping
import numpy as _numpy  # For file I/O only


class ValueIteration():
    """
    Value Iteration solver for the finite belief-MDP approximation.
    Supports Localization and Mapping variants (SLAM is intractable).
    """

    def __init__(self, MDP_n: BeliefMDP_n_M_Localization | BeliefMDP_n_M_Mapping, epsilon=1e-6):
        self.MDP_n = MDP_n
        self.epsilon = epsilon  # convergence threshold

        # Determine problem type
        if isinstance(MDP_n, BeliefMDP_n_M_Localization):
            self.problem_type = 'localization'
        elif isinstance(MDP_n, BeliefMDP_n_M_Mapping):
            self.problem_type = 'mapping'

        # initialize value function to zeros
        cardinality = self.MDP_n.BQ.cardinality
        if self.problem_type == 'localization':
            self.V = np.zeros(cardinality)
        else:  # mapping
            # For mapping, V is over (cardinality, m_n) - belief and state
            self.V = np.zeros((cardinality, self.MDP_n.SQ.m_n))

        # initialize old value function to a large value to ensure first iteration runs
        # (if initialized to zeros, convergence check would pass immediately)
        self.V_old = np.full_like(self.V, np.inf)

        # Track iteration statistics
        self.iteration_count = 0
        self.policy = None  # Will be computed after convergence

    def run(self, verbose: bool = True):
        """
        Run value iteration algorithm using vectorized operations.

        Args:
            verbose: If True, print progress (iteration count, max |V - V_old|).
        """
        self._verbose = getattr(self, '_verbose', verbose)
        if self._verbose:
            card = self.MDP_n.BQ.cardinality
            n_u = self.MDP_n.AQ.n_u
            print(f"  Value iteration: problem={self.problem_type}, |Π|={card}, n_u={n_u}, ε={self.epsilon}")

        if self.problem_type == 'localization':
            return self._run_localization()
        else:  # mapping
            return self._run_mapping()

    def _run_localization(self):
        """
        Vectorized value iteration for localization using sparse matrix operations.

        V(π) = min_u [ρ_n(π, u) + β * Σ_{π'} p(π'|π, u) * V(π')]
        where ρ_n(π, u) = r_exploration(π) + c_effort(u) (cost from MDP, stored in c_n_M).

        Uses sparse matrix-vector multiplication: p_n_M[k] @ V.
        """
        c_n_M = self.MDP_n.c_n_M
        p_n_M = self.MDP_n.p_n_M
        n_u = self.MDP_n.AQ.n_u
        verbose = getattr(self, '_verbose', True)

        # Fail fast on bad cost/transition data (catch upstream bugs: beliefs, ρ_n, η_n)
        if np.any(np.isnan(c_n_M)) or np.any(np.isinf(c_n_M)):
            raise ValueError(
                "c_n_M contains NaN or Inf (upstream bug). "
                "Cost ρ_n = r_exploration + c_effort must be finite; check beliefs/codebook and delete c_n_M_localization cache if stale."
            )
        for k in range(n_u):
            if np.any(np.isnan(p_n_M[k].data)) or np.any(np.isinf(p_n_M[k].data)):
                raise ValueError(
                    f"p_n_M[{k}] contains NaN or Inf (upstream bug). "
                    "Transition η_n must be finite; check beliefs and delete p_n_M_localization cache if stale."
                )

        q_values = None
        while self.is_not_converged():
            self.V_old = self.V.copy()
            self.iteration_count += 1

            q_values = np.zeros((self.MDP_n.BQ.cardinality, n_u), dtype=np.float32)

            for k in range(n_u):
                # Σ_{π'} p(π'|π, u_k) * V(π') for all π
                expected_future_value = p_n_M[k] @ self.V  # (cardinality,)
                q_values[:, k] = c_n_M[:, k] + self.MDP_n.β * expected_future_value

            self.V = np.min(q_values, axis=1)  # (cardinality,)

            max_diff = float(np.max(np.abs(self.V - self.V_old)))
            if verbose:
                # Print every iteration for small problems; every 10th for larger
                card = self.MDP_n.BQ.cardinality
                if self.iteration_count <= 5 or self.iteration_count % 10 == 0 or max_diff < self.epsilon:
                    print(f"    iter {self.iteration_count}: max |V - V_old| = {max_diff:.2e}")

        # Extract optimal policy after convergence
        self.policy = self._extract_policy_localization(c_n_M, p_n_M, q_values)

        if verbose:
            print(f"  Converged in {self.iteration_count} iterations (max_diff = {max_diff:.2e})")
        return self.V

    def _run_mapping(self):
        """
        Vectorized value iteration for mapping.

        V(π, x) = min_u [ρ_n(π, x, u) + β * Σ_{x', π'} p(x', π'|x, π, u) * V(π', x')]
        """
        c_n_M = self.MDP_n.c_n_M
        p_n_M = self.MDP_n.p_n_M
        verbose = getattr(self, '_verbose', True)
        max_diff = float('inf')

        cardinality = self.MDP_n.BQ.cardinality
        m_n = self.MDP_n.SQ.m_n
        n_u = self.MDP_n.AQ.n_u

        while self.is_not_converged():
            self.V_old = self.V.copy()
            self.iteration_count += 1
            # p_n_M is list of sparse (n_states, n_states), n_states = m_n*cardinality, state_idx = x*card + π
            V_flat = self.V.T.reshape(-1)  # (m_n*cardinality,)
            future_list = []
            for k in range(n_u):
                Ev = p_n_M[k] @ V_flat  # (n_states,)
                future_list.append(Ev.reshape(m_n, cardinality).T)  # (cardinality, m_n)
            future_values = np.stack(future_list, axis=2)  # (cardinality, m_n, n_u)
            if c_n_M.ndim == 3:
                # State-dependent mapping cost provided directly: (cardinality, m_n, n_u)
                q_values = c_n_M + self.MDP_n.β * future_values
            elif c_n_M.ndim == 2:
                # State-independent mapping cost: (cardinality, n_u) -> broadcast over x
                q_values = c_n_M[:, np.newaxis, :] + self.MDP_n.β * future_values
            else:
                raise ValueError(
                    f"Invalid mapping c_n_M shape {c_n_M.shape}. "
                    "Expected (cardinality, n_u) or (cardinality, m_n, n_u)."
                )
            self.V = np.min(q_values, axis=2)  # (cardinality, m_n)

            max_diff = float(np.max(np.abs(self.V - self.V_old)))
            if verbose and (self.iteration_count <= 5 or self.iteration_count % 10 == 0 or max_diff < self.epsilon):
                print(f"    iter {self.iteration_count}: max |V - V_old| = {max_diff:.2e}")

        # Extract optimal policy after convergence
        self.policy = self._extract_policy_mapping(c_n_M, p_n_M)

        if verbose:
            print(f"  Converged in {self.iteration_count} iterations (max_diff = {max_diff:.2e})")
        return self.V

    def is_not_converged(self):
        """
        Check if value iteration has converged.
        """
        if np.max(np.abs(self.V - self.V_old)) < self.epsilon:
            return False
        return True

    def _extract_policy_localization(self, c_n_M, p_n_M, q_values=None):
        """
        Extract optimal policy for localization: π*(π) = argmin_u Q(π, u)

        Returns:
            policy: Array of shape (cardinality,) with optimal action indices
        """
        if q_values is None:
            # Recompute Q-values
            n_u = self.MDP_n.AQ.n_u
            q_values = np.zeros((self.MDP_n.BQ.cardinality, n_u), dtype=np.float32)
            for k in range(n_u):
                expected_future_value = p_n_M[k] @ self.V
                q_values[:, k] = c_n_M[:, k] + self.MDP_n.β * expected_future_value

        return np.argmin(q_values, axis=1)  # (cardinality,)

    def _extract_policy_mapping(self, c_n_M, p_n_M):
        """
        Extract optimal policy for mapping: π*(π, x) = argmin_u Q(π, x, u)

        Returns:
            policy: Array of shape (cardinality, m_n) with optimal action indices
        """
        cardinality = self.MDP_n.BQ.cardinality
        m_n = self.MDP_n.SQ.m_n
        n_u = self.MDP_n.AQ.n_u
        V_flat = self.V.T.reshape(-1)
        future_list = []
        for k in range(n_u):
            Ev = p_n_M[k] @ V_flat
            future_list.append(Ev.reshape(m_n, cardinality).T)
        future_values = np.stack(future_list, axis=2)  # (cardinality, m_n, n_u)
        if c_n_M.ndim == 3:
            q_values = c_n_M + self.MDP_n.β * future_values
        elif c_n_M.ndim == 2:
            q_values = c_n_M[:, np.newaxis, :] + self.MDP_n.β * future_values
        else:
            raise ValueError(
                f"Invalid mapping c_n_M shape {c_n_M.shape}. "
                "Expected (cardinality, n_u) or (cardinality, m_n, n_u)."
            )
        return np.argmin(q_values, axis=2)  # (cardinality, m_n)

    def _get_metadata(self):
        """Get metadata for saving value iteration results."""
        mdp = self.MDP_n
        state_bounds = mdp.state_bounds.astype(float)
        state_bounds_save = state_bounds.get() if hasattr(state_bounds, 'get') else state_bounds

        metadata = {
            'M': mdp.M,
            'n': mdp.n,
            'β': float(mdp.β),
            'epsilon': float(self.epsilon),
            'state_bounds': state_bounds_save,
            'max_val': mdp.motion_model.max_v if hasattr(mdp.motion_model, 'max_v') else mdp.motion_model.max_a,
            'dt': float(mdp.motion_model.dt),
            'map_shape': _numpy.array([mdp.map_H, mdp.map_W]),
            'sigma_w': float(mdp.σ_w),
            'sigma_v': float(mdp.σ_v),
            'model': mdp.motion_model.__class__.__name__,
            'state_dim': mdp.state_dim,
            'cardinality': mdp.BQ.cardinality,
            'm_n': mdp.SQ.m_n,
            'n_u': mdp.AQ.n_u,
            'problem_type': self.problem_type,
            'iteration_count': int(self.iteration_count),
            'converged': True,
            'final_max_diff': float(np.max(np.abs(self.V - self.V_old))),
        }

        if self.problem_type == 'localization':
            metadata['N_n'] = mdp.SQ.m_n
            metadata['exploration_type'] = getattr(mdp, 'exploration_type', 'information gain')
        else:  # mapping
            metadata['N_n'] = mdp.len_M

        return metadata

    def _get_cache_path(self):
        """Generate cache file path for value iteration results."""
        mdp = self.MDP_n
        problem_dir = 'LOC' if self.problem_type == 'localization' else 'MAP'
        cache_dir = Path(__file__).parent.parent.parent / f"cache/{problem_dir}/value_iteration"
        cache_dir.mkdir(parents=True, exist_ok=True)

        metadata = self._get_metadata()
        state_bounds_flat = metadata['state_bounds'].flatten().tolist() if hasattr(
            metadata['state_bounds'], 'flatten') else metadata['state_bounds']
        metadata['state_bounds'] = state_bounds_flat

        metadata_str = str(sorted(metadata.items()))
        metadata_hash = hashlib.md5(metadata_str.encode()).hexdigest()[:8]

        max_val = metadata['max_val']
        filename = f"value_iteration_{self.problem_type}_M{mdp.M}_n{mdp.n}_beta{mdp.β}_eps{self.epsilon}_map{mdp.map_H}x{mdp.map_W}_max{max_val}_{metadata_hash}.npz"

        return cache_dir / filename

    def save_results(self, cache_path: Path = None):
        """
        Save value iteration results (value function, policy, metadata) to cache.

        Args:
            cache_path: Optional path to save to. If None, generates path automatically.
        """
        if cache_path is None:
            cache_path = self._get_cache_path()

        # Convert arrays to numpy for saving
        if hasattr(self.V, 'get'):
            V_save = self.V.get()
        else:
            V_save = self.V

        if self.policy is None:
            # Extract policy if not already computed
            if self.problem_type == 'localization':
                self.policy = self._extract_policy_localization(self.MDP_n.c_n_M, self.MDP_n.p_n_M)
            else:
                self.policy = self._extract_policy_mapping(self.MDP_n.c_n_M, self.MDP_n.p_n_M)

        if hasattr(self.policy, 'get'):
            policy_save = self.policy.get()
        else:
            policy_save = self.policy

        metadata = self._get_metadata()

        save_dict = metadata.copy()
        save_dict['V'] = V_save.astype(_numpy.float32)
        save_dict['policy'] = policy_save.astype(_numpy.int32)
        save_dict['kernel_version'] = 'v1-value-iteration'

        _numpy.savez_compressed(cache_path, **save_dict)
        print(f"Saved value iteration results to {cache_path}")

        # Print file size
        file_size_mb = cache_path.stat().st_size / (1024 * 1024)
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Iterations: {self.iteration_count}")
        print(f"  Final max |V - V_old|: {metadata['final_max_diff']:.2e}")

    def load_results(self, cache_path: Path = None) -> bool:
        """
        Load value iteration results from cache.

        Args:
            cache_path: Optional path to load from. If None, generates path automatically.

        Returns:
            True if successfully loaded, False otherwise.
        """
        if cache_path is None:
            cache_path = self._get_cache_path()

        if not cache_path.exists():
            return False

        try:
            data = _numpy.load(str(cache_path))

            if data.get('kernel_version') != 'v1-value-iteration':
                return False

            # Verify metadata matches
            expected_metadata = self._get_metadata()
            # Remove iteration-specific fields for comparison
            expected_metadata.pop('iteration_count', None)
            expected_metadata.pop('converged', None)
            expected_metadata.pop('final_max_diff', None)

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

            # Load value function and policy
            V_loaded = data['V'].astype(_numpy.float32)
            policy_loaded = data['policy'].astype(_numpy.int32)

            # Convert to backend array (CuPy if available)
            if hasattr(np, 'asarray'):
                self.V = np.asarray(V_loaded)
                self.policy = np.asarray(policy_loaded)
            else:
                self.V = V_loaded
                self.policy = policy_loaded

            self.iteration_count = int(data.get('iteration_count', 0))

            print(f"Loaded value iteration results from {cache_path}")
            print(f"  Iterations: {self.iteration_count}")
            print(f"  Final max |V - V_old|: {data.get('final_max_diff', 0):.2e}")

            return True

        except Exception as e:
            print(f"Error loading value iteration cache: {e}")
            import traceback
            traceback.print_exc()
            return False
