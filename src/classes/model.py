# Use CuPy backend (drop-in replacement for NumPy)
import warnings
from typing import Optional, Tuple, List
from .mapping import BaseMap, LidarGridMapVec, LandmarkMap
from .obstacle import Obstacle
from ..utils.angle import rot_mat_2d
from ..utils.array_backend import np
import matplotlib.pyplot as plt
# Use CuPy's ndimage if available and USE_CUPY is enabled, otherwise scipy's
import os
use_cupy = os.getenv("USE_CUPY", "true").lower() in ("true", "1", "yes")
if use_cupy:
    try:
        from cupyx.scipy import ndimage
    except ImportError:
        from scipy import ndimage
else:
    from scipy import ndimage
# Use CuPy backend (drop-in replacement for NumPy)


def wrap_angle(theta):
    """Wrap angle to [-π, π) using atan2 for numerical stability."""
    return np.arctan2(np.sin(theta), np.cos(theta))


def angular_distance(theta1, theta2):
    """Compute distance on S^1 (arc-length metric)."""
    diff = wrap_angle(theta1 - theta2)
    return np.abs(diff)


class UnicycleModel:
    """
    Unicycle kinematic model on SE(2) ≅ R² × S¹.

    State: x = (p_x, p_y, θ) ∈ R² × S¹
    Control: u = (v, ω) ∈ R² (linear velocity, angular velocity)

    Dynamics:
        p_{t+1} = p_t + v·Δt·[cos(θ), sin(θ)]ᵀ + w^p
        θ_{t+1} = (θ_t + ω·Δt + w^θ) mod 2π

    The angular component uses proper S¹ topology via wrap_angle.
    """

    def __init__(self, p_x: float, p_y: float, theta: float, dt: float,
                 max_v: float = 1.0, max_omega: float = np.pi):
        """
        Args:
            p_x, p_y: Initial position
            theta: Initial heading (radians)
            dt: Time step
            max_v: Maximum linear velocity
            max_omega: Maximum angular velocity
        """
        self.x = np.array([p_x, p_y, wrap_angle(theta)], dtype=float)
        self.dt = dt
        self.max_v = max_v
        self.max_omega = max_omega
        self._position_bounds = None  # (2, 2) for position only
        self._state_bounds = None     # (3, 2) for full state (position + angle)

    def set_state_bounds(self, bounds):
        """
        Set bounds for the full state.

        Args:
            bounds: (3, 2) array where bounds[:2] are position bounds
                    and bounds[2] are angle bounds (typically [-π, π])
        """
        bounds = np.asarray(bounds, dtype=float)
        if bounds.shape != (3, 2):
            raise ValueError("UnicycleModel bounds must be shape (3, 2)")
        self._state_bounds = bounds
        self._position_bounds = bounds[:2]
        self.lower = bounds[:, 0]
        self.upper = bounds[:, 1]

    def set_position_bounds(self, bounds):
        """Set bounds for position only (θ is always in [-π, π))."""
        bounds = np.asarray(bounds, dtype=float)
        if bounds.shape != (2, 2):
            raise ValueError("Position bounds must be shape (2, 2)")
        self._position_bounds = bounds
        # Create full state bounds with angle in [-π, π]
        self._state_bounds = np.vstack([bounds, [[-np.pi, np.pi]]])
        self.lower = self._state_bounds[:, 0]
        self.upper = self._state_bounds[:, 1]

    def _project_state(self, x):
        """
        Project state to valid region: clip position, wrap angle to preserve compactness.

        This ensures the state space remains compact by:
        1. Wrapping the angle component to [-π, π) (always preserves S¹ compactness)
        2. Clamping position components to bounds if bounds are set

        Args:
            x: State vector (3,) or (N, 3) - [p_x, p_y, θ]

        Returns:
            Projected state vector with angle wrapped and position clamped
        """
        x = np.asarray(x, dtype=float)

        if x.ndim == 1:
            x_proj = x.copy()
            # Wrap angle first (always do this to preserve S¹ compactness)
            x_proj[2] = wrap_angle(x_proj[2])
            # Clip position if bounds set (preserves R² compactness)
            if self._position_bounds is not None:
                x_proj[:2] = np.clip(x_proj[:2],
                                     self._position_bounds[:, 0],
                                     self._position_bounds[:, 1])
            return x_proj
        else:
            # Batched: (N, 3)
            x_proj = x.copy()
            # Wrap angle first (always do this to preserve S¹ compactness)
            x_proj[:, 2] = wrap_angle(x_proj[:, 2])
            # Clip position if bounds set (preserves R² compactness)
            if self._position_bounds is not None:
                x_proj[:, :2] = np.clip(x_proj[:, :2],
                                        self._position_bounds[:, 0],
                                        self._position_bounds[:, 1])
            return x_proj

    def _nominal_step(self, x, u):
        """
        Deterministic dynamics: x_{t+1} = f(x_t, u_t, 0)

        Args:
            x: State (3,) or (N, 3)
            u: Control (2,) - (v, ω)

        Returns:
            x_next: Next state (3,) or (N, 3)
        """
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)

        if x.ndim == 1:
            p_x, p_y, theta = x[0], x[1], x[2]
            v, omega = u[0], u[1]

            p_x_next = p_x + v * self.dt * np.cos(theta)
            p_y_next = p_y + v * self.dt * np.sin(theta)
            theta_next = theta + omega * self.dt  # wrap happens in _project_state

            return np.array([p_x_next, p_y_next, theta_next])
        else:
            # Batched: (N, 3)
            p = x[:, :2]
            theta = x[:, 2]
            v, omega = u[0], u[1]

            dp = v * self.dt * np.stack([np.cos(theta), np.sin(theta)], axis=1)
            p_next = p + dp
            theta_next = theta + omega * self.dt

            return np.concatenate([p_next, theta_next[:, np.newaxis]], axis=1)

    def update(self, u):
        """Update internal state with control u = (v, ω)."""
        self.x = self._project_state(self._nominal_step(self.x, u))
        return self.x

    def f_bar(self, x, u):
        """Deterministic transition function."""
        x = np.asarray(x, dtype=float)
        x_next = self._nominal_step(x, u)
        return self._project_state(x_next)

    def f(self, x, u, w):
        """
        Stochastic transition: f(x, u, w) = f_bar(x, u) + w (with wrapping for θ)

        Args:
            x: State (3,) or (N, 3)
            u: Control (2,)
            w: Noise (3,) or (N, 3) - (w_px, w_py, w_θ)

        Returns:
            x_next: Next state with noise applied
        """
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        x_nom = self._nominal_step(x, u)
        return self._project_state(x_nom + w)

    @property
    def position(self):
        """Current position (p_x, p_y)."""
        return self.x[:2]

    @property
    def heading(self):
        """Current heading θ."""
        return self.x[2]


class RangeBearingSensor:
    """
    Range-bearing sensor for landmark-based SLAM.

    Based on Thrun et al. (2005) Section 6.6.2. The measurement space is
    R × Θ where R = [ε, r_max] and Θ = [-π, π) (topologized as S¹).

    Observation model for landmark m_i at robot pose (p, θ):
        g(x, m^i, v^i) = [||m^i - p|| + v^{r,i}, P_Θ(atan2(m^i - p) + v^{φ,i})]ᵀ

    where:
        P_Θ(φ) = atan2(sin φ, cos φ)       (wrapping to [-π, π))
        v^i = (v^{r,i}, v^{φ,i}) ~ μ_v     (noise)

    Visibility predicate:
        V^i(x, m^i) = 𝟙[||m^i - p|| ∈ [ε, r_max]]

    The minimum range ε > 0 excludes the singularity of atan2 at the origin,
    ensuring bearing is well-defined for all visible landmarks.

    Noise model:
        - Range: Truncated Gaussian T_[ε-r*, r_max-r*](0, σ_r²)
        - Bearing: Wrapped normal (approximated as Gaussian with arc-length distance)
    """

    def __init__(self, r_max: float = 10.0, epsilon: float = 0.01,
                 sigma_r: float = 0.1, sigma_phi: float = 0.05):
        """
        Args:
            r_max: Maximum sensing range
            epsilon: Minimum sensing range (must be > 0 to avoid atan2 singularity)
            sigma_r: Standard deviation of range noise
            sigma_phi: Standard deviation of bearing noise
        """
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0 to avoid atan2 singularity")
        if epsilon >= r_max:
            raise ValueError("epsilon must be < r_max")

        self.r_max = r_max
        self.epsilon = epsilon
        self.sigma_r = sigma_r
        self.sigma_phi = sigma_phi
        self.R = np.diag([sigma_r**2, sigma_phi**2])

    def _project_bearing(self, phi):
        """
        Projection operator P_Θ: R → [-π, π).

        Wraps angle to [-π, π) using atan2 for numerical stability.
        This implements the canonical quotient map R → R/2πZ ≅ S¹.
        """
        return wrap_angle(phi)

    def _compute_range_bearing(self, x, m):
        """
        Compute raw range and bearing to all landmarks.

        Computes:
            r_i = ||m^i - p||₂           (raw range, may be < ε or > r_max)
            φ_i = atan2(m^i - p) - θ     (relative bearing if θ provided, else global bearing)

        Args:
            x: Robot state - can be:
               - (2,) or (N, 2): (p_x, p_y) - bearings relative to x-axis
               - (3,) or (N, 3): (p_x, p_y, θ) - bearings relative to robot heading
               - (4,) or (N, 4): (p_x, p_y, v_x, v_y) - bearings relative to x-axis
            m: Landmark positions (M, 2)

        Returns:
            ranges: (N, M) or (M,) raw distances (not projected)
            bearings: (N, M) or (M,) angles in [-π, π) (wrapped)
        """
        x = np.asarray(x, dtype=float)
        m = np.asarray(m, dtype=float)

        if m.ndim == 1:
            m = m[np.newaxis, :]

        single_pose = x.ndim == 1
        if single_pose:
            x = x[np.newaxis, :]

        state_dim = x.shape[1]
        p = x[:, :2]          # (N, 2) - position

        # Compute relative positions: (N, M, 2)
        delta = m[np.newaxis, :, :] - p[:, np.newaxis, :]

        # Range: (N, M)
        ranges = np.sqrt(np.sum(delta**2, axis=2))

        # Global bearing (relative to x-axis): (N, M)
        # np.arctan2(0, 0) returns 0.0; treat this singular case explicitly.
        global_bearing = np.arctan2(delta[:, :, 1], delta[:, :, 0])

        # Compute bearings based on state dimension
        if state_dim == 3:
            # State includes heading: use relative bearing
            theta = x[:, 2]  # (N,)
            bearings = wrap_angle(global_bearing - theta[:, np.newaxis])
        else:
            # State doesn't include heading: use global bearing (relative to x-axis)
            bearings = global_bearing

        if single_pose:
            return ranges[0], bearings[0]
        return ranges, bearings

    def visibility_mask(self, x, m):
        """
        Compute visibility mask for landmarks.

        Visibility function V^i(x, m^i) = 1 if ||m^i - p|| ∈ [ε, r_max], else 0.

        Args:
            x: Robot state - can be:
               - (2,) or (N, 2): (p_x, p_y)
               - (3,) or (N, 3): (p_x, p_y, θ)
               - (4,) or (N, 4): (p_x, p_y, v_x, v_y)
            m: Landmark positions (M, 2) or batched (len_M, M, 2)

        Returns:
            mask: (N, M), (M,), (N, len_M, M), or (len_M, M) boolean array
        """
        x = np.asarray(x, dtype=float)
        m = np.asarray(m, dtype=float)

        single_pose = x.ndim == 1
        if single_pose:
            x = x[np.newaxis, :]

        p = x[:, :2]  # (N, 2)

        if m.ndim == 2:
            m_batched = m[np.newaxis, :, :]  # (1, M, 2)
        else:
            m_batched = m  # (len_M, M, 2)

        # Compute ranges with broadcasting: (N, len_M, M)
        delta = m_batched[np.newaxis, :, :, :] - p[:, np.newaxis, np.newaxis, :]
        ranges = np.sqrt(np.sum(delta**2, axis=-1))

        # Visible if range is in [epsilon, r_max]
        mask = (ranges >= self.epsilon) & (ranges <= self.r_max)

        if m.ndim == 2:
            mask = mask[:, 0, :]
            return mask[0] if single_pose else mask
        return mask[0] if single_pose else mask

    def g_bar(self, x, m):
        """
        Deterministic observation function (noise-free).

        Handles both single and batched maps by broadcasting.

        Returns y^{*,i} = (r^{*,i}, φ^{*,i}) where:
            r^{*,i} = ||m^i - p||
            φ^{*,i} = P_Θ(atan2(m^i - p) - θ) if state includes heading, else P_Θ(atan2(m^i - p))

        For non-visible landmarks, returns sentinel values [nan, nan] (denoting ⊥).

        Args:
            x: Robot state - can be:
               - (2,) or (N, 2): (p_x, p_y) - bearings relative to x-axis
               - (3,) or (N, 3): (p_x, p_y, θ) - bearings relative to robot heading
               - (4,) or (N, 4): (p_x, p_y, v_x, v_y) - bearings relative to x-axis
            m: Landmark positions (M, 2) or batched landmarks (len_M, M, 2)
            map_obj: Unused for landmark arrays (kept for compatibility)

        Returns:
            z: Observations
               - Single map: (N, M, 2) or (M, 2) observations [range, bearing]
               - Batched maps: (N, len_M, M, 2) observations
               Non-visible landmarks get [nan, nan] (sentinel ⊥)
        """
        x = np.asarray(x, dtype=float)
        m = np.asarray(m, dtype=float)

        single_pose = x.ndim == 1
        if single_pose:
            x = x[np.newaxis, :]

        p = x[:, :2]  # (N, 2)
        state_dim = x.shape[1]

        if m.ndim == 2:
            m_batched = m[np.newaxis, :, :]  # (1, M, 2)
        else:
            m_batched = m  # (len_M, M, 2)

        # Compute relative positions: (N, len_M, M, 2)
        delta = m_batched[np.newaxis, :, :, :] - p[:, np.newaxis, np.newaxis, :]

        # Range: (N, len_M, M)
        ranges = np.sqrt(np.sum(delta**2, axis=-1))

        # Global bearing (relative to x-axis): (N, len_M, M)
        # np.arctan2(0, 0) returns 0.0; treat this singular case explicitly.
        global_bearing = np.arctan2(delta[..., 1], delta[..., 0])

        # Compute bearings based on state dimension
        if state_dim == 3:
            theta = x[:, 2][:, np.newaxis, np.newaxis]
            bearings = wrap_angle(global_bearing - theta)
        else:
            bearings = global_bearing

        # Apply bearing projection (wrap to [-π, π))
        projected_bearings = self._project_bearing(bearings)

        # Stack into observations: (N, len_M, M, 2)
        z = np.stack([ranges, projected_bearings], axis=-1)

        # Mask non-visible: set to nan/nan (sentinel ⊥)
        mask = (ranges >= self.epsilon) & (ranges <= self.r_max)
        z[~mask, 0] = np.nan
        z[~mask, 1] = np.nan

        if m.ndim == 2:
            z = z[:, 0, :, :]
            return z[0] if single_pose else z
        return z[0] if single_pose else z

    def g(self, x, m, v=None):
        """
        Stochastic observation model.

        Implements g(x, m^i, v^i) = [r^* + v^{r,i}, P_Θ(φ^* + v^{φ,i})]ᵀ
        where v^i ~ μ_v is noise and y^* = g_bar(x, m^i) is the noise-free observation.

        Noise model:
        - Range: Gaussian N(0, σ_r²)
        - Bearing: Wrapped normal (Gaussian with arc-length distance)

        Args:
            x: Robot state - can be:
               - (2,) or (N, 2): (p_x, p_y)
               - (3,) or (N, 3): (p_x, p_y, θ)
               - (4,) or (N, 4): (p_x, p_y, v_x, v_y)
            m: Landmark positions (M, 2)
            v: Observation noise (N, M, 2) or (M, 2) or None
               If None, samples from μ_v (Gaussian for range, wrapped normal for bearing)

        Returns:
            z_noisy: (N, M, 2) or (M, 2) noisy observations
                    Non-visible landmarks get [nan, nan] (sentinel ⊥)
        """
        x = np.asarray(x, dtype=float)
        single_pose = x.ndim == 1
        if single_pose:
            x = x[np.newaxis, :]

        # Get noise-free observations
        z_star = self.g_bar(x, m)
        if z_star.ndim == 2:
            z_star = z_star[np.newaxis, :, :]

        visible = np.isfinite(z_star[..., 0])

        if v is None:
            # Sample noise from μ_v
            # For range: Gaussian N(0, σ_r²)
            # For bearing: wrapped normal (sample from N(0, σ_φ²) then wrap)

            # Get noise-free ranges for noise sampling
            ranges_raw, _ = self._compute_range_bearing(x, m)

            # Sample range noise
            v_r = np.random.randn(*ranges_raw.shape) * self.sigma_r

            # Sample bearing noise: wrapped normal
            v_phi = np.random.randn(*ranges_raw.shape) * self.sigma_phi

            v = np.stack([v_r, v_phi], axis=-1)  # (N, M, 2)
        else:
            # Ensure v has correct shape to match z_star
            v = np.asarray(v, dtype=float)
            if v.ndim == 2:
                v = v[np.newaxis, :, :]
            # Ensure v matches z_star shape
            if v.shape != z_star.shape:
                raise ValueError(f"v shape {v.shape} does not match z_star shape {z_star.shape}")

        # Add noise and apply bearing projection
        z_noisy = z_star.copy()
        z_noisy[visible] = z_star[visible] + v[visible]

        # Apply bearing projection to ensure closure in [-π, π)
        z_noisy[..., 1] = self._project_bearing(z_noisy[..., 1])

        # Non-visible landmarks remain as [nan, nan]
        z_noisy[~visible, 0] = np.nan
        z_noisy[~visible, 1] = np.nan

        if single_pose:
            return z_noisy[0]
        return z_noisy

    def get_visible_landmarks(self, x, m) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get indices and observations of visible landmarks.

        Returns only landmarks where V^i(x, m^i) = 1, i.e., ||m^i - p|| ∈ [ε, r_max].

        Args:
            x: Robot state (2,), (3,), or (4,)
            m: (M, 2) landmark positions

        Returns:
            indices: (K,) indices of visible landmarks
            observations: (K, 2) range-bearing observations [range, bearing]
                         where range ∈ [ε, r_max] and bearing ∈ [-π, π)
        """
        z = self.g_bar(x, m)
        visible = np.isfinite(z[:, 0])
        indices = np.where(visible)[0]
        return indices, z[visible]

class SingleIntegratorModel:
    def __init__(self, i_x, i_y, dt, max_v):
        self.x = np.array([i_x, i_y], dtype=float)
        self.dt = dt
        self.max_v = max_v
        self._state_bounds = None

    def set_state_bounds(self, bounds):
        bounds = np.asarray(bounds, dtype=float)
        if bounds.shape != (2, 2):
            raise ValueError("SingleIntegratorModel bounds must be shape (2, 2)")
        self._state_bounds = bounds
        self.lower = self._state_bounds[:, 0]
        self.upper = self._state_bounds[:, 1]

    def _project_state(self, x):
        """
        Project state to valid region (clamp to bounds) to preserve compactness.

        This ensures the state space remains compact by clamping all state components
        to their valid bounds. If bounds are not set, returns state unchanged.

        Args:
            x: State vector (2,) or (N, 2) - [p_x, p_y]

        Returns:
            Projected state vector with all components clamped to [lower, upper]
        """
        if self._state_bounds is None:
            return x

        # Ensure x is an array
        x = np.asarray(x, dtype=float)

        # Clip to bounds to preserve compactness
        return np.clip(x, self.lower, self.upper)

    def _nominal_step(self, x, u):
        return x + u * self.dt

    def update(self, v):
        self.x = self._project_state(self._nominal_step(self.x, v))
        return self.x

    def f_bar(self, x, u):
        x = np.asarray(x, dtype=float)
        x_star = self._nominal_step(x, u)
        return self._project_state(x_star)

    def f(self, x, u, w):
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        x_nom = self._nominal_step(x, u)
        return self._project_state(x_nom + w)

class DoubleIntegratorModel:
    def __init__(self, p_x, p_y, v_x, v_y, dt, max_a):
        self.x = np.array([p_x, p_y, v_x, v_y], dtype=float)
        self.dt = dt
        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.B = np.array([[0.5 * dt**2, 0],
                           [0, 0.5 * dt**2],
                           [dt, 0],
                           [0, dt]])
        self.Q_t = np.array([[(dt**3) / 3, 0, (dt**2) / 2, 0],
                             [0, (dt**3) / 3, 0, (dt**2) / 2],
                             [(dt**2) / 2, 0, dt, 0],
                             [0, (dt**2) / 2, 0, dt]])
        self.max_a = max_a
        self._state_bounds = None

    def set_state_bounds(self, bounds):
        bounds = np.asarray(bounds, dtype=float)
        if bounds.shape != (4, 2):
            raise ValueError("DoubleIntegratorModel bounds must be shape (4, 2)")
        self._state_bounds = bounds
        self.lower = self._state_bounds[:, 0]
        self.upper = self._state_bounds[:, 1]

    def _project_state(self, x):
        """
        Project state to valid region (clamp to bounds) to preserve compactness.

        This ensures the state space remains compact by clamping all state components
        to their valid bounds. If bounds are not set, returns state unchanged.

        Args:
            x: State vector (4,) or (N, 4) - [p_x, p_y, v_x, v_y]

        Returns:
            Projected state vector with all components clamped to [lower, upper]
        """
        if self._state_bounds is None:
            return x

        # Ensure x is an array
        x = np.asarray(x, dtype=float)

        # Clip to bounds to preserve compactness
        return np.clip(x, self.lower, self.upper)

    def _nominal_step(self, x, u):
        return x @ self.A.T + self.B @ u

    def f_bar(self, x, u):
        x = np.asarray(x, dtype=float)
        x_nom = self._nominal_step(x, u)
        return self._project_state(x_nom)

    def f(self, x, u, w):
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        x_nom = self._nominal_step(x, u)
        return self._project_state(x_nom + w)

class LIDAR:
    def __init__(self, fov=2 * np.pi, r_max=100, B=360):
        """
        :param fov: sensor field of view. Accepts radians (default) or degrees (> 2π).
        :param B: number of beams - evenly spaced, starting at 0 rad (along +x), CCW.
        :param r_max: max distance the robot can see. If no obstacle, laser end point = max_dist
        """
        self.r_max = r_max
        # Normalize FOV to radians if user passed degrees (common when using 360)
        fov_rad = np.deg2rad(fov) if fov > 2 * np.pi + 1e-6 else float(fov)
        self.fov = fov_rad
        self.B = int(B)
        self.resolution = self.fov / self.B
        # Evenly spaced angles in [0, fov), endpoint excluded to avoid duplicate at fov
        self.angles = np.linspace(0.0, self.fov, self.B, endpoint=False)

    def get_intersection(self, a1, a2, b1, b2):
        """
        Vectorized line segment intersection computation.

        :param a1: (x1,y1) or (N, 2) line segment 1 - starting position
        :param a2: (x1',y1') or (N, 2) line segment 1 - ending position
        :param b1: (x2,y2) or (N, 2) line segment 2 - starting position
        :param b2: (x2',y2') or (N, 2) line segment 2 - ending position
        :return: point of intersection(s), if intersect; None or array of Nones if do not intersect
        #adopted from https://github.com/LinguList/TreBor/blob/master/polygon.py
        """
        # Convert to arrays and ensure 2D
        a1 = np.asarray(a1)
        a2 = np.asarray(a2)
        b1 = np.asarray(b1)
        b2 = np.asarray(b2)

        single = a1.ndim == 1
        if single:
            a1 = a1[np.newaxis, :]
            a2 = a2[np.newaxis, :]
            b1 = b1[np.newaxis, :]
            b2 = b2[np.newaxis, :]

        # Vectorized perpendicular: perp([x, y]) = [-y, x]
        da = a2 - a1  # (N, 2)
        db = b2 - b1  # (N, 2)
        dp = a1 - b1  # (N, 2)

        dap = np.stack([-da[:, 1], da[:, 0]], axis=1)  # (N, 2)
        denom = np.sum(dap * db, axis=1)  # (N,)
        num = np.sum(dap * dp, axis=1)  # (N,)

        # Check for parallel lines (zero denominator)
        parallel_mask = np.abs(denom) < 1e-10

        # Compute intersections (avoid division by zero)
        t = np.where(parallel_mask, np.nan, num / denom)  # (N,)
        intersections = b1 + t[:, np.newaxis] * db  # (N, 2)

        # Check if intersections are within line segments
        delta = 1e-3
        a_min_x = np.minimum(a1[:, 0], a2[:, 0])  # (N,)
        a_max_x = np.maximum(a1[:, 0], a2[:, 0])
        a_min_y = np.minimum(a1[:, 1], a2[:, 1])
        a_max_y = np.maximum(a1[:, 1], a2[:, 1])

        b_min_x = np.minimum(b1[:, 0], b2[:, 0])
        b_max_x = np.maximum(b1[:, 0], b2[:, 0])
        b_min_y = np.minimum(b1[:, 1], b2[:, 1])
        b_max_y = np.maximum(b1[:, 1], b2[:, 1])

        condx_a = (a_min_x - delta <= intersections[:, 0]) & (intersections[:, 0] <= a_max_x + delta)
        condx_b = (b_min_x - delta <= intersections[:, 0]) & (intersections[:, 0] <= b_max_x + delta)
        condy_a = (a_min_y - delta <= intersections[:, 1]) & (intersections[:, 1] <= a_max_y + delta)
        condy_b = (b_min_y - delta <= intersections[:, 1]) & (intersections[:, 1] <= b_max_y + delta)

        valid_mask = ~parallel_mask & condx_a & condy_a & condx_b & condy_b

        # Set invalid intersections to NaN
        intersections[~valid_mask] = np.nan

        if single:
            if valid_mask[0]:
                return intersections[0]
            else:
                return None
        else:
            # Return array with NaN for invalid intersections
            return intersections

    def get_laser_ref(self, segments, robot_poses=None):
        """
        Vectorized laser reflection computation for multiple robot poses and angles.
        Fully vectorized version that processes all segments at once.

        :param segments: start and end points of all segments as ((x1,y1,x1',y1'),
                                (x2,y2,x2',y2'), (x3,y3,x3',y3'), (...))
               robot_poses: robot's pose(s) in the global coordinate system (N, 2) or (2,)
        :return: (N, B) array indicating the distance traveled by each laser beam
        """
        if robot_poses is None:
            robot_poses = np.array([[0.0, 0.0]])
        if robot_poses.ndim == 1:
            robot_poses = robot_poses[np.newaxis, :]

        N = robot_poses.shape[0]
        S = len(segments)  # Number of segments

        # Initialize all laser reflections to r_max
        dist_theta = self.r_max * np.ones((N, self.B))

        # Early return if no segments (no obstacles) - all rays hit max range
        if S == 0:
            return dist_theta

        # Precompute ray endpoints for all poses and angles
        # robot_poses: (N, 2), angles: (B,)
        # ray_dirs: (N, B, 2) = cos/sin for each pose-angle combination
        cos_angles = np.cos(self.angles)  # (B,)
        sin_angles = np.sin(self.angles)  # (B,)
        ray_dirs = np.stack([cos_angles, sin_angles], axis=1)  # (B, 2)

        # Expand for broadcasting: (N, 1, 2) + (1, B, 2) * r_max = (N, B, 2)
        robot_poses_expanded = robot_poses[:, np.newaxis, :]  # (N, 1, 2)
        ray_dirs_expanded = ray_dirs[np.newaxis, :, :]  # (1, B, 2)
        ray_endpoints = robot_poses_expanded + self.r_max * ray_dirs_expanded  # (N, B, 2)

        # Convert all segments to arrays upfront: (S, 4) -> (S, 2, 2)
        # Now we know S > 0, so this will create a proper 2D array
        segments_array = np.array(segments)  # (S, 4)
        seg_starts = segments_array[:, :2]  # (S, 2) - start points
        seg_ends = segments_array[:, 2:]  # (S, 2) - end points

        # Expand segments to (S, N, B, 2) for all combinations
        # seg_starts: (S, 1, 1, 2) -> (S, N, B, 2)
        seg_starts_expanded = seg_starts[:, np.newaxis, np.newaxis, :]  # (S, 1, 1, 2)
        seg_ends_expanded = seg_ends[:, np.newaxis, np.newaxis, :]  # (S, 1, 1, 2)

        # Expand robot poses: (N, 1, 2) -> reshape to (1, N, 1, 2) -> broadcast to (S, N, B, 2)
        robot_expanded = robot_poses_expanded.reshape(1, N, 1, 2)  # (1, N, 1, 2)
        # ray_endpoints is already (N, B, 2), expand to (1, N, B, 2)
        ray_expanded = ray_endpoints[np.newaxis, :, :, :]  # (1, N, B, 2)

        # Broadcast to (S, N, B, 2) for all combinations
        seg_starts_all = np.broadcast_to(seg_starts_expanded, (S, N, self.B, 2))  # (S, N, B, 2)
        seg_ends_all = np.broadcast_to(seg_ends_expanded, (S, N, self.B, 2))  # (S, N, B, 2)
        robot_all = np.broadcast_to(robot_expanded, (S, N, self.B, 2))  # (S, N, B, 2)
        ray_all = np.broadcast_to(ray_expanded, (S, N, self.B, 2))  # (S, N, B, 2)

        # Flatten to (S*N*B, 2) for vectorized intersection computation
        seg_starts_flat = seg_starts_all.reshape(-1, 2)  # (S*N*B, 2)
        seg_ends_flat = seg_ends_all.reshape(-1, 2)  # (S*N*B, 2)
        robot_flat = robot_all.reshape(-1, 2)  # (S*N*B, 2)
        ray_flat = ray_all.reshape(-1, 2)  # (S*N*B, 2)

        # Compute all intersections at once: (S*N*B, 2)
        intersections = self.get_intersection(seg_starts_flat, seg_ends_flat,
                                              robot_flat, ray_flat)  # (S*N*B, 2)

        if intersections is not None:
            # Check for valid intersections (not NaN)
            valid = ~np.isnan(intersections).any(axis=1)  # (S*N*B,)

            if valid.any():
                # Compute distances for valid intersections
                distances = np.sqrt(np.sum((intersections - robot_flat)**2, axis=1))  # (S*N*B,)

                # Reshape to (S, N, B) to find minimum across segments
                distances_3d = distances.reshape(S, N, self.B)  # (S, N, B)
                valid_3d = valid.reshape(S, N, self.B)  # (S, N, B)

                # Set invalid intersections to infinity so they don't affect minimum
                distances_3d[~valid_3d] = np.inf

                # Find minimum distance across all segments for each (pose, angle) pair
                min_distances = np.min(distances_3d, axis=0)  # (N, B)

                # Update dist_theta where we found valid intersections closer than r_max
                update_mask = (min_distances < dist_theta) & (min_distances < np.inf)
                dist_theta[update_mask] = min_distances[update_mask]

        return dist_theta

    def get_laser_ref_per_segment(self, segments, robot_poses=None):
        """
        Vectorized laser reflection that returns per-segment distances.
        Useful for batched processing where we need to group segments by map.

        Returns:
            distances_3d: (S, N, B) array of distances for each segment
            valid_3d: (S, N, B) boolean array indicating valid intersections
        """
        if robot_poses is None:
            robot_poses = np.array([[0.0, 0.0]])
        if robot_poses.ndim == 1:
            robot_poses = robot_poses[np.newaxis, :]

        N = robot_poses.shape[0]
        S = len(segments)

        if S == 0:
            return np.full((0, N, self.B), np.inf), np.zeros((0, N, self.B), dtype=bool)

        # Same computation as get_laser_ref but return per-segment results
        cos_angles = np.cos(self.angles)
        sin_angles = np.sin(self.angles)
        ray_dirs = np.stack([cos_angles, sin_angles], axis=1)

        robot_poses_expanded = robot_poses[:, np.newaxis, :]
        ray_dirs_expanded = ray_dirs[np.newaxis, :, :]
        ray_endpoints = robot_poses_expanded + self.r_max * ray_dirs_expanded

        segments_array = np.array(segments)
        seg_starts = segments_array[:, :2]
        seg_ends = segments_array[:, 2:]

        seg_starts_expanded = seg_starts[:, np.newaxis, np.newaxis, :]
        seg_ends_expanded = seg_ends[:, np.newaxis, np.newaxis, :]
        robot_expanded = robot_poses_expanded.reshape(1, N, 1, 2)
        ray_expanded = ray_endpoints[np.newaxis, :, :, :]

        seg_starts_all = np.broadcast_to(seg_starts_expanded, (S, N, self.B, 2))
        seg_ends_all = np.broadcast_to(seg_ends_expanded, (S, N, self.B, 2))
        robot_all = np.broadcast_to(robot_expanded, (S, N, self.B, 2))
        ray_all = np.broadcast_to(ray_expanded, (S, N, self.B, 2))

        seg_starts_flat = seg_starts_all.reshape(-1, 2)
        seg_ends_flat = seg_ends_all.reshape(-1, 2)
        robot_flat = robot_all.reshape(-1, 2)
        ray_flat = ray_all.reshape(-1, 2)

        intersections = self.get_intersection(seg_starts_flat, seg_ends_flat, robot_flat, ray_flat)

        if intersections is not None:
            valid = ~np.isnan(intersections).any(axis=1)
            distances = np.sqrt(np.sum((intersections - robot_flat)**2, axis=1))
            distances_3d = distances.reshape(S, N, self.B)
            valid_3d = valid.reshape(S, N, self.B)
            distances_3d[~valid_3d] = np.inf
            return distances_3d, valid_3d
        else:
            return np.full((S, N, self.B), np.inf), np.zeros((S, N, self.B), dtype=bool)

    def g_bar(self, X, m, map_obj: BaseMap):
        """
        Deterministic observation model - ray casting to determine distances to obstacles.

        Handles both single and batched maps by normalizing shapes.

        Args:
            X: Array of robot positions (m_n, 2)
            m: Either:
               - Single occupancy grid (H, W)
               - Batched occupancy grids (len_M, H, W)
            map_obj: Map object with occupancy_map attribute containing left_lower, right_upper

        Returns:
            y_star: Array of distances
               - Single map: (m_n, B)
               - Batched maps: (m_n, len_M, B)
        """
        X = np.asarray(X)
        m = np.asarray(m)

        single_pose = X.ndim == 1
        if single_pose:
            X = X[np.newaxis, :]

        single_map = m.ndim == 2
        if single_map:
            m = m[np.newaxis, :, :]

        y_star = self.g_bar_batched(X, m, map_obj)

        if single_map:
            y_star = y_star[:, 0, :]
        return y_star[0] if single_pose else y_star

    def g_bar_localization(self, X, obstacle_segments):
        """
        Deterministic observation model for localization - ray casting using pre-computed obstacle segments.

        For localization, the map is known and obstacle segments are pre-computed.
        This avoids recomputing obstacle segments from the map on every call.

        Args:
            X: Array of robot positions (m_n, 2)
            obstacle_segments: Pre-computed list of obstacle segments from the known map

        Returns:
            y_star: Array of distances (m_n, B)
        """
        X = X[np.newaxis, :] if X.ndim == 1 else X
        y_star = self.get_laser_ref(obstacle_segments, X)
        return y_star if y_star.ndim == 2 else y_star[np.newaxis, :]

    def get_laser_ref_batched(self, segments_list, segment_map_indices, robot_poses):
        """
        Batched laser reflection computation that processes segments from multiple maps.
        FULLY VECTORIZED - NO LOOPS. Processes all segments together, then groups by map.

        Args:
            segments_list: List of all segments from all maps
            segment_map_indices: Array (total_segments,) indicating which map each segment belongs to
            robot_poses: Robot poses (m_n, 2)

        Returns:
            y_star_all: (m_n, len_M, B) array with distances for each map
        """
        if robot_poses.ndim == 1:
            robot_poses = robot_poses[np.newaxis, :]

        m_n = robot_poses.shape[0]
        len_M = int(segment_map_indices.max() + 1) if len(segment_map_indices) > 0 else 0
        B = self.B

        # Initialize output to r_max
        y_star_all = np.full((m_n, len_M, B), self.r_max, dtype=np.float64)

        if len(segments_list) == 0:
            return y_star_all

        # Convert to GPU array
        segment_map_indices = np.asarray(segment_map_indices)

        # Process ALL segments together - fully vectorized, no loops
        distances_all, _ = self.get_laser_ref_per_segment(segments_list, robot_poses)
        # distances_all: (S, m_n, B) where S is total number of segments

        # Group by map and take minimum per map using FULLY VECTORIZED operations - NO LOOPS
        # Sort segments by map index to enable vectorized grouping
        sort_indices = np.argsort(segment_map_indices)
        sorted_distances = distances_all[sort_indices, :, :]  # (S, m_n, B)
        sorted_map_indices = segment_map_indices[sort_indices]  # (S,)

        # Find boundaries between maps using vectorized operations
        # Convert Python bool to backend array type (CuPy if using GPU)
        first_true = np.array([True], dtype=bool)
        map_diffs = sorted_map_indices[1:] != sorted_map_indices[:-1]
        map_changes = np.concatenate((first_true, map_diffs))
        segment_boundaries = np.where(map_changes)[0]  # Indices where map changes
        # Convert Python int to backend array type
        end_idx = np.array([len(sorted_map_indices)], dtype=np.int64)
        segment_boundaries = np.concatenate((segment_boundaries, end_idx))  # Add end

        # Compute minimum per map using vectorized reduceat-style operations
        # Process all maps: extract segments per map and reduce
        unique_maps, map_start_indices = np.unique(sorted_map_indices, return_index=True)
        end_idx_array = np.array([len(sorted_map_indices)], dtype=map_start_indices.dtype)
        map_end_indices = np.concatenate((map_start_indices[1:], end_idx_array))

        # Vectorized reduction: compute minimum per map
        # Reshape to (S, m_n*B) for easier manipulation
        sorted_distances_2d = sorted_distances.reshape(len(sorted_distances), m_n * B)  # (S, m_n*B)

        # TODO: Replace with np.maximum.reduceat once CuPy adds support for reduceat:
        # negated_distances = -sorted_distances_2d  # (S, m_n*B)
        # max_negated = np.maximum.reduceat(negated_distances, map_start_indices, axis=0)  # (len_M, m_n*B)
        # min_distances_2d = -max_negated  # (len_M, m_n*B) - this is minimum per map
        # min_distances_2d = np.minimum(min_distances_2d, self.r_max)  # Clamp to r_max
        # CuPy issue: https://github.com/cupy/cupy/issues (reduceat not yet implemented for maximum/minimum)
        # Workaround: Use vectorized slicing + np.min per map
        # Note: np.min() on slices is fully vectorized on GPU - the loop is only O(len_M) for organization

        # Preallocate output array
        min_distances_2d = np.full((len(unique_maps), m_n * B), self.r_max, dtype=np.float64)

        # Process each map: extract segments and compute minimum
        # The computation (np.min) is fully vectorized - loop is only for organizing variable-length groups
        for i, (start_idx, end_idx) in enumerate(zip(map_start_indices, map_end_indices)):
            # Extract this map's segments: (num_segments_map, m_n*B)
            map_segments_2d = sorted_distances_2d[start_idx:end_idx, :]
            # Compute minimum across segments: (m_n*B,) - fully vectorized GPU operation
            min_distances_2d[i, :] = np.min(map_segments_2d, axis=0)

        # Clamp to r_max
        min_distances_2d = np.minimum(min_distances_2d, self.r_max)  # (len_M, m_n*B)

        # Reshape back and transpose to (m_n, len_M, B)
        map_results = min_distances_2d.reshape(len(unique_maps), m_n, B)  # (len_M, m_n, B)
        y_star_all = map_results.transpose(1, 0, 2)  # (m_n, len_M, B)

        # Reorder to match original map order (unique_maps might not be 0..len_M-1)
        # Use advanced indexing to assign to correct map positions
        y_star_all_reordered = np.full((m_n, len_M, B), self.r_max, dtype=np.float64)
        y_star_all_reordered[:, unique_maps, :] = y_star_all

        return y_star_all_reordered

    def g_bar_batched(self, X, M, map_obj: BaseMap):
        """
        Batched deterministic observation model - ray casting for multiple maps simultaneously.

        Processes all maps using vectorized operations. Uses list comprehension for
        obstacle extraction (necessary due to connected components algorithm), but all
        ray casting operations are fully vectorized on GPU.

        Args:
            X: Array of robot positions (m_n, 2)
            M: Array of occupancy grids (len_M, H, W)
            map_obj: Map object with occupancy_map attribute

        Returns:
            y_star_all: Array of distances (m_n, len_M, B)
        """
        X = X[np.newaxis, :] if X.ndim == 1 else X
        len_M = M.shape[0]

        # Extract obstacles for all maps - list comprehension is necessary here
        # because connected components labeling is inherently per-map
        # But this keeps data on GPU (M[map_idx] is a slice, stays on GPU)
        map_segments_list = [self.get_obstacles_from_map(M[map_idx], map_obj) for map_idx in range(len_M)]

        # Flatten segments with map indices - use list operations but convert to GPU arrays
        all_segments = []
        segment_map_indices_list = []

        # Build segment list and indices - this is necessary for grouping
        for map_idx, segments in enumerate(map_segments_list):
            all_segments.extend(segments)
            segment_map_indices_list.extend([map_idx] * len(segments))

        # Convert to GPU array for vectorized operations
        segment_map_indices = np.array(
            segment_map_indices_list) if segment_map_indices_list else np.array([], dtype=np.int32)

        # Use batched laser reflection computation
        return self.get_laser_ref_batched(all_segments, segment_map_indices, X)

    def g(self, X, m, v, map_obj: BaseMap):
        """
        Stochastic observation model: g(x, m, v) = g_bar(x, m) + v

        Args:
            X: Array of robot positions (m_n, 2)
            m: Occupancy grid (H, W)
            v: Observation noise (m_n, B)
            map_obj: Map object with occupancy_map attribute

        Returns:
            y_noisy: Noisy observations (m_n, B)
        """
        v = v[np.newaxis, :] if X.ndim == 1 else v
        y_ideal = self.g_bar(X, m, map_obj)
        y_noisy = y_ideal + v
        return y_noisy

    def get_obstacles_from_map(self, m, map_obj: BaseMap):
        """
        Convert map to obstacle segments.

        Args:
            m: Map representation (shape depends on map type)
            map_obj: Map object implementing BaseMap interface

        Returns:
            List of obstacle segments
        """
        # Try to use the map's get_obstacle_segments method first
        if hasattr(map_obj, 'get_obstacle_segments'):
            segments = map_obj.get_obstacle_segments(m)
            if segments:
                return segments

        # Fallback: for occupancy grids, use connected components
        if hasattr(map_obj, 'occupancy_map'):
            # Find connected components of occupied cells
            connected_components = self._find_connected_components(m)

            all_obstacle_segments = []

            # Create obstacles for each connected component
            for component in connected_components:
                if len(component) > 0:
                    obstacle_segments = self._create_obstacle_from_component(
                        component, m.shape, map_obj)
                    all_obstacle_segments.extend(obstacle_segments)
            return all_obstacle_segments

        # If no method available, return empty list
        return []

    def _find_connected_components(self, m):
        """
        Find connected components of occupied cells (value = 1) in the occupancy grid.
        Uses 4-connectivity (cells sharing a side are connected) with ndimage.label (CuPy required).

        Args:
            m: (H, W) array of occupancy grid

        Returns:
            List of connected components, where each component is a list of (i, j) coordinates

        Raises:
            RuntimeError: If CuPy operations fail. CuPy is required for this operation.
        """
        # Convert to array if needed, ensuring it's the correct backend type
        import numpy as _numpy

        # Check which ndimage module we're using by checking the module name
        # cupyx.scipy.ndimage requires CuPy arrays, scipy.ndimage requires NumPy arrays
        ndimage_module_name = getattr(ndimage, '__name__', '')
        is_cupy_ndimage = 'cupyx' in ndimage_module_name

        # Convert to appropriate array type
        if is_cupy_ndimage:
            # Using CuPy's ndimage - requires CuPy array
            import cupy as _cupy
            if isinstance(m, _numpy.ndarray):
                # Explicitly convert NumPy array to CuPy array
                m = _cupy.asarray(m)
            elif not hasattr(m, 'device'):  # Not a CuPy array (check for 'device' attribute)
                # Try conversion via backend np first
                m = np.array(m)
                # If still not CuPy (doesn't have 'device'), force explicit conversion
                if not hasattr(m, 'device'):
                    m = _cupy.asarray(_numpy.asarray(m))
        else:
            # Using scipy's ndimage - requires NumPy array
            # Check if it's a CuPy array by trying to import cupy and check type
            try:
                import cupy as _cupy
                if isinstance(m, _cupy.ndarray):
                    m = m.get()
                else:
                    m = _numpy.asarray(m)
            except ImportError:
                # CuPy not available, so it must be NumPy
                m = _numpy.asarray(m)

        # Label connected components
        # Returns (labeled_array, num_features)
        result = ndimage.label(m)
        if isinstance(result, tuple):
            labeled, num_features = result
        else:
            # Fallback: if it returns just the array, compute num_features
            labeled = result
            num_features = int(labeled.max())

        # Extract components
        components = []
        for label in range(1, num_features + 1):
            component = np.argwhere(labeled == label)
            # Convert to list of tuples (handles both CPU and GPU arrays)
            if hasattr(component, 'get'):
                component = component.get()
            components.append([tuple(coord) for coord in component])

        return components

    def _create_obstacle_from_component(self, component, grid_shape, map_obj: LidarGridMapVec):
        """
        Create obstacle segments from a connected component of cells.

        Args:
            component: List of (i, j) coordinates representing connected cells
            grid_shape: Shape of the grid (H, W)
            map_obj: Map object with occupancy_map attribute

        Returns:
            List of line segments representing the obstacle boundary
        """
        if not component:
            return []

        # Use the actual resolution from the map object
        H, W = grid_shape
        left_lower = map_obj.occupancy_map.left_lower
        right_upper = map_obj.occupancy_map.right_upper

        cell_w = map_obj.occupancy_map.resolution
        cell_h = map_obj.occupancy_map.resolution

        # Calculate bounding box
        i_coords = [cell[0] for cell in component]
        j_coords = [cell[1] for cell in component]

        min_i, max_i = min(i_coords), max(i_coords)
        min_j, max_j = min(j_coords), max(j_coords)
        # Calculate dimensions in number of cells
        num_cells_i = max_i - min_i + 1
        num_cells_j = max_j - min_j + 1

        # Calculate physical dimensions
        dx = num_cells_j * cell_w  # width (columns)
        dy = num_cells_i * cell_h  # height (rows)

        # Calculate centroid at cell centers (account for 0.5 offset)
        # IMPORTANT: Array row 0 is the TOP of the map. World y increases upward from bottom.
        # Convert row index i to world row by flipping: i_world = H - 1 - i
        res_x = cell_w
        res_y = cell_h

        center_i = (min_i + max_i + 1) / 2.0
        center_j = (min_j + max_j + 1) / 2.0

        # Map (i,j) -> (x,y) using cell edge convention, flipping row index
        i_world = H - center_i
        j_world = center_j
        centroid = left_lower + np.array([j_world * res_x,
                                          i_world * res_y])

        # Create obstacle and get its line segments
        obstacle = Obstacle(centroid, dx=dx, dy=dy, angle=0)
        return obstacle._Obstacle__get_points(centroid)  # tuple: (bottom, top, left, right)


class LIDAR_with_Noise(LIDAR):
    def __init__(self, fov, r_max, B, noise_std):
        super().__init__(fov, r_max, B)
        self.noise_std = noise_std

    def get_laser_ref(self, segments, robot_pose=None):
        if robot_pose is None:
            robot_pose = np.array([0.0, 0.0])
        angles, dist_theta = super().get_laser_ref(segments, robot_pose)
        dist_theta += np.random.normal(0,
                                       self.noise_std, size=dist_theta.shape)
        return angles, dist_theta
