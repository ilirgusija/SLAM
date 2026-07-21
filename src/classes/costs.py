"""Shared stage-cost primitives for both approximation forks.

Both belief-quantized MDPs and finite-memory information MDPs evaluate
    rho(b, u) = lam * r(b) + c_effort(u)
on a *belief* b. Finite-memory policies obtain b via Psi(b_prior, h) before
calling these same costs — they do not define a separate cost family.
"""

from __future__ import annotations

from typing import Any

from ..utils.array_backend import np, is_cupy
import numpy as _numpy

if is_cupy:
    from cupyx.scipy.special import entr
else:
    from scipy.special import entr


def control_effort(U: Any) -> Any:
    """Batched control effort c_effort(u) = ||u||_2 (matches BasePOMDP)."""
    U = np.asarray(U, dtype=float)
    if U.ndim == 1:
        U = U[np.newaxis, :]
    return np.linalg.norm(U, axis=1)


def shannon_entropy_rows(B_flat: Any) -> Any:
    """Row-wise Shannon entropy for flattened beliefs of shape (n_B, d)."""
    B_flat = np.asarray(B_flat)
    B_np = B_flat.get() if hasattr(B_flat, "get") else _numpy.asarray(B_flat)
    if _numpy.any(_numpy.isnan(B_np)) or _numpy.any(_numpy.isinf(B_np)):
        raise ValueError("Beliefs must not contain NaN or Inf.")
    H_vals = np.sum(entr(B_flat), axis=1)
    H_np = H_vals.get() if hasattr(H_vals, "get") else _numpy.asarray(H_vals)
    if _numpy.any(_numpy.isnan(H_np) | _numpy.isinf(H_np)):
        raise ValueError("Entropy produced NaN or Inf; check belief rows.")
    return H_vals


def compose_rho(r: Any, effort: Any, lam: float = 1.0) -> Any:
    """rho(b, u) = lam * r(b) + c_effort(u), shape (n_B, n_u)."""
    r = np.atleast_1d(np.asarray(r, dtype=float))
    effort = np.atleast_1d(np.asarray(effort, dtype=float))
    return float(lam) * r[:, np.newaxis] + effort[np.newaxis, :]


def stage_cost(mdp: Any, belief: Any, u: Any, lam: float = 1.0) -> float:
    """Evaluate shared rho on a single belief and action via the MDP's methods.

    Requires ``mdp.r_exploration`` and ``mdp.c_effort`` (BeliefMDP_n / POMDP API).
    """
    b = np.asarray(belief, dtype=float)
    r = np.asarray(mdp.r_exploration(b), dtype=float).ravel()
    u_arr = np.asarray(u, dtype=float)
    if u_arr.ndim == 1:
        u_arr = u_arr[np.newaxis, :]
    effort = np.asarray(mdp.c_effort(u_arr), dtype=float).ravel()
    return float(float(lam) * float(r[0]) + float(effort[0]))
