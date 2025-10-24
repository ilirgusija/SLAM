import numpy as np
from scipy.optimize import linprog


# Define probability metrics
def tvd(p, q):
    return 2 * float(np.max(np.abs(p - q)))

def W1_x(p_state, q_state, X_points):
    # Ensure valid distributions
    p_state = np.asarray(p_state, dtype=np.float64)
    q_state = np.asarray(q_state, dtype=np.float64)
    p_state = p_state / p_state.sum()
    q_state = q_state / q_state.sum()

    n = p_state.size
    # Cost matrix (Euclidean distance)
    C = np.linalg.norm(X_points[:, None, :] - X_points[None, :, :], axis=2)
    c = C.ravel()

    # Flow variables f_{ij} in R^{n*n}
    # Constraints: for each i, sum_j f_{ij} = p_i; for each j, sum_i f_{ij} = q_j; f_{ij} >= 0
    A_eq_rows = []
    b_eq = []
    # Row sums equal p
    for i in range(n):
        row = np.zeros(n * n)
        row[i * n:(i + 1) * n] = 1.0
        A_eq_rows.append(row)
        b_eq.append(p_state[i])
    # Column sums equal q
    for j in range(n):
        col = np.zeros(n * n)
        col[j::n] = 1.0
        A_eq_rows.append(col)
        b_eq.append(q_state[j])
    A_eq = np.vstack(A_eq_rows)
    b_eq = np.array(b_eq)

    bounds = [(0.0, None) for _ in range(n * n)]
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if not res.success:
        raise RuntimeError(f"Wasserstein-1 LP failed: {res.message}")
    return float(res.fun)

def W1_m(p_map, q_map, M_points):
    """
    Wasserstein-1 distance on the map space with Hamming ground metric.

    Given distributions p_map and q_map over binary maps M_points \in {0,1}^{n x d}
    (each row is a flattened HxW map), the W1 distance under Hamming cost
    equals the sum over bits of the absolute difference of bit marginals:

        W1_Hamming(p, q) = sum_{k=1..d} | E_p[m_k] - E_q[m_k] |

    This avoids solving a large LP over 2^{HW} support.
    """
    # Ensure valid distributions
    p_map = np.asarray(p_map, dtype=np.float64)
    q_map = np.asarray(q_map, dtype=np.float64)
    p_sum = p_map.sum()
    q_sum = q_map.sum()
    if p_sum == 0 or q_sum == 0:
        return 0.0
    p_map = p_map / p_sum
    q_map = q_map / q_sum

    # Ensure binary map points as float for dot products
    M_points = np.asarray(M_points, dtype=np.float64)
    # Bitwise marginals under p and q: (d,)
    p_bits = p_map @ M_points
    q_bits = q_map @ M_points
    return float(np.sum(np.abs(p_bits - q_bits)))


def W1_state_simple(p_state, q_state, X_points):
    """
    Simple sorting-based Wasserstein-1 distance on state space.

    - If X_points is 1D: compute cumulative L1 between sorted CDFs.
    - If X_points is 2D (Nx2): compute W1 along x and y independently and sum.

    This is a fast heuristic suitable for grid-like quantizers and
    diagnostics; for exact Wasserstein on arbitrary supports use W1_x.
    """
    p = np.asarray(p_state, dtype=np.float64)
    q = np.asarray(q_state, dtype=np.float64)
    pts = np.asarray(X_points, dtype=np.float64)

    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum == 0 or q_sum == 0:
        return 0.0
    p = p / p_sum
    q = q / q_sum

    if pts.ndim == 1:
        order = np.argsort(pts)
        p_s = p[order]
        q_s = q[order]
        return float(np.sum(np.abs(np.cumsum(p_s) - np.cumsum(q_s))))

    if pts.ndim == 2 and pts.shape[1] == 2:
        # x-axis
        order_x = np.argsort(pts[:, 0])
        p_x = p[order_x]
        q_x = q[order_x]
        w1_x = np.sum(np.abs(np.cumsum(p_x) - np.cumsum(q_x)))

        # y-axis
        order_y = np.argsort(pts[:, 1])
        p_y = p[order_y]
        q_y = q[order_y]
        w1_y = np.sum(np.abs(np.cumsum(p_y) - np.cumsum(q_y)))

        return float(w1_x + w1_y)

    raise ValueError("X_points must be 1D or 2D with shape (N,2)")
