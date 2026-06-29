"""Path-evaluation metrics: energy, travel cost, posterior variance, and RMSE.

These are solver-agnostic: each planner produces a path and a per-vertex sample
allocation, and these functions score it.
"""
import numpy as np
from .config import SIGMA2
from .geometry import th_level

UNVISITED_NOISE = 1e6  # effective noise variance for vertices with no samples


def extract_path_from_edges(selected_edges, start, target, n_vertices):
    """Walk start -> target following the selected directed edges."""
    path, cur = [start], start
    for _ in range(n_vertices + 5):       # guard against malformed edge sets
        if cur == target:
            break
        nxt = next((v for u, v in selected_edges if u == cur), None)
        if nxt is None:
            break
        path.append(nxt)
        cur = nxt
    return path


def compute_energy_along_path(path, s_chosen, edge_cost, R_0, unit_mass):
    """Load-aware energy  E = sum_j d(v_j, v_{j+1}) * R_j   (paper, Sec. III-B).

    R_j is the robot mass while traversing edge j: the base mass R_0 plus the
    mass of every sample collected at v_1 .. v_j.
    """
    if not path or len(path) < 2:
        return 0.0
    energy, load = 0.0, 0.0
    for u, v in zip(path, path[1:]):
        load += unit_mass * float(s_chosen[u])        # samples picked up at u
        energy += edge_cost[(u, v)] * (R_0 + load)
    return float(energy)


def compute_travel_cost(path, edge_cost):
    """Plain path length  D = sum_j d(v_j, v_{j+1})."""
    if not path or len(path) < 2:
        return 0.0
    return float(sum(edge_cost[(u, v)] for u, v in zip(path, path[1:])))


def compute_posterior_variance(K_VV, K_TV, K_TT, s_chosen, M=None):
    """trace(M * posterior_cov) for a given sampling allocation.

    Per-vertex noise is sigma^2 / l_v (variance reduction from averaging);
    unvisited vertices get a huge variance so they contribute nothing.
    Returns (post_var, posterior_cov), or (None, None) on a singular system.
    """
    n = K_VV.shape[0]
    M = np.eye(K_TT.shape[0]) if M is None else M
    noise = np.full(n, UNVISITED_NOISE)
    for v in range(n):
        if s_chosen[v] > 0:
            noise[v] = SIGMA2 / float(s_chosen[v])
    try:
        K_inv = np.linalg.solve(K_VV + np.diag(noise), np.eye(n))
        cov = K_TT - K_TV @ K_inv @ K_TV.T
        return float(np.trace(M @ cov)), cov
    except Exception:
        return None, None


def compute_prior_variance(K_TT, M=None):
    """trace(M * K_TT): posterior variance with no observations."""
    M = np.eye(K_TT.shape[0]) if M is None else M
    return float(np.trace(M @ K_TT))


def compute_rmse(A_est, V, T):
    """RMSE of the linear estimator f(T) ~= A_est @ f(V) against ground truth."""
    if A_est is None:
        return None
    A_est = np.asarray(A_est)
    fT_est = A_est @ th_level(V)
    return float(np.sqrt(np.mean((th_level(T) - fT_est) ** 2)))
