"""Field generation and graph construction for LIPP experiments.

Defines the synthetic lunar-rover environment used in the paper: a Th-concentration
field over a bounded 2-D region, an elevation profile that makes traversal cost
slope-dependent, an RBF kernel for the GP prior, and the directed graph the
planners operate on.
"""
import numpy as np
from .config import UPHILL_SCALE, DOWNHILL_SCALE


def random_points_in_circle(n, radius=0.45, center=(0.5, 0.5), rng=None):
    """Sample `n` points uniformly inside a disk."""
    rng = rng if rng is not None else np.random
    r = radius * np.sqrt(rng.rand(n))
    theta = 2 * np.pi * rng.rand(n)
    return np.column_stack([center[0] + r * np.cos(theta),
                            center[1] + r * np.sin(theta)])


def elevation(X):
    """Smooth synthetic elevation h(x, y); drives the slope-dependent edge cost."""
    x, y = X[:, 0], X[:, 1]
    return (0.25 * np.sin(2 * np.pi * (x - 0.2)) * np.cos(2 * np.pi * (y + 0.1))
            + 0.4 * np.exp(-6 * ((x - 0.45) ** 2 + (y - 0.55) ** 2))
            + 0.3 * np.exp(-8 * ((x - 0.7) ** 2 + (y - 0.4) ** 2))
            + 0.04 * np.sin(30 * x) * np.cos(20 * y))


def th_level(X):
    """Ground-truth scalar field f(x, y): Th surface concentration."""
    x, y = X[:, 0], X[:, 1]
    return (np.cos(3 * np.pi * x) * np.sin(3 * np.pi * y)
            + 0.5 * np.exp(-10 * ((x - 0.3) ** 2 + (y - 0.7) ** 2)))


def rbf_kernel(X1, X2, lengthscale=0.2):
    """Squared-exponential kernel matrix k(X1, X2)."""
    sq1 = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
    sq2 = np.sum(X2 ** 2, axis=1).reshape(1, -1)
    return np.exp(-0.5 * (sq1 + sq2 - 2 * X1 @ X2.T) / lengthscale ** 2)


def make_edges(n_vertices, start, goal, rng, density=0.5):
    """Build the directed edge list.

    Start and goal are fully connected to/from every other vertex; the
    remaining undirected pairs are kept with probability ~`density` and made
    bidirectional. Order-preserving de-duplication gives the final list.
    """
    critical = [(i, j)
                for i in range(n_vertices) for j in range(n_vertices)
                if i != j and ({i, j} & {start, goal})]

    other_pairs = [(i, j)
                   for i in range(n_vertices) for j in range(i + 1, n_vertices)
                   if not ({i, j} & {start, goal})]
    rng.shuffle(other_pairs)
    keep = int(round(density * len(other_pairs)))

    non_critical = []
    for i, j in other_pairs[:keep]:
        non_critical += [(i, j), (j, i)]

    return list(dict.fromkeys(critical + non_critical))


def build_edge_dicts(V, Vz, edges):
    """Map each directed edge to its slope-adjusted traversal cost d_uv.

        d_uv = ||V_u - V_v|| * (1 + alpha * (h_v - h_u)),

    with alpha = UPHILL_SCALE on ascending edges and DOWNHILL_SCALE on
    descending ones (equal by default).
    """
    edge_cost = {}
    for i, j in edges:
        dist = float(np.linalg.norm(V[i] - V[j]))
        elev_diff = Vz[j] - Vz[i]
        slope = UPHILL_SCALE if elev_diff > 0 else DOWNHILL_SCALE
        edge_cost[(i, j)] = dist * (1.0 + slope * elev_diff)
    return edge_cost
