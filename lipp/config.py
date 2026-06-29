"""Configuration, physical constants, and the per-instance problem bundle.

`ExperimentConfig` holds the knobs that change between runs; `Problem` holds the
data of a single generated instance shared by all three planners.
"""
from dataclasses import dataclass, fields
import numpy as np

# --- Physical / model constants (fixed across every experiment) ---
SIGMA2 = 0.05          # measurement-noise variance of one unit sample (sigma^2)
UPHILL_SCALE = 0.05    # cost-slope alpha for uphill edges   (d_uv = d * (1 + alpha*dh))
DOWNHILL_SCALE = 0.05  # cost-slope alpha for downhill edges
A_MAX_FALLBACK = 5.0   # bound on estimator coefficients if the K_VV solve fails


@dataclass
class ExperimentConfig:
    """All tunable parameters for one experiment."""
    # Graph generation
    exp_id: int = 0
    seed: int = 1234
    density: float = 1.0        # fraction of non-critical edges kept, in (0, 1]
    n_vertices: int = 10        # number of sampling vertices
    n_test: int = 5             # number of test locations

    # Solver
    time_limit: int = 100       # Gurobi time limit (s)

    # Robot / physics (shared by all methods)
    R_0: float = 5.0            # base robot mass without samples
    unit_mass: float = 1.0      # lambda: mass of one unit sample

    # LIPP
    B: float = 5.0              # energy budget
    S_max: int = 3              # max samples per vertex

    # C-IPP and Greedy
    dist_lim: float = 2.0       # distance budget
    S: int = 3                  # fixed samples taken at each visited vertex

    # Output
    out_dir: str = "data/unified"

    @classmethod
    def from_dict(cls, d):
        """Build a config from a plain dict, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Problem:
    """One generated instance: graph + GP setup shared by all planners."""
    V: np.ndarray          # sampling-vertex coordinates  (n x 2)
    T: np.ndarray          # test-location coordinates     (m x 2)
    edges: list            # directed edge list
    edge_cost: dict        # (u, v) -> slope-adjusted traversal cost d_uv
    K_VV: np.ndarray       # kernel over sampling vertices (n x n)
    K_TV: np.ndarray       # kernel test-vs-vertices       (m x n)
    K_TT: np.ndarray       # kernel over test locations    (m x m)

    @property
    def n_vertices(self):
        return self.V.shape[0]

    @property
    def n_test(self):
        return self.T.shape[0]

    @property
    def start(self):
        return 0

    @property
    def target(self):
        return self.n_vertices - 1
