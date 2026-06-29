"""Top-level experiment runner.

Generates one shared instance from a config, runs LIPP + C-IPP + Greedy on it,
saves a JSON record and a compact NPZ, and prints a summary table.
"""
import os
import json
import numpy as np

from .config import ExperimentConfig, Problem
from .geometry import (random_points_in_circle, elevation, rbf_kernel,
                       make_edges, build_edge_dicts)
from .metrics import compute_prior_variance
from .solvers import run_lipp, run_cipp, run_greedy


def build_problem(config):
    """Sample one shared graph + GP setup from the config's seed."""
    rng = np.random.RandomState(config.seed)
    n, m = config.n_vertices, config.n_test
    start, target = 0, n - 1

    V = random_points_in_circle(n, rng=rng)
    T = random_points_in_circle(m, rng=rng)
    Vz = elevation(V)
    edges = make_edges(n, start, target, rng, density=config.density)
    edge_cost = build_edge_dicts(V, Vz, edges)

    return Problem(V=V, T=T, edges=edges, edge_cost=edge_cost,
                   K_VV=rbf_kernel(V, V),
                   K_TV=rbf_kernel(T, V),
                   K_TT=rbf_kernel(T, T))


def run_experiment(config):
    """Run all three planners on one shared instance; save and return results."""
    if isinstance(config, dict):
        config = ExperimentConfig.from_dict(config)
    os.makedirs(config.out_dir, exist_ok=True)

    problem = build_problem(config)
    prior_var = compute_prior_variance(problem.K_TT)

    print(f"[exp {config.exp_id}] LIPP ...");   lipp = run_lipp(problem, config)
    print(f"[exp {config.exp_id}] C-IPP ...");  cipp = run_cipp(problem, config)
    print(f"[exp {config.exp_id}] Greedy ..."); greedy = run_greedy(problem, config)

    result = {
        "exp_id": config.exp_id, "seed": config.seed,
        "n_vertices": config.n_vertices, "n_test": config.n_test,
        "density": config.density, "R_0": config.R_0,
        "unit_mass": config.unit_mass, "B": config.B, "S_max": config.S_max,
        "dist_lim": config.dist_lim, "S": config.S,
        "V": problem.V.tolist(), "T": problem.T.tolist(),
        "start": problem.start, "target": problem.target,
        "edges": [list(e) for e in problem.edges],
        "prior_var": prior_var,
        "LIPP": lipp, "CIPP": cipp, "Greedy": greedy,
    }
    _save(config, problem, result, lipp, cipp, greedy)
    _print_summary(config.exp_id, prior_var, lipp, cipp, greedy)
    return result


def _save(config, problem, result, lipp, cipp, greedy):
    base = os.path.join(config.out_dir, f"exp{config.exp_id}")
    with open(base + "_results.json", "w") as f:
        json.dump(result, f, indent=2)

    def arr(res, key):
        return np.array(res[key]) if res.get(key) else np.array([])

    np.savez_compressed(
        base + "_data.npz",
        V=problem.V, T=problem.T,
        lipp_path=arr(lipp, "path"),     lipp_samples=arr(lipp, "samples"),
        cipp_path=arr(cipp, "path"),     cipp_samples=arr(cipp, "samples"),
        greedy_path=arr(greedy, "path"), greedy_samples=arr(greedy, "samples"),
    )


def _print_summary(exp_id, prior_var, lipp, cipp, greedy):
    def fmt(res, key):
        return f"{res[key]:.4f}" if res.get(key) is not None else "N/A"

    print(f"\n{'=' * 60}\n  Experiment {exp_id} Summary\n{'=' * 60}")
    print(f"  Prior variance: {prior_var:.4f}")
    print(f"  {'Method':<10}{'PostVar':>10}{'Energy':>10}{'Travel':>12}{'RMSE':>10}")
    print(f"  {'-' * 52}")
    for res in (lipp, cipp, greedy):
        print(f"  {res['method']:<10}{fmt(res, 'post_var'):>10}"
              f"{fmt(res, 'energy'):>10}{fmt(res, 'travel_cost'):>12}"
              f"{fmt(res, 'rmse'):>10}")
    print(f"{'=' * 60}\n")
