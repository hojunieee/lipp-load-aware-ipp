#!/usr/bin/env python3
"""Qualitative comparison figure: Greedy vs. C-IPP vs. LIPP on a fixed graph.

Reproduces the paper's heatmap comparison (Fig. 2) using HAND-CODED node and
test coordinates read off the original figure, rather than searching for a seed.
Edit NODES / TESTS below to nudge the layout; everything downstream adapts.

Run from the repository root (the folder that contains `lipp/`):

    python experiments/qualitative_heatmap.py

Output: data/qualitative/heatmap_comparison.pdf / .png
"""
import os
import sys
from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42     # Type 42 (TrueType) -> IEEE-compliant
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt               # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lipp import ExperimentConfig, Problem, run_lipp, run_cipp, run_greedy  # noqa: E402
from lipp.geometry import (th_level, rbf_kernel, elevation,  # noqa: E402
                           build_edge_dicts)

# ---------------------------------------------------------------------------
# Hand-coded layout (read off the reference figure). `a` is start, `i` target;
# the package treats vertex 0 as start and the last vertex as target, so keep
# `a` first and `i` last in this dict.
# ---------------------------------------------------------------------------
NODES = {
    "a": (0.46, 0.54),   # Start
    "b": (0.30, 0.10),
    "c": (0.50, 0.85),
    "d": (0.84, 0.72),
    "e": (0.42, 0.67),
    "f": (0.29, 0.66),
    "g": (0.21, 0.92),
    "h": (0.13, 0.31),
    "i": (0.62, 0.17),   # Target
}
TESTS = [
    (0.22, 0.80),
    (0.35, 0.78),
    (0.32, 0.59),
    (0.58, 0.63),
    (0.78, 0.42),
]

# Budgets / physics. C-IPP and Greedy route under the distance budget; LIPP's
# energy budget is then set to BUDGET_FRACTION of the energy C-IPP actually spent
# (the same scheme as the paper's distance/energy figure).
CONFIG = dict(R_0=1.0, unit_mass=1.0, S_max=3, S=3, dist_lim=1.75, time_limit=100)
BUDGET_FRACTION = 0.5     # B_LIPP = fraction * (energy C-IPP consumes)
OUT_DIR = "data/qualitative"

# Marker sizing (matplotlib scatter area in points^2).
NODE_SIZE = 160           # node circle size; lower = smaller circles
NODE_FONT = 8             # letter inside each node
TEST_SIZE = 70            # red test-location squares


def build_handcoded_problem():
    labels = list(NODES.keys())
    V = np.array([NODES[k] for k in labels], dtype=float)
    T = np.array(TESTS, dtype=float)
    n = len(V)

    # Complete directed graph. Edge travel cost uses the SAME model as the main
    # experiments: elevation-adjusted via build_edge_dicts (d_uv scaled by slope),
    # so "Dis" here is the package's traversal cost, not raw Euclidean length.
    edges = [(i, j) for i in range(n) for j in range(n) if i != j]
    Vz = elevation(V)
    edge_cost = build_edge_dicts(V, Vz, edges)

    problem = Problem(V=V, T=T, edges=edges, edge_cost=edge_cost,
                      K_VV=rbf_kernel(V, V), K_TV=rbf_kernel(T, V),
                      K_TT=rbf_kernel(T, T))
    return problem, labels


def _empty_result():
    return {"path": None, "samples": None, "energy": None,
            "travel_cost": None, "post_var": None}


def solve(problem):
    """Run Greedy + C-IPP under the distance budget, then LIPP under an energy
    budget set to BUDGET_FRACTION of C-IPP's consumed energy.

    Returns (results dict, b_lipp) where b_lipp is the derived LIPP budget.
    """
    base = ExperimentConfig(n_vertices=problem.n_vertices,
                            n_test=problem.n_test, **CONFIG)
    out = {}
    for name, fn in (("Greedy", run_greedy), ("C-IPP", run_cipp)):
        try:
            out[name] = fn(problem, base)
        except Exception as e:                # e.g. Gurobi license too small
            print(f"  [{name}] failed: {e}")
            out[name] = _empty_result()

    # LIPP energy budget = fraction of what C-IPP actually spent.
    b_cipp = out["C-IPP"].get("energy")
    if b_cipp is None:
        print("  [LIPP] skipped: C-IPP energy unavailable")
        out["LIPP"], b_lipp = _empty_result(), None
    else:
        b_lipp = BUDGET_FRACTION * b_cipp
        print(f"  B_CIPP={b_cipp:.2f}  ->  B_LIPP={b_lipp:.2f} "
              f"({BUDGET_FRACTION:.0%} of C-IPP)")
        try:
            out["LIPP"] = run_lipp(problem, replace(base, B=b_lipp))
        except Exception as e:
            print(f"  [LIPP] failed: {e}")
            out["LIPP"] = _empty_result()
    return out, b_lipp


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _field_grid(res=300):
    g = np.linspace(0, 1, res)
    xx, yy = np.meshgrid(g, g)
    z = elevation(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    return z


def plot_panel(ax, problem, result, labels, title, field):
    start, target = 0, problem.n_vertices - 1   # name-agnostic (goal/target)
    V = problem.V

    im = ax.imshow(field, extent=[0, 1, 0, 1], origin="lower",
                   cmap="viridis", aspect="equal", zorder=0)

    # Test locations.
    ax.scatter(problem.T[:, 0], problem.T[:, 1], marker="s", s=TEST_SIZE,
               c="red", edgecolors="black", linewidths=1.0, zorder=3)

    # Path arrows (skip gracefully if a method returned no path).
    path = result.get("path")
    if path:
        for u, v in zip(path, path[1:]):
            ax.add_patch(FancyArrowPatch(
                V[u], V[v], arrowstyle="-|>", mutation_scale=16,
                color="white", lw=2.5, shrinkA=7, shrinkB=7, zorder=4))

    # Nodes.
    for idx, (x, y) in enumerate(V):
        if idx == start:
            fc, tc = "deepskyblue", "white"
        elif idx == target:
            fc, tc = "navy", "white"
        else:
            fc, tc = "white", "black"
        ax.scatter(x, y, s=NODE_SIZE, c=fc, edgecolors="black",
                   linewidths=1.2, zorder=5)
        ax.text(x, y, labels[idx], ha="center", va="center", fontsize=NODE_FONT,
                fontweight="bold", color=tc, zorder=6)

    ax.text(*(V[start] - [0, 0.05]), "Start", color="deepskyblue",
            fontsize=8, fontweight="bold", ha="center", va="top", zorder=6)
    ax.text(*(V[target] - [0, 0.05]), "Target", color="navy",
            fontsize=8, fontweight="bold", ha="center", va="top", zorder=6)

    # Sample-count boxes at visited vertices.
    samples = result.get("samples")
    if samples is not None:
        for idx, (x, y) in enumerate(V):
            if samples[idx] > 0:
                ax.text(x + 0.022, y + 0.045, str(int(samples[idx])), fontsize=8,
                        ha="center", va="center", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  ec="black", lw=1.0))

    def fmt(key):
        v = result.get(key)
        return f"{v:.2f}" if v is not None else "--"
    ax.set_title(f"{title}\nEnergy={fmt('energy')} | Dis={fmt('travel_cost')} | "
                 f"PostVar={fmt('post_var')}", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    return im


def make_figure(problem, results, labels, out_base, b_lipp):
    field = _field_grid()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    im = None
    for ax, name in zip(axes, ("Greedy", "C-IPP", "LIPP")):
        im = plot_panel(ax, problem, results[name], labels,
                        "Ours (LIPP)" if name == "LIPP" else name, field)

    # Shared colorbar.
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Elevation")

    # Budget box, top-right.
    b_txt = f"{b_lipp:.2f}" if b_lipp is not None else "--"
    fig.text(0.995, 0.99,
             f"Distance Limit: {CONFIG['dist_lim']}\n"
             f"LIPP Energy: {b_txt} ({BUDGET_FRACTION:.0%} of C-IPP)",
             ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black"))

    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=200, bbox_inches="tight")
    print(f"Saved -> {out_base}.pdf / .png")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    problem, labels = build_handcoded_problem()
    print(f"Graph: {problem.n_vertices} nodes, {problem.n_test} tests, "
          f"{len(problem.edges)} edges")
    results, b_lipp = solve(problem)
    for name in ("Greedy", "C-IPP", "LIPP"):
        r = results[name]
        e = f"{r['energy']:.2f}" if r.get("energy") is not None else "--"
        print(f"  {name:<7} energy={e}  path={r.get('path')}")
    make_figure(problem, results, labels,
                os.path.join(OUT_DIR, "heatmap_comparison"), b_lipp)


if __name__ == "__main__":
    main()