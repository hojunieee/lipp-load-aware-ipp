#!/usr/bin/env python3
"""Runtime vs. graph size: LIPP (near-zero mass) vs. C-IPP.

Fixes the unit sample mass at a near-zero value (default lambda = 0.001) so the
load/energy coupling is negligible, then sweeps the number of sampling vertices.
What remains is the *structural* cost difference between the two formulations:
LIPP still instantiates the sampling-indexed variables (z, A_{t,v,c}, L, R, T) and
must branch over the sample levels, giving the O(m n^2 S_max^2) vs O(m n^2) size
gap, while C-IPP has no sampling decision at all. This mirrors Fig. 8 of the paper
but removes the lambda confound, so the growing gap is attributable to model size
and sampling-selection branching rather than to the coupling.

Run from the repository root (the folder that contains `lipp/`):

    python experiments/runtime_size_sweep.py                     # 5..30, paper-ish
    python experiments/runtime_size_sweep.py --n-graphs 30 --n-max 20   # quicker

Outputs (in --out-dir):
    raw_results.json      every per-(size, graph, method) record
    summary.csv           per-(method, size) aggregate solve-time statistics
    runtime_vs_size.pdf / .png
"""
import os
import sys
import csv
import json
import argparse
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lipp import ExperimentConfig, build_problem, run_lipp, run_cipp  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Graph-size sweep
    p.add_argument("--n-min", type=int, default=5)
    p.add_argument("--n-max", type=int, default=30)
    p.add_argument("--n-step", type=int, default=5)
    p.add_argument("--n-graphs", type=int, default=10, help="graphs per size")
    p.add_argument("--n-test", type=int, default=5)
    p.add_argument("--density", type=float, default=0.15,
                   help="edge density (paper's runtime fig uses ~0.15)")
    # Fixed problem settings
    p.add_argument("--unit-mass", type=float, default=0.001, help="lambda (near 0)")
    p.add_argument("--r0", type=float, default=1.0)
    p.add_argument("--s-max", type=int, default=3)
    p.add_argument("--s", type=int, default=2)
    p.add_argument("--dist-lim", type=float, default=2.0)
    p.add_argument("--budget", type=float, default=2.0, help="LIPP energy budget B")
    p.add_argument("--time-limit", type=int, default=100)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="data/runtime_size_sweep")
    return p.parse_args()


def run_sweep(args, sizes):
    """For each graph size, build n_graphs instances and run both planners."""
    records = []
    for si, n in enumerate(sizes):
        for g in range(args.n_graphs):
            seed = args.seed_base + si * args.n_graphs + g
            base = ExperimentConfig(
                seed=seed, n_vertices=n, n_test=args.n_test, density=args.density,
                R_0=args.r0, unit_mass=args.unit_mass, S_max=args.s_max, S=args.s,
                dist_lim=args.dist_lim, B=args.budget, time_limit=args.time_limit,
            )
            problem = build_problem(base)
            cipp = run_cipp(problem, base)
            lipp = run_lipp(problem, base)
            records.append(_record(n, seed, "CIPP", cipp))
            records.append(_record(n, seed, "LIPP", lipp))
        # progress per size
        med = lambda meth: np.median([r["solve_time"] for r in records
                                      if r["n_vertices"] == n and r["method"] == meth
                                      and r["solve_time"] is not None])
        print(f"[size {n:>2} | {si + 1}/{len(sizes)}] "
              f"median  C-IPP {med('CIPP'):6.2f}s   LIPP {med('LIPP'):6.2f}s",
              flush=True)
    return records


def _record(n, seed, method, res):
    return {
        "n_vertices": n, "seed": seed, "method": method,
        "solve_time": res.get("solve_time"),
        "status": res.get("status"),
        "final_gap_pct": res.get("final_gap_pct"),
        "solved": res.get("path") is not None,
    }


def summarize(records, sizes):
    rows = []
    for method in ("CIPP", "LIPP"):
        for n in sizes:
            subset = [r for r in records
                      if r["method"] == method and r["n_vertices"] == n]
            times = np.array([r["solve_time"] for r in subset
                              if r["solve_time"] is not None])
            if times.size == 0:
                continue
            rows.append({
                "method": method, "n_vertices": n, "n": int(times.size),
                "mean": float(times.mean()), "median": float(np.median(times)),
                "p25": float(np.percentile(times, 25)),
                "p75": float(np.percentile(times, 75)),
                "std": float(times.std()),
                "n_timelimit": sum(1 for r in subset if r["status"] == 9),
                "n_unsolved": sum(1 for r in subset if not r["solved"]),
            })
    return rows


def make_plot(summary, sizes, s_max, out_base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series(method):
        s = sorted((r for r in summary if r["method"] == method),
                   key=lambda r: r["n_vertices"])
        x = [r["n_vertices"] for r in s]
        clamp = lambda v: max(v, 1e-3)              # keep positive for log axis
        return (x, [clamp(r["median"]) for r in s],
                [clamp(r["p25"]) for r in s], [clamp(r["p75"]) for r in s])

    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    for method, color in (("CIPP", "C3"), ("LIPP", "C0")):
        x, med, p25, p75 = series(method)
        if not x:
            continue
        ax.fill_between(x, p25, p75, alpha=0.18, color=color)
        ax.plot(x, med, "o-", color=color, lw=2, label=method)

    # Reference: S_max^2 x C-IPP (the nominal variable-count growth factor).
    cipp = sorted((r for r in summary if r["method"] == "CIPP"),
                  key=lambda r: r["n_vertices"])
    if cipp:
        xs = [r["n_vertices"] for r in cipp]
        ref = [max(s_max ** 2 * r["median"], 1e-3) for r in cipp]
        ax.plot(xs, ref, "k--", lw=1.3, alpha=0.7,
                label=f"${s_max}^2$ $\\times$ C-IPP")

    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xlabel("Number of sampling vertices  $n$")
    ax.set_ylabel("Solve time to gap (s)")
    ax.set_title("Runtime vs. graph size ($\\lambda \\to 0$)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=200)
    print(f"Saved plot -> {out_base}.pdf / .png")


def main():
    args = parse_args()
    sizes = list(range(args.n_min, args.n_max + 1, args.n_step))
    os.makedirs(args.out_dir, exist_ok=True)

    records = run_sweep(args, sizes)
    summary = summarize(records, sizes)

    with open(os.path.join(args.out_dir, "raw_results.json"), "w") as f:
        json.dump(records, f, indent=2)
    with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print(f"\n=== Solve-time vs. size (lambda={args.unit_mass}, seconds) ===")
    print(f"{'method':<8}{'n':>4}{'#graphs':>8}{'median':>9}{'mean':>9}"
          f"{'p25':>8}{'p75':>8}{'#TL':>5}")
    for r in summary:
        print(f"{r['method']:<8}{r['n_vertices']:>4}{r['n']:>8}{r['median']:>9.2f}"
              f"{r['mean']:>9.2f}{r['p25']:>8.2f}{r['p75']:>8.2f}{r['n_timelimit']:>5}")

    make_plot(summary, sizes, args.s_max,
              os.path.join(args.out_dir, "runtime_vs_size"))


if __name__ == "__main__":
    main()
