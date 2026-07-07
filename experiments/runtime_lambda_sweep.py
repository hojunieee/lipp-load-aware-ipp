#!/usr/bin/env python3
"""Runtime vs. unit sample mass (lambda) sweep.

Tests the hypothesis that LIPP's solve time decreases toward C-IPP as lambda -> 0
(the load/energy coupling relaxes tightly in the limit), while staying strictly
above it (LIPP still carries the sampling variables and big-M estimator links).

Design:
  * Graph size is held FIXED; only lambda varies, so any change in solve time is
    attributable to the load-dependent coupling, not to model size.
  * For each random graph (seed) the instance is built ONCE and shared across all
    lambda values, giving a paired comparison.
  * C-IPP is run once per graph: its MIQP does not depend on lambda (lambda only
    enters the post-hoc energy accounting), so it is the lambda-independent floor.

Run from the repository root (the folder that contains `lipp/`):

    python experiments/runtime_lambda_sweep.py                  # paper settings
    python experiments/runtime_lambda_sweep.py --n-graphs 30    # quicker look

Outputs (in --out-dir):
    raw_results.json      every per-(graph, method, lambda) record
    summary.csv           per-lambda aggregate solve-time statistics
    runtime_vs_lambda.pdf / .png
"""
import os
import sys
import csv
import json
import argparse
from dataclasses import replace

import numpy as np

# Make the `lipp` package importable regardless of the current directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lipp import ExperimentConfig, build_problem, run_lipp, run_cipp  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-graphs", type=int, default=500)
    p.add_argument("--n-vertices", type=int, default=10)
    p.add_argument("--n-test", type=int, default=5)
    p.add_argument("--r0", type=float, default=1.0)
    p.add_argument("--s-max", type=int, default=3)
    p.add_argument("--s", type=int, default=2)
    p.add_argument("--dist-lim", type=float, default=2.0)
    p.add_argument("--budget", type=float, default=2.0, help="LIPP energy budget B")
    # Log-spaced lambda sweep (default 10 points over [1e-3, 1]).
    p.add_argument("--lambda-min", type=float, default=0.001)
    p.add_argument("--lambda-max", type=float, default=2.0)
    p.add_argument("--n-lambda", type=int, default=10)
    p.add_argument("--lambdas", type=str, default=None,
                   help="comma-separated override; if set, ignores lambda-min/max/n")
    p.add_argument("--time-limit", type=int, default=30)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="data/runtime_sweep")
    return p.parse_args()


def run_sweep(args, lambdas):
    """Build each graph once; run C-IPP once and LIPP for every lambda."""
    records = []
    for g in range(args.n_graphs):
        seed = args.seed_base + g
        base = ExperimentConfig(
            seed=seed, n_vertices=args.n_vertices, n_test=args.n_test,
            R_0=args.r0, S_max=args.s_max, S=args.s,
            dist_lim=args.dist_lim, B=args.budget, time_limit=args.time_limit,
        )
        problem = build_problem(base)

        print(f"[graph {g + 1}/{args.n_graphs}] solving ...", flush=True)

        cipp = run_cipp(problem, base)
        records.append(_record(g, seed, "CIPP", None, cipp))
        print(f"    CIPP            -> {cipp['solve_time']:6.2f}s", flush=True)

        for lam in lambdas:
            lipp = run_lipp(problem, replace(base, unit_mass=lam))
            records.append(_record(g, seed, "LIPP", lam, lipp))
            print(f"    LIPP  lambda={lam:<6} -> {lipp['solve_time']:6.2f}s",
                  flush=True)
    return records


def _record(graph, seed, method, lam, res):
    return {
        "graph": graph, "seed": seed, "method": method, "lambda": lam,
        "solve_time": res.get("solve_time"),
        "status": res.get("status"),
        "final_gap_pct": res.get("final_gap_pct"),
        "post_var": res.get("post_var"),
        "energy": res.get("energy"),
        "solved": res.get("path") is not None,
    }


def summarize(records, lambdas):
    """Aggregate solve-time stats per method/lambda over the graphs."""
    rows = []

    def stats(label, lam, subset):
        times = np.array([r["solve_time"] for r in subset
                          if r["solve_time"] is not None])
        if times.size == 0:
            return None
        # Gurobi status 2 = OPTIMAL/within-gap, 9 = TIME_LIMIT.
        n_tl = sum(1 for r in subset if r["status"] == 9)
        return {
            "method": label, "lambda": lam, "n": int(times.size),
            "mean": float(times.mean()), "median": float(np.median(times)),
            "p25": float(np.percentile(times, 25)),
            "p75": float(np.percentile(times, 75)),
            "std": float(times.std()), "n_timelimit": n_tl,
            "n_unsolved": sum(1 for r in subset if not r["solved"]),
        }

    cipp_rows = [r for r in records if r["method"] == "CIPP"]
    s = stats("CIPP", None, cipp_rows)
    if s:
        rows.append(s)
    for lam in lambdas:
        s = stats("LIPP", lam,
                  [r for r in records if r["method"] == "LIPP" and r["lambda"] == lam])
        if s:
            rows.append(s)
    return rows


def make_plot(summary, lambdas, s_max, out_base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Avoid Type 3 fonts (required by IEEE / many conferences)
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    lipp = [s for s in summary if s["method"] == "LIPP"]
    cipp = next((s for s in summary if s["method"] == "CIPP"), None)
    if not lipp:
        print("No LIPP results to plot.")
        return

    xs = [s["lambda"] for s in lipp]
    med = [s["median"] for s in lipp]
    p25 = [s["p25"] for s in lipp]
    p75 = [s["p75"] for s in lipp]

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.fill_between(xs, p25, p75, alpha=0.2, color="#f0b753", label="LIPP IQR")
    ax.plot(xs, med, "o-", color="#f0b753", lw=2, label="LIPP (median)")

    if cipp is not None:
        ax.axhline(cipp["median"], ls="--", color="#5dae6b", lw=1.6,
                   label="C-IPP (median, $\\lambda$-independent)")
        ax.fill_between(lambdas, cipp["p25"], cipp["p75"], alpha=0.12, color="#5dae6b")
        # Reference: S_max^2 x C-IPP (nominal variable-count growth factor).
        ax.axhline(s_max ** 2 * cipp["median"], ls="--", color="gray", lw=1.3,
                   alpha=0.8, label=f"${s_max}^2$ $\\times$ C-IPP")

    ax.set_xscale("log")
    ax.set_xlabel("Unit sample mass  $\\lambda$")
    ax.set_ylabel("Solve time to gap (s)")
    ax.set_title("Runtime vs. sample mass (graph size = 20)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=200)
    print(f"Saved plot -> {out_base}.pdf / .png")


def main():
    args = parse_args()

    if args.lambdas:                                   # explicit override
        lambdas = [float(x) for x in args.lambdas.split(",")]
    else:
        lambdas = list(np.logspace(np.log10(args.lambda_min),
                                   np.log10(args.lambda_max), args.n_lambda))
        
    os.makedirs(args.out_dir, exist_ok=True)

    records = run_sweep(args, lambdas)
    summary = summarize(records, lambdas)

    with open(os.path.join(args.out_dir, "raw_results.json"), "w") as f:
        json.dump(records, f, indent=2)
    with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print("\n=== Solve-time summary (seconds) ===")
    print(f"{'method':<8}{'lambda':>8}{'n':>5}{'median':>9}{'mean':>9}"
          f"{'p25':>8}{'p75':>8}{'#TL':>5}")
    for s in summary:
        lam = "-" if s["lambda"] is None else f"{s['lambda']:g}"
        print(f"{s['method']:<8}{lam:>8}{s['n']:>5}{s['median']:>9.2f}"
              f"{s['mean']:>9.2f}{s['p25']:>8.2f}{s['p75']:>8.2f}{s['n_timelimit']:>5}")

    make_plot(summary, lambdas, args.s_max,
              os.path.join(args.out_dir, "runtime_vs_lambda"))


if __name__ == "__main__":
    main()
