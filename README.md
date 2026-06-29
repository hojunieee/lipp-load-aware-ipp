# LIPP experiment package

Experiment code for **LIPP: Load-Aware Informative Path Planning with Physical
Sampling**. Generates a random instance, runs LIPP alongside the C-IPP and
Greedy baselines on the *same* graph, and saves consolidated results.

## Layout

```
lipp/
├── config.py        # ExperimentConfig, Problem, physical constants
├── geometry.py      # points, elevation, Th field, RBF kernel, edge/graph builders
├── metrics.py       # energy, travel cost, posterior variance, RMSE, path extraction
├── solvers/
│   ├── common.py    # shared Gurobi helpers (flow constraints, result builder, gap)
│   ├── lipp.py      # LIPP load-aware MIQP (main contribution)
│   ├── cipp.py      # C-IPP baseline MIQP + lazy subtour elimination
│   └── greedy.py    # Greedy heuristic baseline
├── runner.py        # build_problem + run_experiment (+ saving / summary)
└── __main__.py      # `python -m lipp` runs one default experiment
```

Dependencies: `numpy`, `networkx`, `gurobipy` (with a license large enough for
the chosen `n_vertices` / `S_max`).

## Usage

Run one experiment with the defaults:

```bash
python -m lipp
```

Or from Python, with custom settings:

```python
from lipp import ExperimentConfig, run_experiment

cfg = ExperimentConfig(n_vertices=10, S_max=3, B=5.0, dist_lim=2.0,
                       out_dir="data/unified/exp0")
result = run_experiment(cfg)        # also accepts a plain dict
```

Each run writes `exp{id}_results.json` (full record) and `exp{id}_data.npz`
(compact paths/samples) to `out_dir`, and prints a summary table.

## Key parameters

| Field        | Meaning                                  | Used by          |
|--------------|------------------------------------------|------------------|
| `R_0`        | base robot mass                          | all              |
| `unit_mass`  | mass per unit sample (lambda)            | all              |
| `B`          | energy budget                            | LIPP             |
| `dist_lim`   | distance budget                          | C-IPP, Greedy    |
| `S_max`      | max samples per vertex                   | LIPP             |
| `S`          | fixed samples per visited vertex         | C-IPP, Greedy    |

Setting `unit_mass` (lambda) -> 0 makes LIPP recover C-IPP, as discussed in the
paper.
