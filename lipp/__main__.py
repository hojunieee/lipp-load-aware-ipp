"""Run a single default experiment:  python -m lipp"""
from .config import ExperimentConfig
from .runner import run_experiment

if __name__ == "__main__":
    config = ExperimentConfig(
        exp_id=0, time_limit=100, seed=1234, density=1.0,
        n_vertices=50, n_test=20,
        R_0=5.0, unit_mass=1.0,        # robot physics (shared)
        B=50.0, S_max=3,                # LIPP
        dist_lim=2.0, S=3,             # C-IPP / Greedy
        out_dir="data/unified/exp0",
    )
    run_experiment(config)
