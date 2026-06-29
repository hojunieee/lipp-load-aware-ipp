"""LIPP: Load-aware Informative Path Planning -- experiment package.

Public API:
    ExperimentConfig, Problem      -- configuration and instance data
    build_problem, run_experiment  -- generate an instance and run all planners
    run_lipp, run_cipp, run_greedy -- the individual planners
"""
from .config import ExperimentConfig, Problem
from .runner import build_problem, run_experiment
from .solvers import run_lipp, run_cipp, run_greedy

__all__ = ["ExperimentConfig", "Problem", "build_problem", "run_experiment",
           "run_lipp", "run_cipp", "run_greedy"]
