"""Shared building blocks for the Gurobi-based planners (LIPP and C-IPP)."""
import gurobipy as gp
from gurobipy import GRB

from ..metrics import (compute_travel_cost, compute_posterior_variance,
                       compute_rmse)


def add_flow_constraints(model, edges, n_vertices, start, target, name):
    """Add binary edge variables + single-path flow conservation.

    Enforces one outgoing edge at `start`, one incoming at `target`, no
    back-flow at either end, and conservation (in == out <= 1) elsewhere.
    Returns the edge-variable dict.
    """
    e = model.addVars(edges, vtype=GRB.BINARY, name=name)
    for v in range(n_vertices):
        inflow = gp.quicksum(e[i, j] for (i, j) in edges if j == v)
        outflow = gp.quicksum(e[i, j] for (i, j) in edges if i == v)
        if v == start:
            model.addConstr(outflow == 1)
            model.addConstr(inflow == 0)
        elif v == target:
            model.addConstr(inflow == 1)
            model.addConstr(outflow == 0)
        else:
            model.addConstr(inflow == outflow)
            model.addConstr(inflow <= 1)
    return e


def relative_gap(model):
    """Relative MIP gap (%) from incumbent and bound, or None if unavailable."""
    try:
        return abs(model.ObjVal - model.ObjBound) / max(abs(model.ObjVal), 1e-12) * 100
    except Exception:
        return None


def assemble_result(method, status, path, samples, A_est, *, problem, M,
                    energy, solve_time, final_gap_pct=None, selected_edges=None):
    """Build the common per-method result dict (metrics + serialisable fields)."""
    has_samples = samples is not None
    return {
        "method": method,
        "status": status,
        "path": path,
        "path_samples": [(int(v), int(samples[v])) for v in path]
                        if path and has_samples else None,
        "samples": samples.tolist() if has_samples else None,
        "A_est": A_est.tolist() if A_est is not None else None,
        "travel_cost": compute_travel_cost(path, problem.edge_cost),
        "energy": energy,
        "post_var": compute_posterior_variance(
            problem.K_VV, problem.K_TV, problem.K_TT, samples, M)[0]
            if has_samples else None,
        "rmse": compute_rmse(A_est, problem.V, problem.T),
        "solve_time": solve_time,
        "final_gap_pct": final_gap_pct,
        "selected_edges": [list(e) for e in selected_edges]
                          if selected_edges is not None else None,
    }
