"""C-IPP (Dutta-style) MIQP planner -- the load-unaware baseline.

Routes under a plain distance budget and applies post-hoc uniform sampling
(S samples at every visited vertex). Subtours are removed lazily via a
callback. Recovered as the lambda -> 0 limit of LIPP.
"""
import time
import numpy as np
import networkx as nx
import gurobipy as gp
from gurobipy import GRB

from ..config import SIGMA2
from ..metrics import extract_path_from_edges, compute_energy_along_path
from .common import add_flow_constraints, assemble_result, relative_gap


def run_cipp(problem, config):
    edges, edge_cost = problem.edges, problem.edge_cost
    K_VV, K_TV, K_TT = problem.K_VV, problem.K_TV, problem.K_TT
    n, n_t = problem.n_vertices, problem.n_test
    start, goal = problem.start, problem.goal

    R_0, lam, S = config.R_0, config.unit_mass, config.S
    M = np.eye(n_t)
    KVV_reg = K_VV + SIGMA2 * np.eye(n)          # fixed noise N = sigma^2 I

    mdl = gp.Model("CIPP")
    mdl.Params.LogToConsole = 0
    mdl.Params.TimeLimit = config.time_limit
    mdl.Params.MIPGap = 0.02
    mdl.Params.Threads = 0
    mdl.Params.Presolve = 2
    mdl.Params.LazyConstraints = 1

    x = add_flow_constraints(mdl, edges, n, start, goal, name="x")
    y = mdl.addVars(n, vtype=GRB.BINARY, name="y")
    K = mdl.addVars(n_t, n, lb=-GRB.INFINITY, name="K")     # estimator (no noise model)

    # Vertex activation from incoming flow.
    for v in range(n):
        if v in (start, goal):
            mdl.addConstr(y[v] == 1)
        else:
            mdl.addConstr(y[v] == gp.quicksum(x[i, j] for (i, j) in edges if j == v))

    # SOS-1 forces K[t, v] = 0 unless vertex v is visited.
    for t in range(n_t):
        for v in range(n):
            z_aux = mdl.addVar(lb=0, ub=1)
            mdl.addConstr(z_aux + y[v] == 1)
            mdl.addSOS(GRB.SOS_TYPE1, [K[t, v], z_aux])

    # Distance budget.
    mdl.addConstr(gp.quicksum(edge_cost[e] * x[e] for e in edges) <= config.dist_lim)

    # Objective: same trace form as LIPP but with fixed noise (in KVV_reg) and
    # no sampling variables. M is diagonal, so off-diagonal terms vanish.
    obj = gp.QuadExpr()
    for t in range(n_t):
        for s in range(n_t):
            w = float(M[t, s])
            if abs(w) < 1e-12:
                continue
            for v1 in range(n):
                for v2 in range(n):
                    coeff = float(KVV_reg[v1, v2])
                    if abs(coeff) > 1e-12:
                        obj.add(w * coeff * K[s, v1] * K[t, v2])
            for v in range(n):
                ktv = float(K_TV[s, v])
                if abs(ktv) > 1e-12:
                    obj.add(-2.0 * w * ktv * K[t, v])
            obj.add(w * float(K_TT[s, t]))
    mdl.setObjective(obj, GRB.MINIMIZE)

    def subtour_callback(model, where):
        """Lazily forbid strongly-connected components that exclude start/goal."""
        if where != GRB.Callback.MIPSOL:
            return
        try:
            vals = model.cbGetSolution(x)
            g = nx.DiGraph()
            g.add_nodes_from(range(n))
            g.add_edges_from(e for e in edges if vals[e] > 0.5)
            for comp in nx.strongly_connected_components(g):
                if len(comp) >= 2 and start not in comp and goal not in comp:
                    model.cbLazy(gp.quicksum(x[i, j] for (i, j) in edges
                                             if i in comp and j in comp)
                                 <= len(comp) - 1)
        except Exception:
            pass

    clock0 = time.time()
    mdl.optimize(subtour_callback)
    solve_time = time.time() - clock0

    if mdl.SolCount == 0:
        return assemble_result("CIPP", int(mdl.status), None, None, None,
                               problem=problem, M=M, energy=None,
                               solve_time=solve_time)

    sel_edges = [e for e in edges if x[e].X > 0.5]
    path = extract_path_from_edges(sel_edges, start, goal, n)

    # Post-hoc uniform sampling: S samples at every visited vertex.
    samples = np.array([S if y[v].X > 0.5 else 0 for v in range(n)], dtype=int)
    A_est = np.array([[K[t, v].X for v in range(n)] for t in range(n_t)])
    energy = compute_energy_along_path(path, samples, edge_cost, R_0, lam)

    return assemble_result("CIPP", int(mdl.status), path, samples, A_est,
                           problem=problem, M=M, energy=energy,
                           solve_time=solve_time,
                           final_gap_pct=relative_gap(mdl),
                           selected_edges=sel_edges)
