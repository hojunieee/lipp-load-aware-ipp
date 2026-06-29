"""LIPP: the load-aware MIQP planner (paper, Sec. IV).

Jointly optimises routing (chi), visitation order (o), and per-vertex sample
count (z / l) under a load-dependent energy budget. The GP posterior variance is
reformulated via the linear estimator A (LLSE), and the bilinear R_u * chi_uv
energy term is linearised exactly with McCormick auxiliaries Tuv.
"""
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from ..config import SIGMA2, A_MAX_FALLBACK
from ..metrics import extract_path_from_edges
from .common import add_flow_constraints, assemble_result, relative_gap


def run_lipp(problem, config):
    V, T = problem.V, problem.T
    edges, edge_cost = problem.edges, problem.edge_cost
    K_VV, K_TV, K_TT = problem.K_VV, problem.K_TV, problem.K_TT
    n, n_t = problem.n_vertices, problem.n_test
    start, target = problem.start, problem.target

    R_0, lam = config.R_0, config.unit_mass
    B, S_max = config.B, config.S_max
    M = np.eye(n_t)

    L_max = n * S_max * lam            # safe upper bound on cumulative load
    R_max = R_0 + L_max

    # Data-driven big-M for the estimator coefficients (tighter than a guess).
    try:
        A_star = np.linalg.solve(K_VV + SIGMA2 * np.eye(n), K_TV.T).T
        A_max = max(1e-6, float(np.max(np.abs(A_star))))
    except Exception:
        A_max = A_MAX_FALLBACK

    mdl = gp.Model("LIPP")
    mdl.Params.LogToConsole = 0
    mdl.Params.Threads = 6
    mdl.Params.TimeLimit = config.time_limit
    mdl.Params.MIPGap = 0.02
    mdl.Params.Presolve = 2
    mdl.Params.MIPFocus = 2

    # ----- Decision variables -----
    chi = add_flow_constraints(mdl, edges, n, start, target, name="chi")
    y = mdl.addVars(n, vtype=GRB.BINARY, name="y")                       # vertex visit
    z = mdl.addVars(n, range(1, S_max + 1), vtype=GRB.BINARY, name="z")  # sample level
    Aac = mdl.addVars(n_t, n, range(1, S_max + 1),
                      lb=-A_max, ub=A_max, name="Aac")   # estimator per sample level
    A = mdl.addVars(n_t, n, lb=-A_max, ub=A_max, name="A")               # estimator
    R = mdl.addVars(n, lb=R_0, ub=R_max, name="R")                       # robot mass
    L = mdl.addVars(n, lb=0.0, ub=L_max, name="L")                       # carried load
    l_var = mdl.addVars(n, vtype=GRB.INTEGER, lb=0, ub=S_max, name="l")  # sample count
    Tuv = mdl.addVars(edges, lb=0.0, name="Tuv")                         # McCormick aux

    # ----- Estimator <-> sampling links (A aggregation + big-M activation) -----
    for t in range(n_t):
        for v in range(n):
            mdl.addConstr(A[t, v] == gp.quicksum(Aac[t, v, c]
                                                 for c in range(1, S_max + 1)))
            if v not in (start, target):                 # A active only if v visited
                mdl.addConstr(A[t, v] <= A_max * y[v])
                mdl.addConstr(A[t, v] >= -A_max * y[v])
            for c in range(1, S_max + 1):                # A_{t,v,c} = 0 unless z = 1
                mdl.addConstr(Aac[t, v, c] <= A_max * z[v, c])
                mdl.addConstr(Aac[t, v, c] >= -A_max * z[v, c])

    # ----- Sampling activation: one level iff visited; integer count l_v -----
    for v in range(n):
        mdl.addConstr(gp.quicksum(z[v, c] for c in range(1, S_max + 1)) == y[v])
        mdl.addConstr(l_var[v] == gp.quicksum(c * z[v, c]
                                              for c in range(1, S_max + 1)))

    # ----- Vertex activation from incoming flow -----
    for v in range(n):
        if v in (start, target):
            mdl.addConstr(y[v] == 1)
        else:
            mdl.addConstr(y[v] == gp.quicksum(chi[i, j] for (i, j) in edges if j == v))

    # ----- MTZ subtour elimination / visitation order -----
    o = mdl.addVars(n, vtype=GRB.INTEGER, lb=0, ub=n - 1, name="o")
    mdl.addConstr(o[start] == 0)
    for u, v in edges:
        mdl.addConstr(o[v] >= o[u] + 1 - n * (1 - chi[u, v]))

    # ----- Load propagation and robot mass (R_v = R_0 + L_v) -----
    mdl.addConstr(L[start] == lam * gp.quicksum(c * z[start, c]
                                                for c in range(1, S_max + 1)))
    for u, v in edges:
        mdl.addConstr(L[v] >= L[u]
                      + lam * gp.quicksum(c * z[u, c] for c in range(1, S_max + 1))
                      - L_max * (1 - chi[u, v]))
    for v in range(n):
        mdl.addConstr(R[v] == R_0 + L[v])

    # ----- Exact McCormick linearisation of R_u * chi_uv  ->  Tuv -----
    for u, v in edges:
        mdl.addConstr(Tuv[u, v] <= R[u])
        mdl.addConstr(Tuv[u, v] <= R_max * chi[u, v])
        mdl.addConstr(Tuv[u, v] >= R[u] - R_max * (1 - chi[u, v]))

    # ----- Energy budget -----
    mdl.addConstr(gp.quicksum(edge_cost[u, v] * Tuv[u, v] for (u, v) in edges) <= B)

    # Optional path-length budget (Sec. V-A): cap execution time directly by
    # supplying geometric distances euclid[(u, v)] = ||V_u - V_v|| and adding
    #   mdl.addConstr(gp.quicksum(euclid[u, v] * chi[u, v] for (u, v) in edges) <= b)

    # ----- Objective: trace(M (A(k_VV+N)A^T - 2 k_TV A^T + k_TT)) (MIQP form) -----
    # The k_VV term uses Aac since A_{t,v} = sum_c A_{t,v,c}; the noise term
    # sigma^2/l_v * A^2 collapses to sigma^2/c * A_{t,v,c}^2 because A_{t,v,c}
    # is nonzero only at the active sample level c.
    obj = gp.QuadExpr()
    for t in range(n_t):
        w = float(M[t, t])
        for v1 in range(n):
            for v2 in range(n):
                kvv = float(K_VV[v1, v2])
                if abs(kvv) < 1e-12:
                    continue
                for c1 in range(1, S_max + 1):
                    for c2 in range(1, S_max + 1):
                        obj.add(w * kvv * Aac[t, v1, c1] * Aac[t, v2, c2])
        for v in range(n):
            for c in range(1, S_max + 1):
                obj.add(w * (SIGMA2 / c) * Aac[t, v, c] * Aac[t, v, c])
        for v in range(n):
            ktv = float(K_TV[t, v])
            if abs(ktv) > 1e-12:
                for c in range(1, S_max + 1):
                    obj.add(-2.0 * w * ktv * Aac[t, v, c])
        obj.add(w * float(K_TT[t, t]))
    mdl.setObjective(obj, GRB.MINIMIZE)

    clock0 = time.time()
    mdl.optimize()
    solve_time = time.time() - clock0

    if mdl.SolCount == 0:
        return assemble_result("LIPP", int(mdl.status), None, None, None,
                               problem=problem, M=M, energy=None,
                               solve_time=solve_time)

    sel_edges = [(i, j) for (i, j) in edges if chi[i, j].X > 0.5]
    path = extract_path_from_edges(sel_edges, start, target, n)
    samples = np.array([int(round(l_var[v].X)) for v in range(n)])
    A_est = np.array([[sum(Aac[t, v, c].X for c in range(1, S_max + 1))
                       for v in range(n)] for t in range(n_t)])
    energy = float(sum(edge_cost[u, v] * Tuv[u, v].X for (u, v) in edges))  # from model

    return assemble_result("LIPP", int(mdl.status), path, samples, A_est,
                           problem=problem, M=M, energy=energy,
                           solve_time=solve_time,
                           final_gap_pct=relative_gap(mdl),
                           selected_edges=sel_edges)
