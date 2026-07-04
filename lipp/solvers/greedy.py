"""Greedy IPP baseline.

At each step add the unvisited vertex with the best posterior-variance reduction
per unit travel distance, as long as the detour still leaves enough budget to
reach the goal. Fixed S samples are taken at every chosen vertex. Paths never
route back through an already-visited vertex (including the start).
"""
import numpy as np
import networkx as nx

from ..config import SIGMA2
from ..metrics import compute_energy_along_path, compute_posterior_variance
from .common import assemble_result


def _shortest_avoiding(g, src, dst, forbidden, edge_cost):
    """Shortest src->dst path that avoids `forbidden` nodes (src and dst exempt).

    Returns (route, cost), or (None, None) if no such path exists.
    """
    if src == dst:
        return [src], 0.0
    keep = [node for node in g.nodes
            if node not in forbidden or node in (src, dst)]
    try:
        route = nx.shortest_path(g.subgraph(keep), src, dst, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None, None
    cost = sum(edge_cost[(u, w)] for u, w in zip(route, route[1:]))
    return route, cost


def run_greedy(problem, config):
    edges, edge_cost = problem.edges, problem.edge_cost
    K_VV, K_TV, K_TT = problem.K_VV, problem.K_TV, problem.K_TT
    n = problem.n_vertices
    start, goal = problem.start, n - 1          # name-agnostic (goal/target)

    R_0, lam, S = config.R_0, config.unit_mass, config.S
    M = np.eye(problem.n_test)

    g = nx.DiGraph()
    for (i, j) in edges:
        g.add_edge(i, j, weight=edge_cost[(i, j)])
    try:
        lengths = dict(nx.all_pairs_dijkstra_path_length(g, weight="weight"))
        paths = dict(nx.all_pairs_dijkstra_path(g, weight="weight"))
    except Exception:
        lengths, paths = {}, {}

    samples = np.zeros(n, dtype=int)
    samples[start] = S
    samples[goal] = S          # goal is sampled on arrival (like the start / C-IPP)
    visited = {start}
    path_nodes = [start]
    selected_edges = []
    remaining = float(config.dist_lim)
    current = start
    post_var_cur, _ = compute_posterior_variance(K_VV, K_TV, K_TT, samples, M)

    while True:
        best = None  # (score, vertex, route, post_var)
        for v in range(n):
            if v in visited or v == goal:       # never revisit; goal is terminal only
                continue
            # Route to v that avoids visited nodes AND the goal (the goal is
            # reachable only as the final leg, never as a pass-through).
            route, d_to_v = _shortest_avoiding(g, current, v,
                                               (visited | {goal}) - {current},
                                               edge_cost)
            if route is None:
                continue
            try:
                d_v_to_t = lengths[v][goal]
            except KeyError:
                continue
            if d_to_v + d_v_to_t > remaining + 1e-9:        # must still reach goal
                continue

            trial = samples.copy()
            trial[v] = S
            pv, _ = compute_posterior_variance(K_VV, K_TV, K_TT, trial, M)
            if pv is None:
                continue
            benefit = post_var_cur - pv
            if benefit <= 0:
                continue
            score = benefit / (d_to_v + 1e-12)
            if best is None or score > best[0]:
                best = (score, v, route, pv)

        if best is None:
            break

        _, best_v, route, best_pv = best
        for u, w in zip(route, route[1:]):
            selected_edges.append((u, w))
            path_nodes.append(w)
            visited.add(w)
            remaining -= edge_cost[(u, w)]
            current = w
        samples[best_v] = S
        post_var_cur = best_pv

    # Return to goal via a path that avoids visited nodes (no back-tracking
    # through the start or any sampled vertex).
    if current != goal:
        route, cost = _shortest_avoiding(g, current, goal, visited - {current},
                                         edge_cost)
        if route is not None and cost <= remaining + 1e-9:
            for u, w in zip(route, route[1:]):
                selected_edges.append((u, w))
                path_nodes.append(w)
                visited.add(w)
                remaining -= edge_cost[(u, w)]
            current = goal

    # GP posterior-mean estimator under the final noise structure.
    noise = np.where(samples > 0, SIGMA2 / np.maximum(samples, 1), 1e6)
    try:
        A_est = np.linalg.solve(K_VV + np.diag(noise), K_TV.T).T
    except Exception:
        A_est = None

    energy = compute_energy_along_path(path_nodes, samples, edge_cost, R_0, lam)
    return assemble_result("Greedy", "GREEDY", path_nodes, samples, A_est,
                           problem=problem, M=M, energy=energy, solve_time=0.0,
                           selected_edges=selected_edges)