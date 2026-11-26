"""Component-aware network location and representation (numerical API).

No GIS dependency. Distances use a symmetric positive-weight CSR graph.
Single-site objectives are exact up to floating-point arithmetic; multi-site
BUILD/SWAP cost curves are heuristic, and selection policies act on those curves.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from numbers import Integral

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _indices(values, n, name):
    a = np.asarray(values)
    if a.ndim != 1 or a.size == 0 or a.dtype.kind not in 'iu':
        raise ValueError(f"{name} must be a nonempty integer vector")
    if np.any(a >= n) or np.any(a < 0):
        raise ValueError(f"{name} outside graph")
    return a.astype(np.int64, copy=False)


def _vector(values, size, name, default):
    a = np.full(size, default, dtype=float) if values is None else np.asarray(
        values, dtype=float
    )
    if a.shape != (size,) or not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError(f"{name} must be finite, nonnegative and aligned with demand")
    return a


def _positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _finite(value, name):
    if not np.isfinite(value):
        raise ValueError(f"{name} overflowed or is not finite")
    return float(value)


def _reject_unused(options):
    """Refuse an explicitly supplied option that the selected mode never uses.

    ``options`` maps a name to (value, applies, rule). ``None`` means omitted.
    """
    for name, (value, applies, rule) in options.items():
        if value is not None and not applies:
            raise ValueError(f'{name} only applies to {rule}; it would have no effect here')


@dataclass(frozen=True)
class CurvePoint:
    """Evaluated allocation: network sum for median, radius for minimax.

    Minimax radius includes connectors when requested."""

    k: int
    nodes: tuple[int, ...]
    cost: float


@dataclass(frozen=True)
class AdaptiveResult:
    centers: np.ndarray
    assignment: np.ndarray  # index into centers; -1 means excluded
    network_distance: np.ndarray  # inf for excluded rows
    access_distance: np.ndarray
    site_weight: np.ndarray
    site_count: np.ndarray
    site_mean_access: np.ndarray  # nan for zero-mass sites
    site_max_access: np.ndarray  # nan for empty sites
    component: np.ndarray  # original graph component per demand row
    objective: float
    network_cost: float
    total_weight: float
    modeled_weight: float
    coverage: float
    mean_access: float
    max_access: float
    curve: tuple[CurvePoint, ...]
    selection: str
    candidate_domain: str
    guarantee: str
    diagnostics: dict


class RoutingContext:
    """Validated graph, component labels and lazily cached component subgraphs."""

    def __init__(self, graph, *, max_temp_mb=64, max_matrix_mb=512, physical_edges=None):
        if getattr(graph, 'format', None) != 'csr':
            raise ValueError('graph must be CSR')
        if not graph.has_canonical_format:
            raise ValueError('CSR must be sorted and free of duplicate entries')
        if graph.dtype.kind not in 'iuf':
            raise ValueError('graph weights must be real numeric values')
        self.graph = csr_matrix(graph, dtype=float, copy=True)
        n, m = self.graph.shape
        if n == 0 or n != m:
            raise ValueError('graph must be nonempty and square')
        if not np.isfinite(self.graph.data).all() or np.any(self.graph.data <= 0):
            raise ValueError('stored graph weights must be finite and strictly positive')
        if np.any(self.graph.diagonal() != 0) or (self.graph != self.graph.T).nnz:
            raise ValueError('graph must be loop-free and symmetric')
        self.n = n
        self.n_components, self.labels = connected_components(self.graph, directed=False)
        self.max_temp_mb = _positive(max_temp_mb, 'max_temp_mb')
        self.max_matrix_mb = _positive(max_matrix_mb, 'max_matrix_mb')
        self._components = {}
        # CSR routing length (parallel physical links already collapsed).
        mass = np.bincount(
            self.labels,
            weights=np.asarray(self.graph.sum(axis=1)).ravel() / 2,
            minlength=self.n_components,
        )
        if physical_edges is not None:
            u, v, length = physical_edges
            from net_center.edge_snap import _adjacency

            u = _indices(u, n, 'physical edge starts')
            v = _indices(v, n, 'physical edge ends')
            length = _vector(length, len(u), 'physical edge lengths', 0.)
            if (len(v) != len(u) or np.any(length <= 0) or np.any(u == v) or
                    np.any(self.labels[u] != self.labels[v])):
                raise ValueError('invalid physical edges for component length')
            if (self.graph != _adjacency(u, v, length, n)).nnz:
                raise ValueError('physical edge inventory disagrees with routing graph')
            mass = np.bincount(self.labels[u], weights=length,
                               minlength=self.n_components)
        if not np.isfinite(mass).all():
            raise ValueError('component length overflowed')
        self._component_mass = mass
        self.largest = int(np.argmax(mass))
        self.largest_basis = ('physical_length' if physical_edges is not None
                              else 'csr_weight')

    def _pin_original_largest(self, node):
        """Keep the original component rank through length-preserving splits."""
        node = _integer(node, 'largest component anchor', 0)
        if node >= self.n or self.largest_basis != 'physical_length':
            raise ValueError('largest component anchor requires physical network')
        component = int(self.labels[node])
        if not np.isclose(self._component_mass[component], self._component_mass.max(),
                          rtol=1e-12, atol=1e-9):
            raise ValueError('original largest component disagrees with split lengths')
        self.largest = component
        self.largest_basis = 'physical_length_original_rank'

    def local(self, component):
        component = int(component)
        if component not in self._components:
            ids = np.flatnonzero(self.labels == component)
            self._components[component] = (ids, self.graph[ids][:, ids].tocsr())
        return self._components[component]

    def blocks(self, graph, sources, *, limit=np.inf):
        rows = max(1, int(self.max_temp_mb * 2**20 // (8 * graph.shape[0])))
        for start in range(0, len(sources), rows):
            stop = min(start + rows, len(sources))
            block = dijkstra(graph, directed=True, indices=sources[start:stop], limit=limit)
            yield slice(start, stop), np.atleast_2d(block)

    def matrix(self, graph, sources, candidates):
        required = len(sources) * len(candidates) * 8 / 2**20
        if required > self.max_matrix_mb:
            raise MemoryError(
                f'candidate matrix needs {required:.2f} MiB; '
                f'limit is {self.max_matrix_mb:.2f} MiB'
            )
        result = np.empty((len(sources), len(candidates)), dtype=float)
        for rows, block in self.blocks(graph, sources):
            result[rows] = block[:, candidates]
        if not np.isfinite(result).all():
            raise ValueError('component-local distance matrix is not finite')
        return result


def _weighted_quantiles(distances, weights):
    """Inverse weighted empirical CDF; zero-weight rows do not affect quantiles."""
    positive = weights > 0
    values, mass = distances[positive], weights[positive]
    order = np.argsort(values, kind='stable')
    values, mass = values[order], mass[order]
    cumulative = np.cumsum(mass, dtype=float)
    total = _finite(cumulative[-1], 'quantile weight')
    return {str(percent): float(values[min(len(values)-1, int(np.searchsorted(
        cumulative, total * (percent / 100.), side='left'
    )))]) for percent in (50, 90, 95, 99)}


def _coalesce(nodes, weights, offsets):
    unique, inverse = np.unique(nodes, return_inverse=True)
    mass = np.bincount(inverse, weights=weights, minlength=len(unique))
    farthest = np.zeros(len(unique), dtype=float)
    np.maximum.at(farthest, inverse, offsets)
    if not np.isfinite(mass).all():
        raise ValueError('coalesced weights overflowed')
    return unique, mass, farthest


def _single(context, graph, demand, mass, candidates, offsets, objective, prune):
    """Streaming exact vertex optimization; no persistent demand-by-node matrix."""
    if objective == 'median':
        positive = mass > 0
        demand, mass = demand[positive], mass[positive]
        if not len(demand):
            return int(candidates[0]), 0.0
        limit = np.inf
        if prune:
            # Reference p is a feasible candidate. Triangle inequality bounds
            # every optimum by d(p,v) <= 2 J(p)/sum(w).
            p = int(candidates[0])
            reference = dijkstra(graph, directed=True, indices=p)[demand]
            baseline = _finite(mass @ reference, 'reference cost')
            radius = 2 * (baseline / _finite(mass.sum(), 'weight sum'))
            limit = (float(reference.max()) + radius) * (1 + 1e-9) + 1e-9
        scores = np.zeros(len(candidates), dtype=float)
        for rows, block in context.blocks(graph, demand, limit=limit):
            # All mass values strictly positive: no 0 * inf.
            scores += mass[rows] @ block[:, candidates]
    else:
        scores = np.zeros(len(candidates), dtype=float)
        for rows, block in context.blocks(graph, demand):
            scores = np.maximum(scores, (block[:, candidates] + offsets[rows, None]).max(0))
    best = int(np.argmin(scores))
    return int(candidates[best]), _finite(scores[best], 'single-site objective')


def _nearest(D, choice):
    """Reduce selected columns without allocating a demand-by-k temporary."""
    best = np.full(D.shape[0], np.inf)
    for j in choice:
        np.minimum(best, D[:, j], out=best)
    return best


def _check_combinations(n, cap, max_combinations):
    count = 0
    for k in range(1, min(cap, n) + 1):
        count += comb(n, k)
        if count > max_combinations:
            raise ValueError('exact enumeration exceeds max_combinations per component')


def _exact_curve(D, mass, candidates, cap, max_combinations):
    """Exhaustive reference solver, with a precomputed hard search gate."""
    cap = min(cap, len(candidates))
    _check_combinations(len(candidates), cap, max_combinations)
    result = []
    for k in range(1, cap + 1):
        best = None
        for choice in combinations(range(len(candidates)), k):
            cost = _finite(mass @ _nearest(D, choice), 'exact site cost')
            trial = (cost, choice)
            if best is None or trial < best:
                best = trial
        cost, choice = best
        result.append(CurvePoint(k, tuple(int(candidates[j]) for j in choice), cost))
    return tuple(result)


def _curve(D, mass, candidates, cap, *, swap_passes, block_cells=262144):
    """Deterministic BUILD + best improving 1-swap; exact cost evaluations."""
    def costs(base):
        result = np.empty(D.shape[1], dtype=float)
        width = max(1, block_cells // len(mass))
        for start in range(0, D.shape[1], width):
            end = min(start + width, D.shape[1])
            result[start:end] = mass @ np.minimum(base[:, None], D[:, start:end])
        if not np.isfinite(result).all():
            raise ValueError('multi-site objective overflowed')
        return result

    def cost(choice):
        return _finite(mass @ _nearest(D, choice), 'multi-site cost')

    chosen = [int(np.argmin(costs(np.full(len(mass), np.inf))))]
    result = [CurvePoint(1, (int(candidates[chosen[0]]),), cost(chosen))]
    for k in range(2, min(cap, len(candidates)) + 1):
        scores = costs(_nearest(D, chosen))
        scores[chosen] = np.inf
        chosen = sorted([*chosen, int(np.argmin(scores))])
        for _ in range(swap_passes):
            base_cost = cost(chosen)
            best = None
            for j in range(k):
                rest = chosen[:j] + chosen[j + 1:]
                trial_cost = costs(_nearest(D, rest))
                trial_cost[chosen] = np.inf
                candidate = int(np.argmin(trial_cost))
                value = float(trial_cost[candidate])
                trial = (value, j, candidate)
                if value < base_cost - 1e-12 * max(1., base_cost) and (
                    best is None or trial < best
                ):
                    best = trial
            if best is None:
                break
            _, j, candidate = best
            chosen[j] = candidate
            chosen.sort()
        result.append(CurvePoint(k, tuple(int(candidates[j]) for j in chosen),
                                 cost(chosen)))
    return tuple(result)


def _allocate(curves, cap):
    """Exact DP over supplied (possibly heuristic) component cost curves."""
    frontier = {0: CurvePoint(0, (), 0.)}
    for curve in curves:
        new = {}
        for previous in frontier.values():
            for item in curve:
                k = previous.k + item.k
                if k > cap:
                    continue
                trial = CurvePoint(k, tuple(sorted(previous.nodes + item.nodes)),
                                   _finite(previous.cost + item.cost, 'allocation cost'))
                old = new.get(k)
                if old is None or (trial.cost, trial.nodes) < (old.cost, old.nodes):
                    new[k] = trial
        if not new:
            raise ValueError('infeasible center budget')
        frontier = new
    return tuple(frontier[k] for k in sorted(frontier))


def _gap(context, graph, local_demand, row_mass, candidates, curve, rng,
         replicates, swap_passes, min_members, pool):
    """One-SE Gap using uniform component vertices and preserved row weights."""
    kmax = min(len(curve), max(1, len(candidates) - 1))
    if kmax <= 1 or curve[0].cost == 0 or len(pool) < 3:
        reason = ('zero_baseline' if curve[0].cost == 0 else
                  'insufficient_reference_support' if len(pool) < 3 else
                  'insufficient_k_range')
        return curve[0], {'outcome': reason, 'gap': [], 'tested_k': 0,
                          'reference_pool_size': len(pool)}
    # Explicit finite log floor, shared by observed and all reference curves.
    floor = max(np.finfo(float).tiny, curve[0].cost * 1e-12)
    logs = np.empty((replicates, kmax))
    for b in range(replicates):
        sample = rng.choice(pool, len(local_demand), replace=True)
        unique, mass, _ = _coalesce(sample, row_mass, np.zeros(len(sample)))
        D = context.matrix(graph, unique, unique)
        ref = _curve(D, mass, unique, kmax, swap_passes=swap_passes)
        for j in range(kmax):
            # At or above the distinct support size the medoid loss is zero.
            value = ref[j].cost if j < len(ref) else 0.
            logs[b, j] = np.log(max(floor, value))
    observed = np.array([p.cost for p in curve[:kmax]])
    gap = logs.mean(0) - np.log(np.maximum(floor, observed))
    se = logs.std(0, ddof=1) * np.sqrt(1 + 1 / replicates)
    chosen = kmax - 1
    outcome = 'upper_bound_reached'
    for j in range(kmax - 1):
        if gap[j] >= gap[j + 1] - se[j + 1]:
            chosen, outcome = j, 'one_se_rule'
            break
    selected = curve[chosen]
    if selected.k > 1 and min_members > 1:
        selected_nodes = np.asarray(selected.nodes, dtype=np.int64)
        # Caller passes local node ids for this curve.
        unique, counts_by_node = np.unique(
            local_demand, return_counts=True
        )
        D = context.matrix(graph, unique, selected_nodes)
        counts = np.bincount(D.argmin(1), weights=counts_by_node, minlength=selected.k)
        if np.any(counts < min_members):
            selected, outcome = curve[0], 'inadmissible_partition_kept_one'
    return selected, {'outcome': outcome, 'gap': gap.tolist(), 'se': se.tolist(),
                      'log_floor': floor, 'tested_k': kmax,
                      'reference_pool_size': len(pool)}


def solve_adaptive(graph, demand_nodes, *, weights=None, offsets=None,
                   objective='median', k=1, selection='fixed', max_centers=12,
                   component_policy='strict', candidate_domain='vertices',
                   candidate_nodes=None, include_offsets=False, penalty=None,
                   access_budget=None, gap_replicates=None, seed=None, min_members=None,
                   swap_passes=None, prune=None, context=None,
                   max_temp_mb=64, max_matrix_mb=512, coordinates=None,
                   reference_nodes=None, algorithm=None, max_combinations=None,
                   facility_cost=None, vertex_candidates=None):
    """Solve a weighted location/representation problem on a symmetric CSR graph.

    ``k`` is a global fixed budget (or one per populated component for
    ``per_component``). Non-fixed selection requires ``k=None``. All demand
    rows, including zero-weight rows, require coverage unless ``largest`` is
    explicitly selected. Weights affect the median but not vertex minimax.
    Offsets always enter reported access and budget; they enter the minimax
    objective only if ``include_offsets=True``. For Gap use demand candidates.

    Tuning options left as ``None`` take their defaults when the selected mode
    uses them: penalty 1, gap_replicates 10, seed 0, min_members 1, swap_passes
    3, prune True, algorithm 'heuristic', max_combinations 100000. Supplying an
    option that the selected mode never uses raises ``ValueError``. Resolved
    configuration is returned in ``diagnostics['configured_options']``;
    ``execution_by_component`` records the solver paths actually executed.

    A component whose modeled rows all have zero weight does not affect the
    median objective. It receives exactly one site, placed where the largest
    access distance (network plus connector) of its rows is smallest.
    """
    ctx = context if context is not None else RoutingContext(
        graph, max_temp_mb=max_temp_mb, max_matrix_mb=max_matrix_mb
    )
    if context is not None and graph is not ctx.graph:
        raise ValueError('when reusing context, pass context.graph as graph')
    q = _indices(demand_nodes, ctx.n, 'demand_nodes')
    w = _vector(weights, len(q), 'weights', 1.)
    off = _vector(offsets, len(q), 'offsets', 0.)
    total = _finite(w.sum(), 'weight sum')
    if total <= 0:
        raise ValueError('at least one positive demand weight is required')
    cap = _integer(max_centers, 'max_centers')
    if algorithm is not None and algorithm not in ('heuristic', 'exact'):
        raise ValueError('unknown multi-site algorithm')
    if prune is not None and not isinstance(prune, (bool, np.bool_)):
        raise ValueError('prune must be True or False')
    if selection not in ('fixed', 'balanced', 'access_budget', 'gap', 'penalized'):
        raise ValueError('unknown selection policy')
    if objective not in ('median', 'vertex_center'):
        raise ValueError('objective must be median or vertex_center')
    if component_policy not in ('strict', 'largest', 'per_component', 'allocate'):
        raise ValueError('unknown component policy')
    if candidate_domain not in ('vertices', 'demand', 'restricted'):
        raise ValueError('unknown candidate domain')
    if (candidate_domain == 'restricted') != (candidate_nodes is not None):
        raise ValueError('restricted domain requires candidate_nodes, others forbid it')
    if selection == 'fixed':
        k = _integer(k, 'k')
        if k > cap:
            raise ValueError('k exceeds max_centers')
    elif k is not None:
        raise ValueError('adaptive selection requires k=None')
    if objective == 'vertex_center' and (selection != 'fixed' or k != 1):
        raise ValueError('vertex minimax currently supports one center per solve/component')
    if component_policy == 'per_component' and (selection != 'fixed' or k != 1):
        raise ValueError('per_component requires fixed k=1; '
                         'use allocate for global budgets')
    if selection == 'access_budget':
        if access_budget is None or not np.isfinite(access_budget) or access_budget < 0:
            raise ValueError('access_budget must be finite and nonnegative')
    elif access_budget is not None:
        raise ValueError('access_budget only applies to access_budget selection')
    if selection == 'penalized':
        if facility_cost is None or not np.isfinite(facility_cost) or facility_cost < 0:
            raise ValueError('penalized selection requires nonnegative facility_cost')
    elif facility_cost is not None:
        raise ValueError('facility_cost only applies to penalized selection')
    # A multi-site curve can be evaluated only for the median with more than one
    # site per solve. Fixed k=1 and per_component always solve one site each.
    multi_site = objective == 'median' and not (
        selection == 'fixed' and (k == 1 or component_policy == 'per_component')
    )
    gap = selection == 'gap'
    _reject_unused({
        'penalty': (penalty, selection == 'balanced', "selection='balanced'"),
        'gap_replicates': (gap_replicates, gap, "selection='gap'"),
        'seed': (seed, gap, "selection='gap'"),
        'min_members': (min_members, gap, "selection='gap'"),
        'coordinates': (coordinates, gap, "selection='gap'"),
        'reference_nodes': (reference_nodes, gap, "selection='gap'"),
        'prune': (prune, objective == 'median', "objective='median'"),
        'algorithm': (algorithm, multi_site,
                      'multi-site median solves (not fixed k=1 or per_component)'),
    })
    algorithm = 'heuristic' if algorithm is None else algorithm
    _reject_unused({
        'max_combinations': (max_combinations, multi_site and algorithm == 'exact',
                             "algorithm='exact' in multi-site median solves"),
        'swap_passes': (swap_passes, multi_site and algorithm == 'heuristic',
                        "algorithm='heuristic' in multi-site median solves"),
    })
    penalty = 1. if penalty is None else penalty
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError('penalty must be finite and nonnegative')
    swap_passes = _integer(3 if swap_passes is None else swap_passes, 'swap_passes', 0)
    max_combinations = _integer(100000 if max_combinations is None else max_combinations,
                                'max_combinations')
    prune = True if prune is None else bool(prune)
    reference = None if reference_nodes is None else np.unique(
        _indices(reference_nodes, ctx.n, 'reference_nodes')
    )
    if gap and algorithm != 'heuristic':
        raise ValueError('Gap currently requires the heuristic solver for both curves')
    if gap and candidate_domain != 'demand':
        raise ValueError('Gap requires demand-node medoids')
    gap_replicates = _integer(10 if gap_replicates is None else gap_replicates,
                              'gap_replicates', 2)
    seed = _integer(0 if seed is None else seed, 'seed', 0)
    min_members = _integer(1 if min_members is None else min_members, 'min_members')
    configured = {}
    if objective == 'median':
        configured['prune'] = prune
    if multi_site:
        configured['algorithm'] = algorithm
        if algorithm == 'exact':
            configured['max_combinations'] = max_combinations
        else:
            configured['swap_passes'] = swap_passes
    if selection == 'balanced':
        configured['penalty'] = float(penalty)
    if gap:
        configured.update(gap_replicates=gap_replicates, seed=seed,
                          min_members=min_members)
    if coordinates is not None:
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.shape != (ctx.n, 2) or not np.isfinite(coordinates).all():
            raise ValueError('coordinates must be finite n-by-2 array')
    restricted = None if candidate_nodes is None else np.unique(
        _indices(candidate_nodes, ctx.n, 'candidate_nodes')
    )
    if vertex_candidates is not None and candidate_domain != 'vertices':
        raise ValueError('vertex_candidates only applies to vertices domain')
    eligible_vertices = (None if vertex_candidates is None else np.unique(
        _indices(vertex_candidates, ctx.n, 'vertex_candidates')
    ))
    labels = ctx.labels[q]
    modeled = np.ones(len(q), dtype=bool)
    if component_policy == 'largest':
        modeled = labels == ctx.largest
    if not modeled.any() or w[modeled].sum() <= 0:
        raise ValueError('component policy leaves no positive modeled demand')
    components = np.unique(labels[modeled])
    C = len(components)
    if component_policy == 'strict' and C != 1:
        raise ValueError('demand spans disconnected components; select a component policy')
    if cap < C or (selection == 'fixed' and component_policy != 'per_component' and k < C):
        raise ValueError('budget is smaller than number of populated components')
    if objective == 'vertex_center' and C > 1 and component_policy != 'per_component':
        raise ValueError('multi-component vertex minimax requires per_component')
    curves, gap_choices, gap_audit, zero_weight = [], [], {}, []
    execution = {}
    for comp in components:
        rows = np.flatnonzero(modeled & (labels == comp))
        ids, sub = ctx.local(comp)
        local_q = np.searchsorted(ids, q[rows])
        unique, mass, farthest = _coalesce(local_q, w[rows], off[rows])
        # The median objective does not depend on a site serving only zero weight.
        no_mass = objective == 'median' and not np.any(mass > 0)
        if candidate_domain == 'demand':
            cand = unique
        elif candidate_domain == 'restricted':
            cand = np.searchsorted(ids, restricted[ctx.labels[restricted] == comp])
        elif eligible_vertices is not None:
            cand = np.searchsorted(
                ids, eligible_vertices[ctx.labels[eligible_vertices] == comp]
            )
        else:
            cand = np.arange(len(ids), dtype=np.int64)
        if not len(cand):
            raise ValueError(f'component {comp} has no eligible candidate')
        local_cap = min(len(cand), cap - C + 1)
        if selection == 'balanced':
            local_cap = min(local_cap, len(unique))
        if selection == 'gap':
            local_cap = min(local_cap, max(1, len(unique) - 1))
        if selection == 'fixed':
            local_cap = min(local_cap, 1 if component_policy == 'per_component' else k-C+1)
        if no_mass:
            # Exactly one site, at the smallest largest access distance (network
            # plus connector) of these rows. Its median cost is zero.
            node, _ = _single(ctx, sub, unique, mass, cand, farthest, 'vertex_center',
                              prune)
            local_curve = (CurvePoint(1, (node,), 0.),)
            zero_weight.append(int(comp))
            solver = 'zero_weight_minimax'
        elif local_cap == 1 or objective == 'vertex_center':
            node, value = _single(ctx, sub, unique, mass, cand,
                                  farthest if include_offsets else np.zeros(len(unique)),
                                  objective, prune)
            local_curve = (CurvePoint(1, (node,), value),)
            solver = 'streaming_median' if objective == 'median' else 'vertex_minimax'
        else:
            # Gate exact combinatorial work before allocating the dense matrix.
            if algorithm == 'exact':
                _check_combinations(len(cand), local_cap, max_combinations)
            D = ctx.matrix(sub, unique, cand)
            if algorithm == 'exact':
                local_curve = _exact_curve(D, mass, cand, local_cap, max_combinations)
            else:
                local_curve = _curve(D, mass, cand, local_cap, swap_passes=swap_passes)
            del D
            solver = 'exact_curve' if algorithm == 'exact' else 'heuristic_curve'
        execution[str(comp)] = {
            'solver': solver,
            'evaluated_k': [p.k for p in local_curve],
            'median_pruning_enabled': prune if solver == 'streaming_median' else None,
        }
        if gap:
            if no_mass:
                selected = local_curve[0]
                audit = {'outcome': 'zero_weight_component', 'gap': [], 'tested_k': 0,
                         'reference_pool_size': 0}
            else:
                pool = (np.arange(len(ids), dtype=np.int64) if reference is None else
                        np.searchsorted(ids, reference[ctx.labels[reference] == comp]))
                if coordinates is not None:
                    xy = coordinates[ids]
                    lo, hi = xy[unique].min(0), xy[unique].max(0)
                    pool = pool[np.all((xy[pool] >= lo) & (xy[pool] <= hi), axis=1)]
                rng = np.random.default_rng(np.random.SeedSequence([seed, int(comp)]))
                selected, audit = _gap(ctx, sub, local_q, w[rows], cand, local_curve,
                                       rng, gap_replicates, swap_passes, min_members,
                                       pool)
            gap_choices.append(CurvePoint(selected.k, tuple(int(ids[j]) for j in
                                                           selected.nodes), selected.cost))
            gap_audit[str(comp)] = audit
            execution[str(comp)]['gap_reference_replicates'] = (
                gap_replicates if audit['tested_k'] else 0
            )
        curves.append(tuple(CurvePoint(p.k, tuple(int(ids[j]) for j in p.nodes), p.cost)
                            for p in local_curve))
    offset_cost = _finite(w[modeled] @ off[modeled], 'connector cost')
    modeled_weight = _finite(w[modeled].sum(), 'modeled weight')
    if objective == 'vertex_center':
        chosen = CurvePoint(C, tuple(sorted(p[0].nodes[0] for p in curves)),
                            max(p[0].cost for p in curves))
        frontier = (chosen,)
    else:
        frontier = _allocate(curves, cap)
        if selection == 'fixed':
            target = C if component_policy == 'per_component' else k
            chosen = next((p for p in frontier if p.k == target), None)
            if chosen is None:
                raise ValueError('fixed k exceeds available candidates' + (
                    '; a component with zero total weight receives exactly one site'
                    if zero_weight else ''))
        elif selection == 'balanced':
            base = frontier[0].cost
            n = len(np.unique(q[modeled]))
            # Never reward adding sites beyond the distinct demand support.
            eligible = [p for p in frontier if p.k <= n]
            chosen = min(eligible, key=lambda p: (
                (p.cost / base if base else 0.) + penalty * (p.k-C) / max(1, n-C), p.k
            ))
        elif selection == 'penalized':
            chosen = min(frontier, key=lambda p: (
                _finite(p.cost + facility_cost * p.k, 'penalized cost'), p.k
            ))
        elif selection == 'access_budget':
            chosen = next((p for p in frontier if
                           (p.cost + offset_cost) / modeled_weight <= access_budget +
                           1e-12 * max(1., access_budget)), None)
            if chosen is None:
                raise ValueError('access budget unmet on evaluated cost curves')
        else:
            selected_nodes = tuple(sorted(j for p in gap_choices for j in p.nodes))
            if len(selected_nodes) > cap:
                raise ValueError('independent component Gap selections exceed global cap')
            chosen = CurvePoint(len(selected_nodes), selected_nodes,
                                _finite(sum(p.cost for p in gap_choices), 'Gap cost'))
    centers = np.asarray(chosen.nodes, dtype=np.int64)
    assignment = np.full(len(q), -1, dtype=np.int64)
    road = np.full(len(q), np.inf)
    for comp in components:
        rows = np.flatnonzero(modeled & (labels == comp))
        ids, sub = ctx.local(comp)
        local_q = np.searchsorted(ids, q[rows])
        for j in np.flatnonzero(ctx.labels[centers] == comp):
            d = dijkstra(sub, directed=True, indices=int(np.searchsorted(ids, centers[j])))
            dist = d[local_q]
            improve = dist < road[rows]
            road[rows[improve]] = dist[improve]
            assignment[rows[improve]] = j
    if not np.isfinite(road[modeled]).all() or np.any(assignment[modeled] < 0):
        raise RuntimeError('selected sites failed to cover modeled demand')
    access = road + off
    if not np.isfinite(access[modeled]).all():
        raise ValueError('access distance overflowed')
    a = assignment[modeled]
    site_weight = np.bincount(a, weights=w[modeled], minlength=len(centers))
    site_count = np.bincount(a, minlength=len(centers))
    site_sum = np.bincount(a, weights=w[modeled]*access[modeled], minlength=len(centers))
    if not np.isfinite(site_sum).all():
        raise ValueError('site access sum overflowed')
    mean = np.divide(site_sum, site_weight, out=np.full(len(centers), np.nan),
                     where=site_weight > 0)
    maximum = np.full(len(centers), -np.inf)
    np.maximum.at(maximum, a, access[modeled])
    maximum[site_count == 0] = np.nan
    road_cost = _finite(w[modeled] @ road[modeled], 'road cost')
    mean_access = _finite(w[modeled] @ access[modeled] / modeled_weight, 'mean access')
    if selection == 'access_budget' and mean_access > (
        access_budget + 1e-12 * max(1., access_budget)
    ):
        raise RuntimeError('recomputed mean access exceeds the requested budget')
    radius = float((access if include_offsets else road)[modeled].max())
    objective_value = (road_cost + (offset_cost if include_offsets else 0.)
                       if objective == 'median' else radius)
    expected_cost = road_cost if objective == 'median' else radius
    if not np.isclose(expected_cost, chosen.cost, rtol=1e-9, atol=1e-9):
        raise RuntimeError('evaluated cost disagrees with routed assignments')
    selection_score = (
        _finite(road_cost + facility_cost * len(centers), 'penalized cost')
        if selection == 'penalized' else None
    )
    exact = objective == 'vertex_center' or (len(centers) == C and selection == 'fixed')
    guarantee = ('exact_single_site_per_component' if exact else
                 'exact_evaluated_site_curves' if algorithm == 'exact' else
                 'heuristic_site_curves')
    return AdaptiveResult(
        centers, assignment, road, access, site_weight, site_count, mean, maximum, labels,
        _finite(objective_value, 'objective'), road_cost, total, modeled_weight,
        modeled_weight/total, mean_access, float(access[modeled].max()), frontier,
        selection, candidate_domain, guarantee,
        {'populated_components': C, 'selected_k': len(centers),
                                 'algorithm': algorithm if any(
                                     item['solver'] in ('exact_curve', 'heuristic_curve')
                                     for item in execution.values()
                                 ) else None,
                                 'configured_options': configured,
                                 'execution_by_component': execution,
                                 'zero_weight_components': zero_weight,
                                 'largest_basis': ctx.largest_basis,
                                 'gap_reference': None if not gap else (
                                     ('explicit_nodes' if reference is not None
                                      else 'component_vertices') +
                                     ('_in_demand_bounding_box'
                                      if coordinates is not None else '')),
                                 'facility_cost': facility_cost,
                                 'penalized_selection_cost': selection_score,
                                 'cap_reached': len(centers) == cap,
                                 'gap_by_component': gap_audit,
                                 'connector_cost': offset_cost,
                                 'modeled_row_fraction': float(modeled.mean()),
                                 'mean_network_access': road_cost / modeled_weight,
                                 'weighted_access_percentiles': _weighted_quantiles(
                                     access[modeled], w[modeled]
                                 ),
                                 'objective': objective,
                                 'component_policy': component_policy,
                                 'includes_offsets': bool(include_offsets),
                                 'curve_cost': 'network_sum' if objective == 'median'
                                 else ('maximum_component_access_radius' if include_offsets
                                       else 'maximum_component_network_radius')}
    )


def solve_groups(graph, demand_nodes, groups, *, weights=None, offsets=None,
                 base_vertex_count=None, largest_component_anchor=None, **options):
    """Return {group: (original row indices, AdaptiveResult)}; fail as a unit."""
    opts = dict(options)
    context = RoutingContext(graph, max_temp_mb=opts.pop('max_temp_mb', 64),
                             max_matrix_mb=opts.pop('max_matrix_mb', 512),
                             physical_edges=opts.pop('physical_edges', None))
    if largest_component_anchor is not None:
        if opts.get('component_policy') != 'largest':
            raise ValueError('largest component anchor requires largest policy')
        context._pin_original_largest(largest_component_anchor)
    q = _indices(demand_nodes, context.n, 'demand_nodes')
    w = _vector(weights, len(q), 'weights', 1.)
    off = _vector(offsets, len(q), 'offsets', 0.)
    labels = np.asarray(groups, dtype=object)
    if labels.shape != q.shape or any(not isinstance(g, str) or not g for g in labels):
        raise ValueError('groups must be aligned nonempty strings')
    if base_vertex_count is not None:
        base_vertex_count = _integer(base_vertex_count, 'base_vertex_count')
        if base_vertex_count > context.n:
            raise ValueError('base_vertex_count exceeds graph size')
        if opts.get('candidate_domain', 'vertices') != 'vertices':
            raise ValueError('base_vertex_count only applies to vertices domain')
        if 'vertex_candidates' in opts:
            raise ValueError('base_vertex_count conflicts with vertex_candidates')
    output = {}
    for group in sorted(set(labels)):
        rows = np.flatnonzero(labels == group)
        group_options = dict(opts)
        if base_vertex_count is not None:
            group_options['vertex_candidates'] = np.union1d(
                np.arange(base_vertex_count), q[rows]
            )
        output[group] = (rows, solve_adaptive(context.graph, q[rows], weights=w[rows],
                                              offsets=off[rows], context=context,
                                              **group_options))
    return output
