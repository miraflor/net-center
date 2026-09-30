"""The two location problems, and the algorithms that solve them.

READING THIS FILE WITHOUT A PROGRAMMING BACKGROUND
--------------------------------------------------
Three ideas cover almost everything below.

*Arrays.* An array is a grid of numbers. ``D`` is our main one: row ``v``,
column ``j`` holds the road distance from demand point ``v`` to intersection
``j``. So ``D`` has one row per demand and one column per intersection.

*Vectorisation.* Rather than looping over rows one at a time, we ask numpy to
do the same arithmetic to every row simultaneously. ``a + b`` where both are
grids adds them cell by cell, in optimised C code. This is why the code below
looks like algebra rather than like step-by-step instructions -- each statement
is performing thousands of small calculations at once.

*Axis.* Operations take an ``axis`` telling them which direction to work in.
``axis=0`` collapses down the columns, ``axis=1`` collapses across the rows.
``D.max(axis=0)`` gives, for each intersection, the distance to its FARTHEST
demand point.

WHAT THE TWO PROBLEMS ARE
-------------------------
Both ask "where is the middle of this road network?" and they disagree.

The MEDIAN minimises the total distance everyone travels: the sum of
``weight * distance``. It is pulled towards where the people are. Hakimi proved
in 1964 that a best median always sits exactly on an intersection, so finding
it is just "score every intersection, keep the best" -- one pass over ``D``.

The CENTRE minimises the distance for the worst-off person: the largest
distance to anyone. It is pulled towards the geographic extremes and is decided
by the two or three most remote places. Crucially Hakimi's result does NOT hold
here: the best point can sit partway along a road, between two intersections.
Finding it needs the sweep described next.

HOW THE CENTRE SWEEP WORKS
--------------------------
Walk along one road segment from intersection ``u`` to intersection ``w``, and
call your position ``t`` metres from ``u``. The segment is ``L`` metres long.

To reach some demand point ``v`` you either walk back to ``u`` and travel from
there, or forward to ``w`` and travel from there. So your distance is

    d(t, v) = min(a_v + t,  b_v + L - t)

where ``a_v`` is the road distance from ``u`` to ``v`` and ``b_v`` from ``w``
to ``v``. As ``t`` grows this rises at first (going via ``u`` is still the
better route) and later falls (going via ``w`` takes over). Plotted against
``t`` it is a tent shape.

Your eccentricity at ``t`` is the worst of these over all demands -- the
highest tent above you. With many tents that upper outline is a jagged ridge of
peaks and valleys, and we want its lowest valley.

Tent ``v`` is still rising exactly while ``t <= t*_v = (b_v - a_v + L) / 2``.
Sort the demands by that switchover point. Between two consecutive switchovers
the set of still-rising tents is a fixed block at one end of the sorted list
and the set of already-falling tents a fixed block at the other, so across that
whole stretch the ridge simplifies to

    ecc(t) = max(A + t,  B + L - t)

where ``A`` is the largest ``a`` among the risers and ``B`` the largest ``b``
among the fallers -- two numbers that do not change within the stretch. That is
a simple V shape, and the bottom of a V is found by setting its two lines
equal. So each stretch is solved by arithmetic, with no searching or sampling.

Sorting costs ``k log k`` per segment for ``k`` demands. Everything else is
running maximums, which numpy computes in a single pass.

WHICH SEGMENTS ARE SWEPT
------------------------
The best intersection already gives a feasible radius ``R``. A segment is swept
only if a lower bound on the eccentricity of every point on it is below ``R``.
Two valid bounds are used, cheapest first:

1. From the endpoint eccentricities alone (one subtraction per segment).
   A point at ``t`` is at most ``t`` from ``u``, so by the triangle inequality
   ``ecc(t) >= ecc(u) - t``; likewise ``ecc(t) >= ecc(w) - (L - t)``. The
   larger of the two is smallest where they are equal, so for every ``t``

       ecc(t) >= (ecc(u) + ecc(w) - L) / 2.

2. From every demand (``k`` operations per segment), applied only to segments
   that pass bound 1. Every route leaves the segment through ``u`` or ``w``,
   so ``d(t, v) >= min(a_v, b_v)`` and ``ecc(t) >= max_v min(a_v, b_v)``.

A segment is swept only if it passes both bounds. On road networks bound 1
alone usually leaves a handful of segments, so the ``k``-per-segment work of
bound 2 is spent almost nowhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from net_center import _checks

# How many array cells to process at once. Peak memory in the sweep is roughly
# 15 arrays of this size in 8-byte floats, so 1,000,000 is about 120 MB per
# worker. Lower it on a memory-constrained machine; it never changes results.
DEFAULT_MAX_CELLS = 1_000_000

# Below this much work, worker startup is usually not worth attempting.
MIN_PARALLEL_WORK = 5_000_000


@dataclass
class CenterResult:
    """One solved location and its objective value.

    A vertex result has ``node`` set. A continuous-network result has ``edge``
    and ``t`` set, where ``t`` is distance from that edge's ``u`` endpoint.
    ``xy`` is attached later by :mod:`net_center.solve` because the mathematical
    core intentionally knows nothing about map coordinates.
    """

    kind: str
    objective: float
    node: int | None = None
    edge: int | None = None
    t: float | None = None
    xy: tuple[float, float] | None = None


# ---------------------------------------------------------------------------
# Input checks
# ---------------------------------------------------------------------------


def _check_distance_matrix(D) -> np.ndarray:
    """Validate the demand-by-node distance matrix used by every solver.

    NaN propagates through ``min()`` and ``max()``, and an infinity shows up in
    one of them, so two reductions check every cell. This avoids the two
    full-size boolean masks that ``isfinite(D).all()`` and ``(D < 0).any()``
    would allocate.
    """
    D = np.asarray(D)
    if D.ndim != 2:
        raise ValueError("D must be a 2-D array shaped (demands, nodes)")
    if D.shape[0] == 0:
        raise ValueError("D contains no demand points")
    if D.shape[1] == 0:
        raise ValueError("D contains no network nodes")
    if not (np.issubdtype(D.dtype, np.integer) or np.issubdtype(D.dtype, np.floating)):
        raise ValueError("D must contain real numeric distances")
    smallest, largest = D.min(), D.max()
    if not (np.isfinite(smallest) and np.isfinite(largest)):
        raise ValueError("D contains NaN or infinite distances")
    if smallest < 0:
        raise ValueError("D contains negative distances")
    return D


def _check_edges(n_nodes: int, edge_u, edge_w, edge_len):
    """Normalise and validate the segment arrays used by the centre sweep."""
    edge_u = _checks.index_array(edge_u, "edge_u and edge_w")
    edge_w = _checks.index_array(edge_w, "edge_u and edge_w")
    edge_len = np.asarray(edge_len, dtype=np.float64).reshape(-1)
    if not (len(edge_u) == len(edge_w) == len(edge_len)):
        raise ValueError("edge_u, edge_w and edge_len must be the same length")
    if len(edge_u):
        if edge_u.min() < 0 or edge_w.min() < 0:
            raise ValueError("edge endpoints must be non-negative node indices")
        if edge_u.max() >= n_nodes or edge_w.max() >= n_nodes:
            raise ValueError("edge endpoint index is outside D's node columns")
        if not np.isfinite(edge_len).all():
            raise ValueError("edge lengths must be finite")
        if (edge_len <= 0).any():
            raise ValueError("edge lengths must be strictly positive")
    return edge_u, edge_w, edge_len


# ---------------------------------------------------------------------------
# Median and vertex centre
# ---------------------------------------------------------------------------


def _median_objective(D: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    """Total (weighted) distance from every node to the demand points.

    Sums are accumulated in float64 even when ``D`` is stored as float32:
    adding thousands of float32 values in float32 loses tens of metres.
    Weighted sums are formed in row blocks so that a float32 ``D`` is never
    converted to float64 in one piece, which would silently allocate a copy
    twice the size of the matrix that ``--float32`` was meant to shrink.
    """
    if weights is None:
        return D.sum(axis=0, dtype=np.float64)
    total = np.zeros(D.shape[1], dtype=np.float64)
    rows = max(1, DEFAULT_MAX_CELLS // D.shape[1])
    for start in range(0, D.shape[0], rows):
        block = D[start : start + rows].astype(np.float64, copy=False)
        total += weights[start : start + rows] @ block
        del block  # release before the next block is allocated
    return total


def vertex_eccentricity(D: np.ndarray) -> np.ndarray:
    """For every node, return its distance to the farthest demand point."""
    return _check_distance_matrix(D).max(axis=0)


def vertex_center(D: np.ndarray) -> CenterResult:
    """Best *node* under the minimax objective (the Jordan centre)."""
    ecc = vertex_eccentricity(D)
    node = int(np.argmin(ecc))
    return CenterResult("vertex_center", float(ecc[node]), node=node)


def weighted_median(D: np.ndarray, weights=None) -> CenterResult:
    """Best node under total weighted distance.

    Hakimi's vertex-optimality result means an optimum for vertex demand can be
    chosen at a network vertex, so the continuous edge interiors do not need a
    second search for this objective.
    """
    D = _check_distance_matrix(D)
    w = None if weights is None else _checks.demand_weights(weights, D.shape[0])
    obj = _median_objective(D, w)
    node = int(np.argmin(obj))
    return CenterResult("median", float(obj[node]), node=node)


# ---------------------------------------------------------------------------
# Absolute centre: pruning bounds and the exact sweep
# ---------------------------------------------------------------------------


def _eccentricity_bound(ecc, edge_u, edge_w, edge_len) -> np.ndarray:
    """Lower bound ``(ecc(u) + ecc(w) - L) / 2`` for every point on each edge.

    Derived in the module docstring (bound 1). Costs O(1) per edge because the
    vertex eccentricities are already known from the vertex centre.
    """
    return 0.5 * (ecc[edge_u].astype(np.float64) + ecc[edge_w] - edge_len)


def _endpoint_bound(D, edge_u, edge_w, max_cells) -> np.ndarray:
    """Lower bound ``max_v min(D[v, u], D[v, w])`` for every point on each edge.

    Derived in the module docstring (bound 2). Costs O(k) per edge, so it is
    only applied to edges that survive :func:`_eccentricity_bound`.
    """
    m = len(edge_u)
    step = max(1, max_cells // D.shape[0])
    out = np.empty(m, dtype=np.float64)
    for s in range(0, m, step):
        e = min(s + step, m)
        out[s:e] = np.minimum(D[:, edge_u[s:e]], D[:, edge_w[s:e]]).max(axis=0)
    return out


def _sweep(D, edge_u, edge_w, edge_len, max_cells):
    """Solve the exact minimax point on every supplied segment.

    Arrays inside each block have shape ``(segments, demands)``. The block size
    is derived from ``max_cells`` so the temporary arrays have a predictable
    upper bound independent of the total number of road segments.
    """
    m = len(edge_u)
    k = D.shape[0]
    step = max(1, max_cells // k)
    best_val = np.empty(m, dtype=np.float64)
    best_t = np.empty(m, dtype=np.float64)

    for s in range(0, m, step):
        e = min(s + step, m)

        # Distances from both endpoints to every demand. Making these arrays
        # contiguous pays for itself because the next operation sorts each row.
        a = np.ascontiguousarray(D[:, edge_u[s:e]].T, dtype=np.float64)
        b = np.ascontiguousarray(D[:, edge_w[s:e]].T, dtype=np.float64)
        L = edge_len[s:e][:, None]
        n_e = a.shape[0]

        # A demand switches from the u-route to the w-route at t*. A valid
        # shortest-path matrix should already put t* in [0,L]; clipping is a
        # defensive guard against tiny numerical violations.
        tstar = np.clip(0.5 * (b - a + L), 0.0, L)
        order = np.argsort(tstar, axis=1, kind="stable")
        ts = np.take_along_axis(tstar, order, axis=1)
        a_s = np.take_along_axis(a, order, axis=1)
        b_s = np.take_along_axis(b, order, axis=1)
        del a, b, order, tstar

        # k switch points create k+1 intervals. On each interval the upper
        # envelope reduces to max(A+t, B+L-t), a V whose minimum is analytic.
        # -inf padding means "no riser" / "no faller" at the two outer ends.
        neg = np.full((n_e, 1), -np.inf)
        lo = np.concatenate([np.zeros((n_e, 1)), ts], axis=1)
        hi = np.concatenate([ts, L], axis=1)
        A = np.concatenate(
            [np.maximum.accumulate(a_s[:, ::-1], axis=1)[:, ::-1], neg], axis=1
        )
        B = np.concatenate([neg, np.maximum.accumulate(b_s, axis=1)], axis=1)
        del a_s, b_s, ts

        tc = np.clip(0.5 * (B + L - A), lo, hi)
        val = np.maximum(A + tc, B + L - tc)
        del A, B, lo, hi

        j = np.argmin(val, axis=1)
        rows = np.arange(n_e)
        best_val[s:e] = val[rows, j]
        best_t[s:e] = tc[rows, j]

    return best_val, best_t


def _roundoff_margin(dtype, largest: float) -> float:
    """Scale-aware allowance for rounding in the stored distances.

    D may deliberately be stored as float32. The centre is then exact for the
    stored distances, not for the unrounded float64 values SciPy originally
    produced. Pruning must never discard an edge merely because rounding made
    its lower bound a hair too large, so bounds are compared with this margin.
    """
    if np.issubdtype(dtype, np.floating):
        return 8.0 * float(np.finfo(dtype).eps) * max(1.0, largest)
    return 0.0


def absolute_center(
    D: np.ndarray,
    edge_u: np.ndarray,
    edge_w: np.ndarray,
    edge_len: np.ndarray,
    n_jobs: int = 1,
    backend: str = "threading",
    max_cells: int = DEFAULT_MAX_CELLS,
    min_parallel_work: int = MIN_PARALLEL_WORK,
    tol: float = 1e-9,
) -> CenterResult:
    """Return the exact absolute 1-centre for vertex demand.

    The best vertex gives an incumbent radius. Two mathematically safe lower
    bounds (see the module docstring) then discard segments that cannot improve
    it, and only the survivors are swept exactly. If no segment improves on
    the best vertex by more than the numerical margin, the vertex result is
    returned (``kind == "vertex_center"``).
    """
    D = _check_distance_matrix(D)
    edge_u, edge_w, edge_len = _check_edges(D.shape[1], edge_u, edge_w, edge_len)
    max_cells = _checks.positive_int(max_cells, "max_cells")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be a finite non-negative number")
    min_parallel_work = _checks.non_negative_int(min_parallel_work, "min_parallel_work")
    n_jobs = _checks.positive_int(n_jobs, "n_jobs")
    backend = _checks.backend(backend)

    # Incumbent: the vertex centre. Its eccentricities also feed bound 1.
    ecc = D.max(axis=0)
    best_node = int(np.argmin(ecc))
    radius = float(ecc[best_node])
    best = CenterResult("vertex_center", radius, node=best_node)
    if len(edge_u) == 0:
        return best

    margin = max(float(tol), _roundoff_margin(D.dtype, float(ecc.max())))
    threshold = radius + margin

    # Bound 1 on every edge, then bound 2 on the survivors only.
    keep = np.flatnonzero(_eccentricity_bound(ecc, edge_u, edge_w, edge_len) < threshold)
    if keep.size:
        bound2 = _endpoint_bound(D, edge_u[keep], edge_w[keep], max_cells)
        keep = keep[bound2 < threshold]
    if keep.size == 0:
        return best

    too_small = keep.size * D.shape[0] < min_parallel_work
    if n_jobs == 1 or keep.size < 2 * n_jobs or too_small:
        vals, offs = _sweep(D, edge_u[keep], edge_w[keep], edge_len[keep], max_cells)
    else:
        from joblib import Parallel, delayed

        blocks = [blk for blk in np.array_split(keep, n_jobs) if blk.size]
        parts = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_sweep)(D, edge_u[b], edge_w[b], edge_len[b], max_cells) for b in blocks
        )
        vals = np.concatenate([p[0] for p in parts])
        offs = np.concatenate([p[1] for p in parts])

    i = int(np.argmin(vals))
    if vals[i] >= radius - margin:
        return best
    return CenterResult(
        "absolute_center",
        float(vals[i]),
        edge=int(keep[i]),
        t=float(offs[i]),
    )
