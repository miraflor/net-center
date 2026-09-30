"""Shortest-path distance matrices with bounded temporary memory.

The expensive object in this project is usually not the sparse road graph but
``D``: the ``k x n`` matrix of distances from ``k`` demand nodes to ``n``
network nodes.  This module keeps that cost explicit and predictable:

* callers choose the stored floating-point dtype;
* SciPy Dijkstra output is computed in bounded row chunks;
* parallelism is opt-in because extra workers also increase temporary memory;
* parallel results are streamed into a preallocated output matrix instead of
  being accumulated and then copied by ``numpy.vstack``.

SciPy supplies Dijkstra itself.  The code here only validates and schedules the
work around it.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from net_center import _checks

MIN_PARALLEL_WORK = 20_000_000
DEFAULT_TEMP_MB = 256


def estimate_distance_matrix_mb(n_sources: int, n_nodes: int, dtype=np.float64) -> float:
    """Approximate memory occupied by the *stored* distance matrix in MiB."""
    if n_sources < 0 or n_nodes < 0:
        raise ValueError("n_sources and n_nodes must be non-negative")
    itemsize = np.dtype(dtype).itemsize
    return float(n_sources) * float(n_nodes) * itemsize / (1024.0**2)


def _validate_graph(graph: csr_matrix) -> csr_matrix:
    """Return CSR input after checking assumptions required by Dijkstra."""
    graph = graph.tocsr(copy=False)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("graph must be a square sparse matrix")
    if graph.data.size:
        if not np.isfinite(graph.data).all():
            raise ValueError("graph contains NaN or infinite edge lengths")
        if (graph.data < 0).any():
            raise ValueError("Dijkstra requires non-negative edge lengths")

    # net-center models an undirected network, which is only meaningful when the
    # cost matrix itself is symmetric.  Fail here rather than guess.
    delta = graph - graph.T
    if delta.nnz and not np.allclose(delta.data, 0.0, rtol=1e-12, atol=1e-12):
        raise ValueError(
            "graph must be symmetric because net-center models an undirected network"
        )
    return graph


def _chunk(graph: csr_matrix, sources: np.ndarray, dtype) -> np.ndarray:
    """Compute one bounded block of source-to-node distances."""
    # The graph has been checked to be symmetric, so every road is already
    # stored in both directions and ``directed=True`` gives the same distances
    # as ``directed=False``.  It is faster because SciPy's undirected mode
    # builds a transposed copy of the graph on every call and then relaxes each
    # edge twice (once from each stored direction).
    block = dijkstra(graph, directed=True, indices=sources)
    # SciPy computes in float64.  Casting each bounded block immediately
    # prevents float32 users from holding one complete float64 matrix
    # alongside the complete float32 matrix.
    return block.astype(dtype, copy=False)


def _source_chunks(
    sources: np.ndarray, n_nodes: int, max_temp_mb: float, min_chunks: int = 1
):
    """Split sources into row blocks for Dijkstra.

    Each block's float64 output stays within ``max_temp_mb``.  When
    ``min_chunks > 1`` (parallel execution), sources are also split into at
    least that many blocks, so every worker receives work even when the whole
    problem would fit in one memory block.
    """
    if not np.isfinite(max_temp_mb) or max_temp_mb <= 0:
        raise ValueError("max_temp_mb must be a finite positive number")
    bytes_per_row = max(1, n_nodes) * np.dtype(np.float64).itemsize
    rows_for_memory = max(1, int(max_temp_mb * 1024**2 // bytes_per_row))
    rows_for_workers = -(-len(sources) // max(1, min_chunks))  # ceiling division
    rows = max(1, min(rows_for_memory, rows_for_workers))
    for start in range(0, len(sources), rows):
        yield sources[start : start + rows]


def distance_matrix(
    graph: csr_matrix,
    sources,
    n_jobs: int = 1,
    backend: str = "loky",
    dtype=np.float64,
    min_parallel_work: int = MIN_PARALLEL_WORK,
    max_temp_mb: float = DEFAULT_TEMP_MB,
) -> np.ndarray:
    """Return road distances from each source to every network node.

    Parameters
    ----------
    graph
        Square symmetric sparse adjacency matrix with non-negative edge lengths.
    sources
        Node indices from which shortest paths should be calculated.
    n_jobs
        Maximum workers.  ``1`` is the deliberately conservative default.
    backend
        Joblib backend used only when ``n_jobs > 1`` and the problem is large
        enough to parallelise.  ``"loky"`` uses processes and can provide real
        CPU parallelism at the cost of process/memory overhead; ``"threading"``
        shares Python memory and may be preferable on memory-constrained systems.
        Which is faster is workload- and SciPy-build-dependent, so no speedup is
        assumed by the API.
    dtype
        Storage dtype for the returned matrix.  ``float32`` roughly halves the
        stored matrix size; SciPy still creates each temporary Dijkstra block in
        float64.
    min_parallel_work
        Skip worker startup below this ``sources x nodes`` work estimate.
    max_temp_mb
        Target maximum float64 output size of *one* Dijkstra chunk in MiB.  With
        multiple workers, several chunks may exist concurrently; reduce this
        value when memory rather than CPU is the limiting resource.

    Notes
    -----
    When parallel execution is used, sources are split into at least
    ``n_jobs`` blocks.  Parallel results are consumed as an ordered generator
    and written directly into the preallocated result matrix, which avoids
    retaining every returned block and then making a second full copy with
    ``numpy.vstack``.
    """
    graph = _validate_graph(graph)
    sources = _checks.index_array(np.atleast_1d(sources), "sources")
    if sources.size == 0:
        raise ValueError("no starting points given")
    if graph.shape[0] == 0:
        raise ValueError("graph has no nodes")
    if sources.max() >= graph.shape[0] or sources.min() < 0:
        raise ValueError("starting point index is outside the network")

    dtype = np.dtype(dtype)
    if dtype.kind != "f":
        raise ValueError("distance-matrix dtype must be a floating-point type")
    min_parallel_work = _checks.non_negative_int(min_parallel_work, "min_parallel_work")
    n_jobs = _checks.positive_int(n_jobs, "n_jobs")
    backend = _checks.backend(backend)

    n_nodes = graph.shape[0]
    work = int(sources.size) * int(n_nodes)
    parallel = n_jobs > 1 and sources.size > 1 and work >= min_parallel_work
    chunks = list(
        _source_chunks(sources, n_nodes, max_temp_mb, min_chunks=n_jobs if parallel else 1)
    )
    out = np.empty((sources.size, n_nodes), dtype=dtype)

    if not parallel:
        blocks = (_chunk(graph, chunk, dtype) for chunk in chunks)
    else:
        from joblib import Parallel, delayed

        # ``return_as='generator'`` preserves submission order, so each
        # completed block can be streamed straight into its output rows.
        blocks = Parallel(n_jobs=n_jobs, backend=backend, return_as="generator")(
            delayed(_chunk)(graph, chunk, dtype) for chunk in chunks
        )

    row = 0
    for chunk, block in zip(chunks, blocks, strict=True):
        out[row : row + len(chunk)] = block
        row += len(chunk)
    return out
