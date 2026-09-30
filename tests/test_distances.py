import numpy as np
import pytest

from net_center.distances import distance_matrix, estimate_distance_matrix_mb
from net_center.topology import adjacency_from_edges


def path_graph(n=20):
    u = np.arange(n - 1)
    w = np.arange(1, n)
    length = np.ones(n - 1)
    return adjacency_from_edges(u, w, length, n)


def test_chunking_does_not_change_distances():
    graph = path_graph(50)
    sources = np.array([0, 10, 20, 30, 49])
    large = distance_matrix(graph, sources, max_temp_mb=100)
    tiny = distance_matrix(graph, sources, max_temp_mb=0.001)
    assert tiny == pytest.approx(large)


def test_float32_storage_is_supported():
    graph = path_graph(20)
    D = distance_matrix(graph, [0, 19], dtype=np.float32, max_temp_mb=0.001)
    assert D.dtype == np.float32
    assert D[0, -1] == pytest.approx(19.0)


def test_memory_estimate():
    assert estimate_distance_matrix_mb(1024, 1024, np.float64) == pytest.approx(8.0)
    assert estimate_distance_matrix_mb(1024, 1024, np.float32) == pytest.approx(4.0)


def test_invalid_source_is_rejected():
    graph = path_graph(4)
    with pytest.raises(ValueError, match="outside"):
        distance_matrix(graph, [4])


def test_nonpositive_temp_budget_is_rejected():
    graph = path_graph(4)
    with pytest.raises(ValueError, match="positive"):
        distance_matrix(graph, [0], max_temp_mb=0)


def test_parallel_threaded_chunks_match_serial():
    graph = path_graph(60)
    sources = np.array([0, 10, 20, 30, 40, 59])
    serial = distance_matrix(graph, sources, n_jobs=1, max_temp_mb=0.001)
    parallel = distance_matrix(
        graph,
        sources,
        n_jobs=2,
        backend="threading",
        min_parallel_work=0,
        max_temp_mb=0.001,
    )
    assert parallel == pytest.approx(serial)


def test_invalid_worker_count_is_rejected():
    graph = path_graph(4)
    with pytest.raises(ValueError, match="positive integer"):
        distance_matrix(graph, [0], n_jobs=0)


def test_negative_matrix_dimensions_are_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        estimate_distance_matrix_mb(-1, 10)


def test_asymmetric_graph_is_rejected():
    from scipy.sparse import csr_matrix

    graph = csr_matrix(np.array([[0.0, 1.0], [2.0, 0.0]]))
    with pytest.raises(ValueError, match="symmetric"):
        distance_matrix(graph, [0])


def test_fractional_source_indices_are_rejected_instead_of_truncated():
    graph = path_graph(4)
    with pytest.raises(ValueError, match="integer node indices"):
        distance_matrix(graph, [0.5])


# ---------------------------------------------------------------------------
# Regressions added in the September 2026 review.
# ---------------------------------------------------------------------------


def test_parallel_request_splits_work_that_fits_one_memory_block(monkeypatch):
    """v0.1.0 ran serially whenever all sources fitted into one memory block,
    which at the default 256 MiB covers most real problems, so --jobs had no
    effect. Sources are now split into at least n_jobs blocks."""
    import net_center.distances as distances

    graph = path_graph(60)
    sources = np.arange(0, 60, 3)
    serial = distance_matrix(graph, sources, n_jobs=1)
    calls = []
    original = distances._chunk
    monkeypatch.setattr(distances, "_chunk", lambda *a: calls.append(1) or original(*a))
    parallel = distance_matrix(
        graph, sources, n_jobs=3, backend="threading", min_parallel_work=0
    )
    assert len(calls) == 3
    assert np.array_equal(parallel, serial)


def test_directed_dijkstra_equals_undirected_scipy_on_random_graphs():
    from scipy.sparse.csgraph import dijkstra

    rng = np.random.default_rng(4)
    for _ in range(5):
        n = 80
        u = rng.integers(0, n, 200)
        w = rng.integers(0, n, 200)
        keep = u != w
        graph = adjacency_from_edges(u[keep], w[keep], rng.uniform(1, 50, keep.sum()), n)
        sources = rng.choice(n, 10, replace=False)
        expected = dijkstra(graph, directed=False, indices=sources)
        assert np.array_equal(distance_matrix(graph, sources), expected)
