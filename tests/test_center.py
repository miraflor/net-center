import numpy as np
import pytest
from scipy.sparse.csgraph import dijkstra

from net_center.center import absolute_center, vertex_center, weighted_median
from net_center.topology import adjacency_from_edges as make_csr


def full_D(n, u, w, length):
    return dijkstra(make_csr(u, w, length, n), directed=False)


def test_interior_optimum_on_a_path():
    # A --3-- B --7-- C, demand at the two ends.
    # ecc(B) = 7, but the point 2 m along BC has ecc 5.
    u = np.array([0, 1])
    w = np.array([1, 2])
    length = np.array([3.0, 7.0])
    D = full_D(3, u, w, length)[[0, 2]]

    assert vertex_center(D).objective == pytest.approx(7.0)

    res = absolute_center(D, u, w, length, n_jobs=1)
    assert res.kind == "absolute_center"
    assert res.objective == pytest.approx(5.0)
    assert res.edge == 1
    assert res.t == pytest.approx(2.0)


def test_triangle_optimum_stays_at_a_vertex():
    # Equilateral triangle: every interior point is worse than every vertex.
    u = np.array([0, 1, 0])
    w = np.array([1, 2, 2])
    length = np.array([1.0, 1.0, 1.0])
    D = full_D(3, u, w, length)

    res = absolute_center(D, u, w, length, n_jobs=1)
    assert res.kind == "vertex_center"
    assert res.objective == pytest.approx(1.0)


def test_weights_move_the_median():
    # Star centred on 0, with a distant leaf 3.
    u = np.array([0, 0, 0])
    w = np.array([1, 2, 3])
    length = np.array([1.0, 1.0, 10.0])
    D = full_D(4, u, w, length)

    assert weighted_median(D).node == 0
    assert weighted_median(D, [1, 1, 1, 10]).node == 3


def random_network(rng, n=25):
    perm = rng.permutation(n)
    u = [perm[i] for i in range(1, n)]
    w = [perm[rng.integers(0, i)] for i in range(1, n)]
    for _ in range(n // 2):
        a, b = rng.integers(0, n, 2)
        if a != b:
            u.append(int(a))
            w.append(int(b))
    u = np.asarray(u, dtype=np.int64)
    w = np.asarray(w, dtype=np.int64)
    length = rng.uniform(1.0, 40.0, size=len(u))
    return u, w, length


def brute_force_radius(D, u, w, length, samples=4001):
    best = np.inf
    for e in range(len(u)):
        a = D[:, u[e]][:, None]
        b = D[:, w[e]][:, None]
        t = np.linspace(0.0, length[e], samples)[None, :]
        ecc = np.minimum(a + t, b + length[e] - t).max(axis=0)
        best = min(best, ecc.min())
    return best


@pytest.mark.parametrize("seed", range(12))
def test_sweep_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    u, w, length = random_network(rng)
    D = full_D(25, u, w, length)
    assert np.isfinite(D).all()

    res = absolute_center(D, u, w, length, n_jobs=1)
    expected = brute_force_radius(D, u, w, length)
    # Sampling can only overestimate; the sweep is exact.
    assert res.objective <= expected + 1e-9
    assert res.objective == pytest.approx(expected, abs=length.max() / 1000)


def test_parallel_matches_serial():
    rng = np.random.default_rng(7)
    u, w, length = random_network(rng, n=60)
    D = full_D(60, u, w, length)
    serial = absolute_center(D, u, w, length, n_jobs=1)
    parallel = absolute_center(D, u, w, length, n_jobs=4)
    assert parallel.objective == pytest.approx(serial.objective)


def test_chunking_does_not_change_the_answer():
    rng = np.random.default_rng(3)
    u, w, length = random_network(rng, n=40)
    D = full_D(40, u, w, length)
    big = absolute_center(D, u, w, length, n_jobs=1, max_cells=10**7)
    small = absolute_center(D, u, w, length, n_jobs=1, max_cells=50)
    assert big.objective == pytest.approx(small.objective)


def test_center_never_worse_than_vertex_center():
    rng = np.random.default_rng(11)
    for _ in range(20):
        u, w, length = random_network(rng, n=30)
        D = full_D(30, u, w, length)
        assert absolute_center(D, u, w, length, n_jobs=1).objective <= (
            vertex_center(D).objective + 1e-9
        )


def test_parallel_edges_are_min_combined():
    # tocsr() would sum these into one 30 m edge; the min rule keeps 10 m.
    csr = make_csr(np.array([0, 0]), np.array([1, 1]), np.array([10.0, 20.0]), 2)
    assert csr[0, 1] == pytest.approx(10.0)
    assert csr[1, 0] == pytest.approx(10.0)


def test_random_networks_respect_the_triangle_inequality():
    rng = np.random.default_rng(8)
    u, w, length = random_network(rng)
    D = full_D(25, u, w, length)
    assert (D[u, w] <= length + 1e-9).all()


def test_weights_are_validated():
    u, w, length = np.array([0]), np.array([1]), np.array([5.0])
    D = full_D(2, u, w, length)
    with pytest.raises(ValueError, match="correspond one to one"):
        weighted_median(D, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="non-negative"):
        weighted_median(D, [1.0, -1.0])
    with pytest.raises(ValueError, match="NaN or infinity"):
        weighted_median(D, [1.0, np.nan])


def test_mismatched_edge_arrays_are_rejected():
    u, w, length = np.array([0]), np.array([1]), np.array([5.0])
    D = full_D(2, u, w, length)
    with pytest.raises(ValueError, match="same length"):
        absolute_center(D, u, w, np.array([5.0, 6.0]))


def test_single_demand_point_works():
    # k=1 exercises the -inf padding at both interval ends.
    u, w, length = np.array([0, 1]), np.array([1, 2]), np.array([3.0, 7.0])
    D = full_D(3, u, w, length)[[0]]
    res = absolute_center(D, u, w, length, n_jobs=1)
    assert res.objective == pytest.approx(0.0)


def test_zero_total_weights_are_rejected():
    u, w, length = np.array([0]), np.array([1]), np.array([5.0])
    D = full_D(2, u, w, length)
    with pytest.raises(ValueError, match="at least one"):
        weighted_median(D, [0.0, 0.0])


def test_nonfinite_distance_matrix_is_rejected():
    D = np.array([[0.0, np.inf]])
    with pytest.raises(ValueError, match="NaN or infinite"):
        vertex_center(D)


def test_nonpositive_edge_length_is_rejected():
    D = np.array([[0.0, 1.0]])
    with pytest.raises(ValueError, match="strictly positive"):
        absolute_center(D, [0], [1], [0.0])


def test_float32_center_is_stable_on_simple_path():
    u = np.array([0, 1])
    w = np.array([1, 2])
    length = np.array([3.0, 7.0])
    D64 = full_D(3, u, w, length)[[0, 2]]
    D32 = D64.astype(np.float32)
    r64 = absolute_center(D64, u, w, length, n_jobs=1)
    r32 = absolute_center(D32, u, w, length, n_jobs=1)
    assert r32.objective == pytest.approx(r64.objective, abs=1e-5)


def test_invalid_center_worker_count_is_rejected():
    u, w, length = np.array([0]), np.array([1]), np.array([5.0])
    D = full_D(2, u, w, length)
    with pytest.raises(ValueError, match="positive integer"):
        absolute_center(D, u, w, length, n_jobs=0)


def test_fractional_edge_indices_are_rejected_instead_of_truncated():
    D = np.array([[0.0, 1.0]])
    with pytest.raises(ValueError, match="integer node indices"):
        absolute_center(D, [0.5], [1.0], [1.0])


def test_fractional_topology_indices_are_rejected_instead_of_truncated():
    with pytest.raises(ValueError, match="integer node indices"):
        make_csr([0.5], [1.0], [1.0], 2)


# ---------------------------------------------------------------------------
# Regressions and properties added in the September 2026 review.
# ---------------------------------------------------------------------------


def _exact_edge_minima(D, u, w, length):
    from net_center.center import _sweep

    return _sweep(D, u, w, length, 10**6)[0]


@pytest.mark.parametrize("seed", range(10))
def test_both_pruning_bounds_are_below_the_exact_edge_minimum(seed):
    from net_center.center import _eccentricity_bound, _endpoint_bound

    rng = np.random.default_rng(100 + seed)
    u, w, length = random_network(rng, n=40)
    D = full_D(40, u, w, length)[rng.choice(40, size=12, replace=False)]
    exact = _exact_edge_minima(D, u, w, length)
    assert (_eccentricity_bound(D.max(axis=0), u, w, length) <= exact + 1e-9).all()
    assert (_endpoint_bound(D, u, w, 10**6) <= exact + 1e-9).all()


@pytest.mark.parametrize("seed", range(10))
def test_pruned_result_equals_sweeping_every_edge(seed):
    rng = np.random.default_rng(200 + seed)
    u, w, length = random_network(rng, n=40)
    D = full_D(40, u, w, length)[rng.choice(40, size=15, replace=False)]
    exact = min(_exact_edge_minima(D, u, w, length).min(), vertex_center(D).objective)
    assert absolute_center(D, u, w, length).objective == pytest.approx(exact, abs=1e-9)


def _cycle(n=24):
    u = np.arange(n)
    w = (u + 1) % n
    return u, w, np.full(n, 10.0)


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_parallel_sweep_branch_matches_serial(backend, monkeypatch):
    # On a cycle every point has the same eccentricity, so no edge can be
    # pruned; the v0.1.0 tests never reached the parallel branch.
    import net_center.center as center

    u, w, length = _cycle()
    D = full_D(len(u), u, w, length)
    serial = absolute_center(D, u, w, length, n_jobs=1)
    calls = []
    if backend == "threading":
        original = center._sweep
        monkeypatch.setattr(center, "_sweep", lambda *a: calls.append(1) or original(*a))
    parallel = absolute_center(
        D, u, w, length, n_jobs=2, backend=backend, min_parallel_work=0
    )
    assert parallel.objective == pytest.approx(serial.objective)
    assert parallel.kind == serial.kind
    if backend == "threading":
        assert len(calls) == 2


def test_float32_weighted_median_does_not_copy_the_matrix_to_float64():
    import tracemalloc

    rng = np.random.default_rng(5)
    # Larger than one float64 work block (DEFAULT_MAX_CELLS cells), as in use.
    D32 = rng.uniform(0, 5e4, size=(1500, 4000)).astype(np.float32)
    weights = rng.uniform(1, 100, size=1500)
    tracemalloc.start()
    res = weighted_median(D32, weights)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    # v0.1.0 evaluated ``weights @ D32``, which converts all of D32 to a
    # float64 copy: a peak of twice D32.nbytes. Blocked evaluation stays small.
    assert peak < 0.5 * D32.nbytes
    exact = weights @ D32.astype(np.float64)
    assert res.node == int(np.argmin(exact))
    assert res.objective == pytest.approx(exact.min(), rel=1e-12)


def test_float32_median_sum_is_accumulated_in_float64():
    D32 = np.full((200_000, 2), 0.1, dtype=np.float32)
    D32[:, 1] = 0.2
    exact = D32.astype(np.float64).sum(axis=0)
    res = weighted_median(D32)
    # A float32 running sum of 200,000 values is off by about 1 %.
    assert res.objective == pytest.approx(exact[0], rel=1e-9)


def test_distance_matrix_check_rejects_bad_values_without_masks():
    with pytest.raises(ValueError, match="NaN or infinite"):
        vertex_center(np.array([[0.0, np.nan]]))
    with pytest.raises(ValueError, match="negative"):
        vertex_center(np.array([[0.0, -1.0]]))
    with pytest.raises(ValueError, match="real numeric"):
        vertex_center(np.array([[True, False]]))
