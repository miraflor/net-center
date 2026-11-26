"""Independent small-instance, numerical and geometry regression oracles."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from net_center.adaptive import RoutingContext, solve_adaptive, solve_groups
from net_center.adaptive_cli import _publish
from net_center.edge_snap import _adjacency, augment_network, snap_to_edges


def line(n=10):
    a = np.zeros((n, n))
    j = np.arange(n-1)
    a[j, j+1] = a[j+1, j] = 1
    return csr_matrix(a)


def disconnected():
    a = np.zeros((7, 7))
    a[:4, :4] = line(4).toarray()
    a[4:, 4:] = line(3).toarray()
    return csr_matrix(a)


def test_median_streaming_random_oracle():
    rng = np.random.default_rng(102)
    for n in (5, 12, 25):
        a = line(n).toarray()
        for _ in range(n):
            i, j = rng.choice(n, size=2, replace=False)
            a[i, j] = a[j, i] = rng.uniform(.1, 10)
        g = csr_matrix(a)
        q = rng.integers(n, size=14)
        w = rng.integers(0, 9, size=14)
        truth = w @ dijkstra(g, indices=q, directed=True)
        for prune in (True, False):
            r = solve_adaptive(g, q, weights=w, prune=prune, max_temp_mb=.0001)
            assert r.network_cost == pytest.approx(truth.min())
            assert truth[r.centers[0]] == pytest.approx(truth.min())
            assert r.site_weight.sum() == w.sum()
            assert r.site_count.sum() == len(q)


def test_exact_candidate_domains_distinct():
    g = csr_matrix([[0, 2, 2, 2], [2, 0, 0, 0], [2, 0, 0, 0], [2, 0, 0, 0]])
    q = [1, 2, 3]
    a = solve_adaptive(g, q)
    b = solve_adaptive(g, q, candidate_domain='demand')
    c = solve_adaptive(g, q, candidate_domain='restricted', candidate_nodes=[2, 3])
    assert a.centers.tolist() == [0]
    assert a.network_cost == 6
    assert b.network_cost == c.network_cost == 8
    assert b.centers.tolist() == [1]
    assert c.centers.tolist() == [2]


def test_connector_minimax_changes_optimum_and_duplicates():
    g = line(11)
    q = [0, 0, 10]
    r = solve_adaptive(g, q, objective='vertex_center', offsets=[0, 6, 0],
                       include_offsets=True)
    assert r.centers.tolist() == [2]
    assert r.objective == 8
    assert solve_adaptive(g, q, objective='vertex_center').centers.tolist() == [5]
    m = solve_adaptive(g, q, offsets=[0, 6, 0], include_offsets=True)
    assert m.network_cost == 10 and m.objective == 16


def test_components_and_mass():
    g = disconnected()
    with pytest.raises(ValueError, match='disconnected'):
        solve_adaptive(g, [0, 3, 4, 6])
    with pytest.raises(ValueError, match='budget'):
        solve_adaptive(g, [0, 3, 4, 6], component_policy='allocate')
    r = solve_adaptive(g, [0, 3, 4, 6], weights=[2, 3, 5, 7],
                       component_policy='per_component')
    assert r.centers.tolist() == [3, 6]
    assert r.site_weight.tolist() == [5, 12]
    r = solve_adaptive(g, [0, 3, 4, 6], weights=[2, 3, 5, 7],
                       component_policy='largest')
    assert r.coverage == pytest.approx(5/17)
    assert r.assignment.tolist() == [0, 0, -1, -1]
    assert np.isinf(r.network_distance[2:]).all()


def test_unused_components_do_not_invalidate_single_site():
    assert solve_adaptive(disconnected(), [4, 6]).network_cost == 2


def test_fixed_and_dp_small_oracle():
    g = disconnected()
    q = np.array([0, 3, 4, 6])
    w = np.array([1., 3., 2., 1.])
    result = solve_adaptive(g, q, weights=w, k=3, component_policy='allocate',
                            candidate_domain='demand')
    D = dijkstra(g, indices=q, directed=True)[:, q]
    truth = min(w @ np.min(D[:, choice], axis=1)
                for choice in combinations(range(4), 3))
    assert result.network_cost == truth
    assert result.guarantee == 'heuristic_site_curves'
    assert result.site_weight.sum() == w.sum()


def test_balanced_selection_and_budget():
    g = line(11)
    r = solve_adaptive(g, [0, 5, 10], selection='balanced', k=None,
                       candidate_domain='demand', max_centers=3)
    base = r.curve[0].cost
    selected = min(r.curve, key=lambda p: (p.cost/base+(p.k-1)/2, p.k))
    assert len(r.centers) == selected.k
    r = solve_adaptive(g, [0, 5, 10], selection='access_budget', k=None,
                       candidate_domain='demand', max_centers=3, access_budget=0)
    assert len(r.centers) == 3 and r.mean_access == 0
    with pytest.raises(ValueError, match='budget unmet'):
        solve_adaptive(g, [0, 5, 10], offsets=[1, 1, 1], k=None,
                       selection='access_budget', access_budget=0)


def test_gap_deterministic_and_degenerate():
    q = [0, 0, 1, 2, 15, 16, 17, 17]
    options = dict(k=None, selection='gap', candidate_domain='demand',
                   gap_replicates=4, seed=12, max_centers=4)
    r = solve_adaptive(line(18), q, **options)
    s = solve_adaptive(line(18), q, **options)
    assert r.centers.tolist() == s.centers.tolist()
    assert r.diagnostics == s.diagnostics
    degenerate = solve_adaptive(line(18), [0, 0], **options)
    assert degenerate.centers.tolist() == [0]


def test_groups_preserve_row_ids():
    r = solve_groups(line(), [0, 8, 1, 9], ['a', 'b', 'a', 'b'])
    assert r['a'][0].tolist() == [0, 2]
    assert r['b'][0].tolist() == [1, 3]
    assert r['a'][1].centers.tolist() == [0]
    assert r['b'][1].centers.tolist() == [8]


@pytest.mark.parametrize('kwargs', [
    {'weights': [0, 0]}, {'weights': [-1, 1]}, {'weights': [np.nan, 1]},
    {'k': True}, {'k': 0}, {'candidate_domain': 'restricted'},
    {'candidate_nodes': [1]}, {'selection': 'gap', 'k': None},
    {'selection': 'balanced'}, {'access_budget': 1}, {'offsets': [-1, 0]},
])
def test_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        solve_adaptive(line(), [0, 9], **kwargs)


def test_memory_gate_and_graph_validation():
    with pytest.raises(MemoryError):
        solve_adaptive(line(), [0, 9], k=2, max_matrix_mb=.00001)
    with pytest.raises(ValueError, match='symmetric'):
        RoutingContext(csr_matrix([[0., 1.], [0., 0.]]))
    with pytest.raises(ValueError):
        RoutingContext(csr_matrix([[0., -1.], [-1., 0.]]))


@dataclass
class SyntheticNetwork:
    node_xy: np.ndarray
    edge_u: np.ndarray
    edge_w: np.ndarray
    edge_len: np.ndarray
    edge_geom: np.ndarray
    csr: csr_matrix
    crs: str = 'EPSG:32651'


def make_network(lines, xy, u, v):
    shapely = pytest.importorskip('shapely')
    geometry = np.array([shapely.LineString(x) for x in lines], dtype=object)
    lengths = shapely.length(geometry)
    return SyntheticNetwork(np.array(xy, dtype=float), np.array(u), np.array(v), lengths,
                            geometry,
                            _adjacency(np.array(u), np.array(v), lengths, len(xy)))


def test_edge_snap_split_curved_and_duplicate():
    net = make_network([[(0, 0), (5, 0), (5, 5)]], [(0, 0), (5, 5)], [0], [1])
    snap = snap_to_edges(net, [[2, 1], [2, 1], [5, 3], [100, 100]], max_distance=3)
    aug = augment_network(net, snap)
    assert aug.demand_nodes.tolist() == [2, 2, 3, -1]
    assert aug.network.edge_len.tolist() == [2, 6, 2]
    np.testing.assert_allclose(dijkstra(aug.network.csr)[:2, :2], dijkstra(net.csr))
    assert len(net.node_xy) == 2
    assert aug.original_edge.tolist() == [0, 0, 0]
    assert aug.original_start.tolist() == [0, 2, 8]


def test_crossings_and_parallel_edges():
    net = make_network([[(-2, 0), (2, 0)], [(0, -2), (0, 2)]],
                       [(-2, 0), (2, 0), (0, -2), (0, 2)], [0, 2], [1, 3])
    aug = augment_network(net, snap_to_edges(net, [[0, 0], [0, .5]]))
    assert aug.demand_nodes.tolist() == [4, 5]
    assert np.isinf(dijkstra(aug.network.csr)[4, 5])
    net = make_network([[(0, 0), (2, 0)], [(0, 0), (1, 1), (2, 0)]],
                       [(0, 0), (2, 0)], [0, 0], [1, 1])
    aug = augment_network(net, snap_to_edges(net, [[1, 1]]))
    assert dijkstra(aug.network.csr)[0, 1] == 2


def test_publication_no_clobber(tmp_path):
    path = tmp_path / 'out.json'
    _publish(path, {'missing': np.inf, 'other': np.nan})
    assert 'null' in path.read_text()
    first = path.read_bytes()
    with pytest.raises(FileExistsError):
        _publish(path, {'other': 5})
    assert path.read_bytes() == first
    assert list(tmp_path.iterdir()) == [path]


def test_exact_curves_against_exhaustive_oracle():
    rng = np.random.default_rng(71)
    for _ in range(10):
        g = line(8)
        q = rng.integers(8, size=7)
        weights = rng.integers(1, 8, size=len(q))
        D = dijkstra(g, indices=q, directed=True)
        result = solve_adaptive(g, q, weights=weights, k=3, algorithm='exact')
        for point in result.curve:
            expected = min(weights @ D[:, subset].min(axis=1)
                           for subset in combinations(range(8), point.k))
            assert point.cost == pytest.approx(expected)
        assert result.guarantee == 'exact_evaluated_site_curves'
    with pytest.raises(ValueError, match='max_combinations'):
        solve_adaptive(line(20), [0, 19], k=3, algorithm='exact', max_combinations=50)


def test_penalized_selection_and_exact_disconnected_oracle():
    g = disconnected()
    q = np.array([0, 3, 4, 6])
    weights = np.array([1., 3., 2., 1.])
    result = solve_adaptive(g, q, weights=weights, k=None, selection='penalized',
                            facility_cost=2., algorithm='exact', max_centers=4,
                            component_policy='allocate')
    D = dijkstra(g, indices=q, directed=True)
    oracle = min((weights @ D[:, subset].min(axis=1) + 2*k, k)
                 for k in range(2, 5) for subset in combinations(range(7), k))
    assert result.network_cost + 2*len(result.centers) == oracle[0]
    assert len(result.centers) == oracle[1]


def test_explicit_gap_reference_and_degenerate_audit():
    q = [0, 1, 2, 7, 8, 9]
    r = solve_adaptive(line(), q, k=None, selection='gap', candidate_domain='demand',
                       reference_nodes=[0, 9], gap_replicates=3)
    audit = r.diagnostics['gap_by_component']['0']
    assert audit['outcome'] == 'insufficient_reference_support'
    assert audit['reference_pool_size'] == 2
    assert len(r.centers) == 1
    r = solve_adaptive(line(), q, k=None, selection='gap', candidate_domain='demand',
                       reference_nodes=[0, 2, 5, 7, 9], gap_replicates=3)
    assert r.diagnostics['gap_by_component']['0']['reference_pool_size'] == 5


def test_physical_largest_policy_distinct_from_csr():
    g = disconnected()
    physical = ([0, 1, 2, 4, 5, 4], [1, 2, 3, 5, 6, 5], [1, 1, 1, 1, 1, 9])
    context = RoutingContext(g, physical_edges=physical)
    r = solve_adaptive(context.graph, [0, 4], context=context, component_policy='largest')
    assert r.assignment.tolist() == [-1, 0]
    assert r.diagnostics['largest_basis'] == 'physical_length'


def test_augmentation_rejects_inconsistent_original_routing_graph():
    net = make_network([[(0, 0), (2, 0)]], [(0, 0), (2, 0)], [0], [1])
    net.csr = csr_matrix([[0, 3], [3, 0]])
    with pytest.raises(ValueError, match='disagree'):
        snap_to_edges(net, [[1, 0]])


@pytest.mark.parametrize('kwargs', [
    {'algorithm': 'unknown'}, {'max_combinations': 0}, {'facility_cost': 1},
    {'selection': 'penalized', 'k': None}, {'reference_nodes': [0, 1]},
    {'selection': 'gap', 'candidate_domain': 'demand', 'k': None, 'algorithm': 'exact'},
])
def test_new_options_reject_invalid_combinations(kwargs):
    with pytest.raises(ValueError):
        solve_adaptive(line(), [0, 9], **kwargs)


def test_absolute_wrapper_with_independent_line_oracle(monkeypatch):
    """Exercise wrapper plumbing locally; real solver checked by integration suite."""
    import sys
    from types import ModuleType, SimpleNamespace

    from net_center.absolute_groups import solve_absolute_groups

    def oracle(D, u, v, lengths):
        # Fixture consists of single-edge components; solve its upper envelope
        # by enumerating pairwise intersections of rising/falling affine arms.
        assert len(u) == 1
        L = lengths[0]
        a, b = D[:, u[0]], D[:, v[0]]
        trials = [0., L]
        trials.extend(np.clip((y + L - x) / 2, 0, L) for x in a for y in b)
        radius, t = min((np.minimum(a+t, b+L-t).max(), t) for t in trials)
        return SimpleNamespace(node=None, edge=0, t=t, objective=radius)

    fake = ModuleType('net_center.center')
    fake.absolute_center = oracle
    monkeypatch.setitem(sys.modules, 'net_center.center', fake)
    net = make_network([[(0, 0), (10, 0)], [(20, 0), (30, 0)]],
                       [(0, 0), (10, 0), (20, 0), (30, 0)], [0, 2], [1, 3])
    net.interpolate = lambda edge, t: (net.node_xy[net.edge_u[edge], 0]+t, 0.)
    result = solve_absolute_groups(net, [0, 0, 1, 2, 3], ['a']*5,
                                   offsets=[0, 6, 0, 0, 0], include_offsets=True,
                                   component_policy='per_component')['a'][1]
    assert result['objective'] == 8
    assert result['center_xy'] == [(2., 0.), (25., 0.)]
    assert result['assignment'].tolist() == [0, 0, 0, 1, 1]
    assert result['access_distance'].tolist() == [2, 8, 8, 5, 5]


def test_weighted_percentiles_and_budget_verification():
    result = solve_adaptive(line(11), [0, 5, 10], weights=[90, 10, 0],
                            offsets=[1, 2, 100])
    assert result.diagnostics['weighted_access_percentiles'] == {
        '50': 1., '90': 1., '95': 7., '99': 7.
    }
    assert result.max_access == 110
    assert result.diagnostics['modeled_row_fraction'] == 1
    result = solve_adaptive(line(3), [0, 2], weights=[1, 1], offsets=[.1, .2],
                            k=None, selection='access_budget', access_budget=.15,
                            candidate_domain='demand')
    assert len(result.centers) == 2
    assert result.mean_access == pytest.approx(.15)


def test_cli_with_fixture_graph_and_real_gis_inputs(tmp_path, monkeypatch):
    """Check CLI IO, grouping, candidate augmentation and coverage independently."""
    import json
    import sys
    from types import ModuleType

    from net_center.adaptive_cli import main

    gpd = pytest.importorskip('geopandas')
    shapely = pytest.importorskip('shapely')
    base = make_network([[(0, 0), (10, 0)], [(20, 0), (25, 0)]],
                        [(0, 0), (10, 0), (20, 0), (25, 0)], [0, 2], [1, 3])
    base.n_nodes = len(base.node_xy)
    fake = ModuleType('net_center.graph')
    fake.build_network = lambda *args, **kwargs: base
    fake.snap_points = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, 'net_center.graph', fake)
    demand = tmp_path / 'demand.gpkg'
    gpd.GeoDataFrame({'w': [1, 2, 4, 8]}, geometry=[
        shapely.Point(0, 0), shapely.Point(10, 0), shapely.Point(20, 0),
        shapely.Point(100, 100),
    ], crs=base.crs).to_file(demand, driver='GPKG')
    candidates = tmp_path / 'candidates.gpkg'
    gpd.GeoDataFrame(geometry=[shapely.Point(3, 0)], crs=base.crs).to_file(
        candidates, driver='GPKG'
    )
    path = tmp_path / 'result.json'
    args = ['fixture', '--demand', str(demand), '--out', str(path), '--max-snap', '1',
            '--component-policy', 'largest', '--weight-field', 'w',
            '--candidate-domain', 'restricted', '--candidate-points', str(candidates)]
    assert main(args) == 1
    assert not path.exists()
    assert main([*args, '--on-unsnapped', 'exclude']) == 0
    result = json.loads(path.read_text())
    assert result['schema'] == 'net-center-adaptive/5'
    assert result['groups'][0]['center_xy'] == [[3., 0.]]
    assert result['groups'][0]['full_input_coverage'] == pytest.approx(3/15)
    assert result['groups'][0]['full_input_row_coverage'] == .5
    assert [row['status'] for row in result['snap_audit']] == [
        'assigned', 'assigned', 'excluded_component', 'unsnapped'
    ]
    assert result['candidate_audit'][0]['snapped_xy'] == [3., 0.]


def test_reversed_edge_orientation_rejected():
    net = make_network([[(10, 0), (0, 0)]], [(0, 0), (10, 0)], [0], [1])
    with pytest.raises(ValueError, match='orientation'):
        snap_to_edges(net, [[0, 0], [2, 0], [10, 0]])


def test_endpoint_tolerance_is_explicit_and_respected():
    net = make_network([[(0.05, 0), (10, 0)]], [(0, 0), (10, 0)], [0], [1])
    with pytest.raises(ValueError, match='orientation'):
        snap_to_edges(net, [[2, 0]], endpoint_tolerance=0.01)
    snaps = snap_to_edges(net, [[2, 0]], endpoint_tolerance=0.1)
    aug = augment_network(net, snaps, endpoint_tolerance=0.1)
    assert aug.network.node_xy[aug.demand_nodes[0]].tolist() == pytest.approx([2, 0])


def test_group_vertex_candidates_independent_of_other_attachments():
    base = make_network([[(0, 0), (10, 0)]], [(0, 0), (10, 0)], [0], [1])
    aug = augment_network(base, snap_to_edges(base, [[0, 0], [10, 0], [5, 0]]))
    solo = solve_adaptive(base.csr, [0, 1], objective='vertex_center')
    grouped = solve_groups(aug.network.csr, aug.demand_nodes, ['a', 'a', 'b'],
                           objective='vertex_center', base_vertex_count=2)
    assert grouped['a'][1].objective == solo.objective == 10
    assert grouped['a'][1].centers.tolist() == solo.centers.tolist()
    assert grouped['b'][1].centers.tolist() == [2]
    assert solve_adaptive(aug.network.csr, [0, 1], objective='vertex_center').objective == 5


@pytest.mark.parametrize('physical', [
    ([0, 1, 2, 4, 5, 4], [1, 2, 3, 5, 6, 4], [1, 1, 1, 1, 1, 999]),
    ([0, 1, 2, 4, 5, 0], [1, 2, 3, 5, 6, 2], [1, 1, 1, 1, 1, 999]),
    ([0, 1, 4, 5], [1, 2, 5, 6], [1, 1, 1, 1]),
    ([0, 1, 2, 4, 5], [1, 2, 3, 5, 6], [2, 1, 1, 1, 1]),
])
def test_physical_inventory_rejects_invalid_topology(physical):
    with pytest.raises(ValueError, match='physical edge'):
        RoutingContext(disconnected(), physical_edges=physical)


def test_absolute_wrapper_rejects_invalid_network_before_solving(monkeypatch):
    import sys
    from types import ModuleType

    from net_center.absolute_groups import solve_absolute_groups

    fake = ModuleType('net_center.center')

    def must_not_solve(*_args):
        pytest.fail('solver called with invalid physical edges')

    fake.absolute_center = must_not_solve
    monkeypatch.setitem(sys.modules, 'net_center.center', fake)
    net = make_network([[(10, 0), (0, 0)]], [(0, 0), (10, 0)], [0], [1])
    with pytest.raises(ValueError, match='orientation'):
        solve_absolute_groups(net, [0, 1], ['a', 'a'])
    net = make_network([[(0, 0), (10, 0)]], [(0, 0), (10, 0)], [0], [1])
    net.csr = csr_matrix([[0, 9], [9, 0]])
    with pytest.raises(ValueError, match='disagree'):
        solve_absolute_groups(net, [0, 1], ['a', 'a'])


def test_penalized_cost_and_connector_radius_reporting():
    r = solve_adaptive(line(11), [0, 10], k=None, selection='penalized',
                       facility_cost=3, algorithm='exact', max_centers=2,
                       offsets=[1, 2], include_offsets=True)
    assert r.diagnostics['penalized_selection_cost'] == 6
    assert r.network_cost == 0 and r.objective == 3
    r = solve_adaptive(line(11), [0, 10], objective='vertex_center',
                       offsets=[6, 0], include_offsets=True)
    assert r.curve[0].cost == r.objective == 8
    assert r.diagnostics['curve_cost'] == 'maximum_component_access_radius'


def test_transitive_endpoint_tolerance_matches_base_topology():
    net = make_network([[(0, 0), (0, 10)], [(0.08, 0), (10, 0)],
                        [(0.16, 0), (0, -10)]],
                       [(0, 0), (0, 10), (10, 0), (0, -10)],
                       [0, 0, 0], [1, 2, 3])
    snaps = snap_to_edges(net, [[5, 0]], endpoint_tolerance=.1)
    aug = augment_network(net, snaps, endpoint_tolerance=.1)
    np.testing.assert_allclose(dijkstra(aug.network.csr)[:4, :4], dijkstra(net.csr))
    with pytest.raises(ValueError, match='tolerance chain'):
        snap_to_edges(net, [[5, 0]], endpoint_tolerance=.05)


def test_stale_snap_metadata_rejected():
    from dataclasses import replace

    net = make_network([[(0, 0), (10, 0)]], [(0, 0), (10, 0)], [0], [1])
    snaps = snap_to_edges(net, [[2, 0]])
    with pytest.raises(ValueError, match='snap metadata'):
        augment_network(net, replace(snaps, xy=np.array([[8., 0.]])))


def test_original_largest_rank_pinned_after_augmentation():
    g = disconnected()
    ctx = RoutingContext(g, physical_edges=(
        [0, 1, 2, 4, 5, 4], [1, 2, 3, 5, 6, 5], [1, 1, 1, 1, 1, 1.0000000000001]
    ))
    assert ctx.largest == ctx.labels[4]
    ctx._pin_original_largest(0)
    assert ctx.largest == ctx.labels[0]
    assert ctx.largest_basis == 'physical_length_original_rank'
    with pytest.raises(ValueError, match='disagrees'):
        RoutingContext(g, physical_edges=(
            [0, 1, 2, 4, 5], [1, 2, 3, 5, 6], [1, 1, 1, 1, 1]
        ))._pin_original_largest(4)


def test_cli_group_independence_and_invalid_absolute_options(tmp_path, monkeypatch):
    import json
    import sys
    from types import ModuleType

    import geopandas as gpd
    import shapely

    from net_center.adaptive_cli import main

    base = make_network([[(0, 0), (10, 0)]], [(0, 0), (10, 0)], [0], [1])
    base.n_nodes = len(base.node_xy)
    fake = ModuleType('net_center.graph')
    fake.build_network = lambda *_args, **_kwargs: base
    fake.snap_points = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, 'net_center.graph', fake)
    path = tmp_path / 'demand.gpkg'
    gpd.GeoDataFrame({'g': ['a', 'a', 'b']}, geometry=[shapely.Point(0, 0),
        shapely.Point(10, 0), shapely.Point(5, 0)], crs=base.crs).to_file(path)
    args = ['fixture', '--demand', str(path), '--group-by', 'g']
    out = tmp_path / 'vertex.json'
    assert main([*args, '--objective', 'vertex_center', '--out', str(out)]) == 0
    assert json.loads(out.read_text())['groups'][0]['objective'] == 10
    invalid = tmp_path / 'invalid.json'
    for flag in (['--no-prune'], ['--gap-replicates', '10'], ['--penalty', '1'],
                 ['--component-policy', 'allocate'], ['--algorithm', 'heuristic'],
                 ['--swap-passes', '-1'], ['--max-combinations', '0']):
        assert main([*args, '--objective', 'absolute_center', '--out', str(invalid),
                     *flag]) == 1
        assert not invalid.exists()


def test_zero_weight_component_site_minimizes_access():
    g = disconnected()  # components {0, 1, 2, 3} and {4, 5, 6}
    r = solve_adaptive(g, [0, 6], weights=[1, 0], k=2, component_policy='allocate')
    assert r.centers.tolist() == [0, 6]
    assert r.max_access == 0
    assert r.diagnostics['zero_weight_components'] == [int(r.component[1])]
    # Connectors count: rows at 4 (connector 3) and 6 (connector 0). Network-only
    # minimax would choose 5 (radius 1); access minimax chooses 4 (radius 3).
    r = solve_adaptive(g, [0, 4, 6], weights=[1, 0, 0], offsets=[0, 3, 0], k=2,
                       component_policy='allocate')
    assert r.centers.tolist() == [0, 4]
    assert r.site_max_access[1] == 3
    assert r.network_cost == 0


def test_zero_weight_component_receives_exactly_one_site():
    g = disconnected()
    r = solve_adaptive(g, [0, 6], weights=[1, 0], k=3, component_policy='allocate')
    assert np.count_nonzero(np.isin(r.centers, [4, 5, 6])) == 1
    assert len(r.centers) == 3
    with pytest.raises(ValueError, match='zero total weight'):
        solve_adaptive(g, [0, 6], weights=[1, 0], k=3, component_policy='allocate',
                       candidate_domain='demand')


@pytest.mark.parametrize('kwargs', [
    {'penalty': 1.}, {'gap_replicates': 10}, {'seed': 0}, {'min_members': 1},
    {'coordinates': np.zeros((10, 2))}, {'algorithm': 'heuristic'},
    {'swap_passes': 3}, {'max_combinations': 10},
    {'k': 2, 'max_combinations': 10},
    {'k': 2, 'algorithm': 'exact', 'swap_passes': 1},
    {'objective': 'vertex_center', 'prune': False},
    {'objective': 'vertex_center', 'algorithm': 'exact'},
    {'selection': 'balanced', 'k': None, 'min_members': 2},
    {'selection': 'penalized', 'k': None, 'facility_cost': 1., 'penalty': 2.},
    {'component_policy': 'per_component', 'swap_passes': 2},
])
def test_options_without_effect_are_rejected(kwargs):
    with pytest.raises(ValueError, match='only applies to'):
        solve_adaptive(line(), [0, 9], **kwargs)


@pytest.mark.parametrize('kwargs, expected', [
    ({}, {'prune': True}),
    ({'prune': False}, {'prune': False}),
    ({'k': 2, 'swap_passes': 0},
     {'prune': True, 'algorithm': 'heuristic', 'swap_passes': 0}),
    ({'k': 2, 'algorithm': 'exact', 'max_combinations': 100},
     {'prune': True, 'algorithm': 'exact', 'max_combinations': 100}),
    ({'selection': 'balanced', 'k': None, 'penalty': 2.},
     {'prune': True, 'algorithm': 'heuristic', 'swap_passes': 3, 'penalty': 2.}),
    ({'objective': 'vertex_center'}, {}),
])
def test_configured_options_record_resolved_values(kwargs, expected):
    r = solve_adaptive(line(), [0, 9], **kwargs)
    assert r.diagnostics['configured_options'] == expected
    assert r.diagnostics['algorithm'] == expected.get('algorithm')


def test_gap_reference_label_states_the_support_used():
    q = [0, 0, 1, 2, 15, 16, 17, 17]
    options = dict(k=None, selection='gap', candidate_domain='demand',
                   gap_replicates=4, seed=12, max_centers=4)
    xy = np.column_stack((np.arange(18.), np.zeros(18)))
    labels = {
        'component_vertices': solve_adaptive(line(18), q, **options),
        'component_vertices_in_demand_bounding_box': solve_adaptive(
            line(18), q, coordinates=xy, **options),
        'explicit_nodes': solve_adaptive(line(18), q, reference_nodes=range(0, 18, 2),
                                         **options),
        'explicit_nodes_in_demand_bounding_box': solve_adaptive(
            line(18), q, reference_nodes=range(0, 18, 2), coordinates=xy, **options),
    }
    for label, result in labels.items():
        assert result.diagnostics['gap_reference'] == label
    assert solve_adaptive(line(), [0, 9]).diagnostics['gap_reference'] is None
    r = solve_adaptive(disconnected(), [0, 1, 2, 3, 6], weights=[1, 1, 1, 1, 0],
                       component_policy='allocate', **options)
    audit = r.diagnostics['gap_by_component']
    assert audit[str(int(r.component[4]))]['outcome'] == 'zero_weight_component'


@pytest.mark.parametrize('options, solver, pruning, algorithm', [
    ({'k': 2}, 'heuristic_curve', None, 'heuristic'),
    ({'k': 2, 'algorithm': 'exact'}, 'exact_curve', None, 'exact'),
    ({'selection': 'balanced', 'k': None, 'max_centers': 1, 'algorithm': 'exact'},
     'streaming_median', True, None),
    ({'prune': False}, 'streaming_median', False, None),
    ({'objective': 'vertex_center'}, 'vertex_minimax', None, None),
])
def test_execution_diagnostics_match_solver_path(options, solver, pruning, algorithm):
    result = solve_adaptive(line(), [0, 9], **options)
    diagnostics = result.diagnostics
    record = next(iter(diagnostics['execution_by_component'].values()))
    assert record['solver'] == solver
    assert record['median_pruning_enabled'] is pruning
    assert diagnostics['algorithm'] == algorithm
    assert record['evaluated_k'] == ([1, 2] if options.get('k') == 2 else [1])
    if options.get('algorithm') == 'exact':
        assert diagnostics['configured_options']['algorithm'] == 'exact'


def test_execution_diagnostics_distinguish_zero_weight_component():
    result = solve_adaptive(disconnected(), [0, 3, 6], weights=[1, 1, 0], k=3,
                            component_policy='allocate', algorithm='exact')
    records = result.diagnostics['execution_by_component']
    assert records[str(result.component[0])]['solver'] == 'exact_curve'
    zero = records[str(result.component[2])]
    assert zero == {'solver': 'zero_weight_minimax', 'evaluated_k': [1],
                    'median_pruning_enabled': None}


def test_gap_execution_distinguishes_reference_sampling_and_degeneracy():
    options = dict(k=None, selection='gap', candidate_domain='demand', gap_replicates=3)
    sampled = solve_adaptive(line(), [0, 1, 8, 9], **options)
    skipped = solve_adaptive(line(), [0, 9], **options)
    actual = sampled.diagnostics['execution_by_component']['0']
    omitted = skipped.diagnostics['execution_by_component']['0']
    assert actual['gap_reference_replicates'] == 3
    assert omitted['gap_reference_replicates'] == 0
    assert skipped.diagnostics['algorithm'] is None


def test_cli_all_public_arguments_have_help():
    from net_center.adaptive_cli import parser

    for action in parser()._actions:
        assert action.help and action.help.strip(), action.dest
