"""Integration against the repository's unmodified graph and legacy solvers."""

from __future__ import annotations

import json

# Missing dependencies or a missing baseline are failures, not skipped tests.
import geopandas as gpd
import numpy as np
import pytest
import shapely
from scipy.sparse.csgraph import dijkstra

from net_center import graph
from net_center.absolute_groups import solve_absolute_groups
from net_center.adaptive import solve_adaptive
from net_center.adaptive_cli import main
from net_center.edge_snap import augment_network, snap_to_edges
from net_center.solve import solve as legacy_solve


def network():
    roads = gpd.GeoDataFrame(geometry=[shapely.LineString([(0, 0), (10, 0)])],
                             crs='EPSG:32651')
    return roads, graph.build_network(roads, keep_largest_component=False)


def test_augmented_network_with_legacy_exact_absolute_center():
    _, base = network()
    aug = augment_network(base, snap_to_edges(base, [[2, 0], [8, 0]]))
    np.testing.assert_allclose(dijkstra(aug.network.csr)[:2, :2], dijkstra(base.csr))
    old = legacy_solve(aug.network, demand_nodes=aug.demand_nodes, weights=[1, 2])
    new = solve_adaptive(aug.network.csr, aug.demand_nodes, weights=[1, 2])
    assert old['median'].objective == pytest.approx(new.network_cost)
    assert old['absolute_center'].objective == pytest.approx(3)
    assert old['absolute_center'].xy == pytest.approx((5, 0))


def test_cli_grouped_and_failed_output(tmp_path):
    roads, _ = network()
    road_path = tmp_path / 'roads.gpkg'
    demand_path = tmp_path / 'demand.gpkg'
    roads.to_file(road_path, driver='GPKG')
    gpd.GeoDataFrame({'group': ['a', 'a', 'b'], 'weight': [1., 2., 5.]},
                     geometry=[shapely.Point(2, 1), shapely.Point(8, 1),
                               shapely.Point(4, 1)], crs=roads.crs).to_file(
        demand_path, driver='GPKG'
    )
    out = tmp_path / 'result.json'
    args = [str(road_path), '--demand', str(demand_path), '--group-by', 'group',
            '--weight-field', 'weight', '--out', str(out)]
    assert main([*args, '--max-snap', '0.1']) == 1
    assert not out.exists()
    assert main([*args, '--max-snap', '2']) == 0
    result = json.loads(out.read_text())
    assert result['attached_weight'] == 8
    assert result['groups'][0]['center_xy'][0] == pytest.approx([8, 0])
    assert result['groups'][1]['full_input_coverage'] == 1
    assert main(args) == 1


def test_absolute_connector_and_components():
    roads = gpd.GeoDataFrame(geometry=[
        shapely.LineString([(0, 0), (10, 0)]),
        shapely.LineString([(20, 0), (30, 0)]),
    ], crs='EPSG:32651')
    net = graph.build_network(roads, keep_largest_component=False)
    snaps = snap_to_edges(net, [[0, 0], [0, 6], [10, 0], [20, 0], [30, 0]])
    aug = augment_network(net, snaps)
    result = solve_absolute_groups(
        aug.network, aug.demand_nodes, ['a'] * 5, offsets=snaps.distance,
        include_offsets=True, component_policy='per_component',
    )['a'][1]
    assert result['diagnostics']['selected_k'] == 2
    assert result['objective'] == pytest.approx(8)
    assert sorted(tuple(site['xy']) for site in result['center_locations']) == [
        (2., 0.), (25., 0.)
    ]
    assert sum(site['assigned_count'] for site in result['center_locations']) == 5
    with pytest.raises(ValueError, match='one site'):
        solve_absolute_groups(aug.network, aug.demand_nodes, ['a'] * 5)


def test_cli_absolute_and_restricted(tmp_path):
    roads, _ = network()
    road_path = tmp_path / 'roads.gpkg'
    demand_path = tmp_path / 'demand.gpkg'
    candidate_path = tmp_path / 'candidates.gpkg'
    roads.to_file(road_path, driver='GPKG')
    gpd.GeoDataFrame(geometry=[shapely.Point(0, 0), shapely.Point(10, 0)],
                     crs=roads.crs).to_file(demand_path, driver='GPKG')
    gpd.GeoDataFrame(geometry=[shapely.Point(3, 0)], crs=roads.crs).to_file(
        candidate_path, driver='GPKG'
    )
    args = [str(road_path), '--demand', str(demand_path)]
    out = tmp_path / 'absolute.json'
    assert main([*args, '--objective', 'absolute_center', '--out', str(out)]) == 0
    group = json.loads(out.read_text())['groups'][0]
    assert group['center_locations'][0]['node'] is None
    assert group['center_xy'][0] == pytest.approx([5, 0])
    out = tmp_path / 'restricted.json'
    assert main([*args, '--candidate-domain', 'restricted', '--candidate-points',
                 str(candidate_path), '--out', str(out)]) == 0
    result = json.loads(out.read_text())
    assert result['groups'][0]['center_xy'][0] == pytest.approx([3, 0])
    assert result['candidate_audit'][0]['distance'] == 0


def test_cli_group_candidate_independence_and_absolute_option_errors(tmp_path):
    roads, _ = network()
    road_path = tmp_path / 'roads.gpkg'
    demand_path = tmp_path / 'demand.gpkg'
    roads.to_file(road_path, driver='GPKG')
    gpd.GeoDataFrame({'group': ['a', 'a', 'b']}, geometry=[
        shapely.Point(0, 0), shapely.Point(10, 0), shapely.Point(5, 0)
    ], crs=roads.crs).to_file(demand_path, driver='GPKG')
    args = [str(road_path), '--demand', str(demand_path), '--group-by', 'group']
    out = tmp_path / 'vertex.json'
    assert main([*args, '--objective', 'vertex_center', '--out', str(out)]) == 0
    groups = json.loads(out.read_text())['groups']
    assert groups[0]['objective'] == 10
    assert groups[1]['objective'] == 0
    invalid = tmp_path / 'invalid.json'
    for flags in (['--no-prune'], ['--gap-replicates', '10'], ['--penalty', '1'],
                  ['--component-policy', 'allocate'], ['--algorithm', 'heuristic']):
        assert main([*args, '--objective', 'absolute_center', '--out', str(invalid),
                     *flags]) == 1
        assert not invalid.exists()


def test_cli_rejects_options_without_effect_and_records_values_used(tmp_path):
    roads, _ = network()
    road_path = tmp_path / 'roads.gpkg'
    demand_path = tmp_path / 'demand.gpkg'
    roads.to_file(road_path, driver='GPKG')
    gpd.GeoDataFrame(geometry=[shapely.Point(1, 0), shapely.Point(9, 0)],
                     crs=roads.crs).to_file(demand_path, driver='GPKG')
    args = [str(road_path), '--demand', str(demand_path)]
    invalid = tmp_path / 'invalid.json'
    for flags in (['--penalty', '1'], ['--min-members', '2', '--selection', 'balanced'],
                  ['--seed', '0'], ['--gap-replicates', '10'], ['--algorithm', 'heuristic'],
                  ['--k', '2', '--max-centers', '2', '--max-combinations', '10'],
                  ['--objective', 'vertex_center', '--no-prune']):
        assert main([*args, '--out', str(invalid), *flags]) == 1
        assert not invalid.exists()
    out = tmp_path / 'two.json'
    assert main([*args, '--k', '2', '--max-centers', '2', '--out', str(out)]) == 0
    result = json.loads(out.read_text())
    assert result['schema'] == 'net-center-adaptive/5'
    for name in ('penalty', 'algorithm', 'swap_passes', 'max_combinations', 'seed',
                 'gap_replicates', 'min_members'):
        assert result['settings'][name] is None
    diagnostics = result['groups'][0]['diagnostics']
    assert diagnostics['configured_options'] == {
        'prune': True, 'algorithm': 'heuristic', 'swap_passes': 3
    }
    execution = next(iter(diagnostics['execution_by_component'].values()))
    assert execution['solver'] == 'heuristic_curve'
    assert execution['median_pruning_enabled'] is None


def fragment_fixture(tmp_path):
    """5 x 5 grid of 100 m blocks plus a disconnected 30 m road 3 m from one point."""
    lines = []
    for i in range(6):
        lines.append(shapely.LineString([(100 * j, 100 * i) for j in range(6)]))
        lines.append(shapely.LineString([(100 * i, 100 * j) for j in range(6)]))
    lines.append(shapely.LineString([(235, 250), (265, 250)]))
    road_path = tmp_path / 'roads.gpkg'
    demand_path = tmp_path / 'demand.gpkg'
    gpd.GeoDataFrame(geometry=lines, crs='EPSG:32651').to_file(road_path, driver='GPKG')
    gpd.GeoDataFrame({'w': [1., 2., 3., 4.]}, geometry=[
        shapely.Point(20, 5), shapely.Point(250, 253), shapely.Point(420, 480),
        shapely.Point(90, 330),
    ], crs='EPSG:32651').to_file(demand_path, driver='GPKG')
    return road_path, demand_path


def test_keep_largest_component_matches_legacy_cli(tmp_path):
    from net_center.cli import main as legacy_main

    road_path, demand_path = fragment_fixture(tmp_path)
    args = [str(road_path), '--demand', str(demand_path), '--weight-field', 'w']
    out = tmp_path / 'default.json'
    # Default behavior is unchanged: the point next to the fragment attaches to it.
    assert main([*args, '--out', str(out)]) == 1
    assert not out.exists()
    legacy_out = tmp_path / 'legacy.gpkg'
    with pytest.warns(UserWarning, match='disconnected'):
        assert legacy_main([*args, '--skip-absolute', '--quiet',
                            '--out', str(legacy_out)]) == 0
    legacy = gpd.read_file(legacy_out).set_index('kind')
    out = tmp_path / 'node.json'
    with pytest.warns(UserWarning, match='disconnected'):
        assert main([*args, '--snap-mode', 'node', '--keep-largest-component',
                     '--out', str(out)]) == 0
    result = json.loads(out.read_text())
    group = result['groups'][0]
    assert result['attached_rows'] == 4 and group['coverage'] == 1
    assert group['centers'] == [int(legacy.loc['median', 'node'])]
    assert group['network_cost'] == pytest.approx(legacy.loc['median', 'objective'])
    out = tmp_path / 'edge.json'
    with pytest.warns(UserWarning, match='disconnected'):
        assert main([*args, '--keep-largest-component', '--out', str(out)]) == 0
    result = json.loads(out.read_text())
    assert result['groups'][0]['coverage'] == 1
    assert [row['status'] for row in result['snap_audit']] == ['assigned'] * 4
    # The point at (250, 253) now attaches to the grid road at y=300, 47 m away.
    assert result['snap_audit'][1]['distance'] == pytest.approx(47)
    out = tmp_path / 'absolute.json'
    with pytest.warns(UserWarning, match='disconnected'):
        assert main([*args, '--keep-largest-component', '--objective', 'absolute_center',
                     '--out', str(out)]) == 0
    assert json.loads(out.read_text())['groups'][0]['coverage'] == 1
