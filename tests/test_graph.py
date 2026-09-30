"""Regression tests for the GIS/topology boundary."""

import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components

gpd = pytest.importorskip("geopandas")
shapely_geom = pytest.importorskip("shapely.geometry")
LineString = shapely_geom.LineString

from net_center.graph import build_network, snap_points  # noqa: E402

UTM51N = "EPSG:32651"


def gdf(geoms, crs=UTM51N):
    return gpd.GeoDataFrame(geometry=list(geoms), crs=crs)


def test_crossings_are_not_connected_by_default():
    roads = gdf(
        [
            LineString([(0, 50), (100, 50)]),
            LineString([(50, 0), (50, 100)]),
        ]
    )
    net = build_network(roads, keep_largest_component=False)
    n_components, _ = connected_components(net.csr, directed=False)
    assert net.n_nodes == 4
    assert net.n_edges == 2
    assert n_components == 2


def test_planar_noding_is_explicit_and_splits_crossings():
    roads = gdf(
        [
            LineString([(0, 50), (100, 50)]),
            LineString([(50, 0), (50, 100)]),
        ]
    )
    with pytest.warns(UserWarning, match="planar noding"):
        net = build_network(roads, node=True)
    assert net.n_nodes == 5
    assert net.n_edges == 4
    assert net.edge_len.sum() == pytest.approx(200.0)


def test_shape_points_do_not_become_nodes():
    curve = LineString([(x, 10 * np.sin(x / 40)) for x in range(0, 400, 5)])
    net = build_network(gdf([curve]))
    assert net.n_edges == 1
    assert net.n_nodes == 2


def test_closed_loop_is_kept_not_deleted():
    ring_pts = [
        (1000 + 300 * np.cos(a), 300 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 33)
    ]
    expected = 1300.0 + LineString(ring_pts).length
    net = build_network(gdf([LineString([(0, 0), (1300, 0)]), LineString(ring_pts)]))
    assert net.edge_len.sum() == pytest.approx(expected, abs=0.01)
    assert (net.edge_u != net.edge_w).all()


def test_empty_geometry_does_not_shift_endpoints():
    net = build_network(
        gdf(
            [
                LineString([(0, 0), (100, 0)]),
                LineString(),
                LineString([(100, 0), (200, 0)]),
            ]
        )
    )
    assert net.edge_len == pytest.approx([100.0, 100.0])


def test_projected_metre_input_is_not_reprojected():
    net = build_network(gdf([LineString([(0, 0), (1300, 0)])]))
    assert net.crs.to_string() == UTM51N
    assert net.edge_len[0] == pytest.approx(1300.0)


def test_latlon_input_is_reprojected_to_metres():
    net = build_network(gdf([LineString([(121.0, 14.6), (121.01, 14.6)])], crs="EPSG:4326"))
    assert not net.crs.is_geographic
    assert 1000 < net.edge_len[0] < 1200


def test_explicit_geographic_target_crs_is_rejected():
    with pytest.raises(ValueError, match="projected"):
        build_network(gdf([LineString([(0, 0), (100, 0)])]), target_crs="EPSG:4326")


def test_explicit_non_metre_target_crs_is_rejected():
    with pytest.raises(ValueError, match="metres"):
        build_network(gdf([LineString([(0, 0), (100, 0)])]), target_crs="EPSG:2263")


def test_largest_component_is_chosen_by_road_length_not_node_count():
    # Dense but tiny component: 10 one-metre links.
    tiny = [LineString([(i, 0), (i + 1, 0)]) for i in range(10)]
    # Sparse but substantively larger component: one 100-metre road.
    long = [LineString([(1000, 0), (1100, 0)])]
    with pytest.warns(UserWarning, match="most road length"):
        net = build_network(gdf(tiny + long))
    assert net.n_nodes == 2
    assert net.n_edges == 1
    assert net.edge_len.sum() == pytest.approx(100.0)


def test_missing_crs_is_rejected():
    with pytest.raises(ValueError, match="coordinate reference system"):
        build_network(gdf([LineString([(0, 0), (1, 0)])], crs=None))


def test_invalid_snap_is_rejected():
    with pytest.raises(ValueError, match="snap"):
        build_network(gdf([LineString([(0, 0), (1, 0)])]), snap=0)


def test_snap_points_respects_max_dist():
    net = build_network(gdf([LineString([(0, 0), (100, 0)])]))
    idx, dist = snap_points(net, np.array([[0.0, 5.0], [0.0, 5000.0]]), max_dist=50.0)
    assert idx[0] >= 0 and idx[1] == -1
    assert dist[0] == pytest.approx(5.0)


def test_snap_points_rejects_nonfinite_coordinates():
    net = build_network(gdf([LineString([(0, 0), (100, 0)])]))
    with pytest.raises(ValueError, match="NaN or infinity"):
        snap_points(net, np.array([[np.nan, 5.0]]))


def test_end_to_end_solve_finds_known_midpoint():
    from net_center.solve import solve

    net = build_network(gdf([LineString([(0, 0), (3, 0)]), LineString([(3, 0), (10, 0)])]))
    ends, _ = snap_points(net, np.array([[0.0, 0.0], [10.0, 0.0]]))
    res = solve(net, demand_nodes=ends)
    assert res["vertex_center"].objective == pytest.approx(7.0)
    assert res["absolute_center"].objective == pytest.approx(5.0)
    assert res["absolute_center"].xy[0] == pytest.approx(5.0)


def test_shared_vertex_split_recovers_osm_style_junctions():
    """OSM stores a through-road as one way whose interior vertices ARE the
    junctions, with side streets ending on them. Endpoint-only matching misses
    most of those connections in this synthetic network."""
    lines, n = [], 8
    for i in range(n):
        lines.append(LineString([(x * 100, i * 100) for x in range(n)]))
    for j in range(1, n - 1):
        for i in range(n - 1):
            lines.append(LineString([(j * 100, i * 100), (j * 100, (i + 1) * 100)]))
    g = gdf(lines)
    total = sum(line.length for line in lines)

    kept = build_network(g).edge_len.sum()
    assert kept == pytest.approx(total)

    with pytest.warns(UserWarning, match="disconnected"):
        shattered = build_network(g, split_shared_vertices=False).edge_len.sum()
    assert shattered < 0.2 * total


def test_shared_vertex_split_does_not_weld_a_bridge():
    """An overpass shares no vertex with the road beneath it, so it must stay
    separate. This is what planar noding gets wrong."""
    from scipy.sparse.csgraph import connected_components

    bridge = gdf(
        [
            LineString([(0, 50), (100, 50)]),
            LineString([(50, 0), (50, 100)]),
        ]
    )
    net = build_network(bridge, keep_largest_component=False)
    assert connected_components(net.csr, directed=False)[0] == 2

    with pytest.warns(UserWarning, match="planar noding"):
        planar = build_network(bridge, node=True, keep_largest_component=False)
    assert connected_components(planar.csr, directed=False)[0] == 1


def test_shared_vertex_split_leaves_untouched_lines_alone():
    net = build_network(gdf([LineString([(0, 0), (50, 0), (100, 0)])]))
    assert net.n_edges == 1
    assert net.edge_len[0] == pytest.approx(100.0)


def test_shared_interior_vertices_are_split_on_both_lines():
    """Two continuing ways can share a real OSM junction at interior vertices."""
    roads = gdf(
        [
            LineString([(0, 50), (50, 50), (100, 50)]),
            LineString([(50, 0), (50, 50), (50, 100)]),
        ]
    )
    net = build_network(roads, keep_largest_component=False)
    n_components, _ = connected_components(net.csr, directed=False)
    assert n_components == 1
    assert net.n_nodes == 5
    assert net.n_edges == 4
    assert net.edge_len.sum() == pytest.approx(200.0)


def test_filtered_sliver_does_not_leave_an_orphan_node():
    roads = gdf(
        [
            LineString([(0, 0), (100, 0)]),
            LineString([(1000, 0), (1000.00000001, 0)]),
        ]
    )
    net = build_network(roads, min_length=1e-3, keep_largest_component=False)
    assert net.n_nodes == 2
    assert net.n_edges == 1
    assert net.csr.shape == (2, 2)


def test_tiny_snap_merges_only_identical_coordinates():
    # Replaces an integer-overflow test for the old snap grid. Locations are
    # now matched by distance, so a tiny snap is valid: identical coordinates
    # still join, while a 1e-9 m gap no longer does.
    joined = build_network(
        gdf(
            [LineString([(500000, 0), (500100, 0)]), LineString([(500100, 0), (500200, 0)])]
        ),
        snap=1e-20,
    )
    assert joined.n_nodes == 3
    apart = build_network(
        gdf(
            [
                LineString([(500000, 0), (500100, 0)]),
                LineString([(500100 + 1e-9, 0), (500200, 0)]),
            ]
        ),
        snap=1e-20,
        keep_largest_component=False,
    )
    assert apart.n_nodes == 4


# ---------------------------------------------------------------------------
# Regressions added in the September 2026 review. Each test below failed on
# the v0.1.0 code.
# ---------------------------------------------------------------------------


def _road_distance(net, p, q):
    from scipy.sparse.csgraph import dijkstra

    (i, j), _ = snap_points(net, np.array([p, q]))
    return dijkstra(net.csr, directed=False, indices=[i])[0, j]


@pytest.mark.parametrize("x", [500123.44, 500123.45, 500123.55])
def test_junction_is_not_split_by_floating_point_noise(x):
    """v0.1.0 rounded coordinates to a 0.1 m grid. A centimetre coordinate
    ending in 5 lies exactly on a cell boundary, so 1e-9 m of noise put the
    two copies of one junction into different cells and cut the road."""
    a = LineString([(x - 100, 1600000.0), (x, 1600000.0)])
    b = LineString([(x + 1e-9, 1600000.0), (x + 100, 1600000.0)])
    net = build_network(gdf([a, b]), keep_largest_component=False)
    assert net.n_nodes == 3
    assert connected_components(net.csr, directed=False)[0] == 1


def test_ring_whose_ends_share_a_location_is_split_not_deleted():
    """v0.1.0 tested ring closure by Euclidean distance but merged nodes by
    grid cell. Ends 0.113 m apart in one cell became one node, failed the
    closure test, and the 1.9 km ring was deleted as a self-loop. Now the ring
    test is "both ends are one location", the same rule as node identity."""
    access = LineString([(-500.0, -0.04), (-0.04, -0.04)])
    angles = np.linspace(0, 2 * np.pi, 41)[1:-1]
    for end in [(0.04, 0.04), (0.02, -0.04)]:  # gaps of 0.113 m and 0.06 m
        ring = LineString(
            [(-0.04, -0.04)]
            + [(300 * np.cos(a) - 300, 300 * np.sin(a)) for a in angles]
            + [end]
        )
        net = build_network(gdf([access, ring]), keep_largest_component=False)
        assert net.edge_len.sum() == pytest.approx(access.length + ring.length, abs=1e-6)
        assert (net.edge_u != net.edge_w).all()


def test_lollipop_line_is_split_where_it_touches_itself():
    """A way that ends on one of its own interior vertices (a cul-de-sac
    turning loop). v0.1.0 kept it as one 500 m edge, so the road distance from
    the access road to the loop junction was 500 m instead of 100 m."""
    access = LineString([(0, 0), (100, 0)])
    lolli = LineString([(100, 0), (200, 0), (300, 0), (300, 100), (200, 100), (200, 0)])
    for lines in ([access, lolli], [lolli]):  # also when it is the only line
        net = build_network(gdf(lines), keep_largest_component=False)
        assert _road_distance(net, (100, 0), (200, 0)) == pytest.approx(100.0)
        assert net.edge_len.sum() == pytest.approx(sum(g.length for g in lines))


def test_line_crossing_itself_at_an_interior_vertex_is_not_a_junction():
    # Two interior visits by one line are not linked (unchanged behaviour).
    line = LineString([(0, 0), (100, 0), (100, 100), (50, 50), (100, 0), (200, 0)])
    net = build_network(gdf([line]))
    assert net.n_edges == 1
    assert net.edge_len[0] == pytest.approx(line.length)


def test_dense_vertices_of_one_line_are_not_chained_into_one_location():
    # Vertices 2 cm apart are closer than snap (0.1 m), but nearby interior
    # vertices of the same line are never linked to each other.
    dense = LineString([(x, 0.0) for x in np.linspace(0.0, 10.0, 501)])
    alone = build_network(gdf([dense]))
    assert alone.n_edges == 1
    assert alone.edge_len.sum() == pytest.approx(10.0)

    # A side road ending on one vertex joins there. Only the stretch within
    # snap of that junction collapses into the junction node.
    side = LineString([(5.0, 0.0), (5.0, 50.0)])
    net = build_network(gdf([dense, side]))
    assert connected_components(net.csr, directed=False)[0] == 1
    assert net.edge_len.sum() > 60.0 - 2 * 0.1
    assert _road_distance(net, (0.0, 0.0), (10.0, 0.0)) > 10.0 - 2 * 0.1


def test_node_numbering_follows_snap_cells_as_in_v010():
    # v0.1.0 numbered nodes by snap-grid cell (x cell, then y cell). The same
    # order is kept so that node ids saved from earlier runs stay valid: the
    # node at x = 0.03 comes before the node at x = 0.01 because both lie in
    # the x cell 0 and its y cell is smaller.
    lines = [
        LineString([(0.01, 500.0), (300.0, 500.0)]),
        LineString([(0.03, 100.0), (300.0, 100.0)]),
        LineString([(0.03, 100.0), (0.01, 500.0)]),
    ]
    net = build_network(gdf(lines))
    assert net.node_xy[:2].tolist() == [[0.03, 100.0], [0.01, 500.0]]
    shuffled = build_network(gdf(lines[::-1]))
    assert np.array_equal(shuffled.node_xy, net.node_xy)


def test_vectorised_cutting_matches_a_piece_by_piece_reference():
    import shapely

    from net_center.graph import _cut_lines

    rng = np.random.default_rng(0)
    lines = [
        LineString(rng.uniform(0, 1000, (int(rng.integers(2, 9)), 2))) for _ in range(40)
    ]
    parts = np.asarray(lines, dtype=object)
    xy, line_id = shapely.get_coordinates(parts, return_index=True)
    starts = np.searchsorted(line_id, np.arange(len(parts)))
    ends = np.searchsorted(line_id, np.arange(len(parts)), side="right") - 1
    interior = np.ones(len(xy), dtype=bool)
    interior[starts] = interior[ends] = False
    cut_here = interior & (rng.random(len(xy)) < 0.4)

    expected = []
    for p in range(len(parts)):
        cuts = np.flatnonzero(cut_here[starts[p] : ends[p] + 1]).tolist()
        if not cuts:
            expected.append(np.asarray(parts[p].coords))
            continue
        bounds = [0, *cuts, ends[p] - starts[p]]
        coords = xy[starts[p] : ends[p] + 1]
        expected += [
            coords[a : b + 1] for a, b in zip(bounds[:-1], bounds[1:], strict=True)
        ]

    got = _cut_lines(parts, xy, line_id, cut_here)
    assert len(got) == len(expected)
    for g, e in zip(got, expected, strict=True):
        assert np.array_equal(np.asarray(g.coords), e)
