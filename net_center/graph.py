"""Convert line geometry into the compact undirected network used by net-center.

A road layer is not automatically a routing graph. The important distinction is
between *geometry* (lines that happen to cross on a map) and *topology* (places
where travel can actually move from one road to another).

Safe default
------------
``build_network(..., node=False)`` preserves source topology: it splits at
vertices already shared by distinct input lines, but does not invent junctions
at geometric crossings. This is the safer default for routing-quality data,
including data that distinguishes bridges, tunnels, and grade-separated roads.

Optional planar noding
----------------------
``node=True`` splits every two-dimensional line crossing and therefore assumes
that every drawn crossing is a valid junction. That can be useful for simple
street drawings, but it is wrong for overpasses and underpasses. The option is
explicit because silently inventing a turn is worse than failing loudly.

One location rule
-----------------
Every decision about "the same place" uses one rule: two coordinates within
``snap`` metres of each other are one network location, and the rule is
applied transitively (see :func:`_close_point_labels`). The same rule decides
node identity, which source vertices are shared junctions, and whether a line
is a ring (both ends at one location). Using one rule everywhere matters: when
these decisions used different rules, a junction could be split in two and a
ring could be deleted as a self-loop.

Other protections in this module handle empty geometries, closed rings,
self-touching lines, duplicate parallel edges, metric CRS units, and stranded
connected components.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import shapely
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from net_center.topology import adjacency_from_edges

_LINESTRING = 1


@dataclass(slots=True)
class Network:
    """Road network represented as numeric arrays plus original edge geometry."""

    node_xy: np.ndarray
    edge_u: np.ndarray
    edge_w: np.ndarray
    edge_len: np.ndarray
    edge_geom: np.ndarray
    csr: csr_matrix
    crs: object = None

    @property
    def n_nodes(self) -> int:
        return len(self.node_xy)

    @property
    def n_edges(self) -> int:
        return len(self.edge_u)

    def interpolate(self, edge: int, t: float) -> tuple[float, float]:
        """Return map coordinates ``t`` metres from the u-end of one edge."""
        if edge < 0 or edge >= self.n_edges:
            raise IndexError("edge index is outside the network")
        if not np.isfinite(t):
            raise ValueError("t must be finite")
        if t < -1e-9 or t > self.edge_len[edge] + 1e-9:
            raise ValueError("t falls outside the selected edge")
        # Clamp tiny floating-point excursions at exactly 0 or L.
        t = float(np.clip(t, 0.0, self.edge_len[edge]))
        pt = shapely.line_interpolate_point(self.edge_geom[edge], t)
        return (float(shapely.get_x(pt)), float(shapely.get_y(pt)))


def _explode_lines(geoms: np.ndarray) -> np.ndarray:
    """Flatten multipart geometry into non-empty LineStrings only."""
    parts = shapely.get_parts(geoms)
    parts = parts[shapely.get_type_id(parts) == _LINESTRING]
    return parts[~shapely.is_empty(parts)]


def _node_and_merge(geoms: np.ndarray) -> np.ndarray:
    """Planarise all crossings, then merge degree-2 shape segments.

    This treats *every* 2-D crossing as connected. Call it only when that is a
    valid assumption for the input data.
    """
    merged = shapely.line_merge(shapely.union_all(geoms))
    return _explode_lines(np.asarray([merged], dtype=object))


def _run_bounds(line_id: np.ndarray, n_lines: int) -> tuple[np.ndarray, np.ndarray]:
    """Index of the first and last coordinate of each line in a flat array.

    ``line_id`` is the per-coordinate line index returned by
    ``shapely.get_coordinates(..., return_index=True)``; every line must have
    at least one coordinate (empty geometry is removed before this point).
    """
    lines = np.arange(n_lines)
    starts = np.searchsorted(line_id, lines, side="left")
    ends = np.searchsorted(line_id, lines, side="right") - 1
    return starts, ends


def _check_finite(xy: np.ndarray) -> None:
    """Reject NaN or infinite coordinates before any distance is measured."""
    if not np.isfinite(xy).all():
        raise ValueError("line geometry contains NaN or infinite coordinates")


def _close_point_labels(xy: np.ndarray, tol: float, may_link=None) -> np.ndarray:
    """Label points so that points within ``tol`` metres share a label.

    Two points receive the same label when a chain of points connects them in
    which each step is at most ``tol`` long (single-linkage clustering). Unlike
    rounding coordinates to a grid, this never separates two nearly identical
    points because a grid-cell boundary happens to lie between them.

    ``may_link(i, j)`` can refuse candidate pairs. It receives two index arrays
    and returns a boolean array.
    """
    n = len(xy)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    pairs = cKDTree(xy).query_pairs(r=tol, output_type="ndarray").reshape(-1, 2)
    if may_link is not None and len(pairs):
        pairs = pairs[may_link(pairs[:, 0], pairs[:, 1])]
    links = coo_matrix(
        (np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n)
    )
    _, labels = connected_components(links, directed=False)
    return labels.astype(np.int64, copy=False)


def _split_at_shared_vertices(parts: np.ndarray, snap: float) -> np.ndarray:
    """Split lines at junctions that the source vertices already encode.

    This is the topology-preserving default for OSM-shaped linework. A real
    junction is often one coordinate that belongs to two or more ways without
    being an endpoint of either, so splitting only at line ends would miss it.

    A vertex location is a junction when

    * vertices of two or more *different* lines lie there (T- and X-junctions),
      or
    * a line endpoint lies there. This includes a line's own endpoint: a
      "lollipop" way that ends on one of its own interior vertices (common for
      cul-de-sac turning loops) is split where it touches itself.

    Every line is split at each of its interior vertices that lies at a
    junction. Crucially, no coordinate is invented at a mere geometric
    crossing: a bridge that shares no source vertex with the road beneath it
    stays disconnected unless the caller asks for planar noding (``node=True``).

    "Same location" means within ``snap`` metres (:func:`_close_point_labels`).
    Two vertices of one and the same line are linked only when one of them is
    an endpoint and the line travels more than ``snap`` between them. So a
    densely digitised road is not merged into a single location, and a line
    that crosses itself at an interior vertex is not a junction.
    """
    if len(parts) == 0:
        return parts

    xy, line_id = shapely.get_coordinates(parts, return_index=True)
    _check_finite(xy)
    starts, ends = _run_bounds(line_id, len(parts))
    is_end = np.zeros(len(xy), dtype=bool)
    is_end[starts] = True
    is_end[ends] = True

    # Distance of every vertex along its own line, in metres from the start.
    step = np.r_[0.0, np.hypot(*np.diff(xy, axis=0).T)]
    step[starts] = 0.0  # no step across the boundary between two lines
    along = np.cumsum(step)
    along -= along[starts][line_id]

    def may_link(i, j):
        # Vertices of different lines within snap are one location. Within one
        # line, a vertex is linked to the line's own endpoint only when the
        # line travels more than snap between them -- it left the endpoint and
        # came back, as in a lollipop loop. Without this condition the closely
        # spaced vertices next to the end of a densely digitised road would be
        # merged into its endpoint, shortening the road.
        other_line = line_id[i] != line_id[j]
        returns = (is_end[i] | is_end[j]) & (np.abs(along[i] - along[j]) > snap)
        return other_line | returns

    location = _close_point_labels(xy, snap, may_link)

    # Labels are 0..n_locations-1, so after sorting by label the g-th group is
    # label g. A location is shared when the smallest and largest line id in
    # its group differ.
    order = np.argsort(location, kind="stable")
    group_start = np.flatnonzero(np.r_[True, np.diff(location[order]) != 0])
    min_line = np.minimum.reduceat(line_id[order], group_start)
    max_line = np.maximum.reduceat(line_id[order], group_start)
    junction = min_line != max_line
    junction[location[is_end]] = True

    cut_here = ~is_end & junction[location]
    if not cut_here.any():
        return parts
    return _cut_lines(parts, xy, line_id, cut_here)


def _cut_lines(parts, xy, line_id, cut_here) -> np.ndarray:
    """Split lines at the flagged interior vertices, keeping input order.

    Lines without a cut keep their original geometry object. The pieces of all
    cut lines are built in one vectorised Shapely call: each cut vertex is
    written twice, once as the end of one piece and once as the start of the
    next.
    """
    cuts_per_line = np.bincount(line_id[cut_here], minlength=len(parts))
    pieces_per_line = cuts_per_line + 1
    is_cut_line = cuts_per_line > 0

    vertex = np.flatnonzero(is_cut_line[line_id])
    vertex = np.repeat(vertex, np.where(cut_here[vertex], 2, 1))
    # A new piece begins at the first vertex of each line and at the second
    # copy of each cut vertex.
    first_of_line = np.r_[True, line_id[vertex[1:]] != line_id[vertex[:-1]]]
    second_copy = np.r_[False, vertex[1:] == vertex[:-1]]
    piece = np.cumsum(first_of_line | second_copy) - 1
    new_pieces = shapely.linestrings(xy[vertex], indices=piece)

    # Put each line's pieces in the slots its original geometry occupied, so
    # the order of edges still follows the order of the input features.
    out = np.empty(int(pieces_per_line.sum()), dtype=object)
    first_slot = np.cumsum(pieces_per_line) - pieces_per_line
    uncut_slots = first_slot[~is_cut_line]
    out[uncut_slots] = parts[~is_cut_line]
    free = np.ones(len(out), dtype=bool)
    free[uncut_slots] = False
    out[free] = new_pieces
    return out


def _endpoints(parts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised extraction of the first and last coordinate of each line."""
    xy, line_id = shapely.get_coordinates(parts, return_index=True)
    _check_finite(xy)
    starts, ends = _run_bounds(line_id, len(parts))
    return xy[starts], xy[ends]


def _endpoint_locations(parts: np.ndarray, snap: float):
    """Endpoints, lengths, and the location label of both ends of every line."""
    p0, p1 = _endpoints(parts)
    lengths = np.asarray(shapely.length(parts), dtype=np.float64)
    location = _close_point_labels(np.vstack([p0, p1]), snap)
    m = len(parts)
    return p0, p1, lengths, location[:m], location[m:]


def _split_rings(parts: np.ndarray, is_ring: np.ndarray) -> np.ndarray:
    """Replace each ring by its two halves, appended after the other lines.

    A ring here is any line whose two ends are one network location, whether
    it is exactly closed or only nearly closed. Left whole it would become a
    self-loop and be discarded, deleting real road.
    """
    from shapely.ops import substring

    halves = []
    for geom in parts[is_ring]:
        half = geom.length / 2.0
        halves.append(substring(geom, 0.0, half))
        halves.append(substring(geom, half, geom.length))
    return np.concatenate([parts[~is_ring], np.asarray(halves, dtype=object)])


def _is_metre_crs(crs) -> bool:
    """True when both horizontal axes use metres as their linear unit."""
    from pyproj import CRS

    crs = CRS.from_user_input(crs)
    if not crs.is_projected:
        return False
    axes = crs.axis_info[:2]
    if len(axes) < 2:
        return False
    return all(np.isclose(axis.unit_conversion_factor, 1.0) for axis in axes)


def _working_crs(gdf, target_crs=None):
    """Choose a projected metre-based CRS and reject ambiguous unit mistakes."""
    from pyproj import CRS

    source_crs = CRS.from_user_input(gdf.crs)

    if target_crs is not None:
        chosen = CRS.from_user_input(target_crs)
        if not chosen.is_projected:
            raise ValueError("target_crs must be projected, not latitude/longitude")
        if not _is_metre_crs(chosen):
            raise ValueError(
                "target_crs must use metres because all net-center distances and "
                "the snap distance are defined in metres"
            )
        return chosen

    if source_crs.is_projected and _is_metre_crs(source_crs):
        # Do not reproject good projected input: changing CRS needlessly changes
        # every measured edge length a little.
        return source_crs

    # Geographic input, or projected input in feet/another unit: choose a local
    # UTM CRS so lengths and tolerances have the documented metre interpretation.
    chosen = gdf.estimate_utm_crs()
    if chosen is None:
        raise ValueError(
            "could not infer a local metre-based CRS; pass target_crs explicitly"
        )

    if source_crs.is_geographic:
        minx, miny, maxx, maxy = gdf.total_bounds
        if maxx - minx > 6.0:
            warnings.warn(
                "input spans more than 6 degrees of longitude; one inferred UTM "
                "zone may distort a very large study area. Pass an explicit "
                "metre-based projected CRS appropriate to the full extent.",
                UserWarning,
                stacklevel=3,
            )
    else:
        warnings.warn(
            "input CRS is projected but not metre-based; reprojecting to an "
            "estimated local UTM CRS so reported distances remain metres.",
            UserWarning,
            stacklevel=3,
        )
    return CRS.from_user_input(chosen)


def _number_nodes(p0, p1, loc_u, loc_w, snap: float):
    """Give the locations used by the surviving edges dense ids ``0..n-1``.

    Each node is placed at the first endpoint coordinate that maps to it.
    Locations used only by discarded edges receive no id, so filtering cannot
    leave isolated orphan nodes in the graph.

    Nodes are numbered by ``(round(x / snap), round(y / snap))`` and then by
    exact ``(x, y)``. The first key reproduces the numbering of net-center
    v0.1.0, which sorted nodes by snap-grid cell, so node ids saved from
    earlier runs stay valid. It is computed in floating point and therefore
    cannot overflow. The numbering never depends on the order of the input
    features.
    """
    location = np.concatenate([loc_u, loc_w])
    used, first = np.unique(location, return_index=True)
    xy = np.vstack([p0, p1])[first]
    cell = np.rint(xy / snap)
    rank = np.lexsort((xy[:, 1], xy[:, 0], cell[:, 1], cell[:, 0]))
    node_id = np.empty(len(used), dtype=np.int64)
    node_id[rank] = np.arange(len(used), dtype=np.int64)
    ids = node_id[np.searchsorted(used, location)]
    m = len(loc_u)
    return xy[rank], ids[:m], ids[m:]


def _largest_component_by_length(labels, u, w, lengths) -> int:
    """Choose the connected component containing the most road length."""
    n_components = int(labels.max()) + 1
    totals = np.zeros(n_components, dtype=np.float64)
    # Every retained edge has endpoints in one component. Count its length once.
    np.add.at(totals, labels[u], lengths)
    return int(np.argmax(totals))


def build_network(
    source,
    layer: str | None = None,
    target_crs=None,
    snap: float = 0.1,
    node: bool = False,
    split_shared_vertices: bool = True,
    min_length: float = 1e-9,
    keep_largest_component: bool = True,
) -> Network:
    """Build a :class:`Network` from a vector file or GeoDataFrame.

    Parameters
    ----------
    source
        Shapefile/GeoPackage/etc. path or a GeoDataFrame containing linework.
    layer
        Optional layer name for multi-layer files.
    target_crs
        Optional projected CRS whose horizontal units are metres. Geographic
        input is otherwise moved to an inferred local UTM CRS automatically.
    snap
        Node-identity tolerance in metres. Coordinates closer than this are
        one network location, and the rule is applied transitively. It governs
        line endpoints and the shared source vertices used for junction
        recovery; arbitrary points are never projected onto roads. A line
        whose two ends are one location (a closed or nearly closed ring) is
        split into two halves so that it is not deleted as a self-loop.
    node
        If true, split every geometric line crossing. Use only for a planar
        network where every 2-D crossing is a genuine junction. This also
        connects bridges to whatever passes beneath them.
    split_shared_vertices
        If true (the default), split lines at vertices already shared by two or
        more distinct input LineStrings. This recovers OSM-style T- and
        X-junctions without inventing a junction at a bridge/underpass crossing.
    min_length
        Segments at or below this length are discarded as slivers.
    keep_largest_component
        If true, retain the connected component with the greatest road length.
    """
    import geopandas as gpd

    if not np.isfinite(snap) or snap <= 0:
        raise ValueError("snap must be a finite positive distance in metres")
    if not np.isfinite(min_length) or min_length < 0:
        raise ValueError("min_length must be a finite non-negative distance")

    gdf = (
        source
        if isinstance(source, gpd.GeoDataFrame)
        else gpd.read_file(source, layer=layer)
    )

    # Remove missing/empty geometry before endpoint indexing; an empty geometry
    # contributes no coordinates and can otherwise shift subsequent endpoints.
    gdf = gdf[~gdf.geometry.is_empty & ~gdf.geometry.isna()]
    gdf = gdf[gdf.geom_type.isin(["LineString", "MultiLineString"])]
    if len(gdf) == 0:
        raise ValueError("no usable line geometries found in source")
    if gdf.crs is None:
        raise ValueError(
            "source has no coordinate reference system; set one before building"
        )

    crs = _working_crs(gdf, target_crs)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    geoms = np.asarray(gdf.geometry.values, dtype=object)
    if node:
        warnings.warn(
            "planar noding is enabled: every geometric crossing will become a "
            "junction, including bridges/underpasses unless the input has already "
            "been separated geometrically.",
            UserWarning,
            stacklevel=2,
        )
        parts = _node_and_merge(geoms)
    else:
        parts = _explode_lines(geoms)
        if split_shared_vertices:
            parts = _split_at_shared_vertices(parts, snap)

    if len(parts) == 0:
        raise ValueError("no line segments survived preprocessing")

    # Locations of both ends of every line. A line whose ends are one location
    # is a ring: split it in two so it survives the self-loop filter below.
    p0, p1, lengths, loc_u, loc_w = _endpoint_locations(parts, snap)
    is_ring = (loc_u == loc_w) & (lengths > min_length)
    if is_ring.any():
        parts = _split_rings(parts, is_ring)
        p0, p1, lengths, loc_u, loc_w = _endpoint_locations(parts, snap)

    usable = (loc_u != loc_w) & np.isfinite(lengths) & (lengths > min_length)
    if not usable.any():
        raise ValueError("no usable segments after removing loops and slivers")
    parts, lengths = parts[usable], lengths[usable]
    node_xy, u, w = _number_nodes(
        p0[usable], p1[usable], loc_u[usable], loc_w[usable], snap
    )
    csr = adjacency_from_edges(u, w, lengths, len(node_xy))

    if keep_largest_component:
        n_components, labels = connected_components(csr, directed=False)
        if n_components > 1:
            keep_label = _largest_component_by_length(labels, u, w, lengths)
            node_keep = labels == keep_label
            remap = np.full(len(node_xy), -1, dtype=np.int64)
            remap[node_keep] = np.arange(int(node_keep.sum()))
            edge_keep = node_keep[u] & node_keep[w]

            kept_length = float(lengths[edge_keep].sum())
            total_length = float(lengths.sum())
            warnings.warn(
                f"network split into {n_components} disconnected pieces; keeping "
                f"the component with the most road length "
                f"({kept_length:,.1f} of {total_length:,.1f} m).",
                UserWarning,
                stacklevel=2,
            )

            u, w = remap[u[edge_keep]], remap[w[edge_keep]]
            lengths, parts = lengths[edge_keep], parts[edge_keep]
            node_xy = node_xy[node_keep]
            csr = adjacency_from_edges(u, w, lengths, len(node_xy))

    return Network(
        node_xy=np.asarray(node_xy, dtype=np.float64),
        edge_u=np.asarray(u, dtype=np.int64),
        edge_w=np.asarray(w, dtype=np.int64),
        edge_len=np.asarray(lengths, dtype=np.float64),
        edge_geom=np.asarray(parts, dtype=object),
        csr=csr,
        crs=crs,
    )


def snap_points(net: Network, xy: np.ndarray, max_dist: float | None = None):
    """Attach point demand to the nearest *network node*.

    This is intentionally node snapping, not nearest-edge snapping. The returned
    distance tells the caller how far each demand point was moved, so a sensible
    ``max_dist`` can prevent long simplified edges from causing hidden shifts.
    """
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(xy).all():
        raise ValueError("demand coordinates contain NaN or infinity")
    if max_dist is not None and (not np.isfinite(max_dist) or max_dist < 0):
        raise ValueError("max_dist must be a finite non-negative distance")
    if net.n_nodes == 0:
        raise ValueError("network has no nodes")

    dist, idx = cKDTree(net.node_xy).query(xy, k=1)
    idx = idx.astype(np.int64)
    if max_dist is not None:
        idx = np.where(dist <= max_dist, idx, -1)
    return idx, dist
