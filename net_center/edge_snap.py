"""Continuous demand attachment for the existing metre-based Network model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class EdgeSnaps:
    edge: np.ndarray
    offset: np.ndarray
    distance: np.ndarray
    xy: np.ndarray
    accepted: np.ndarray


@dataclass(frozen=True)
class Augmentation:
    network: object
    demand_nodes: np.ndarray
    original_edge: np.ndarray  # original edge id per augmented edge
    original_start: np.ndarray  # original edge start measure per augmented edge


def _geometry(net, *, endpoint_tolerance=0.1):
    import shapely

    if not np.isfinite(endpoint_tolerance) or endpoint_tolerance < 0:
        raise ValueError("endpoint_tolerance must be finite and nonnegative")
    geom = np.asarray(net.edge_geom, dtype=object)
    xy = np.asarray(net.node_xy, dtype=float)
    u, v = np.asarray(net.edge_u), np.asarray(net.edge_w)
    if (xy.ndim != 2 or xy.shape[1] != 2 or not len(xy) or
            not np.isfinite(xy).all() or u.dtype.kind not in 'iu' or
            v.dtype.kind not in 'iu' or geom.ndim != 1 or
            u.shape != geom.shape or v.shape != geom.shape or
            np.any(u < 0) or np.any(v < 0) or np.any(u >= len(xy)) or
            np.any(v >= len(xy)) or np.any(u == v)):
        raise ValueError('invalid physical edge endpoint arrays')
    lengths = np.asarray(net.edge_len, dtype=float)
    if not len(geom) or geom.shape != lengths.shape:
        raise ValueError('network must contain aligned edge geometry and length')
    if (np.any(shapely.get_type_id(geom) != 1) or
            not np.isfinite(lengths).all() or np.any(lengths <= 0) or
            not np.allclose(shapely.length(geom), lengths, rtol=1e-10, atol=1e-9)):
        raise ValueError('expected nonempty metric LineStrings with matching lengths')
    # Edge geometry is ordered from u to v. Node representatives can differ
    # from actual endpoints after source topology merging. Do not reverse or
    # silently repair data; require an explicit, bounded tolerance.
    first = shapely.get_coordinates(shapely.get_point(geom, 0))
    last = shapely.get_coordinates(shapely.get_point(geom, -1))
    if not np.isfinite(first).all() or not np.isfinite(last).all():
        raise ValueError('nonfinite physical edge endpoint geometry')
    forward_u = np.linalg.norm(first - xy[u], axis=1)
    forward_v = np.linalg.norm(last - xy[v], axis=1)
    reverse_u = np.linalg.norm(last - xy[u], axis=1)
    reverse_v = np.linalg.norm(first - xy[v], axis=1)
    tol = endpoint_tolerance + 1e-9
    if np.any(reverse_u + reverse_v + 1e-9 < forward_u + forward_v):
        raise ValueError('physical edge endpoint geometry/orientation disagrees with nodes')
    # The base graph merges endpoint coordinates transitively. A legitimate
    # representative can be farther than the pairwise tolerance. For those
    # nodes only, verify a chain of retained endpoint coordinates to the
    # representative. No new graph links are created by this check.
    distant = np.unique(np.r_[u[forward_u > tol], v[forward_v > tol]])
    if len(distant):
        from scipy.sparse.csgraph import connected_components
        from scipy.spatial import cKDTree

        for node in distant:
            points = np.unique(np.vstack((xy[node], first[u == node], last[v == node])),
                               axis=0)
            pairs = cKDTree(points).query_pairs(tol, output_type='ndarray')
            links = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                               shape=(len(points), len(points))).tocsr()
            if connected_components(links, directed=False, return_labels=False) != 1:
                raise ValueError('physical edge endpoint geometry/orientation '
                                 'disagrees with nodes; no retained tolerance chain')
    if net.csr.shape != (len(xy), len(xy)) or (
        net.csr != _adjacency(u, v, lengths, len(xy))
    ).nnz:
        raise ValueError('physical edges disagree with routing graph')
    return geom, lengths


def snap_to_edges(net, xy, *, max_distance=None, endpoint_tolerance=0.1):
    """Project onto closest physical edge; exact ties use the smallest edge id.

    ``xy`` must be in the existing network's projected CRS. No new junction
    is introduced where distinct physical edges cross in two dimensions.
    """
    import shapely

    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) == 0 or not np.isfinite(xy).all():
        raise ValueError('xy must be a nonempty finite n-by-2 array')
    if max_distance is not None and (not np.isfinite(max_distance) or max_distance < 0):
        raise ValueError('max_distance must be finite and nonnegative')
    geom, lengths = _geometry(net, endpoint_tolerance=endpoint_tolerance)
    points = shapely.points(xy)
    pairs = shapely.STRtree(geom).query_nearest(points, all_matches=True)
    edge = np.full(len(xy), len(geom), dtype=np.int64)
    np.minimum.at(edge, pairs[0], pairs[1])
    if np.any(edge == len(geom)):
        raise RuntimeError('nearest-edge query did not return all demand rows')
    offset = np.clip(shapely.line_locate_point(geom[edge], points), 0., lengths[edge])
    projected = shapely.line_interpolate_point(geom[edge], offset)
    snapped = shapely.get_coordinates(projected)
    dist = shapely.distance(points, projected)
    if not np.isfinite(dist).all() or not np.isfinite(offset).all():
        raise ValueError('nonfinite snap result')
    accepted = (np.ones(len(xy), dtype=bool) if max_distance is None
                else dist <= max_distance)
    return EdgeSnaps(edge, offset, dist, snapped, accepted)


def _adjacency(u, v, lengths, n):
    # Same minimum-parallel-edge semantics as the existing topology module.
    lo, hi = np.minimum(u, v), np.maximum(u, v)
    order = np.lexsort((lengths, hi, lo))
    lo, hi, lengths = lo[order], hi[order], lengths[order]
    keep = np.ones(len(lo), dtype=bool)
    keep[1:] = (lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1])
    lo, hi, lengths = lo[keep], hi[keep], lengths[keep]
    return coo_matrix((np.r_[lengths, lengths], (np.r_[lo, hi], np.r_[hi, lo])),
                      shape=(n, n)).tocsr()


def augment_network(net, snaps, *, endpoint_tolerance=0.1):
    """Split touched edges once, retaining original vertices and edge lineage.

    Exact coincident offsets share a node. Distinct interior offsets on distinct
    physical edges never merge, even if their map coordinates coincide.
    Rejected rows return node -1. The original Network is not mutated.
    """
    import shapely
    from shapely.ops import substring

    geom, lengths = _geometry(net, endpoint_tolerance=endpoint_tolerance)
    edge = np.asarray(snaps.edge)
    offsets = np.asarray(snaps.offset, dtype=float)
    accepted = np.asarray(snaps.accepted)
    if (edge.ndim != 1 or edge.dtype.kind not in 'iu' or accepted.dtype.kind != 'b' or
            offsets.shape != edge.shape or accepted.shape != edge.shape or
            np.any(edge < 0) or np.any(edge >= len(geom)) or
            not np.isfinite(offsets).all() or np.any(offsets < 0) or
            np.any(offsets > lengths[edge])):
        raise ValueError('invalid edge snap arrays')
    snapped_xy = np.asarray(snaps.xy, dtype=float)
    distance = np.asarray(snaps.distance, dtype=float)
    expected_xy = shapely.get_coordinates(
        shapely.line_interpolate_point(geom[edge], offsets)
    )
    if (snapped_xy.shape != (len(edge), 2) or distance.shape != edge.shape or
            not np.isfinite(distance).all() or np.any(distance < 0) or
            not np.isfinite(snapped_xy).all() or
            not np.allclose(snapped_xy, expected_xy, rtol=0., atol=1e-8)):
        raise ValueError('snap metadata does not match this physical network')
    node_xy = np.asarray(net.node_xy, dtype=float).tolist()
    demand = np.full(len(edge), -1, dtype=np.int64)
    us, vs, ls, gs, lineage, starts = [], [], [], [], [], []
    by_edge = {}
    for row in np.flatnonzero(accepted):
        by_edge.setdefault(int(edge[row]), []).append(int(row))
    for e, line in enumerate(geom):
        L = float(lengths[e])
        u, v = int(net.edge_u[e]), int(net.edge_w[e])
        rows = by_edge.get(e, [])
        # Only normalize the exact endpoints; no global coordinate merging.
        measures = sorted({float(offsets[r]) for r in rows if 0 < offsets[r] < L})
        nodes = {0.: u, L: v}
        for t in measures:
            point = shapely.line_interpolate_point(line, t)
            nodes[t] = len(node_xy)
            node_xy.append([float(point.x), float(point.y)])
        for row in rows:
            demand[row] = nodes[float(offsets[row])]
        chain = [0., *measures, L]
        for a, b in zip(chain[:-1], chain[1:], strict=True):
            piece = line if not measures else substring(line, a, b)
            if (piece.geom_type != 'LineString' or piece.length <= 0 or
                    not np.isclose(piece.length, b-a, rtol=1e-9, atol=1e-9)):
                raise ValueError('edge split produced invalid or degenerate geometry')
            us.append(nodes[a])
            vs.append(nodes[b])
            ls.append(b-a)
            gs.append(piece)
            lineage.append(e)
            starts.append(a)
    u, v, length = np.array(us), np.array(vs), np.array(ls)
    if np.any(u == v) or np.any(length <= 0):
        raise ValueError('augmented network contains invalid edge')
    network = type(net)(node_xy=np.asarray(node_xy), edge_u=u, edge_w=v,
                        edge_len=length, edge_geom=np.asarray(gs, dtype=object),
                        csr=_adjacency(u, v, length, len(node_xy)), crs=net.crs)
    _geometry(network, endpoint_tolerance=endpoint_tolerance)
    return Augmentation(network, demand, np.array(lineage), np.array(starts))
