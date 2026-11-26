"""Component-aware wrapper around the existing exact absolute 1-center solver."""

from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import dijkstra

from net_center.adaptive import (
    RoutingContext,
    _finite,
    _indices,
    _integer,
    _vector,
    _weighted_quantiles,
)


def solve_absolute_groups(net, demand_nodes, groups, *, weights=None, offsets=None,
                          include_offsets=False, component_policy='strict',
                          max_centers=12, max_temp_mb=64, max_matrix_mb=512,
                          physical_edges=None, endpoint_tolerance=0.1,
                          largest_component_anchor=None):
    """One continuous minimax center per group/component, not absolute k-center.

    The legacy edge sweep works for additive fixed origin offsets because its
    tent breakpoints are unchanged and its eccentricity bounds remain valid.
    Returns original group-row positions and JSON-ready records with explicit
    vertex/physical-edge location records; no fictitious vertex IDs are used.
    """
    from net_center.center import absolute_center
    from net_center.edge_snap import _geometry

    _geometry(net, endpoint_tolerance=endpoint_tolerance)
    actual_edges = (net.edge_u, net.edge_w, net.edge_len)
    if physical_edges is not None and (len(physical_edges) != 3 or not all(
        np.array_equal(given, actual)
        for given, actual in zip(physical_edges, actual_edges, strict=True)
    )):
        raise ValueError('physical_edges must describe the actual solver network')

    ctx = RoutingContext(net.csr, max_temp_mb=max_temp_mb, max_matrix_mb=max_matrix_mb,
                         physical_edges=actual_edges)
    if largest_component_anchor is not None:
        if component_policy != 'largest':
            raise ValueError('largest component anchor requires largest policy')
        ctx._pin_original_largest(largest_component_anchor)
    q = _indices(demand_nodes, ctx.n, 'demand_nodes')
    w = _vector(weights, len(q), 'weights', 1.)
    off = _vector(offsets, len(q), 'offsets', 0.)
    labels = np.asarray(groups, dtype=object)
    if labels.shape != q.shape or any(not isinstance(g, str) or not g for g in labels):
        raise ValueError('groups must be aligned nonempty strings')
    cap = _integer(max_centers, 'max_centers')
    if component_policy not in ('strict', 'largest', 'per_component'):
        raise ValueError('absolute center supports strict, largest or per_component')
    output = {}
    for group in sorted(set(labels)):
        rows = np.flatnonzero(labels == group)
        demand, mass, origin = q[rows], w[rows], off[rows]
        total = _finite(mass.sum(), 'group weight')
        comp_labels = ctx.labels[demand]
        modeled = (comp_labels == ctx.largest if component_policy == 'largest'
                   else np.ones(len(rows), dtype=bool))
        modeled_weight = _finite(mass[modeled].sum(), 'modeled weight')
        if modeled_weight <= 0:
            raise ValueError(f'group {group!r} has no positive modeled weight')
        components = np.unique(comp_labels[modeled])
        if (component_policy == 'strict' and len(components) != 1) or len(components) > cap:
            raise ValueError('absolute center requires one site per populated component')
        assignment = np.full(len(rows), -1, dtype=np.int64)
        road = np.full(len(rows), np.inf)
        sites = []
        for comp in components:
            selected = np.flatnonzero(modeled & (comp_labels == comp))
            ids, sub = ctx.local(comp)
            local = np.searchsorted(ids, demand[selected])
            unique, inverse = np.unique(local, return_inverse=True)
            inverse = inverse.reshape(-1)
            D = ctx.matrix(sub, unique, np.arange(len(ids)))
            if include_offsets:
                max_offsets = np.zeros(len(unique), dtype=float)
                np.maximum.at(max_offsets, inverse, origin[selected])
                D += max_offsets[:, None]
            edges = np.flatnonzero(ctx.labels[net.edge_u] == comp)
            u = np.searchsorted(ids, net.edge_u[edges])
            v = np.searchsorted(ids, net.edge_w[edges])
            result = absolute_center(D, u, v, net.edge_len[edges])
            del D
            if result.node is not None:
                node = int(ids[result.node])
                xy = net.node_xy[node]
                dist = dijkstra(sub, directed=True, indices=result.node)[local]
                location = {'node': node, 'edge': None, 'edge_offset': None}
            else:
                edge = int(edges[result.edge])
                t = float(result.t)
                xy = net.interpolate(edge, t)
                du = dijkstra(sub, directed=True, indices=int(u[result.edge]))[local]
                dv = dijkstra(sub, directed=True, indices=int(v[result.edge]))[local]
                dist = np.minimum(du + t, dv + net.edge_len[edge] - t)
                location = {'node': None, 'edge': edge, 'edge_offset': t}
            road[selected] = dist
            assignment[selected] = len(sites)
            access = dist + origin[selected]
            radius = _finite((access if include_offsets else dist).max(), 'radius')
            if not np.isclose(radius, result.objective, rtol=1e-8, atol=1e-8):
                raise RuntimeError('absolute-center radius does not match routed demand')
            site_weight = _finite(mass[selected].sum(), 'site weight')
            sites.append({**location, 'xy': xy, 'component': int(comp),
                          'objective': radius, 'assigned_weight': site_weight,
                          'assigned_count': len(selected),
                          'max_access': float(access.max()),
                          'mean_access': (_finite(mass[selected] @ access / site_weight,
                                                  'mean access') if site_weight else None)})
        access = road + origin
        if not np.isfinite(access[modeled]).all() or np.any(assignment[modeled] < 0):
            raise RuntimeError('absolute centers failed to cover modeled demand')
        output[group] = (rows, {
            'center_locations': sites, 'center_xy': [site['xy'] for site in sites],
            'assignment': assignment, 'network_distance': road, 'access_distance': access,
            'component': comp_labels, 'objective': max(s['objective'] for s in sites),
            'total_weight': total, 'modeled_weight': modeled_weight,
            'coverage': modeled_weight / total,
            'mean_access': _finite(mass[modeled] @ access[modeled] / modeled_weight,
                                   'mean access'),
            'max_access': float(access[modeled].max()), 'selection': 'fixed',
            'candidate_domain': 'continuous',
            'guarantee': 'exact_single_site_per_component_with_numerical_tolerance',
            'diagnostics': {'selected_k': len(sites), 'objective': 'absolute_center',
                            'component_policy': component_policy,
                            'largest_basis': ctx.largest_basis,
                            'modeled_row_fraction': float(modeled.mean()),
                            'weighted_access_percentiles': _weighted_quantiles(
                                access[modeled], mass[modeled]
                            ),
                            'includes_offsets': bool(include_offsets)},
        })
    return output
