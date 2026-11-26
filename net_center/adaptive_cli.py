"""Additive GIS CLI: python -m net_center.adaptive_cli --help."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from net_center.adaptive import RoutingContext, solve_groups
from net_center.edge_snap import _geometry, augment_network, snap_to_edges


def parser():
    p = argparse.ArgumentParser(description='Adaptive network location and representation')
    p.add_argument('network', help='Input line network file')
    p.add_argument('--layer', help='Network layer name')
    p.add_argument('--crs', help='Projected working CRS with metre units')
    p.add_argument('--snap', type=float, default=0.1, help='Node identity tolerance (m)')
    p.add_argument('--node-crossings', action='store_true', help='Planar networks only')
    p.add_argument('--no-shared-vertex-noding', action='store_true',
                   help='Disable recovery of junctions at shared source vertices')
    p.add_argument('--keep-largest-component', action='store_true',
                   help='build the network as the legacy net-center CLI does: keep only '
                        'the component with the most road length before snapping; '
                        'the maximum snap distance still applies')
    p.add_argument('--demand', required=True, help='Input demand point file')
    p.add_argument('--demand-layer', help='Demand layer name')
    p.add_argument('--group-by', help='Field defining independent demand groups')
    p.add_argument('--weight-field', help='Nonnegative demand weight field')
    p.add_argument('--snap-mode', choices=['edge', 'node'], default='edge',
                   help='Attach to edge interiors (default) or existing nodes')
    p.add_argument('--max-snap', type=float, help='Maximum connector length in metres')
    p.add_argument('--on-unsnapped', choices=['error', 'exclude'], default='error',
                   help='Fail (default) or audit and exclude unattached demand rows')
    p.add_argument('--component-policy', choices=['strict', 'largest', 'per_component',
                                                  'allocate'], default='strict',
                   help='Treatment of demand spanning disconnected components')
    p.add_argument('--candidate-domain',
                   choices=['vertices', 'demand', 'restricted', 'continuous'],
                   help='Eligible facility locations; continuous for absolute center only')
    p.add_argument('--candidate-points', help='Point file for restricted candidates')
    p.add_argument('--candidate-layer', help='Restricted candidate layer name')
    p.add_argument('--objective', choices=['median', 'vertex_center', 'absolute_center'],
                   default='median', help='Location objective (default: median)')
    p.add_argument('--selection',
                   choices=['fixed', 'balanced', 'access_budget', 'gap', 'penalized'],
                   default='fixed', help='Site-count selection policy (default: fixed)')
    p.add_argument('--k', type=int, help='Fixed site count (default: 1 for fixed)')
    p.add_argument('--max-centers', type=int, default=12,
                   help='Maximum sites per group (default: 12)')
    p.add_argument('--access-budget', type=float, help='Mean access budget in metres')
    p.add_argument('--include-offsets', action='store_true',
                   help='Include fixed demand connectors in the location objective')
    p.add_argument('--penalty', type=float, default=None,
                   help='Balanced selection complexity penalty (default: 1)')
    p.add_argument('--facility-cost', type=float, help='Cost per site in weight-metres')
    p.add_argument('--algorithm', choices=['heuristic', 'exact'], default=None,
                   help='Multi-site curve method (default: heuristic)')
    p.add_argument('--max-combinations', type=int, default=None,
                   help='Exact subset enumeration cap per component (default: 100000)')
    p.add_argument('--swap-passes', type=int, default=None,
                   help='Maximum heuristic SWAP passes (default: 3)')
    p.add_argument('--gap-replicates', type=int, default=None,
                   help='Gap reference replicates (default: 10)')
    p.add_argument('--min-members', type=int, default=None,
                   help='Minimum row count per selected Gap site (default: 1)')
    p.add_argument('--seed', type=int, default=None, help='Gap random seed (default: 0)')
    p.add_argument('--max-temp-mb', type=float, default=64,
                   help='Temporary routing memory target in MiB (default: 64)')
    p.add_argument('--max-matrix-mb', type=float, default=512,
                   help='Dense distance matrix limit in MiB (default: 512)')
    p.add_argument('--no-prune', action='store_true',
                   help='Disable streaming single-median distance pruning')
    p.add_argument('--out', required=True, type=Path,
                   help='New JSON output path; must not already exist')
    return p


def _json(value):
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (list, tuple)):
        return [_json(x) for x in value]
    if isinstance(value, dict):
        return {key: _json(v) for key, v in value.items()}
    return value


def _publish(path, output):
    """Write one complete JSON artifact; do not replace an existing output."""
    if path.exists():
        raise FileExistsError(f'output already exists: {path}')
    if not path.parent.is_dir():
        raise ValueError('output directory must already exist')
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False,
                                         dir=path.parent, prefix='.net-center-') as stream:
            name = stream.name
            json.dump(_json(output), stream, allow_nan=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic no-clobber creation on filesystems supporting hard links.
        os.link(name, path)
    finally:
        if name is not None:
            os.unlink(name)


def run(args):
    import geopandas as gpd

    from net_center.graph import build_network, snap_points

    if args.out.exists():
        raise FileExistsError(f'output already exists: {args.out}')
    if args.selection != 'fixed' and args.k is not None:
        raise ValueError('--k is only valid with fixed selection')
    if args.candidate_domain is None:
        args.candidate_domain = ('continuous' if args.objective == 'absolute_center'
                                 else 'vertices')
    if (args.candidate_domain == 'restricted') != bool(args.candidate_points):
        raise ValueError('restricted domain requires --candidate-points; others forbid it')
    if args.candidate_layer is not None and args.candidate_points is None:
        raise ValueError('--candidate-layer requires --candidate-points')
    if args.objective == 'absolute_center':
        if (args.component_policy == 'allocate' or args.no_prune or any(
            value is not None for value in (args.penalty, args.swap_passes,
                                            args.gap_replicates, args.min_members,
                                            args.seed, args.max_combinations,
                                            args.algorithm)
        )):
            raise ValueError(
                'absolute center does not accept multi-site, Gap or pruning options'
            )
        if (args.candidate_domain != 'continuous' or
                args.selection != 'fixed' or args.k not in (None, 1) or
                args.access_budget is not None or args.facility_cost is not None or
                args.algorithm is not None):
            raise ValueError('absolute center requires fixed k=1 and continuous candidates')
    elif args.candidate_domain == 'continuous':
        raise ValueError('continuous candidates require absolute_center')
    # Omitted tuning options stay None here and in the JSON settings. The solver
    # applies defaults only where they are used and refuses options that have no
    # effect; group diagnostics distinguish configuration from execution.
    net = build_network(args.network, layer=args.layer, target_crs=args.crs,
                        snap=args.snap, node=args.node_crossings,
                        split_shared_vertices=not args.no_shared_vertex_noding,
                        keep_largest_component=args.keep_largest_component)
    data = gpd.read_file(args.demand, layer=args.demand_layer)
    if data.empty or data.crs is None:
        raise ValueError('demand must contain points and a CRS')
    if data.geometry.isna().any() or data.geometry.is_empty.any() or not (
        data.geom_type == 'Point'
    ).all():
        raise ValueError('all input demand rows must be nonempty points')
    data = data.to_crs(net.crs)
    xy = np.column_stack((data.geometry.x, data.geometry.y))
    weights = (np.ones(len(data)) if args.weight_field is None else
               data[args.weight_field].to_numpy(dtype=float))
    if (not np.isfinite(weights).all() or np.any(weights < 0) or
            not np.isfinite(weights.sum())):
        raise ValueError('weights must be finite and nonnegative')
    if args.group_by is not None and data[args.group_by].isna().any():
        raise ValueError('group labels must not be missing')
    groups = (np.full(len(data), 'all', dtype=object) if args.group_by is None else
              data[args.group_by].astype(str).to_numpy())
    base_n = net.n_nodes
    largest_anchor = None
    if args.component_policy == 'largest':
        base_context = RoutingContext(
            net.csr, physical_edges=(net.edge_u, net.edge_w, net.edge_len),
            max_temp_mb=args.max_temp_mb, max_matrix_mb=args.max_matrix_mb,
        )
        largest_anchor = int(np.flatnonzero(base_context.labels == base_context.largest)[0])
        del base_context
    candidate_xy = np.empty((0, 2), dtype=float)
    if args.candidate_points is not None:
        sites = gpd.read_file(args.candidate_points, layer=args.candidate_layer)
        if (sites.empty or sites.crs is None or sites.geometry.isna().any() or
                sites.geometry.is_empty.any() or not (sites.geom_type == 'Point').all()):
            raise ValueError('candidate file must contain nonempty points with a CRS')
        sites = sites.to_crs(net.crs)
        candidate_xy = np.column_stack((sites.geometry.x, sites.geometry.y))
    all_xy = np.vstack((xy, candidate_xy))
    snaps = None
    aug = None
    if args.snap_mode == 'edge':
        snaps = snap_to_edges(net, all_xy, max_distance=args.max_snap,
                              endpoint_tolerance=args.snap)
        aug = augment_network(net, snaps, endpoint_tolerance=args.snap)
        net, all_nodes, all_distances = aug.network, aug.demand_nodes, snaps.distance
    else:
        _geometry(net, endpoint_tolerance=args.snap)
        all_nodes, all_distances = snap_points(net, all_xy, max_dist=args.max_snap)
    nodes, distances = all_nodes[:len(xy)], all_distances[:len(xy)]
    candidate_nodes = all_nodes[len(xy):] if args.candidate_points else None
    if candidate_nodes is not None and np.any(candidate_nodes < 0):
        raise ValueError('at least one restricted candidate failed snapping')
    ok = nodes >= 0
    if args.max_snap is None and np.any(distances[ok] > 1000):
        print('warning: demand snap exceeds 1 km; inspect snap audit', file=sys.stderr)
    if not ok.all() and args.on_unsnapped == 'error':
        raise ValueError(f'{np.count_nonzero(~ok)} demand rows exceeded snap limit')
    if not ok.any():
        raise ValueError('no demand rows were attached')
    if set(groups) != set(groups[ok]):
        raise ValueError('at least one input group has no attached demand')
    row_ids = np.flatnonzero(ok)
    physical_edges = (net.edge_u, net.edge_w, net.edge_len)
    if args.objective == 'absolute_center':
        from net_center.absolute_groups import solve_absolute_groups

        results = solve_absolute_groups(
            net, nodes[ok], groups[ok], weights=weights[ok], offsets=distances[ok],
            component_policy=args.component_policy, include_offsets=args.include_offsets,
            max_centers=args.max_centers, max_temp_mb=args.max_temp_mb,
            max_matrix_mb=args.max_matrix_mb, physical_edges=physical_edges,
            endpoint_tolerance=args.snap, largest_component_anchor=largest_anchor,
        )
    else:
        results = solve_groups(
            net.csr, nodes[ok], groups[ok], weights=weights[ok], offsets=distances[ok],
            objective=args.objective, selection=args.selection,
            k=(args.k if args.k is not None else 1) if args.selection == 'fixed' else None,
            max_centers=args.max_centers, component_policy=args.component_policy,
            candidate_domain=args.candidate_domain, candidate_nodes=candidate_nodes,
            include_offsets=args.include_offsets, penalty=args.penalty,
            access_budget=args.access_budget, facility_cost=args.facility_cost,
            gap_replicates=args.gap_replicates, seed=args.seed,
            min_members=args.min_members,
            max_temp_mb=args.max_temp_mb, max_matrix_mb=args.max_matrix_mb,
            prune=False if args.no_prune else None,
            coordinates=net.node_xy if args.selection == 'gap' else None,
            reference_nodes=np.arange(base_n) if args.selection == 'gap' else None,
            physical_edges=physical_edges, algorithm=args.algorithm,
            largest_component_anchor=largest_anchor,
            base_vertex_count=base_n if args.candidate_domain == 'vertices' else None,
            max_combinations=args.max_combinations, swap_passes=args.swap_passes,
        )
    records = []
    for group, (rows, result) in results.items():
        if args.objective == 'absolute_center':
            record = result
            for location in record['center_locations']:
                edge = location['edge']
                if edge is not None:
                    location['original_edge'] = (int(aug.original_edge[edge])
                                                 if aug is not None else edge)
                    location['original_offset'] = (location['edge_offset'] +
                        (float(aug.original_start[edge]) if aug is not None else 0.))
        else:
            record = asdict(result)
            record['center_xy'] = net.node_xy[result.centers]
        full_weight = float(weights[groups == group].sum())
        record['group'] = group
        record['input_row_ids'] = row_ids[rows]
        record['full_input_weight'] = full_weight
        record['full_input_coverage'] = record['modeled_weight']/full_weight
        record['full_input_row_coverage'] = (
            float(np.count_nonzero(record['assignment'] >= 0)) /
            int(np.count_nonzero(groups == group))
        )
        records.append(record)
    assignment_status = np.full(len(data), 'unsnapped', dtype=object)
    for record in records:
        input_ids = record['input_row_ids']
        assignment_status[input_ids] = np.where(
            record['assignment'] >= 0, 'assigned', 'excluded_component'
        )
    audit = []
    for i in range(len(data)):
        item = {'row': i, 'group': groups[i], 'weight': weights[i],
                'node': int(nodes[i]), 'distance': distances[i],
                'accepted': bool(ok[i]), 'status': assignment_status[i], 'input_xy': xy[i]}
        if snaps is not None:
            item.update(original_edge=int(snaps.edge[i]), edge_offset=snaps.offset[i],
                        snapped_xy=snaps.xy[i])
        elif ok[i]:
            item['snapped_xy'] = net.node_xy[nodes[i]]
        audit.append(item)
    candidate_audit = [
        {'row': i, 'input_xy': candidate_xy[i], 'node': int(node),
         'distance': all_distances[len(xy)+i], 'snapped_xy': net.node_xy[node]}
        for i, node in enumerate(candidate_nodes if candidate_nodes is not None else [])
    ]
    _publish(args.out, {'schema': 'net-center-adaptive/5', 'crs': str(net.crs),
                        'settings': {key: str(value) if isinstance(value, Path) else value
                                     for key, value in vars(args).items()},
                        'candidate_audit': candidate_audit,
                        'distance_unit': 'metre', 'snap_mode': args.snap_mode,
                        'input_rows': len(data), 'input_weight': float(weights.sum()),
                        'attached_rows': int(ok.sum()),
                        'attached_weight': float(weights[ok].sum()),
                        'snap_audit': audit,
                        'groups': records})
    print(f'Wrote {args.out}')


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        run(args)
    except (ValueError, RuntimeError, OSError, KeyError, MemoryError, ImportError) as exc:
        print(f'net-center adaptive: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
