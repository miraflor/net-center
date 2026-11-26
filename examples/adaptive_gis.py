"""GIS adaptive example with in-memory geometry; requires the GIS extras."""

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

from net_center.absolute_groups import solve_absolute_groups
from net_center.adaptive import solve_adaptive
from net_center.edge_snap import augment_network, snap_to_edges
from net_center.graph import build_network


def main():
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (10, 0)])], crs="EPSG:32651",
    )
    net = build_network(roads, keep_largest_component=False)
    # Observations 2 m above and 1 m below an edge, projected to x=2 and x=8.
    snaps = snap_to_edges(net, [[2., 2.], [8., -1.]], max_distance=3.)
    if not snaps.accepted.all():
        raise RuntimeError("Unexpected rejected demand")
    aug = augment_network(net, snaps)
    result = solve_adaptive(
        aug.network.csr, aug.demand_nodes, weights=[2., 1.],
        offsets=snaps.distance, include_offsets=True,
    )
    if not np.isclose(result.objective, 11.):
        raise RuntimeError("Unexpected median objective")
    print("Median:", aug.network.node_xy[result.centers], result.objective)
    groups = solve_absolute_groups(
        aug.network, aug.demand_nodes, ["all", "all"],
        offsets=snaps.distance, include_offsets=True,
    )
    print("Continuous minimax result:", groups)


if __name__ == "__main__":
    main()
