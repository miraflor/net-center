"""Small, self-contained example that does not require a GIS file on disk."""

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

from net_center import build_network, snap_points, solve

# A simple 10 m path with one junction at x=3.
roads = gpd.GeoDataFrame(
    geometry=[
        LineString([(0, 0), (3, 0)]),
        LineString([(3, 0), (10, 0)]),
    ],
    crs="EPSG:32651",  # projected CRS with metre units
)

net = build_network(roads)

# Demand at the two ends of the road. snap_points finds their node indices from
# coordinates, so the example does not depend on how nodes are numbered.
demand_nodes, _ = snap_points(net, np.array([[0.0, 0.0], [10.0, 0.0]]))

# The best vertex centre is x=3 with radius 7 m, while the absolute centre is
# x=5 with radius 5 m.
results = solve(net, demand_nodes=demand_nodes)

for name, result in results.items():
    print(name, result)
