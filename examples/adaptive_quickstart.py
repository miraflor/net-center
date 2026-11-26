"""Numerical adaptive example; no GIS dependencies or external data required."""

from scipy.sparse import csr_matrix

from net_center.adaptive import solve_adaptive, solve_groups


def main():
    # Two disconnected paths of three nodes each, with lengths in metres.
    graph = csr_matrix([
        [0., 10., 0., 0., 0., 0.],
        [10., 0., 10., 0., 0., 0.],
        [0., 10., 0., 0., 0., 0.],
        [0., 0., 0., 0., 5., 0.],
        [0., 0., 0., 5., 0., 5.],
        [0., 0., 0., 0., 5., 0.],
    ])
    demand = [0, 2, 3, 5]
    weights = [3., 1., 2., 1.]
    result = solve_adaptive(
        graph, demand, weights=weights, k=3, candidate_domain="demand",
        component_policy="allocate", algorithm="exact",
    )
    # A third site saves 20 weighted metres on the first path, versus 10 on
    # the second. The heavier node on the second path receives its one site.
    if result.centers.tolist() != [0, 2, 3] or result.network_cost != 10.:
        raise RuntimeError("Unexpected exact allocation")
    print("Fixed centers:", result.centers, "site weights:", result.site_weight)
    print("Mean access:", result.mean_access)
    adaptive = solve_adaptive(
        graph, demand, weights=weights, k=None, selection="balanced",
        candidate_domain="demand", component_policy="allocate", max_centers=4,
    )
    print("Balanced centers:", adaptive.centers)
    groups = solve_groups(
        graph, demand, ["west", "west", "east", "east"], weights=weights,
        candidate_domain="demand",
    )
    for label, (rows, group) in groups.items():
        print(label, "rows:", rows, "center:", group.centers)


if __name__ == "__main__":
    main()
