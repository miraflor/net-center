# net-center

**Network location optimization and adaptive spatial representation.**

`net-center` is a Python library and command-line toolkit for locating facilities or representative sites on undirected spatial networks. It uses shortest-path distance rather than straight-line distance: barriers, road connectivity and circuitous routes matter. Demand can represent people, establishments, settlements, observations or other weighted locations.

The original single-facility solvers remain available unchanged. The adaptive workflow adds grouped demand, continuous edge attachment, multiple medians, automatic site counts, explicit disconnected-component policies and assignment diagnostics.

## Capabilities and guarantees

| Problem | Eligible locations | Guarantee |
|---|---|---|
| Weighted 1-median | Network vertices, demand vertices or restricted candidates | Exact for the specified finite candidate set |
| Vertex 1-center | Eligible network vertices | Exact minimax, unweighted |
| Absolute 1-center | Anywhere along physical network edges | Exact up to numerical tolerance |
| Weighted network k-median | Network vertices or restricted candidates | BUILD/SWAP heuristic; bounded exhaustive enumeration for small cases |
| Weighted network k-medoids | Distinct demand vertices | Same finite-candidate algorithms |
| Component allocation | Evaluated component site-count curves | Exact dynamic programming over those curves |
| Adaptive representation | Evaluated solutions | Explicit selection criterion, not a universal optimality claim |

The absolute-center wrapper solves one continuous minimax location per group or populated component, not a general continuous k-center problem. Exactness is relative to the supplied network, candidate universe, modeled demand and floating-point tolerance.

## Install

From a clone of the repository:

```bash
python -m pip install -e .          # numerical core: NumPy, SciPy, Joblib
python -m pip install -e ".[gis]"    # optional GIS tools
python -m pip install -e ".[dev]"    # development and tests
```

Python 3.10–3.13 are tested in CI. GIS workflows require Shapely, GeoPandas and PyProj. The core numerical API does not require the GIS extras. The distribution is `net-center`, import package `net_center`, and installed console command `net-center`.

## Quick start: exact single-center location

The original command computes the weighted 1-median, vertex center and absolute center (default demand is every network node).

```bash
net-center roads.gpkg
net-center roads.gpkg --demand points.gpkg --weight-field population \
  --max-snap 500 --out centers.gpkg
```

Use `net-center --help` for topology, parallel execution and memory options. This workflow attaches demand to the nearest existing node, retains the largest road-length component, accepts point or polygon representative-point demand, and may exclude rows beyond the snap threshold with warnings. Connector distances are not part of its objective.

```python
from net_center import build_network, snap_points, solve

net = build_network("roads.gpkg")
# Coordinates must already be in net.crs (metres).
nodes, distances = snap_points(net, [[305100, 1615000], [305250, 1615200]], max_dist=500)
if (nodes < 0).any():
    raise ValueError("Unattached demand")
results = solve(net, demand_nodes=nodes, weights=[1200, 800])
print(results["median"], results["absolute_center"])
```

See [the single-center algorithm guide](docs/ALGORITHMS.md) and [the original quickstart](examples/quickstart.py).

## Quick start: adaptive network location

Use the separate module CLI (no new console entry point is installed). Input is a prepared line network and a nonempty point demand layer with a declared CRS. Output is a new JSON file; existing output paths are refused.

```bash
# Independent demand groups, choose representative medoids automatically.
python -m net_center.adaptive_cli roads.gpkg --demand points.gpkg \
  --weight-field population --group-by region --max-snap 500 \
  --selection balanced --candidate-domain demand \
  --component-policy allocate --max-centers 12 --out representatives.json

# Fixed k, bounded exact reference calculation on small instances.
python -m net_center.adaptive_cli roads.gpkg --demand points.gpkg \
  --candidate-domain demand --k 2 --algorithm exact \
  --max-combinations 100000 --out exact.json

# Exact continuous minimax center per populated component.
python -m net_center.adaptive_cli roads.gpkg --demand points.gpkg \
  --objective absolute_center --include-offsets \
  --component-policy per_component --out continuous.json
```

Run `python -m net_center.adaptive_cli --help` for all options. For a legacy-equivalent network and node attachment, use `--snap-mode node --keep-largest-component`; compare the same demand rows, weights, thresholds and candidate domain, and do not include connectors in the objective. The legacy and adaptive outputs have different formats and exclusion defaults.

### Numerical Python API

The adaptive API is available from its submodules, not as new top-level exports. For example, a four-node path with weighted demand at its ends:

```python
from scipy.sparse import csr_matrix
from net_center.adaptive import solve_adaptive

G = csr_matrix([[0., 10., 0., 0.], [10., 0., 10., 0.],
                [0., 10., 0., 10.], [0., 0., 10., 0.]])
r = solve_adaptive(G, [0, 3], weights=[2., 1.],
                   candidate_domain="demand", k=2, algorithm="exact")
print(r.centers, r.site_weight, r.mean_access)
```

Use `solve_groups` for independent groups with a shared graph; pass `base_vertex_count` for group-specific original-plus-own-demand vertex candidates. Use `net_center.edge_snap.snap_to_edges` and `augment_network` to attach points to physical edges, and `net_center.absolute_groups.solve_absolute_groups` for continuous minimax. See [adaptive API and CLI details](docs/ADAPTIVE_LOCATION.md), [numerical example](examples/adaptive_quickstart.py) and [GIS example](examples/adaptive_gis.py).

## Mathematical objectives

Let $d_G(q,s)$ be shortest-path distance, $w_i \ge 0$ demand weight, $q_i$ an attached demand vertex, and $S$ a set of eligible sites.

**Weighted network median:** minimize

```math
J(S)=\sum_i w_i\min_{s\in S}d_G(q_i,s).
```

For one site on a connected component, an optimum over a finite candidate set is found exactly. With unrestricted network vertices, classical vertex optimality for the median applies to vertex demand. Multiple finite-candidate sites use greedy BUILD and bounded best-improving SWAP by default; bounded exact enumeration is available with `algorithm="exact"`. The heuristic is not guaranteed globally optimal.

**Vertex minimax:** minimize $\max_i d_G(q_i,s)$ over eligible vertices. **Absolute minimax:** minimize the same maximum over any point of a network edge. On an edge of length $L$ with endpoints $u,v$, a demand's distance to offset $t$ is

```math
\min\{d_G(q_i,u)+t,\ d_G(q_i,v)+L-t\}.
```

The exact edge sweep finds the minimum of the upper envelope using sorted breakpoints, prefix/suffix maxima and safe lower bounds. Demand is finite, not distributed continuously along the entire network. Adaptive edge attachment inserts the finite interior demand positions as vertices.

**Connectors:** the adaptive workflow records each observation's Euclidean attachment offset $a_i$. Access is $a_i+\min_s d_G(q_i,s)$. For medians, $\sum_i w_i a_i$ is constant for a fixed modeled demand set; `include_offsets=True` adds it to the reported objective. For minimax, offsets may change the optimal location, and enter the objective only when explicitly enabled. Access diagnostics and the access-budget criterion always include them. Demand weights do not weight minimax radii; zero-weight rows still count for minimax and coverage.

## Candidate domains and grouping

| Domain | Meaning |
|---|---|
| `vertices` | All supplied vertices in the numerical API; original plus each group's own inserted demand vertices in the GIS CLI |
| `demand` | Distinct observed attached demand vertices (medoids) |
| `restricted` | Explicit candidate node IDs or snapped candidate point layer |
| `continuous` | Absolute 1-center GIS objective only |

Group labels identify independent problems. Site budgets apply per group, not across groups. Other groups' inserted vertices may be used for routing but are not automatically eligible facilities. Candidate points are not extra demand. The solver validates finite input, candidate reachability, positive modeled group mass and memory limits.

## Demand attachment, topology and components

Both workflows use undirected positive-length networks and projected metre units. Input topology is preferred over geometric crossings; planar noding is opt-in. One-way streets, turn restrictions and arbitrary travel costs are not modeled. See [network assumptions](docs/NETWORK_ASSUMPTIONS.md).

| Behavior | Original CLI | Adaptive CLI |
|---|---|---|
| Default attachment | Nearest existing node | Nearest physical edge, split at accepted interior positions |
| Input demand | Optional point/polygon layer | Required point layer |
| Network retention | Largest road-length component | All components unless `--keep-largest-component` |
| Unsnapped rows | Exclude with warning | Error unless `--on-unsnapped exclude` |
| Output | Three single-center results, optional GIS output | Complete group/assignment JSON |

The adaptive component policies are `strict` (one populated component), `largest` (exclude rows outside the largest component after snapping), `per_component` (fixed one site per populated component), and `allocate` (global site budget per group, at least one site per populated component). In GIS, largest means original physical road length; CSR-only numerical calls default to collapsed routing edge weights. `--keep-largest-component` filters the graph before snapping and can increase connector distances. It is not equivalent to post-snap exclusion. No links are invented.

In a positive-mass median group, an entirely zero-weight component receives exactly one site chosen by secondary minimax access; this is a coverage convention, not a weighted-median theorem. A fixed-k request may consequently be infeasible. All-zero modeled groups fail.

## Selecting the number of sites

| Selection | Criterion |
|---|---|
| `fixed` | Specified k (default 1) |
| `balanced` | Network loss relative to the one-per-component baseline plus a normalized site-count penalty |
| `access_budget` | Smallest evaluated site count satisfying weighted mean access, including connectors |
| `penalized` | Network loss plus user-defined facility cost times k |
| `gap` | One-standard-error Gap rule with a network-medoid reference model |

For adaptive policies pass `k=None` in Python and omit `--k` on the CLI. These are median policies; vertex and absolute minimax support fixed one-site subproblems only. Gap requires demand candidates and heuristic curves. It uses **unsquared** weighted network-medoid loss, draws from component vertices (not uniform network length), preserves row weights and records the reference support and outcomes. The cap and admissibility checks may prevent a split. See [the full mathematical and parameter description](docs/ADAPTIVE_LOCATION.md). Failure under an access budget means no evaluated solution met it, not a proof of global infeasibility.

## Results and diagnostics

The adaptive JSON schema `net-center-adaptive/5` records center locations, source row indices, assignments, coverage and exclusions, site weights/counts, mean and maximum access, weighted P50/P90/P95/P99, evaluated cost curves and selection diagnostics. Assignments index the group's center list; unavailable/nonfinite values become JSON null. Excluded and unsnapped rows are identified. Weighted percentiles use modeled positive mass; maximum access includes zero-weight rows. `settings` is invocation configuration, `diagnostics.configured_options` resolves tuning defaults for finite-site objectives, and `diagnostics.execution_by_component` records observed solver paths and Gap replicates. The continuous wrapper has separate diagnostics. Output is published only after all groups succeed and requires hard-link support for no-clobber publication.

## Performance and limitations

- The original solvers construct a demand-by-node matrix; `--float32`, `--max-temp-mb`, `--max-cells`, `--jobs` and `--backend` are legacy controls. Absolute edge sweeps use safe pruning. See [algorithms](docs/ALGORITHMS.md).
- Adaptive single-site median and vertex minimax use streaming shortest paths; median pruning uses a triangle-inequality bound. `--max-temp-mb` targets routing blocks, not total RSS. Multi-site curves retain a dense unique-demand-by-candidate matrix within `--max-matrix-mb`; exhaustive enumeration has a combinatorial gate. No adaptive national-scale performance guarantee is claimed.
- Not implemented: directed/asymmetric/time-dependent costs, turn restrictions, general weighted or multi-facility minimax, continuous demand density along roads, capacities, out-of-core distance matrices, automatic OSM acquisition, or arbitrary multimodal cost models. The GIS solver uses positive physical lengths in metres. No network barycenter or squared-distance Fréchet mean solver is provided.

## Documentation and architecture

| Resource | Scope |
|---|---|
| [Adaptive location](docs/ADAPTIVE_LOCATION.md) | Detailed API, selection, safeguards, output schema and examples |
| [Algorithms](docs/ALGORITHMS.md) | Original mathematical derivations plus adaptive methods |
| [Architecture](docs/ARCHITECTURE.md) | Both execution paths and module boundaries |
| [Network assumptions](docs/NETWORK_ASSUMPTIONS.md) | Topology, units, demand and component semantics |
| [Technical note (PDF)](docs/TECHNICAL_NOTE.pdf) · [LaTeX source](docs/TECHNICAL_NOTE.tex) | Unified scientific treatment |
| [Validation](docs/VALIDATION.md) | Tests, CI, numerical limits and reproducibility |
| [Review notes](docs/REVIEW_NOTES.md) | Historical review of the first public release |
| [Changelog](CHANGELOG.md) | Release history and unreleased expansion |
| [Development guide](GITHUB_SETUP.md) | Safe contribution and release workflow |

```text
net_center/                 numerical and GIS implementation
  center.py, solve.py       original exact single-center solvers
  distances.py, topology.py sparse routing and adjacency
  graph.py, cli.py          original GIS network and CLI
  adaptive.py              grouped finite-candidate location and selection
  edge_snap.py             nearest-edge attachment and augmentation
  absolute_groups.py       grouped continuous minimax wrapper
  adaptive_cli.py          adaptive GIS workflow and JSON publication
examples/                   single-center, adaptive numerical and GIS examples
tests/                      original and adaptive regression suites
docs/                       technical and user documentation
.github/workflows/ci.yml    cross-version CI and installed-wheel smoke tests
```

The earlier detailed single-center README and technical note remain available in the [historical revision](https://github.com/miraflor/net-center/tree/df24244e81e17b79e497b979fb3773382c87fed7); the current guide scopes those methods without discarding their derivations in `ALGORITHMS.md`.

## Validation

```bash
python -m pip install -e ".[dev]"
python -m pip check
python -m ruff check .
python -m pytest -q
```

The merged adaptive implementation recorded 201 passing tests on the [post-merge CI run](https://github.com/miraflor/net-center/actions/runs/45778162775), with Python 3.10–3.13 jobs and the wheel/source-distribution smoke test successful. That count refers to the tested revision, not a promise for every future run. See [validation](docs/VALIDATION.md) for test coverage and limits. Exact versus heuristic guarantees are deliberately distinguished.

## Scientific references

- Hakimi, S. L. (1964). Optimum Locations of Switching Centers and the Absolute Centers and Medians of a Graph. *Operations Research*, 12(3), 450–459. https://doi.org/10.1287/opre.12.3.450
- Hakimi, S. L. (1965). Optimum Distribution of Switching Centers in a Communication Network and Some Related Graph Theoretic Problems. *Operations Research*, 13(3), 462–475. https://doi.org/10.1287/opre.13.3.462
- Kariv, O., & Hakimi, S. L. (1979). An Algorithmic Approach to Network Location Problems. I: The p-Centers. *SIAM Journal on Applied Mathematics*, 37(3), 513–538. https://doi.org/10.1137/0137040
- Handler, G. Y., & Mirchandani, P. B. (1979). *Location on Networks: Theory and Algorithms*. MIT Press.
- Teitz, M. B., & Bart, P. (1968). Heuristic Methods for Estimating the Generalized Vertex Median of a Weighted Graph. *Operations Research*, 16(5), 955–961. https://doi.org/10.1287/opre.16.5.955
- Kaufman, L., & Rousseeuw, P. J. (1990). *Finding Groups in Data*. Wiley.
- Tibshirani, R., Walther, G., & Hastie, T. (2001). Estimating the Number of Clusters in a Data Set via the Gap Statistic. *JRSS B*, 63(2), 411–423. https://doi.org/10.1111/1467-9868.00293
- Daskin, M. S. (2013). *Network and Discrete Location*, 2nd ed. Wiley.

The algorithms are established formulations and heuristics; the specific network reference distribution, normalized selection criterion and engineering safeguards are implementation choices, not claims of new location theorems. Additional sources and software provenance are in [Algorithms](docs/ALGORITHMS.md).

## License

MIT. See [LICENSE](LICENSE).

## Author and Citation

**James Matthew Miraflor**  
[Scientific Computing Laboratory](https://scl.dcs.upd.edu.ph/)  
[Department of Computer Science](https://dcs.upd.edu.ph/)  
College of Engineering  
University of the Philippines Diliman

**net-center** is authored and maintained by James Matthew Miraflor, with AI assistance in software development, debugging, testing, and documentation. Responsibility for the project's methodological decisions, scientific validity, and published outputs rests with the author.

When citing software, record the repository URL, release or commit SHA, and the specific algorithm and settings used. This guide describes the merged adaptive code, which has not yet been assigned a new release number.
