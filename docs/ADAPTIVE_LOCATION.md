# Adaptive network location and representation

This additive extension supplies weighted network median and representative-site
optimization. It does not change the existing `net-center` CLI or the exact
absolute 1-center implementation. A new wrapper exposes that same solver for
continuous group/component minimax problems with optional connector costs.
Its separate CLI is:

```bash
python -m net_center.adaptive_cli --help
python -m net_center.adaptive_cli roads.gpkg --demand points.gpkg \
  --weight-field population --group-by region --max-snap 500 \
  --selection balanced --candidate-domain demand --component-policy allocate \
  --max-centers 12 --out representatives.json
```

The network is undirected, weighted by positive lengths in metres. For the GIS
entry point all input demand rows must be nonempty points, with a declared CRS.
No geometry is downloaded; supply a prepared line layer. Nearest-edge snapping
is the new command's default. The old command retains its original defaults.

## Python API

```python
from net_center.graph import build_network
from net_center.edge_snap import snap_to_edges, augment_network
from net_center.adaptive import solve_adaptive

net = build_network('roads.gpkg', keep_largest_component=False)
# XY must be in net.crs; values here are illustrative.
xy = [[305100, 1615000], [305250, 1615200]]
snaps = snap_to_edges(net, xy, max_distance=500)
if not snaps.accepted.all():
    raise ValueError('unattached demand')
aug = augment_network(net, snaps)
r = solve_adaptive(aug.network.csr, aug.demand_nodes,
                   weights=[1200, 800], offsets=snaps.distance,
                   k=None, selection='balanced',
                   component_policy='allocate', candidate_domain='vertices',
                   max_centers=12)
print(r.centers, r.site_weight, r.mean_access)
```

`solve_groups(graph, demand_nodes, groups, weights=..., offsets=..., **options)`
returns a dictionary of group labels mapped to original input row indices and
`AdaptiveResult`. Labels must be nonempty strings. Each group uses the same validated routing graph and cached component
subgraphs and has its own center cap. With `base_vertex_count=N`, the
`vertices` domain is restricted to the first N original vertices and that
group's own accepted demand vertices. GIS supplies this argument automatically.
Without it, numerical callers declare the supplied graph as their candidate
universe. `solve_adaptive` accepts `vertex_candidates` for a restricted
vertex universe; `candidate_nodes` is reserved for the `restricted` domain. A failed group raises rather than silently emitting partial results.

## What is optimized?

Given demand nodes q_i and nonnegative weights w_i, a set S of sites minimizes
`J(S) = sum_i w_i min_{s in S} d(q_i,s)`. Candidate domains are `vertices`
(original and group-own demand vertices in GIS, all supplied vertices by
default in numerical calls), `demand` (distinct observed locations), or
`restricted` (explicit candidate IDs in Python or a GIS candidate point layer). The latter two do not
claim to optimize over every road vertex. For k=1 a connected component is
solved exactly by streaming weighted shortest-path reductions. In the
unweighted `vertex_center` problem the maximum distance is minimized instead;
all demand rows, including zero-weight rows, count for minimax. General
k-center is not implemented. The existing absolute 1-center solver remains
available directly and through the new grouped continuous minimax wrapper.

For k>1 the default heuristic greedy BUILD chooses a new site with the greatest cost decrease, then
up to `swap_passes` best-improving single replacements are made. All costs
are computed on exact network distances, but global optimality of the set is
not guaranteed by the heuristic. Bounded exhaustive enumeration is also
available for small finite candidate sets. Fixed k may yield an unused or zero-mass site, especially
when more candidates than distinct demand positions are requested. Reported
site counts and weights reveal this; it is not hidden. Ties use increasing
node IDs; swap improvements have a numerical tolerance.

### Components and coverage

- `strict`: all demand must occupy one connected component (unused components
  are irrelevant).
- `largest`: explicitly restrict to the largest component by total CSR routing
  edge weight for CSR-only calls, or total physical edge length if supplied to
  `RoutingContext` / `solve_groups`. The GIS CLI uses original physical lengths,
  including parallel physical links, before demand augmentation and pins that
rank after verifying the split inventory. Metadata must reconstruct exactly
the same minimum-parallel-edge CSR; self-loops and missing or extra pairs
are rejected. A collapsed CSR cannot prove the existence of longer parallel
physical links; numerical callers remain responsible for their provenance.
Excluded rows
  retain assignment -1 and infinite access; coverage is based on original mass.
- `per_component`: fixed k=1 per populated component (cap applies globally).
- `allocate`: a global budget distributed among populated components. At least
  one site per component is mandatory. A dynamic program minimizes the sum
  of costs across the enumerated component curves.

The package does not add links to force connectivity. All input demand rows
(including zero-weight rows) require a finite assignment unless explicitly
excluded. At least one positive weight per group is required; a zero-mass
component in a positive-mass group is handled and assigned. Because the median
objective does not depend on its site, such a component receives exactly one
site, placed where the largest access distance (network plus connector) of its
rows is smallest; `diagnostics.zero_weight_components` lists these components.
No implicit loss of mass occurs.

By default the adaptive CLI keeps every network component, so a demand point
next to a small disconnected road piece attaches to that piece. The option
`--keep-largest-component` builds the network as the legacy `net-center` CLI
does: only the component with the most road length is kept, and demand
and candidate points are snapped against that retained network. Node and edge IDs then refer to that
reduced network, which is the same network the legacy CLI builds from the
same input and topology options. This is not `--component-policy largest`,
which attaches demand first and then excludes rows outside the largest
component. Pre-filtering can increase connector distances and does not
override `--max-snap`. The CLI refuses unsnapped rows by default; `--on-unsnapped
exclude` explicitly permits exclusion but still refuses a completely absent
or zero-modeled-mass group. The JSON has both attached and full-input coverage.

### Site count policies

- `fixed`: give integer `k` (default 1). Exact only for the single-site
  subproblems by default. Use `algorithm="exact"` for bounded exhaustive
  enumeration on small problems. With `allocate`, k is global within a group.
- `balanced`: give `k=None` in Python; omit `--k` at the CLI. For C populated
  components and n distinct modeled demand vertices, choose the evaluated K
  minimizing `J(K)/J(C) + penalty * (K-C)/max(1,n-C)`. The default penalty is 1.
  If the baseline loss is zero, its normalized term is zero. This is an
  explicit modeling criterion, not a statistical theorem. Smaller K wins ties.
- `access_budget`: choose the first evaluated K for which the weighted mean
  network-plus-connector access distance is within the stated budget. The
  budget always includes connectors. Comparisons use a documented numerical
  tolerance of `1e-12 * max(1, budget)`, and the final routed result is checked.
  Failure means no evaluated feasible
  solution was found, not a proof of global infeasibility.
- `penalized`: minimize `J(K) + facility_cost * K` over evaluated allocations.
  Facility cost must be in the same units as the weighted network objective
  (weight-metres for the GIS API). A fixed connector sum does not change K.
  `objective` is the location objective, not the selection score; the
  network-only penalized criterion is separately returned as
  `diagnostics.penalized_selection_cost`.
- `gap`: use `candidate_domain='demand'`. Fit weighted medoids to observed
  demand and to replicated draws with replacement from component vertices;
  when coordinates are provided, restrict reference vertices to the bounding
  box of the component's observed demand coordinates. `reference_nodes` can
  restrict the reference to explicit graph vertices; the GIS CLI uses original
  pre-augmentation vertices so other groups do not change the null support.
  A sparse original graph may supply fewer than three eligible reference nodes;
  the audit then records insufficient support and retains one site. Preserve the individual
  row weights across draws; multiplicities are coalesced. The dispersion is
  the weighted sum of unsquared nearest-medoid network distances (power 1),
  not the squared Euclidean within-cluster dispersion of the original Gap
  formulation. See the `d.power` discussion in the R `cluster`
  [clusGap documentation](https://search.r-project.org/CRAN/refmans/cluster/html/clusGap.html).
  This is a network-medoid variant, not a reproduction of R `clusGap`.
  Use the smallest k
  satisfying the one-standard-error rule, or report `upper_bound_reached`.
  A common cost floor max(float64 tiny, J(1)*1e-12) makes logs finite; it is an
  explicit numerical convention and can affect zero-cost cases. Degenerate
  observations or insufficient support retain one site with an audit outcome.
  An inadmissible selected split (child count below `min_members`) retains one
  site with a different audit outcome. Each component is selected independently;
  exceeding the global cap fails rather than silently overriding the rule.
  The null samples vertices, not uniform network length. It is sensitive to
  vertex density, extent, reference distribution, seed and approximation.
  References are not downsampled: memory-gate failure is explicit.
  `diagnostics.gap_reference` describes the configured reference support: `component_vertices`
  or `explicit_nodes`, with the suffix `_in_demand_bounding_box` when
  coordinates restrict it. It is null for other selection policies.
  `gap_by_component` records the actual pool size and outcome;
  `execution_by_component` reports the number of reference replicates
  actually generated (zero for degenerate/skipped Gap tests).

## Distance and snapping semantics

Each projected observation is mapped to the closest physical LineString;
exact ties prefer the smallest edge ID. Every accepted interior offset is
inserted once on that physical edge. Distinct crossing edges remain separate;
parallel edges stay physically distinct and CSR routing uses the minimum
cost per pair. Original edge IDs and start measures are available through
`Augmentation`. The original Network is not mutated. Distances among original
vertices are preserved, within floating-point tolerance. Geometric lengths
must match routing edge lengths; this version does not generalize costs or
model one-way travel. Coordinates on the pre-existing node merge tolerance
may differ slightly from physical edge endpoints, as in the original graph.
Ordered geometry endpoints are checked against the declared nodes using
`endpoint_tolerance` (default 0.1 m; CLI uses `--snap`). Transitive merges
require a verifiable retained endpoint chain if direct tolerance fails. A
better reversed correspondence is rejected; if the original source chain
cannot be established, the validator fails rather than inventing a repair.
Snap coordinates are checked against their edge and offset before splitting;
CSR and geometry consistency are rechecked afterward. The absolute wrapper
uses the same validator and the actual augmented edge inventory.

Connector offsets do not affect median location, but may affect vertex minimax
if `include_offsets=True`. For medians the option adds the constant connector
cost to the reported objective. Access diagnostics always include connectors.
For continuous minimax use `solve_absolute_groups` or the CLI
`--objective absolute_center`. This permits fixed one-center-per-component
problems, not general k-center. The absolute CLI rejects explicitly supplied
multi-site, Gap, penalty or median pruning options; it does not ignore them.
The median and vertex paths follow the same rule: an option that the selected
mode never uses is rejected, for example `--penalty` without `balanced`,
`--min-members`, `--seed` or `--gap-replicates` without `gap`,
`--max-combinations` without `--algorithm exact`, or `--algorithm`,
`--swap-passes` and `--max-combinations` when every solve has one site (fixed
k=1 or `per_component`). Omitted tuning values are null in the JSON settings;
boolean flags and other parser defaults are recorded as resolved by the CLI.
For the finite-site objectives, `diagnostics.configured_options` contains the
mode-specific defaults and supplied tuning values. This is configuration, not
a claim that each option affected execution: data, candidate counts and the
site cap can prevent a multi-site curve or reference sample from running.
`diagnostics.algorithm` is null unless an observed multi-site curve was
computed. `diagnostics.execution_by_component` is keyed by component ID and
records the executed `solver` (`streaming_median`, `vertex_minimax`,
`zero_weight_minimax`, `exact_curve` or `heuristic_curve`), `evaluated_k`
(the observed site counts actually evaluated), and `median_pruning_enabled`
(true/false for the streaming median, null otherwise). The latter reports
whether the pruning branch was enabled, not a count of candidates eliminated.
For Gap it also reports `gap_reference_replicates` (zero if no reference
sampling took place). This diagnostic describes curve evaluation, not just
the selected solution; a selected k=1 can follow a multi-site search.
The continuous-center wrapper retains its separate radius/location diagnostics.
A maximum connector offset at each duplicate
node is added to its demand row in the existing absolute-center sweep. The
wrapper verifies the final routed radius and reports physical edge ID and offset
when a center is not a node. `site_mean_access` is
NaN for zero mass and `site_max_access` is NaN for an empty site. The JSON
writer uses null for all unavailable/nonfinite values; assignment and coverage
fields distinguish exclusion. The CLI emits one complete no-clobber JSON
file, with stable source row indices, graph node IDs, CRS and center positions.
It requires a filesystem supporting hard links for atomic publication.

## Computation and limits

The single-site streaming median uses a safe optional triangle-inequality
bound. With a feasible reference candidate p and total positive mass W,
`J(v) >= W*d(p,v)-J(p)`. No improving vertex lies beyond `2*J(p)/W` from p;
therefore a Dijkstra cutoff of `max_i d(p,q_i)+2*J(p)/W` plus a numerical
margin preserves every possible optimum. Zero-weight rows are omitted from
the median reduction to avoid `0*inf`. Vertex minimax uses a streaming max.
The graph and component subgraphs remain in memory, and Dijkstra batches
are limited by `max_temp_mb` with a minimum of one full row. This setting
is a temporary block target, not a bound on process RSS. Multi-site methods
retain a dense unique-demand by eligible-candidate matrix per component;
`max_matrix_mb` refuses an oversized matrix. Other arrays and temporary
blocks are extra. No national-scale benchmark or performance guarantee is
claimed. Full-network multi-site optimization can be expensive; choose
`demand` candidates or an explicitly restricted set for large instances.

## Interfaces and diagnostics

```bash
# Exact small multi-median reference computation (combinatorial gate).
python -m net_center.adaptive_cli roads.gpkg --demand points.gpkg \
  --k 2 --algorithm exact --max-combinations 100000 \
  --candidate-domain demand --out exact.json

# Restrict eligible locations to an explicit point layer.
python -m net_center.adaptive_cli roads.gpkg --demand points.gpkg \
  --candidate-domain restricted --candidate-points facilities.gpkg \
  --max-snap 100 --k 2 --out restricted.json

# One exact continuous minimax location per demand-bearing component.
python -m net_center.adaptive_cli roads.gpkg --demand points.gpkg \
  --objective absolute_center --include-offsets \
  --component-policy per_component --out absolute.json

# Select sites with a user-defined facility cost in weight-metres.
python -m net_center.adaptive_cli roads.gpkg --demand points.gpkg \
  --selection penalized --facility-cost 1000 \
  --candidate-domain demand --out penalized.json
```

The adaptive CLI accepts the original topology options `--snap`,
`--node-crossings`, `--no-shared-vertex-noding` and `--crs`. Candidate points
are attached in the same augmentation as demand, with rejection of any
unsnapped candidate. Multiple candidates at a single node collapse to one.
Candidates restrict location only; they are not additional demand and their
connector offsets do not enter the demand objective.

The exact multi-site mode enumerates candidate subsets for every evaluated
component k. The sum of binomial counts must not exceed `max_combinations`
per component; the check precedes matrix allocation. Component DP then gives
an exact allocation across exact cost curves. For adaptive selection this
means exact costs for the evaluated choices, not a universal optimal k.
The heuristic and exact modes use the same finite candidate domain.
Gap uses heuristic curves only, on both observed and reference samples.

JSON schema `net-center-adaptive/5` includes settings, candidate audits,
row-level assignment statuses (`assigned`, `excluded_component`, `unsnapped`),
full input mass and row coverage, and weighted 50/90/95/99 access percentiles.
Percentiles are inverse weighted empirical CDF values among modeled positive
mass; maximum access includes zero-weight modeled rows. `assignment` indexes
the group center list. For continuous locations use `center_locations`;
node and edge identifiers are mutually exclusive. Original edge lineage is
included in the CLI for interior edge centers. No output is published until
all groups have succeeded. Median curve costs are network sums, whereas vertex minimax curve costs are
network or connector-inclusive maximum radii according to `include_offsets`.
The diagnostic `curve_cost` states which is used. Finite-site solutions are
verified against their evaluated curve after routing. The JSON settings describe
the invocation and parser defaults, not a cryptographic fingerprint of input
source files. Use `python -m net_center.adaptive_cli --help` for documented
arguments, defaults and their meanings.

## Mathematical references

- Hakimi, S. L. (1964), *Optimum Locations of Switching Centers and the
  Absolute Centers and Medians of a Graph*, Operations Research 12(3),
  450–459. https://doi.org/10.1287/opre.12.3.450
- Hakimi, S. L. (1965), *Optimum Distribution of Switching Centers in a
  Communication Network and Some Related Graph Theoretic Problems*,
  Operations Research 13(3), 462–475. https://doi.org/10.1287/opre.13.3.462
- Teitz, M. B. and Bart, P. (1968), *Heuristic Methods for Estimating the
  Generalized Vertex Median of a Weighted Graph*, Operations Research 16(5),
  955–961. https://doi.org/10.1287/opre.16.5.955
- Kaufman, L. and Rousseeuw, P. J. (1990), *Finding Groups in Data*, Wiley.
- Tibshirani, R., Walther, G. and Hastie, T. (2001), *Estimating the Number
  of Clusters in a Data Set via the Gap Statistic*, JRSS B 63(2), 411–423.
  https://doi.org/10.1111/1467-9868.00293
- R `cluster`, [`clusGap` documentation](https://search.r-project.org/CRAN/refmans/cluster/html/clusGap.html),
  especially `d.power` and `Tibs2001SEmax`.

These are established formulations and heuristics. The specific normalized
selection criterion and network reference distribution are implementation
choices, not claims of new facility-location theorems.
