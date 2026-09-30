# net-center

**Exact 1-median and 1-center location on undirected spatial networks.**

`net-center` is a Python package and command-line tool for solving classical **network location problems** on road and other line-based transport networks.

Given a network and an optional set of demand locations, it computes:

| Result | Objective | Candidate location |
|---|---|---|
| **Weighted 1-median** | Minimize total weighted shortest-path distance | Network nodes |
| **Vertex 1-center** | Minimize maximum shortest-path distance | Network nodes |
| **Absolute 1-center** | Minimize maximum shortest-path distance | Anywhere on the network |

Unlike an ordinary geographic centroid, `net-center` measures distance **through the network**. Barriers, bridges, ferry links, circuitous roads, and network topology therefore affect the solution.

The package is domain-neutral: demand locations can represent people, establishments, customers, facilities, settlements, observations, or any other points for which network distance matters.

---

## Why network centers?

A coordinate centroid answers a geometric question. It does not necessarily answer an accessibility question.

Two locations may be close in straight-line distance but far apart through a road network because of rivers, coastlines, mountains, limited crossings, disconnected streets, or other topological constraints.

`net-center` instead solves location problems using the shortest-path metric $`d_G`$ induced by the network.

### Weighted 1-median

For demand locations $`i=1,\dots,k`$, non-negative weights $`w_i`$, and a candidate location $`x`$,

```math
x^* = \arg\min_x \sum_{i=1}^{k} w_i d_G(x,i).
```

This is the **minisum** objective: find the location minimizing aggregate weighted travel distance.

For vertex demand on a network, a median optimum can be chosen at a network vertex. `net-center` therefore solves this problem over network nodes.

### Vertex 1-center

```math
x^* = \arg\min_{x\in V} \max_i d_G(x,i).
```

This is the **minimax** objective restricted to network vertices. It minimizes the distance to the farthest demand location.

### Absolute 1-center

```math
x^* = \arg\min_{x\in G} \max_i d_G(x,i).
```

Here the candidate location may lie **anywhere on the network**, including in the interior of an edge.

This distinction matters: the true minimax solution need not coincide with a junction.

---

## Literature and algorithmic lineage

`net-center` implements classical network-location problems rather than proposing a new facility-location algorithm.

The mathematical core follows the literature beginning with:

- **Hakimi, S. L. (1964).** “Optimum Locations of Switching Centers and the Absolute Centers and Medians of a Graph.” *Operations Research*, 12(3), 450–459.  
  https://doi.org/10.1287/opre.12.3.450

  Hakimi introduced the absolute center and absolute median framework for weighted graphs and established the central distinction used by `net-center`: median optima may be taken at vertices, while absolute-center optima can lie inside edges.

- **Kariv, O., & Hakimi, S. L. (1979).** “An Algorithmic Approach to Network Location Problems. I: The p-Centers.” *SIAM Journal on Applied Mathematics*, 37(3), 513–538.  
  https://doi.org/10.1137/0137040

  This is the principal algorithmic reference for the continuous absolute-center problem. `net-center` uses the classical edgewise distance structure and breakpoint decomposition underlying absolute-center algorithms, while implementing the computation with vectorized prefix/suffix envelope evaluation and additional engineering optimizations.

- **Handler, G. Y., & Mirchandani, P. B. (1979).** *Location on Networks: Theory and Algorithms*. MIT Press.  
  https://mitpress.mit.edu/9780262080903/location-on-networks/

  A systematic treatment of network median and center problems and the classical edgewise formulation used in the package.

- **Hakimi, S. L. (1965).** “Optimum Distribution of Switching Centers in a Communication Network and Some Related Graph Theoretic Problems.” *Operations Research*, 13(3), 462–475.  
  https://doi.org/10.1287/opre.13.3.462

  Extends the network-location framework to the $`p`$-median problem. `net-center` currently solves only $`p=1`$.

- **Daskin, M. S. (2013).** *Network and Discrete Location: Models, Algorithms, and Applications*, 2nd ed. Wiley.  
  https://doi.org/10.1002/9781118537015

  A modern treatment of facility-location models, including the distinction between vertex and absolute center problems.

The implementation should therefore be described as a **computational implementation of classical network-location theory**, with package-specific work concentrated in GIS topology construction, numerical safeguards, vectorization, memory management, pruning, and parallel execution.

See `docs/ALGORITHMS.md` and `docs/TECHNICAL_NOTE.tex` for the detailed derivation and implementation audit.

### Provenance of each implementation step

| Step in `net-center` | Source | Relation to the source |
|---|---|---|
| Shortest-path distance matrix | Dijkstra (1959), computed with SciPy (Virtanen et al., 2020) | Published algorithm, library implementation. |
| Weighted 1-median over network nodes | Hakimi (1964): for vertex demand, a median optimum can be taken at a vertex | Direct implementation: a weighted sum over the distance matrix, then `argmin`. |
| Vertex 1-center | The node-restricted form of the minimax problem of Hakimi (1964) | Direct implementation: a row maximum over the distance matrix, then `argmin`. |
| Absolute 1-center: tent functions, sorted breakpoints, analytic minimum on each interval | Hakimi (1964); Kariv and Hakimi (1979, part I); textbook form in Handler and Mirchandani (1979) and Daskin (2013) | The interval decomposition is the published one. Evaluating it with vectorized prefix and suffix running maxima over many edges at once is an implementation choice (`docs/ALGORITHMS.md`). |
| Edge pruning with the two bounds in [Edge pruning](#edge-pruning) | Derived in that section from the triangle inequality and the edge distance formula | No published source is claimed. The bounds only discard edges that cannot improve on the best vertex center, so the optimum does not change, and the test suite checks the bounds against a sweep of every edge. |
| Demand snapped to the nearest node; duplicate demand coalesced | Modeling choice stated in [Demand semantics](#demand-semantics) | Not a location algorithm. Coalescing is exact: weights are summed for the median, and repetition does not change a maximum. |
| Road topology: shared-source-vertex noding and optional planar noding | GIS preprocessing. Planar noding uses GEOS through Shapely; `docs/ALGORITHMS.md` discusses its algorithmic background, including Bentley and Ottmann (1979) | Not a location algorithm. `docs/ALGORITHMS.md` states that no novelty is claimed for the shared-vertex rule. |
| Blocked computation, float32 storage and parallel execution | Engineering | These change memory use and speed, not the objective (see [Memory controls](#memory-controls)). |

In summary, the location methods are published, and the package-specific parts are engineering steps and the pruning bounds. None of them changes the optimization objective. The section on AI-assisted implementation work in `docs/ALGORITHMS.md` lists these parts in more detail.

---

## Installation

### Full GIS installation

```bash
python -m pip install -e ".[gis]"
```

### Development installation

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

### Numerical core only

If you already have a sparse graph and do not need GeoPandas/Shapely file I/O:

```bash
python -m pip install -e .
```

---

## Quick start

The simplest command is:

```bash
net-center roads.gpkg
```

With explicit demand locations and weights:

```bash
net-center roads.gpkg \
  --layer roads \
  --demand demand.gpkg \
  --weight-field weight \
  --max-snap 1000 \
  --out centers.gpkg
```

This computes the weighted 1-median, vertex 1-center, and absolute 1-center.

If no demand layer is supplied, every network node is used as a demand location.

---

## Python API

```python
from net_center import build_network, snap_points, solve

net = build_network("roads.gpkg")

# Coordinates must be in net.crs.
demand_xy = [
    [305000.0, 1615000.0],
    [306250.0, 1616200.0],
]

nodes, snap_distance = snap_points(
    net,
    demand_xy,
    max_dist=1000,
)

results = solve(
    net,
    demand_nodes=nodes,
    weights=[1200, 800],
)

print(results["median"])
print(results["vertex_center"])
print(results["absolute_center"])
```

The absolute-center result is either a network node or an edge plus an offset `t` in metres. `solve()` attaches the corresponding map coordinate in `.xy`.

---

## Demand semantics

Demand is represented internally at **network nodes**.

When a GIS demand layer is supplied:

1. empty geometries are removed;
2. point geometries are used directly;
3. non-point geometries are converted to representative points;
4. coordinates are transformed to the network CRS;
5. each point is snapped to its nearest network node.

This is an explicit modeling choice. Demand is **not** currently placed continuously along edges.

### Repeated demand locations

Several demand observations may snap to the same node.

`net-center` coalesces them exactly:

- for the median, supplied weights or observation counts are summed;
- for the center, duplicate copies are irrelevant because repetition does not change a maximum.

This can substantially reduce shortest-path work.

### Snap-distance control

Use:

```bash
--max-snap 1000
```

to reject demand points farther than the specified distance from the nearest network node.

If no maximum is supplied, points are still snapped, but large snap distances should be inspected carefully, especially on simplified road networks.

---

## Road-network topology

Correct topology is critical.

A geometric crossing does not necessarily imply a valid turn. A bridge, tunnel, or flyover may cross another road in two dimensions without connecting to it.

`net-center` therefore distinguishes three topology modes.

### Default: shared-source-vertex noding

```bash
net-center roads.gpkg
```

If two input LineStrings already contain the same source vertex, or a line ends on a vertex of another line or of itself, that location is treated as a genuine junction and the lines are split there. The second case covers a line that ends on one of its own interior vertices, such as a cul-de-sac turning loop drawn as one line.

This recovers common T- and X-junctions already encoded by the source data without automatically connecting every geometric crossing.

### Strict endpoint-only mode

```bash
net-center roads.gpkg --no-shared-vertex-noding
```

Use this only if the input is already segmented at every true junction.

### Explicit planar noding

```bash
net-center roads.gpkg --node-crossings
```

This treats every geometric crossing as connected.

Use it only for genuinely planar networks. On ordinary road data it can incorrectly connect a bridge to the road beneath it.

### Node identity (`--snap`)

Coordinates closer than `--snap` metres (default `0.1`) are treated as one network location, and the rule is applied transitively. The same rule decides node identity, which source vertices are shared junctions, and whether a line is a ring. A line whose two ends are one location (a closed or nearly closed ring) is split into two halves, so that it is not deleted as a self-loop. Within one line, nearby vertices are linked only when one of them is an endpoint and the line travels more than `--snap` metres between them, so a densely digitised road is not merged into one location.

---

## Coordinate systems and units

All reported network distances, edge lengths, offsets, and snapping distances are in **metres**.

- projected metre-based input is preserved;
- geographic input is reprojected to an inferred local UTM CRS unless a target CRS is supplied;
- projected input using non-metre units is reprojected;
- an explicit target CRS must be projected and metre-based.

For large or multi-zone study areas, supply a projection appropriate to the full study area rather than relying on an automatically inferred UTM zone.

---

## Shortest-path distance matrix

Let:

- $`k`$ = number of unique demand nodes,
- $`n`$ = number of network nodes.

`net-center` computes

```math
D_{ij}=d_G(q_i,v_j),
```

giving a demand-by-node matrix

```math
D\in\mathbb{R}^{k\times n}.
```

Distances are computed using Dijkstra's shortest-path algorithm through SciPy.

This matrix is then reused by the location solvers.

---

## Weighted 1-median implementation

For each candidate node $`j`$,

```math
M_j=\sum_i w_iD_{ij}.
```

The solution is

```math
j^*=\arg\min_j M_j.
```

In the implementation this is a weighted reduction over the shortest-path matrix followed by `argmin`.

When explicit weights are absent, every demand observation has unit weight.

---

## Vertex 1-center implementation

For each candidate node $`j`$,

```math
E_j=\max_iD_{ij}.
```

The vertex center is

```math
j^*=\arg\min_j E_j.
```

This is the node-restricted minimax solution.

---

## Exact absolute 1-center

Consider an edge $`(u,w)`$ of length $`L`$, and let $`t\in[0,L]`$ denote distance from endpoint $`u`$.

For demand node $`v`$,

```math
d(t,v)
=
\min\left\{
d(u,v)+t,\;
d(w,v)+L-t
\right\}.
```

Each demand therefore contributes a piecewise-linear "tent" function along the edge.

The edge eccentricity is

```math
E(t)=\max_v d(t,v).
```

The route for a demand switches between the two edge endpoints at

```math
t_v^*
=
\frac{d(w,v)-d(u,v)+L}{2}.
```

Sorting these breakpoints partitions the edge into intervals in which the upper envelope reduces to

```math
E(t)=\max\{A+t,\;B+L-t\},
```

where $`A`$ and $`B`$ are fixed on the interval.

The minimum on each interval is therefore analytic. `net-center` evaluates the necessary prefix/suffix maxima and selects the best candidate.

This is an exact continuous-edge solution for the stored shortest-path metric, modulo floating-point precision and the configured numerical tolerance.

---

## Edge pruning

Sweeping every edge is unnecessary.

The best vertex center first supplies an incumbent radius:

```math
R_V=\min_j\max_iD_{ij}.
```

Two lower bounds on the eccentricity anywhere on edge $`(u,w)`$ of length $`L`$ are then used, cheapest first. The first needs only the vertex eccentricities $`E(v)=\max_iD_{iv}`$, which are already known, and costs one operation per edge:

```math
LB'_e
=
\tfrac12\bigl(E(u)+E(w)-L\bigr).
```

It follows from the triangle inequality: a point at distance $`t`$ from $`u`$ has eccentricity at least $`E(u)-t`$ and at least $`E(w)-(L-t)`$, and the larger of the two is never below $`LB'_e`$. The second bound costs $`k`$ operations per edge and is evaluated only for edges that pass the first:

```math
LB_e
=
\max_i \min\{D_{iu},D_{iw}\}.
```

If either bound cannot improve the incumbent, the edge is discarded before the more expensive breakpoint sweep.

This is an implementation acceleration; it does not change the optimization objective.

---

## Performance and memory

For $`k`$ unique demand nodes and $`n`$ network nodes, the stored shortest-path matrix requires approximately:

```text
float64: 8 × k × n bytes
float32: 4 × k × n bytes
```

Example:

```text
2,000 demand nodes × 12,000 network nodes
```

is approximately:

```text
float64: 183 MiB
float32:  92 MiB
```

Use:

```bash
net-center roads.gpkg --float32
```

to roughly halve matrix storage.

The solver uses a scale-aware numerical margin so float32 rounding cannot incorrectly prune a borderline edge, but the final result remains limited by the precision of the stored matrix.

Median sums are accumulated in float64 even with `--float32`, and the weighted median reads the matrix in bounded blocks, so `--float32` does not create a hidden full-size float64 copy.

---

## Parallel execution

The conservative default is one worker:

```bash
net-center roads.gpkg
```

For larger problems:

```bash
net-center roads.gpkg --jobs 4 --backend loky
```

or:

```bash
net-center roads.gpkg --jobs 4 --backend threading
```

`loky` uses processes; `threading` uses shared-memory threads. With more than one worker, the shortest-path sources are split into at least as many blocks as there are workers.

Performance depends on graph size, demand count, SciPy build, operating system, memory bandwidth, and available RAM. Benchmark the actual workload rather than assuming that more workers are always faster.

---

## Memory controls

Shortest-path calculations are performed in bounded source blocks.

```bash
--max-temp-mb
```

controls the target float64 Dijkstra result size of one block per worker.

The absolute-center sweep is likewise blocked:

```bash
--max-cells
```

controls the maximum number of segment-demand cells handled at once.

These settings change memory use, not the mathematical objective.

---

## Output

Console output reports each solved location with:

- objective value;
- node or edge identifier;
- edge offset when applicable;
- map coordinates.

For the median, the CLI also reports the mean weighted network distance.

Spatial output can be written with:

```bash
net-center roads.gpkg \
  --demand demand.gpkg \
  --out centers.gpkg
```

The output includes:

- `kind`
- `objective`
- `node`
- `edge`
- `t_m`
- geometry

---

## Disconnected networks

All selected demand locations must be reachable through the analyzed network.

If the shortest-path matrix contains infinite distances, `net-center` fails rather than returning a plausible but mathematically meaningless center.

Depending on the application, repair the topology, restrict the study to a connected component, or solve components separately.

---

## What `net-center` does not currently implement

The current package is deliberately narrower than the full network-location literature.

Not implemented:

- directed networks;
- one-way streets;
- turn restrictions;
- asymmetric travel costs;
- time-dependent costs;
- congestion-dependent travel time;
- weighted minimax centers;
- demand located continuously along edges;
- $`p>1`$ center or median problems;
- automatic inference of bridge/tunnel connectivity from attributes;
- out-of-core storage for distance matrices too large for RAM.

---

## Network median versus geographic centroid

The weighted network median is **not** an arithmetic centroid.

A weighted Euclidean centroid minimizes squared Euclidean distance and can be written

```math
\bar{x}
=
\frac{\sum_i w_ix_i}{\sum_iw_i}.
```

The network median instead minimizes

```math
\sum_i w_id_G(x,i).
```

It is therefore better interpreted as a **network-accessibility center under a minisum objective** than as a literal coordinate centroid.

A closer network analogue to a Euclidean centroid would be a network Fréchet mean or barycenter,

```math
x^*
=
\arg\min_{x\in G}
\sum_iw_i d_G(x,i)^2,
```

which is **not currently implemented**.

---

## Choosing an objective

Use the **weighted 1-median** when the goal is to minimize aggregate weighted network distance.

Use the **vertex 1-center** when the worst-served demand matters and the facility must coincide with an existing network node.

Use the **absolute 1-center** when the worst-served demand matters and the facility may be located anywhere along the network.

There is no universal definition of "the center" of a network. The appropriate objective depends on the application.

---

## Repository map

```text
net-center/
├─ .github/
│  └─ workflows/
│     └─ ci.yml
├─ docs/
│  ├─ ALGORITHMS.md
│  ├─ ARCHITECTURE.md
│  ├─ NETWORK_ASSUMPTIONS.md
│  ├─ REVIEW_NOTES.md
│  ├─ TECHNICAL_NOTE.tex
│  ├─ TECHNICAL_NOTE.pdf
│  └─ VALIDATION.md
├─ examples/
│  └─ quickstart.py
├─ net_center/
│  ├─ __init__.py
│  ├─ _checks.py
│  ├─ center.py
│  ├─ cli.py
│  ├─ distances.py
│  ├─ graph.py
│  ├─ solve.py
│  └─ topology.py
├─ tests/
│  ├─ test_center.py
│  ├─ test_cli.py
│  ├─ test_distances.py
│  ├─ test_graph.py
│  ├─ test_package.py
│  └─ test_solve.py
├─ .gitignore
├─ CHANGELOG.md
├─ GITHUB_SETUP.md
├─ LICENSE
├─ README.md
└─ pyproject.toml
```

For understanding the implementation, read approximately in this order:

```text
solve.py
  ↓
center.py
  ↓
distances.py
  ↓
graph.py
  ↓
topology.py
```

---

## Validation

The test suite includes:

- analytically checkable median and center cases;
- continuous edge-interior center cases;
- random-network comparison with independent brute-force edge sampling;
- serial/parallel invariance;
- block-size invariance;
- float32 behavior;
- T-junction recovery;
- interior-interior shared-vertex recovery;
- bridge versus true-junction topology regressions;
- parallel-edge handling;
- closed rings, and rings whose ends are only nearly closed;
- junctions whose two copies differ by floating-point noise;
- lines that end on one of their own interior vertices;
- pruning bounds checked against a sweep of every edge;
- float32 median memory use and summation precision;
- disconnected networks;
- CRS and unit handling;
- sliver and self-loop filtering;
- worker and memory-control validation;
- command-line runs, including one-line error messages for unreadable input and unwritable output;
- package metadata checks.

Run:

```bash
python -m pytest -q
```

See `docs/VALIDATION.md` for the validation philosophy and `docs/ALGORITHMS.md` for the full mathematical and bibliographic discussion.

---

## References

Bentley, J. L., & Ottmann, T. A. (1979). Algorithms for reporting and counting geometric intersections. *IEEE Transactions on Computers*, C-28(9), 643–647.

Daskin, M. S. (2013). *Network and Discrete Location: Models, Algorithms, and Applications* (2nd ed.). Wiley. https://doi.org/10.1002/9781118537015

Dijkstra, E. W. (1959). A note on two problems in connexion with graphs. *Numerische Mathematik*, 1, 269–271.

Hakimi, S. L. (1964). Optimum locations of switching centers and the absolute centers and medians of a graph. *Operations Research*, 12(3), 450–459. https://doi.org/10.1287/opre.12.3.450

Hakimi, S. L. (1965). Optimum distribution of switching centers in a communication network and some related graph theoretic problems. *Operations Research*, 13(3), 462–475. https://doi.org/10.1287/opre.13.3.462

Handler, G. Y., & Mirchandani, P. B. (1979). *Location on Networks: Theory and Algorithms*. MIT Press.

Kariv, O., & Hakimi, S. L. (1979). An algorithmic approach to network location problems. I: The p-centers. *SIAM Journal on Applied Mathematics*, 37(3), 513–538. https://doi.org/10.1137/0137040

Kariv, O., & Hakimi, S. L. (1979). An algorithmic approach to network location problems. II: The p-medians. *SIAM Journal on Applied Mathematics*, 37(3), 539–560. https://doi.org/10.1137/0137041

Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods*, 17, 261–272. https://doi.org/10.1038/s41592-019-0686-2

---

## License

MIT. See `LICENSE`.
