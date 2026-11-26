# Algorithms, and who they belong to

**Scope:** The original stages, complexity table and historical engineering audit below describe the legacy single-center workflow. The adaptive workflow is documented in the added section before References and in [Adaptive location](ADAPTIVE_LOCATION.md).

Every algorithm in this package was invented by someone else, most of them
decades ago. This document records what is used, who created it, and — kept
carefully separate at the end — which parts are implementation work done with
AI assistance rather than anything original.

Nothing here is presented as a new algorithm. The package implements the
classical absolute 1-centre directly because that continuous edge-interior case
was the gap motivating the original codebase.

## Notation

- `n` — junctions (graph vertices)
- `m` — road segments (graph edges)
- `k` — demand points; `k ≤ n`, and usually `k ≪ n`
- `d(x, y)` — shortest road distance
- `L` — the length of a particular segment

---

## Stage 1 — Turning a drawing into a network

### Recovering source topology without inventing crossings

A line layer records geometry, not necessarily road topology. Two lines may
cross on the page without a valid turn between them (for example, a bridge over
a road), while a genuine OSM-style junction can be stored as a coordinate in
the *interior* of one or both LineStrings.

The default therefore uses a deliberately conservative rule: a source-vertex
location is a junction when at least two distinct input LineStrings use it, or
when a line ends there. The second case includes a line that ends on one of its
own interior vertices, such as a cul-de-sac turning loop. Every line is split at
each of its interior vertices that lies at a junction. This recovers T- and
X-junctions already encoded by the source but does not create a new coordinate
at a mere geometric crossing.

This shared-vertex recovery is straightforward array bookkeeping rather than a
new graph algorithm; no novelty is claimed.

### Optional planar splitting of all geometric crossings

When the caller explicitly enables planar noding, every two-dimensional
crossing is treated as connected. That is appropriate only when the input is
genuinely planar.

The sweep-line method for reporting all intersections among a set of segments
is due to **Jon Bentley and Thomas Ottmann (1979)**. Producing output that
remains topologically consistent once coordinates are rounded to finite
precision — the practical difficulty — is **snap rounding**, introduced by
**Daniel Greene and F. Frances Yao (1986)** and made practical by
**John Hobby (1999)**.

This package does not implement those algorithms directly. Explicit planar
noding calls `shapely.union_all`, which binds to **GEOS**, itself a port of the
**JTS Topology Suite** written by **Martin Davis**.

### Merging away shape points

A curved road is drawn with many coordinates. Only the junctions matter for
routing; the rest describe shape. Merging chains of segments that meet
end-to-end at a point where nothing else joins can reduce `n` dramatically.
The distance matrix costs `O(kn)` memory and becomes `O(n²)` when every node is
used as demand, so unnecessary shape vertices can dominate the problem.
Implemented by `shapely.line_merge`, again GEOS/JTS (Martin Davis). This
merging runs only in the explicit planar-noding mode. In the default and
endpoint-only modes lines are not merged, so a point where two lines meet end to
end, with no third line, remains a node of degree two.

### Matching near-identical coordinates

Endpoints meant to coincide often differ in the last decimal places. Two
coordinates closer than the snap distance are treated as one location, and the
rule is applied transitively (single-linkage clustering over a k-d tree). Within
one line, two vertices are linked only when one of them is an endpoint and the
line travels more than the snap distance between them, so a densely digitised
road is not merged into one location. One rule decides node identity,
shared-vertex junctions, and ring closure. This is folklore rather than
anyone's algorithm. Until September 2026 coordinates were rounded to a grid and
compared as integers; see bugs 9 and 10 below.

### Finding the connected pieces

A network digitised from real data usually contains stranded fragments.
Identifying connected components in linear time is standard depth-first search,
in the form given by **John Hopcroft and Robert Tarjan (1973)**. Used via
`scipy.sparse.csgraph.connected_components`.

### Attaching demand points to the network

Demand points rarely sit exactly on a road, so each is matched to its nearest
junction using a **k-d tree**, invented by **Jon Bentley (1975)**. Used via
`scipy.spatial.cKDTree`.

---

## Stage 2 — Shortest paths

Distances come from **Dijkstra's algorithm**, published by
**Edsger W. Dijkstra in 1959** in a paper slightly over two pages long. It is
run once per demand point. Used via `scipy.sparse.csgraph.dijkstra`; not
reimplemented here.

Cost is `O(k · (m + n log n))` with a heap, and the result occupies
`O(k · n)` memory. On any network large enough to matter, the memory is what
stops you rather than the arithmetic.

---

## Stage 3 — The 1-median (minisum)

Find `x` minimising `Σ_v w_v · d(x, v)`.

The governing result is **Hakimi's vertex optimality theorem**, from
**S. L. Hakimi (1964)**: along any segment the objective is concave, so it
attains its minimum at an endpoint, and therefore an optimal median can always
be taken at a junction. Hakimi extended this to `p` medians in **(1965)**.

This collapses a continuous problem into a discrete one, and the implementation
is a single weighted sum over the distance matrix followed by an `argmin`.

Maximising closeness centrality — **Linton Freeman (1978)**, building on
**Alex Bavelas (1950)** — is the same computation with unit weights.

On a tree the 1-median is computable in linear time, shown by
**A. J. Goldman (1971)**.

---

## Stage 4 — The absolute 1-centre (minimax)

Find `x` anywhere on the network minimising `max_v d(x, v)`.

Vertex optimality **fails** here, and this is the whole difficulty. The optimum
can lie strictly inside a segment. Both the problem and the term *absolute
centre* are due to **Hakimi (1964)**, in the same paper as the median theorem.

The efficient algorithm is **Oded Kariv and S. L. Hakimi (1979)**, part I,
which solves the absolute `p`-centre and gives `O(mn log n)` for `p = 1`. The
same paper proves the `p`-centre problem NP-hard for general `p`; part II does
the same for the `p`-median. **Edward Minieka (1970)** treated the related
`m`-centre problem and the variant where every point of the network, not only
the junctions, counts as demand.

### The construction implemented here

For a segment from `u` to `w` of length `L`, and a position `t` metres from
`u`:

```
d(t, v) = min(a_v + t,  b_v + L − t)        a_v = d(u, v),  b_v = d(w, v)
```

a tent-shaped function of `t`. Eccentricity is the upper envelope of `k` tents:
piecewise linear with slopes ±1, not concave, so its minimum can be interior.

Tent `v` is on its rising branch exactly when `t ≤ t*_v = (b_v − a_v + L)/2`.
Sorting demands by `t*_v` makes the rising set a suffix and the falling set a
prefix of the sorted order, so between consecutive breakpoints

```
ecc(t) = max(A + t,  B + L − t)
```

with `A` a suffix maximum of `a` and `B` a prefix maximum of `b`, both constant
on the interval. That is a convex V, minimised where the two lines meet,
clamped to the interval. `O(k log k)` per segment, `O(mk log k)` overall.

The triangle inequality guarantees `t*_v ∈ [0, L]`, so no interval is degenerate.

This decomposition is the standard reading of Kariv and Hakimi's local-centre
computation, presented in textbook form by **Handler and Mirchandani (1979)**
and **Daskin (2013)**. **Nimrod Megiddo (1983)** later removed the logarithmic
factor using parametric search; that refinement is *not* implemented here,
because at these problem sizes the sort is not the bottleneck.

### Special cases in the literature, not implemented

On a tree the centre is far easier — **Camille Jordan (1869)** characterised it
as one vertex or two adjacent ones, in a paper about trees written a century
before anyone had a computer.

---

## Complexity summary

| Stage | Cost | Source |
|---|---|---|
| Noding | `O((s + i) log s)` for `s` segments, `i` crossings | Bentley & Ottmann 1979 |
| Shortest paths | `O(k(m + n log n))` | Dijkstra 1959 |
| 1-median | `O(kn)` | Hakimi 1964 |
| Jordan centre | `O(kn)` | — |
| Absolute centre | `O(mk log k)` worst case | Kariv & Hakimi 1979 |

In practice the absolute centre stage is far below its worst case, because of
the bound described in the next section.

---

## AI-assisted implementation work

The following were developed in a working session with Claude (Anthropic). They
are engineering on top of the published algorithms above — none is a new
algorithm, and none should be cited as a contribution to the literature. They
are listed because the person reviewing this code should know which lines came
from a textbook and which came from a conversation, and therefore where to look
hardest for mistakes.

### Optimisations

1. **Segment pruning by a lower bound.** From any point on a segment,
   `d(x, v) ≥ min(d(u, v), d(w, v))`, so `max_v min(d(u,v), d(w,v))` is a lower
   bound on that segment's eccentricity. Segments whose bound already exceeds
   the best known junction radius cannot contain a better centre and are not
   swept. This follows directly from the distance formula; no novelty is
   claimed.

2. **Batched vectorisation of the sweep.** The interval decomposition is
   Kariv–Hakimi's; expressing it as prefix/suffix running maxima over a
   `(segments × demands)` array so many segments are solved together is an
   implementation choice, as is the `-inf` padding that removes boundary
   special cases.

3. **Memory-budgeted sweep blocking.** Both the lower-bound calculation and the
   exact sweep process segments in cell-bounded blocks so temporary memory does
   not scale directly with the full edge count.

4. **Contiguous row-wise sort inputs.** Endpoint-distance columns are copied
   into row-contiguous arrays before the per-edge switch-point sort. This costs
   one controlled copy and makes the subsequent vectorised operations simpler
   and generally faster on large blocks.

5. **Conservative parallelism.** `n_jobs` is a ceiling, not a promise. Small
   jobs remain serial because worker startup can dominate useful work. The
   package defaults to one worker and exposes both process and thread backends for
   deliberate benchmarking rather than assuming one is universally superior.

6. **Bounded shortest-path chunks.** SciPy returns Dijkstra distances in
   float64. Source nodes are split into bounded row groups, each result is cast
   promptly to the requested storage dtype, and rows are written into the final
   preallocated matrix. This keeps a float32 request from requiring one full
   float64 matrix at the same time as the complete float32 matrix.

7. **Streaming parallel assembly.** Parallel Dijkstra blocks are consumed as an
   ordered joblib generator and written directly to final output rows. The code
   therefore avoids retaining all completed blocks and then constructing a
   second complete matrix with `numpy.vstack`.

8. **Exact demand coalescing.** Repeated observations snapped to the same node
   are collapsed before Dijkstra. Multiplicity is irrelevant to a maximum, so
   the centre is unchanged; counts or user weights are summed for the median,
   preserving the minisum objective exactly.

The centre pruning step also uses a dtype-and-scale-aware safety margin so
float32 rounding cannot discard a borderline candidate edge.

The following were added in a September 2026 review, also with Claude:

9. **Eccentricity pre-bound.** By the triangle inequality, a point at distance
   `t` from `u` has eccentricity at least `ecc(u) - t` and at least
   `ecc(w) - (L - t)`; the larger of the two is never below
   `(ecc(u) + ecc(w) - L) / 2`. This bound costs one operation per segment, uses
   the junction eccentricities that the Jordan centre already needs, and is
   applied before the bound of item 1, which costs `k` operations per segment.
   On the tested
   road-like networks it alone leaves about five of 18,600 segments; the
   absolute-centre stage became about nine times faster.

10. **Directed Dijkstra on the validated symmetric graph.** SciPy's undirected
    mode builds a transposed graph on every call and relaxes each edge twice.
    The graph is already checked to be symmetric, so directed mode gives
    identical distances about 10 % faster.

11. **Vectorised line cutting.** Shared-vertex noding builds all pieces in one
    `shapely.linestrings` call instead of one Python-level call per piece:
    about three times faster graph construction on OSM-shaped input.

12. **Mask-free validation.** The distance matrix is checked with one `min()`
    and one `max()` (NaN propagates through both) instead of full-size boolean
    masks.

### An optimisation that was tried and rejected

**Best-first bound tightening** — sweeping the most promising segments first,
tightening the incumbent radius from the result, and re-pruning the remainder.
This is standard branch-and-bound practice and it was implemented and measured.
It was **removed** because the first lower-bound pass already made the second
pass too small to justify the added branching and code complexity on the tested
workloads. It is recorded here so future optimisation work starts from a
measurement rather than an assumption.

### Correctness bugs found and fixed

The failures below all had the potential to produce plausible wrong answers
rather than obvious crashes, which is the dangerous kind. They are kept here as
regression history.

1. **Parallel edges silently summed.** `coo_matrix(...).tocsr()` adds duplicate
   entries, so two roads joining the same pair of junctions became one road of
   their combined length. This made `d(u,w) > L`, breaking the triangle
   inequality the sweep depends on, which pushed breakpoints outside `[0, L]`,
   inverted intervals, and returned a radius **below** the true one. Fixed by
   collapsing duplicates with a minimum (`topology.adjacency_from_edges`), plus
   a defensive clamp in the sweep.

2. **Closed loops deleted.** A ring road merges into a single line whose ends
   coincide; the self-loop filter then removed it, deleting real road. On a test
   case a **1.9 km ring vanished**. Fixed by cutting such loops in half.

3. **Empty geometries corrupting endpoints.** A geometry with no coordinates
   left an empty run in the coordinate index, so the "last coordinate" lookup
   reached into the previous line's data and attached segments to the wrong
   junctions. Fixed by filtering empties before the lookup.

4. **Needless reprojection.** Always converting to the estimated local UTM zone
   is right for latitude/longitude input and wrong for already-projected
   metre-based input. Projected metre input is now left untouched, while
   explicit target CRSs are checked to ensure the documented metre semantics.

5. **Nearly closed rings deleted.** GIS rings are sometimes visually closed but
   have a final coordinate that differs from the first by floating-point dust.
   Exact `is_closed` testing can miss them; endpoint snapping then maps both ends
   to one node and the resulting self-loop is discarded. The graph builder now
   splits such rings before that can happen. The first fix was incomplete; see
   bug 10.

6. **False junctions at grade-separated crossings.** Treating every geometric
   crossing as a routable junction can connect a bridge to the road below it.
   Planar noding is explicit rather than the default.

7. **Shared-vertex noding stopped at T-junctions.** An intermediate patch split
   a through-road when another line *ended* on one of its interior vertices.
   That still missed a genuine junction where two ways both continued through
   the same shared source vertex. The current rule defines a junction by
   participation of at least two distinct input LineStrings, or by a line
   endpoint, covering both T and X cases without inventing a node at an
   ordinary geometric crossing.

8. **Filtered slivers left orphan graph nodes.** Node IDs were assigned before
   tiny edges and snapped self-loops were removed. With component filtering
   disabled, a discarded edge could leave an isolated CSR row/column and later
   produce infinite distances in an otherwise connected surviving network.
   Node IDs are now compacted immediately after edge filtering.

Found in the September 2026 review of v0.1.0:

9. **Junctions split at grid-cell boundaries.** Node identity rounded
   coordinates to a 0.1 m grid. A centimetre coordinate ending in 5 lies exactly
   on a cell boundary, so floating-point noise of 1e-9 m put the two copies of
   one junction into different cells and cut the road. Fixed by the
   distance-based location rule above.

10. **Rings deleted when closure and identity disagreed.** The fix for bug 5
    tested ring closure by Euclidean distance but merged endpoints by grid cell.
    Ends 0.113 m apart in one cell became one node without passing the closure
    test, and a 1.9 km ring was deleted again. A line is now a ring exactly when
    both ends are one location.

11. **Lines that end on themselves were not split.** A cul-de-sac turning loop
    drawn as one line was kept as one edge, so the road distance from the loop
    entrance to the loop junction was 500 m instead of 100 m in a test.

12. **Crash with NumPy 2.0.0.** That release, allowed by `numpy>=1.24`, returns a
    two-dimensional inverse from `numpy.unique(..., axis=0,
    return_inverse=True)`; the shared-vertex step did not flatten it, so graph
    construction failed for any input with two or more lines. CI installs only
    the newest versions and could not detect it.

13. **Float32 median.** The weighted median evaluated `weights @ D`, which
    converts a float32 matrix to a full float64 copy (184 MiB for a 92 MiB
    matrix). The unweighted median summed in float32, so on a 2,000 × 12,000
    test the node objectives were off by up to 58.9 m. Both are now accumulated
    in float64 in bounded blocks.

14. **Parallel Dijkstra silently serial.** Sources were split by memory only,
    so any problem that fitted one 256 MiB block ran serially whatever
    `n_jobs` was.

15. **Command-line tracebacks.** pyogrio's file errors derive from
    `RuntimeError`, which the CLI did not catch, so a mistyped input path printed
    a traceback instead of a one-line error.

Each has a regression test in `tests/test_graph.py`, `tests/test_distances.py`,
`tests/test_center.py`, or `tests/test_cli.py`, except bug 12, which was
verified by running the test suite under NumPy 2.0.0.

### Verification

Correctness of the sweep is checked against brute force: twelve random networks
are sampled at 4,001 points along every segment, and the sweep's answer must be
no worse than the sampled minimum and within the sampling resolution of it.
Further tests confirm hand-computable cases, that the answer never exceeds the
Jordan centre, and that results are invariant to worker count and block size.
Both pruning bounds are checked against the exact minimum of every segment on
random networks, and the pruned answer is compared with a sweep of every
segment.

Brute force is a genuinely independent check — it shares no code with the sweep
— but it only verifies the sweep given a distance matrix. It does not verify
that the distance matrix describes the roads you meant. That remains the
reviewer's job, and stage 1 is where the real risk lives.

---

## Adaptive location methods (current implementation)

The preceding stages and their historical engineering audit describe the original single-site pipeline. The adaptive workflow shares the graph and absolute-center foundations but uses separate demand attachment, component, candidate and memory semantics. See [Adaptive location](ADAPTIVE_LOCATION.md) for the full API and output contracts and [the unified technical note](TECHNICAL_NOTE.pdf) for derivations.

### Edge attachment and metric preservation

`edge_snap.py` projects each demand/candidate observation to a physical edge, resolves exact ties by edge ID, and inserts unique accepted interior offsets. Subsegments retain the original edge's length in total and lineage; original-vertex shortest paths are preserved within floating-point tolerance. Ordered endpoints, physical lengths, CSR consistency and snap records are validated. Parallel physical edges remain distinct. This is geometric preprocessing, not a new routing algorithm. Finite demand positions are now vertices; continuous demand density is not implemented.

### Streaming single-site optimization

For demand weights $w_i$, attached nodes $q_i$, and eligible vertex $v$, minimize $J(v)=\sum_i w_i d_G(q_i,v)$ or the maximum distance. Instead of retaining the original full matrix, the adaptive solver reduces bounded Dijkstra batches to a score vector. For a feasible candidate $p$ and total positive mass $W$,

```math
J(v) \ge Wd_G(p,v)-J(p).
```

An improving candidate must lie within $2J(p)/W$ of $p$. A safe cutoff for demand-source searches is $\max_i d_G(p,q_i)+2J(p)/W$ plus the implementation's numerical margin. This follows directly from triangle inequalities; no novel published algorithm is claimed. Zero-weight rows are omitted from the weighted sum to avoid `0*inf` but retained for coverage. Vertex minimax streams a maximum and can include fixed connector offsets. The absolute-group wrapper adds the same offset to both endpoint distances of each demand (max offset at repeated nodes), which leaves the edge-switch difference unchanged. The original edge sweep and lower bounds therefore remain valid.

### Multiple medians and medoids

For finite candidate set $A$ and $S\subseteq A$ of size $k$, minimize

```math
J(S)=\sum_i w_i \min_{s\in S} d_G(q_i,s).
```

Demand-node candidates give weighted medoids; unrestricted supplied vertices or explicit restricted candidates give finite network medians. The GIS group-specific vertex universe contains original vertices plus the group's own attached demand vertices, not other groups' inserted positions. Greedy BUILD chooses sites with the best weighted improvement, followed by a bounded best-improving single SWAP loop (interchange/PAM-style methods; Teitz and Bart 1968, Kaufman and Rousseeuw 1990). Each larger curve point starts from the preceding site's set and adds a site; accepted swaps decrease cost. Distances are exact up to numerical precision, while the site set is heuristic. The exact alternative enumerates finite subsets after checking the sum of binomial counts against a per-component cap. The dense matrix has shape unique demand by eligible candidates; it is not a full graph all-pairs matrix unless those sets cover the graph.

### Component curves and adaptive selection

For demand-bearing components $c=1,\ldots,C$, finite coverage requires one site per component. Given available component costs $J_c(j)$, the allocation DP retains minimum cost and site tuples for every evaluated total $K$:

```math
F_c(K)=\min_j\{F_{c-1}(K-j)+J_c(j)\},\quad F_0(0)=0.
```

Only feasible counts within the cap enter. Exactness of allocation is relative to these curves. For a zero-weight component in a positive-mass median group, exactly one secondary connector-inclusive minimax site is required, with zero median cost. This extra modeling convention can constrain fixed-k feasibility.

Let $J(K)=F_C(K)$, $n$ distinct modeled demand nodes, $W$ modeled mass, $a_i$ fixed connectors. Fixed selects a prescribed feasible count. Balanced minimizes $J(K)/J(C)+\lambda(K-C)/\max(1,n-C)$ for evaluated $K\le n$, with zero normalized loss when the baseline is zero. Access budget selects the first evaluated count satisfying $(J(K)+\sum_iw_ia_i)/W\le B$ up to numerical tolerance. Penalized minimizes $J(K)+fK$ (facility cost in weight-distance units). These criteria are explicit choices, not global statistical or optimization theorems beyond the evaluated curves.

Gap is a network-medoid variant of Tibshirani, Walther and Hastie (2001). The dispersion is a weighted sum of unsquared network distances, not the squared Euclidean dispersion of the original formulation. Draw with replacement from component vertices, optionally restricted to the demand bounding box (GIS uses original vertices). Preserve row weights. For $B$ reference replicates, compute $g_j=\operatorname{mean}_b\log J^*_{jb}-\log J_j$ and $s_j=\operatorname{sd}_b(\log J^*_{jb})\sqrt{1+1/B}$ with the documented common positive floor. Select the smallest $j$ satisfying $g_j\ge g_{j+1}-s_{j+1}$, or audit upper-bound contact. Degenerate support and inadmissible splits can retain one site; exceeding the global cap fails. Null support is vertex-uniform, not length-uniform. The exact option is not used for Gap.

### Access, coverage and numerical guarantees

Each modeled row has nearest-site road distance $r_i$ and access $a_i+r_i$. Connector costs add a fixed term to the median but can move minimax solutions. Weighted mean and percentiles use positive modeled mass, and maximum access includes zero-weight rows. Exclusions are explicit; the GIS CLI records full-input coverage and attachment coverage. The solver recomputes the selected objective from routed assignments. Configured options and actual component solver paths are separately reported. Output publication is all-groups-or-error and no-clobber.

The adaptive single-site path stores bounded distance blocks instead of a full persistent matrix. Multi-site and absolute component paths have dense matrix limits; combinatorial and memory gates do not imply a total RSS bound. The original complexity table above refers to the single-site path. Multi-site BUILD/SWAP depends on candidate and demand sizes, site counts and swap passes; exhaustive enumeration grows combinatorially; Gap repeats reference curves. General multi-site minimax, directed costs, target-terminated routing and out-of-core distances remain outside scope.

### Additional methodological references

- Hakimi (1965), *Optimum Distribution of Switching Centers in a Communication Network and Some Related Graph Theoretic Problems*. https://doi.org/10.1287/opre.13.3.462
- Teitz and Bart (1968), *Heuristic Methods for Estimating the Generalized Vertex Median of a Weighted Graph*. https://doi.org/10.1287/opre.16.5.955
- Kaufman and Rousseeuw (1990), *Finding Groups in Data*. Wiley.
- Tibshirani, Walther and Hastie (2001), *Estimating the Number of Clusters in a Data Set via the Gap Statistic*. https://doi.org/10.1111/1467-9868.00293
- R `cluster` [clusGap documentation](https://search.r-project.org/CRAN/refmans/cluster/html/clusGap.html), `d.power` and `Tibs2001SEmax`.

---

## References

Bavelas, A. (1950). Communication patterns in task-oriented groups.
*Journal of the Acoustical Society of America*, 22(6), 725–730.

Bentley, J. L. (1975). Multidimensional binary search trees used for
associative searching. *Communications of the ACM*, 18(9), 509–517.

Bentley, J. L., & Ottmann, T. A. (1979). Algorithms for reporting and counting
geometric intersections. *IEEE Transactions on Computers*, C-28(9), 643–647.

Daskin, M. S. (2013). *Network and Discrete Location: Models, Algorithms and
Applications* (2nd ed.). Wiley.

Dijkstra, E. W. (1959). A note on two problems in connexion with graphs.
*Numerische Mathematik*, 1, 269–271.

Freeman, L. C. (1978). Centrality in social networks: conceptual clarification.
*Social Networks*, 1(3), 215–239.

Goldman, A. J. (1971). Optimal center location in simple networks.
*Transportation Science*, 5(2), 212–221.

Greene, D. H., & Yao, F. F. (1986). Finite-resolution computational geometry.
*27th Annual Symposium on Foundations of Computer Science*, 143–152.

Hakimi, S. L. (1964). Optimum locations of switching centers and the absolute
centers and medians of a graph. *Operations Research*, 12(3), 450–459.

Hakimi, S. L. (1965). Optimum distribution of switching centers in a
communication network and some related graph theoretic problems.
*Operations Research*, 13(3), 462–475.

Handler, G. Y., & Mirchandani, P. B. (1979). *Location on Networks: Theory and
Algorithms*. MIT Press.

Hobby, J. D. (1999). Practical segment intersection with finite precision
output. *Computational Geometry*, 13(4), 199–214.

Hopcroft, J., & Tarjan, R. (1973). Algorithm 447: efficient algorithms for
graph manipulation. *Communications of the ACM*, 16(6), 372–378.

Jordan, C. (1869). Sur les assemblages de lignes. *Journal für die reine und
angewandte Mathematik*, 70, 185–190.

Kariv, O., & Hakimi, S. L. (1979). An algorithmic approach to network location
problems. I: The p-centers. *SIAM Journal on Applied Mathematics*, 37(3),
513–538.

Kariv, O., & Hakimi, S. L. (1979). An algorithmic approach to network location
problems. II: The p-medians. *SIAM Journal on Applied Mathematics*, 37(3),
539–560.

Megiddo, N. (1983). Applying parallel computation algorithms in the design of
serial algorithms. *Journal of the ACM*, 30(4), 852–865.

Minieka, E. (1970). The m-center problem. *SIAM Review*, 12(1), 138–139.

Mirchandani, P. B., & Francis, R. L. (Eds.) (1990). *Discrete Location Theory*.
Wiley.

### Software

GEOS and the JTS Topology Suite (Martin Davis and contributors) — noding and
line merging, via Shapely.
SciPy (`sparse.csgraph`, `spatial`) — Dijkstra, connected components, k-d tree.
NumPy — array operations throughout.
joblib — optional threading/process parallel scheduling.
