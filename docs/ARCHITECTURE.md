# Architecture

`netcenter` is deliberately layered so the mathematics, routing, and GIS code
can be reviewed independently.

## Execution path

```text
road line file
    |
    v
graph.py
    |  CRS + geometry cleanup + source-topology recovery
    |  -> Network arrays + sparse adjacency
    v
solve.py
    |  demand validation and coalescing
    v
distances.py
    |  SciPy Dijkstra -> demand x node distance matrix D
    v
center.py
    |-- weighted_median(D)
    |-- vertex_center(D)
    `-- absolute_center(D, edges)
    |
    v
solve.py
    |  node / edge+t -> map x,y
    v
CLI / Python caller
```

## `topology.py`

Pure numeric topology helper. It turns edge endpoint arrays into a CSR sparse
matrix and collapses parallel graph connections by minimum length. No GIS
dependency.

## `graph.py`

The GIS boundary and therefore the highest-risk layer.

Responsibilities:

- choose/validate a metre-based working CRS;
- flatten line geometry and remove empties;
- recover junctions already encoded as shared source vertices, including lines
  that end on one of their own interior vertices;
- optionally perform full planar noding when explicitly requested;
- preserve closed and nearly closed rings;
- apply one location rule (coordinates within the snap distance, transitively)
  to node identity, shared-vertex junctions, and ring closure;
- remove slivers/self-loops and compact orphaned node IDs;
- select the largest connected component by road length;
- provide node coordinates and edge interpolation back to map space.

## `distances.py`

Scheduling layer around `scipy.sparse.csgraph.dijkstra`.

Responsibilities:

- validate sparse graph and source indices;
- estimate stored matrix memory;
- break Dijkstra output into bounded float64 chunks;
- cast each chunk promptly to the requested storage dtype;
- call Dijkstra in directed mode, which is valid because the graph is checked
  to be symmetric;
- optionally parallelise large batches, with at least one block per worker;
- stream parallel blocks directly into a preallocated final matrix.

## `center.py`

Mathematical core. It contains no GeoPandas/Shapely imports. Inputs are plain
NumPy arrays, which keeps the location algorithms independently testable.
Absolute-centre pruning applies a bound that costs one operation per edge
before the bound that costs one operation per demand point per edge.

## `_checks.py`

Shared input checks: positive and non-negative integers, backend names,
node-index arrays, and median weights. `center.py`, `distances.py`, `solve.py`,
and `topology.py` all use them, so every entry point applies the same rule and
raises the same message.

## `solve.py`

Small orchestration layer. It coalesces repeated demand nodes, builds `D`, calls
the three location solvers, and attaches map coordinates to results.

## `cli.py`

No core modelling logic belongs here. It handles arguments, file loading,
progress messages, and result writing. File errors from the GIS readers and
writers are reported as one-line messages.
