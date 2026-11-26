# Network assumptions

These assumptions are part of the model, not merely command-line settings.

## Shared network model

- Undirected, positive-length edges. One-way roads, turn restrictions, asymmetric or time-dependent costs and weighted minimax are not represented.
- Geometry is not connectivity. By default, shared source vertices recover junctions (including endpoint/interior and self-touching cases); mere 2-D crossings do not connect. `split_shared_vertices=False` / `--no-shared-vertex-noding` assumes already segmented topology. `node=True` / `--node-crossings` explicitly planarizes crossings and is unsafe for overpasses.
- The graph builder does not inspect OSM bridge, tunnel, layer, access or turn attributes. Prepare routing-quality input; a shared coordinate is evidence, not proof of a turn.
- Distances in the GIS workflows are metres. Geographic/non-metre input is transformed to a suitable inferred local UTM CRS when possible; explicit target CRS must be projected and metre-based. Choose an appropriate projection for the study extent. Numerical callers supply consistent distance units.
- Node identity tolerance (`snap`, default 0.1 m) is transitive. It is distinct from maximum demand connector distance. Nearby shape vertices on one line are not indiscriminately merged. Closed and nearly closed rings are split so they survive self-loop filtering. Choose tolerance above coordinate noise but well below the shortest real segment.
- Parallel physical edges remain available for continuous attachment and location, while CSR retains the minimum routing weight for each pair. Adaptive geometry validation requires ordered endpoints, physical length consistency and verifiable transitive merges. Inconsistent geometry is rejected rather than silently reversed or repaired.

## Legacy versus adaptive demand model

| Aspect | `net-center` / `solve` GIS workflow | `python -m net_center.adaptive_cli` |
|---|---|---|
| Demand | Optional; points or polygon representative points, default every node | Required nonempty point layer |
| Attachment | Nearest existing node | Nearest physical edge by default; `--snap-mode node` available |
| Connector | Snap QA; not part of objective | Access always includes connector; objective optionally includes it |
| Unattached rows | Warn and exclude beyond max snap | Fail by default; explicit audited exclusion available |
| Network retention | Largest component by physical road length | All components by default |
| Output | Single-center results, optional point file | Complete JSON with group-level results and row coverage |

Finite edge-interior observations become vertices by splitting the original physical edges at accepted offsets. The continuous absolute 1-center can itself lie on an edge interior. Neither model integrates demand density continuously along an entire edge. Candidate point layers constrain facilities, not demand. Snapping to the closest edge can select a small isolated road; it does not imply that travel is possible to the rest of the graph.

## Components and coverage

The adaptive workflow exposes four explicit policies:

- `strict`: require all modeled demand to occupy one component; unused components do not matter.
- `largest`: attach first and exclude demand outside the largest component (physical lengths in GIS, collapsed CSR weights unless physical metadata supplied in numerical calls).
- `per_component`: one site per demand-bearing component, with a per-group global cap.
- `allocate`: distribute a per-group budget over populated components, requiring one site in each. Applies to median allocation; general multi-site minimax is not implemented.

`--keep-largest-component` is a separate preprocessing option: remove small components before attaching points. It matches legacy network construction, but may enlarge connector distances and does not bypass `--max-snap`. No policy adds bridges or links to manufacture reachability. Exclusion changes the modeled demand denominator; inspect both row and mass coverage before comparing methods.

All modeled rows require a finite assignment even when their weight is zero. Each modeled group must have positive mass. A zero-weight component in a positive-mass median group gets exactly one secondary minimax-access site, even if extra sites could lower maximum access without affecting the weighted objective. This is a modeling convention. Minimax remains unweighted; weights summarize access, not the radius.

## Facility locations and distance assumptions

The numerical finite-site domains are `vertices`, `demand`, and `restricted`. The GIS vertices domain includes original vertices and only that group's own accepted demand vertices. A shared routing graph does not imply a shared augmented candidate set. `continuous` is reserved for the GIS absolute 1-center objective. The exact and heuristic multi-median algorithms use the same finite candidate domain.

Connector distances are fixed nonnegative offsets, not routing edges. They cannot affect the median site's optimal location for fixed modeled demand, but may affect minimax. Access budget always includes offsets. The GIS metric is physical length; arbitrary per-edge costs, boarding penalties, directed travel, off-network walking routes and continuous network demand require other models.

## Practical checks

Use a defensible maximum snap distance, inspect attachment and excluded-row diagnostics, verify physical topology and CRS, and choose candidate and component policies before interpreting the objective. Gap reference vertices are a modeling choice affected by vertex density and study extent. No guarantee is made that the nearest Euclidean attachment is the appropriate real-world access point. See [adaptive guide](ADAPTIVE_LOCATION.md) and [validation](VALIDATION.md).
