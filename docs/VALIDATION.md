# Validation strategy and evidence

Network location software can return plausible but incorrect answers. Tests cover independent mathematical reference cases, network topology, input contracts, diagnostics, execution and distribution installation. Exactness claims are conditional on modeled topology, candidates and numerical tolerance; the heuristic is not guaranteed globally optimal.

## Original single-center suite

The original 111 tests cover weighted medians, vertex and absolute centers, hand-computable paths and triangles, random networks compared with dense edge sampling, safe pruning bounds and pruned/unpruned sweeps, float32 storage and float64 accumulation, parallel/block invariance, memory estimates and invalid inputs. GIS cases cover shared vertices, grade-separated crossings, rings and self-touching lines, slivers, parallel edges, CRS, component ranking and tolerance. CLI and package tests verify input/output handling, metadata and public callable interfaces. Dense sampling is a numerical reference, not a proof of exactness.

## Adaptive suite

The 90 adaptive tests present at the merged revision are in `tests/test_adaptive_expansion.py` and `tests/test_adaptive_integration.py`. They cover:

- ordered physical edge geometry, stale snap metadata, transitive merges, split lineage, distance preservation and original component ranking;
- streaming median and minimax comparisons with dense reference calculations, pruning on/off, weighted duplicate rows, connectors, exact small multi-median enumeration and heuristic cost curves;
- candidate restrictions, independent group domains, component allocation and explicit zero-weight component convention;
- fixed, balanced, budget, penalized and Gap selection; reference support, reproducibility, cap and infeasible cases;
- configuration versus executed-solver audit, finite/nonfinite JSON, coverage, row assignments, weighted access quantiles and no-clobber output;
- actual repository integration with the existing `Network` and exact absolute-center solver, legacy comparison, CLI restrictions and GIS inputs.

The tests support the stated numerical contracts but do not guarantee a global optimum for BUILD/SWAP, a correct real-world road model, or national-scale runtime. Gap results depend on null support, seed, loss variant and candidate universe. Budget failure is limited to evaluated curves.

## Verified CI evidence

The merged commit `e7710b9b436f1b344d60d2327babb42e0e36f266` passed the [post-merge main run](https://github.com/miraflor/net-center/actions/runs/45778162775): all four Python jobs (3.10, 3.11, 3.12, 3.13) and the package job succeeded. The Python 3.12 job log reports **201 passed**, no skips. The wheel and source distribution built successfully; the installed-wheel smoke test imported from `site-packages`, and `pip check` succeeded. The preceding PR and push runs also passed after the CI dependency/smoke-test correction. These are historical observations for that commit, not results from future documentation or code changes.

## Local validation

```bash
python -m pip install -e ".[dev]"
python -m pip check
python -m ruff check .
python -m compileall -q net_center examples
python -m pytest -q
python -m net_center.cli --version
python -m net_center.cli --help
python -m net_center.adaptive_cli --help
python examples/quickstart.py
python examples/adaptive_quickstart.py
python examples/adaptive_gis.py
```

Use fail-fast execution; check each exit status. The editable installation matters: otherwise the package metadata test can skip and the version may be `0+unknown`. Ensure GIS dependencies and drivers are present to avoid skipped integration tests. CI also builds the distribution and runs an isolated import/CLI smoke test outside the source tree. A skipped or missing required check is not successful validation.

## Reproducibility checklist

Record the exact commit/release, Python and dependency versions, input graph and demand/candidate data fingerprints, CRS, topology tolerance, maximum snap distance, component/coverage policy, candidate domain, objective and connector treatment, site-count rule, bounds, seed and solver options. The adaptive schema records configuration and execution separately but does not hash input files. Persist the raw inputs or an immutable provenance record separately. Report both modeled and full-input mass/row coverage. Do not compare objectives across different demand subsets or incompatible connector and candidate conventions.
