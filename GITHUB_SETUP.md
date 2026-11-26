# Development and contribution guide

Use Git rather than replacing a repository from an archive. Work on a clean feature branch and preserve local changes before updating.

```bash
git clone https://github.com/miraflor/net-center.git
cd net-center
python -m pip install -e ".[dev]"
python -m pip check
python -m ruff check .
python -m compileall -q net_center examples
python -m pytest -q
python -m net_center.cli --help
python -m net_center.adaptive_cli --help
python examples/quickstart.py
python examples/adaptive_quickstart.py
python examples/adaptive_gis.py
```

Run each command only after its predecessor succeeds. On Windows PowerShell 5.1, check `$LASTEXITCODE` immediately after each native command; `$ErrorActionPreference = 'Stop'` alone does not turn a nonzero native exit code into a terminating exception.

## Changes and review

1. Fetch origin and verify the intended base revision and clean tracked/index state. Create a feature branch; never reset away work or overwrite an unexpected file.
2. Make only the intended changes, run tests and lint, and inspect the staged diff. Verify the exact staged file list and content before committing.
3. Push only the verified commit, then verify the remote head. Open a PR against `main` with scope, assumptions and validation results.
4. Require all expected CI jobs to complete successfully for the exact PR head before merging. Missing, pending, canceled, skipped or failing required checks block merge. No automatic bypass or force push.
5. Verify the post-merge main CI before publishing releases.

## Distribution validation

Build with `python -m build`. Install the resulting wheel with its declared dependencies in a fresh environment; run `python -m pip check`. Smoke-test from outside the checkout using Python isolated mode (`python -I`) so imports resolve to the installed wheel rather than repository sources. The package CI implements this procedure and tests Python 3.10–3.13. For GIS integration, install the optional GIS dependencies in a separate test environment if necessary.

The package version is defined in `pyproject.toml`. Keep release notes, metadata and tags consistent; do not publish a new release solely by changing a documentation heading. The current adaptive expansion is unreleased relative to version 0.2.0 metadata.

For the technical note, compile `docs/TECHNICAL_NOTE.tex` twice with pdfLaTeX, using a separate build directory; inspect the PDF before replacing the committed `docs/TECHNICAL_NOTE.pdf`.

Do not commit caches, virtual environments, build directories or large local GIS inputs. Keep small fixtures redistributable and clearly licensed. See [validation](docs/VALIDATION.md) and [changelog](CHANGELOG.md).
