# Replacing an existing GitHub repo with this codebase

This ZIP is laid out as a **repository root**. There is no extra wrapper folder
inside it.

## Existing local repository

1. Open the existing repo folder in File Explorer.
2. Keep the hidden `.git` folder exactly where it is.
3. Delete or replace the old project files **except `.git`**.
4. Extract the replacement ZIP directly into the repo folder.
5. Open GitHub Desktop and review the changed files.
6. Open a terminal in the repo folder and run:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m netcenter.cli --help
```

A clean tree should report **111 passed**.

7. Commit only after the tests pass.

Suggested commit message:

```text
Fix node identity, ring closure, float32 median, and parallel Dijkstra
```

Suggested description:

```text
Replaces grid-based node identity with a distance tolerance, so junctions are
not split by floating-point noise and nearly closed rings are not deleted;
splits lines that end on themselves; fixes a crash with NumPy 2.0.0; removes a
hidden float64 copy and float32 summation error in the median; makes --jobs
split shortest-path work across workers; reports GIS file errors as one-line
CLI messages; adds a faster pruning bound; rebuilds the technical-note PDF from
its source. See CHANGELOG.md, section Unreleased.
```

## New repository

Create an empty GitHub repository, clone it, and place these files directly in
the clone root. Do not add another wrapper directory around them.

## Do not commit

The supplied `.gitignore` excludes Python caches, virtual environments, build
artifacts, and common local GIS outputs. Keep source datasets out of the repo
unless they are deliberately small, redistributable fixtures with clear
licensing.
