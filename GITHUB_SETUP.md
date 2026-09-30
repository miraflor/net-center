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
python -m pip uninstall -y netcenter
python -m pip install -e ".[dev]"
python -m pytest -q
python -m net_center.cli --help
```

The first command removes the package installed under the old name `netcenter`. If that package is not installed, pip only prints a warning.

A clean tree should report **111 passed**.

7. Commit only after the tests pass.

Suggested commit message:

```text
Rename package to net-center and add provenance to README
```

Suggested description:

```text
Renames the distribution to net-center, the import package to net_center (directory netcenter/ became net_center/), and the command to net-center, to match the GitHub repository name. Adds a per-step provenance table and two references to the README. Rebuilds docs/TECHNICAL_NOTE.pdf from its source. See CHANGELOG.md, section Unreleased.
```

## New repository

Create an empty GitHub repository, clone it, and place these files directly in
the clone root. Do not add another wrapper directory around them.

## Do not commit

The supplied `.gitignore` excludes Python caches, virtual environments, build
artifacts, and common local GIS outputs. Keep source datasets out of the repo
unless they are deliberately small, redistributable fixtures with clear
licensing.
