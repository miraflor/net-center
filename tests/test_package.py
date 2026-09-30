"""Small packaging-level regressions that catch release metadata drift."""

from importlib.metadata import PackageNotFoundError, version

import pytest

import net_center


def test_exported_version_matches_installed_distribution():
    # A fresh clone has no installed distribution, so this would otherwise fail
    # for a contributor who simply runs pytest. CI installs first, where the
    # check is meaningful and does run.
    try:
        installed = version("net-center")
    except PackageNotFoundError:
        pytest.skip("net-center is not installed; run `pip install -e .` first")
    assert net_center.__version__ == installed


def test_public_solve_is_callable():
    assert callable(net_center.solve)
