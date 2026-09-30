"""End-to-end checks of the command-line interface (added in the September
2026 review; v0.1.0 exercised the CLI only through --help and --version)."""

import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
pytest.importorskip("pyogrio")
from shapely.geometry import LineString, Point  # noqa: E402

from net_center.cli import main  # noqa: E402

UTM51N = "EPSG:32651"


@pytest.fixture
def roads(tmp_path):
    path = tmp_path / "roads.gpkg"
    gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (3, 0)]), LineString([(3, 0), (10, 0)])], crs=UTM51N
    ).to_file(path)
    return path


def test_cli_solves_and_writes_output(roads, tmp_path, capsys):
    out = tmp_path / "centers.gpkg"
    assert main([str(roads), "--quiet", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "absolute_center" in printed and "edge 1 @ 2.0 m" in printed
    result = gpd.read_file(out)
    absolute = result[result["kind"] == "absolute_center"].iloc[0]
    assert absolute.geometry.x == pytest.approx(5.0)
    assert absolute["objective"] == pytest.approx(5.0)


def test_cli_uses_demand_layer_and_weights(roads, tmp_path, capsys):
    demand = tmp_path / "demand.gpkg"
    gpd.GeoDataFrame(
        {"pop": [1.0, 10.0]}, geometry=[Point(0, 0.5), Point(10, 0.5)], crs=UTM51N
    ).to_file(demand)
    code = main([str(roads), "--quiet", "--demand", str(demand), "--weight-field", "pop"])
    assert code == 0
    median_line = next(
        line for line in capsys.readouterr().out.splitlines() if line.startswith("median")
    )
    assert "(10.000, 0.000)" in median_line


def test_missing_input_file_gives_a_one_line_error(tmp_path):
    # v0.1.0 printed a full traceback: pyogrio's DataSourceError derives from
    # RuntimeError, which the CLI did not catch.
    with pytest.raises(SystemExit, match="^error: "):
        main([str(tmp_path / "does_not_exist.gpkg"), "--quiet"])


def test_unwritable_output_gives_a_one_line_error(roads, tmp_path):
    with pytest.raises(SystemExit, match="^error: "):
        main([str(roads), "--quiet", "--out", str(tmp_path / "no_such_dir" / "x.gpkg")])


def test_weight_field_must_exist(roads, tmp_path):
    demand = tmp_path / "demand.gpkg"
    gpd.GeoDataFrame({"pop": [1.0]}, geometry=[Point(0, 0)], crs=UTM51N).to_file(demand)
    with pytest.raises(SystemExit, match="not in the demand layer"):
        main([str(roads), "--quiet", "--demand", str(demand), "--weight-field", "people"])


def test_cli_and_python_api_agree(roads):
    from net_center import build_network, solve

    net = build_network(str(roads))
    api = solve(net)
    assert np.isclose(api["absolute_center"].objective, 5.0)
