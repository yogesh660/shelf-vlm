import json
from pathlib import Path

from shelf.cli import main

ROOT = Path(__file__).parents[2]


def test_schema(capsys):
    assert main(["schema"]) == 0
    assert json.loads(capsys.readouterr().out)["title"] == "ShelfReport"


def test_rows(capsys):
    assert main(["rows", str(ROOT / "tests/fixtures/annotations_mini.csv"), "shelf_a.jpg"]) == 0
    out = capsys.readouterr().out
    assert "25 products on 3 shelves" in out


def test_rows_unknown_image():
    assert main(["rows", str(ROOT / "tests/fixtures/annotations_mini.csv"), "nope.jpg"]) == 1


def test_score(capsys):
    args = ["score", "--planogram", str(ROOT / "examples/planogram.json")]
    assert main([*args, "--report", str(ROOT / "examples/report.json")]) == 0
    assert "compliance 79%" in capsys.readouterr().out


def test_data_stats_without_data(tmp_path, capsys):
    assert main(["data", "stats", "--dest", str(tmp_path)]) == 0
    assert "make data" in capsys.readouterr().out
