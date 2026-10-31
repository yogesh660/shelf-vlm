import json
from pathlib import Path

from shelf.cli import main

ROOT = Path(__file__).parents[2]


def test_schema(capsys):
    assert main(["schema"]) == 0
    assert json.loads(capsys.readouterr().out)["title"] == "ShelfReport"


def test_eval_gaps(capsys):
    assert main(["eval-gaps", str(ROOT / "tests/fixtures/annotations_mini.csv")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["images"] > 0
    assert result["tp"] + result["fn"] == result["images"]


def test_eval_gaps_empty(tmp_path, capsys):
    path = tmp_path / "empty.csv"
    path.write_text("")
    assert main(["eval-gaps", str(path)]) == 1
    assert json.loads(capsys.readouterr().out)["images"] == 0


def test_eval_detector(tmp_path, capsys):
    predictions = tmp_path / "predictions.csv"
    predictions.write_text("shelf_a.jpg,0,50,90,250,0.9\n")
    assert (
        main(
            [
                "eval-detector",
                str(ROOT / "tests/fixtures/annotations_mini.csv"),
                str(predictions),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["tp"] == 1
    assert result["fn"] == 27


def test_eval_vlm(tmp_path, capsys, monkeypatch):
    from shelf import vlm

    class FakeClient:
        def complete(self, messages, json_schema):
            return (ROOT / "examples/report.json").read_text()

    images = tmp_path / "images"
    images.mkdir()
    (images / "shelf_a.jpg").write_bytes(b"image")
    (images / "shelf_b.jpg").write_bytes(b"image")
    output = tmp_path / "result.json"
    monkeypatch.setattr(vlm, "OpenAICompatibleClient", FakeClient)
    assert (
        main(
            [
                "eval-vlm",
                str(ROOT / "tests/fixtures/annotations_mini.csv"),
                str(images),
                "--sample-size",
                "1",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["images"] == 1
    assert result["valid_first_attempt_rate"] == 1
    assert json.loads(output.read_text()) == result


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
