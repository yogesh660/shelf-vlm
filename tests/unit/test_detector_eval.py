from pathlib import Path

import pytest

from shelf.data.sku110k import ImageAnnotations
from shelf.detector_eval import evaluate_detector, read_predictions
from shelf.geometry import Box
from shelf.metrics import Detection


def box(x1=0):
    return Box(x1=x1, y1=0, x2=x1 + 10, y2=10)


def test_read_predictions_filters_by_score(tmp_path: Path):
    path = tmp_path / "predictions.csv"
    path.write_text("a.jpg,0,0,10,10,0.9\na.jpg,20,0,30,10,0.2\n")
    assert read_predictions(path, 0.5) == {"a.jpg": [box()]}


def test_read_predictions_rejects_bad_rows(tmp_path: Path):
    path = tmp_path / "predictions.csv"
    path.write_text("a.jpg,0,0\n")
    with pytest.raises(ValueError, match="expected 6 columns"):
        read_predictions(path)


@pytest.mark.parametrize("score", ["-0.1", "1.1", "nan"])
def test_read_predictions_rejects_bad_scores(tmp_path: Path, score: str):
    path = tmp_path / "predictions.csv"
    path.write_text(f"a.jpg,0,0,10,10,{score}\n")
    with pytest.raises(ValueError, match="score must be"):
        read_predictions(path)


def test_evaluate_detector_includes_missing_and_unknown_images():
    gold = {
        "a.jpg": ImageAnnotations("a.jpg", 100, 100, [box(), box(20)]),
        "missing.jpg": ImageAnnotations("missing.jpg", 100, 100, [box()]),
    }
    predictions = {"a.jpg": [box(), box(40)], "unknown.jpg": [box()]}
    result = evaluate_detector(predictions, gold)
    assert result.images == 3
    assert result.detection == Detection(tp=1, fp=2, fn=2)
    assert result.counts.mae == pytest.approx(2 / 3)
    assert result.counts.mape == pytest.approx(0.5)


@pytest.mark.parametrize("threshold", [0, -1, 1.1, float("nan")])
def test_evaluate_detector_rejects_bad_iou(threshold):
    with pytest.raises(ValueError, match="IoU"):
        evaluate_detector({}, {}, threshold)
