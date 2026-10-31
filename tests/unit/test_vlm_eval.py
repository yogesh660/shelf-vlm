import json
from pathlib import Path

import pytest

from shelf.data.sku110k import ImageAnnotations
from shelf.geometry import Box
from shelf.vlm_eval import evaluate_vlm


def report(facings: int) -> str:
    return json.dumps(
        {
            "shelf_count": 1,
            "products": [
                {
                    "brand": "unknown",
                    "category": None,
                    "shelf": 0,
                    "position": 0,
                    "facings": facings,
                    "box": None,
                }
            ],
            "gaps": [],
        }
    )


class FakeClient:
    def __init__(self, replies):
        self.replies = iter(replies)

    def complete(self, messages, json_schema):
        return next(self.replies)


def annotations(names=("a.jpg", "b.jpg")):
    boxes = [Box(x1=0, y1=0, x2=10, y2=10), Box(x1=10, y1=0, x2=20, y2=10)]
    return {name: ImageAnnotations(name, 20, 10, list(boxes)) for name in names}


def test_evaluates_validity_retries_and_count_error(tmp_path: Path):
    for name in ("a.jpg", "b.jpg"):
        (tmp_path / name).write_bytes(b"image")
    result = evaluate_vlm(FakeClient([report(2), "bad", report(1)]), annotations(), tmp_path)
    assert result.images == 2
    assert result.valid_first_attempt == 1
    assert result.valid_after_retry == 2
    assert result.counts.mae == 0.5
    assert result.counts.mape == 0.25


def test_invalid_final_reply_counts_as_zero(tmp_path: Path):
    (tmp_path / "a.jpg").write_bytes(b"image")
    result = evaluate_vlm(FakeClient(["bad", "still bad"]), annotations(("a.jpg",)), tmp_path)
    assert result.valid_after_retry == 0
    assert result.counts.mae == 2


def test_sampling_is_seeded_and_order_independent(tmp_path: Path):
    anns = annotations(("c.jpg", "a.jpg", "b.jpg"))
    for name in anns:
        (tmp_path / name).write_bytes(b"image")
    first = evaluate_vlm(FakeClient([report(2)]), anns, tmp_path, sample_size=1, seed=7)
    second = evaluate_vlm(
        FakeClient([report(2)]), dict(reversed(list(anns.items()))), tmp_path, sample_size=1, seed=7
    )
    assert first.results[0].image == second.results[0].image


def test_rejects_nonpositive_sample_size(tmp_path: Path):
    with pytest.raises(ValueError, match="positive"):
        evaluate_vlm(FakeClient([]), {}, tmp_path, sample_size=0)
