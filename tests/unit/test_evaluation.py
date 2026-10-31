import pytest

from shelf.data.sku110k import ImageAnnotations
from shelf.evaluation import evaluate_gaps
from shelf.geometry import Box
from shelf.metrics import Detection


def shelf(name="test.jpg", count=5):
    return ImageAnnotations(
        name, 100, 20, [Box(x1=i * 10, y1=0, x2=(i + 1) * 10, y2=20) for i in range(count)]
    )


def test_recovers_removed_product_without_mutating_annotations():
    ann = shelf()
    result = evaluate_gaps([ann])
    assert result.images == 1
    assert result.detection == Detection(1, 0, 0)
    assert len(ann.boxes) == 5


def test_skips_short_rows_and_empty_inputs():
    assert evaluate_gaps([]).images == 0
    result = evaluate_gaps([shelf(count=2), shelf(count=0)])
    assert result.skipped == 2
    assert result.detection == Detection(0, 0, 0)


def test_order_independent():
    images = [shelf("b"), shelf("a")]
    assert evaluate_gaps(images, seed=42) == evaluate_gaps(reversed(images), seed=42)


def test_unlabelled_natural_gaps_count_as_false_positives():
    ann = shelf()
    ann.boxes.append(Box(x1=90, y1=0, x2=100, y2=20))
    result = evaluate_gaps([ann])
    assert result.detection.fp >= 1


def test_no_prediction_baseline_misses_every_synthetic_gap():
    # Five tightly packed products have no gaps before removal.
    from shelf.rows import find_gaps, group_rows

    assert find_gaps(group_rows(shelf().boxes)) == []
    assert evaluate_gaps([shelf()]).detection.recall == 1.0


@pytest.mark.parametrize("threshold", [0, -1, 1.1, float("nan")])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="IoU"):
        evaluate_gaps([], threshold=threshold)
