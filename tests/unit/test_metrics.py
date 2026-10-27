import pytest

from shelf.geometry import Box
from shelf.metrics import Detection, count_error, match_boxes


def box(x):
    return Box(x1=x, y1=0, x2=x + 10, y2=10)


def test_perfect_match():
    boxes = [box(0), box(20)]
    d = match_boxes(boxes, boxes)
    assert (d.tp, d.fp, d.fn) == (2, 0, 0)
    assert d.f1 == 1.0


def test_each_gold_box_matched_once():
    # two predictions on the same product: one tp, one fp
    d = match_boxes([box(0), box(1)], [box(0)])
    assert (d.tp, d.fp, d.fn) == (1, 1, 0)


def test_below_threshold_is_a_miss():
    d = match_boxes([box(6)], [box(0)], threshold=0.5)
    assert (d.tp, d.fp, d.fn) == (0, 1, 1)


def test_empty():
    assert match_boxes([], []).f1 == 0.0
    assert match_boxes([], [box(0)]).recall == 0.0


def test_detections_add_up():
    total = Detection(1, 2, 3) + Detection(4, 5, 6)
    assert (total.tp, total.fp, total.fn) == (5, 7, 9)


def test_count_error():
    err = count_error({"A": 8, "B": 2}, {"A": 10, "B": 2, "C": 1})
    assert err.mae == pytest.approx((2 + 0 + 1) / 3)
    assert err.mape == pytest.approx((0.2 + 0 + 1.0) / 3)
    assert count_error({}, {}).mae == 0.0
