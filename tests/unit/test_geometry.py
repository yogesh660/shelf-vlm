import pytest
from pydantic import ValidationError

from shelf.geometry import Box, iou, vertical_overlap


def box(x1, y1, x2, y2):
    return Box(x1=x1, y1=y1, x2=x2, y2=y2)


def test_iou():
    a = box(0, 0, 10, 10)
    assert iou(a, a) == 1.0
    assert iou(a, box(20, 20, 30, 30)) == 0.0
    assert iou(a, box(5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_touching_boxes_do_not_overlap():
    assert iou(box(0, 0, 10, 10), box(10, 0, 20, 10)) == 0.0


def test_vertical_overlap_uses_shorter_box():
    tall = box(0, 0, 10, 100)
    short = box(20, 40, 30, 60)
    assert vertical_overlap(tall, short) == 1.0
    assert vertical_overlap(box(0, 0, 10, 10), box(0, 5, 10, 15)) == 0.5


@pytest.mark.parametrize("coords", [(0, 0, 0, 10), (5, 0, 1, 10), (0, 10, 10, 10)])
def test_degenerate_boxes_rejected(coords):
    with pytest.raises(ValidationError):
        box(*coords)
