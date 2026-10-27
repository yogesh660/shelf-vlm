import json
from pathlib import Path

import pytest

from shelf.planogram import Planogram, score
from shelf.schema import ShelfReport

EXAMPLES = Path(__file__).parents[2] / "examples"


def plan(*shelves):
    return Planogram(
        shelves=[{"slots": [{"brand": b, "facings": f} for b, f in s]} for s in shelves]
    )


def report(*shelves):
    products = [
        {"brand": b, "shelf": i, "position": j, "facings": f}
        for i, s in enumerate(shelves)
        for j, (b, f) in enumerate(s)
    ]
    return ShelfReport(shelf_count=len(shelves), products=products)


def test_perfect_shelf():
    p = plan([("A", 2), ("B", 2)])
    result = score(p, report([("A", 2), ("B", 2)]))
    assert result.score == 1.0
    assert result.issues == 0


def test_missing_facings():
    result = score(plan([("A", 4)]), report([("A", 1)]))
    assert result.shelves[0].missing == {"A": 3}
    assert result.score == pytest.approx(0.25)


def test_wrong_shelf_is_misplaced_not_just_extra():
    p = plan([("A", 2)], [("B", 2)])
    r = report([("A", 2), ("B", 1)], [("B", 1)])
    result = score(p, r)
    assert result.shelves[0].misplaced == {"B": 1}
    assert result.shelves[1].missing == {"B": 1}


def test_unplanned_brand_is_extra_but_not_misplaced():
    result = score(plan([("A", 2)]), report([("A", 2), ("Z", 1)]))
    assert result.shelves[0].extra == {"Z": 1}
    assert result.shelves[0].misplaced == {}
    assert result.score == pytest.approx(2 / 3)


def test_order():
    p = plan([("A", 1), ("B", 1), ("C", 1)])
    assert score(p, report([("A", 1), ("B", 1), ("C", 1)])).shelves[0].order_ok
    assert not score(p, report([("B", 1), ("A", 1), ("C", 1)])).shelves[0].order_ok
    # a missing brand is a missing issue, not an order issue as well
    assert score(p, report([("A", 1), ("C", 1)])).shelves[0].order_ok


def test_shelf_count_mismatch():
    result = score(plan([("A", 1)], [("B", 1)]), report([("A", 1)]))
    assert not result.shelf_count_ok
    assert result.shelves[1].missing == {"B": 1}


def test_bundled_example():
    p = Planogram.model_validate(json.loads((EXAMPLES / "planogram.json").read_text()))
    r = ShelfReport.model_validate_json((EXAMPLES / "report.json").read_text())
    result = score(p, r)
    assert result.score == pytest.approx(15 / 19)
    assert result.shelves[0].misplaced == {"Sunsilk": 1}
