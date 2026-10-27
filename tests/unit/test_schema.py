import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from shelf.schema import ShelfReport

EXAMPLE = Path(__file__).parents[2] / "examples" / "report.json"


@pytest.fixture
def report():
    return ShelfReport.model_validate_json(EXAMPLE.read_text())


def test_example_is_valid(report):
    assert report.shelf_count == 3
    assert len(report.products) == 6


def test_facings_and_share(report):
    counts = report.facings_by_brand()
    assert counts["Dove"] == 4
    assert counts["Sunsilk"] == 3
    share = report.share_of_shelf()
    assert sum(share.values()) == pytest.approx(1.0)
    assert share["Dove"] == pytest.approx(4 / 16)


def test_shelf_sequence_expands_facings(report):
    assert report.shelf_sequence(1) == ["Head & Shoulders"] * 3 + ["Sunsilk"] * 2


def test_empty_report_has_no_share():
    assert ShelfReport(shelf_count=1, products=[]).share_of_shelf() == {}


def product(shelf, position, brand="X"):
    return {"brand": brand, "shelf": shelf, "position": position}


@pytest.mark.parametrize(
    "data",
    [
        {"shelf_count": 2, "products": [product(2, 0)]},  # shelf out of range
        {"shelf_count": 1, "products": [product(0, 0), product(0, 0, "Y")]},  # same slot twice
        {"shelf_count": 1, "products": [], "mood": "happy"},  # unknown field
        {"shelf_count": 0, "products": []},
    ],
)
def test_invalid_reports(data):
    with pytest.raises(ValidationError):
        ShelfReport.model_validate(data)


def test_schema_is_closed_for_guided_decoding():
    # vLLM's guided decoding and OpenAI strict mode both need additionalProperties: false
    schema = ShelfReport.model_json_schema()
    assert schema["additionalProperties"] is False
    for name in ("Product", "Gap"):
        assert schema["$defs"][name]["additionalProperties"] is False
    json.dumps(schema)
