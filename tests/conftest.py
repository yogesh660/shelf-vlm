from pathlib import Path

import pytest

from shelf.data.sku110k import read_annotations

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def mini_annotations():
    return read_annotations(FIXTURES / "annotations_mini.csv")
