"""Group detector boxes into shelf rows and find gaps between products.

This is the non-VLM baseline for gap detection: products on one shelf sit at roughly the
same height, and an empty slot shows up as horizontal space wider than a typical product.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from shelf.geometry import Box, vertical_overlap

# a box joins a row if it overlaps the row's typical box vertically by at least this much
ROW_OVERLAP = 0.5
# space between neighbours counts as a gap when wider than this many median product widths
GAP_FACTOR = 0.8


@dataclass
class Row:
    boxes: list[Box] = field(default_factory=list)

    @property
    def y_center(self) -> float:
        return statistics.median(b.cy for b in self.boxes)

    @property
    def median_height(self) -> float:
        return statistics.median(b.height for b in self.boxes)

    def reference(self) -> Box:
        # a box with the row's median vertical extent; one tall pack can't drag the row
        half = self.median_height / 2
        x1 = min(b.x1 for b in self.boxes)
        return Box(x1=x1, y1=self.y_center - half, x2=x1 + 1, y2=self.y_center + half)


def group_rows(boxes: list[Box], min_overlap: float = ROW_OVERLAP) -> list[Row]:
    """Rows top to bottom, boxes in each row left to right."""
    rows: list[Row] = []
    for box in sorted(boxes, key=lambda b: b.cy):
        best = max(rows, key=lambda r: vertical_overlap(box, r.reference()), default=None)
        if best is not None and vertical_overlap(box, best.reference()) >= min_overlap:
            best.boxes.append(box)
        else:
            rows.append(Row([box]))
    for row in rows:
        row.boxes.sort(key=lambda b: b.x1)
    rows.sort(key=lambda r: r.y_center)
    return rows


@dataclass(frozen=True)
class FoundGap:
    shelf: int
    after_position: int  # -1 = before the first product
    box: Box


def find_gaps(
    rows: list[Row],
    shelf_x1: float | None = None,
    shelf_x2: float | None = None,
    factor: float = GAP_FACTOR,
) -> list[FoundGap]:
    """Gaps between neighbours, and at the row ends if the shelf extent is known.

    shelf_x1/x2 default to the leftmost/rightmost product over all rows, which catches an
    empty end on one shelf when the shelves above and below are full.
    """
    all_boxes = [b for r in rows for b in r.boxes]
    if not all_boxes:
        return []
    left = min(b.x1 for b in all_boxes) if shelf_x1 is None else shelf_x1
    right = max(b.x2 for b in all_boxes) if shelf_x2 is None else shelf_x2

    gaps = []
    for shelf, row in enumerate(rows):
        min_width = factor * statistics.median(b.width for b in row.boxes)
        top = min(b.y1 for b in row.boxes)
        bottom = max(b.y2 for b in row.boxes)
        edges = [left, *[x for b in row.boxes for x in (b.x1, b.x2)], right]
        # pairs: (left edge, first x1), (box0 x2, box1 x1), ..., (last x2, right edge)
        for i in range(0, len(edges), 2):
            start, end = edges[i], edges[i + 1]
            if end - start > min_width:
                gaps.append(FoundGap(shelf, i // 2 - 1, Box(x1=start, y1=top, x2=end, y2=bottom)))
    return gaps
