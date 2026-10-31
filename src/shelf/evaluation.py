"""Reproducible, annotation-only synthetic-gap baseline (no model or images needed)."""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass

from shelf.data.sku110k import ImageAnnotations
from shelf.metrics import Detection, match_boxes
from shelf.rows import find_gaps, group_rows


@dataclass(frozen=True)
class GapEvaluation:
    images: int
    skipped: int
    detection: Detection


def evaluate_gaps(
    images: Iterable[ImageAnnotations], *, seed: int = 0, threshold: float = 0.5
) -> GapEvaluation:
    """Remove one interior product per image and match found gaps to its original box.

    Images without a row of at least three products are skipped. Selection is stable under
    input reordering. All predictions count: natural gaps are unlabelled in SKU-110K, so
    precision measures the synthetic task, not real-world empty-slot precision.
    """
    if not 0 < threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    rng = random.Random(seed)
    total = Detection(0, 0, 0)
    evaluated = skipped = 0
    for ann in sorted(images, key=lambda a: a.image):
        rows = group_rows(ann.boxes)
        candidates = [b for row in rows for b in row.boxes[1:-1]]
        if not candidates:
            skipped += 1
            continue
        removed = rng.choice(candidates)
        remaining = list(ann.boxes)
        remaining.remove(removed)
        predicted = [gap.box for gap in find_gaps(group_rows(remaining))]
        total += match_boxes(predicted, [removed], threshold)
        evaluated += 1
    return GapEvaluation(evaluated, skipped, total)
