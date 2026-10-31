"""Evaluate detector prediction CSVs against SKU-110K annotations."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from shelf.data.sku110k import ImageAnnotations
from shelf.geometry import Box
from shelf.metrics import CountError, Detection, count_error, match_boxes


@dataclass(frozen=True)
class DetectorEvaluation:
    images: int
    detection: Detection
    counts: CountError


def read_predictions(csv_path: Path, score_threshold: float = 0.0) -> dict[str, list[Box]]:
    """Read headerless rows: image,x1,y1,x2,y2,score."""
    if not 0 <= score_threshold <= 1:
        raise ValueError("score threshold must be in [0, 1]")
    images: dict[str, list[Box]] = {}
    with csv_path.open(newline="") as fh:
        for line, row in enumerate(csv.reader(fh), 1):
            if not row:
                continue
            if len(row) != 6:
                raise ValueError(f"{csv_path}:{line}: expected 6 columns, got {len(row)}")
            name, x1, y1, x2, y2, score = row
            try:
                confidence = float(score)
                box = Box(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
            except ValueError as exc:
                raise ValueError(f"{csv_path}:{line}: invalid prediction") from exc
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError(f"{csv_path}:{line}: score must be in [0, 1]")
            if confidence >= score_threshold:
                images.setdefault(name, []).append(box)
    return images


def evaluate_detector(
    predictions: dict[str, list[Box]],
    gold: dict[str, ImageAnnotations],
    iou_threshold: float = 0.5,
) -> DetectorEvaluation:
    """Micro box P/R/F1 and per-image count MAE/MAPE over the union of image names."""
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    names = set(predictions) | set(gold)
    detection = Detection(0, 0, 0)
    for name in names:
        detection += match_boxes(
            predictions.get(name, []), gold[name].boxes if name in gold else [], iou_threshold
        )
    pred_counts = {name: len(boxes) for name, boxes in predictions.items()}
    gold_counts = {name: len(ann.boxes) for name, ann in gold.items()}
    return DetectorEvaluation(len(names), detection, count_error(pred_counts, gold_counts))
