"""Metrics used in every eval: box matching, counting error, gap detection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from shelf.geometry import Box, iou


@dataclass(frozen=True)
class Detection:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def __add__(self, other: Detection) -> Detection:
        return Detection(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)


def match_boxes(pred: list[Box], gold: list[Box], threshold: float = 0.5) -> Detection:
    """Greedy one-to-one matching by IoU, best pairs first (the usual VOC-style rule)."""
    pairs = sorted(
        ((iou(p, g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gold)),
        reverse=True,
    )
    used_pred: set[int] = set()
    used_gold: set[int] = set()
    for overlap, i, j in pairs:
        if overlap < threshold:
            break
        if i in used_pred or j in used_gold:
            continue
        used_pred.add(i)
        used_gold.add(j)
    tp = len(used_pred)
    return Detection(tp=tp, fp=len(pred) - tp, fn=len(gold) - tp)


@dataclass(frozen=True)
class CountError:
    mae: float
    mape: float  # over keys with a non-zero gold count


def count_error(pred: Mapping[str, int], gold: Mapping[str, int]) -> CountError:
    """Per-key counting error, e.g. facings per brand or products per image."""
    keys = set(pred) | set(gold)
    if not keys:
        return CountError(0.0, 0.0)
    errors = [abs(pred.get(k, 0) - gold.get(k, 0)) for k in keys]
    relative = [abs(pred.get(k, 0) - g) / g for k, g in gold.items() if g]
    return CountError(
        mae=sum(errors) / len(errors),
        mape=sum(relative) / len(relative) if relative else 0.0,
    )
