"""Boxes in pixel coordinates, (x1, y1) top-left, (x2, y2) bottom-right."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator


class Box(BaseModel):
    model_config = ConfigDict(frozen=True)

    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def _ordered(self) -> Box:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(f"degenerate box {self.x1, self.y1, self.x2, self.y2}")
        return self

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2


def iou(a: Box, b: Box) -> float:
    w = min(a.x2, b.x2) - max(a.x1, b.x1)
    h = min(a.y2, b.y2) - max(a.y1, b.y1)
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / (a.area + b.area - inter)


def vertical_overlap(a: Box, b: Box) -> float:
    """Overlap of the y-ranges as a fraction of the shorter box's height."""
    overlap = min(a.y2, b.y2) - max(a.y1, b.y1)
    return max(overlap, 0.0) / min(a.height, b.height)
