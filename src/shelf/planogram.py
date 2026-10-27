"""Planogram compliance: what the shelf should look like vs what it looks like.

Scored per shelf on facings per brand, which is what a field auditor checks:
  missing    planned facings that aren't there
  extra      facings that aren't in the plan for that shelf
  misplaced  the extra/missing pair is the same brand on another shelf (wrong shelf,
             not a missing product)
  order      do the brand blocks appear left to right in the planned order?
"""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, ConfigDict, Field

from shelf.schema import ShelfReport


class Slot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    brand: str
    facings: int = Field(ge=1)


class PlannedShelf(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slots: list[Slot] = Field(description="Left to right.")

    def counts(self) -> Counter[str]:
        c: Counter[str] = Counter()
        for slot in self.slots:
            c[slot.brand] += slot.facings
        return c

    def brand_order(self) -> list[str]:
        return _blocks([s.brand for s in self.slots])


class Planogram(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = ""
    shelves: list[PlannedShelf] = Field(description="Top to bottom.")


class ShelfResult(BaseModel):
    shelf: int
    expected: int
    observed: int
    matched: int
    missing: dict[str, int]
    extra: dict[str, int]
    misplaced: dict[str, int]
    order_ok: bool


class ComplianceResult(BaseModel):
    score: float = Field(description="Matched facings / max(planned, observed). 1.0 = perfect.")
    shelves: list[ShelfResult]
    shelf_count_ok: bool

    @property
    def issues(self) -> int:
        return sum(
            sum(s.missing.values()) + sum(s.extra.values()) + (not s.order_ok) for s in self.shelves
        )


def _blocks(seq: list[str]) -> list[str]:
    """['A', 'A', 'B', 'A'] -> ['A', 'B', 'A']"""
    return [b for i, b in enumerate(seq) if i == 0 or seq[i - 1] != b]


def _order_ok(planned: list[str], observed: list[str]) -> bool:
    # planned brands must keep their relative order among what was observed; a missing
    # brand is already counted as missing, it shouldn't also count as an order error
    present = [b for b in planned if b in observed]
    seen = [b for b in observed if b in planned]
    return _blocks(seen) == _blocks(present)


def score(plan: Planogram, report: ShelfReport) -> ComplianceResult:
    n = max(len(plan.shelves), report.shelf_count)
    expected_by_shelf = [
        plan.shelves[i].counts() if i < len(plan.shelves) else Counter[str]() for i in range(n)
    ]
    observed_by_shelf = [Counter(report.shelf_sequence(i)) for i in range(n)]

    results = []
    for i in range(n):
        exp, obs = expected_by_shelf[i], observed_by_shelf[i]
        missing = exp - obs
        extra = obs - exp
        # an extra facing of a brand that is missing on another shelf was put on the wrong shelf
        missing_elsewhere: Counter[str] = Counter()
        for j in range(n):
            if j != i:
                missing_elsewhere += expected_by_shelf[j] - observed_by_shelf[j]
        misplaced = extra & missing_elsewhere
        planned_order = plan.shelves[i].brand_order() if i < len(plan.shelves) else []
        results.append(
            ShelfResult(
                shelf=i,
                expected=sum(exp.values()),
                observed=sum(obs.values()),
                matched=sum((exp & obs).values()),
                missing=dict(missing),
                extra=dict(extra),
                misplaced=dict(misplaced),
                order_ok=_order_ok(planned_order, _blocks(report.shelf_sequence(i))),
            )
        )

    expected_total = sum(r.expected for r in results)
    observed_total = sum(r.observed for r in results)
    matched_total = sum(r.matched for r in results)
    denom = max(expected_total, observed_total)
    return ComplianceResult(
        score=matched_total / denom if denom else 1.0,
        shelves=results,
        shelf_count_ok=len(plan.shelves) == report.shelf_count,
    )
