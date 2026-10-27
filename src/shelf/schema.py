"""The report a shelf analysis produces, from the VLM or from detector + rules.

This is also the JSON schema the VLM is forced to answer in (guided decoding, ADR 0002),
so field names and descriptions are written for the model as much as for people.
"""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, ConfigDict, Field, model_validator

from shelf.geometry import Box


class Product(BaseModel):
    model_config = ConfigDict(extra="forbid")

    brand: str = Field(description="Brand name as printed on the pack, e.g. 'Dove'.")
    category: str | None = Field(
        default=None, description="Product category, e.g. 'shampoo'. Null if unclear."
    )
    shelf: int = Field(ge=0, description="Shelf index, 0 is the top shelf.")
    position: int = Field(ge=0, description="Left-to-right position on the shelf, from 0.")
    facings: int = Field(
        default=1, ge=1, description="Identical packs side by side, counted as one entry."
    )
    box: Box | None = Field(default=None, description="Pixel box if known, else null.")


class Gap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shelf: int = Field(ge=0, description="Shelf index, 0 is the top shelf.")
    after_position: int = Field(
        ge=-1, description="Position of the product left of the gap; -1 if at the left edge."
    )
    box: Box | None = Field(default=None, description="Pixel box if known, else null.")


class ShelfReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shelf_count: int = Field(ge=1, description="Number of shelves visible in the photo.")
    products: list[Product] = Field(description="Every product facing group, all shelves.")
    gaps: list[Gap] = Field(default_factory=list, description="Empty spaces on the shelves.")

    @model_validator(mode="after")
    def _consistent(self) -> ShelfReport:
        shelves = [p.shelf for p in self.products] + [g.shelf for g in self.gaps]
        for shelf in shelves:
            if shelf >= self.shelf_count:
                raise ValueError(f"shelf {shelf} out of range (shelf_count={self.shelf_count})")
        seen: set[tuple[int, int]] = set()
        for p in self.products:
            if (p.shelf, p.position) in seen:
                raise ValueError(f"two products at shelf {p.shelf}, position {p.position}")
            seen.add((p.shelf, p.position))
        return self

    def facings_by_brand(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for p in self.products:
            counts[p.brand] += p.facings
        return counts

    def share_of_shelf(self) -> dict[str, float]:
        """Share of facings per brand. The standard retail-audit definition, not shelf length."""
        counts = self.facings_by_brand()
        total = sum(counts.values())
        return {brand: n / total for brand, n in counts.most_common()} if total else {}

    def shelf_sequence(self, shelf: int) -> list[str]:
        """Brands on one shelf, left to right, one entry per facing."""
        row = sorted((p for p in self.products if p.shelf == shelf), key=lambda p: p.position)
        return [p.brand for p in row for _ in range(p.facings)]
