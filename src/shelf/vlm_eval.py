"""Reproducible zero-shot VLM evaluation over an annotated image sample."""

from __future__ import annotations

import mimetypes
import random
from dataclasses import dataclass
from pathlib import Path

from shelf.data.sku110k import ImageAnnotations
from shelf.metrics import CountError, count_error
from shelf.schema import ShelfReport
from shelf.vlm import ChatClient, extract_report


@dataclass(frozen=True)
class VLMImageResult:
    image: str
    gold_count: int
    predicted_count: int
    attempts: int
    errors: tuple[str, ...]
    report: ShelfReport | None


@dataclass(frozen=True)
class VLMEvaluation:
    results: tuple[VLMImageResult, ...]
    counts: CountError

    @property
    def images(self) -> int:
        return len(self.results)

    @property
    def valid_first_attempt(self) -> int:
        return sum(result.report is not None and result.attempts == 1 for result in self.results)

    @property
    def valid_after_retry(self) -> int:
        return sum(result.report is not None for result in self.results)


def evaluate_vlm(
    client: ChatClient,
    annotations: dict[str, ImageAnnotations],
    images: Path,
    *,
    sample_size: int = 50,
    seed: int = 0,
) -> VLMEvaluation:
    """Evaluate validity and total-facing count error on a fixed sample.

    Invalid final replies receive a predicted count of zero so count error reflects the
    end-to-end pipeline rather than silently excluding failures.
    """
    if sample_size <= 0:
        raise ValueError("sample size must be positive")
    population = sorted(annotations.values(), key=lambda ann: ann.image)
    selected = random.Random(seed).sample(population, min(sample_size, len(population)))
    selected.sort(key=lambda ann: ann.image)
    results = []
    for ann in selected:
        path = images / ann.image
        mime = mimetypes.guess_type(path)[0] or "image/jpeg"
        extraction = extract_report(client, path.read_bytes(), mime)
        predicted = (
            sum(product.facings for product in extraction.report.products)
            if extraction.report is not None
            else 0
        )
        results.append(
            VLMImageResult(
                image=ann.image,
                gold_count=len(ann.boxes),
                predicted_count=predicted,
                attempts=extraction.attempts,
                errors=tuple(extraction.errors),
                report=extraction.report,
            )
        )
    predicted_counts = {result.image: result.predicted_count for result in results}
    gold_counts = {result.image: result.gold_count for result in results}
    return VLMEvaluation(tuple(results), count_error(predicted_counts, gold_counts))
