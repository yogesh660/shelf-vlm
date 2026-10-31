"""shelf command line.

shelf schema                                   JSON schema the VLM answers in
shelf data download --dest data/sku110k        fetch SKU-110K
shelf data stats --dest data/sku110k           image and box counts per split
shelf rows ANNOTATIONS.csv IMAGE               shelf rows + gaps from ground-truth boxes
shelf score --planogram P.json --report R.json planogram compliance
shelf analyse IMAGE                            ask the configured VLM (see README)
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from pathlib import Path

from shelf.schema import ShelfReport


def cmd_schema(_: argparse.Namespace) -> int:
    print(json.dumps(ShelfReport.model_json_schema(), indent=2))
    return 0


def cmd_eval_gaps(args: argparse.Namespace) -> int:
    from shelf.data.sku110k import read_annotations
    from shelf.evaluation import evaluate_gaps

    if not 0 < args.iou <= 1:
        print("IoU threshold must be in (0, 1]", file=sys.stderr)
        return 1
    result = evaluate_gaps(
        read_annotations(args.annotations).values(), seed=args.seed, threshold=args.iou
    )
    detection = result.detection
    print(
        json.dumps(
            {
                "protocol": "one-interior-product-removed-v1",
                "seed": args.seed,
                "iou_threshold": args.iou,
                "images": result.images,
                "skipped": result.skipped,
                "tp": detection.tp,
                "fp": detection.fp,
                "fn": detection.fn,
                "precision": detection.precision,
                "recall": detection.recall,
                "f1": detection.f1,
            },
            indent=2,
        )
    )
    return 0 if result.images else 1


def cmd_eval_detector(args: argparse.Namespace) -> int:
    from shelf.data.sku110k import read_annotations
    from shelf.detector_eval import evaluate_detector, read_predictions

    try:
        predictions = read_predictions(args.predictions, args.score)
        result = evaluate_detector(predictions, read_annotations(args.annotations), args.iou)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    detection = result.detection
    print(
        json.dumps(
            {
                "score_threshold": args.score,
                "iou_threshold": args.iou,
                "images": result.images,
                "tp": detection.tp,
                "fp": detection.fp,
                "fn": detection.fn,
                "precision": detection.precision,
                "recall": detection.recall,
                "f1": detection.f1,
                "count_mae": result.counts.mae,
                "count_mape": result.counts.mape,
            },
            indent=2,
        )
    )
    return 0 if result.images else 1


def cmd_eval_vlm(args: argparse.Namespace) -> int:
    from shelf.data.sku110k import read_annotations
    from shelf.vlm import OpenAICompatibleClient
    from shelf.vlm_eval import evaluate_vlm

    try:
        result = evaluate_vlm(
            OpenAICompatibleClient(),
            read_annotations(args.annotations),
            args.images,
            sample_size=args.sample_size,
            seed=args.seed,
        )
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    payload = {
        "protocol": "zero-shot-vlm-v1",
        "seed": args.seed,
        "images": result.images,
        "valid_first_attempt": result.valid_first_attempt,
        "valid_first_attempt_rate": (
            result.valid_first_attempt / result.images if result.images else 0.0
        ),
        "valid_after_retry": result.valid_after_retry,
        "valid_after_retry_rate": (
            result.valid_after_retry / result.images if result.images else 0.0
        ),
        "count_mae": result.counts.mae,
        "count_mape": result.counts.mape,
        "results": [
            {
                "image": item.image,
                "gold_count": item.gold_count,
                "predicted_count": item.predicted_count,
                "attempts": item.attempts,
                "errors": item.errors,
                "report": item.report.model_dump(mode="json") if item.report else None,
            }
            for item in result.results
        ],
    }
    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0 if result.images else 1


def cmd_data_download(args: argparse.Namespace) -> int:
    from shelf.data.sku110k import download

    download(args.dest)
    return 0


def cmd_data_stats(args: argparse.Namespace) -> int:
    from shelf.data.sku110k import read_annotations, split_paths, stats

    for split, path in split_paths(args.dest).items():
        if not path.exists():
            print(f"{split:<6} missing ({path}); run `make data` first")
            continue
        s = stats(read_annotations(path))
        print(
            f"{split:<6} {s.images:>6} images {s.boxes:>9} boxes  "
            f"median {s.boxes_per_image_median:.0f}/image  max {s.boxes_per_image_max}"
        )
    return 0


def cmd_rows(args: argparse.Namespace) -> int:
    from shelf.data.sku110k import read_annotations
    from shelf.rows import find_gaps, group_rows

    images = read_annotations(args.annotations)
    if args.image not in images:
        print(f"{args.image} not in {args.annotations}", file=sys.stderr)
        return 1
    ann = images[args.image]
    rows = group_rows(ann.boxes)
    gaps = find_gaps(rows)
    print(f"{args.image}: {len(ann.boxes)} products on {len(rows)} shelves")
    for i, row in enumerate(rows):
        in_row = [g for g in gaps if g.shelf == i]
        print(f"  shelf {i}: {len(row.boxes):>3} products, {len(in_row)} gap(s)")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    from shelf.planogram import Planogram, score

    plan = Planogram.model_validate_json(args.planogram.read_text())
    report = ShelfReport.model_validate_json(args.report.read_text())
    result = score(plan, report)
    print(f"compliance {result.score:.0%}  ({result.issues} issue(s))")
    for s in result.shelves:
        notes = []
        if s.missing:
            notes.append(f"missing {s.missing}")
        if s.misplaced:
            notes.append(f"misplaced {s.misplaced}")
        if s.extra:
            notes.append(f"extra {s.extra}")
        if not s.order_ok:
            notes.append("out of order")
        print(f"  shelf {s.shelf}: {s.matched}/{s.expected} planned facings  {'; '.join(notes)}")
    print("\nshare of shelf:")
    for brand, share in report.share_of_shelf().items():
        print(f"  {brand:<16} {share:.0%}")
    return 0


def cmd_analyse(args: argparse.Namespace) -> int:
    from shelf.vlm import OpenAICompatibleClient, extract_report

    mime = mimetypes.guess_type(args.image)[0] or "image/jpeg"
    result = extract_report(OpenAICompatibleClient(), args.image.read_bytes(), mime)
    if not result.ok or result.report is None:
        print(f"no valid report after {result.attempts} attempts: {result.errors}", file=sys.stderr)
        return 1
    print(result.report.model_dump_json(indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="shelf")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("schema", help="print the report JSON schema").set_defaults(func=cmd_schema)

    p = sub.add_parser("eval-gaps", help="evaluate synthetic gaps from annotation boxes")
    p.add_argument("annotations", type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iou", type=float, default=0.5)
    p.set_defaults(func=cmd_eval_gaps)

    p = sub.add_parser("eval-detector", help="evaluate detector prediction boxes")
    p.add_argument("annotations", type=Path)
    p.add_argument("predictions", type=Path)
    p.add_argument("--score", type=float, default=0.5)
    p.add_argument("--iou", type=float, default=0.5)
    p.set_defaults(func=cmd_eval_detector)

    p = sub.add_parser("eval-vlm", help="evaluate zero-shot VLM validity and counting")
    p.add_argument("annotations", type=Path)
    p.add_argument("images", type=Path)
    p.add_argument("--sample-size", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path)
    p.set_defaults(func=cmd_eval_vlm)

    data = sub.add_parser("data", help="SKU-110K dataset").add_subparsers(required=True)
    p = data.add_parser("download")
    p.add_argument("--dest", type=Path, default=Path("data/sku110k"))
    p.set_defaults(func=cmd_data_download)
    p = data.add_parser("stats")
    p.add_argument("--dest", type=Path, default=Path("data/sku110k"))
    p.set_defaults(func=cmd_data_stats)

    p = sub.add_parser("rows", help="shelf rows and gaps from annotation boxes")
    p.add_argument("annotations", type=Path)
    p.add_argument("image")
    p.set_defaults(func=cmd_rows)

    p = sub.add_parser("score", help="planogram compliance of a report")
    p.add_argument("--planogram", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("analyse", help="ask the configured VLM about a shelf photo")
    p.add_argument("image", type=Path)
    p.set_defaults(func=cmd_analyse)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
