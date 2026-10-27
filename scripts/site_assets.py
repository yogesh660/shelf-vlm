"""Build the case-study page assets: architecture images, the gap-finder picture and chart data.

    make site

Writes site/img/*.svg and site/data.js. The gap-finder picture is the real output of
shelf.rows on the bundled test shelf; chart data comes from the bundled example planogram and
report (labelled as an example on the page) and from the SKU-110K paper's split sizes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from shelf.data.sku110k import read_annotations
from shelf.planogram import Planogram, score
from shelf.rows import find_gaps, group_rows
from shelf.schema import ShelfReport
from svgdiagram import FONT, MONO, Box, Diagram

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"

# Goldman et al., CVPR 2019 (also listed in the Ultralytics dataset docs)
SKU110K_SPLITS = {"train": 8219, "val": 588, "test": 2936}


def architecture_wide() -> str:
    d = Diagram(1240, 650, "shelf-vlm architecture")
    d.group(24, 24, 1192, 336, "Inference: photo to audit report")
    photo = d.box(
        Box(44, 150, 160, 96, "Shelf photo", "phone camera,\noptional planogram", "input", "io")
    )
    det = d.box(
        Box(
            240,
            64,
            220,
            100,
            "RT-DETR",
            "one box per product",
            "detect · M1",
            "model",
            planned=True,
        )
    )
    rows = d.box(Box(500, 64, 220, 100, "rows.py", "boxes -> shelves -> gaps", "baseline", "code"))
    vlm = d.box(
        Box(
            240,
            218,
            220,
            116,
            "Qwen3-VL + LoRA",
            "brands, categories,\nfacings, gaps",
            "read · M3",
            "model",
            planned=True,
        )
    )
    rep = d.box(
        Box(
            500,
            218,
            220,
            116,
            "ShelfReport",
            "closed JSON schema;\none retry on error",
            "contract",
            "code",
        )
    )
    comp = d.box(
        Box(
            760,
            120,
            210,
            112,
            "planogram.py",
            "missing · extra\nmisplaced · order",
            "decide",
            "code",
        )
    )
    plan = d.box(Box(760, 262, 210, 72, "Planogram", "planned layout", "input", "data"))
    out = d.box(
        Box(1000, 120, 196, 112, "Audit report", "JSON + overlay,\nshare of shelf", "output", "io")
    )
    d.arrow(photo.edge("r"), det.edge("l"), bend="hvh")
    d.arrow(photo.edge("r"), vlm.edge("l"), bend="hvh")
    d.arrow(det.edge("r"), rows.edge("l"))
    d.arrow(vlm.edge("r"), rep.edge("l"))
    d.arrow(det.edge("b"), vlm.edge("t"), "boxes as hints", dashed=True)
    d.arrow(rows.edge("r"), (comp.x, comp.cy - 20), bend="hvh")
    d.arrow(rep.edge("r"), (comp.x, comp.cy + 20), bend="hvh")
    d.arrow(plan.edge("t"), comp.edge("b"))
    d.arrow(comp.edge("r"), out.edge("l"))

    d.group(24, 392, 1192, 230, "Training and serving", planned=True)
    serve = d.box(
        Box(
            240,
            440,
            220,
            112,
            "vLLM",
            "adapter loaded by\nregistry alias",
            "serve · M4",
            "model",
            planned=True,
        )
    )
    reg = d.box(
        Box(
            500,
            440,
            220,
            112,
            "MLflow registry",
            "runs, metrics,\nadapters",
            "track",
            "ops",
            planned=True,
        )
    )
    train = d.box(
        Box(
            760,
            440,
            210,
            112,
            "Fine-tuning",
            "RT-DETR · LoRA SFT\nTRL + PEFT",
            "train · M1-M3",
            "model",
            planned=True,
        )
    )
    data = d.box(
        Box(
            1000,
            440,
            196,
            112,
            "Training data",
            "SKU-110K (loader built)\nown photos · DVC",
            "data",
            "data",
        )
    )
    d.box(Box(44, 440, 160, 112, "GPU", "Kaggle · Colab\nModal", "compute", "ops", planned=True))
    d.arrow(data.edge("l"), train.edge("r"))
    d.arrow(train.edge("l"), reg.edge("r"))
    d.arrow(reg.edge("l"), serve.edge("r"))
    d.arrow(serve.edge("t"), vlm.edge("b"), "serves")
    d.legend(244, 636)
    return d.svg()


def architecture_tall() -> str:
    d = Diagram(460, 1500, "shelf-vlm architecture")
    d.group(20, 20, 420, 860, "Inference")
    photo = d.box(
        Box(38, 60, 384, 76, "Shelf photo", "phone camera, optional planogram", "input", "io")
    )
    det = d.box(
        Box(38, 168, 186, 100, "RT-DETR", "product boxes", "detect · M1", "model", planned=True)
    )
    rows = d.box(Box(236, 168, 186, 100, "rows.py", "shelves + gaps", "baseline", "code"))
    vlm = d.box(
        Box(
            38,
            300,
            384,
            92,
            "Qwen3-VL + LoRA",
            "brands, categories, facings, gaps",
            "read · M3",
            "model",
            planned=True,
        )
    )
    rep = d.box(
        Box(
            38,
            424,
            384,
            92,
            "ShelfReport",
            "closed JSON schema; one retry on error",
            "contract",
            "code",
        )
    )
    plan = d.box(Box(38, 548, 102, 100, "Plan", "layout", "input", "data"))
    comp = d.box(
        Box(
            152,
            548,
            270,
            100,
            "planogram.py",
            "missing · extra\nmisplaced · order",
            "decide",
            "code",
        )
    )
    out = d.box(
        Box(38, 680, 384, 84, "Audit report", "JSON + overlay, share of shelf", "output", "io")
    )
    d.arrow((det.cx, photo.y + photo.h), det.edge("t"))
    d.arrow(det.edge("r"), rows.edge("l"))
    d.arrow(det.edge("b"), (det.cx, vlm.y), dashed=True)
    d.arrow(vlm.edge("b"), rep.edge("t"))
    d.arrow((comp.cx, rep.y + rep.h), (comp.cx, comp.y))
    d.arrow(comp.edge("b"), (comp.cx, out.y))
    d.arrow(plan.edge("r"), comp.edge("l"))
    # rows feed compliance directly; route down the right margin, clear of the VLM boxes
    d.path([rows.edge("r"), (432, rows.cy), (432, comp.cy), comp.edge("r")])

    d.group(20, 910, 420, 570, "Training and serving", planned=True)
    data = d.box(
        Box(
            38,
            950,
            384,
            84,
            "Training data",
            "SKU-110K (loader built) · own photos",
            "data",
            "data",
        )
    )
    train = d.box(
        Box(
            38,
            1066,
            384,
            84,
            "Fine-tuning",
            "RT-DETR · LoRA SFT · TRL + PEFT",
            "train · M1-M3",
            "model",
            planned=True,
        )
    )
    reg = d.box(
        Box(
            38,
            1182,
            384,
            84,
            "MLflow registry",
            "runs, metrics, adapters",
            "track",
            "ops",
            planned=True,
        )
    )
    serve = d.box(
        Box(
            38,
            1298,
            384,
            84,
            "vLLM",
            "serves the adapter to the VLM step",
            "serve · M4",
            "model",
            planned=True,
        )
    )
    d.box(Box(38, 1404, 384, 60, "GPU: Kaggle · Colab · Modal", "", "", "ops", planned=True))
    d.arrow(data.edge("b"), train.edge("t"))
    d.arrow(train.edge("b"), reg.edge("t"))
    d.arrow(reg.edge("b"), serve.edge("t"))
    return d.svg()


def gap_picture() -> str:
    ann = read_annotations(ROOT / "tests/fixtures/annotations_mini.csv")["shelf_a.jpg"]
    rows = group_rows(ann.boxes)
    gaps = find_gaps(rows)
    s, ox, oy = 0.9, 40, 70
    bottom = max(b.y2 for b in ann.boxes)
    w, h = int(ann.width * s + 2 * ox), int(oy + bottom * s + 56)

    def r(x1: float, y1: float, x2: float, y2: float) -> str:
        return f'x="{ox + x1 * s:.1f}" y="{oy + y1 * s:.1f}" width="{(x2 - x1) * s:.1f}" height="{(y2 - y1) * s:.1f}"'

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img">',
        "<title>Gap finder output on the bundled test shelf</title>",
        '<defs><pattern id="hatch" width="10" height="10" patternUnits="userSpaceOnUse" '
        'patternTransform="rotate(45)"><rect width="10" height="10" fill="#2a2110"/>'
        '<line x1="0" y1="0" x2="0" y2="10" stroke="#f3c26b" stroke-width="3" opacity=".55"/></pattern></defs>',
        f'<rect width="{w}" height="{h}" rx="24" fill="#0b0c12"/>',
        f'<text x="{ox}" y="40" font-family="{MONO}" font-size="13" letter-spacing="1.5" fill="#9a9aa6">'
        f"SHELF_A.JPG · {len(ann.boxes)} PRODUCTS · {len(rows)} SHELVES · {len(gaps)} GAPS</text>",
    ]
    for i, row in enumerate(rows):
        top = min(b.y1 for b in row.boxes) - 8
        bottom = max(b.y2 for b in row.boxes) + 8
        parts.append(
            f'<rect {r(-10, top, ann.width + 10, bottom)} rx="10" fill="none" stroke="#ffffff22"/>'
        )
        parts.append(
            f'<text x="{ox + 6}" y="{oy + bottom * s + 18}" font-family="{MONO}" font-size="12" '
            f'fill="#7a7a88">shelf {i}</text>'
        )
        for b in row.boxes:
            parts.append(
                f'<rect {r(b.x1, b.y1, b.x2, b.y2)} rx="6" fill="#1a1733" stroke="#8c7cff" stroke-width="1.5"/>'
            )
    for g in gaps:
        b = g.box
        parts.append(
            f'<rect {r(b.x1, b.y1, b.x2, b.y2)} rx="6" fill="url(#hatch)" stroke="#f3c26b" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{ox + b.cx * s}" y="{oy + b.cy * s + 6}" text-anchor="middle" font-family="{FONT}" '
            f'font-size="18" font-weight="700" fill="#f7d797">gap</text>'
        )
    parts.append("</svg>\n")
    return "".join(parts)


def chart_data() -> dict[str, object]:
    plan = Planogram.model_validate_json((ROOT / "examples/planogram.json").read_text())
    report = ShelfReport.model_validate_json((ROOT / "examples/report.json").read_text())
    result = score(plan, report)
    planned: dict[str, int] = {}
    for shelf in plan.shelves:
        for brand, n in shelf.counts().items():
            planned[brand] = planned.get(brand, 0) + n
    observed = report.facings_by_brand()
    brands = list(planned) + [b for b in observed if b not in planned]
    return {
        "facings": [
            {"brand": b, "planned": planned.get(b, 0), "observed": observed.get(b, 0)}
            for b in brands
        ],
        "shelves": [
            {
                "shelf": s.shelf,
                "matched": s.matched,
                "missing": sum(s.missing.values()),
                "extra": sum(s.extra.values()),
            }
            for s in result.shelves
        ],
        "share": report.share_of_shelf(),
        "compliance": round(result.score, 4),
        "splits": SKU110K_SPLITS,
    }


def main() -> None:
    (SITE / "img").mkdir(parents=True, exist_ok=True)
    (SITE / "img" / "architecture-wide.svg").write_text(architecture_wide())
    (SITE / "img" / "architecture-tall.svg").write_text(architecture_tall())
    (SITE / "img" / "gap-finder.svg").write_text(gap_picture())
    (SITE / "data.js").write_text(
        "// generated by scripts/site_assets.py - do not edit by hand\n"
        f"window.SITE_DATA = {json.dumps(chart_data(), separators=(',', ':'))};\n"
    )
    print("wrote site/img/*.svg and site/data.js")


if __name__ == "__main__":
    main()
