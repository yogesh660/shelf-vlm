"""SKU-110K: ~11.7k densely packed shelf images with product boxes (Goldman et al., CVPR 2019).

Archive layout after unpacking:
    SKU110K_fixed/images/{train,val,test}_*.jpg
    SKU110K_fixed/annotations/annotations_{train,val,test}.csv

The CSVs have no header. Columns: image, x1, y1, x2, y2, class, image_width, image_height.
Every box has the same class ("object"); there are no brand labels, which is why the
project adds its own labelled photos later (docs/ROADMAP.md, M2).

The dataset is for research use; check the terms in the original repository before
using it for anything else: https://github.com/eg4000/SKU110K_CVPR19
"""

from __future__ import annotations

import csv
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from shelf.geometry import Box

URL = "http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz"
SPLITS = ("train", "val", "test")


@dataclass
class ImageAnnotations:
    image: str
    width: int
    height: int
    boxes: list[Box] = field(default_factory=list)


def read_annotations(csv_path: Path) -> dict[str, ImageAnnotations]:
    images: dict[str, ImageAnnotations] = {}
    with csv_path.open(newline="") as fh:
        for row in csv.reader(fh):
            if not row:
                continue
            name, x1, y1, x2, y2, _cls, w, h = row
            ann = images.setdefault(name, ImageAnnotations(name, int(w), int(h)))
            try:
                ann.boxes.append(Box(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)))
            except ValueError:
                continue  # the original release has a handful of zero-size boxes
    return images


def split_paths(root: Path) -> dict[str, Path]:
    base = root / "SKU110K_fixed" / "annotations"
    return {s: base / f"annotations_{s}.csv" for s in SPLITS}


@dataclass(frozen=True)
class SplitStats:
    images: int
    boxes: int
    boxes_per_image_median: float
    boxes_per_image_max: int


def stats(images: dict[str, ImageAnnotations]) -> SplitStats:
    counts = sorted(len(a.boxes) for a in images.values())
    if not counts:
        return SplitStats(0, 0, 0.0, 0)
    mid = len(counts) // 2
    median = counts[mid] if len(counts) % 2 else (counts[mid - 1] + counts[mid]) / 2
    return SplitStats(len(counts), sum(counts), float(median), counts[-1])


def download(dest: Path, url: str = URL) -> Path:
    """Download and unpack into dest. Skips work that is already done."""
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / Path(url).name
    if all(p.exists() for p in split_paths(dest).values()):
        print(f"already unpacked in {dest}")
        return dest
    if not archive.exists():
        part = archive.with_suffix(".part")
        print(f"downloading {url} (about 13.6 GB) ...")
        with urllib.request.urlopen(url) as resp, part.open("wb") as out:
            shutil.copyfileobj(resp, out, length=1 << 22)
        part.rename(archive)
    print(f"unpacking {archive} ...")
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")
    return dest
