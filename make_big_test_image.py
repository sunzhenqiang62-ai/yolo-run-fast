#!/usr/bin/env python3
"""Stitch the 640px dataset crops into one large aerial-style image.

The repo has no real 21537x17132 aerial TIFF, so we tile the val/train 640
crops into an ~NxM grid to produce a single big image used as the common input
for the C++ vs Python benchmark. Deterministic (sorted file order) so both
backends see identical pixels.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def collect_images(src: Path, limit: int | None) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in exts)
    if not files:
        raise SystemExit(f"no images found under {src}")
    if limit:
        files = files[:limit]
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset/images", help="dir of 640 crops")
    ap.add_argument("--out", default="big_test.tif")
    ap.add_argument("--cols", type=int, default=0, help="grid columns (0 = auto ~square)")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--limit", type=int, default=0, help="max source crops (0 = all)")
    args = ap.parse_args()

    files = collect_images(Path(args.src), args.limit or None)
    n = len(files)
    cols = args.cols or max(1, int(math.ceil(math.sqrt(n))))
    rows = int(math.ceil(n / cols))
    tile = args.tile

    canvas = np.full((rows * tile, cols * tile, 3), 114, dtype=np.uint8)
    for idx, fp in enumerate(files):
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            continue
        if img.shape[0] != tile or img.shape[1] != tile:
            img = cv2.resize(img, (tile, tile))
        r, c = divmod(idx, cols)
        canvas[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = img

    out = Path(args.out)
    # Write a standard, tifffile-readable TIFF so the Python fast path
    # (MemmapTileReader -> tifffile.memmap, which expects RGB) and the C++ path
    # (cv2.imread -> BGR) both see identical pixels. cv2.imwrite produces a TIFF
    # that tifffile rejects ("bad magic number"), forcing a slow PIL fallback
    # that crashes on large crops, so we use tifffile directly and store RGB.
    if out.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        tifffile.imwrite(str(out), cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    else:
        cv2.imwrite(str(out), canvas)
    print(f"wrote {out} : {canvas.shape[1]}x{canvas.shape[0]} from {n} crops "
          f"({cols}x{rows} grid)")


if __name__ == "__main__":
    main()
