#!/usr/bin/env python3
"""Benchmark GPU optimization configs on a large aerial TIFF."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from predict_aerial import predict_aerial


def run_case(
    name: str,
    image: Path,
    model: Path,
    out_dir: Path,
    **kwargs: object,
) -> dict:
    case_dir = out_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    json_path = case_dir / "result.json"
    t0 = time.perf_counter()
    count, stats = predict_aerial(
        model_path=model,
        image_path=image,
        json_path=json_path,
        preview_path=None,
        skip_preview=True,
        profile=True,
        conf=0.5,
        device="0",
        **kwargs,
    )
    elapsed = time.perf_counter() - t0
    return {
        "name": name,
        "count": count,
        "elapsed_s": round(elapsed, 2),
        "cache_ms": round(stats.cache_ms, 1),
        "read_ms": round(stats.read_ms, 1),
        "infer_ms": round(stats.infer_ms, 1),
        "merge_ms": round(stats.merge_ms, 1),
        "total_tiles": stats.total_tiles,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GPU aerial inference configs.")
    parser.add_argument(
        "-i", "--image",
        default="14_A1.tif",
        help="Input TIFF path (default: 14_A1.tif in cwd)",
    )
    parser.add_argument(
        "-m", "--model",
        default="runs/zhuangji_obb-2/weights/best.pt",
    )
    parser.add_argument(
        "-o", "--out-dir",
        default="runs/benchmark_gpu",
    )
    args = parser.parse_args()

    image = Path(args.image)
    model = Path(args.model)
    out_dir = Path(args.out_dir)

    cases = [
        ("baseline_pt_b8", {"backend": "pt", "batch_size": 8}),
        ("turbo_pt", {
            "backend": "pt",
            "batch_size": 64,
            "half": True,
            "prefetch": True,
            "prefetch_workers": 8,
        }),
    ]

    results = []
    for name, kwargs in cases:
        print(f"\n{'=' * 60}\nBenchmark: {name}\n{'=' * 60}")
        try:
            results.append(run_case(name, image, model, out_dir, **kwargs))
        except Exception as exc:
            results.append({"name": name, "error": str(exc)})

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n{'=' * 60}\nSUMMARY -> {summary_path}\n{'=' * 60}")
    for row in sorted(results, key=lambda r: r.get("elapsed_s", 9999)):
        print(row)


if __name__ == "__main__":
    main()
