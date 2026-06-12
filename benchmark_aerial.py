#!/usr/bin/env python3
"""Benchmark aerial detection strategies on a large TIFF."""

from __future__ import annotations

import json
import time
from pathlib import Path

from predict_aerial import predict_aerial


def run_case(name: str, image: Path, out_dir: Path, **kwargs: object) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{name}.json"
    t0 = time.perf_counter()
    count, stats = predict_aerial(
        model_path=Path("runs/zhuangji_obb-2/weights/best.pt"),
        image_path=image,
        json_path=json_path,
        preview_path=None,
        skip_preview=True,
        profile=True,
        conf=0.5,
        **kwargs,
    )
    elapsed = time.perf_counter() - t0
    return {
        "name": name,
        "count": count,
        "elapsed_s": round(elapsed, 1),
        "coarse_tiles": stats.coarse_tiles,
        "fine_tiles": stats.fine_tiles,
        "total_tiles": stats.total_tiles,
        "read_ms": round(stats.read_ms, 1),
        "infer_ms": round(stats.infer_ms, 1),
        "merge_ms": round(stats.merge_ms, 1),
    }


def main() -> None:
    image = Path(r"C:\Users\m1512\Downloads\14_A1.tif")
    out_dir = Path("runs/benchmark_aerial")
    cases = [
        ("two_stage_pt", {"strategy": "two-stage", "backend": "pt", "batch_size": 8}),
        ("two_stage_openvino", {"strategy": "two-stage", "backend": "openvino", "batch_size": 8}),
        ("batch_fullscan_pt", {"strategy": "full-scan", "backend": "pt", "batch_size": 8, "overlap": 128}),
    ]
    results = []
    for name, kwargs in cases:
        print(f"\n{'=' * 60}\nBenchmark: {name}\n{'=' * 60}")
        try:
            results.append(run_case(name, image, out_dir, **kwargs))
        except Exception as exc:
            results.append({"name": name, "error": str(exc)})
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nBenchmark summary written to {summary_path}")
    for row in results:
        print(row)


if __name__ == "__main__":
    main()
