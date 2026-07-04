#!/usr/bin/env python3
"""Export the recommended best.pt weights to ONNX for the C++ ORT pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--weights",
        default="runs/zhuangji_obb-2/weights/best.pt",
        help="PyTorch checkpoint",
    )
    ap.add_argument(
        "--out",
        default="",
        help="Output .onnx path (default: alongside weights as best.onnx)",
    )
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32, help="Fixed batch for C++ ORT engine")
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.is_file():
        raise SystemExit(f"weights not found: {weights}")

    out = Path(args.out) if args.out else weights.with_suffix(".onnx")
    out.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        batch=args.batch,
        simplify=True,
        dynamic=False,
        opset=17,
    )
    exported_path = Path(str(exported))
    if exported_path.resolve() != out.resolve():
        out.write_bytes(exported_path.read_bytes())
        print(f"copied {exported_path} -> {out}")
    else:
        print(f"exported {out}")
    print(f"ONNX ready: {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
