"""Fine-tune YOLO26n-OBB on zhuangji dataset."""
from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parent
DATA_YAML = PROJECT / "zhuangji.yaml"
MODEL = "yolo26n-obb.pt"


def pick_batch() -> int:
    if not torch.cuda.is_available():
        return 4
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram_gb >= 12:
        return 16
    if vram_gb >= 8:
        return 8
    return 4


def main() -> None:
    device = 0 if torch.cuda.is_available() else "cpu"
    batch = pick_batch()

    model = YOLO(MODEL)
    model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=batch,
        device=device,
        project=str(PROJECT / "runs"),
        name="zhuangji_obb",
        pretrained=True,
        patience=20,
        save=True,
        plots=True,
        workers=0,
    )


if __name__ == "__main__":
    main()
