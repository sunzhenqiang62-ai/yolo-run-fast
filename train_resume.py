"""Continue zhuangji OBB training from last weights until loss stabilizes."""
from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parent
DATA_YAML = PROJECT / "zhuangji.yaml"
CHECKPOINT = PROJECT / "runs" / "zhuangji_obb" / "weights" / "last.pt"
RUN_NAME = "zhuangji_obb-2"

# Fresh run from epoch-46 weights (last.pt has no optimizer state for true resume)
EPOCHS = 250
# Early stop on mAP plateau only after long stall (fitness metric, not loss)
PATIENCE = 50


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
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    device = 0 if torch.cuda.is_available() else "cpu"
    batch = pick_batch()

    model = YOLO(str(CHECKPOINT))
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=640,
        batch=batch,
        device=device,
        project=str(PROJECT / "runs"),
        name=RUN_NAME,
        patience=PATIENCE,
        save=True,
        plots=True,
        workers=0,
    )


if __name__ == "__main__":
    main()
