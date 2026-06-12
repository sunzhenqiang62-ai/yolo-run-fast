"""Fine-tune zhuangji OBB from best weights on GPU."""
from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parent
DATA_YAML = PROJECT / "zhuangji.yaml"
CHECKPOINT = PROJECT / "runs" / "zhuangji_obb-2" / "weights" / "best.pt"
RUN_NAME = "zhuangji_obb_gpu"

EPOCHS = 150
PATIENCE = 30


def pick_batch() -> int:
    if not torch.cuda.is_available():
        return 4
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram_gb >= 12:
        return 16
    if vram_gb >= 8:
        return 8
    return 4


def pick_workers() -> int:
    return 4 if torch.cuda.is_available() else 0


def main() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — install GPU PyTorch before running train_gpu.py")

    batch = pick_batch()
    workers = pick_workers()

    model = YOLO(str(CHECKPOINT))
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=640,
        batch=batch,
        device=0,
        project=str(PROJECT / "runs"),
        name=RUN_NAME,
        patience=PATIENCE,
        save=True,
        plots=True,
        workers=workers,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
