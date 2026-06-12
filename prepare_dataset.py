"""Split flat YOLO OBB dataset into train/val and write dataset YAML."""
from __future__ import annotations

import random
import shutil
from pathlib import Path

SRC = Path(r"E:\openmm\zhuangji_data")
PROJECT = Path(r"E:\yolo26n")
DATASET = PROJECT / "dataset"
VAL_RATIO = 0.2
SEED = 42


def main() -> None:
    random.seed(SEED)
    images = sorted((SRC / "images").glob("*.jpg"))
    stems = [p.stem for p in images]
    label_dir = SRC / "labels"
    missing = [s for s in stems if not (label_dir / f"{s}.txt").exists()]
    if missing:
        raise SystemExit(f"Missing labels for {len(missing)} images, e.g. {missing[0]}")

    random.shuffle(stems)
    val_count = max(1, int(len(stems) * VAL_RATIO))
    val_stems = set(stems[:val_count])
    train_stems = stems[val_count:]

    for split, split_stems in (("train", train_stems), ("val", list(val_stems))):
        img_out = DATASET / "images" / split
        lbl_out = DATASET / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for stem in split_stems:
            shutil.copy2(SRC / "images" / f"{stem}.jpg", img_out / f"{stem}.jpg")
            shutil.copy2(label_dir / f"{stem}.txt", lbl_out / f"{stem}.txt")

    yaml_path = PROJECT / "zhuangji.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {DATASET.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: zhuangji",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Total: {len(stems)} | train: {len(train_stems)} | val: {len(val_stems)}")
    print(f"YAML: {yaml_path}")


if __name__ == "__main__":
    main()
