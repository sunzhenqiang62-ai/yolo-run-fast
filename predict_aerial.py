#!/usr/bin/env python3
"""Sliding-window OBB inference on large aerial TIFF images (optimized)."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

Image.MAX_IMAGE_PIXELS = None


@dataclass
class TileSpec:
    x0: int
    y0: int
    tw: int
    th: int


@dataclass
class RawHit:
    polygon: list[list[float]]
    score: float
    xyxy: list[float]


@dataclass
class ProfileStats:
    read_ms: float = 0.0
    infer_ms: float = 0.0
    merge_ms: float = 0.0
    preview_ms: float = 0.0
    cache_ms: float = 0.0
    coarse_tiles: int = 0
    fine_tiles: int = 0
    total_tiles: int = 0

    def report(self, total_s: float) -> None:
        print(
            f"Profile: total={total_s:.1f}s | "
            f"cache={self.cache_ms:.0f}ms read={self.read_ms:.0f}ms infer={self.infer_ms:.0f}ms "
            f"merge={self.merge_ms:.0f}ms preview={self.preview_ms:.0f}ms | "
            f"tiles coarse={self.coarse_tiles} fine={self.fine_tiles} total={self.total_tiles}"
        )


class TileReaderProto(Protocol):
    def read_bgr(self, spec: TileSpec, tile_size: int) -> np.ndarray: ...


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def _read_thumbnail(path: Path, max_long_edge: int) -> tuple[np.ndarray, float]:
    with Image.open(path) as img:
        img = img.convert("RGB")
        w, h = img.size
        scale = min(1.0, max_long_edge / max(w, h))
        if scale < 1.0:
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        rgb = np.array(img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    actual_scale = bgr.shape[1] / w
    return bgr, actual_scale


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_j = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (
            boxes[order[1:], 3] - boxes[order[1:], 1]
        )
        union = area_i + area_j - inter
        iou_vals = np.where(union > 0, inter / union, 0.0)
        order = order[1:][iou_vals <= iou]
    return keep


def build_tile_grid(
    orig_w: int,
    orig_h: int,
    tile_size: int,
    overlap: int,
    stride: int | None = None,
) -> list[TileSpec]:
    step = stride if stride is not None else max(1, tile_size - overlap)
    tiles: list[TileSpec] = []
    y0 = 0
    while y0 < orig_h:
        x0 = 0
        th = min(tile_size, orig_h - y0)
        while x0 < orig_w:
            tw = min(tile_size, orig_w - x0)
            tiles.append(TileSpec(x0, y0, tw, th))
            if x0 + tw >= orig_w:
                break
            x0 += step
        if y0 + th >= orig_h:
            break
        y0 += step
    return tiles


class MemmapTileReader:
    """Fast tile reader using tifffile when possible, else streaming PIL."""

    def __init__(self, path: Path, profile: ProfileStats | None = None) -> None:
        self.path = path
        self._profile = profile
        self._arr: np.ndarray | None = None
        self._pil: Image.Image | None = None
        self._mode = "pil"

    def __enter__(self) -> MemmapTileReader:
        t0 = time.perf_counter()
        try:
            import tifffile

            self._arr = tifffile.memmap(self.path)
            self._mode = "memmap"
        except Exception:
            self._pil = Image.open(self.path)
            self._mode = "pil"
        if self._profile is not None:
            self._profile.cache_ms = (time.perf_counter() - t0) * 1000
        return self

    def __exit__(self, *args: object) -> None:
        self._arr = None
        if self._pil is not None:
            self._pil.close()
            self._pil = None

    def read_bgr(self, spec: TileSpec, tile_size: int) -> np.ndarray:
        if self._mode == "memmap" and self._arr is not None:
            tile = self._arr[spec.y0 : spec.y0 + spec.th, spec.x0 : spec.x0 + spec.tw]
            if tile.ndim == 2:
                bgr = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
            elif tile.shape[2] >= 3:
                bgr = cv2.cvtColor(tile[:, :, :3], cv2.COLOR_RGB2BGR)
            else:
                bgr = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
            return _pad_tile(np.ascontiguousarray(bgr), spec, tile_size)
        assert self._pil is not None
        tile = self._pil.crop((spec.x0, spec.y0, spec.x0 + spec.tw, spec.y0 + spec.th))
        bgr = cv2.cvtColor(np.array(tile.convert("RGB")), cv2.COLOR_RGB2BGR)
        return _pad_tile(bgr, spec, tile_size)


def _pad_tile(bgr: np.ndarray, spec: TileSpec, tile_size: int) -> np.ndarray:
    pad_h = tile_size - spec.th
    pad_w = tile_size - spec.tw
    if pad_h > 0 or pad_w > 0:
        bgr = cv2.copyMakeBorder(
            bgr, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
    return bgr


def _resolve_openvino_path(model_path: Path, int8: bool = False) -> Path | None:
    if model_path.is_dir() and (model_path / "best.xml").exists():
        return model_path
    stem = model_path.stem
    parent = model_path.parent
    if int8:
        candidates = [
            parent / f"{stem}_int8_openvino_model",
            parent / f"{stem}_openvino_model_int8",
            parent / "best_int8_openvino_model",
            parent / "best_openvino_model_int8",
        ]
    else:
        candidates = [
            parent / f"{stem}_openvino_model",
            parent / "best_openvino_model",
        ]
    for cand in candidates:
        if cand.is_dir() and next(cand.glob("*.xml"), None) is not None:
            return cand
    return None


def _resolve_engine_path(model_path: Path) -> Path | None:
    if model_path.suffix == ".engine" and model_path.is_file():
        return model_path
    parent = model_path.parent
    stem = model_path.stem
    for cand in (parent / f"{stem}.engine", parent / "best.engine"):
        if cand.is_file():
            return cand
    return None


def load_model(
    model_path: Path,
    backend: str,
    tile_size: int,
    device: str = "0",
    batch_size: int = 8,
    int8: bool = False,
    data_yaml: Path | None = None,
    half: bool = False,
    compile_model: bool = False,
) -> tuple[YOLO, str]:
    """Load YOLO model; supports PyTorch, TensorRT engine, and OpenVINO."""
    backend = backend.lower()
    export_device = device if device not in ("cpu", "") else "0"

    if backend in ("engine", "tensorrt", "auto"):
        engine_path = _resolve_engine_path(model_path)
        if engine_path is None:
            try:
                print(f"Exporting TensorRT engine from {model_path} (batch={batch_size}, half={half}) ...")
                pt_model = YOLO(str(model_path))
                export_dir = pt_model.export(
                    format="engine",
                    device=export_device,
                    half=half,
                    imgsz=tile_size,
                    batch=batch_size,
                    workspace=4,
                )
                engine_path = Path(str(export_dir))
            except Exception as exc:
                print(f"TensorRT export failed ({exc}); falling back to PyTorch.")
                backend = "pt"
        if engine_path is not None and backend != "pt":
            print(f"Using TensorRT engine: {engine_path}")
            model = YOLO(str(engine_path), task="obb")
            _maybe_warmup(model, tile_size, device, half, batch_size)
            return model, "engine"
        if backend in ("engine", "tensorrt"):
            backend = "pt"

    if backend == "openvino":
        ov_path = _resolve_openvino_path(model_path, int8=int8)
        if ov_path is None:
            label = "OpenVINO INT8" if int8 else "OpenVINO"
            print(f"Exporting {label} model from {model_path} ...")
            try:
                pt_model = YOLO(str(model_path))
                export_kwargs: dict = {"format": "openvino", "imgsz": tile_size}
                if int8:
                    calib = data_yaml or Path("zhuangji.yaml")
                    export_kwargs["int8"] = True
                    export_kwargs["data"] = str(calib)
                export_dir = pt_model.export(**export_kwargs)
                ov_path = Path(str(export_dir))
            except Exception as exc:
                print(f"OpenVINO export failed ({exc}); falling back to PyTorch.")
                return _load_pt_model(model_path, device, half, compile_model, tile_size, batch_size)
        try:
            backend_label = "openvino-int8" if int8 else "openvino"
            print(f"Using OpenVINO backend ({backend_label}): {ov_path}")
            return YOLO(str(ov_path), task="obb"), backend_label
        except Exception as exc:
            print(f"OpenVINO load failed ({exc}); falling back to PyTorch.")

    return _load_pt_model(model_path, device, half, compile_model, tile_size, batch_size)


def _load_pt_model(
    model_path: Path,
    device: str,
    half: bool,
    compile_model: bool,
    tile_size: int,
    batch_size: int,
) -> tuple[YOLO, str]:
    model = YOLO(str(model_path))
    if compile_model and hasattr(torch, "compile"):
        try:
            model.model = torch.compile(model.model, mode="reduce-overhead")
            print("torch.compile enabled")
        except Exception as exc:
            print(f"torch.compile skipped ({exc})")
    _maybe_warmup(model, tile_size, device, half, batch_size)
    return model, "pt"


def _maybe_warmup(
    model: YOLO,
    tile_size: int,
    device: str,
    half: bool,
    batch_size: int,
) -> None:
    if device in ("cpu", ""):
        return
    dummy = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
    batch = [dummy] * min(batch_size, 4)
    try:
        model.predict(
            batch,
            imgsz=tile_size,
            device=device,
            half=half,
            verbose=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _extract_hits(result: object, spec: TileSpec) -> list[RawHit]:
    if result.obb is None or len(result.obb) == 0:
        return []
    polys = result.obb.xyxyxyxy.cpu().numpy()
    scores = result.obb.conf.cpu().numpy()
    hits: list[RawHit] = []
    for poly, score in zip(polys, scores):
        poly = poly.reshape(4, 2).astype(np.float64)
        poly[:, 0] = np.clip(poly[:, 0], 0, spec.tw)
        poly[:, 1] = np.clip(poly[:, 1], 0, spec.th)
        poly[:, 0] += spec.x0
        poly[:, 1] += spec.y0
        xs = poly[:, 0]
        ys = poly[:, 1]
        hits.append(
            RawHit(
                polygon=poly.tolist(),
                score=float(score),
                xyxy=[float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
            )
        )
    return hits


class BatchPredictor:
    """Run batched tile inference with optional prefetch and auto batch-size reduction."""

    def __init__(
        self,
        model: YOLO,
        tile_size: int,
        conf: float,
        iou: float,
        device: str,
        batch_size: int,
        half: bool = False,
        prefetch: bool = False,
        prefetch_workers: int = 4,
    ) -> None:
        self.model = model
        self.tile_size = tile_size
        self.conf = conf
        self.iou = iou
        self.device = device
        self.batch_size = max(1, batch_size)
        self.half = half
        self.prefetch = prefetch
        self.prefetch_workers = max(1, prefetch_workers)

    def _run_infer(self, batch_images: list[np.ndarray]) -> list:
        results = self.model.predict(
            batch_images,
            imgsz=self.tile_size,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            half=self.half,
            verbose=False,
        )
        if torch.cuda.is_available() and self.device not in ("cpu", ""):
            torch.cuda.synchronize()
        return results

    def predict_tiles(
        self,
        reader: TileReaderProto,
        tiles: list[TileSpec],
        profile: ProfileStats,
        label: str = "",
    ) -> list[RawHit]:
        all_hits: list[RawHit] = []
        total = len(tiles)
        if total == 0:
            return all_hits

        batch_size = self.batch_size
        processed = 0
        t_infer_start = time.perf_counter()
        pool = ThreadPoolExecutor(max_workers=self.prefetch_workers) if self.prefetch else None
        pending = None

        def _load_batch(specs: list[TileSpec]) -> list[np.ndarray]:
            t_read = time.perf_counter()
            if pool and len(specs) > 1:
                futures = [pool.submit(reader.read_bgr, s, self.tile_size) for s in specs]
                imgs = [f.result() for f in futures]
            else:
                imgs = [reader.read_bgr(s, self.tile_size) for s in specs]
            profile.read_ms += (time.perf_counter() - t_read) * 1000
            return imgs

        try:
            while processed < total:
                end = min(processed + batch_size, total)
                batch_specs = tiles[processed:end]

                if pending is not None:
                    batch_images = pending.result()
                    pending = None
                else:
                    batch_images = _load_batch(batch_specs)

                next_end = min(end + batch_size, total)
                if pool is not None and next_end > end:
                    pending = pool.submit(_load_batch, tiles[end:next_end])

                try:
                    t_inf = time.perf_counter()
                    results = self._run_infer(batch_images)
                    profile.infer_ms += (time.perf_counter() - t_inf) * 1000
                except RuntimeError as exc:
                    if pending is not None:
                        pending.cancel()
                        pending = None
                    if batch_size > 1 and "out of memory" in str(exc).lower():
                        batch_size = max(1, batch_size // 2)
                        print(f"  OOM: reducing batch size to {batch_size}")
                        continue
                    raise

                for spec, result in zip(batch_specs, results):
                    all_hits.extend(_extract_hits(result, spec))

                processed = end
                if processed == batch_size or processed % max(batch_size * 10, 50) == 0 or processed == total:
                    elapsed = time.perf_counter() - t_infer_start
                    eta = elapsed / processed * (total - processed) if processed else 0
                    prefix = f"{label} " if label else ""
                    print(f"  {prefix}[{processed}/{total}] hits={len(all_hits)} eta={eta:.0f}s")
        finally:
            if pool is not None:
                pool.shutdown(wait=False)

        return all_hits


def _expand_rect(x1: float, y1: float, x2: float, y2: float, margin: int, orig_w: int, orig_h: int) -> tuple[int, int, int, int]:
    return (
        max(0, int(x1) - margin),
        max(0, int(y1) - margin),
        min(orig_w, int(x2) + margin),
        min(orig_h, int(y2) + margin),
    )


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _tile_intersects_rect(spec: TileSpec, rect: tuple[int, int, int, int]) -> bool:
    tx1, ty1 = spec.x0, spec.y0
    tx2, ty2 = spec.x0 + spec.tw, spec.y0 + spec.th
    return not (tx2 <= rect[0] or rect[2] <= tx1 or ty2 <= rect[1] or rect[3] <= ty1)


def build_hot_regions(
    coarse_hits: list[RawHit],
    orig_w: int,
    orig_h: int,
    margin: int,
) -> list[tuple[int, int, int, int]]:
    if not coarse_hits:
        return [(0, 0, orig_w, orig_h)]

    rects: list[tuple[int, int, int, int]] = []
    for hit in coarse_hits:
        x1, y1, x2, y2 = hit.xyxy
        rects.append(_expand_rect(x1, y1, x2, y2, margin, orig_w, orig_h))

    merged = True
    while merged:
        merged = False
        new_rects: list[tuple[int, int, int, int]] = []
        used = [False] * len(rects)
        for i, ri in enumerate(rects):
            if used[i]:
                continue
            ax1, ay1, ax2, ay2 = ri
            for j in range(i + 1, len(rects)):
                if used[j]:
                    continue
                if _rects_overlap(ri, rects[j]):
                    bx1, by1, bx2, by2 = rects[j]
                    ax1, ay1 = min(ax1, bx1), min(ay1, by1)
                    ax2, ay2 = max(ax2, bx2), max(ay2, by2)
                    used[j] = True
                    merged = True
            new_rects.append((ax1, ay1, ax2, ay2))
            used[i] = True
        rects = new_rects
    return rects


def filter_tiles_by_regions(tiles: list[TileSpec], regions: list[tuple[int, int, int, int]]) -> list[TileSpec]:
    if not regions:
        return tiles
    if len(regions) == 1 and regions[0] == (0, 0, 0, 0):
        return tiles
    return [t for t in tiles if any(_tile_intersects_rect(t, r) for r in regions)]


def merge_hits(hits: list[RawHit], nms_iou: float = 0.35) -> list[dict]:
    if not hits:
        return []
    boxes_arr = np.array([h.xyxy for h in hits], dtype=np.float32)
    scores_arr = np.array([h.score for h in hits], dtype=np.float32)
    keep = _nms_xyxy(boxes_arr, scores_arr, iou=nms_iou)
    detections = [
        {"score": hits[i].score, "polygon": hits[i].polygon, "xyxy": hits[i].xyxy}
        for i in keep
    ]
    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections


def save_preview(
    image_path: Path,
    detections: list[dict],
    preview_path: Path,
    preview_max_edge: int = 4096,
) -> None:
    preview_bgr, scale = _read_thumbnail(image_path, preview_max_edge)
    for det in detections:
        pts = np.array(det["polygon"], dtype=np.float32) * scale
        pts = pts.astype(np.int32)
        cv2.polylines(preview_bgr, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        x, y = int(pts[0, 0]), int(pts[0, 1])
        cv2.putText(
            preview_bgr,
            f"{det['score']:.2f}",
            (x, max(0, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(preview_path), preview_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])


class CachedTileReader:
    """Load the full image into RAM once; slice tiles without repeated decode."""

    def __init__(self, path: Path, profile: ProfileStats | None = None) -> None:
        self.path = path
        self._bgr: np.ndarray | None = None
        self._profile = profile

    def __enter__(self) -> CachedTileReader:
        t0 = time.perf_counter()
        with Image.open(self.path) as img:
            rgb = np.asarray(img.convert("RGB"))
        self._bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if self._profile is not None:
            self._profile.cache_ms = (time.perf_counter() - t0) * 1000
        return self

    def __exit__(self, *args: object) -> None:
        self._bgr = None

    def read_bgr(self, spec: TileSpec, tile_size: int) -> np.ndarray:
        assert self._bgr is not None
        bgr = self._bgr[spec.y0 : spec.y0 + spec.th, spec.x0 : spec.x0 + spec.tw].copy()
        return _pad_tile(bgr, spec, tile_size)


def _pick_reader(
    image_path: Path,
    cache_image: bool,
    profile: ProfileStats,
) -> TileReaderProto:
    if cache_image:
        return CachedTileReader(image_path, profile)
    return MemmapTileReader(image_path, profile)


def predict_aerial(
    model_path: Path,
    image_path: Path,
    json_path: Path,
    preview_path: Path | None,
    tile_size: int = 640,
    overlap: int = 96,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = "cpu",
    preview_max_edge: int = 4096,
    batch_size: int = 8,
    backend: str = "pt",
    strategy: str = "two-stage",
    coarse_stride: int = 1280,
    coarse_conf: float = 0.35,
    hot_margin: int = 320,
    profile: bool = False,
    skip_preview: bool = False,
    int8: bool = False,
    data_yaml: Path | None = None,
    half: bool = False,
    prefetch: bool = False,
    cache_image: bool = False,
    compile_model: bool = False,
    prefetch_workers: int = 4,
) -> tuple[int, ProfileStats]:
    t_total = time.perf_counter()
    stats = ProfileStats()
    orig_w, orig_h = _image_size(image_path)

    model, backend_used = load_model(
        model_path,
        backend,
        tile_size,
        device=device,
        batch_size=batch_size,
        int8=int8,
        data_yaml=data_yaml,
        half=half,
        compile_model=compile_model,
    )
    if backend_used.startswith("openvino") and batch_size > 1:
        print(f"OpenVINO LATENCY mode: reducing batch-size {batch_size} -> 1")
        batch_size = 1
    print(f"Backend: {backend_used} | half={half} | batch={batch_size} | cache={cache_image} | prefetch={prefetch}")

    fine_tiles = build_tile_grid(orig_w, orig_h, tile_size, overlap)
    predictor = BatchPredictor(
        model, tile_size, conf, iou, device, batch_size,
        half=half, prefetch=prefetch, prefetch_workers=prefetch_workers,
    )

    all_hits: list[RawHit] = []
    reader_ctx = _pick_reader(image_path, cache_image, stats)

    with reader_ctx as reader:
        if strategy == "two-stage":
            coarse_tiles = build_tile_grid(
                orig_w, orig_h, tile_size, overlap=0, stride=coarse_stride
            )
            stats.coarse_tiles = len(coarse_tiles)
            print(
                f"Image: {orig_w}x{orig_h} | two-stage | "
                f"coarse={len(coarse_tiles)} (stride={coarse_stride}) fine_pool={len(fine_tiles)}"
            )
            coarse_predictor = BatchPredictor(
                model, tile_size, coarse_conf, iou, device, batch_size,
                half=half, prefetch=prefetch, prefetch_workers=prefetch_workers,
            )
            print("Stage 1: coarse scan")
            coarse_hits = coarse_predictor.predict_tiles(
                reader, coarse_tiles, stats, label="coarse"
            )
            regions = build_hot_regions(coarse_hits, orig_w, orig_h, hot_margin)
            print(f"  hot regions: {len(regions)} (from {len(coarse_hits)} coarse hits)")
            selected_fine = filter_tiles_by_regions(fine_tiles, regions)
            stats.fine_tiles = len(selected_fine)
            stats.total_tiles = stats.coarse_tiles + stats.fine_tiles
            print(f"Stage 2: fine scan ({len(selected_fine)} tiles)")
            all_hits = predictor.predict_tiles(reader, selected_fine, stats, label="fine")
            all_hits.extend(coarse_hits)
        else:
            stats.fine_tiles = len(fine_tiles)
            stats.total_tiles = len(fine_tiles)
            print(
                f"Image: {orig_w}x{orig_h} | full-scan | "
                f"tiles={len(fine_tiles)} ({tile_size}px, overlap {overlap}px)"
            )
            all_hits = predictor.predict_tiles(reader, fine_tiles, stats, label="scan")

    t_merge = time.perf_counter()
    detections = merge_hits(all_hits, nms_iou=0.35)
    stats.merge_ms = (time.perf_counter() - t_merge) * 1000

    payload = {
        "image": str(image_path),
        "size": {"width": orig_w, "height": orig_h},
        "count": len(detections),
        "backend": backend_used,
        "strategy": strategy,
        "detections": detections,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Found {len(detections)} detection(s) after merge")
    print(f"JSON: {json_path}")

    if not skip_preview and preview_path is not None:
        t_prev = time.perf_counter()
        save_preview(image_path, detections, preview_path, preview_max_edge)
        stats.preview_ms = (time.perf_counter() - t_prev) * 1000
        print(f"Preview: {preview_path}")

    total_s = time.perf_counter() - t_total
    if profile:
        stats.report(total_s)
    else:
        print(f"Total: {total_s:.1f}s")
    return len(detections), stats


def main() -> None:
    parser = argparse.ArgumentParser(description="OBB sliding-window inference on large aerial images.")
    parser.add_argument("--image", "-i", required=True)
    parser.add_argument("--model", "-m", default="runs/zhuangji_obb-2/weights/best.pt")
    parser.add_argument("--json", required=True)
    parser.add_argument("--preview", default=None, help="Output preview JPEG (optional with --skip-preview)")
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=int, default=96, help="Fine tile overlap (default: 96)")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--backend",
        choices=["pt", "openvino", "engine", "tensorrt", "auto"],
        default="pt",
    )
    parser.add_argument("--half", action="store_true", help="FP16 inference (GPU / TensorRT)")
    parser.add_argument("--prefetch", action="store_true", help="Prefetch next tile batch while GPU runs")
    parser.add_argument("--prefetch-workers", type=int, default=4)
    parser.add_argument("--cache-image", action="store_true", help="Load full image into RAM (auto if <=512MB)")
    parser.add_argument("--no-cache-image", action="store_true", help="Disable RAM cache even for small images")
    parser.add_argument("--compile", dest="compile_model", action="store_true", help="torch.compile (PyTorch backend)")
    parser.add_argument(
        "--turbo",
        action="store_true",
        help="Max speed: auto backend, FP16, batch=32, prefetch, skip preview",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Use OpenVINO INT8 quantized model (requires --backend openvino)",
    )
    parser.add_argument(
        "--data",
        default="zhuangji.yaml",
        help="Dataset YAML for INT8 calibration (default: zhuangji.yaml)",
    )
    parser.add_argument(
        "--strategy",
        choices=["two-stage", "full-scan"],
        default="two-stage",
        help="two-stage: coarse hot-region scan then fine tiles (default)",
    )
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Disable two-stage and scan all fine tiles",
    )
    parser.add_argument("--coarse-stride", type=int, default=1280)
    parser.add_argument("--coarse-conf", type=float, default=0.35)
    parser.add_argument("--hot-margin", type=int, default=320)
    parser.add_argument("--profile", action="store_true", help="Print per-stage timing breakdown")
    parser.add_argument("--skip-preview", action="store_true")
    args = parser.parse_args()

    if args.turbo:
        args.backend = "pt"
        args.half = True
        args.prefetch = True
        args.skip_preview = True
        if args.batch_size == 8:
            args.batch_size = 64
        if args.device == "cpu" and torch.cuda.is_available():
            args.device = "0"

    strategy = "full-scan" if args.full_scan else args.strategy
    if args.skip_preview and not args.preview:
        preview_path = None
    elif args.preview:
        preview_path = Path(args.preview)
    else:
        preview_path = Path(args.json).with_suffix(".jpg")

    cache_image = args.cache_image and not args.no_cache_image

    torch.set_num_threads(max(1, torch.get_num_threads()))
    predict_aerial(
        model_path=Path(args.model),
        image_path=Path(args.image),
        json_path=Path(args.json),
        preview_path=preview_path,
        tile_size=args.tile_size,
        overlap=args.overlap,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        batch_size=args.batch_size,
        backend=args.backend,
        strategy=strategy,
        coarse_stride=args.coarse_stride,
        coarse_conf=args.coarse_conf,
        hot_margin=args.hot_margin,
        profile=args.profile,
        skip_preview=args.skip_preview,
        int8=args.int8,
        data_yaml=Path(args.data),
        half=args.half,
        prefetch=args.prefetch,
        cache_image=cache_image,
        compile_model=args.compile_model,
        prefetch_workers=args.prefetch_workers,
    )


if __name__ == "__main__":
    main()
