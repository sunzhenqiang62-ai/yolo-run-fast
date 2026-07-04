"""Build and run inference commands for the aerial OBB GUI."""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app_paths import AppPaths

LogFn = Callable[[str], None]


@dataclass
class InferenceConfig:
    image_path: Path
    json_path: Path
    conf: float = 0.25
    backend: str = "cpp"  # cpp | pt | openvino
    preview_path: Path | None = None
    profile: bool = True


@dataclass
class InferenceResult:
    ok: bool
    returncode: int
    elapsed_s: float
    stdout: str = ""
    stderr: str = ""
    payload: dict | None = None
    error: str = ""


_TOTAL_RE = re.compile(r"Total:\s*([\d.]+)s", re.IGNORECASE)


def build_cpp_command(paths: AppPaths, cfg: InferenceConfig) -> list[str]:
    if not paths.cpp_exe or not paths.onnx_model:
        raise RuntimeError("未找到 C++ 推理程序或 ONNX 模型")
    cmd = [
        str(paths.cpp_exe),
        "--image",
        str(cfg.image_path),
        "--json",
        str(cfg.json_path),
        "--model",
        str(paths.onnx_model),
        "--conf",
        str(cfg.conf),
    ]
    if cfg.profile:
        cmd.append("--profile")
    if cfg.preview_path is not None:
        cmd.extend(["--preview", str(cfg.preview_path)])
    return cmd


def build_python_command(paths: AppPaths, cfg: InferenceConfig) -> list[str]:
    if not paths.python_exe or not paths.predict_script:
        raise RuntimeError("未找到 Python 或 predict_aerial.py")
    model = paths.pt_model or paths.onnx_model
    if model is None:
        raise RuntimeError("未找到 .pt 或 .onnx 模型")
    backend = "openvino" if cfg.backend == "openvino" else "pt"
    cmd = [
        str(paths.python_exe),
        str(paths.predict_script),
        "--image",
        str(cfg.image_path),
        "--json",
        str(cfg.json_path),
        "--model",
        str(model),
        "--conf",
        str(cfg.conf),
        "--backend",
        backend,
        "--profile",
    ]
    if cfg.preview_path is not None:
        cmd.extend(["--preview", str(cfg.preview_path)])
    else:
        cmd.append("--skip-preview")
    return cmd


def build_command(paths: AppPaths, cfg: InferenceConfig) -> list[str]:
    if cfg.backend == "cpp":
        return build_cpp_command(paths, cfg)
    return build_python_command(paths, cfg)


def parse_timing(stdout: str, stderr: str, elapsed_s: float) -> float:
    for text in (stdout, stderr):
        match = _TOTAL_RE.search(text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
    return elapsed_s


def load_result_json(json_path: Path) -> dict:
    return json.loads(json_path.read_text(encoding="utf-8"))


def run_inference(
    paths: AppPaths,
    cfg: InferenceConfig,
    log: LogFn | None = None,
) -> InferenceResult:
    def emit(line: str) -> None:
        if log:
            log(line.rstrip())

    try:
        cmd = build_command(paths, cfg)
    except RuntimeError as exc:
        return InferenceResult(ok=False, returncode=-1, elapsed_s=0.0, error=str(exc))

    cfg.json_path.parent.mkdir(parents=True, exist_ok=True)
    if cfg.preview_path is not None:
        cfg.preview_path.parent.mkdir(parents=True, exist_ok=True)

    emit("命令: " + " ".join(f'"{part}"' if " " in part else part for part in cmd))
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(paths.root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        return InferenceResult(
            ok=False,
            returncode=-1,
            elapsed_s=time.perf_counter() - t0,
            error=str(exc),
        )
    elapsed = time.perf_counter() - t0

    for line in (proc.stdout or "").splitlines():
        emit(line)
    for line in (proc.stderr or "").splitlines():
        emit(line)

    timing = parse_timing(proc.stdout or "", proc.stderr or "", elapsed)
    if proc.returncode != 0:
        return InferenceResult(
            ok=False,
            returncode=proc.returncode,
            elapsed_s=timing,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            error=f"推理失败 (exit {proc.returncode})",
        )

    if not cfg.json_path.is_file():
        return InferenceResult(
            ok=False,
            returncode=proc.returncode,
            elapsed_s=timing,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            error=f"未生成 JSON: {cfg.json_path}",
        )

    payload = load_result_json(cfg.json_path)
    return InferenceResult(
        ok=True,
        returncode=0,
        elapsed_s=timing,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        payload=payload,
    )
