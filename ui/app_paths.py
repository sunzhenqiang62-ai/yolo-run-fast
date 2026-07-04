"""Resolve install/dev paths for zhuangji-aerial GUI and subprocess launchers."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


def get_app_root() -> Path:
    env = os.environ.get("AERIAL_OBB_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parent.parent


@dataclass
class AppPaths:
    root: Path
    cpp_exe: Path | None
    onnx_model: Path | None
    predict_script: Path | None
    python_exe: Path | None
    pt_model: Path | None

    @property
    def has_cpp(self) -> bool:
        return self.cpp_exe is not None and self.onnx_model is not None

    @property
    def has_python(self) -> bool:
        return self.python_exe is not None and self.predict_script is not None


def _first_existing(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.is_file():
            return path
    return None


def find_python(root: Path) -> Path | None:
    candidates = [
        root / ".venv" / "Scripts" / "python.exe",
        root.parent / ".venv" / "Scripts" / "python.exe",
    ]
    found = _first_existing(candidates)
    if found:
        return found
    for name in ("python", "python3", "py"):
        resolved = shutil.which(name)
        if not resolved:
            continue
        path = Path(resolved)
        if "WindowsApps" in str(path):
            continue
        return path
    return None


def resolve_paths(root: Path | None = None) -> AppPaths:
    root = (root or get_app_root()).resolve()
    cpp_exe = _first_existing(
        [
            root / "bin" / "aerial_obb.exe",
            root / "cpp" / "build" / "Release" / "aerial_obb.exe",
            root / "cpp" / "build" / "Debug" / "aerial_obb.exe",
            root / "cpp" / "build" / "aerial_obb.exe",
        ]
    )
    onnx_model = _first_existing(
        [
            root / "models" / "best.onnx",
            root / "runs" / "zhuangji_obb-2" / "weights" / "best.onnx",
        ]
    )
    predict_script = root / "predict_aerial.py"
    if not predict_script.is_file():
        predict_script = None
    pt_model = _first_existing(
        [
            root / "runs" / "zhuangji_obb-2" / "weights" / "best.pt",
            root.parent / "runs" / "zhuangji_obb-2" / "weights" / "best.pt",
        ]
    )
    return AppPaths(
        root=root,
        cpp_exe=cpp_exe,
        onnx_model=onnx_model,
        predict_script=predict_script,
        python_exe=find_python(root),
        pt_model=pt_model,
    )


def default_json_path(root: Path) -> Path:
    return root / "out" / "result.json"


def default_preview_path(json_path: Path) -> Path:
    return json_path.with_name(json_path.stem + "_preview.jpg")
