#!/usr/bin/env python3
"""装机检测航拍推理 — 图形界面 (tkinter + subprocess)."""

from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

UI_DIR = Path(__file__).resolve().parent
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))

from app_paths import AppPaths, default_json_path, default_preview_path, get_app_root, resolve_paths
from inference_runner import InferenceConfig, InferenceResult, run_inference


class AerialGuiApp:
    TITLE = "装机检测航拍推理"
    FILE_TYPES = [
        ("影像文件", "*.tif;*.tiff;*.png;*.jpg;*.jpeg;*.bmp"),
        ("TIFF", "*.tif;*.tiff"),
        ("所有文件", "*.*"),
    ]

    def __init__(self, root: tk.Tk, paths: AppPaths) -> None:
        self.root = root
        self.paths = paths
        self.worker: threading.Thread | None = None
        self.preview_image: tk.PhotoImage | None = None

        root.title(self.TITLE)
        root.minsize(920, 640)
        root.geometry("1024x720")

        self.image_var = tk.StringVar()
        self.json_var = tk.StringVar(value=str(default_json_path(paths.root)))
        self.conf_var = tk.DoubleVar(value=0.25)
        self.preview_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="就绪")
        self.summary_var = tk.StringVar(value="尚未运行推理")

        self._build_ui()
        self._refresh_backend_options()
        self._log(f"安装目录: {paths.root}")

    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 4}
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        io = ttk.LabelFrame(outer, text="输入 / 输出", padding=8)
        io.pack(fill=tk.X, **pad)
        self._file_row(io, "输入影像", self.image_var, self._pick_image)
        self._file_row(io, "输出 JSON", self.json_var, self._pick_json)

        settings = ttk.LabelFrame(outer, text="推理设置", padding=8)
        settings.pack(fill=tk.X, **pad)

        row = ttk.Frame(settings)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="置信度阈值", width=12).pack(side=tk.LEFT)
        ttk.Scale(
            row,
            from_=0.05,
            to=0.95,
            variable=self.conf_var,
            orient=tk.HORIZONTAL,
            command=self._on_conf_scale,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self.conf_label = ttk.Label(row, text="0.25", width=6)
        self.conf_label.pack(side=tk.LEFT)

        row2 = ttk.Frame(settings)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="推理后端", width=12).pack(side=tk.LEFT)
        self.backend_combo = ttk.Combobox(row2, state="readonly", width=36)
        self.backend_combo.pack(side=tk.LEFT, padx=6)
        ttk.Checkbutton(row2, text="生成预览图", variable=self.preview_var).pack(side=tk.LEFT, padx=12)

        actions = ttk.Frame(outer)
        actions.pack(fill=tk.X, **pad)
        self.run_btn = ttk.Button(actions, text="开始推理", command=self._on_run)
        self.run_btn.pack(side=tk.LEFT)
        ttk.Button(actions, text="打开输出目录", command=self._open_out_dir).pack(side=tk.LEFT, padx=8)
        ttk.Button(actions, text="打开 CLI 帮助", command=self._open_cli).pack(side=tk.LEFT)

        self.progress = ttk.Progressbar(outer, mode="indeterminate")
        self.progress.pack(fill=tk.X, **pad)

        status_row = ttk.Frame(outer)
        status_row.pack(fill=tk.X, **pad)
        ttk.Label(status_row, text="状态:").pack(side=tk.LEFT)
        ttk.Label(status_row, textvariable=self.status_var).pack(side=tk.LEFT, padx=6)

        summary = ttk.LabelFrame(outer, text="结果摘要", padding=8)
        summary.pack(fill=tk.X, **pad)
        ttk.Label(summary, textvariable=self.summary_var, wraplength=960, justify=tk.LEFT).pack(
            anchor=tk.W
        )

        panes = ttk.Panedwindow(outer, orient=tk.HORIZONTAL)
        panes.pack(fill=tk.BOTH, expand=True, **pad)

        left = ttk.Frame(panes)
        panes.add(left, weight=3)
        ttk.Label(left, text="检测列表").pack(anchor=tk.W)
        cols = ("idx", "score", "bbox")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=14)
        self.tree.heading("idx", text="#")
        self.tree.heading("score", text="置信度")
        self.tree.heading("bbox", text="边界框 (xyxy)")
        self.tree.column("idx", width=40, anchor=tk.CENTER)
        self.tree.column("score", width=90, anchor=tk.CENTER)
        self.tree.column("bbox", width=320, anchor=tk.W)
        scroll = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        right = ttk.Frame(panes)
        panes.add(right, weight=2)
        ttk.Label(right, text="预览").pack(anchor=tk.W)
        self.preview_label = ttk.Label(right, text="推理完成后显示预览图", anchor=tk.CENTER)
        self.preview_label.pack(fill=tk.BOTH, expand=True, pady=6)
        ttk.Button(right, text="用系统查看器打开预览", command=self._open_preview_file).pack()

        log_frame = ttk.LabelFrame(outer, text="运行日志", padding=6)
        log_frame.pack(fill=tk.BOTH, expand=False, **pad)
        self.log_text = tk.Text(log_frame, height=8, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _file_row(self, parent: ttk.Frame, label: str, var: tk.StringVar, browse_cmd) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=12).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Button(row, text="浏览...", command=browse_cmd, width=10).pack(side=tk.LEFT)

    def _refresh_backend_options(self) -> None:
        options: list[tuple[str, str]] = []
        if self.paths.has_cpp:
            options.append(("cpp", "C++ ONNX Runtime (推荐，无需 Python 依赖)"))
        if self.paths.has_python:
            options.append(("pt", "Python PyTorch (.pt)"))
            options.append(("openvino", "Python OpenVINO (需 pip install openvino)"))
        if not options:
            options.append(("cpp", "C++ ONNX (未检测到可执行文件)"))
        self._backend_map = {label: key for key, label in options}
        labels = [label for _, label in options]
        self.backend_combo["values"] = labels
        if labels:
            self.backend_combo.current(0)

    def _selected_backend(self) -> str:
        label = self.backend_combo.get()
        return self._backend_map.get(label, "cpp")

    def _on_conf_scale(self, _value: str) -> None:
        self.conf_label.configure(text=f"{self.conf_var.get():.2f}")

    def _pick_image(self) -> None:
        path = filedialog.askopenfilename(title="选择输入影像", filetypes=self.FILE_TYPES)
        if path:
            self.image_var.set(path)
            if not self.json_var.get().strip():
                self.json_var.set(str(default_json_path(self.paths.root)))

    def _pick_json(self) -> None:
        path = filedialog.asksaveasfilename(
            title="选择 JSON 输出路径",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("所有文件", "*.*")],
            initialfile="result.json",
        )
        if path:
            self.json_var.set(path)

    def _log(self, message: str) -> None:
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def _ui(self, fn, *args) -> None:
        self.root.after(0, lambda: fn(*args))

    def _set_busy(self, busy: bool) -> None:
        state = tk.DISABLED if busy else tk.NORMAL
        self.run_btn.configure(state=state)
        if busy:
            self.progress.start(12)
            self.status_var.set("推理进行中…")
        else:
            self.progress.stop()
            self.status_var.set("就绪")

    def _on_run(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        image = self.image_var.get().strip()
        json_out = self.json_var.get().strip()
        if not image:
            messagebox.showwarning(self.TITLE, "请选择输入影像。")
            return
        if not json_out:
            messagebox.showwarning(self.TITLE, "请指定 JSON 输出路径。")
            return
        if not Path(image).is_file():
            messagebox.showerror(self.TITLE, f"输入文件不存在:\n{image}")
            return

        backend = self._selected_backend()
        if backend == "cpp" and not self.paths.has_cpp:
            messagebox.showerror(self.TITLE, "未找到 aerial_obb.exe 或 ONNX 模型。")
            return
        if backend in ("pt", "openvino") and not self.paths.has_python:
            messagebox.showerror(self.TITLE, "未找到 Python 环境或 predict_aerial.py。")
            return

        json_path = Path(json_out)
        preview_path = default_preview_path(json_path) if self.preview_var.get() else None
        cfg = InferenceConfig(
            image_path=Path(image),
            json_path=json_path,
            conf=float(self.conf_var.get()),
            backend=backend,
            preview_path=preview_path,
            profile=True,
        )

        self.log_text.delete("1.0", tk.END)
        self._clear_results()
        self._set_busy(True)

        def worker() -> None:
            def log_line(line: str) -> None:
                self._ui(self._log, line)

            result = run_inference(self.paths, cfg, log=log_line)
            self._ui(self._on_inference_done, result, preview_path)

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def _clear_results(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.summary_var.set("尚未运行推理")
        self.preview_image = None
        self.preview_label.configure(image="", text="推理完成后显示预览图")
        self._last_preview_path: Path | None = None

    def _on_inference_done(self, result: InferenceResult, preview_path: Path | None) -> None:
        self._set_busy(False)
        if not result.ok:
            self.status_var.set("推理失败")
            self.summary_var.set(result.error or "未知错误")
            messagebox.showerror(self.TITLE, result.error or "推理失败")
            return

        payload = result.payload or {}
        count = payload.get("count", 0)
        backend = payload.get("backend", self._selected_backend())
        strategy = payload.get("strategy", "-")
        self.summary_var.set(
            f"检测数量: {count}  |  耗时: {result.elapsed_s:.1f}s  |  "
            f"后端: {backend}  |  策略: {strategy}"
        )
        self.status_var.set("推理完成")
        self._populate_tree(payload.get("detections", []))
        if preview_path and preview_path.is_file():
            self._last_preview_path = preview_path
            self._show_preview(preview_path)

    def _populate_tree(self, detections: list) -> None:
        for idx, det in enumerate(detections, start=1):
            score = det.get("score", 0.0)
            xyxy = det.get("xyxy", [])
            bbox = ", ".join(f"{v:.1f}" for v in xyxy) if xyxy else "-"
            self.tree.insert("", tk.END, values=(idx, f"{score:.4f}", bbox))

    def _show_preview(self, path: Path) -> None:
        try:
            from PIL import Image, ImageTk

            img = Image.open(path)
            max_w, max_h = 420, 420
            img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
            self.preview_image = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=self.preview_image, text="")
        except Exception:
            self.preview_label.configure(
                image="",
                text=f"预览已保存:\n{path}\n(安装 Pillow 可在界面内显示 JPEG)",
            )

    def _open_preview_file(self) -> None:
        path = getattr(self, "_last_preview_path", None)
        if path and path.is_file():
            os.startfile(path)
        else:
            messagebox.showinfo(self.TITLE, "暂无预览图。请勾选“生成预览图”后重新推理。")

    def _open_out_dir(self) -> None:
        out = Path(self.json_var.get().strip() or default_json_path(self.paths.root))
        folder = out.parent if out.suffix else out
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(folder)

    def _open_cli(self) -> None:
        launcher = self.paths.root / "aerial_obb_launcher.bat"
        if launcher.is_file():
            os.startfile(str(launcher))
            return
        bat = self.paths.root / "aerial_obb.bat"
        if bat.is_file():
            os.startfile(str(bat))
            return
        messagebox.showinfo(self.TITLE, "未找到 CLI 启动脚本。")


def run_self_test(root: Path | None = None) -> int:
    paths = resolve_paths(root)
    print(f"root={paths.root}")
    print(f"cpp={paths.cpp_exe}")
    print(f"onnx={paths.onnx_model}")
    print(f"python={paths.python_exe}")

    sample = paths.root / "dist" / "_install_test" / "out" / "result.json"
    if not sample.is_file():
        sample = paths.root / "runs" / "smoke_test" / "cpp_out.json"
    if sample.is_file():
        from inference_runner import load_result_json

        data = load_result_json(sample)
        assert "count" in data and "detections" in data
        print(f"json_ok count={data['count']}")
    else:
        print("json_ok skipped (no sample json)")

    image = paths.root / "big_test.tif"
    if not (paths.has_cpp and image.is_file()):
        print("infer_ok skipped")
        return 0

    out_json = paths.root / "out" / "_gui_selftest.json"
    cfg = InferenceConfig(
        image_path=image,
        json_path=out_json,
        conf=0.25,
        backend="cpp",
        preview_path=None,
        profile=True,
    )
    result = run_inference(paths, cfg, log=print)
    if not result.ok:
        print(f"infer_fail: {result.error}")
        return 1
    print(f"infer_ok count={result.payload.get('count')} elapsed={result.elapsed_s:.1f}s")
    return 0


def main() -> None:
    if "--self-test" in sys.argv:
        idx = sys.argv.index("--self-test")
        root_arg = None
        if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("-"):
            root_arg = Path(sys.argv[idx + 1])
        raise SystemExit(run_self_test(root_arg))

    os.environ.setdefault("AERIAL_OBB_ROOT", str(get_app_root()))
    paths = resolve_paths()
    root = tk.Tk()
    try:
        root.call("tk", "scaling", 1.25)
    except tk.TclError:
        pass
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    AerialGuiApp(root, paths)
    root.mainloop()


if __name__ == "__main__":
    main()
