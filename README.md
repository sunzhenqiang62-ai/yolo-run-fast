# YOLO26n-OBB 装机检测 (zhuangji OBB)

基于 [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO26n-OBB 的定向边界框（OBB）微调项目，用于检测 **装机 (zhuangji)** 目标。

## 项目简介

- **任务**: OBB 目标检测（单类别）
- **基座模型**: `yolo26n-obb.pt`（约 **2.65M** 参数，6.3 GFLOPs）
- **数据集**: 85 张裁剪图像，train/val = 68/17（20% 验证集，seed=42）
- **类别**: `zhuangji`（class id `0`）

## 当前进展

| 模块 | 状态 | 说明 |
|------|------|------|
| 模型微调 | 完成 | 推荐权重 `runs/zhuangji_obb-2/weights/best.pt`，mAP50-95 **0.727** |
| 640×640 切片推理 | 完成 | 标准 `yolo obb predict` |
| 大图航拍推理 | 完成 | [`predict_aerial.py`](predict_aerial.py)，支持超大 TIFF |
| 速度优化 | 完成 | two-stage 粗筛 + 批量推理 + OpenVINO，**14s / 全图**（原 ~27min） |
| INT8 量化 | 已验证 | OpenVINO INT8，模型 3.3MB，精度损失极小，速度提升不明显 |
| 基准测试 | 完成 | 见 [`runs/benchmark_aerial/summary.json`](runs/benchmark_aerial/summary.json) |

## 环境安装

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt
```

依赖：`ultralytics>=8.4.0`、`torch`、`torchvision`。

Intel CPU 加速（推荐）：

```bash
pip install openvino nncf   # nncf 仅 INT8 导出时需要
```

### GPU 环境（NVIDIA RTX 4070，CUDA 12.4）

CPU 版 PyTorch 无法使用 GPU。请单独安装 CUDA 12.4 版 PyTorch：

```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

验证：

```python
import torch
print(torch.__version__)          # 2.6.0+cu124
print(torch.cuda.is_available())  # True
print(torch.cuda.get_device_name(0))  # NVIDIA GeForce RTX 4070 Laptop GPU
```

## 数据准备

1. 将原始 YOLO OBB 数据放在源目录（默认 `E:\openmm\zhuangji_data`，含 `images/` 与 `labels/`）。
2. 运行划分脚本，生成 `dataset/` 并写入 `zhuangji.yaml`：

```bash
python prepare_dataset.py
```

`zhuangji.yaml` 示例：

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: zhuangji
```

## 训练

**初次训练**（从预训练权重开始，100 epochs，patience=20）：

```bash
python train.py
```

输出目录：`runs/zhuangji_obb/`。

**续训（CPU）**（从初次训练的 `last.pt` 继续，最多 250 epochs，patience=50）：

```bash
python train_resume.py
```

输出目录：`runs/zhuangji_obb-2/`（若同名目录已存在，Ultralytics 会自动递增后缀）。

**GPU 微调**（从 CPU 最佳权重继续，RTX 4070，batch=8）：

```bash
python train_gpu.py
```

输出目录：`runs/zhuangji_obb_gpu/`。

### 训练配置

| 项 | 初次训练 | CPU 续训 | GPU 微调 |
|----|----------|----------|----------|
| 输入尺寸 | 640 | 640 | 640 |
| Batch | 4 | 4 | 8 |
| 设备 | CPU | CPU | CUDA:0 (RTX 4070) |
| Workers | 0 | 0 | 4 |
| Epochs / Patience | 100 / 20 | 250 / 50 | 150 / 30 |
| PyTorch | 2.7.0+cpu | 2.7.0+cpu | 2.6.0+cu124 |

## 最佳权重

| 运行 | 路径 | mAP50-95 | 说明 |
|------|------|----------|------|
| 初次训练 | `runs/zhuangji_obb/weights/best.pt` | 0.701 | 46 epochs 早停 |
| **CPU 续训（推荐推理）** | **`runs/zhuangji_obb-2/weights/best.pt`** | **0.727** | 95 epochs 早停，综合指标最高 |
| GPU 微调 | `runs/zhuangji_obb_gpu/weights/best.pt` | 0.723 | 65 epochs 早停（best @ epoch 15） |

请使用 **`runs/zhuangji_obb-2/weights/best.pt`** 进行推理与部署（mAP50-95 略高于 GPU 运行）。

## 验证指标

### CPU vs GPU 对比

| 运行 | 设备 | 最佳 Epoch | Precision | Recall | mAP50 | mAP50-95 |
|------|------|------------|-----------|--------|-------|----------|
| `zhuangji_obb-2` | CPU (i7-14650HX) | 45 | 0.988 | 0.997 | 0.994 | **0.727** |
| `zhuangji_obb_gpu` | GPU (RTX 4070) | 15 | 0.988 | 0.985 | 0.994 | 0.723 |

GPU 训练约 **4 分钟**（65 epochs）；CPU 续训约 **27 分钟**（95 epochs）。CPU 续训在 mAP50-95 上略优（+0.003），故推荐 CPU 最佳权重用于推理。

完整逐 epoch 记录见各运行目录下的 `results.csv`。

## 推理

### 640×640 切片（标准）

Python：

```python
from ultralytics import YOLO

model = YOLO("runs/zhuangji_obb-2/weights/best.pt")
results = model.predict("dataset/images/val/3a50aa95-original_cropped_17920_14080.jpg", imgsz=640)
results[0].show()
```

CLI：

```bash
yolo obb predict model=runs/zhuangji_obb-2/weights/best.pt source=dataset/images/val imgsz=640
```

### 大图航拍推理（`predict_aerial.py`）

对超大 TIFF 航拍图（如 `21537×17132`）做分块 OBB 检测。默认启用 **two-stage 两阶段粗筛**：

1. **Stage 1 粗扫**：stride=1280，约 238 块，快速定位热区
2. **Stage 2 精扫**：仅在热区内 640 窗口精细检测，约 226 块
3. 合并 NMS 输出 JSON + 预览图

**推荐命令（Intel CPU）：**

```powershell
python predict_aerial.py `
  -i "C:\path\to\large.tif" `
  --json out.json `
  --preview out.jpg `
  --conf 0.5 `
  --strategy two-stage `
  --backend openvino `
  --profile
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--strategy two-stage` | 启用 | 粗扫 + 热区精扫，tile 数 1428 → ~464 |
| `--full-scan` | 关 | 全图逐块扫描（召回最高，速度较慢） |
| `--batch-size` | 8 | 批量推理；OpenVINO 自动降为 1；`--turbo` 默认 64 |
| `--backend openvino` | pt | Intel CPU 推荐；首次自动导出，失败回退 PyTorch |
| `--backend engine` | — | TensorRT（需 Python≥3.10 + tensorrt-cu12） |
| `--int8` | 关 | OpenVINO INT8 量化 |
| `--overlap` | 96 | 精扫块重叠像素 |
| `--coarse-conf` | 0.35 | 粗扫置信度 |
| `--hot-margin` | 320 | 热区外扩像素 |
| `--half` | 关 | FP16 半精度（GPU） |
| `--prefetch` | 关 | 后台预读下一批 tile |
| `--profile` | 关 | 分项耗时统计 |
| `--skip-preview` | 关 | 跳过预览图 |
| `--turbo` | 关 | FP16 + batch64 + prefetch + GPU + 跳过预览 |

**推荐命令（NVIDIA GPU）：**

```bash
python predict_aerial.py \
  -i /path/to/large.tif \
  --json out.json \
  --turbo --profile --conf 0.5 --device 0
```

#### GPU benchmark（`21537×17132`，RTX 3090，conf=0.5）

| 配置 | Pipeline 耗时 | 推理 | 检测数 |
|------|--------------|------|--------|
| 基线 (batch=8, FP32) | 7.4s | 1.4s | 550 |
| **`--turbo` (FP16, batch=64, prefetch)** | **4.4s** | **1.8s** | 550 |

运行 GPU benchmark：

```bash
python benchmark_gpu.py -i /path/to/14_A1.tif
```

#### 航拍 benchmark（Intel CPU）

| 配置 | 耗时 | 检测数 | tile 数 |
|------|------|--------|---------|
| 旧版串行全扫 | ~1654s | 587 | 1428 |
| batch 全扫 (PyTorch) | 238s | 587 | 1428 |
| two-stage (PyTorch) | 96s | 550 | 464 |
| **two-stage (OpenVINO FP32)** | **14s** | 551 | 464 |
| two-stage (OpenVINO INT8) | 14s | 561 | 462 |

OpenVINO 导出模型会在首次运行时自动生成于 `runs/zhuangji_obb-2/weights/best_openvino_model/`。INT8 模型使用 `--int8`，校准数据默认 `zhuangji.yaml`。

运行完整 benchmark：

```bash
python benchmark_aerial.py
```

## 仓库结构

```
├── train.py                 # 初次训练
├── train_resume.py          # CPU 续训
├── train_gpu.py             # GPU 微调（需 cu124 PyTorch）
├── prepare_dataset.py       # 划分 train/val
├── predict_aerial.py        # 大图航拍分块 OBB 推理（two-stage / GPU turbo / OpenVINO）
├── benchmark_aerial.py      # 航拍速度 benchmark（CPU）
├── benchmark_gpu.py         # GPU 优化 benchmark
├── zhuangji.yaml            # 数据集配置
├── requirements.txt
├── yolo26n-obb.pt           # 预训练 OBB 权重
├── dataset/                 # 图像与 OBB 标签
├── runs/zhuangji_obb/       # 初次训练
├── runs/zhuangji_obb-2/     # CPU 续训（推荐 best.pt）
├── runs/zhuangji_obb_gpu/   # GPU 微调
└── runs/benchmark_aerial/   # 航拍 benchmark 汇总
```

## License

模型与代码遵循 Ultralytics 及相关依赖的各自许可证。
