# Tello Fruit Tracker
**SSD-based fruit detection & autonomous tracking system for DJI Tello drone**
SSDによる果物検出 + DJI Tello自律追跡システム

---

## Overview / 概要

This project implements real-time object detection (apple / orange / banana) using a fine-tuned SSD (Single Shot Detector) model, combined with autonomous drone control for follow-tracking via DJI Tello.

SSD（Single Shot Detector）をファインチューニングして果物（apple / orange / banana）をリアルタイム検出し、DJI Telloドローンで対象を自律追跡するシステムです。

---

## Features / 機能

- **SSD object detection** — VGG16 backbone, fine-tuned on custom fruit dataset
- **Autonomous tracking** — Center-error PID-style yaw/forward control
- **Continuity controller** — DETECT → HOLD → PREDICT → LOST state machine to handle missed detections
- **Pinhole depth estimation** — Estimates target distance from bounding-box height using camera intrinsics
- **Evaluation pipeline** — Per-session CSV logging + batch metrics (detect ratio, center error, settle time, etc.)

---

## Directory Structure / ディレクトリ構成

```
tello-fruit-tracker/
│
├── main.py                    # Entry point (manual flight + stream)
├── main_tracking.py           # Entry point (autonomous tracking)
│
├── utils/
│   ├── ctrl.py                # Tello connection, RC control, stream HUD
│   ├── model.py               # Model wrapper (load weights, predict, annotate)
│   ├── tracking_continuity.py # State machine: DETECT/HOLD/PREDICT/LOST
│   ├── pinhole.py             # Pinhole camera model, depth estimation
│   ├── ssd_model.py           # SSD network definition (VGG + Extras + Detect)
│   ├── ssd_predict_show.py    # Inference helper + OpenCV/matplotlib drawing
│   ├── match.py               # Box matching / IoU / encode (from ssd.pytorch)
│   ├── data_augumentation.py  # Data augmentation transforms
│   ├── eval_logger.py         # CSV/XLSX logging (EvalRow schema)
│   ├── eval_metrics.py        # Metrics computation from log files (pandas)
│   └── eval_batch.py          # Batch aggregation of multiple sessions
│
├── weights/
│   └── ssd_finetuned_200_filter.pth   # Fine-tuned model weights (not committed)
│
├── logs/                      # Auto-generated session CSV logs
│
├── train_base_cfg_filterv1.py # Training script
├── testcam.py                 # Camera stream test
├── test_Tello_state.py        # Tello sensor state test
│
├── .gitignore
└── README.md
```

---

## Requirements / 動作環境

```
Python >= 3.9
torch >= 2.0
torchvision
opencv-python
djitellopy
numpy
pandas
openpyxl        # xlsx logging (optional)
keyboard        # key input (Linux/macOS: requires sudo)
```

Install / インストール:
```bash
pip install torch torchvision opencv-python djitellopy numpy pandas openpyxl keyboard
```

---

## Quick Start / 使い方

### 1. Connect to Tello Wi-Fi
Telloの Wi-Fi（SSID: `TELLO-xxxxxx`）にPCを接続してください。

### 2. Run tracking / 追跡実行
```bash
python main_tracking.py
```

### 3. Run manual control / 手動操作
```bash
python main.py
```

### Key bindings / キー操作
| Key | Action |
|-----|--------|
| `T` | Takeoff / 離陸 |
| `L` | Land / 着陸 |
| `W/A/S/D` | Forward/Left/Back/Right |
| `↑/↓` | Up / Down |
| `←/→` | Yaw left / right |
| `Q` | Quit / 終了 |

---

## Model / モデル

- Architecture: SSD300 (VGG16 backbone)
- Input size: 300 × 300
- Classes: `apple`, `orange`, `banana` (+ background)
- Dataset: Custom VOC-format dataset
- Weights: `weights/ssd_finetuned_200_filter.pth` *(not included in repo)*

---

## Evaluation / 評価

Session logs are saved to `logs/` as CSV. To compute metrics for a single session:

セッションログは `logs/` にCSV保存されます。単一セッションの指標計算：

```bash
python -m utils.eval_metrics --log logs/session_001.csv --out logs/metrics_001.csv
```

Batch aggregation across sessions / 複数セッションの一括集計:

```bash
python -m utils.eval_batch
```

Key metrics / 主な指標:
- `detect_ratio` — Fraction of frames with successful detection
- `center_err_mean_px` — Mean pixel distance from frame center
- `z_err_p95_abs_m` — 95th percentile absolute depth error
- `settle_time_s` — Time to reach stable distance

---

## License / ライセンス

Parts of `utils/match.py` are derived from [amdegroot/ssd.pytorch](https://github.com/amdegroot/ssd.pytorch) (MIT License).

The rest of this project is released under the MIT License. See `LICENSE` for details.

---

## Acknowledgements / 謝辞

- [amdegroot/ssd.pytorch](https://github.com/amdegroot/ssd.pytorch) — SSD implementation reference
- [DJITelloPy](https://github.com/damiafuentes/DJITelloPy) — Tello SDK wrapper
