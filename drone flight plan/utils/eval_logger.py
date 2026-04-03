# utils/eval_logger.py
from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union

try:
    import openpyxl  # type: ignore
except Exception:
    openpyxl = None  # xlsx未使用なら不要

BBOX = Tuple[float, float, float, float]


# -----------------------------
# Schema
# -----------------------------
HEADER = [
    # time
    "t", "frame_idx",

    # detection / continuity
    "mode", "label", "conf",
    "age_det_s", "infer_dt_s",

    # bbox / screen
    "cx", "cy",
    "x1", "y1", "x2", "y2",
    "frame_w", "frame_h",
    "in_frame",
    "area_ratio",

    # range (pinhole)
    "z_est_m", "z_ema_m", "z_err_m", "target_z_m",

    # control (rc)
    "rc_vx", "rc_vy", "rc_vz", "rc_yaw",
    "autopilot", "manual_active",
    "track_mode",
]


@dataclass
class EvalRow:
    t: float
    frame_idx: int

    mode: str
    label: str
    conf: Optional[float]

    age_det_s: Optional[float]
    infer_dt_s: Optional[float]

    cx: Optional[float]
    cy: Optional[float]
    x1: Optional[float]
    y1: Optional[float]
    x2: Optional[float]
    y2: Optional[float]
    frame_w: int
    frame_h: int
    in_frame: Optional[int]
    area_ratio: Optional[float]

    z_est_m: Optional[float]
    z_ema_m: Optional[float]
    z_err_m: Optional[float]
    target_z_m: Optional[float]

    rc_vx: int
    rc_vy: int
    rc_vz: int
    rc_yaw: int
    autopilot: int
    manual_active: int
    track_mode: int

    def as_list(self) -> List[Union[str, int, float]]:
        def v(x):
            return "" if x is None else x

        return [
            self.t, self.frame_idx,
            self.mode, self.label, v(self.conf),
            v(self.age_det_s), v(self.infer_dt_s),
            v(self.cx), v(self.cy),
            v(self.x1), v(self.y1), v(self.x2), v(self.y2),
            self.frame_w, self.frame_h,
            v(self.in_frame),
            v(self.area_ratio),
            v(self.z_est_m), v(self.z_ema_m), v(self.z_err_m), v(self.target_z_m),
            self.rc_vx, self.rc_vy, self.rc_vz, self.rc_yaw,
            self.autopilot, self.manual_active,
            self.track_mode,
        ]


# -----------------------------
# Logger base
# -----------------------------
class EvalLoggerBase:
    def write(self, row: EvalRow) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


# -----------------------------
# CSV Logger
# -----------------------------
class CsvLogger(EvalLoggerBase):
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.f = open(path, "w", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        self.w.writerow(HEADER)

    def write(self, row: EvalRow) -> None:
        self.w.writerow(row.as_list())

    def close(self) -> None:
        try:
            self.f.close()
        except Exception:
            pass


# -----------------------------
# XLSX Logger (buffered)
# -----------------------------
class XlsxLogger(EvalLoggerBase):
    """
    XLSXは逐次保存が重いので、基本はメモリに貯めて close() で保存する。
    長時間ログが必要なら flush_every を指定。
    """
    def __init__(self, path: str, flush_every: int = 0):
        if openpyxl is None:
            raise RuntimeError("openpyxl が見つかりません。xlsx出力を使うなら openpyxl をインストールしてください。")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.flush_every = int(flush_every)

        self.wb = openpyxl.Workbook(write_only=True)
        self.ws = self.wb.create_sheet("log")
        self.ws.append(HEADER)

        self._count = 0

    def write(self, row: EvalRow) -> None:
        self.ws.append(row.as_list())
        self._count += 1
        if self.flush_every > 0 and (self._count % self.flush_every == 0):
            # 途中保存（落ちた時の保険）。重いので任意。
            self.wb.save(self.path)

    def close(self) -> None:
        try:
            self.wb.save(self.path)
        except Exception:
            pass


# -----------------------------
# Factory
# -----------------------------
def create_logger(path: str) -> EvalLoggerBase:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xlsx":
        return XlsxLogger(path)
    # default csv
    return CsvLogger(path)


# -----------------------------
# Helper computations
# -----------------------------
def bbox_center(bb: Optional[BBOX]) -> Tuple[Optional[float], Optional[float]]:
    if bb is None:
        return None, None
    x1, y1, x2, y2 = bb
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def bbox_in_frame(bb: Optional[BBOX], frame_w: int, frame_h: int) -> Optional[int]:
    if bb is None:
        return None
    x1, y1, x2, y2 = bb
    return int((x1 >= 0) and (y1 >= 0) and (x2 <= frame_w) and (y2 <= frame_h))


def bbox_area_ratio(bb: Optional[BBOX], frame_w: int, frame_h: int) -> Optional[float]:
    if bb is None:
        return None
    x1, y1, x2, y2 = bb
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    return float((w * h) / max(1.0, float(frame_w * frame_h)))
