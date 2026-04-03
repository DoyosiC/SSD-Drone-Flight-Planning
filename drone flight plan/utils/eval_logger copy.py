# utils/eval_logger.py
from __future__ import annotations

import csv
import os
import time
from typing import Optional, Tuple

BBOX = Tuple[float, float, float, float]


class CsvLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.f = open(path, "w", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        self.w.writerow([
            "t",
            "mode",
            "label",
            "conf",
            "cx", "cy",
            "x1", "y1", "x2", "y2",
            "frame_w", "frame_h",
            "area_ratio",
            "age_det",
            "infer_dt",
            "vx_cmd", "vy_cmd", "vz_cmd", "yaw_cmd",
            "autopilot",
            "manual_active",
            "track_mode",
        ])

    def log(self, *,
            mode: str,
            label: str,
            conf: float,
            bbox: Optional[BBOX],
            frame_wh: Tuple[int, int],
            area_ratio: Optional[float],
            age_det: float,
            infer_dt: float,
            cmd: Tuple[int, int, int, int],
            autopilot: bool,
            manual_active: bool,
            track_mode: int):
        t = time.time()
        fw, fh = frame_wh
        vx, vy, vz, yaw = cmd

        if bbox is None:
            row = [t, mode, label, conf,
                   "", "", "", "", "", "",
                   fw, fh,
                   ("" if area_ratio is None else area_ratio),
                   age_det, infer_dt,
                   vx, vy, vz, yaw,
                   int(autopilot), int(manual_active), int(track_mode)]
        else:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            row = [t, mode, label, conf,
                   cx, cy, x1, y1, x2, y2,
                   fw, fh,
                   ("" if area_ratio is None else area_ratio),
                   age_det, infer_dt,
                   vx, vy, vz, yaw,
                   int(autopilot), int(manual_active), int(track_mode)]

        self.w.writerow(row)

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass
