# utils/eval_metrics.py
from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple


@dataclass
class Metrics:
    total_time: float
    detect_ratio: float
    lost_ratio: float
    mean_age_det: float
    mean_abs_center_err_px: Optional[float]
    mean_follow_segment_s: Optional[float]
    follow_success_ratio: Optional[float]


def _to_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def compute_metrics_from_csv(
    csv_path: str,
    *,
    follow_modes: Tuple[str, ...] = ("DETECT", "HOLD", "PREDICT"),
    success_seg_sec: float = 3.0,
) -> Metrics:
    """
    CSVログ（utils/eval_logger.py）から論文用の基本指標を算出する。

    - detect_ratio: mode==DETECT の割合（ログ行基準）
    - lost_ratio: mode==LOST の割合
    - mean_age_det: age_det の平均
    - mean_abs_center_err_px: |cx - frame_w/2| の平均（cxがある行のみ）
    - mean_follow_segment_s: follow_modes の連続区間の平均長
    - follow_success_ratio: follow区間が success_seg_sec 以上の割合
    """
    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    if len(rows) < 2:
        return Metrics(0.0, 0.0, 1.0, 1e9, None, None, None)

    t0 = _to_float(rows[0].get("t", "")) or 0.0
    t1 = _to_float(rows[-1].get("t", "")) or t0
    total_time = max(1e-6, t1 - t0)

    n = len(rows)
    n_detect = sum(1 for x in rows if x.get("mode") == "DETECT")
    n_lost = sum(1 for x in rows if x.get("mode") == "LOST")

    ages = [_to_float(x.get("age_det", "")) for x in rows]
    ages_f = [a for a in ages if a is not None and a < 1e8]
    mean_age = (sum(ages_f) / len(ages_f)) if ages_f else 1e9

    # center error
    ce: List[float] = []
    for x in rows:
        cx = _to_float(x.get("cx", ""))
        fw = _to_float(x.get("frame_w", ""))
        if cx is None or fw is None:
            continue
        ce.append(abs(cx - fw * 0.5))
    mean_abs_center_err = (sum(ce) / len(ce)) if ce else None

    # follow segments (continuous intervals where mode in follow_modes)
    segs: List[float] = []
    cur_start: Optional[float] = None

    for x in rows:
        t = _to_float(x.get("t", ""))
        if t is None:
            continue
        mode = x.get("mode", "")

        if mode in follow_modes:
            if cur_start is None:
                cur_start = t
        else:
            if cur_start is not None:
                segs.append(t - cur_start)
                cur_start = None

    if cur_start is not None:
        segs.append((t1 - cur_start))

    mean_seg = (sum(segs) / len(segs)) if segs else None
    success = (sum(1 for s in segs if s >= success_seg_sec) / len(segs)) if segs else None

    return Metrics(
        total_time=total_time,
        detect_ratio=n_detect / n,
        lost_ratio=n_lost / n,
        mean_age_det=mean_age,
        mean_abs_center_err_px=mean_abs_center_err,
        mean_follow_segment_s=mean_seg,
        follow_success_ratio=success,
    )
