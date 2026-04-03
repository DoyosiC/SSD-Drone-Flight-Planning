# utils/eval_batch.py
from __future__ import annotations

import os
import glob
import math
from dataclasses import asdict
from typing import List, Dict, Any, Optional

from utils.eval_metrics import compute_metrics_from_csv, Metrics


def _mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))


def _std(xs: List[float], mean: Optional[float] = None) -> float:
    n = len(xs)
    if n <= 1:
        return 0.0
    m = _mean(xs) if mean is None else mean
    v = sum((x - m) ** 2 for x in xs) / (n - 1)  # sample std
    return math.sqrt(v)


def aggregate_metrics(metrics_list: List[Metrics]) -> Dict[str, Dict[str, float]]:
    """
    Metrics のリストから mean/std を返す。
    None の項目は集計から除外。
    """
    keys = list(asdict(metrics_list[0]).keys())
    out: Dict[str, Dict[str, float]] = {}

    for k in keys:
        vals: List[float] = []
        for m in metrics_list:
            v = getattr(m, k)
            if v is None:
                continue
            vals.append(float(v))

        if len(vals) == 0:
            continue

        mval = _mean(vals)
        sval = _std(vals, mval)
        out[k] = {"mean": mval, "std": sval, "n": float(len(vals))}
    return out


def run_batch(
    log_dir: str,
    pattern: str = "ics_tracking_*.csv",
    success_seg_sec: float = 3.0,
) -> None:
    paths = sorted(glob.glob(os.path.join(log_dir, pattern)))
    if not paths:
        print(f"[BATCH] No csv found: dir={log_dir} pattern={pattern}")
        return

    metrics_list: List[Metrics] = []
    failed: List[str] = []

    for p in paths:
        try:
            m = compute_metrics_from_csv(p, success_seg_sec=success_seg_sec)
            metrics_list.append(m)
        except Exception:
            failed.append(p)

    print(f"[BATCH] files: {len(paths)}  ok: {len(metrics_list)}  failed: {len(failed)}")
    if failed:
        print("[BATCH] failed files:")
        for f in failed:
            print("  -", f)

    if not metrics_list:
        return

    agg = aggregate_metrics(metrics_list)

    # 論文向けに重要な順で表示
    order = [
        "total_time",
        "detect_ratio",
        "lost_ratio",
        "mean_age_det",
        "mean_abs_center_err_px",
        "mean_follow_segment_s",
        "follow_success_ratio",
    ]

    print("\n[BATCH] mean ± std (n=number of valid files)")
    for k in order:
        if k not in agg:
            continue
        mean = agg[k]["mean"]
        std = agg[k]["std"]
        n = int(agg[k]["n"])
        if k in ("detect_ratio", "lost_ratio", "follow_success_ratio"):
            print(f"  {k:24s}: {mean:.4f} ± {std:.4f}  (n={n})")
        elif k in ("mean_abs_center_err_px",):
            print(f"  {k:24s}: {mean:.1f} ± {std:.1f}  (n={n})")
        else:
            print(f"  {k:24s}: {mean:.3f} ± {std:.3f}  (n={n})")


if __name__ == "__main__":
    # 例: python -m utils.eval_batch
    LOG_DIR = "./logs"
    run_batch(LOG_DIR, pattern="ics_tracking_*.csv", success_seg_sec=3.0)
