# utils/eval_metrics.py
from __future__ import annotations

import os
import math
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple, List

import pandas as pd


# -----------------------------
# Helpers
# -----------------------------
def _to_float_series(s: pd.Series) -> pd.Series:
    # logger writes "" for None in CSV; pandas reads as NaN
    return pd.to_numeric(s, errors="coerce")


def _safe_mean(x: pd.Series) -> Optional[float]:
    x = x.dropna()
    if len(x) == 0:
        return None
    return float(x.mean())


def _safe_std(x: pd.Series) -> Optional[float]:
    x = x.dropna()
    if len(x) == 0:
        return None
    return float(x.std(ddof=1)) if len(x) >= 2 else 0.0


def _safe_min(x: pd.Series) -> Optional[float]:
    x = x.dropna()
    if len(x) == 0:
        return None
    return float(x.min())


def _safe_max(x: pd.Series) -> Optional[float]:
    x = x.dropna()
    if len(x) == 0:
        return None
    return float(x.max())


def _safe_quantile(x: pd.Series, q: float) -> Optional[float]:
    x = x.dropna()
    if len(x) == 0:
        return None
    return float(x.quantile(q))


def _bool_rate(x: pd.Series) -> Optional[float]:
    x = x.dropna()
    if len(x) == 0:
        return None
    # expects 0/1 int
    return float(x.mean())


def _run_lengths(mask: pd.Series) -> List[int]:
    """
    Return lengths of consecutive True runs.
    mask is boolean Series.
    """
    vals = mask.fillna(False).astype(bool).to_numpy()
    runs: List[int] = []
    cur = 0
    for v in vals:
        if v:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
                cur = 0
    if cur > 0:
        runs.append(cur)
    return runs


def _estimate_fps(df: pd.DataFrame) -> Optional[float]:
    if "t" not in df.columns:
        return None
    t = _to_float_series(df["t"]).dropna()
    if len(t) < 2:
        return None
    dt = t.diff().dropna()
    dt = dt[dt > 0]
    if len(dt) == 0:
        return None
    # robust using median
    return float(1.0 / dt.median())


# -----------------------------
# Metrics schema
# -----------------------------
@dataclass
class Metrics:
    # general
    n_rows: int
    fps_est: Optional[float]

    # 1) detection stability (confidence)
    conf_mean: Optional[float]
    conf_std: Optional[float]
    conf_p10: Optional[float]
    conf_p50: Optional[float]
    conf_p90: Optional[float]
    detect_ratio: Optional[float]     # mode == DETECT rate
    hold_ratio: Optional[float]       # mode == HOLD rate
    lost_ratio: Optional[float]       # mode == LOST rate

    detect_run_mean_s: Optional[float]
    detect_run_max_s: Optional[float]
    hold_run_mean_s: Optional[float]
    hold_run_max_s: Optional[float]

    infer_dt_mean_s: Optional[float]
    infer_dt_p95_s: Optional[float]
    infer_dt_max_s: Optional[float]

    # 2) in-frame keep rate (during tracking)
    in_frame_rate_all: Optional[float]
    in_frame_rate_detect: Optional[float]
    in_frame_rate_hold: Optional[float]

    center_err_mean_px: Optional[float]   # optional if present
    center_err_p95_px: Optional[float]

    # 3) distance behavior (z tracking)
    z_ema_mean_m: Optional[float]
    z_ema_std_m: Optional[float]
    z_err_mean_m: Optional[float]
    z_err_p95_abs_m: Optional[float]
    settle_time_s: Optional[float]        # time until |z_err| <= band and stays
    overshoot_m: Optional[float]          # max(z_ema - target) when approaching
    approach_success_rate: Optional[float]  # fraction of time |z_err| <= band

    # control behavior (for analysis)
    rc_vx_mean: Optional[float]
    rc_vx_p95_abs: Optional[float]
    rc_yaw_p95_abs: Optional[float]


# -----------------------------
# Core computation
# -----------------------------
def compute_metrics(df: pd.DataFrame, settle_band_m: float = 0.05, settle_hold_s: float = 1.0) -> Metrics:
    """
    Compute metrics for:
      1) confidence stability
      2) in-frame keep rate
      3) distance response behavior

    settle_band_m: threshold for |z_err| to be considered "settled"
    settle_hold_s: must stay within band for this duration to count as settled
    """

    n = int(len(df))
    fps = _estimate_fps(df)

    # --- mode ratios ---
    mode = df["mode"].astype(str) if "mode" in df.columns else pd.Series([""] * n)
    detect_mask = (mode == "DETECT")
    hold_mask = (mode == "HOLD")
    lost_mask = (mode == "LOST")

    detect_ratio = float(detect_mask.mean()) if n > 0 else None
    hold_ratio = float(hold_mask.mean()) if n > 0 else None
    lost_ratio = float(lost_mask.mean()) if n > 0 else None

    # --- confidence stats (only when detect/hold has conf) ---
    conf = _to_float_series(df["conf"]) if "conf" in df.columns else pd.Series([math.nan] * n)
    # Some rows may be empty when LOST; we analyze for DETECT only and for all non-NaN.
    conf_valid = conf.dropna()

    conf_mean = _safe_mean(conf_valid)
    conf_std = _safe_std(conf_valid)
    conf_p10 = _safe_quantile(conf_valid, 0.10)
    conf_p50 = _safe_quantile(conf_valid, 0.50)
    conf_p90 = _safe_quantile(conf_valid, 0.90)

    # --- infer_dt stats ---
    infer_dt = _to_float_series(df["infer_dt_s"]) if "infer_dt_s" in df.columns else pd.Series([math.nan] * n)
    infer_dt_mean = _safe_mean(infer_dt)
    infer_dt_p95 = _safe_quantile(infer_dt, 0.95)
    infer_dt_max = _safe_max(infer_dt)

    # --- run length stats (convert to seconds via fps) ---
    def run_stats(mask: pd.Series) -> Tuple[Optional[float], Optional[float]]:
        runs = _run_lengths(mask)
        if len(runs) == 0:
            return None, None
        if fps is None:
            # in frames
            return float(sum(runs) / len(runs)), float(max(runs))
        # in seconds
        return float((sum(runs) / len(runs)) / fps), float(max(runs) / fps)

    detect_run_mean_s, detect_run_max_s = run_stats(detect_mask)
    hold_run_mean_s, hold_run_max_s = run_stats(hold_mask)

    # --- in-frame rate ---
    in_frame = _to_float_series(df["in_frame"]) if "in_frame" in df.columns else pd.Series([math.nan] * n)

    in_frame_rate_all = _bool_rate(in_frame)
    in_frame_rate_detect = _bool_rate(in_frame[detect_mask]) if n > 0 else None
    in_frame_rate_hold = _bool_rate(in_frame[hold_mask]) if n > 0 else None

    # --- center error (if cx,cy and frame_w/h exist) ---
    center_err = None
    if all(c in df.columns for c in ["cx", "cy", "frame_w", "frame_h"]):
        cx = _to_float_series(df["cx"])
        cy = _to_float_series(df["cy"])
        fw = _to_float_series(df["frame_w"])
        fh = _to_float_series(df["frame_h"])
        ex = cx - fw * 0.5
        ey = cy - fh * 0.5
        center_err = (ex.pow(2) + ey.pow(2)).pow(0.5)

    center_err_mean = _safe_mean(center_err) if center_err is not None else None
    center_err_p95 = _safe_quantile(center_err, 0.95) if center_err is not None else None

    # --- distance behavior (z_ema, z_err, target_z) ---
    z_ema = _to_float_series(df["z_ema_m"]) if "z_ema_m" in df.columns else pd.Series([math.nan] * n)
    z_err = _to_float_series(df["z_err_m"]) if "z_err_m" in df.columns else pd.Series([math.nan] * n)
    target_z = _to_float_series(df["target_z_m"]) if "target_z_m" in df.columns else pd.Series([math.nan] * n)

    z_ema_mean = _safe_mean(z_ema)
    z_ema_std = _safe_std(z_ema)

    z_err_mean = _safe_mean(z_err)
    z_err_abs = z_err.abs()
    z_err_p95_abs = _safe_quantile(z_err_abs, 0.95)

    # settle time: first time index where |z_err| within band for >= settle_hold_s
    settle_time_s = None
    if fps is not None and "t" in df.columns:
        within = (z_err_abs <= float(settle_band_m))
        runs = _run_lengths(within)
        # need run length >= settle_hold_s * fps
        min_frames = int(math.ceil(float(settle_hold_s) * fps))
        if min_frames <= 1:
            min_frames = 1

        if within.notna().any():
            # find first run that satisfies
            vals = within.fillna(False).astype(bool).to_numpy()
            start = None
            cur = 0
            for i, v in enumerate(vals):
                if v:
                    if start is None:
                        start = i
                    cur += 1
                    if cur >= min_frames:
                        # settle time = t[start]
                        t0 = _to_float_series(df["t"]).iloc[start]
                        t_first = _to_float_series(df["t"]).iloc[0]
                        if pd.notna(t0) and pd.notna(t_first):
                            settle_time_s = float(t0 - t_first)
                        break
                else:
                    start = None
                    cur = 0

    # overshoot: max(z_ema - target_z) when both valid
    overshoot = None
    if "z_ema_m" in df.columns and "target_z_m" in df.columns:
        diff = (z_ema - target_z)
        overshoot = _safe_max(diff)

    # approach success: fraction within band (when z_err exists)
    approach_success_rate = None
    if z_err_abs is not None:
        approach_success_rate = _bool_rate((z_err_abs <= float(settle_band_m)).astype(float))

    # --- control behavior ---
    rc_vx = _to_float_series(df["rc_vx"]) if "rc_vx" in df.columns else pd.Series([math.nan] * n)
    rc_yaw = _to_float_series(df["rc_yaw"]) if "rc_yaw" in df.columns else pd.Series([math.nan] * n)

    rc_vx_mean = _safe_mean(rc_vx)
    rc_vx_p95_abs = _safe_quantile(rc_vx.abs(), 0.95) if rc_vx is not None else None
    rc_yaw_p95_abs = _safe_quantile(rc_yaw.abs(), 0.95) if rc_yaw is not None else None

    return Metrics(
        n_rows=n,
        fps_est=fps,

        conf_mean=conf_mean,
        conf_std=conf_std,
        conf_p10=conf_p10,
        conf_p50=conf_p50,
        conf_p90=conf_p90,
        detect_ratio=detect_ratio,
        hold_ratio=hold_ratio,
        lost_ratio=lost_ratio,

        detect_run_mean_s=detect_run_mean_s,
        detect_run_max_s=detect_run_max_s,
        hold_run_mean_s=hold_run_mean_s,
        hold_run_max_s=hold_run_max_s,

        infer_dt_mean_s=infer_dt_mean,
        infer_dt_p95_s=infer_dt_p95,
        infer_dt_max_s=infer_dt_max,

        in_frame_rate_all=in_frame_rate_all,
        in_frame_rate_detect=in_frame_rate_detect,
        in_frame_rate_hold=in_frame_rate_hold,

        center_err_mean_px=center_err_mean,
        center_err_p95_px=center_err_p95,

        z_ema_mean_m=z_ema_mean,
        z_ema_std_m=z_ema_std,
        z_err_mean_m=z_err_mean,
        z_err_p95_abs_m=z_err_p95_abs,
        settle_time_s=settle_time_s,
        overshoot_m=overshoot,
        approach_success_rate=approach_success_rate,

        rc_vx_mean=rc_vx_mean,
        rc_vx_p95_abs=rc_vx_p95_abs,
        rc_yaw_p95_abs=rc_yaw_p95_abs,
    )


# -----------------------------
# I/O
# -----------------------------
def read_log(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xlsx":
        return pd.read_excel(path, sheet_name=0)
    return pd.read_csv(path)


def metrics_to_dataframe(m: Metrics) -> pd.DataFrame:
    d = asdict(m)
    return pd.DataFrame([d])


def save_metrics(m: Metrics, out_path: str) -> None:
    df = metrics_to_dataframe(m)
    ext = os.path.splitext(out_path)[1].lower()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if ext == ".xlsx":
        with pd.ExcelWriter(out_path, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="metrics")
    else:
        df.to_csv(out_path, index=False)


def compute_metrics_from_file(log_path: str,
                              out_path: Optional[str] = None,
                              settle_band_m: float = 0.05,
                              settle_hold_s: float = 1.0) -> Metrics:
    df = read_log(log_path)
    m = compute_metrics(df, settle_band_m=settle_band_m, settle_hold_s=settle_hold_s)
    if out_path:
        save_metrics(m, out_path)
    return m


# -----------------------------
# CLI
# -----------------------------
def _fmt(x: Optional[float], nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x))):
        return "-"
    return f"{x:.{nd}f}"


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=str, required=True, help="path to log (.csv or .xlsx)")
    p.add_argument("--out", type=str, default="", help="output metrics file (.csv or .xlsx)")
    p.add_argument("--settle_band", type=float, default=0.05, help="|z_err| band [m]")
    p.add_argument("--settle_hold", type=float, default=1.0, help="time within band to count as settled [s]")
    args = p.parse_args()

    out_path = args.out if args.out else None
    m = compute_metrics_from_file(
        args.log,
        out_path=out_path,
        settle_band_m=float(args.settle_band),
        settle_hold_s=float(args.settle_hold),
    )

    print("=== METRICS ===")
    print("rows:", m.n_rows, "fps_est:", _fmt(m.fps_est, 2))
    print("[1] Detection stability")
    print(" conf mean/std:", _fmt(m.conf_mean, 3), _fmt(m.conf_std, 3),
          "p10/p50/p90:", _fmt(m.conf_p10, 3), _fmt(m.conf_p50, 3), _fmt(m.conf_p90, 3))
    print(" detect/hold/lost ratio:", _fmt(m.detect_ratio, 3), _fmt(m.hold_ratio, 3), _fmt(m.lost_ratio, 3))
    print(" detect run mean/max [s]:", _fmt(m.detect_run_mean_s, 2), _fmt(m.detect_run_max_s, 2))
    print(" hold   run mean/max [s]:", _fmt(m.hold_run_mean_s, 2), _fmt(m.hold_run_max_s, 2))
    print(" infer_dt mean/p95/max [s]:", _fmt(m.infer_dt_mean_s, 3), _fmt(m.infer_dt_p95_s, 3), _fmt(m.infer_dt_max_s, 3))

    print("[2] In-frame keep rate")
    print(" in_frame all/detect/hold:", _fmt(m.in_frame_rate_all, 3), _fmt(m.in_frame_rate_detect, 3), _fmt(m.in_frame_rate_hold, 3))
    print(" center_err mean/p95 [px]:", _fmt(m.center_err_mean_px, 1), _fmt(m.center_err_p95_px, 1))

    print("[3] Distance behavior")
    print(" z_ema mean/std [m]:", _fmt(m.z_ema_mean_m, 3), _fmt(m.z_ema_std_m, 3))
    print(" z_err mean [m], p95|err| [m]:", _fmt(m.z_err_mean_m, 3), _fmt(m.z_err_p95_abs_m, 3))
    print(" settle_time [s]:", _fmt(m.settle_time_s, 2),
          "overshoot [m]:", _fmt(m.overshoot_m, 3),
          "success_rate:", _fmt(m.approach_success_rate, 3))

    print("[Control]")
    print(" rc_vx mean:", _fmt(m.rc_vx_mean, 2),
          "rc_vx p95| |:", _fmt(m.rc_vx_p95_abs, 2),
          "rc_yaw p95| |:", _fmt(m.rc_yaw_p95_abs, 2))

    if out_path:
        print("saved:", out_path)


if __name__ == "__main__":
    main()
