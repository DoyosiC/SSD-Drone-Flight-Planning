"""
exp3_range_eval.py  —  Exp.3: Range Estimation Accuracy Evaluation
====================================================================
Usage (ground-truth CSV entry mode):
    python exp3_range_eval.py --mode record --log logs/session_001.xlsx --gt_cm 40

Usage (batch analysis from gt_table CSV):
    python exp3_range_eval.py --mode analyze --gt_table exp3_gt.csv --out exp3_results/

Ground-truth table CSV format (exp3_gt.csv):
    log_path,gt_cm
    logs/session_40cm_1.xlsx,40
    logs/session_40cm_2.xlsx,40
    logs/session_80cm_1.xlsx,80
    ...

Output
------
- exp3_results/exp3_per_trial.csv   : per-trial metrics
- exp3_results/exp3_summary.csv     : mean±std per gt_cm
- exp3_results/exp3_summary.xlsx    : same in Excel (if openpyxl available)
- Console table for copy-paste into paper
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Dict

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── eval_metrics helpers ─────────────────────────────────────────────────────
def _to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _safe_mean(x: pd.Series) -> Optional[float]:
    x = x.dropna()
    return float(x.mean()) if len(x) > 0 else None


def _safe_std(x: pd.Series) -> Optional[float]:
    x = x.dropna()
    if len(x) < 2:
        return 0.0
    return float(x.std(ddof=1))


def _safe_rmse(err: pd.Series) -> Optional[float]:
    err = err.dropna()
    if len(err) == 0:
        return None
    return float(math.sqrt((err ** 2).mean()))


# ── Per-trial result ──────────────────────────────────────────────────────────
@dataclass
class TrialResult:
    log_path: str
    gt_m: float                     # ground truth distance [m]

    n_rows: int
    n_valid_z: int                  # rows where z_est_m is not NaN

    z_est_mean_m: Optional[float]   # mean of raw estimates
    z_est_std_m: Optional[float]

    z_ema_mean_m: Optional[float]   # mean of EMA-filtered estimates
    z_ema_std_m: Optional[float]

    err_mean_m: Optional[float]     # z_ema_mean - gt  (signed)
    err_abs_mean_m: Optional[float] # |z_ema - gt| mean
    err_rmse_m: Optional[float]     # RMSE of (z_ema - gt)
    err_pct: Optional[float]        # (|err| / gt) * 100  [%]

    detect_ratio: Optional[float]


def analyze_trial(log_path: str, gt_cm: float) -> TrialResult:
    """Load one log file and compute range estimation metrics against gt_cm."""
    gt_m = gt_cm / 100.0
    ext = os.path.splitext(log_path)[1].lower()
    df = pd.read_excel(log_path) if ext == ".xlsx" else pd.read_csv(log_path)
    n_rows = len(df)

    z_est = _to_float_series(df["z_est_m"]) if "z_est_m" in df.columns else pd.Series(dtype=float)
    z_ema = _to_float_series(df["z_ema_m"]) if "z_ema_m" in df.columns else pd.Series(dtype=float)

    n_valid_z = int(z_est.notna().sum())

    # Error computed on EMA (smoothed) estimate — this is what the controller uses
    err_series = z_ema - gt_m

    err_mean = _safe_mean(err_series)
    err_abs_mean = _safe_mean(err_series.abs())
    err_rmse = _safe_rmse(err_series)
    err_pct = (err_abs_mean / gt_m * 100.0) if (err_abs_mean is not None and gt_m > 0) else None

    mode_col = df["mode"].astype(str) if "mode" in df.columns else pd.Series([""] * n_rows)
    detect_ratio = float((mode_col == "DETECT").mean()) if n_rows > 0 else None

    return TrialResult(
        log_path=log_path,
        gt_m=gt_m,
        n_rows=n_rows,
        n_valid_z=n_valid_z,
        z_est_mean_m=_safe_mean(z_est),
        z_est_std_m=_safe_std(z_est),
        z_ema_mean_m=_safe_mean(z_ema),
        z_ema_std_m=_safe_std(z_ema),
        err_mean_m=err_mean,
        err_abs_mean_m=err_abs_mean,
        err_rmse_m=err_rmse,
        err_pct=err_pct,
        detect_ratio=detect_ratio,
    )


# ── Summary per distance ──────────────────────────────────────────────────────
def summarize(trials: List[TrialResult]) -> pd.DataFrame:
    """Aggregate trials by gt_cm → mean±std table for the paper."""
    rows = []
    for t in trials:
        rows.append({
            "gt_cm": round(t.gt_m * 100),
            "gt_m": t.gt_m,
            "log": os.path.basename(t.log_path),
            "n_rows": t.n_rows,
            "n_valid_z": t.n_valid_z,
            "z_ema_mean_m": t.z_ema_mean_m,
            "z_ema_std_m": t.z_ema_std_m,
            "err_mean_m": t.err_mean_m,
            "err_abs_mean_m": t.err_abs_mean_m,
            "err_rmse_m": t.err_rmse_m,
            "err_pct": t.err_pct,
            "detect_ratio": t.detect_ratio,
        })
    per_trial_df = pd.DataFrame(rows)

    # Aggregate per gt_cm
    summary_rows = []
    for gt_cm, grp in per_trial_df.groupby("gt_cm"):
        def ms(col: str):
            vals = grp[col].dropna()
            if len(vals) == 0:
                return None, None
            return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) >= 2 else 0.0

        z_mu, z_sd = ms("z_ema_mean_m")
        e_mu, e_sd = ms("err_mean_m")
        ea_mu, ea_sd = ms("err_abs_mean_m")
        rmse_mu, rmse_sd = ms("err_rmse_m")
        pct_mu, pct_sd = ms("err_pct")
        dr_mu, _ = ms("detect_ratio")

        summary_rows.append({
            "gt_cm": gt_cm,
            "gt_m": gt_cm / 100.0,
            "n_trials": len(grp),
            "z_ema_mean_m": z_mu,
            "z_ema_std_m": z_sd,
            "err_mean_m (bias)": e_mu,
            "err_abs_mean_m": ea_mu,
            "err_abs_std_m": ea_sd,
            "RMSE_m": rmse_mu,
            "err_pct (%)": pct_mu,
            "detect_ratio": dr_mu,
        })

    return per_trial_df, pd.DataFrame(summary_rows).sort_values("gt_cm")


# ── Console printer ───────────────────────────────────────────────────────────
def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:.{nd}f}"


def print_summary_table(summary_df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("Exp.3  Range Estimation Accuracy  —  Summary Table")
    print("=" * 70)
    header = f"{'GT[cm]':>7} {'n':>3} {'z_ema[m]':>10} {'bias[m]':>9} {'|err|[m]':>9} {'RMSE[m]':>9} {'err%':>7} {'det_r':>6}"
    print(header)
    print("-" * 70)
    for _, r in summary_df.iterrows():
        z_str = f"{_fmt(r['z_ema_mean_m'],3)}±{_fmt(r['z_ema_std_m'],3)}"
        e_str = _fmt(r["err_mean_m (bias)"], 3)
        ea_str = f"{_fmt(r['err_abs_mean_m'],3)}±{_fmt(r['err_abs_std_m'],3)}"
        rmse_str = _fmt(r["RMSE_m"], 3)
        pct_str = _fmt(r["err_pct (%)"], 1)
        dr_str = _fmt(r["detect_ratio"], 2)
        print(f"{int(r['gt_cm']):>7} {int(r['n_trials']):>3}  {z_str:>12}  {e_str:>8}  {ea_str:>14}  {rmse_str:>8}  {pct_str:>6}  {dr_str:>6}")
    print("=" * 70)
    print("  bias = z_ema_mean - gt  (+ means overestimated)")
    print("  err% = (|err| / gt) * 100")


# ── Record mode: append one GT entry ─────────────────────────────────────────
def record_mode(log_path: str, gt_cm: float, gt_table_path: str = "exp3_gt.csv"):
    """Quick helper: append a log+gt_cm pair to the GT table CSV."""
    entry = pd.DataFrame([{"log_path": log_path, "gt_cm": gt_cm}])
    if os.path.exists(gt_table_path):
        existing = pd.read_csv(gt_table_path)
        entry = pd.concat([existing, entry], ignore_index=True)
    entry.to_csv(gt_table_path, index=False)
    print(f"[RECORD] Appended: {log_path} / gt={gt_cm}cm  →  {gt_table_path}")
    # Also run quick analysis immediately
    try:
        t = analyze_trial(log_path, gt_cm)
        print(f"         z_ema_mean={_fmt(t.z_ema_mean_m,3)}m  |err|={_fmt(t.err_abs_mean_m,3)}m  ({_fmt(t.err_pct,1)}%)")
    except Exception as e:
        print(f"[WARN] quick analysis failed: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Exp.3 Range Estimation Accuracy Evaluator")
    parser.add_argument("--mode", choices=["record", "analyze"], default="analyze",
                        help="record: add one GT entry; analyze: process gt_table")
    parser.add_argument("--log", type=str, default="",
                        help="[record mode] path to log file (.csv/.xlsx)")
    parser.add_argument("--gt_cm", type=float, default=None,
                        help="[record mode] ground-truth distance in cm (e.g. 40)")
    parser.add_argument("--gt_table", type=str, default="exp3_gt.csv",
                        help="[analyze mode] CSV with columns: log_path, gt_cm")
    parser.add_argument("--out", type=str, default="exp3_results",
                        help="[analyze mode] output directory")
    args = parser.parse_args()

    if args.mode == "record":
        if not args.log or args.gt_cm is None:
            parser.error("--log and --gt_cm are required in record mode")
        record_mode(args.log, args.gt_cm, args.gt_table)
        return

    # ── analyze mode ──
    if not os.path.exists(args.gt_table):
        print(f"[ERR] GT table not found: {args.gt_table}")
        print("  Create it by running:")
        print("    python exp3_range_eval.py --mode record --log <log.xlsx> --gt_cm <cm>")
        sys.exit(1)

    gt_df = pd.read_csv(args.gt_table)
    if not {"log_path", "gt_cm"}.issubset(gt_df.columns):
        print("[ERR] GT table must have columns: log_path, gt_cm")
        sys.exit(1)

    trials: List[TrialResult] = []
    for _, row in gt_df.iterrows():
        log_path = str(row["log_path"])
        gt_cm = float(row["gt_cm"])
        if not os.path.exists(log_path):
            print(f"[WARN] log not found, skipped: {log_path}")
            continue
        try:
            t = analyze_trial(log_path, gt_cm)
            trials.append(t)
            print(f"[OK] {os.path.basename(log_path)}  gt={gt_cm}cm  "
                  f"|err|={_fmt(t.err_abs_mean_m,3)}m  ({_fmt(t.err_pct,1)}%)")
        except Exception as e:
            print(f"[ERR] {log_path}: {e}")

    if not trials:
        print("[ERR] No valid trials found.")
        sys.exit(1)

    per_trial_df, summary_df = summarize(trials)
    print_summary_table(summary_df)

    # Save outputs
    os.makedirs(args.out, exist_ok=True)
    per_trial_path = os.path.join(args.out, "exp3_per_trial.csv")
    summary_csv_path = os.path.join(args.out, "exp3_summary.csv")
    summary_xlsx_path = os.path.join(args.out, "exp3_summary.xlsx")

    per_trial_df.to_csv(per_trial_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)

    try:
        with pd.ExcelWriter(summary_xlsx_path, engine="openpyxl") as w:
            summary_df.to_excel(w, sheet_name="summary", index=False)
            per_trial_df.to_excel(w, sheet_name="per_trial", index=False)
        print(f"\n[SAVED] {summary_xlsx_path}")
    except ImportError:
        print(f"[INFO] openpyxl not found; Excel output skipped.")

    print(f"[SAVED] {per_trial_path}")
    print(f"[SAVED] {summary_csv_path}")


if __name__ == "__main__":
    main()
