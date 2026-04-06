"""
exp6_endtoend.py  —  Exp.6: End-to-End Approach Task
======================================================
実機で「離陸 → 検出 → 自律接近 → 目標距離30cmで安定停止」を繰り返し、
1試行ごとに成否・収束時間・z_errを記録する。

Usage
-----
  # 1試行実行（自動的にログ保存・成否判定）
  python exp6_endtoend.py --mode run --trial 1

  # 全試行の集計（n=10以上が揃ってから）
  python exp6_endtoend.py --mode analyze --log_dir logs/ --out exp6_results/

  # 既存ログへの手動成否追記（自動判定に疑問がある場合）
  python exp6_endtoend.py --mode annotate --result_csv exp6_results/exp6_trials.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import keyboard
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.ctrl import CTRL, Tools
from utils.model import Model
from utils.pinhole import Intrinsics, PinholeCamera, Pose
from utils.tracking_continuity import ContinuityController
from utils.eval_logger import create_logger, EvalRow, bbox_center, bbox_in_frame, bbox_area_ratio
from utils.eval_metrics import compute_metrics_from_file, _fmt

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
WEIGHT_PATH        = "./weights/ssd_finetuned_filterv2.pth"
TARGET_LABEL       = "apple"
APPLE_SIZE_M       = 0.08
USE_BBOX_AXIS      = "height"
FOV_H_DEG          = 82.0
TARGET_DISTANCE_M  = 0.30
DIST_TOL_M         = 0.05       # ±5cm で「成功」判定
SUCCESS_HOLD_S     = 1.5        # この時間 DIST_TOL_M 以内を維持で「成功」
TRIAL_TIMEOUT_S    = 45.0       # 試行タイムアウト
MIN_DISTANCE_M     = 0.20
MAX_DISTANCE_M     = 5.0
MAX_VX             = 38
KP_FORWARD         = 150.0
EMA_ALPHA_Z        = 0.35
CENTERING_MODE     = "yaw"
KP_CENTER_X        = 0.15
MAX_YAW            = 40
CONF_THRESH        = 0.5
INFER_HZ           = 4.0
HOLD_SEC           = 0.5
LOG_DIR            = "./logs"
QUIT_KEY           = "q"
WINDOW_NAME        = "Exp.6 End-to-end"


def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def clamp(v: float, lo: float, hi: float) -> int:
    return int(max(lo, min(hi, v)))


# ─────────────────────────────────────────────
# Trial result record
# ─────────────────────────────────────────────
@dataclass
class TrialRecord:
    trial_idx: int
    timestamp: str
    log_path: str
    success: int                     # 1=success, 0=fail
    success_auto: int                # auto-detected success (before manual override)
    settle_time_s: Optional[float]   # time from first detection to first stable arrival
    first_detect_s: Optional[float]  # time from session start to first detection
    z_err_mean_m: Optional[float]
    z_err_p95_abs_m: Optional[float]
    center_err_mean_px: Optional[float]
    detect_ratio: Optional[float]
    lost_ratio: Optional[float]
    n_rows: int
    notes: str = ""

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


TRIAL_CSV_HEADER = [f.name for f in fields(TrialRecord)]


# ─────────────────────────────────────────────
# Success auto-detection from log
# ─────────────────────────────────────────────
def detect_success(log_path: str) -> tuple[int, Optional[float], Optional[float]]:
    """
    Returns (success: 0/1, settle_time_s, first_detect_s) by re-reading the log.

    Success criterion:
        |z_ema - TARGET_DISTANCE_M| <= DIST_TOL_M  for >= SUCCESS_HOLD_S seconds
    """
    try:
        ext = os.path.splitext(log_path)[1].lower()
        df = pd.read_excel(log_path) if ext == ".xlsx" else pd.read_csv(log_path)
    except Exception as e:
        print(f"[WARN] detect_success read error: {e}")
        return 0, None, None

    if len(df) < 2:
        return 0, None, None

    t_col = pd.to_numeric(df["t"], errors="coerce") if "t" in df.columns else None
    z_ema = pd.to_numeric(df["z_ema_m"], errors="coerce") if "z_ema_m" in df.columns else None
    mode_col = df["mode"].astype(str) if "mode" in df.columns else None

    if t_col is None or z_ema is None:
        return 0, None, None

    t0 = float(t_col.dropna().iloc[0]) if len(t_col.dropna()) > 0 else 0.0

    # First detection time
    first_detect_s: Optional[float] = None
    if mode_col is not None:
        det_idx = (mode_col == "DETECT")
        if det_idx.any():
            first_det_t = float(t_col[det_idx].dropna().iloc[0])
            first_detect_s = first_det_t - t0

    # Find first run where |z_ema - target| <= DIST_TOL_M for >= SUCCESS_HOLD_S
    within = ((z_ema - TARGET_DISTANCE_M).abs() <= DIST_TOL_M)

    settle_time_s: Optional[float] = None
    success = 0

    cur_start_idx = None
    cur_start_t = None
    vals_t = t_col.to_numpy()
    vals_w = within.fillna(False).to_numpy()

    for i, v in enumerate(vals_w):
        if math.isnan(vals_t[i]):
            continue
        if v:
            if cur_start_idx is None:
                cur_start_idx = i
                cur_start_t = vals_t[i]
            else:
                duration = vals_t[i] - cur_start_t
                if duration >= SUCCESS_HOLD_S:
                    success = 1
                    settle_time_s = cur_start_t - t0
                    break
        else:
            cur_start_idx = None
            cur_start_t = None

    return success, settle_time_s, first_detect_s


# ─────────────────────────────────────────────
# Single trial runner
# ─────────────────────────────────────────────
def run_trial(trial_idx: int, ctrl: CTRL, model: Model) -> TrialRecord:
    """
    Run one end-to-end approach trial. Returns a TrialRecord.
    """
    log_path = os.path.join(LOG_DIR, f"exp6_trial{trial_idx:02d}_{now_str()}.xlsx")
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = create_logger(log_path)

    state: Dict[str, object] = {
        "vx": 0, "vy": 0, "vz": 0, "yaw": 0,
        "running": True, "autopilot": True,
        "z_ema": None, "infer_dt": 0.0, "manual_active": False,
    }
    state_lock = threading.Lock()

    tello = ctrl.tello
    fr = tello.get_frame_read()
    categories = list(model.voc_classes)
    target_idx = categories.index(TARGET_LABEL) if TARGET_LABEL in categories else None

    cont = ContinuityController(infer_hz=INFER_HZ, hold_sec=HOLD_SEC, ema_alpha=0.5)

    det_lock = threading.Lock()
    det_shared = {"boxes": [], "labels": [], "scores": [], "infer_dt": 0.0}

    def infer_worker():
        period = 1.0 / max(0.1, INFER_HZ)
        while True:
            with state_lock:
                if not state["running"]:
                    break
            frame_bgr = fr.frame
            if frame_bgr is None:
                time.sleep(0.01)
                continue
            frame_in = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            t0 = time.time()
            try:
                _, boxes, labels, scores = model.predict_frame(frame_in, conf=CONF_THRESH)
            except Exception:
                boxes, labels, scores = [], [], []
            dt = time.time() - t0
            with det_lock:
                det_shared.update({"boxes": boxes, "labels": labels, "scores": scores, "infer_dt": dt})
            with state_lock:
                state["infer_dt"] = dt
            time.sleep(max(0.0, period - (time.time() - t0)))

    def rc_sender():
        while True:
            with state_lock:
                if not state["running"]:
                    break
                vx, vy, vz, yaw = state["vx"], state["vy"], state["vz"], state["yaw"]
            try:
                tello.send_rc_control(clamp(vy,-100,100), clamp(vx,-100,100),
                                      clamp(vz,-100,100), clamp(yaw,-100,100))
            except Exception:
                pass
            time.sleep(0.05)

    threading.Thread(target=infer_worker, daemon=True).start()
    threading.Thread(target=rc_sender, daemon=True).start()

    cam: Optional[PinholeCamera] = None
    trial_start = time.time()
    frame_idx = 0
    success_announced = False

    print(f"\n[TRIAL {trial_idx}] Starting. timeout={TRIAL_TIMEOUT_S}s  target={TARGET_DISTANCE_M}m±{DIST_TOL_M}m")
    print(f"  [{QUIT_KEY.upper()}] to end trial manually")

    try:
        while True:
            elapsed = time.time() - trial_start
            if elapsed >= TRIAL_TIMEOUT_S:
                print(f"[TRIAL {trial_idx}] Timeout ({TRIAL_TIMEOUT_S}s).")
                break

            with state_lock:
                if not state["running"]:
                    break

            frame_bgr = fr.frame
            if frame_bgr is None:
                if (cv2.waitKey(1) & 0xFF) == ord(QUIT_KEY):
                    break
                continue

            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            out = frame.copy()

            if cam is None:
                h0, w0 = out.shape[:2]
                intr = Intrinsics.from_fov(width=w0, height=h0, fov_deg_h=FOV_H_DEG)
                cam = PinholeCamera(intr=intr, pose_w2c=Pose.identity())

            with det_lock:
                boxes   = det_shared["boxes"]
                labels  = det_shared["labels"]
                scores  = det_shared["scores"]
                infer_dt = float(det_shared["infer_dt"])

            best_bb = None
            best_sc = 0.0
            z_est: Optional[float] = None

            for i, bb in enumerate(boxes):
                li = int(labels[i]) if i < len(labels) else -1
                sc = float(scores[i]) if i < len(scores) else 0.0
                x1, y1, x2, y2 = [int(v) for v in bb]
                color = (0, 255, 255) if (target_idx is not None and li == target_idx) else (0, 255, 0)
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                if target_idx is not None and li == target_idx and sc >= CONF_THRESH:
                    area = (x2 - x1) * (y2 - y1)
                    if best_bb is None or area > (best_bb[2]-best_bb[0])*(best_bb[3]-best_bb[1]):
                        best_bb = (x1, y1, x2, y2)
                        best_sc = sc

            if best_bb is not None:
                cont.update_with_detection(best_bb, TARGET_LABEL, best_sc)
            else:
                if HOLD_SEC > 0 and cont.state.age_det() <= HOLD_SEC:
                    cont.state.mode = "HOLD"
                else:
                    cont.state.bbox = None
                    cont.state.cx_s = None
                    cont.state.mode = "LOST"

            st = cont.state

            if cam is not None and st.bbox is not None and st.mode in ("DETECT", "HOLD"):
                x1, y1, x2, y2 = [int(v) for v in st.bbox]
                w_px = max(1, x2 - x1)
                h_px = max(1, y2 - y1)
                px_len = h_px if USE_BBOX_AXIS == "height" else w_px
                f_pix  = cam.intr.fy if USE_BBOX_AXIS == "height" else cam.intr.fx
                Z = (float(f_pix) * float(APPLE_SIZE_M)) / float(max(1, px_len))
                if MIN_DISTANCE_M <= Z <= MAX_DISTANCE_M:
                    z_est = float(Z)

            with state_lock:
                if z_est is not None:
                    if state["z_ema"] is None:
                        state["z_ema"] = z_est
                    else:
                        state["z_ema"] = EMA_ALPHA_Z * z_est + (1.0 - EMA_ALPHA_Z) * float(state["z_ema"])
                z_ema = state["z_ema"]

            # Autopilot
            if z_est is not None and st.bbox is not None:
                x1, y1, x2, y2 = [int(v) for v in st.bbox]
                cx = 0.5 * (x1 + x2)
                h, w = out.shape[:2]
                ex = (cx - (w * 0.5)) / max(1.0, w * 0.5)
                z_use = float(z_ema) if z_ema is not None else float(z_est)
                e = z_use - TARGET_DISTANCE_M
                vx_cmd = 0 if abs(e) <= DIST_TOL_M else int(np.clip(KP_FORWARD * e * 10.0, -MAX_VX, MAX_VX))
                if z_use <= MIN_DISTANCE_M:
                    vx_cmd = min(0, vx_cmd)
                yaw_cmd = int(np.clip(KP_CENTER_X * ex * 100.0, -MAX_YAW, MAX_YAW))
                with state_lock:
                    state["vx"] = vx_cmd
                    state["yaw"] = yaw_cmd

                # Real-time success notification
                if z_ema is not None and abs(float(z_ema) - TARGET_DISTANCE_M) <= DIST_TOL_M:
                    if not success_announced:
                        print(f"[TRIAL {trial_idx}] IN RANGE! z={float(z_ema):.3f}m at t={elapsed:.1f}s")
                        success_announced = True
            else:
                with state_lock:
                    state["vx"] = state["vy"] = state["yaw"] = 0

            # HUD
            with state_lock:
                vx = int(state["vx"]); vy = int(state["vy"])
                vz = int(state["vz"]); yaw = int(state["yaw"])

            z_txt = f"{float(z_ema):.2f}m" if z_ema is not None else "- "
            in_r = z_ema is not None and abs(float(z_ema) - TARGET_DISTANCE_M) <= DIST_TOL_M
            hud_color = (0, 255, 80) if in_r else (255, 255, 255)
            cv2.putText(out, f"Trial {trial_idx}  {elapsed:.1f}s / {TRIAL_TIMEOUT_S}s",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
            cv2.putText(out, f"Z:{z_txt}  Target:{TARGET_DISTANCE_M}m  Mode:{st.mode}",
                        (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_color, 1)
            if in_r:
                cv2.putText(out, "  IN RANGE", (10, 66),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,80), 2)

            # Log
            bb = tuple(map(float, st.bbox)) if st.bbox is not None else None
            cx_v, cy_v = bbox_center(bb)
            in_fr  = bbox_in_frame(bb, out.shape[1], out.shape[0])
            area_r = bbox_area_ratio(bb, out.shape[1], out.shape[0])
            z_err  = (float(z_ema) - TARGET_DISTANCE_M) if z_ema is not None else None

            row = EvalRow(
                t=time.time(), frame_idx=frame_idx,
                mode=str(st.mode), label=str(st.label),
                conf=float(st.conf) if st.conf else None,
                age_det_s=float(st.age_det()), infer_dt_s=float(infer_dt),
                cx=float(cx_v) if cx_v is not None else None,
                cy=float(cy_v) if cy_v is not None else None,
                x1=float(bb[0]) if bb else None, y1=float(bb[1]) if bb else None,
                x2=float(bb[2]) if bb else None, y2=float(bb[3]) if bb else None,
                frame_w=out.shape[1], frame_h=out.shape[0],
                in_frame=int(in_fr) if in_fr is not None else None,
                area_ratio=float(area_r) if area_r is not None else None,
                z_est_m=float(z_est) if z_est is not None else None,
                z_ema_m=float(z_ema) if z_ema is not None else None,
                z_err_m=float(z_err) if z_err is not None else None,
                target_z_m=float(TARGET_DISTANCE_M),
                rc_vx=vx, rc_vy=vy, rc_vz=vz, rc_yaw=yaw,
                autopilot=1, manual_active=0, track_mode=0,
            )
            logger.write(row)

            out_show = cv2.resize(out, None, fx=0.9, fy=0.9, interpolation=cv2.INTER_AREA)
            cv2.imshow(WINDOW_NAME, out_show)
            if (cv2.waitKey(1) & 0xFF) == ord(QUIT_KEY):
                print(f"[TRIAL {trial_idx}] Manual end.")
                break

            frame_idx += 1

    finally:
        logger.close()
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except Exception:
            pass
        with state_lock:
            state["running"] = False

    # Post-trial metrics
    success_auto, settle_time_s, first_detect_s = detect_success(log_path)
    try:
        m = compute_metrics_from_file(log_path)
        z_err_mean = m.z_err_mean_m
        z_err_p95  = m.z_err_p95_abs_m
        ce_mean    = m.center_err_mean_px
        det_ratio  = m.detect_ratio
        lost_ratio = m.lost_ratio
        n_rows     = m.n_rows
    except Exception:
        z_err_mean = z_err_p95 = ce_mean = det_ratio = lost_ratio = None
        n_rows = frame_idx

    result_str = "SUCCESS" if success_auto else "FAIL"
    print(f"\n[TRIAL {trial_idx}] Auto-result: {result_str}")
    print(f"  settle_time={_fmt(settle_time_s, 2)}s  z_err_p95={_fmt(z_err_p95, 3)}m  "
          f"center_err={_fmt(ce_mean, 1)}px  detect_ratio={_fmt(det_ratio, 3)}")

    # Manual override prompt
    user_ans = input(f"  Confirm result? [{result_str}]  Enter=keep / 0=fail / 1=success  notes: ").strip()
    notes = ""
    manual_success = success_auto
    if user_ans == "0":
        manual_success = 0
    elif user_ans == "1":
        manual_success = 1
    elif user_ans and user_ans not in ("0", "1"):
        notes = user_ans

    return TrialRecord(
        trial_idx=trial_idx,
        timestamp=now_str(),
        log_path=log_path,
        success=manual_success,
        success_auto=success_auto,
        settle_time_s=settle_time_s,
        first_detect_s=first_detect_s,
        z_err_mean_m=z_err_mean,
        z_err_p95_abs_m=z_err_p95,
        center_err_mean_px=ce_mean,
        detect_ratio=det_ratio,
        lost_ratio=lost_ratio,
        n_rows=n_rows if n_rows else frame_idx,
        notes=notes,
    )


# ─────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────
def analyze_trials(log_dir: str, out_dir: str):
    """Aggregate exp6 logs and produce paper-ready table."""
    pattern_xlsx = os.path.join(log_dir, "exp6_trial*.xlsx")
    pattern_csv  = os.path.join(log_dir, "exp6_trial*.csv")
    files = sorted(glob.glob(pattern_xlsx)) + sorted(glob.glob(pattern_csv))

    # Also check for existing trial result CSV
    trial_csv = os.path.join(out_dir, "exp6_trials.csv")
    if os.path.exists(trial_csv):
        df = pd.read_csv(trial_csv)
        print(f"[INFO] Loaded existing trial results: {trial_csv} ({len(df)} trials)")
    else:
        # Build from log files
        os.makedirs(out_dir, exist_ok=True)
        if not files:
            print(f"[ERR] No exp6 log files found in {log_dir}")
            sys.exit(1)
        records = []
        for f in files:
            m = re.search(r"exp6_trial(\d+)", os.path.basename(f))
            t_idx = int(m.group(1)) if m else -1
            try:
                success_auto, settle_s, first_det_s = detect_success(f)
                metrics = compute_metrics_from_file(f)
                records.append(TrialRecord(
                    trial_idx=t_idx,
                    timestamp="",
                    log_path=f,
                    success=success_auto,
                    success_auto=success_auto,
                    settle_time_s=settle_s,
                    first_detect_s=first_det_s,
                    z_err_mean_m=metrics.z_err_mean_m,
                    z_err_p95_abs_m=metrics.z_err_p95_abs_m,
                    center_err_mean_px=metrics.center_err_mean_px,
                    detect_ratio=metrics.detect_ratio,
                    lost_ratio=metrics.lost_ratio,
                    n_rows=metrics.n_rows,
                ).as_dict())
                print(f"[OK] trial={t_idx}  success={success_auto}  settle={_fmt(settle_s,2)}s")
            except Exception as e:
                print(f"[ERR] {f}: {e}")

        df = pd.DataFrame(records, columns=TRIAL_CSV_HEADER)
        df.to_csv(trial_csv, index=False)
        print(f"[SAVED] {trial_csv}")

    # Summary statistics
    n = len(df)
    n_success = int(df["success"].sum())
    success_rate = n_success / n if n > 0 else 0.0

    def ms(col):
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) == 0:
            return None, None
        return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) >= 2 else 0.0

    st_mu, st_sd = ms("settle_time_s")
    ze_mu, ze_sd = ms("z_err_mean_m")
    ze95_mu, _   = ms("z_err_p95_abs_m")
    ce_mu, ce_sd = ms("center_err_mean_px")
    dr_mu, _     = ms("detect_ratio")

    print("\n" + "=" * 60)
    print("Exp.6  End-to-End Approach Task  —  Summary")
    print("=" * 60)
    print(f"  Total trials       : {n}")
    print(f"  Success count      : {n_success}")
    print(f"  Success rate       : {success_rate*100:.1f}%  ({n_success}/{n})")
    print(f"  Settle time [s]    : {_fmt(st_mu,2)} ± {_fmt(st_sd,2)}")
    print(f"  z_err mean [m]     : {_fmt(ze_mu,3)} ± {_fmt(ze_sd,3)}")
    print(f"  z_err p95  [m]     : {_fmt(ze95_mu,3)}")
    print(f"  Center err [px]    : {_fmt(ce_mu,1)} ± {_fmt(ce_sd,1)}")
    print(f"  Detect ratio       : {_fmt(dr_mu,3)}")
    print("=" * 60)
    if n < 10:
        print(f"  [WARN] n={n} < 10. International conference typically requires n>=10.")

    # Per-trial table
    print("\n  Per-trial breakdown:")
    print(f"  {'Trial':>5}  {'Success':>7}  {'Settle[s]':>9}  {'z_err_p95[m]':>12}  {'ctr_err[px]':>11}")
    for _, r in df.sort_values("trial_idx").iterrows():
        s = "OK" if int(r["success"]) == 1 else "FAIL"
        print(f"  {int(r['trial_idx']):>5}  {s:>7}  "
              f"{_fmt(r['settle_time_s'],2):>9}  "
              f"{_fmt(r['z_err_p95_abs_m'],3):>12}  "
              f"{_fmt(r['center_err_mean_px'],1):>11}")

    # Save summary
    os.makedirs(out_dir, exist_ok=True)
    summary = {
        "n_trials": n, "n_success": n_success, "success_rate": success_rate,
        "settle_time_mean_s": st_mu, "settle_time_std_s": st_sd,
        "z_err_mean_m": ze_mu, "z_err_std_m": ze_sd, "z_err_p95_abs_m": ze95_mu,
        "center_err_mean_px": ce_mu, "center_err_std_px": ce_sd,
        "detect_ratio": dr_mu,
    }
    summary_path = os.path.join(out_dir, "exp6_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"\n[SAVED] {summary_path}")

    try:
        xlsx_path = os.path.join(out_dir, "exp6_summary.xlsx")
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
            pd.DataFrame([summary]).to_excel(w, sheet_name="summary", index=False)
            df.to_excel(w, sheet_name="per_trial", index=False)
        print(f"[SAVED] {xlsx_path}")
    except ImportError:
        pass


# ─────────────────────────────────────────────
# Annotate mode: manually review/edit results
# ─────────────────────────────────────────────
def annotate_mode(trial_csv: str):
    """Interactive CLI to review and override success flags."""
    if not os.path.exists(trial_csv):
        print(f"[ERR] File not found: {trial_csv}")
        sys.exit(1)
    df = pd.read_csv(trial_csv)
    print(f"\n  Loaded {len(df)} trials from {trial_csv}")
    print("  For each trial: Enter=keep / 0=fail / 1=success / q=quit\n")
    changed = False
    for i, row in df.iterrows():
        cur = int(row["success"])
        cur_str = "OK" if cur else "FAIL"
        ans = input(f"  Trial {int(row['trial_idx']):02d}  [{cur_str}]  settle={_fmt(row['settle_time_s'],2)}s  > ").strip()
        if ans == "q":
            break
        if ans == "0":
            df.at[i, "success"] = 0
            changed = True
        elif ans == "1":
            df.at[i, "success"] = 1
            changed = True
    if changed:
        df.to_csv(trial_csv, index=False)
        print(f"[SAVED] {trial_csv}")
    else:
        print("  No changes.")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Exp.6 End-to-End Approach Task")
    parser.add_argument("--mode", choices=["run", "analyze", "annotate"], default="analyze")
    parser.add_argument("--trial", type=int, default=1, help="[run] trial index")
    parser.add_argument("--log_dir", type=str, default=LOG_DIR, help="[analyze] log directory")
    parser.add_argument("--out", type=str, default="exp6_results", help="[analyze] output directory")
    parser.add_argument("--result_csv", type=str, default="exp6_results/exp6_trials.csv",
                        help="[annotate] path to trial result CSV")
    args = parser.parse_args()

    if args.mode == "analyze":
        analyze_trials(args.log_dir, args.out)
        return

    if args.mode == "annotate":
        annotate_mode(args.result_csv)
        return

    # ── run mode ──
    print("\n" + "=" * 60)
    print(f"  Exp.6 End-to-End Approach Task")
    print(f"  Trial     : {args.trial}")
    print(f"  Target    : {TARGET_DISTANCE_M}m ± {DIST_TOL_M}m")
    print(f"  Timeout   : {TRIAL_TIMEOUT_S}s")
    print(f"  Success   : z within ±{DIST_TOL_M}m for {SUCCESS_HOLD_S}s")
    print("=" * 60)
    input("  Press Enter when drone is ready...")

    ctrl = CTRL()
    model = Model(weight_path=WEIGHT_PATH)
    ctrl.set_speed(40)

    os.makedirs(args.out, exist_ok=True)
    trial_csv = os.path.join(args.out, "exp6_trials.csv")

    print("[INFO] T=takeoff  G=land  ESC=abort  Q=end trial")
    try:
        while not keyboard.is_pressed("t"):
            if keyboard.is_pressed("esc"):
                print("[ABORT]")
                ctrl.cleanup()
                return
            time.sleep(0.05)
        ctrl.takeoff()
        time.sleep(1.5)

        record = run_trial(args.trial, ctrl, model)
        ctrl.land()
        time.sleep(1.0)

        # Append to trial CSV
        row_dict = record.as_dict()
        if os.path.exists(trial_csv):
            existing = pd.read_csv(trial_csv)
            new_df = pd.concat([existing, pd.DataFrame([row_dict])], ignore_index=True)
        else:
            new_df = pd.DataFrame([row_dict], columns=TRIAL_CSV_HEADER)
        new_df.to_csv(trial_csv, index=False)
        print(f"[SAVED] {trial_csv}  (total: {len(new_df)} trials)")
        print(f"\n  Next trial: python exp6_endtoend.py --mode run --trial {args.trial + 1}")

    finally:
        ctrl.cleanup()


if __name__ == "__main__":
    main()
