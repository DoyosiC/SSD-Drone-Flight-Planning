"""
exp5_ablation_main.py  —  Exp.5: Ablation Study
=================================================
main_tracking.py を条件切り替えして実行するラッパー。
実験条件を定義し、1条件ずつ実機実験を案内しながらログを管理する。

Ablation conditions
-------------------
  FULL   : 提案手法フル（baseline）
  NO_HOLD: hold_sec = 0  （HOLDなし）
  NO_EMA : EMA_ALPHA_Z = 1.0  （EMAなし、生推定値をそのまま使用）
  HZ_2   : infer_hz = 2.0
  HZ_8   : infer_hz = 8.0
  HZ_15  : infer_hz = 15.0

Usage
-----
  # 1) インタラクティブ実験モード（1条件ずつ案内）
  python exp5_ablation_main.py --mode run --condition FULL

  # 2) 結果集計（全条件のログが揃ってから）
  python exp5_ablation_main.py --mode analyze --log_dir logs/ --out exp5_results/
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import keyboard
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.ctrl import CTRL, Tools
from utils.model import Model
from utils.pinhole import Intrinsics, PinholeCamera, Pose
from utils.tracking_continuity import ContinuityController
from utils.eval_logger import create_logger, EvalRow, bbox_center, bbox_in_frame, bbox_area_ratio
from utils.eval_metrics import compute_metrics_from_file, _fmt


# ─────────────────────────────────────────────
# Ablation condition definitions
# ─────────────────────────────────────────────
ABLATION_CONDITIONS: Dict[str, dict] = {
    "FULL": {
        "label": "Full system (proposed)",
        "hold_sec": 0.5,
        "ema_alpha_z": 0.35,
        "infer_hz": 4.0,
    },
    "NO_HOLD": {
        "label": "w/o continuity (HOLD disabled)",
        "hold_sec": 0.0,
        "ema_alpha_z": 0.35,
        "infer_hz": 4.0,
    },
    "NO_EMA": {
        "label": "w/o EMA filter (raw z_est)",
        "hold_sec": 0.5,
        "ema_alpha_z": 1.0,      # alpha=1 → z_ema = z_est (no smoothing)
        "infer_hz": 4.0,
    },
    "HZ_2": {
        "label": "infer_hz = 2.0 Hz",
        "hold_sec": 0.5,
        "ema_alpha_z": 0.35,
        "infer_hz": 2.0,
    },
    "HZ_8": {
        "label": "infer_hz = 8.0 Hz",
        "hold_sec": 0.5,
        "ema_alpha_z": 0.35,
        "infer_hz": 8.0,
    },
    "HZ_15": {
        "label": "infer_hz = 15.0 Hz",
        "hold_sec": 0.5,
        "ema_alpha_z": 0.35,
        "infer_hz": 15.0,
    },
}

# ─────────────────────────────────────────────
# Shared config (matches main_tracking.py)
# ─────────────────────────────────────────────
WEIGHT_PATH       = "./weights/ssd_finetuned_filterv2.pth"
TARGET_LABEL      = "apple"
APPLE_SIZE_M      = 0.08
USE_BBOX_AXIS     = "height"
FOV_H_DEG         = 82.0
TARGET_DISTANCE_M = 0.30
DIST_TOL_M        = 0.08
MIN_DISTANCE_M    = 0.20
MAX_DISTANCE_M    = 5.0
MAX_VX            = 38
KP_FORWARD        = 150.0
CENTERING_MODE    = "yaw"
KP_CENTER_X       = 0.15
MAX_YAW           = 40
CONF_THRESH       = 0.5
LOG_DIR           = "./logs"
QUIT_KEY          = "q"
WINDOW_NAME_TMPL  = "Ablation: {cond}"
SESSION_DURATION_S = 30.0    # 1セッションの目標時間 [s]


def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def clamp(v: float, lo: float, hi: float) -> int:
    return int(max(lo, min(hi, v)))


# ─────────────────────────────────────────────
# Single session runner
# ─────────────────────────────────────────────
def run_session(condition_name: str, trial_idx: int, ctrl: CTRL, model: Model) -> Optional[str]:
    """
    Run one 30-second tracking session under the given ablation condition.
    Returns the log file path on completion, None on error.
    """
    cond = ABLATION_CONDITIONS[condition_name]
    hold_sec    = float(cond["hold_sec"])
    ema_alpha_z = float(cond["ema_alpha_z"])
    infer_hz    = float(cond["infer_hz"])

    log_path = os.path.join(
        LOG_DIR,
        f"ablation_{condition_name}_trial{trial_idx:02d}_{now_str()}.xlsx"
    )
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = create_logger(log_path)

    state: Dict[str, object] = {
        "vx": 0, "vy": 0, "vz": 0, "yaw": 0,
        "running": True,
        "autopilot": True,
        "z_ema": None,
        "infer_dt": 0.0,
        "manual_active": False,
    }
    state_lock = threading.Lock()

    tello = ctrl.tello
    fr = tello.get_frame_read()
    categories = list(model.voc_classes)
    target_idx = categories.index(TARGET_LABEL) if TARGET_LABEL in categories else None

    cont = ContinuityController(
        infer_hz=infer_hz,
        hold_sec=hold_sec,
        ema_alpha=0.5,
    )

    det_lock = threading.Lock()
    det_shared = {"boxes": [], "labels": [], "scores": [], "infer_dt": 0.0}

    def infer_worker():
        period = 1.0 / max(0.1, infer_hz)
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
    session_start = time.time()
    frame_idx = 0
    window_name = WINDOW_NAME_TMPL.format(cond=condition_name)

    print(f"\n[SESSION] {condition_name} trial={trial_idx}  hold={hold_sec}s  "
          f"alpha={ema_alpha_z}  hz={infer_hz}")
    print(f"  → duration: {SESSION_DURATION_S}s  |  press [{QUIT_KEY.upper()}] to end early")

    try:
        while True:
            elapsed = time.time() - session_start
            if elapsed >= SESSION_DURATION_S:
                print(f"[SESSION] {SESSION_DURATION_S}s elapsed. Session complete.")
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
                if hold_sec > 0:
                    if cont.state.age_det() <= hold_sec:
                        cont.state.mode = "HOLD"
                    else:
                        cont.state.bbox = None
                        cont.state.cx_s = None
                        cont.state.cy_s = None
                        cont.state.mode = "LOST"
                else:
                    cont.update_no_detection()

            st = cont.state

            if cam is not None and st.bbox is not None and st.mode in ("DETECT", "HOLD", "PREDICT"):
                x1, y1, x2, y2 = [int(v) for v in st.bbox]
                w_px = max(1, x2 - x1)
                h_px = max(1, y2 - y1)
                if USE_BBOX_AXIS == "height":
                    px_len, f_pix = h_px, cam.intr.fy
                elif USE_BBOX_AXIS == "width":
                    px_len, f_pix = w_px, cam.intr.fx
                else:
                    px_len, f_pix = (h_px, cam.intr.fy) if h_px >= w_px else (w_px, cam.intr.fx)
                Z = (float(f_pix) * float(APPLE_SIZE_M)) / float(max(1, px_len))
                if MIN_DISTANCE_M <= Z <= MAX_DISTANCE_M:
                    z_est = float(Z)

            with state_lock:
                if z_est is not None:
                    if state["z_ema"] is None:
                        state["z_ema"] = z_est
                    else:
                        # Use ablation-specific alpha
                        state["z_ema"] = ema_alpha_z * z_est + (1.0 - ema_alpha_z) * float(state["z_ema"])
                z_ema = state["z_ema"]

            # Autopilot
            with state_lock:
                autopilot = bool(state["autopilot"])

            if autopilot and z_est is not None and st.bbox is not None:
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
            elif autopilot and (st.bbox is None or z_est is None):
                with state_lock:
                    state["vx"] = state["vy"] = state["yaw"] = 0

            # HUD
            with state_lock:
                vx = int(state["vx"]); vy = int(state["vy"])
                vz = int(state["vz"]); yaw = int(state["yaw"])
                ap_flag = 1 if autopilot else 0
                manual_flag = 0

            remain = max(0.0, SESSION_DURATION_S - elapsed)
            z_txt = f"{float(z_ema):.2f}m" if z_ema is not None else "-"
            cv2.putText(out, f"COND:{condition_name}  T{trial_idx}  {remain:.1f}s left",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
            cv2.putText(out, f"Mode:{st.mode}  Z:{z_txt}  hz:{infer_hz}  hold:{hold_sec}s",
                        (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # Log
            bb = tuple(map(float, st.bbox)) if st.bbox is not None else None
            cx_v, cy_v = bbox_center(bb)
            in_fr = bbox_in_frame(bb, out.shape[1], out.shape[0])
            area_r = bbox_area_ratio(bb, out.shape[1], out.shape[0])
            z_err = (float(z_ema) - TARGET_DISTANCE_M) if z_ema is not None else None

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
                autopilot=ap_flag, manual_active=manual_flag,
                track_mode=0,
            )
            logger.write(row)

            out_show = cv2.resize(out, None, fx=0.9, fy=0.9, interpolation=cv2.INTER_AREA)
            cv2.imshow(window_name, out_show)
            if (cv2.waitKey(1) & 0xFF) == ord(QUIT_KEY):
                print("[USER] Early exit.")
                break

            frame_idx += 1

    finally:
        logger.close()
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass
        with state_lock:
            state["running"] = False

    print(f"[SESSION] Log saved: {log_path}")
    return log_path


# ─────────────────────────────────────────────
# Analysis: compare conditions
# ─────────────────────────────────────────────
def analyze_ablation(log_dir: str, out_dir: str):
    """
    Load all ablation logs from log_dir, group by condition, compute metrics,
    and produce a comparison table.

    Log file naming convention:
        ablation_{CONDITION}_trial{N}_{timestamp}.xlsx
    """
    import glob, re

    pattern = os.path.join(log_dir, "ablation_*.xlsx")
    files = sorted(glob.glob(pattern))
    if not files:
        # Also try CSV
        pattern = os.path.join(log_dir, "ablation_*.csv")
        files = sorted(glob.glob(pattern))

    if not files:
        print(f"[ERR] No ablation log files found in {log_dir}")
        sys.exit(1)

    results: Dict[str, List[dict]] = {k: [] for k in ABLATION_CONDITIONS}

    for f in files:
        bn = os.path.basename(f)
        m = re.match(r"ablation_([A-Z0-9_]+)_trial(\d+)_", bn)
        if m is None:
            print(f"[WARN] skipped (name mismatch): {bn}")
            continue
        cond = m.group(1)
        if cond not in ABLATION_CONDITIONS:
            print(f"[WARN] unknown condition '{cond}': {bn}")
            continue
        try:
            metrics = compute_metrics_from_file(f)
            results[cond].append({
                "file": bn,
                "detect_ratio": metrics.detect_ratio,
                "hold_ratio":   metrics.hold_ratio,
                "lost_ratio":   metrics.lost_ratio,
                "center_err_mean_px": metrics.center_err_mean_px,
                "center_err_p95_px":  metrics.center_err_p95_px,
                "z_err_mean_m":       metrics.z_err_mean_m,
                "z_err_p95_abs_m":    metrics.z_err_p95_abs_m,
                "settle_time_s":      metrics.settle_time_s,
                "infer_dt_mean_s":    metrics.infer_dt_mean_s,
                "infer_dt_p95_s":     metrics.infer_dt_p95_s,
            })
            print(f"[OK] {cond}  {bn}")
        except Exception as e:
            print(f"[ERR] {bn}: {e}")

    # Build summary
    summary_rows = []
    for cond, trial_list in results.items():
        if not trial_list:
            continue
        df = pd.DataFrame(trial_list)

        def ms(col):
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) == 0:
                return None, None
            return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) >= 2 else 0.0

        det_mu,  det_sd  = ms("detect_ratio")
        lost_mu, lost_sd = ms("lost_ratio")
        ce_mu,   ce_sd   = ms("center_err_mean_px")
        ce95_mu, _       = ms("center_err_p95_px")
        ze_mu,   ze_sd   = ms("z_err_mean_m")
        ze95_mu, _       = ms("z_err_p95_abs_m")
        st_mu,   st_sd   = ms("settle_time_s")
        idt_mu,  _       = ms("infer_dt_mean_s")

        summary_rows.append({
            "condition": cond,
            "label": ABLATION_CONDITIONS[cond]["label"],
            "n_trials": len(trial_list),
            "detect_ratio": det_mu,
            "detect_ratio_std": det_sd,
            "lost_ratio": lost_mu,
            "lost_ratio_std": lost_sd,
            "center_err_mean_px": ce_mu,
            "center_err_mean_px_std": ce_sd,
            "center_err_p95_px": ce95_mu,
            "z_err_mean_m": ze_mu,
            "z_err_mean_m_std": ze_sd,
            "z_err_p95_abs_m": ze95_mu,
            "settle_time_s": st_mu,
            "settle_time_s_std": st_sd,
            "infer_dt_mean_s": idt_mu,
        })

    if not summary_rows:
        print("[ERR] No results to summarize.")
        sys.exit(1)

    sum_df = pd.DataFrame(summary_rows)

    # Console table
    print("\n" + "=" * 90)
    print("Exp.5  Ablation Study  —  Summary Table")
    print("=" * 90)
    cols = ["condition", "n_trials", "detect_ratio", "lost_ratio",
            "center_err_mean_px", "z_err_p95_abs_m", "settle_time_s", "infer_dt_mean_s"]
    hdr = f"{'Condition':<12} {'n':>3}  {'det_r':>6}  {'lost_r':>6}  "
    hdr += f"{'ctr_err[px]':>11}  {'ze_p95[m]':>9}  {'settle[s]':>9}  {'inf_dt[s]':>9}"
    print(hdr)
    print("-" * 90)
    for _, r in sum_df.iterrows():
        print(
            f"{str(r['condition']):<12} {int(r['n_trials']):>3}  "
            f"{_fmt(r['detect_ratio'],3):>6}  {_fmt(r['lost_ratio'],3):>6}  "
            f"{_fmt(r['center_err_mean_px'],1):>11}  "
            f"{_fmt(r['z_err_p95_abs_m'],3):>9}  "
            f"{_fmt(r['settle_time_s'],2):>9}  "
            f"{_fmt(r['infer_dt_mean_s'],3):>9}"
        )
    print("=" * 90)

    # Save
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "exp5_ablation_summary.csv")
    sum_df.to_csv(csv_path, index=False)
    print(f"\n[SAVED] {csv_path}")

    try:
        xlsx_path = os.path.join(out_dir, "exp5_ablation_summary.xlsx")
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
            sum_df.to_excel(w, sheet_name="summary", index=False)
        print(f"[SAVED] {xlsx_path}")
    except ImportError:
        pass


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Exp.5 Ablation Study Runner / Analyzer")
    parser.add_argument("--mode", choices=["run", "analyze"], default="analyze")
    parser.add_argument("--condition", type=str, default="FULL",
                        choices=list(ABLATION_CONDITIONS.keys()),
                        help="[run mode] ablation condition to run")
    parser.add_argument("--trial", type=int, default=1,
                        help="[run mode] trial index (1-5 recommended)")
    parser.add_argument("--log_dir", type=str, default=LOG_DIR,
                        help="[analyze mode] directory containing ablation logs")
    parser.add_argument("--out", type=str, default="exp5_results",
                        help="[analyze mode] output directory")
    args = parser.parse_args()

    if args.mode == "analyze":
        analyze_ablation(args.log_dir, args.out)
        return

    # ── run mode ──
    cond_info = ABLATION_CONDITIONS[args.condition]
    print("\n" + "=" * 60)
    print(f"  Exp.5 Ablation Run")
    print(f"  Condition : {args.condition}")
    print(f"  Label     : {cond_info['label']}")
    print(f"  hold_sec  : {cond_info['hold_sec']}")
    print(f"  ema_alpha : {cond_info['ema_alpha_z']}")
    print(f"  infer_hz  : {cond_info['infer_hz']}")
    print(f"  Trial     : {args.trial}")
    print("=" * 60)
    input("  Press Enter when drone is ready and positioned...")

    ctrl = CTRL()
    model = Model(weight_path=WEIGHT_PATH)
    ctrl.set_speed(40)

    print("[INFO] T=takeoff  G=land  ESC=abort")
    try:
        while not keyboard.is_pressed("t"):
            if keyboard.is_pressed("esc"):
                print("[ABORT]")
                ctrl.cleanup()
                return
            time.sleep(0.05)
        ctrl.takeoff()
        time.sleep(1.5)
        log_path = run_session(args.condition, args.trial, ctrl, model)
        ctrl.land()
        if log_path:
            print(f"\n[DONE] Session complete. Log: {log_path}")
            print(f"  Next: run with --condition {args.condition} --trial {args.trial + 1}")
    finally:
        ctrl.cleanup()


if __name__ == "__main__":
    main()
