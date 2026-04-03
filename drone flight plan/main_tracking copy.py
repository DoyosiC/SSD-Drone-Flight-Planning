# main_tracking.py
from __future__ import annotations

import time
import threading
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
import keyboard
import os
from datetime import datetime

from utils import tracking_continuity
from utils.ctrl import CTRL, Tools
from utils.model import Model
from utils.pinhole import Intrinsics, PinholeCamera, Pose

from utils.tracking_continuity import ContinuityController, bbox_area_ratio
from utils.eval_logger import CsvLogger
from utils.eval_metrics import compute_metrics_from_csv


# =========================
# 検出ターゲット（学習済みラベル名に合わせる）
TARGET_LABEL         = "apple"

# ピンホール距離推定
APPLE_SIZE_M         = 0.08
USE_BBOX_AXIS        = "height"      # "height" | "width" | "max"
FOV_H_DEG            = 82.0

# 自動接近（前進・位置合わせ）
AUTOPILOT_START_ON   = False
TARGET_DISTANCE_M    = 0.30
DIST_TOL_M           = 0.05
MAX_VX               = 38
KP_FORWARD           = 150.0
EMA_ALPHA_Z          = 0.35

# センタリング
CENTERING_MODE       = "yaw"         # "yaw" | "strafe"
KP_CENTER_X          = 0.15
MAX_YAW              = 40
MAX_VY               = 30

# 安全
MIN_DISTANCE_M       = 0.20
MAX_DISTANCE_M       = 5.0
QUIT_KEY             = "q"

# ウィンドウ
WINDOW_NAME          = "Tello DET + RANGE (Apple)"
HUD_ALPHA            = 0.0
RESIZE_FX, RESIZE_FY = 0.9, 0.9

# モデル
WEIGHT_PATH          = "./weights/ssd_finetuned_200_filter.pth"
CONF_THRESH          = 0.5

# 連続性補完（推論制御）
INFER_HZ             = 4.0
HOLD_SEC             = 0.5  # 実験２ではここは0.0
EMA_ALPHA_TRACK      = 0.5
LOST_SEC             = 1.5  # これ以上検出が古いなら停止
ENABLE_CONTINUITY_HOLD    = True
ENABLE_CONTINUITY_PREDICT = False

# 評価ログ
LOG_PATH             = "./logs/ics_tracking.csv"
LOG_DIR              = "./logs"
SUCCESS_SEG_SEC      = 3.0

# 色（BGR）
CLR_OTHER            = (0, 255, 0)
CLR_TARGET           = (0, 255, 255)
CLR_IN_RANGE         = (255, 0, 0)


# =========================
# 共有状態（スレッド間）
# =========================
lock = threading.Lock()

state: Dict[str, object] = {
    "vx": 0.0,
    "vy": 0.0,
    "vz": 0.0,
    "yaw": 0.0,
    "running": True,
    "autopilot": AUTOPILOT_START_ON,

    # distance EMA
    "z_ema": None,

    # tracking snapshot for logger
    "frame_w": 0,
    "frame_h": 0,
    "mode": "LOST",
    "label": "",
    "conf": 0.0,
    "bbox": None,          # (x1,y1,x2,y2)
    "area_ratio": None,
    "age_det": 1e9,
    "infer_dt": 0.0,

    "manual_active": False,

    "track_mode": 1,  #1;Yaw-only 2:Yaw+distance 3:distance-only
}

det_lock = threading.Lock()
det_shared = {
    "best_bb": None,        # (x1,y1,x2,y2) or None
    "best_sc": 0.0,
    "boxes": [],
    "labels": [],
    "scores": [],
    "infer_dt": 0.0,
    "last_update_t": 0.0,
}

logger: Optional[CsvLogger] = None


def clamp(v: float, lo: float, hi: float) -> int:
    return int(max(lo, min(hi, v)))


def _wait_release(key: str, step_sleep: float = 0.02):
    while keyboard.is_pressed(key):
        time.sleep(step_sleep)


# =========================
# RC連続送信スレッド（ここでログも吐く）
# =========================
def rc_sender(ctrl: CTRL, rate_hz: int = 20):
    global logger
    period = 1.0 / float(rate_hz)
    t = ctrl.tello

    while True:
        with lock:
            running = bool(state["running"])
            vx = float(state["vx"])
            vy = float(state["vy"])
            vz = float(state["vz"])
            yaw = float(state["yaw"])

            # logger snapshot
            fw = int(state["frame_w"])
            fh = int(state["frame_h"])
            mode = str(state["mode"])
            label = str(state["label"])
            conf = float(state["conf"])
            bbox = state["bbox"]
            area_ratio = state["area_ratio"]
            age_det = float(state["age_det"])
            infer_dt = float(state["infer_dt"])
            autopilot = bool(state["autopilot"])
            manual_active = bool(state["manual_active"])
            track_mode = int(state["track_mode"])

        if not running:
            break

        try:
            # send_rc_control(lr, fb, ud, yaw)
            t.send_rc_control(
                clamp(vy,  -100, 100),
                clamp(vx,  -100, 100),
                clamp(vz,  -100, 100),
                clamp(yaw, -100, 100),
            )
        except Exception as e:
            print(f"[rc_sender] send_rc_control failed: {e}")

        # 評価ログ（制御周期で記録）
        if logger is not None and fw > 0 and fh > 0:
            try:
                logger.log(
                    mode=mode,
                    label=label,
                    conf=conf,
                    bbox=bbox,
                    frame_wh=(fw, fh),
                    area_ratio=area_ratio,
                    age_det=age_det,
                    infer_dt=infer_dt,
                    cmd=(int(vx), int(vy), int(vz), int(yaw)),
                    autopilot=autopilot,
                    manual_active=manual_active,
                    track_mode=track_mode,
                )
            except Exception as e:
                print(f"[rc_sender] logger failed: {e}")

        time.sleep(period)


def infer_worker(fr, model: Model, target_idx: int | None):
    """推論だけを別スレッドで実行し、最新結果だけ det_shared に保存する。"""
    global det_shared

    while True:
        with lock:
            if not bool(state["running"]):
                break

        frame = fr.frame
        if frame is None:
            time.sleep(0.005)
            continue

        # 推論対象フレームはコピー（表示スレッドと干渉しない）
        frame_bgr = frame.copy()

        # 推論周期（INFER_HZ）
        # ※ ここで sleep することで推論頻度を制御
        time.sleep(max(0.0, 1.0 / max(0.1, float(INFER_HZ))))

        t0 = time.time()
        try:
            _, boxes, labels, scores = model.predict_frame(frame_bgr, conf=CONF_THRESH)
        except Exception:
            boxes, labels, scores = [], [], []
        infer_dt = time.time() - t0

        # ターゲットbbox選択（最大面積）
        best_bb = None
        best_sc = 0.0
        if target_idx is not None and boxes is not None and len(boxes) > 0:
            best_area = -1.0
            for i, bb in enumerate(boxes):
                li = int(labels[i]) if i < len(labels) else -1
                sc = float(scores[i]) if i < len(scores) else 0.0
                if li != target_idx or sc < CONF_THRESH:
                    continue
                x1, y1, x2, y2 = [float(v) for v in bb]
                area = max(1.0, x2 - x1) * max(1.0, y2 - y1)
                if area > best_area:
                    best_area = area
                    best_bb = (x1, y1, x2, y2)
                    best_sc = sc

        with det_lock:
            det_shared["best_bb"] = best_bb
            det_shared["best_sc"] = float(best_sc)
            det_shared["boxes"] = boxes if boxes is not None else []
            det_shared["labels"] = labels if labels is not None else []
            det_shared["scores"] = scores if scores is not None else []
            det_shared["infer_dt"] = float(infer_dt)
            det_shared["last_update_t"] = time.time()



# =========================
# 検出＋距離推定＋自動接近（連続性補完あり）
# =========================
def run_det_range(ctrl: CTRL, model: Model):
    tello = ctrl.tello
    fr = tello.get_frame_read()
    
    try:
        cv2.setUseOptimized(True)
        cv2.setNumThreads(0)
    except Exception:
        pass

    # pinhole
    intr = None
    cam = None

    categories = list(model.voc_classes)
    try:
        target_idx = categories.index(TARGET_LABEL)
        print(f"[INFO] TARGET_LABEL='{TARGET_LABEL}' -> id={target_idx}")
    except ValueError:
        print(f"[WARN] label '{TARGET_LABEL}' not found in model.voc_classes. autopilot range disabled.")
        target_idx = None

    # continuity controller
    hold_sec_use = HOLD_SEC if ENABLE_CONTINUITY_HOLD else 0.0
    cont = ContinuityController(infer_hz=INFER_HZ, hold_sec=hold_sec_use, ema_alpha=EMA_ALPHA_TRACK)
    # ---- start inference thread (must) ----
    inf_th = threading.Thread(target=infer_worker, args=(fr, model, target_idx), daemon=True)
    inf_th.start()
    print("[INF] inference worker started")



    prev_t = time.time()
    avg_fps = 0.0

    while True:
        with lock:
            if not bool(state["running"]):
                break
            autopilot = bool(state["autopilot"])
            manual_active = bool(state["manual_active"])
            track_mode = int(state["track_mode"])
        
        frame_bgr = fr.frame
        # frame_bgr= cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if frame_bgr is None:
            if (cv2.waitKey(1) & 0xFF) == ord(QUIT_KEY):
                time.sleep(0.001)  # display/control loop throttle

                break
            continue

        h0, w0 = frame_bgr.shape[:2]
        with lock:
            state["frame_w"] = w0
            state["frame_h"] = h0

        # init intrinsics once
        if cam is None:
            intr = Intrinsics.from_fov(width=w0, height=h0, fov_deg_h=FOV_H_DEG)
            cam = PinholeCamera(intr=intr, pose_w2c=Pose.identity())
            print(f"[PINHOLE] K=\n{cam.intr.K}\nFOV(h,v)={cam.intr.fov_deg()}")

        # --- 推論（レート制限） ---
        def _handle_no_detection(cont: ContinuityController):
            st = cont.state

            if (not ENABLE_CONTINUITY_HOLD) and (not ENABLE_CONTINUITY_PREDICT):
                st.bbox = None; st.cx_s = None; st.cy_s = None
                st.label = ""; st.conf = 0.0; st.mode = "LOST"
                return

            if ENABLE_CONTINUITY_HOLD and (not ENABLE_CONTINUITY_PREDICT):
                age = st.age_det()
                if age <= cont.hold_sec:
                    st.mode = "HOLD"
                else:
                    st.bbox = None; st.cx_s = None; st.cy_s = None
                    st.label = ""; st.conf = 0.0; st.mode = "LOST"
                return

            cont.update_no_detection()

        boxes, labels, scores = [], [], []

        # ---- inference result (from worker thread) ----
        with det_lock:
            best_bb = det_shared["best_bb"]          # (x1,y1,x2,y2) or None
            best_sc = float(det_shared["best_sc"])
            boxes   = det_shared["boxes"]
            labels  = det_shared["labels"]
            scores  = det_shared["scores"]
            infer_dt = float(det_shared["infer_dt"])
            last_update_t = float(det_shared["last_update_t"])

        # ---- continuity update ----
        if best_bb is not None:
            cont.update_with_detection(best_bb, TARGET_LABEL, best_sc)
        else:
            _handle_no_detection(cont)




        st = cont.state

        # --- 距離推定（DETECT/HOLD/PREDICT で bbox を使う） ---
        z_est = None
        best_bb_i = None
        if cam is not None and st.bbox is not None and target_idx is not None and st.mode in ("DETECT", "HOLD", "PREDICT"):
            x1, y1, x2, y2 = [int(v) for v in st.bbox]
            w_px = max(1, x2 - x1)
            h_px = max(1, y2 - y1)

            if USE_BBOX_AXIS == "height":
                px_len = h_px
                f_pix = cam.intr.fy
            elif USE_BBOX_AXIS == "width":
                px_len = w_px
                f_pix = cam.intr.fx
            else:
                if w_px >= h_px:
                    px_len, f_pix = w_px, cam.intr.fx
                else:
                    px_len, f_pix = h_px, cam.intr.fy

            Z = (f_pix * APPLE_SIZE_M) / float(max(1, px_len))
            if MIN_DISTANCE_M <= Z <= MAX_DISTANCE_M:
                z_est = float(Z)
                best_bb_i = (x1, y1, x2, y2)

        # 距離 EMA
        with lock:
            if z_est is not None:
                if state["z_ema"] is None:
                    state["z_ema"] = z_est
                else:
                    state["z_ema"] = EMA_ALPHA_Z * z_est + (1.0 - EMA_ALPHA_Z) * float(state["z_ema"])

        # --- 自動接近（手動中は抑制） ---
        # 検出が古すぎたら停止（連続性補完でも制御暴走させないため）
        age_det = st.age_det()
        if autopilot and (not manual_active):
            if age_det > LOST_SEC or st.mode == "LOST" or best_bb_i is None:
                with lock:
                    state["vx"] = state["vy"] = state["yaw"] = 0
            else:
                x1, y1, x2, y2 = best_bb_i
                cx = 0.5 * (x1 + x2)
                ex = (cx - (w0 * 0.5)) / max(1.0, w0 * 0.5)

                with lock:
                    z_use = float(state["z_ema"]) if state["z_ema"] is not None else (z_est if z_est is not None else 999.0)

                # --- 前後制御（目標距離との差） ---
                e = z_use - TARGET_DISTANCE_M
                if abs(e) <= DIST_TOL_M:
                    vx_cmd = 0
                else:
                    vx_cmd = int(np.clip(KP_FORWARD * e * 10.0, -MAX_VX, MAX_VX))

                if z_use <= MIN_DISTANCE_M:
                    vx_cmd = min(0, vx_cmd)

                # --- センタリング ---
                if CENTERING_MODE == "yaw":
                    yaw_cmd = int(np.clip(KP_CENTER_X * ex * 100.0, -MAX_YAW, MAX_YAW))
                    vy_cmd = 0
                else:
                    vy_cmd = int(np.clip(KP_CENTER_X * ex * 100.0, -MAX_VY, MAX_VY))
                    yaw_cmd = 0

                # --- ここでモードマスク（最後にやる） ---
                if track_mode == 1:
                    # Yaw-only：前後/横移動禁止（yawは残す）
                    vx_cmd = 0
                    vy_cmd = 0
                elif track_mode == 2:
                    # Yaw + Distance：全て有効
                    pass
                elif track_mode == 3:
                    # Distance-only：旋回/横移動禁止（vxだけ残す）
                    yaw_cmd = 0
                    vy_cmd = 0
                else:
                    vx_cmd = vy_cmd = yaw_cmd = 0

                with lock:
                    state["vx"] = vx_cmd
                    state["vy"] = vy_cmd
                    state["yaw"] = yaw_cmd


        # --- 可視化（あなたの元main.pyと同等、ただし BGRのまま） ---
        out = frame_bgr.copy()

        # 既存の全bbox描画（軽量化したいならここを切る）
        # ※ ただし boxes が空の周期もある（推論レート制限）
        if boxes is not None and len(boxes) > 0:
            for i, bb in enumerate(boxes):
                li = int(labels[i]) if i < len(labels) else -1
                sc = float(scores[i]) if i < len(scores) else 0.0
                x1, y1, x2, y2 = [int(v) for v in bb]
                name = categories[li] if 0 <= li < len(categories) else f"id:{li}"
                color = CLR_TARGET if (target_idx is not None and li == target_idx) else CLR_OTHER
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                txt = f"{name}:{sc:.2f}"
                tw = max(60, 10 * len(txt))
                cv2.rectangle(out, (x1, max(0, y1 - 18)), (x1 + tw, y1), color, -1)
                cv2.putText(out, txt, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 1, cv2.LINE_AA)

        # 選択対象（連続性補完したbbox）を強調
        if best_bb_i is not None:
            x1, y1, x2, y2 = best_bb_i
            with lock:
                z_use = float(state["z_ema"]) if state["z_ema"] is not None else (z_est if z_est is not None else 999.0)
            in_range = (z_est is not None) and (abs(z_use - TARGET_DISTANCE_M) <= DIST_TOL_M)
            strong_color = CLR_IN_RANGE if in_range else CLR_TARGET
            cv2.rectangle(out, (x1, y1), (x2, y2), strong_color, 3)
            if z_est is not None:
                cv2.putText(out, f"Z~{z_use:.2f} m [{st.mode}]",
                            (x1 + 3, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, strong_color, 2, cv2.LINE_AA)

        # HUD
        try: bat = tello.get_battery()
        except: bat = "N/A"
        try: alt = tello.get_height()
        except: alt = "N/A"
        spd = ctrl.speed

        now = time.time()
        dt = max(1e-6, now - prev_t)
        fps = 1.0 / dt
        avg_fps = 0.9 * avg_fps + 0.1 * fps if avg_fps > 0 else fps
        prev_t = now

        x, y, w_hud, h_hud = 10, 10, 470, 170
        if HUD_ALPHA > 0.0:
            overlay = out.copy()
            cv2.rectangle(overlay, (x, y), (x + w_hud, y + h_hud), (0, 0, 0), -1)
            out = cv2.addWeighted(overlay, HUD_ALPHA, out, 1.0 - HUD_ALPHA, 0.0)
        else:
            cv2.rectangle(out, (x, y), (x + w_hud, y + h_hud), (0, 0, 0), 1)

        def put_line(text: str, row: int):
            cv2.putText(out, text, (x + 12, y + 25 + 24 * row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        with lock:
            ap = "ON" if state["autopilot"] else "OFF"
            z_text = f"{float(state['z_ema']):.2f} m" if state["z_ema"] is not None else "-"
        put_line(f"TrackMode: {track_mode} (1:Yaw-only,2:Yaw+Dist,3:Dist-only)", 0)
        put_line(f"AutoApproach: {ap}  (R=toggle)", 1)
        put_line(f"Mode: {st.mode}  age={age_det:.2f}s  infer_dt={infer_dt*1000:.1f}ms", 2)

        put_line(f"Est.Range: {z_text}  (Target: {TARGET_DISTANCE_M:.2f} m)", 3)
        put_line(f"Battery: {bat}%  Alt: {alt} cm", 4)
        put_line(f"Speed: {spd}  FPS: {avg_fps:.1f}", 5)
        put_line(f"[{QUIT_KEY.upper()}] to quit", 6)

        # --- logger用スナップショット更新（ロック内でまとめて） ---
        with lock:
            state["mode"] = st.mode
            state["label"] = st.label
            state["conf"] = float(st.conf)
            state["bbox"] = st.bbox
            state["age_det"] = float(age_det)
            state["infer_dt"] = float(infer_dt)
            state["area_ratio"] = bbox_area_ratio(st.bbox, w0, h0)

        out = cv2.resize(out, None, fx=RESIZE_FX, fy=RESIZE_FY, interpolation=cv2.INTER_AREA)
        cv2.imshow(WINDOW_NAME, out)
        time.sleep(0.001)  # display/control loop throttle

        if (cv2.waitKey(1) & 0xFF) == ord(QUIT_KEY):
            time.sleep(0.001)  # display/control loop throttle
            break

    try:
        cv2.destroyWindow(WINDOW_NAME)
    except Exception:
        pass


# =========================
# メイン（main.py を踏襲）
# =========================
def main():
    global logger

    ctrl = CTRL()
    tools = Tools(ctrl.tello)
    model = Model(weight_path=WEIGHT_PATH)

    ctrl.set_speed(40)
    print(f"[INIT] speed = {ctrl.speed}")

    # ---- logger path (unique) ----
    os.makedirs(LOG_DIR, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"ics_tracking_{run_id}.csv")

    # logger start
    logger = CsvLogger(log_path)
    print(f"[EVAL] logging -> {log_path}")

    # RC送信スレッド
    sender = threading.Thread(target=rc_sender, args=(ctrl, 20), daemon=True)
    sender.start()

    # 検出＋距離推定＋自動接近スレッド（連続性補完あり）
    det_thread = threading.Thread(target=run_det_range, args=(ctrl, model), daemon=True)
    det_thread.start()

    print("[INFO] 操作: W/S 前後, A/D 左右, I/K 上下, J/L ヨー, Space 停止")
    print("[INFO] 速度: '+' 加速, '-' 減速, '0' 既定値")
    print("[INFO] 離陸/着陸: T / G,  自動接近: R  切替,  終了: ESC")

    time.sleep(0.5)

    try:
        while True:
            # 速度調整
            if keyboard.is_pressed('+'):
                cur = ctrl.inc_speed(); print(f"[SPEED] Up -> {cur}");  _wait_release('+')
            if keyboard.is_pressed('-'):
                cur = ctrl.dec_speed(); print(f"[SPEED] Down -> {cur}"); _wait_release('-')
            if keyboard.is_pressed('0'):
                ctrl.set_speed(CTRL.DRONE_SPEED); _wait_release('0')

            # 離着陸
            if keyboard.is_pressed('t'):
                _wait_release('t'); ctrl.takeoff()
            if keyboard.is_pressed('g'):
                _wait_release('g'); ctrl.land()

            # 自動接近 切替
            if keyboard.is_pressed('r'):
                _wait_release('r')
                with lock:
                    state["autopilot"] = not bool(state["autopilot"])
                    print(f"[AUTO] AutoApproach -> {state['autopilot']}")

            # 追従モード　切替
            if keyboard.is_pressed('1'):
                _wait_release('1')
                with lock:
                    state["track_mode"] = 1
                print("[MODE] Track Mode -> 1 (Yaw-only)")

            if keyboard.is_pressed('2'):
                _wait_release('2')
                with lock:
                    state["track_mode"] = 2
                print(f"[MODE] Track Mode -> 2 (Yaw+Distance)")

            if keyboard.is_pressed('3'):
                _wait_release('3')
                with lock:
                    state["track_mode"] = 3
                print(f"[MODE] Track Mode -> 3 (Distance-only)")
            

            # デフォルトはホバー（0）— キー押下時のみ移動
            vx = vy = vz = yaw = 0
            if keyboard.is_pressed('w'): vx =  ctrl.speed
            if keyboard.is_pressed('s'): vx = -ctrl.speed
            if keyboard.is_pressed('d'): vy =  ctrl.speed
            if keyboard.is_pressed('a'): vy = -ctrl.speed
            if keyboard.is_pressed('i'): vz =  ctrl.u
            if keyboard.is_pressed('k'): vz = -ctrl.d
            if keyboard.is_pressed('l'): yaw =  ctrl.speed
            if keyboard.is_pressed('j'): yaw = -ctrl.speed

            manual_active = any([vx, vy, vz, yaw])
            with lock:
                state["manual_active"] = bool(manual_active)

            if manual_active:
                with lock:
                    state["vx"], state["vy"], state["vz"], state["yaw"] = vx, vy, vz, yaw
            else:
                with lock:
                    if not bool(state["autopilot"]):
                        state["vx"] = state["vy"] = state["vz"] = state["yaw"] = 0

            # 非常停止
            if keyboard.is_pressed('space'):
                with lock:
                    state["vx"] = state["vy"] = state["vz"] = state["yaw"] = 0

            # 全終了
            if keyboard.is_pressed('esc'):
                print("[ESC] detected. stopping ...")
                break

            time.sleep(0.01)

    finally:
        with lock:
            state["running"] = False

        # 停止・掃除
        try: cv2.destroyWindow(WINDOW_NAME)
        except Exception: pass
        try: ctrl.tello.send_rc_control(0, 0, 0, 0)
        except Exception: pass
        try: ctrl.land()
        except Exception: pass
        try: tools.stop_overlay()
        except Exception: pass
        ctrl.cleanup()

        # logger close + metrics
        if logger is not None:
            logger.close()

        m = compute_metrics_from_csv(log_path, success_seg_sec=SUCCESS_SEG_SEC)
        print("\n[RESULT] Metrics from log")
        print(f"  total_time              : {m.total_time:.2f} s")
        print(f"  detect_ratio            : {m.detect_ratio:.3f}")
        print(f"  lost_ratio              : {m.lost_ratio:.3f}")
        print(f"  mean_age_det            : {m.mean_age_det:.3f} s")
        if m.mean_abs_center_err_px is not None:
            print(f"  mean_abs_center_err_px  : {m.mean_abs_center_err_px:.1f} px")
        if m.mean_follow_segment_s is not None:
            print(f"  mean_follow_segment_s   : {m.mean_follow_segment_s:.2f} s")
        if m.follow_success_ratio is not None:
            print(f"  follow_success_ratio(seg>={SUCCESS_SEG_SEC}s): {m.follow_success_ratio:.3f}")
        print(f"\n[INFO] Log saved: {log_path}")
        print("[DONE] cleanup completed.")


if __name__ == "__main__":
    main()
