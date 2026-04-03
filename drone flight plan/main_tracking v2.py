# main_tracking.py
from __future__ import annotations

import os
import time
import threading
from datetime import datetime
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
import keyboard

from utils.ctrl import CTRL, Tools
from utils.model import Model
from utils.pinhole import Intrinsics, PinholeCamera, Pose

from utils.tracking_continuity import ContinuityController
from utils.eval_logger import (
    create_logger,
    EvalRow,
    bbox_center,
    bbox_in_frame,
    bbox_area_ratio,
)

# =========================
# 研究・評価ターゲット
# =========================
TARGET_LABEL = "apple"

# =========================
# ピンホール距離推定（main.py準拠）
# =========================
APPLE_SIZE_M   = 0.08     # 代表直径[m]
USE_BBOX_AXIS  = "height" # "height" | "width" | "max"
FOV_H_DEG      = 82.0     # 水平FOV[deg]（実機に合わせる）

# =========================
# 自動接近（前進・位置合わせ） main.py準拠
# =========================
AUTOPILOT_START_ON = False

TARGET_DISTANCE_M  = 0.30   # ← 30cmを基本に（あなたの実験結果に合わせる）
DIST_TOL_M         = 0.08
MIN_DISTANCE_M     = 0.20   # ← 安全側
MAX_DISTANCE_M     = 5.0

MAX_VX             = 38
KP_FORWARD         = 150.0
EMA_ALPHA_Z        = 0.35

CENTERING_MODE     = "yaw"  # "yaw" | "strafe"
KP_CENTER_X        = 0.15
MAX_YAW            = 40
MAX_VY             = 30

ALT_LIMIT_CM       = 200
QUIT_KEY           = "q"

# =========================
# 検出モデル（main.pyのパスを優先）
# =========================
WEIGHT_PATH   = "./weights/ssd_finetuned_filterv2.pth"
CONF_THRESH   = 0.5

# =========================
# 評価のための推論周波数・連続性（検出抜け補完）
# =========================
INFER_HZ = 4.0  # 推論は毎フレームではなく固定Hz（停止感を減らす）
ENABLE_HOLD = True
HOLD_SEC = 0.5

# 予測補完（PREDICT）は論文で誤魔化し扱いになりやすいのでデフォルトOFF
ENABLE_PREDICT = False

# =========================
# 表示/HUD（main.py準拠）
# =========================
WINDOW_NAME          = "Tello DET + RANGE + TRACK (Apple)"
HUD_ALPHA            = 0.0
RESIZE_FX, RESIZE_FY = 0.9, 0.9

# 色（BGR）
CLR_OTHER     = (0, 255, 0)
CLR_TARGET    = (0, 255, 255)
CLR_IN_RANGE  = (255, 0, 0)

# =========================
# ログ（xlsx推奨）
# =========================
LOG_DIR = "./logs"


# =========================
# ランタイム共有状態（main.py準拠 + 追跡評価用の最小拡張）
# =========================
state: Dict[str, float | bool | object] = {
    "vx": 0.0,
    "vy": 0.0,
    "vz": 0.0,
    "yaw": 0.0,
    "running": True,
    "autopilot": AUTOPILOT_START_ON,
    "z_ema": None,        # 距離EMA
    "infer_dt": 0.0,      # 推論時間
    "manual_active": False,
}

state_lock = threading.Lock()


def clamp(v: float, lo: float, hi: float) -> int:
    return int(max(lo, min(hi, v)))


def _wait_release(key: str, step_sleep: float = 0.02):
    while keyboard.is_pressed(key):
        time.sleep(step_sleep)


def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# =========================
# RC連続送信スレッド（main.py準拠）
# =========================
def rc_sender(ctrl: CTRL, rate_hz: int = 20):
    period = 1.0 / float(rate_hz)
    t = ctrl.tello
    while True:
        with state_lock:
            if not bool(state["running"]):
                break
            vy = float(state["vy"])
            vx = float(state["vx"])
            vz = float(state["vz"])
            yaw = float(state["yaw"])

        try:
            t.send_rc_control(
                clamp(vy,  -100, 100),
                clamp(vx,  -100, 100),
                clamp(vz,  -100, 100),
                clamp(yaw, -100, 100),
            )
        except Exception as e:
            print(f"[rc_sender] send_rc_control failed: {e}")
        time.sleep(period)


# =========================
# 推論（固定Hz）＋追跡補完＋距離推定＋評価ログ＋HUD
# =========================
def run_det_range_track(ctrl: CTRL, model: Model):
    tello = ctrl.tello
    fr = tello.get_frame_read()

    # OpenCV最適化
    try:
        cv2.setUseOptimized(True)
        cv2.setNumThreads(0)
    except Exception:
        pass

    # pinhole
    cam: Optional[PinholeCamera] = None
    categories = list(model.voc_classes)

    try:
        target_idx = categories.index(TARGET_LABEL)
        print(f"[INFO] TARGET_LABEL='{TARGET_LABEL}' -> voc_idx={target_idx}")
    except ValueError:
        print(f"[WARN] label '{TARGET_LABEL}' がモデル内に見つかりません。追跡評価は不正確になります。")
        target_idx = None

    # continuity controller（検出抜けの連続性）
    hold_sec_use = HOLD_SEC if ENABLE_HOLD else 0.0
    cont = ContinuityController(infer_hz=float(INFER_HZ), hold_sec=float(hold_sec_use), ema_alpha=0.5)

    # logger
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"ics_tracking_{now_str()}.xlsx")
    logger = create_logger(log_path)
    print("[LOG] ->", log_path)

    # 推論スレッド共有
    det_lock = threading.Lock()
    det_shared = {
        "boxes": [],
        "labels": [],
        "scores": [],
        "infer_dt": 0.0,
        "last_t": 0.0,
    }

    # 固定Hz推論スレッド（UI停止を回避）
    def infer_worker():
        period = 1.0 / max(0.1, float(INFER_HZ))
        while True:
            with state_lock:
                if not bool(state["running"]):
                    break

            frame_bgr = fr.frame
            if frame_bgr is None:
                time.sleep(0.01)
                continue

            # main.pyは BGR->RGB に変換していたが、utils.model.Model 側が何を想定しているかで変わる。
            # あなたのmain.pyに合わせるならここでRGB化する。ただし、もし認識が落ちるならここを外す。
            frame_in = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            t0 = time.time()
            try:
                _, boxes, labels, scores = model.predict_frame(frame_in, conf=CONF_THRESH)
            except Exception:
                boxes, labels, scores = [], [], []
            infer_dt = time.time() - t0

            with det_lock:
                det_shared["boxes"] = boxes
                det_shared["labels"] = labels
                det_shared["scores"] = scores
                det_shared["infer_dt"] = float(infer_dt)
                det_shared["last_t"] = time.time()

            with state_lock:
                state["infer_dt"] = float(infer_dt)

            time.sleep(max(0.0, period - (time.time() - t0)))

    threading.Thread(target=infer_worker, daemon=True).start()

    prev_t = time.time()
    avg_fps = 0.0
    frame_idx = 0

    try:
        while True:
            with state_lock:
                if not bool(state["running"]):
                    break

            frame_bgr = fr.frame
            if frame_bgr is None:
                if (cv2.waitKey(1) & 0xFF) == ord(QUIT_KEY):
                    break
                continue

            # main.py表示はRGBベースだったので合わせる（表示とbbox計算を一致させる）
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            out = frame.copy()

            # 初回にpinhole構築
            if cam is None:
                h0, w0 = out.shape[:2]
                intr = Intrinsics.from_fov(width=w0, height=h0, fov_deg_h=FOV_H_DEG)
                cam = PinholeCamera(intr=intr, pose_w2c=Pose.identity())
                print(f"[PINHOLE] K=\n{cam.intr.K}\nFOV(h,v)={cam.intr.fov_deg()}")

            # 推論結果取得（最新）
            with det_lock:
                boxes = det_shared["boxes"]
                labels = det_shared["labels"]
                scores = det_shared["scores"]
                infer_dt = float(det_shared["infer_dt"])

            # すべてのbbox描画 + ターゲット候補選択（最大面積）
            best_bb: Optional[Tuple[int, int, int, int]] = None
            best_sc = 0.0
            z_est: Optional[float] = None

            for i, bb in enumerate(boxes):
                li = int(labels[i]) if i < len(labels) else -1
                sc = float(scores[i]) if i < len(scores) else 0.0
                x1, y1, x2, y2 = [int(v) for v in bb]
                w_px = max(1, x2 - x1)
                h_px = max(1, y2 - y1)

                name = categories[li] if 0 <= li < len(categories) else f"id:{li}"
                color = CLR_TARGET if (target_idx is not None and li == target_idx) else CLR_OTHER

                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                label = f"{name}:{sc:.2f}"
                tw = max(60, 10 * len(label))
                cv2.rectangle(out, (x1, max(0, y1 - 18)), (x1 + tw, y1), color, -1)
                cv2.putText(out, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 1, cv2.LINE_AA)

                # ターゲット選択（confしきい値 + 最大面積）
                if (target_idx is not None) and (li == target_idx) and (sc >= CONF_THRESH):
                    area = w_px * h_px
                    if (best_bb is None) or (area > (best_bb[2]-best_bb[0]) * (best_bb[3]-best_bb[1])):
                        best_bb = (x1, y1, x2, y2)
                        best_sc = sc

            # continuity更新（検出抜け補完＝HOLD）
            if best_bb is not None:
                cont.update_with_detection(best_bb, TARGET_LABEL, best_sc)
            else:
                # main.pyで「見失い時は停止」していたのと整合
                if ENABLE_HOLD and (not ENABLE_PREDICT):
                    if cont.state.age_det() <= cont.hold_sec:
                        cont.state.mode = "HOLD"
                    else:
                        cont.state.bbox = None
                        cont.state.cx_s = None
                        cont.state.cy_s = None
                        cont.state.label = ""
                        cont.state.conf = 0.0
                        cont.state.mode = "LOST"
                else:
                    cont.update_no_detection()

            st = cont.state

            # 距離推定（contのbboxを使う＝HOLD中も距離推定が継続する）
            if cam is not None and st.bbox is not None and st.mode in ("DETECT", "HOLD", "PREDICT"):
                x1, y1, x2, y2 = [int(v) for v in st.bbox]
                w_px = max(1, x2 - x1)
                h_px = max(1, y2 - y1)

                if USE_BBOX_AXIS == "height":
                    px_len = h_px; f_pix = cam.intr.fy
                elif USE_BBOX_AXIS == "width":
                    px_len = w_px; f_pix = cam.intr.fx
                else:
                    if w_px >= h_px: px_len, f_pix = w_px, cam.intr.fx
                    else:            px_len, f_pix = h_px, cam.intr.fy

                Z = (float(f_pix) * float(APPLE_SIZE_M)) / float(max(1, px_len))
                if MIN_DISTANCE_M <= Z <= MAX_DISTANCE_M:
                    z_est = float(Z)

            # 距離EMA
            with state_lock:
                if z_est is not None:
                    if state["z_ema"] is None:
                        state["z_ema"] = z_est
                    else:
                        state["z_ema"] = EMA_ALPHA_Z * z_est + (1.0 - EMA_ALPHA_Z) * float(state["z_ema"])
                z_ema = state["z_ema"]

            # 手動入力判定（main.py準拠）
            manual_active = (
                keyboard.is_pressed('w') or keyboard.is_pressed('s') or
                keyboard.is_pressed('a') or keyboard.is_pressed('d') or
                keyboard.is_pressed('i') or keyboard.is_pressed('k') or
                keyboard.is_pressed('j') or keyboard.is_pressed('l') or
                keyboard.is_pressed('space')
            )
            with state_lock:
                state["manual_active"] = bool(manual_active)

            # 自動接近（main.py準拠：距離＋センタリング）
            with state_lock:
                autopilot = bool(state["autopilot"])

            if autopilot and (not manual_active) and (z_est is not None) and (st.bbox is not None):
                x1, y1, x2, y2 = [int(v) for v in st.bbox]
                cx = 0.5 * (x1 + x2)
                h, w = out.shape[:2]
                ex = (cx - (w * 0.5)) / max(1.0, w * 0.5)

                z_use = float(z_ema) if z_ema is not None else float(z_est)
                e = z_use - TARGET_DISTANCE_M

                if abs(e) <= DIST_TOL_M:
                    vx_cmd = 0
                else:
                    vx_cmd = int(np.clip(KP_FORWARD * e * 10.0, -MAX_VX, MAX_VX))

                # 近すぎ安全：前進禁止
                if z_use <= MIN_DISTANCE_M:
                    vx_cmd = min(0, vx_cmd)

                # センタリング
                if CENTERING_MODE == "yaw":
                    yaw_cmd = int(np.clip(KP_CENTER_X * ex * 100.0, -MAX_YAW, MAX_YAW))
                    vy_cmd = 0
                else:
                    vy_cmd = int(np.clip(KP_CENTER_X * ex * 100.0, -MAX_VY, MAX_VY))
                    yaw_cmd = 0

                with state_lock:
                    state["vx"] = vx_cmd
                    state["vy"] = vy_cmd
                    state["yaw"] = yaw_cmd

            else:
                # 見失い時（LOST）に停止：main.pyの挙動と整合
                if autopilot and (not manual_active) and (st.bbox is None or z_est is None):
                    with state_lock:
                        state["vx"] = 0
                        state["vy"] = 0
                        state["yaw"] = 0

            # 選択bboxの強調（距離/範囲表示）
            if st.bbox is not None:
                x1, y1, x2, y2 = [int(v) for v in st.bbox]
                z_use = float(z_ema) if z_ema is not None else (float(z_est) if z_est is not None else None)
                if z_use is not None:
                    in_range = abs(z_use - TARGET_DISTANCE_M) <= DIST_TOL_M
                    strong_color = CLR_IN_RANGE if in_range else CLR_TARGET
                    cv2.rectangle(out, (x1, y1), (x2, y2), strong_color, 3)
                    txt = f"Z~{z_use:.2f} m"
                    if in_range:
                        txt += " (IN RANGE)"
                    cv2.putText(out, txt, (x1 + 3, y2 + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, strong_color, 2, cv2.LINE_AA)

            # HUD（main.py準拠 + mode/infer_dt）
            try:
                bat = tello.get_battery()
            except Exception:
                bat = "N/A"
            try:
                alt = tello.get_height()
            except Exception:
                alt = "N/A"

            now = time.time()
            dt = max(1e-6, now - prev_t)
            fps = 1.0 / dt
            avg_fps = 0.9 * avg_fps + 0.1 * fps if avg_fps > 0 else fps
            prev_t = now

            xh, yh, w_hud, h_hud = 10, 10, 460, 185
            if HUD_ALPHA > 0.0:
                overlay = out.copy()
                cv2.rectangle(overlay, (xh, yh), (xh + w_hud, yh + h_hud), (0, 0, 0), -1)
                out = cv2.addWeighted(overlay, HUD_ALPHA, out, 1.0 - HUD_ALPHA, 0.0)
            else:
                cv2.rectangle(out, (xh, yh), (xh + w_hud, yh + h_hud), (0, 0, 0), 1)

            def put(text: str, row: int):
                cv2.putText(out, text, (xh + 12, yh + 25 + 24 * row),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            ap = "ON" if autopilot else "OFF"
            z_txt = f"{float(z_ema):.2f} m" if z_ema is not None else "-"
            put(f"AutoApproach: {ap}  (R=toggle)", 0)
            put(f"Mode: {st.mode}  conf={st.conf:.2f}  infer_dt={infer_dt*1000:.0f}ms", 1)
            put(f"Est.Range: {z_txt}  (Target: {TARGET_DISTANCE_M:.2f} m)", 2)
            put(f"Battery: {bat}%  Alt: {alt} cm  FPS: {avg_fps:.1f}", 3)
            put(f"[{QUIT_KEY.upper()}] quit", 4)

            # =========================
            # 3評価のためのログ（毎フレーム）
            # =========================
            with state_lock:
                vx = int(state["vx"])
                vy = int(state["vy"])
                vz = int(state["vz"])
                yaw = int(state["yaw"])
                manual_flag = 1 if bool(state["manual_active"]) else 0
                ap_flag = 1 if autopilot else 0

            bb = tuple(map(float, st.bbox)) if st.bbox is not None else None
            cx, cy = bbox_center(bb)
            in_fr = bbox_in_frame(bb, int(out.shape[1]), int(out.shape[0]))
            area_r = bbox_area_ratio(bb, int(out.shape[1]), int(out.shape[0]))

            z_err = (float(z_ema) - float(TARGET_DISTANCE_M)) if z_ema is not None else None

            row = EvalRow(
                t=time.time(),
                frame_idx=int(frame_idx),

                mode=str(st.mode),
                label=str(st.label),
                conf=float(st.conf) if st.conf is not None else None,

                age_det_s=float(st.age_det()),
                infer_dt_s=float(infer_dt),

                cx=float(cx) if cx is not None else None,
                cy=float(cy) if cy is not None else None,
                x1=float(bb[0]) if bb is not None else None,
                y1=float(bb[1]) if bb is not None else None,
                x2=float(bb[2]) if bb is not None else None,
                y2=float(bb[3]) if bb is not None else None,
                frame_w=int(out.shape[1]),
                frame_h=int(out.shape[0]),
                in_frame=int(in_fr) if in_fr is not None else None,
                area_ratio=float(area_r) if area_r is not None else None,

                z_est_m=float(z_est) if z_est is not None else None,
                z_ema_m=float(z_ema) if z_ema is not None else None,
                z_err_m=float(z_err) if z_err is not None else None,
                target_z_m=float(TARGET_DISTANCE_M),

                rc_vx=int(vx),
                rc_vy=int(vy),
                rc_vz=int(vz),
                rc_yaw=int(yaw),
                autopilot=int(ap_flag),
                manual_active=int(manual_flag),
                track_mode=0,
            )
            logger.write(row)

            # 表示
            out_show = cv2.resize(out, None, fx=RESIZE_FX, fy=RESIZE_FY, interpolation=cv2.INTER_AREA)
            cv2.imshow(WINDOW_NAME, out_show)
            if (cv2.waitKey(1) & 0xFF) == ord(QUIT_KEY):
                break

            frame_idx += 1

    finally:
        try:
            logger.close()
        except Exception:
            pass
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except Exception:
            pass


# =========================
# メイン（main.py準拠）
# =========================
def main():
    ctrl = CTRL()
    tools = Tools(ctrl.tello)
    model = Model(weight_path=WEIGHT_PATH)

    # 起動ログ
    try:
        import torch
        print("[CHK] net device:", next(model.netb.parameters()).device)
        print("[CHK] dbox device:", model.netb.dbox_list.device)
        print("[CUDA] available:", torch.cuda.is_available())
    except Exception:
        pass

    ctrl.set_speed(40)
    print(f"[INIT] speed = {ctrl.speed}")

    # RC送信スレッド
    sender = threading.Thread(target=rc_sender, args=(ctrl, 20), daemon=True)
    sender.start()

    # 検出＋距離推定＋追跡補完＋評価ログ
    det_thread = threading.Thread(target=run_det_range_track, args=(ctrl, model), daemon=True)
    det_thread.start()

    print("[INFO] 操作: W/S 前後, A/D 左右, I/K 上下, J/L ヨー, Space 停止")
    print("[INFO] 速度: '+' 加速, '-' 減速, '0' 既定値")
    print("[INFO] 離陸/着陸: T / G,  自動接近: R 切替,  終了: ESC")

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

            # 自動接近 切替（main.py準拠）
            if keyboard.is_pressed('r'):
                _wait_release('r')
                with state_lock:
                    state["autopilot"] = not bool(state["autopilot"])
                    print(f"[AUTO] AutoApproach -> {state['autopilot']}")

            # 手動操縦（main.py準拠）
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
            with state_lock:
                state["manual_active"] = bool(manual_active)

            if manual_active:
                with state_lock:
                    state["vx"], state["vy"], state["vz"], state["yaw"] = vx, vy, vz, yaw
            else:
                with state_lock:
                    if not bool(state["autopilot"]):
                        state["vx"] = state["vy"] = state["vz"] = state["yaw"] = 0

            # 非常停止
            if keyboard.is_pressed('space'):
                with state_lock:
                    state["vx"] = state["vy"] = state["vz"] = state["yaw"] = 0

            # 全終了
            if keyboard.is_pressed('esc'):
                print("[ESC] detected. stopping ...")
                break

            time.sleep(0.01)

    finally:
        with state_lock:
            state["running"] = False

        try:
            cv2.destroyWindow(WINDOW_NAME)
        except Exception:
            pass
        try:
            ctrl.tello.send_rc_control(0, 0, 0, 0)
        except Exception:
            pass
        try:
            ctrl.land()
        except Exception:
            pass
        try:
            tools.stop_overlay()
        except Exception:
            pass
        ctrl.cleanup()
        print("[DONE] cleanup completed.")


if __name__ == "__main__":
    main()
