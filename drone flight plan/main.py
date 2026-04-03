# main.py
from __future__ import annotations
import time
import threading
from typing import Dict, Tuple

import cv2
import numpy as np
import keyboard

from utils.ctrl import CTRL, Tools
from utils.model import Model
from utils.pinhole import Intrinsics, PinholeCamera, Pose

# =========================
# 検出ターゲット（学習済みラベル名に合わせる）
TARGET_LABEL         = "apple"

# ピンホール距離推定
APPLE_SIZE_M         = 0.08          # リンゴの代表直径[m]（7–9cm想定）
USE_BBOX_AXIS        = "height"      # "height" | "width" | "max"（どの辺長で距離推定するか）
FOV_H_DEG            = 82.0          # カメラの水平FOV[deg]（実機に合わせて調整）

# 自動接近（前進・位置合わせ）
AUTOPILOT_START_ON   = False         # 起動時に自動接近ONにするか
TARGET_DISTANCE_M    = 0.10          # 目標距離[m]（10cm）
DIST_TOL_M           = 0.08          # 目標距離の許容誤差[m]（この範囲に入ったら前後は停止）
MAX_VX               = 38            # 前後速度(±)の最大値（send_rc_control値）
KP_FORWARD           = 150.0         # 前後制御の比例ゲイン（速度=KP*距離誤差 をスケーリング）
EMA_ALPHA            = 0.35          # 距離の指数移動平均（ノイズ平滑）

# センタリング（ヨー回頭 or 横移動のどちらかで中心合わせ）
CENTERING_MODE       = "yaw"         # "yaw" | "strafe"
KP_CENTER_X          = 0.15          # 画像中心からの水平誤差(正規化)に対する比例ゲイン
MAX_YAW              = 40            # ヨー最大（±）
MAX_VY               = 30            # 横移動最大（±）

# 高度・安全
MIN_DISTANCE_M       = 0.10          # これ以下は近すぎ→前進禁止（安全）
MAX_DISTANCE_M       = 5.0           # これ以上は遠すぎ→検出外れ扱い
ALT_LIMIT_CM         = 200           # 高度上限
QUIT_KEY             = "q"           # 映像ウィンドウの終了キー

# ウィンドウ/HUD
WINDOW_NAME          = "Tello DET + RANGE (Apple)"
HUD_ALPHA            = 0.0           # HUD半透明(0=枠線のみで軽量)
RESIZE_FX, RESIZE_FY = 0.9, 0.9      # 表示時のリサイズ係数

# モデル重み
WEIGHT_PATH          = "./weights/ssd_finetuned_filterv2.pth"
CONF_THRESH          = 0.5           # 検出のしきい値

# 色（BGR）
CLR_OTHER            = (0, 255, 0)     # 対象以外：緑
CLR_TARGET           = (0, 255, 255)   # 対象クラス：黄色
CLR_IN_RANGE         = (255, 0, 0)   # 目標距離内の選択対象：青系（見やすいオレンジ/青どちらでもOKなら変更可）


# =========================
# ランタイム共有状態
# =========================
state: Dict[str, float | bool] = {
    "vx": 0.0,      # 前後 +前進 / -後退
    "vy": 0.0,      # 左右 +右   / -左
    "vz": 0.0,      # 上下 +上昇 / -下降
    "yaw": 0.0,     # 回頭 +右   / -左
    "running": True,
    "autopilot": AUTOPILOT_START_ON,
    "z_ema": None,  # 距離のEMA
}

def clamp(v: float, lo: float, hi: float) -> int:
    return int(max(lo, min(hi, v)))


# =========================
# RC連続送信スレッド
# =========================
def rc_sender(ctrl: CTRL, rate_hz: int = 20):
    period = 1.0 / float(rate_hz)
    t = ctrl.tello
    while state["running"]:
        try:

            t.send_rc_control(
                clamp(state["vy"],  -100, 100),
                clamp(state["vx"],  -100, 100),
                clamp(state["vz"],  -100, 100),
                clamp(state["yaw"], -100, 100),
            )
        except Exception as e:
            print(f"[rc_sender] send_rc_control failed: {e}")
        time.sleep(period)


# =========================
# 自動接近 + 距離推定を含む表示スレッド
# =========================
def run_det_range(ctrl: CTRL, model: Model):
    tello = ctrl.tello
    fr = tello.get_frame_read()

    # OpenCV最適化
    try:
        cv2.setUseOptimized(True)
        cv2.setNumThreads(0)
    except Exception:
        pass

    # Intrinsics は最初のフレームから決定
    intr = None
    cam = None

    # ターゲットラベルindex
    categories = list(model.voc_classes)
    try:
        target_idx = categories.index(TARGET_LABEL)
        print(f"[INFO] TARGET_LABEL='{TARGET_LABEL}' -> id={target_idx}")
    except ValueError:
        print(f"[WARN] label '{TARGET_LABEL}' がモデル内に見つかりません。距離推定と自動接近は無効化されます。")
        target_idx = None

    prev_t = time.time()
    avg_fps = 0.0

    while state["running"]:
          # フレーム更新
        frame_bgr = fr.frame
        frame_bgr= cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if frame_bgr is None:
            if (cv2.waitKey(1) & 0xFF) == ord(QUIT_KEY):
                break
            continue

        # 初回にカメラ内部パラメータを構築
        if cam is None:
            h0, w0 = frame_bgr.shape[:2]
            intr = Intrinsics.from_fov(width=w0, height=h0, fov_deg_h=FOV_H_DEG)
            cam = PinholeCamera(intr=intr, pose_w2c=Pose.identity())
            print(f"[PINHOLE] K=\n{cam.intr.K}\nFOV(h,v)={cam.intr.fov_deg()}")

        # 物体検出
        _, boxes, labels, scores = model.predict_frame(frame_bgr, conf=CONF_THRESH)
        out = frame_bgr.copy()

        # 距離推定（最も大きなリンゴbboxを採用）
        z_est = None
        best_bb: Tuple[int, int, int, int] | None = None

        for i, bb in enumerate(boxes):
            li = int(labels[i])
            x1, y1, x2, y2 = [int(v) for v in bb]
            w_px = max(1, x2 - x1)
            h_px = max(1, y2 - y1)

            # 色の決定：対象クラス=黄色、それ以外=緑
            name = categories[li] if 0 <= li < len(categories) else f"id:{li}"
            sc = float(scores[i]) if i < len(scores) else None
            color = CLR_TARGET if (target_idx is not None and li == target_idx) else CLR_OTHER

            # 一旦ラベルのみ描く（距離は後で追記）
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{name}"
            if sc is not None:
                label += f":{sc:.2f}"
            tw = max(60, 10 * len(label))
            cv2.rectangle(out, (x1, max(0, y1 - 18)), (x1 + tw, y1), color, -1)
            cv2.putText(out, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 1, cv2.LINE_AA)

            # ターゲットの距離推定
            if (target_idx is not None) and (li == target_idx):
                if USE_BBOX_AXIS == "height":
                    px_len = h_px; f_pix = cam.intr.fy
                elif USE_BBOX_AXIS == "width":
                    px_len = w_px; f_pix = cam.intr.fx
                else:
                    if w_px >= h_px: px_len, f_pix = w_px, cam.intr.fx
                    else:            px_len, f_pix = h_px, cam.intr.fy

                Z = (f_pix * APPLE_SIZE_M) / float(px_len)
                # 妥当性チェック
                if (MIN_DISTANCE_M <= Z <= MAX_DISTANCE_M):
                    # 最大のbboxを優先（近いほど大きい前提）
                    if (best_bb is None) or (w_px * h_px > (best_bb[2]-best_bb[0]) * (best_bb[3]-best_bb[1])):
                        z_est = float(Z)
                        best_bb = (x1, y1, x2, y2)

        # 距離のEMA
        if z_est is not None:
            if state["z_ema"] is None:
                state["z_ema"] = z_est
            else:
                state["z_ema"] = EMA_ALPHA * z_est + (1.0 - EMA_ALPHA) * float(state["z_ema"])

        # 手動入力中は自動接近を抑制
        manual_active = (
            keyboard.is_pressed('w') or keyboard.is_pressed('s') or
            keyboard.is_pressed('a') or keyboard.is_pressed('d') or
            keyboard.is_pressed('i') or keyboard.is_pressed('k') or
            keyboard.is_pressed('j') or keyboard.is_pressed('l') or
            keyboard.is_pressed('space')
        )

        # 自動接近（前進 + センタリング）：手動中は無効
        if state["autopilot"] and not manual_active and (z_est is not None) and (best_bb is not None):
            x1, y1, x2, y2 = best_bb
            cx = 0.5 * (x1 + x2)
            h, w = out.shape[:2]
            # 画像中心からの正規化誤差（-1..1）
            ex = (cx - (w * 0.5)) / max(1.0, w * 0.5)

            # 前後制御：目標距離との差
            z_use = float(state["z_ema"]) if state["z_ema"] is not None else z_est
            e = z_use - TARGET_DISTANCE_M
            if abs(e) <= DIST_TOL_M:
                vx_cmd = 0
            else:
                vx_cmd = int(np.clip(KP_FORWARD * e * 10.0, -MAX_VX, MAX_VX))

            # 近すぎ安全：前進禁止
            if z_use <= MIN_DISTANCE_M:
                vx_cmd = min(0, vx_cmd)

            # センタリング：ヨー or 横移動
            if CENTERING_MODE == "yaw":
                yaw_cmd = int(np.clip(KP_CENTER_X * ex * 100.0, -MAX_YAW, MAX_YAW))
                vy_cmd = 0
            else:
                vy_cmd = int(np.clip(KP_CENTER_X * ex * 100.0, -MAX_VY, MAX_VY))
                yaw_cmd = 0

            state["vx"]  = vx_cmd
            state["vy"]  = vy_cmd
            state["yaw"] = yaw_cmd
        else:
            # ここでは何もしない（手動ループ側が上書き）
            pass

        # 自動接近ブロックのすぐ後ろに追加
        if state["autopilot"] and not manual_active:
            if z_est is None or best_bb is None:
                state["vx"] = state["vy"] = state["yaw"] = 0  # 見失い時は停止


        # 選択対象（best_bb）の強調表示
        if z_est is not None and best_bb is not None:
            x1, y1, x2, y2 = best_bb
            z_use = float(state['z_ema']) if state['z_ema'] is not None else z_est
            in_range = abs(z_use - TARGET_DISTANCE_M) <= DIST_TOL_M

            # 距離が目標±許容内なら「青系（またはご指定色）」で太枠、そうでなければ黄色で太枠
            strong_color = CLR_IN_RANGE if in_range else CLR_TARGET
            cv2.rectangle(out, (x1, y1), (x2, y2), strong_color, 3)

            # 距離テキスト
            txt = f"Z~{z_use:.2f} m"
            if in_range:
                txt += " (IN RANGE)"
            cv2.putText(out, txt, (x1 + 3, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, strong_color, 2, cv2.LINE_AA)

        # HUD
        try:  bat = tello.get_battery()
        except: bat = "N/A"
        try:  alt = tello.get_height()
        except: alt = "N/A"
        spd = ctrl.speed

        now = time.time()
        dt = max(1e-6, now - prev_t)
        fps = 1.0 / dt
        avg_fps = 0.9 * avg_fps + 0.1 * fps if avg_fps > 0 else fps
        prev_t = now

        x, y, w_hud, h_hud = 10, 10, 430, 160
        if HUD_ALPHA > 0.0:
            overlay = out.copy()
            cv2.rectangle(overlay, (x, y), (x + w_hud, y + h_hud), (0, 0, 0), -1)
            out = cv2.addWeighted(overlay, HUD_ALPHA, out, 1.0 - HUD_ALPHA, 0.0)
        else:
            cv2.rectangle(out, (x, y), (x + w_hud, y + h_hud), (0, 0, 0), 1)

        def put_line(text: str, row: int):
            cv2.putText(out, text, (x + 12, y + 25 + 24 * row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        ap = "ON" if state["autopilot"] else "OFF"
        z_text = f"{float(state['z_ema']):.2f} m" if state["z_ema"] is not None else "-"
        put_line(f"AutoApproach: {ap}  (R=toggle)", 0)
        put_line(f"Est.Range: {z_text}  (Target: {TARGET_DISTANCE_M:.2f} m)", 1)
        put_line(f"Battery: {bat}%  Alt: {alt} cm", 2)
        put_line(f"Speed: {spd}  FPS: {avg_fps:.1f}", 3)
        put_line(f"[{QUIT_KEY.upper()}] to quit", 4)

        # リサイズ＆表示（BGRのまま表示で軽量）
        out = cv2.resize(out, None, fx=RESIZE_FX, fy=RESIZE_FY, interpolation=cv2.INTER_AREA)
        cv2.imshow(WINDOW_NAME, out)
        if (cv2.waitKey(1) & 0xFF) == ord(QUIT_KEY):
            break

    try:
        cv2.destroyWindow(WINDOW_NAME)
    except:
        pass


# =========================
# メインループ（キーボード手動 + 自動接近切替）
# =========================
def main():
    ctrl = CTRL()
    tools = Tools(ctrl.tello)
    model = Model(weight_path=WEIGHT_PATH)

    # 起動ログ（デバイス確認）
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

    # 検出＋距離推定＋自動接近スレッド
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
                state["autopilot"] = not bool(state["autopilot"])
                print(f"[AUTO] AutoApproach -> {state['autopilot']}")

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
            if manual_active:
                state["vx"], state["vy"], state["vz"], state["yaw"] = vx, vy, vz, yaw
            elif not state["autopilot"]:
                state["vx"] = state["vy"] = state["vz"] = state["yaw"] = 0

            # 非常停止
            if keyboard.is_pressed('space'):
                state["vx"] = state["vy"] = state["vz"] = state["yaw"] = 0

            # 全終了
            if keyboard.is_pressed('esc'):
                print("[ESC] detected. stopping ...")
                break

            time.sleep(0.01)
    finally:
        state["running"] = False
        try: cv2.destroyWindow(WINDOW_NAME)
        except: pass
        try: ctrl.tello.send_rc_control(0, 0, 0, 0)
        except: pass
        try: ctrl.land()
        except: pass
        try: tools.stop_overlay()
        except: pass
        ctrl.cleanup()
        print("[DONE] cleanup completed.")


def _wait_release(key: str, step_sleep: float = 0.02):
    while keyboard.is_pressed(key):
        time.sleep(step_sleep)


if __name__ == "__main__":
    main()
