# utils/ctrl.py
from __future__ import annotations

import time
import threading
from typing import Optional, Tuple

import cv2
import keyboard
from djitellopy import Tello

DRONE_SPEED = 30                 # 既定値（cm/s）
DRONE_UPPER_LIMIT = 200
DRONE_LOWER_LIMIT = 5
DRONE_UP_SPEED = 30
DRONE_DOWN_SPEED = 30
DRONE_BATTERY_THRESHOLD = 30

SPEED_MIN = 5
SPEED_MAX = 100
SPEED_STEP = 5

class CTRL:
    DRONE_SPEED = 30                 # 既定値（cm/s）
    DRONE_UPPER_LIMIT = 200
    DRONE_LOWER_LIMIT = 3
    DRONE_UP_SPEED = 30
    DRONE_DOWN_SPEED = 30
    DRONE_BATTERY_THRESHOLD = 30

    SPEED_MIN = 5
    SPEED_MAX = 100
    SPEED_STEP = 5

    def __init__(
        self,
        speed: int = DRONE_SPEED,
        max_height: int = DRONE_UPPER_LIMIT,
        min_height: int = DRONE_LOWER_LIMIT,
        up_speed: int = DRONE_UP_SPEED,
        down_speed: int = DRONE_DOWN_SPEED,
        battery_threshold: int = DRONE_BATTERY_THRESHOLD,
        auto_connect: bool = True,
        auto_stream: bool = True,
    ):
        self.s = int(speed)
        self.u = int(up_speed)
        self.d = int(down_speed)
        self.MAX_HEIGHT = int(max(0, max_height))
        self.MIN_HEIGHT = int(max(0, min_height))
        self.BATTERY_THRESHOLD = int(max(0, battery_threshold))
        self.tello = Tello()
        if auto_connect:
            self._connect_and_check()
        if auto_stream:
            self._start_stream()

    # ------- 内部ユーティリティ -------

    def _connect_and_check(self) -> None:
        try:
            print("Attempting to connect to Tello...")
            self.tello.connect()
            bat = self.tello.get_battery()
            time.sleep(0.2)
            if bat is None:
                raise RuntimeError("Failed to read battery status.")
            print(f"Tello connected. Battery: {bat}%")
            time.sleep(0.2)
            if bat < self.BATTERY_THRESHOLD:
                print(f"[Warning] Battery below threshold ({bat}% < {self.BATTERY_THRESHOLD}%).")
        except Exception as e:
            print(f"Connection Error: {e}")
            print("Ensure your PC is connected to the Tello Wi-Fi (SSID: TELLO-xxxxxx).")
            raise

    def _start_stream(self) -> None:
        try:
            self.tello.streamoff()
        except Exception:
            pass
        try:
            self.tello.streamon()
            print("Camera stream started.")
            time.sleep(0.3)
        except Exception as e:
            print(f"Failed to start camera stream: {e}")
            raise

    @staticmethod
    def _valid_height(h: Optional[int]) -> Optional[int]:
        if h is None:
            return None
        if isinstance(h, int):
            return h if h >= 0 else None
        try:
            return int(h)
        except Exception:
            return None

    # ========= 速度制御API =========
    @property
    def speed(self) -> int:
        return self.s

    @speed.setter
    def speed(self, value: int) -> None:
        self.s = max(self.SPEED_MIN, min(self.SPEED_MAX, int(value)))
        print(f"[CTRL] speed -> {self.s}")

    def set_speed(self, value: int) -> None:
        self.speed = value

    def inc_speed(self, step: int | None = None) -> int:
        if step is None:
            step = self.SPEED_STEP
        self.speed = self.s + int(step)
        return self.s

    def dec_speed(self, step: int | None = None) -> int:
        if step is None:
            step = self.SPEED_STEP
        self.speed = self.s - int(step)
        return self.s

    def ctrl_forward(self):
        self.tello.send_rc_control(0, self.s, 0, 0)

    def ctrl_back(self):
        self.tello.send_rc_control(0, -self.s, 0, 0)

    def ctrl_left(self):
        self.tello.send_rc_control(-self.s, 0, 0, 0)

    def ctrl_right(self):
        self.tello.send_rc_control(self.s, 0, 0, 0)

    def ctrl_up(self):
        h = self._valid_height(self.tello.get_height())
        if h is None:
            print("Height unknown. Sending up command cautiously.")
            self.tello.send_rc_control(0, 0, self.u, 0)
            return
        if h < self.MAX_HEIGHT:
            self.tello.send_rc_control(0, 0, self.u, 0)
        else:
            print(f"Cannot go higher (current: {h} cm, limit: {self.MAX_HEIGHT} cm).")

    def ctrl_down(self):
        h = self._valid_height(self.tello.get_height())
        if h is None:
            print("Height unknown. Sending down command cautiously.")
            self.tello.send_rc_control(0, 0, -self.d, 0)
            return
        if h > self.MIN_HEIGHT:
            self.tello.send_rc_control(0, 0, -self.d, 0)
        else:
            print(f"Cannot go lower (current: {h} cm, limit: {self.MIN_HEIGHT} cm).")

    def ctrl_yaw(self, yaw_rc: int):
        yaw_rc = max(-100, min(100, int(yaw_rc)))
        self.tello.send_rc_control(0, 0, 0, yaw_rc)

    def ctrl_hover(self):
        self.tello.send_rc_control(0, 0, 0, 0)

    def takeoff(self):
        try:
            self.tello.takeoff()
        except Exception as e:
            print(f"Takeoff failed: {e}")

    def land(self):
        try:
            self.tello.land()
        except Exception as e:
            print(f"Land failed: {e}")

    def cleanup(self):
        try:
            print("Cleaning up resources...")
            h = self._valid_height(self.tello.get_height())
            if h is not None and h > 0:
                try:
                    self.tello.land()
                except Exception:
                    pass
            try:
                self.tello.streamoff()
            except Exception:
                pass
        finally:
            try:
                self.tello.end()
            except Exception:
                pass
            print("Drone resources released.")


class Tools:
    def __init__(self, tello: Tello) -> None:
        self.tello = tello
        self._height_display_thread: Optional[threading.Thread] = None

    # 既存のHUD（省略）…

    # ======== 新規：検出オーバーレイHUD ========
    def show_stream_with_detection(
        self,
        ctrl,                # CTRL インスタンス
        model,               # utils.model.Model インスタンス
        window_name: str = "Tello DET",
        resize_fx: float = 0.9,
        resize_fy: float = 0.9,
        hud_pos: Tuple[int, int] = (10, 10),
        hud_width: int = 320,
        hud_height: int = 140,
        font_scale: float = 0.6,
        thickness: int = 1,
        fps_estimate: bool = True,
        quit_key: str = "q",
        conf: float = 0.5,
    ):
        """
        カメラストリームにSSDの検出結果を重畳し、バッテリー/高度/速度/FPSのHUDを付与して表示。
        """
        self._overlay_running = True
        prev_t, avg_fps = time.time(), 0.0

        try:
            fr = self.tello.get_frame_read()
            while self._overlay_running:
                frame_bgr = fr.frame
                if frame_bgr is None:
                    if cv2.waitKey(1) & 0xFF == ord(quit_key):
                        break
                    continue

                # 物体検出を適用（BGRのまま）
                frame_bgr = model.annotate_frame(frame_bgr, conf=conf)

                # リサイズ
                frame_bgr = cv2.resize(frame_bgr, None, fx=resize_fx, fy=resize_fy, interpolation=cv2.INTER_AREA)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # ステータス取得
                try:  bat = self.tello.get_battery()
                except: bat = "N/A"
                try:  alt = self.tello.get_height()
                except: alt = "N/A"
                spd = getattr(ctrl, "speed", "N/A")

                # FPS推定
                if fps_estimate:
                    now = time.time()
                    dt = max(1e-6, now - prev_t)
                    fps = 1.0 / dt
                    avg_fps = 0.9 * avg_fps + 0.1 * fps if avg_fps > 0 else fps
                    prev_t = now

                # 半透明HUD
                x, y = hud_pos
                w, h = hud_width, hud_height
                overlay = frame_rgb.copy()
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
                frame_rgb = cv2.addWeighted(overlay, 0.35, frame_rgb, 0.65, 0.0)

                def put_line(text, line_idx):
                    cv2.putText(frame_rgb, text, (x + 12, y + 25 + 24 * line_idx),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

                put_line(f"Battery: {bat}%", 0)
                put_line(f"Altitude: {alt} cm", 1)
                put_line(f"Speed: {spd}", 2)
                if fps_estimate:
                    put_line(f"FPS: {avg_fps:.1f}", 3)
                put_line(f"[{quit_key.upper()}] to quit", 4)

                cv2.imshow(window_name, frame_rgb)
                if cv2.waitKey(1) & 0xFF == ord(quit_key):
                    break

        finally:
            try:
                cv2.destroyWindow(window_name)
            except Exception:
                pass
            self._overlay_running = False

    def stop_overlay(self):
        self._overlay_running = False
