# utils/tracking_continuity.py
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional, Tuple

BBOX = Tuple[float, float, float, float]  # (x1,y1,x2,y2)

@dataclass
class TrackState:
    bbox: Optional[BBOX] = None
    label: str = ""
    conf: float = 0.0

    cx_s: Optional[float] = None
    cy_s: Optional[float] = None

    vx: float = 0.0
    vy: float = 0.0

    t_last_det: float = 0.0
    t_last_upd: float = 0.0

    mode: str = "LOST"  # DETECT / HOLD / PREDICT / LOST

    def age_det(self) -> float:
        return (time.time() - self.t_last_det) if self.t_last_det > 0 else 1e9


class ContinuityController:
    """
    推論間引き + 検出欠損に対する連続性補完
      - infer_hz で推論を間引く（遅延の雪だるま回避）
      - 検出欠損時は HOLD→PREDICT で継続
      - 画面中心制御用に中心座標を EMA で平滑化
    """

    def __init__(self, infer_hz: float = 8.0, hold_sec: float = 0.5, ema_alpha: float = 0.5):
        self.infer_period = 1.0 / max(1e-6, float(infer_hz))
        self.hold_sec = float(hold_sec)
        self.alpha = float(ema_alpha)

        self.state = TrackState()
        self._t_next_infer = time.time()
        self.last_infer_dt = 0.0

    def due_infer(self) -> bool:
        return time.time() >= self._t_next_infer

    def mark_infer_done(self, infer_dt: float):
        self.last_infer_dt = float(infer_dt)
        self._t_next_infer = time.time() + self.infer_period

    def update_with_detection(self, bbox: BBOX, label: str, conf: float):
        st = self.state
        now = time.time()

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5

        if st.cx_s is None:
            st.cx_s, st.cy_s = cx, cy
            st.vx, st.vy = 0.0, 0.0
        else:
            dt = max(1e-3, now - st.t_last_upd)
            cx_prev, cy_prev = st.cx_s, st.cy_s

            a = self.alpha
            st.cx_s = a * cx + (1 - a) * st.cx_s
            st.cy_s = a * cy + (1 - a) * st.cy_s

            st.vx = (st.cx_s - cx_prev) / dt
            st.vy = (st.cy_s - cy_prev) / dt

        st.bbox = bbox
        st.label = label
        st.conf = float(conf)
        st.t_last_det = now
        st.t_last_upd = now
        st.mode = "DETECT"

    def update_no_detection(self):
        st = self.state
        now = time.time()

        if st.bbox is None or st.cx_s is None or st.cy_s is None:
            st.mode = "LOST"
            return

        age = now - st.t_last_det
        if age <= self.hold_sec:
            st.mode = "HOLD"
            return

        dt = max(1e-3, now - st.t_last_upd)
        st.cx_s += st.vx * dt
        st.cy_s += st.vy * dt
        st.t_last_upd = now
        st.mode = "PREDICT"


def bbox_area_ratio(bbox: Optional[BBOX], frame_w: int, frame_h: int) -> Optional[float]:
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    return (bw * bh) / float(max(1, frame_w * frame_h))
