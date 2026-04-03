from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        """3x3 内部パラメータ行列（float32）"""
        K = np.array([[self.fx, 0.0,     self.cx],
                      [0.0,     self.fy, self.cy],
                      [0.0,     0.0,     1.0   ]], dtype=np.float32)
        return K

    @staticmethod
    def from_fov(width: int, height: int,
                 fov_deg_h: Optional[float] = None,
                 fov_deg_v: Optional[float] = None) -> Intrinsics:
        """
        画角(FOV)から焦点距離を推定して Intrinsics を作成。
        水平/垂直いずれか一方のFOVだけでも良い（両方あれば両座標で整合性を取る）。
        """
        cx = (width - 1) * 0.5
        cy = (height - 1) * 0.5

        fx = fy = None
        if fov_deg_h is not None:
            fx = (width) / (2.0 * math.tan(math.radians(fov_deg_h) * 0.5))
        if fov_deg_v is not None:
            fy = (height) / (2.0 * math.tan(math.radians(fov_deg_v) * 0.5))

        if fx is None and fy is None:
            raise ValueError("fov_deg_h または fov_deg_v の少なくとも一方が必要です。")

        # 片方しか無ければ同値とみなす（正方ピクセル仮定）
        if fx is None and fy is not None:
            fx = fy
        if fy is None and fx is not None:
            fy = fx

        return Intrinsics(float(fx), float(fy), float(cx), float(cy), int(width), int(height))

    def fov_deg(self) -> Tuple[float, float]:
        """現在の fx, fy から水平/垂直のFOV(deg)を返す。"""
        fov_h = math.degrees(2.0 * math.atan((self.width) / (2.0 * self.fx)))
        fov_v = math.degrees(2.0 * math.atan((self.height) / (2.0 * self.fy)))
        return (fov_h, fov_v)


@dataclass(frozen=True)
class Pose:
    """ワールド→カメラの外部パラメータ（R: 3x3, t: 3, float32）"""
    R: np.ndarray  # (3,3)
    t: np.ndarray  # (3,)

    @staticmethod
    def identity() -> Pose:
        return Pose(np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))

    @staticmethod
    def from_euler_xyz(tx: float, ty: float, tz: float,
                       rx_deg: float, ry_deg: float, rz_deg: float) -> Pose:
        """平行移動(tx,ty,tz) と XYZオイラー角(deg)から Pose を構築（Rz*Ry*Rxの順に適用）。"""
        rx = math.radians(rx_deg)
        ry = math.radians(ry_deg)
        rz = math.radians(rz_deg)

        Rx = np.array([[1, 0, 0],
                       [0, math.cos(rx), -math.sin(rx)],
                       [0, math.sin(rx),  math.cos(rx)]], dtype=np.float32)
        Ry = np.array([[ math.cos(ry), 0, math.sin(ry)],
                       [0,             1, 0],
                       [-math.sin(ry), 0, math.cos(ry)]], dtype=np.float32)
        Rz = np.array([[math.cos(rz), -math.sin(rz), 0],
                       [math.sin(rz),  math.cos(rz), 0],
                       [0,             0,            1]], dtype=np.float32)

        R = (Rz @ Ry @ Rx).astype(np.float32)
        t = np.array([tx, ty, tz], dtype=np.float32)
        return Pose(R, t)

    def compose(self, other: Pose) -> Pose:
        """自分の後に other を適用（Poseの合成）。"""
        R = (other.R @ self.R).astype(np.float32)
        t = (other.R @ self.t + other.t).astype(np.float32)
        return Pose(R, t)


class PinholeCamera:
    """
    歪みなしピンホールカメラモデル（右手系, z前向き想定）。
    ・project_world: P_world(3D) → 画素座標(uv)
    ・project_camera: P_cam(3D) → 画素座標(uv)
    ・unproject: 画素(uv) + Z(奥行き, カメラ座標系) → P_cam(3D)
    ・rays_from_pixels: 画素(uv) → カメラ座標系の単位視線ベクトル
    """

    def __init__(self, intr: Intrinsics, pose_w2c: Pose = Pose.identity()):
        self.intr = intr
        self.pose_w2c = pose_w2c  # World → Camera

    # ----- 内部: 正規化座標 ↔ ピクセル -----
    def pixels_to_normalized(self, uv: np.ndarray) -> np.ndarray:
        """
        (N,2) のピクセル座標 → (N,2) の正規化座標（x'= (u-cx)/fx, y'=(v-cy)/fy）
        """
        uv = np.asarray(uv, dtype=np.float32)
        x = (uv[..., 0] - self.intr.cx) / self.intr.fx
        y = (uv[..., 1] - self.intr.cy) / self.intr.fy
        return np.stack([x, y], axis=-1)

    def normalized_to_pixels(self, xy: np.ndarray) -> np.ndarray:
        """
        (N,2) の正規化座標 → (N,2) のピクセル座標
        """
        xy = np.asarray(xy, dtype=np.float32)
        u = xy[..., 0] * self.intr.fx + self.intr.cx
        v = xy[..., 1] * self.intr.fy + self.intr.cy
        return np.stack([u, v], axis=-1)

    # ----- 3D→2D 投影 -----
    def project_camera(self, P_cam: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        カメラ座標系の3D点 (N,3) をピクセル座標 (N,2) に投影。
        Z<=0（背面）には注意。返り値は可視範囲チェックなし。
        """
        P = np.asarray(P_cam, dtype=np.float32)
        Z = np.clip(P[..., 2], eps, None)
        x = P[..., 0] / Z
        y = P[..., 1] / Z
        return self.normalized_to_pixels(np.stack([x, y], axis=-1))

    def project_world(self, P_world: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        ワールド座標の3D点 (N,3) をピクセル座標 (N,2) に投影。
        """
        Pw = np.asarray(P_world, dtype=np.float32)
        Pc = (Pw @ self.pose_w2c.R.T + self.pose_w2c.t).astype(np.float32)
        return self.project_camera(Pc, eps=eps)

    # ----- 2D(+depth)→3D 逆投影 -----
    def unproject(self, uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """
        ピクセル (N,2) とカメラ座標系での Z(奥行き) (N,) から、カメラ座標系3D点 (N,3) を再構成。
        depth は Z>0 を仮定。
        """
        uv = np.asarray(uv, dtype=np.float32)
        depth = np.asarray(depth, dtype=np.float32).reshape(-1)
        xy = self.pixels_to_normalized(uv)  # (N,2)
        X = xy[:, 0] * depth
        Y = xy[:, 1] * depth
        Z = depth
        return np.stack([X, Y, Z], axis=-1).astype(np.float32)

    # ----- 視線ベクトル -----
    def rays_from_pixels(self, uv: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        ピクセル (N,2) から、カメラ座標系の視線方向 (N,3) を返す（原点はカメラ中心）。
        normalize=True なら単位ベクトル。
        """
        xy = self.pixels_to_normalized(uv)
        dirs = np.concatenate([xy, np.ones((xy.shape[0], 1), dtype=np.float32)], axis=-1)
        if normalize:
            n = np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-9
            dirs = dirs / n
        return dirs.astype(np.float32)

    # ----- OpenCV互換ヘルパ -----
    def cv_K(self) -> np.ndarray:
        """OpenCVでそのまま使える 3x3 K（float32）"""
        return self.intr.K

    def cv_RT(self) -> Tuple[np.ndarray, np.ndarray]:
        """OpenCVの projectPoints 等に渡せる (R, t)。Rは3x3, tは3x1。"""
        R = self.pose_w2c.R.astype(np.float32)
        t = self.pose_w2c.t.reshape(3, 1).astype(np.float32)
        return R, t
