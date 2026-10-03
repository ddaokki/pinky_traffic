"""테이프 트랙 시뮬레이터 (ROS 없이 돈다).

바닥에 테이프로 깐 타원 코스를 위에서 본 그림(ground)으로 만들고,
Pinky 카메라 위치/각도에서 본 영상으로 투시 변환(수업 40번 자료 warpPerspective)한다.
같은 방법으로 '정답 라벨' 그림도 만들 수 있어서 YOLO 학습용 합성 데이터도 나온다.

단위 m. 로봇은 반시계 방향으로 돈다 -> 안쪽 선이 left, 바깥쪽 선이 right.
"""
import math
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np

# 라벨 그림의 값
L_NONE, L_LEFT, L_RIGHT, L_CROSSWALK = 0, 1, 2, 3


@dataclass
class TrackSpec:
    length: float = 2.4            # 코스 바깥 크기 (중심선 기준) x
    width: float = 1.6             # y
    radius: float = 0.45           # 모서리 반지름 (중심선)
    lane_width: float = 0.22       # 두 테이프 안쪽 사이 거리
    tape_width: float = 0.024      # 테이프 폭
    crosswalk_s: List[float] = field(default_factory=lambda: [0.9])   # 중심선 따라 횡단보도 시작 위치(m)
    crosswalk_len: float = 0.16    # 진행방향 길이
    stripe_width: float = 0.024
    stripes: int = 4
    floor_bgr: tuple = (95, 100, 105)
    lane_bgr: tuple = (235, 235, 235)        # 흰 테이프
    crosswalk_bgr: tuple = (40, 210, 235)    # 노란 테이프 (차선과 같은 색으로 하려면 lane_bgr 과 같게)
    ppm: int = 400                 # 그림 해상도 px/m
    margin: float = 0.5
    noise: float = 6.0             # 바닥 무늬/조명 얼룩 세기


@dataclass
class CameraSpec:
    width: int = 320
    height: int = 240
    hfov: float = 1.1519           # Pinky Pro URDF (rad)
    cam_x: float = 0.035           # base 기준 앞쪽
    cam_z: float = 0.065           # 바닥에서 높이
    tilt_deg: float = 8.0          # 아래로 숙인 각 (bringup_sim 의 cam_tilt_deg 기본값)


class Track:
    def __init__(self, spec: TrackSpec = None, seed=0):
        self.spec = spec or TrackSpec()
        s = self.spec
        self.rng = np.random.default_rng(seed)
        self._build_centerline()
        self.x0 = -s.margin
        self.y1 = s.width + s.margin
        self.size = (int((s.length + 2 * s.margin) * s.ppm), int((s.width + 2 * s.margin) * s.ppm))  # (w, h)
        self.ground, self.labels = self._render()

    # ---------- 중심선 ----------
    def _build_centerline(self, step=0.005):
        s = self.spec
        r, L, W = s.radius, s.length, s.width
        pts = []

        def line(a, b):
            n = max(2, int(math.dist(a, b) / step))
            for i in range(n):
                t = i / n
                pts.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))

        def arc(c, a0, a1):
            n = max(2, int(abs(a1 - a0) * r / step))
            for i in range(n):
                a = a0 + (a1 - a0) * i / n
                pts.append((c[0] + r * math.cos(a), c[1] + r * math.sin(a)))

        # 아래 변에서 +x 방향으로 출발, 반시계
        line((r, 0), (L - r, 0)); arc((L - r, r), -math.pi / 2, 0)
        line((L, r), (L, W - r)); arc((L - r, W - r), 0, math.pi / 2)
        line((L - r, W), (r, W)); arc((r, W - r), math.pi / 2, math.pi)
        line((0, W - r), (0, r)); arc((r, r), math.pi, 1.5 * math.pi)
        self.pts = np.array(pts)
        d = np.diff(np.vstack([self.pts, self.pts[:1]]), axis=0)
        seg = np.hypot(d[:, 0], d[:, 1])
        self.s = np.concatenate([[0], np.cumsum(seg)[:-1]])
        self.total = float(seg.sum())
        self.theta = np.arctan2(d[:, 1], d[:, 0])

    def pose_at(self, s, lateral=0.0, dtheta=0.0):
        """중심선 위 s(m) 지점의 (x, y, yaw). lateral>0 = 왼쪽으로 치우침."""
        i = int(np.searchsorted(self.s, s % self.total, side='right') - 1)
        x, y = self.pts[i]
        th = self.theta[i]
        return (x - lateral * math.sin(th), y + lateral * math.cos(th), th + dtheta)

    def nearest(self, x, y):
        """(가장 가까운 중심선 인덱스의 s, 부호있는 횡방향 오차[왼쪽 +])."""
        d2 = (self.pts[:, 0] - x) ** 2 + (self.pts[:, 1] - y) ** 2
        i = int(np.argmin(d2))
        th = self.theta[i]
        lateral = -(x - self.pts[i, 0]) * math.sin(th) + (y - self.pts[i, 1]) * math.cos(th)
        return float(self.s[i]), float(lateral)

    # ---------- 그리기 ----------
    def to_px(self, xy):
        xy = np.asarray(xy, dtype=float)
        return np.stack([(xy[..., 0] - self.x0) * self.spec.ppm, (self.y1 - xy[..., 1]) * self.spec.ppm], axis=-1)

    def _offset_line(self, lateral):
        nx, ny = -np.sin(self.theta), np.cos(self.theta)
        return np.stack([self.pts[:, 0] + lateral * nx, self.pts[:, 1] + lateral * ny], axis=1)

    def _render(self):
        s = self.spec
        w, h = self.size
        ground = np.empty((h, w, 3), np.uint8)
        ground[:] = s.floor_bgr
        if s.noise > 0:
            # 바닥 얼룩: 저해상도 잡음을 키워서 더한다 (조명 얼룩 흉내)
            blotch = cv2.resize(self.rng.normal(0, s.noise, (h // 40 + 1, w // 40 + 1)).astype(np.float32), (w, h))
            grain = self.rng.normal(0, s.noise * 0.5, (h, w)).astype(np.float32)
            ground = np.clip(ground.astype(np.float32) + (blotch + grain)[..., None], 0, 255).astype(np.uint8)
        labels = np.zeros((h, w), np.uint8)
        tape = max(1, int(round(s.tape_width * s.ppm)))
        off = s.lane_width / 2 + s.tape_width / 2
        for lateral, label in ((off, L_LEFT), (-off, L_RIGHT)):
            poly = self.to_px(self._offset_line(lateral)).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(ground, [poly], True, s.lane_bgr, tape, cv2.LINE_AA)
            cv2.polylines(labels, [poly], True, int(label), tape)
        for s0 in s.crosswalk_s:
            for k in range(s.stripes):
                lat = (k + 0.5) / s.stripes * s.lane_width - s.lane_width / 2
                a = self.pose_at(s0, lat)[:2]
                b = self.pose_at(s0 + s.crosswalk_len, lat)[:2]
                pa, pb = self.to_px(a).astype(int), self.to_px(b).astype(int)
                sw = max(1, int(round(s.stripe_width * s.ppm)))
                cv2.line(ground, tuple(pa), tuple(pb), s.crosswalk_bgr, sw, cv2.LINE_AA)
                cv2.line(labels, tuple(pa), tuple(pb), int(L_CROSSWALK), sw)
        return ground, labels

    def crosswalk_distance(self, s):
        """현재 s 에서 다음 횡단보도 시작까지 남은 거리 (m)."""
        return min(((c - s) % self.total) for c in self.spec.crosswalk_s) if self.spec.crosswalk_s else float('inf')


class Camera:
    """바닥 평면 -> 카메라 영상 호모그래피."""

    def __init__(self, track: Track, spec: CameraSpec = None):
        self.track = track
        self.spec = spec or CameraSpec()
        c = self.spec
        self.f = (c.width / 2) / math.tan(c.hfov / 2)
        self.K = np.array([[self.f, 0, c.width / 2], [0, self.f, c.height / 2], [0, 0, 1]])
        th = math.radians(c.tilt_deg)
        # 로봇좌표(x 앞, y 왼쪽, z 위) -> 카메라좌표(x 오른쪽, y 아래, z 앞)
        self.R_cr = np.array([[0, -1, 0], [-math.sin(th), 0, -math.cos(th)], [math.cos(th), 0, -math.sin(th)]])
        self.horizon = int(c.height / 2 - self.f * math.tan(th))
        ppm = track.spec.ppm
        # 바닥그림 px -> 세계 m
        self.A = np.array([[1 / ppm, 0, track.x0], [0, -1 / ppm, track.y1], [0, 0, 1]])

    def homography(self, x, y, yaw):
        c = self.spec
        R_rw = np.array([[math.cos(yaw), math.sin(yaw), 0], [-math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        M = self.R_cr @ R_rw
        t = -M @ np.array([x, y, 0.0]) - self.R_cr @ np.array([c.cam_x, 0, c.cam_z])
        H_world = self.K @ np.column_stack([M[:, 0], M[:, 1], t])
        return H_world @ self.A

    def render(self, x, y, yaw, wall_bgr=(150, 150, 150)):
        c = self.spec
        H = self.homography(x, y, yaw)
        img = cv2.warpPerspective(self.track.ground, H, (c.width, c.height), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=self.track.spec.floor_bgr)
        cut = max(0, self.horizon + 6)
        img[:cut] = wall_bgr       # 지평선 위 = 벽
        return img

    def render_labels(self, x, y, yaw):
        c = self.spec
        H = self.homography(x, y, yaw)
        lab = cv2.warpPerspective(self.track.labels, H, (c.width, c.height), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        lab[: max(0, self.horizon + 6)] = 0
        return lab
