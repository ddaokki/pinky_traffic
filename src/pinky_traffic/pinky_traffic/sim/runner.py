"""닫힌 루프 시뮬레이션: 카메라 그림 -> 검출 -> 제어 -> 로봇 운동 -> 다시 카메라.

로봇 1대/2대를 같은 코스에 올려 한 바퀴 돌리고 수치(metrics)를 낸다.
테스트(test/)와 도구(tools/run_sim.py)가 같이 쓴다.
"""
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..core.config import Config
from ..core.controller import LaneController, STOP, CROSSING, BLOCKED, LOST
from ..core.coordinator import LockManager, LocalLock
from ..core.detectors import make_detector
from ..core.perception import draw_debug
from .track import Track, Camera, TrackSpec, CameraSpec

ROBOT_RADIUS = 0.06      # Pinky 대략 반지름 (충돌 판정용)


@dataclass
class SimRobot:
    name: str
    x: float
    y: float
    yaw: float
    v: float = 0.0
    w: float = 0.0
    max_acc: float = 0.6        # m/s^2
    max_wacc: float = 8.0       # rad/s^2
    delay_steps: int = 1        # 영상/명령 지연 (프레임)
    queue: list = field(default_factory=list)

    def command(self, v, w):
        self.queue.append((v, w))

    def step(self, dt):
        if len(self.queue) > self.delay_steps:
            tv, tw = self.queue.pop(0)
            self.v += max(-self.max_acc * dt, min(self.max_acc * dt, tv - self.v))
            self.w += max(-self.max_wacc * dt, min(self.max_wacc * dt, tw - self.w))
        self.yaw += self.w * dt
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt


def front_distance(me: SimRobot, others, half_angle_deg):
    """라이다 흉내: 전방 부채꼴 안에서 가장 가까운 다른 로봇 표면까지 거리."""
    best = None
    for o in others:
        dx, dy = o.x - me.x, o.y - me.y
        dist = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx) - me.yaw
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))
        if abs(math.degrees(bearing)) <= half_angle_deg:
            d = max(0.0, dist - ROBOT_RADIUS)
            best = d if best is None else min(best, d)
    return best


class Simulation:
    def __init__(self, cfg: Config = None, track: Track = None, n_robots=1, starts=None, fps=15.0,
                 camera: CameraSpec = None, use_coordinator=None, seed=0, cfgs=None):
        self.track = track or Track(TrackSpec(), seed=seed)
        self.camera = Camera(self.track, camera or CameraSpec())
        self.dt = 1.0 / fps
        self.t = 0.0
        self.manager = LockManager(lease_sec=4.0, clock=lambda: self.t)
        starts = starts or [0.15 + i * -0.45 for i in range(n_robots)]   # 뒤차는 0.45 m 뒤에서 출발
        self.robots, self.controllers, self.detectors, self.cfgs = [], [], [], []
        for i in range(n_robots):
            c = (cfgs[i] if cfgs else None) or Config(**(cfg.to_dict() if cfg else {}))
            if use_coordinator is not None:
                c.use_coordinator = use_coordinator
            name = f'pinky{i + 1}'
            x, y, yaw = self.track.pose_at(starts[i])
            self.robots.append(SimRobot(name, x, y, yaw))
            self.cfgs.append(c)
            self.detectors.append(make_detector(c, model=False))   # 그림 코스라 YOLO 는 안 쓴다 (색만)
            self.controllers.append(LaneController(c, LocalLock(self.manager, name), name))
        self.log = {r.name: [] for r in self.robots}
        self.frames = {}
        self.min_gap = float('inf')
        self.collisions = 0
        self.stops = {r.name: [] for r in self.robots}   # 횡단보도 정지 기록 (남은거리, 정지시간)
        self._stop_t = {}
        self.both_in_crossing = 0

    def start(self):
        for c in self.controllers:
            c.start(self.t)

    def step(self, draw=False):
        crossing_now = 0
        for i, (robot, ctrl, det) in enumerate(zip(self.robots, self.controllers, self.detectors)):
            img = self.camera.render(robot.x, robot.y, robot.yaw)
            p, masks, small = det.detect(img)
            others = [o for o in self.robots if o is not robot]
            front = front_distance(robot, others, self.cfgs[i].front_angle_deg) if others else None
            prev = ctrl.state
            cmd = ctrl.step(p, front, self.t)
            robot.command(cmd.v, cmd.w)
            s, lateral = self.track.nearest(robot.x, robot.y)
            if cmd.state == STOP and prev != STOP:
                self._stop_t[robot.name] = (self.t, self.track.crosswalk_distance(s))
            if prev == STOP and cmd.state != STOP and robot.name in self._stop_t:
                t0, remain = self._stop_t.pop(robot.name)
                self.stops[robot.name].append({'remain_m': round(remain, 3), 'stopped_s': round(self.t - t0, 2)})
            crossing_now += cmd.state == CROSSING
            self.log[robot.name].append({'t': round(self.t, 3), 's': s, 'lateral': lateral, 'v': robot.v,
                                         'w': robot.w, 'state': cmd.state, 'offset': p.offset, 'ok': p.ok,
                                         'front': front})
            if draw:
                text = f'{robot.name} {cmd.state} v={cmd.v:.2f} w={cmd.w:+.2f} off={p.offset:+.2f}'
                self.frames[robot.name] = draw_debug(small, p, masks, text)
        if crossing_now >= 2:
            self.both_in_crossing += 1
        for robot in self.robots:
            robot.step(self.dt)
        for a in range(len(self.robots)):
            for b in range(a + 1, len(self.robots)):
                gap = math.hypot(self.robots[a].x - self.robots[b].x, self.robots[a].y - self.robots[b].y)
                self.min_gap = min(self.min_gap, gap)
                self.collisions += gap < 2 * ROBOT_RADIUS
        self.t += self.dt

    def run(self, seconds, draw=False, on_step=None):
        self.start()
        for _ in range(int(seconds / self.dt)):
            self.step(draw)
            if on_step:
                on_step(self)
        return self.metrics()

    def metrics(self):
        out = {'time_s': round(self.t, 2), 'collisions': int(self.collisions),
               'min_gap_m': None if self.min_gap == float('inf') else round(self.min_gap, 3),
               'both_in_crossing_steps': self.both_in_crossing, 'robots': {}}
        for robot, ctrl in zip(self.robots, self.controllers):
            rows = self.log[robot.name]
            lat = np.array([r['lateral'] for r in rows]) if rows else np.zeros(1)
            ds = np.diff([r['s'] for r in rows]) if len(rows) > 1 else np.zeros(1)
            ds = np.where(ds < -self.track.total / 2, ds + self.track.total, ds)
            states = [r['state'] for r in rows]
            out['robots'][robot.name] = {
                'distance_m': round(float(np.clip(ds, 0, None).sum()), 2),
                'laps': round(float(np.clip(ds, 0, None).sum() / self.track.total), 2),
                'max_abs_lateral_m': round(float(np.abs(lat).max()), 3),
                'rms_lateral_m': round(float(np.sqrt((lat ** 2).mean())), 3),
                'lost_steps': states.count(LOST),
                'blocked_steps': states.count(BLOCKED),
                'crossings': ctrl.crossings,
                'stops': self.stops[robot.name],
                'final_state': ctrl.state,
            }
        return out

    def topview(self, scale=0.35):
        """위에서 본 코스 + 로봇 위치 그림."""
        img = cv2.resize(self.track.ground, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        colors = [(60, 60, 255), (255, 140, 0), (0, 200, 0)]
        for i, robot in enumerate(self.robots):
            px = (self.track.to_px((robot.x, robot.y)) * scale).astype(int)
            tip = (self.track.to_px((robot.x + 0.12 * math.cos(robot.yaw), robot.y + 0.12 * math.sin(robot.yaw))) * scale).astype(int)
            cv2.circle(img, tuple(px), max(3, int(ROBOT_RADIUS * self.track.spec.ppm * scale)), colors[i % 3], -1)
            cv2.line(img, tuple(px), tuple(tip), (255, 255, 255), 2)
            cv2.putText(img, f'{robot.name}:{self.controllers[i].state}', (px[0] + 10, px[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[i % 3], 1, cv2.LINE_AA)
        return img
