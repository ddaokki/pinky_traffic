"""검출기 + 제어기 + 대시보드 연결을 묶은 한 덩어리 (ROS 를 모른다).

ROS 노드(nodes/lane_driver.py)와 시뮬레이터(tools/run_sim.py --dashboard)가 같이 쓴다.
"""
import os
import threading
import time

import cv2

from .controller import LaneController, NoLock
from .coordinator import DashLink
from .detectors import make_detector
from .perception import draw_debug


class Driver:
    def __init__(self, cfg, name='pinky', use_dashboard=True, autostart=False, log=print, record_dir=None):
        self.cfg, self.name, self.log = cfg, name, log
        self.detector = make_detector(cfg)
        self.pending = []                   # 대시보드에서 온 명령 (다른 스레드) -> 제어 루프에서 처리
        self.mutex = threading.Lock()
        self.link = DashLink(cfg.dashboard_url, name, self._on_command, self._on_params) if use_dashboard else None
        self.controller = LaneController(cfg, self.link or NoLock(), name)
        self.autostart = autostart
        self.fps = 0.0
        self._t_prev = None
        self._n = 0
        self.last = None
        self.debug = None
        self.state = {}
        self.record_dir = record_dir        # 주면 달리는 동안 카메라 화면을 3장에 1장꼴로 저장 (원인 분석용)

    def _on_command(self, cmd):
        with self.mutex:
            self.pending.append(cmd)

    def _on_params(self, params):
        with self.mutex:
            self.pending.append(('params', params))

    def command(self, cmd):
        self._on_command(cmd)

    def _apply_pending(self, now):
        with self.mutex:
            pending, self.pending = self.pending, []
        for item in pending:
            if isinstance(item, tuple):
                changed = self.cfg.update(item[1])
                if changed:
                    self.log(f'[{self.name}] 파라미터 변경: ' + ', '.join(f'{k}={getattr(self.cfg, k)}' for k in changed))
                    if 'backend' in changed or 'weights' in changed:
                        self.detector = make_detector(self.cfg)
            elif item == 'start':
                self.controller.start(now)
            elif item == 'stop':
                self.controller.stop(now)
            elif item == 'estop':
                self.controller.estop(now)

    def process(self, frame, front_m=None, now=None):
        """영상 한 장 -> Command. 디버그 그림은 self.debug 에 남긴다."""
        now = time.time() if now is None else now
        if self.autostart:
            self.autostart = False
            self.controller.start(now)
        self._apply_pending(now)
        self.detector.prefer = self.controller.prefer      # 갈림길에서 어느 쪽 선을 따라갈지
        p, masks, small = self.detector.detect(frame)
        cmd = self.controller.step(p, front_m, now)
        if self._t_prev is not None and now > self._t_prev:
            self.fps = 0.9 * self.fps + 0.1 / (now - self._t_prev) if self.fps else 1.0 / (now - self._t_prev)
        self._t_prev = now
        self.last = (p, cmd)
        self._n += 1
        if self.record_dir and self._n % 3 == 0 and cmd.state != 'idle':
            os.makedirs(self.record_dir, exist_ok=True)
            cv2.imwrite(os.path.join(self.record_dir, f'{self._n:06d}_{cmd.state}.jpg'), small)
        state = {'state': cmd.state, 'v': round(cmd.v, 3), 'w': round(cmd.w, 3), 'offset': round(p.offset, 3),
                 'heading': round(p.heading, 3), 'ok': p.ok, 'left': p.left_seen, 'right': p.right_seen,
                 'crosswalk': p.crosswalk, 'crosswalk_y': round(p.crosswalk_y, 2),
                 'front': None if front_m is None else round(front_m, 2), 'fps': round(self.fps, 1),
                 'ms': round(p.ms, 1), 'backend': self.cfg.backend, 'reason': cmd.reason,
                 'crossings': self.controller.crossings,
                 'route': self.controller.in_route, 'route_seen': p.route_seen,
                 'route_end_y': round(p.route_end_y, 2), 'role': self.cfg.lane_role,
                 'prefer': self.controller.prefer, 'uturn': p.uturn_seen, 'zone_y': round(p.zone_y, 2), 'led': list(self.controller.led)}
        text = f'{self.name} {cmd.state} v={cmd.v:.2f} w={cmd.w:+.2f}'
        self.debug = draw_debug(small, p, masks, text)
        if self.link:
            self.link.report(state)
            if self._n % 2 == 0:          # 영상은 절반만 올린다
                ok, jpeg = cv2.imencode('.jpg', self.debug, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    self.link.frame(jpeg.tobytes())
        self.state = state
        return cmd

    def idle_report(self, reason, now=None):
        """영상이 안 올 때도 대시보드에 살아있음을 알린다."""
        now = time.time() if now is None else now
        self._apply_pending(now)
        if self.link:
            self.link.report({'state': self.controller.state, 'v': 0.0, 'w': 0.0, 'reason': reason,
                              'backend': self.cfg.backend, 'fps': 0.0})

    def close(self):
        if self.link:
            self.link.close()
