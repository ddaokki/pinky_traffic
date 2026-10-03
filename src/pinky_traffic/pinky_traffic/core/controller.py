"""주행 제어: PID + 상태 머신 (수업 25번 자료의 state machine 패턴).

ROS 를 모른다. step(인식결과, 전방거리, 현재시각) -> (v, w) 만 한다.
그래서 시뮬레이터/테스트/실제 노드가 같은 코드를 쓴다.

부호: offset > 0 (차선 중심이 오른쪽) -> 오른쪽으로 돌아야 함 -> angular.z < 0
"""
from dataclasses import dataclass

from .perception import Perception

IDLE = 'idle'
LANE_FOLLOW = 'lane_follow'
APPROACH = 'approach_crosswalk'
STOP = 'stop_at_crosswalk'
CROSSING = 'crossing'
BLOCKED = 'blocked'
LOST = 'lost'
ESTOP = 'estop'


class PID:
    """수업 23번 자료 control_apps.PID 와 같은 구조. dt 는 밖에서 준다(테스트 가능하게)."""

    def __init__(self, p=0.0, i=0.0, d=0.0, limit=1.0):
        self.P, self.I, self.D, self.limit = p, i, d, limit
        self.reset()

    def reset(self):
        self.pre_state = None
        self.integrated = 0.0

    def update(self, state, dt):
        d = 0.0 if self.pre_state is None or dt <= 0 else (state - self.pre_state) / dt
        self.integrated += state * dt
        # 적분 와인드업 방지
        if self.I > 0:
            cap = self.limit / self.I
            self.integrated = max(-cap, min(cap, self.integrated))
        out = self.P * state + self.I * self.integrated + self.D * d
        self.pre_state = state
        return max(-self.limit, min(self.limit, out))


class NoLock:
    """코디네이터를 안 쓸 때: 항상 통과 허가."""

    def request(self, resource):
        return True

    def release(self, resource):
        pass


@dataclass
class Command:
    v: float = 0.0
    w: float = 0.0
    state: str = IDLE
    reason: str = ''


class LaneController:
    def __init__(self, cfg, lock=None, name='pinky'):
        self.cfg = cfg
        self.name = name
        self.lock = lock or NoLock()
        self.pid = PID()
        self.state = IDLE
        self.resume_state = LANE_FOLLOW     # BLOCKED 에서 돌아갈 상태
        self.t_state = 0.0                  # 현재 상태에 들어온 시각
        self.t_last = None
        self.t_seen = None                  # 마지막으로 차선을 본 시각
        self.t_cross_done = -1e9            # 마지막 횡단보도 통과 완료 시각
        self.t_cw_seen = None
        self.last_w = 0.0
        self.crossings = 0
        self.events = []                    # (t, 문자열) 최근 이벤트

    # ----- 외부 명령 -----
    def start(self, now=0.0):
        if self.state in (IDLE, ESTOP, LOST):
            self.pid.reset()
            self.t_seen = None
            self._go(LANE_FOLLOW, now, 'start')

    def stop(self, now=0.0):
        self._release()
        self._go(IDLE, now, 'stop')

    def estop(self, now=0.0):
        self._release()
        self._go(ESTOP, now, 'estop')

    # ----- 내부 -----
    def _go(self, state, now, reason=''):
        if state != self.state:
            self.events.append((now, f'{self.state} -> {state} {reason}'.strip()))
            self.events = self.events[-50:]
            self.state = state
            self.t_state = now

    def _release(self):
        if self.cfg.use_coordinator:
            self.lock.release(self.cfg.resource)

    def _steer(self, p: Perception, dt, v_target):
        cfg = self.cfg
        self.pid.P, self.pid.I, self.pid.D, self.pid.limit = cfg.kp, cfg.ki, cfg.kd, cfg.w_max
        w = -(self.pid.update(p.offset, dt) + cfg.k_heading * p.heading)
        w = max(-cfg.w_max, min(cfg.w_max, w))
        slow = 1.0 - cfg.slow_gain * min(1.0, abs(p.offset) * 1.5 + abs(p.heading) * 0.5)
        v = max(cfg.v_min, v_target * slow)
        self.last_w = w
        return v, w

    def step(self, p: Perception, front_m=None, now=0.0) -> Command:
        cfg = self.cfg
        dt = 0.0 if self.t_last is None else max(0.0, now - self.t_last)
        self.t_last = now

        if self.state in (IDLE, ESTOP):
            return Command(0.0, 0.0, self.state)

        # ---- 전방 장애물 (라이다 또는 yolo 'robot') : 어떤 주행 상태보다 우선 ----
        blocked = front_m is not None and front_m < cfg.obstacle_stop_m
        if p.obstacle_y and p.obstacle_y >= cfg.robot_stop_row:
            blocked = True
        if self.state == BLOCKED:
            cleared = (front_m is None or front_m > cfg.obstacle_stop_m + 0.05) and \
                      not (p.obstacle_y and p.obstacle_y >= cfg.robot_stop_row - 0.05)
            if cleared:
                self._go(self.resume_state, now, 'clear')
            else:
                return Command(0.0, 0.0, BLOCKED, f'front {front_m}')
        elif blocked and self.state in (LANE_FOLLOW, APPROACH, CROSSING, LOST):
            self.resume_state = self.state if self.state != LOST else LANE_FOLLOW
            self._go(BLOCKED, now, f'front={front_m}')
            return Command(0.0, 0.0, BLOCKED)

        if p.ok:
            self.t_seen = now
        if p.crosswalk:
            self.t_cw_seen = now

        # ---- 횡단보도 정지 ----
        if self.state == STOP:
            granted = self.lock.request(cfg.resource) if cfg.use_coordinator else True
            waited = now - self.t_state
            if waited >= cfg.crosswalk_stop_sec and granted:
                self._go(CROSSING, now, f'waited {waited:.1f}s')
            else:
                return Command(0.0, 0.0, STOP, 'wait' if waited < cfg.crosswalk_stop_sec else 'wait lock')

        # ---- 차선을 못 볼 때 ----
        if not p.ok:
            since = 1e9 if self.t_seen is None else now - self.t_seen
            if self.state == CROSSING and now - self.t_state < cfg.crossing_sec:
                # 횡단보도 줄무늬 위에서는 차선이 끊겨 보일 수 있다 -> 직전 방향으로 천천히
                return Command(cfg.v_min, self.last_w * 0.5, CROSSING, 'blind')
            if since <= cfg.lost_grace_sec:
                return Command(cfg.v_min, self.last_w, self.state, 'grace')
            if since > cfg.lost_timeout_sec or self.t_seen is None:
                if self.state != LOST:
                    self._release()
                    self._go(LOST, now, 'no lane')
                return Command(0.0, 0.0, LOST)
            return Command(cfg.v_min, self.last_w * 0.5, self.state, 'searching')

        if self.state == LOST:
            self.pid.reset()
            self._go(LANE_FOLLOW, now, 'lane found')

        if self.state == LANE_FOLLOW:
            cooled = now - self.t_cross_done >= cfg.crosswalk_cooldown_sec
            if p.crosswalk and cooled:
                self._go(APPROACH, now, f'crosswalk y={p.crosswalk_y:.2f}')
            else:
                v_target = cfg.v_max
                if front_m is not None and front_m < cfg.obstacle_slow_m:
                    span = max(1e-3, cfg.obstacle_slow_m - cfg.obstacle_stop_m)
                    v_target *= max(0.3, (front_m - cfg.obstacle_stop_m) / span)
                v, w = self._steer(p, dt, v_target)
                return Command(v, w, LANE_FOLLOW)

        if self.state == APPROACH:
            if cfg.use_coordinator:
                self.lock.request(cfg.resource)       # 미리 줄을 선다
            if p.crosswalk and p.crosswalk_y >= cfg.crosswalk_stop_row:
                self._go(STOP, now, f'y={p.crosswalk_y:.2f}')
                return Command(0.0, 0.0, STOP)
            if not p.crosswalk and self.t_cw_seen is not None and now - self.t_cw_seen > 1.0:
                self._release()
                self._go(LANE_FOLLOW, now, 'crosswalk gone')
            v, w = self._steer(p, dt, cfg.v_approach)
            return Command(v, w, self.state)

        if self.state == CROSSING:
            if cfg.use_coordinator:
                self.lock.request(cfg.resource)       # 통과 중에도 계속 불러 자리를 유지(하트비트)
            if now - self.t_state >= cfg.crossing_sec:
                self.crossings += 1
                self.t_cross_done = now
                self._release()
                self._go(LANE_FOLLOW, now, f'crossed #{self.crossings}')
            v, w = self._steer(p, dt, cfg.v_cross)
            return Command(v, w, self.state)

        return Command(0.0, 0.0, self.state)
