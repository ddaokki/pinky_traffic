"""주행 제어: PID + 상태 머신 (수업 25번 자료의 state machine 패턴).

ROS 를 모른다. step(인식결과, 전방거리, 현재시각) -> (v, w) 만 한다.
그래서 시뮬레이터/테스트/실제 노드가 같은 코드를 쓴다.

부호: offset > 0 (차선 중심이 오른쪽) -> 오른쪽으로 돌아야 함 -> angular.z < 0
"""
import math
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
PARK_TURN = 'park_turn'            # 칸 끝에서 제자리 회전 (나갈 방향으로 돌아선다)
PARKED = 'parked'
WAIT_JUNCTION = 'wait_junction'    # 유턴 구간이 비기를 기다린다 (2대)
POCKET_ADVANCE = 'pocket_advance'  # 2차선: 파란 선을 따라 칸 입구 가운데까지 간다
POCKET_TURN = 'pocket_turn'        # 2차선: 제자리에서 칸 쪽으로(들어갈 때 오른쪽, 나올 때 왼쪽) 돈다
WAIT_EXIT = 'wait_exit'            # 초록 칸에서 돌아선 뒤, 나가도 될 때까지 기다린다

# LED (r, g, b): 달리는 중 초록, 서 있으면 빨강, 주차 통로 안에서는 통로 색, 주차 완료 초록
LED_GREEN, LED_RED, LED_BLUE = (0, 255, 0), (255, 0, 0), (0, 0, 255)
STOPPED_STATES = (IDLE, STOP, BLOCKED, LOST, ESTOP, WAIT_JUNCTION, WAIT_EXIT)


def led_color(state, in_route=False, route_color=''):
    if state == PARKED:
        return LED_GREEN
    if state in STOPPED_STATES:
        return LED_RED
    if in_route and route_color in ('red', 'blue'):
        return LED_RED if route_color == 'red' else LED_BLUE
    return LED_GREEN


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
        self.in_route = False               # 주차 통로에 들어섰다 (STOP/START 전까지 유지)
        self.t_route = 0.0                  # 통로에 들어선 시각
        self.end_hits = 0                   # 칸 끝 선이 정지 행까지 온 연속 프레임 수
        self.events = []                    # (t, 문자열) 최근 이벤트
        self._reset_role()

    def _reset_role(self):
        """lane_role 맵(파란 유턴 / 초록 칸)의 진행 상태."""
        self.prefer = ''                    # 'left' | 'right' : 그쪽 선만 따라간다 (검출기에 전달)
        self.follow_blue = False            # 2차선: 파란 선을 가운데 두고 따라간다 (검출기에 전달)
        self.advance = 0.0                  # 칸 입구에서 더 간 거리 (명령 속도 적분)
        self.turned = 0.0                   # 1차선: 유턴하며 돈 각도 (rad)
        self.exit_turned = False            # 2차선: 칸에서 나오는 좌회전을 했다
        self.junction_held = False          # 유턴 구간 락을 쥐고 있다
        self.uturn_started = False          # 1차선: 파란 선에 올라탔다
        self.uturn_done = False             # 1차선: 파란 선이 끝났다 (2차선으로 돌아가는 중)
        self.pocket_mode = False            # 2차선: 초록 칸으로 빠지는 중
        self.pocket_parked = False          # 2차선: 칸에서 돌아섰다
        self.exiting = False                # 2차선: 칸에서 나오는 중
        self.t_blue = 0.0
        self.t_mode = 0.0
        self.zone_hits = 0

    def _junction(self, want):
        """유턴 구간 락. want=True 는 요청/유지(허가 여부를 돌려준다), False 는 반납. 1대일 때는 항상 허가."""
        cfg = self.cfg
        if want:
            self.junction_held = bool(self.lock.request(cfg.junction_resource)) if cfg.use_coordinator else True
            return self.junction_held
        if self.junction_held and cfg.use_coordinator:
            self.lock.release(cfg.junction_resource)
        self.junction_held = False
        return True

    def _role_step(self, p, now, dt=0.0):
        """lane_role 에 따른 판단. 멈춰 기다려야 하면 Command 를, 아니면 None 을 돌려준다.

        1차선: 파란 선을 만나면 구간 락을 얻고 파란 선을 따라 유턴 -> 파란 선이 끝나면 오른쪽 선만 따라
               (초록 칸 입구를 지나쳐) 2차선으로 돌아간다 -> junction_clear_sec 뒤 락 반납.
        2차선: 출발부터 락을 쥔다(이 로봇이 칸에 들어가기 전에는 1차선 로봇이 유턴하면 안 된다. 2차선에서 마주친다)
               -> 파란 선이 보이면 오른쪽 선만 따라 칸으로 우회전 -> 초록 선 앞에서 정지, 제자리 180도
               -> 락 반납 -> 락을 다시 얻으면(1차선 로봇이 지나간 뒤) 왼쪽 선만 따라 나와 2차선으로 돌아간다.
        """
        cfg = self.cfg
        if cfg.lane_role == 1:
            if p.uturn_near and not self.uturn_started:
                if not self._junction(True):
                    return Command(0.0, 0.0, WAIT_JUNCTION, 'junction busy')
                self.uturn_started, self.t_blue = True, now
                self.events.append((now, 'uturn start'))
            if self.uturn_started and not self.uturn_done:
                self._junction(True)
                self.follow_blue = True
                self.turned += abs(self.last_w) * dt          # 돈 각도 (명령 회전 속도를 더해서 잰다)
                if p.uturn_seen:
                    self.t_blue = now
                elif self.turned < math.radians(cfg.uturn_min_deg):
                    # 유턴이 덜 끝났는데 파란 선이 안 보인다 = 꺾이는 곳에서 선이 카메라 밑으로 들어갔다.
                    # 유턴 방향(오른쪽)으로 제자리 회전하며 다시 찾는다
                    if now - self.t_blue > cfg.lost_grace_sec and now - self.t_blue <= cfg.side_search_sec:
                        self.last_w = -cfg.side_spin_w
                        return Command(0.0, self.last_w, self.state, 'search blue')
                elif now - self.t_blue > 1.0:
                    self.uturn_done, self.follow_blue, self.prefer, self.t_mode = True, False, 'right', now
                    self.events.append((now, 'uturn done'))
            elif self.uturn_done and self.junction_held:
                self._junction(True)
                if now - self.t_mode >= cfg.junction_clear_sec:
                    self.prefer = ''
                    self._junction(False)
        elif cfg.lane_role == 2:
            if not self.pocket_parked:
                if not self._junction(True):
                    return Command(0.0, 0.0, WAIT_JUNCTION, 'junction busy')
                if p.uturn_near and not self.pocket_mode and self.state == LANE_FOLLOW:
                    self.pocket_mode, self.follow_blue, self.advance = True, True, 0.0
                    self._go(POCKET_ADVANCE, now, 'blue near')
                    return Command(cfg.v_min, 0.0, POCKET_ADVANCE)
                hit = self.pocket_mode and self.state == LANE_FOLLOW and p.zone_seen and p.zone_y >= cfg.park_line_row
                self.zone_hits = self.zone_hits + 1 if hit else 0
                if self.zone_hits >= 2:
                    self._go(PARK_TURN, now, f'green y={p.zone_y:.2f}')
                    return Command(0.0, 0.0, PARK_TURN)
            elif self.exiting:
                self._junction(True)
                if p.uturn_near and not self.exit_turned and self.state == LANE_FOLLOW:
                    self.follow_blue, self.advance, self.exit_turned = False, 0.0, True   # 나가는 회전은 한 번만
                    self._go(POCKET_ADVANCE, now, 'blue near (exit)')
                    return Command(cfg.v_min, 0.0, POCKET_ADVANCE)
                if now - self.t_mode >= cfg.exit_follow_sec and self.state == LANE_FOLLOW:
                    self.exiting, self.prefer = False, ''
                    self._junction(False)
        return None

    @property
    def led(self):
        return led_color(self.state, self.in_route, self.cfg.route_color)

    # ----- 외부 명령 -----
    def start(self, now=0.0):
        if self.state in (IDLE, ESTOP, LOST, PARKED):
            self.pid.reset()
            self.t_seen = None
            if self.state != LOST:
                self.in_route = False
                self._junction(False)
                self._reset_role()
            self._go(LANE_FOLLOW, now, 'start')

    def stop(self, now=0.0):
        self._release()
        self._junction(False)
        self._reset_role()
        self.in_route = False
        self._go(IDLE, now, 'stop')

    def estop(self, now=0.0):
        self._release()
        self._junction(False)
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

        if self.state in (IDLE, ESTOP, PARKED):
            return Command(0.0, 0.0, self.state)
        if self.junction_held and self.state in (PARK_TURN, POCKET_ADVANCE, POCKET_TURN):
            self._junction(True)                      # 기동 중에도 락을 계속 쥔다 (하트비트)
        if self.state == PARK_TURN:
            # 각도 센서 없이 시간으로 돈다: 각도 / 회전 속도
            if now - self.t_state >= math.radians(cfg.park_turn_deg) / max(0.1, cfg.park_turn_w):
                if cfg.lane_role == 2:
                    self.pocket_mode, self.pocket_parked = False, True
                    self._junction(False)             # 칸 안에 들어왔다 -> 1차선 로봇이 유턴해도 된다
                    self._go(WAIT_EXIT, now, 'turned')
                    return Command(0.0, 0.0, WAIT_EXIT)
                self._go(PARKED, now, 'turned')
                return Command(0.0, 0.0, PARKED)
            return Command(0.0, cfg.park_turn_w, PARK_TURN)
        if self.state == WAIT_EXIT:
            if now - self.t_state < cfg.exit_wait_sec or not self._junction(True):
                return Command(0.0, 0.0, WAIT_EXIT)
            self.exiting, self.prefer, self.t_mode, self.t_seen = True, '', now, now
            self.pid.reset()
            self._go(LANE_FOLLOW, now, 'exit pocket')
        if self.state == POCKET_ADVANCE:
            # 들어갈 때는 파란 선을 따라, 나올 때는 곧장. 정해진 거리만큼 간 뒤 제자리 회전
            goal = cfg.exit_advance_m if self.exiting else cfg.pocket_advance_m
            self.advance += cfg.v_min * dt
            if self.advance >= goal:
                self.follow_blue = False
                self._go(POCKET_TURN, now, f'advanced {self.advance:.2f}m')
                return Command(0.0, 0.0, POCKET_TURN)
            w = self._steer(p, dt, cfg.v_min)[1] if (p.ok and self.follow_blue) else 0.0
            return Command(cfg.v_min, w, POCKET_ADVANCE)
        if self.state == POCKET_TURN:
            if now - self.t_state >= math.radians(cfg.pocket_turn_deg) / max(0.1, cfg.park_turn_w):
                self.pid.reset()
                self.t_seen, self.t_mode = now, now
                self._go(LANE_FOLLOW, now, 'turned')
            else:
                return Command(0.0, cfg.park_turn_w if self.exiting else -cfg.park_turn_w, POCKET_TURN)
        if cfg.lane_role:
            wait = self._role_step(p, now, dt)
            if wait is not None:
                return wait
            if self.prefer or self.pocket_mode or self.exiting or (self.uturn_started and not self.uturn_done):
                p.crosswalk = False                   # 유턴 구간·칸 안에는 횡단보도가 없다 (가로선 오검출 방지)

        # ---- 주차 통로: 들어서면 기억하고, 칸 끝 벽이 park_stop_m 안에 오면 주차 완료 ----
        if p.route_near and self.state != BLOCKED:
            if not self.in_route:
                self.events.append((now, 'route entered'))
                self.t_route = now
            self.in_route = True
        # 끝 선이 park_line_row 까지 내려왔거나(2프레임 연속), 선을 못 봤어도 벽이 park_stop_m 안이면 주차
        at_line = self.in_route and p.route_end and p.route_end_y >= cfg.park_line_row and \
            now - self.t_route >= cfg.park_min_route_sec
        self.end_hits = self.end_hits + 1 if at_line else 0
        at_wall = self.in_route and front_m is not None and front_m < cfg.park_stop_m
        if self.end_hits >= 2 or at_wall:
            self._release()
            reason = f'front={front_m:.2f}' if at_wall else f'end line y={p.route_end_y:.2f}'
            self._go(PARK_TURN if cfg.park_turn_deg > 0 else PARKED, now, reason)
            return Command(0.0, 0.0, self.state)
        # 유턴 구간은 벽이 가깝다 (통로 안과 같이 더 가까이까지 허용)
        in_uturn = self.uturn_started and self.junction_held      # 유턴을 마치고 구간을 벗어날 때까지
        stop_m = cfg.park_stop_m if self.in_route or in_uturn else cfg.obstacle_stop_m

        # ---- 전방 장애물 (라이다 또는 yolo 'robot') : 어떤 주행 상태보다 우선 ----
        blocked = front_m is not None and front_m < stop_m
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
            if self.prefer and since <= cfg.side_search_sec:
                # 따라가던 쪽 선을 놓쳤다 (선이 그쪽으로 급하게 꺾였다) -> 그쪽으로 제자리 회전하며 찾는다
                w = -cfg.side_spin_w if self.prefer == 'right' else cfg.side_spin_w
                return Command(0.0, w, self.state, f'search {self.prefer}')
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
            # 진짜 횡단보도는 먼 곳에서 먼저 보이고 점점 다가온다. 처음부터 정지 행보다 가까이(발밑)에서
            # 나타난 것은 코너의 테이프 조각 같은 오검출로 보고 무시한다 (2026-10-04 코너에서 오인 정지)
            if p.crosswalk and cooled and p.crosswalk_y < cfg.crosswalk_stop_row - 0.12:
                self._go(APPROACH, now, f'crosswalk y={p.crosswalk_y:.2f}')
            else:
                v_target = cfg.v_min if self.follow_blue else cfg.v_max     # 파란 선 위에서는 천천히
                if front_m is not None and front_m < cfg.obstacle_slow_m:
                    span = max(1e-3, cfg.obstacle_slow_m - stop_m)
                    v_target *= max(0.3, (front_m - stop_m) / span)
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
            v, w = self._steer(p, dt, min(cfg.v_approach, cfg.v_max))
            return Command(v, w, self.state)

        if self.state == CROSSING:
            if cfg.use_coordinator:
                self.lock.request(cfg.resource)       # 통과 중에도 계속 불러 자리를 유지(하트비트)
            if now - self.t_state >= cfg.crossing_sec:
                self.crossings += 1
                self.t_cross_done = now
                self._release()
                self._go(LANE_FOLLOW, now, f'crossed #{self.crossings}')
            v, w = self._steer(p, dt, min(cfg.v_cross, cfg.v_max))
            return Command(v, w, self.state)

        return Command(0.0, 0.0, self.state)
