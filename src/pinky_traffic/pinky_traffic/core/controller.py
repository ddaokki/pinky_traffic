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
SIGN_APPROACH = 'sign_approach'    # 파란 표지판 쪽으로 간다
SIGN_ADVANCE = 'sign_advance'      # 표지판 끝에 도착 -> 곧장 조금 더 가서 표지판 위에 선다
SIGN_TURN = 'sign_turn'            # 표지판 위에서 제자리 90도 (경로에 적힌 쪽으로)
SIGN_SEARCH = 'sign_search'        # 다음 표지판을 찾으며 곧장 간다
POCKET_END = 'pocket_end'          # 초록 선이 보인 뒤 곧장 더 가서 초록 선 위에 선다
WAIT_EXIT = 'wait_exit'            # 초록 칸에서 돌아선 뒤, 상대 로봇이 지나갈 때까지 기다린다

# LED (r, g, b): 달리는 중 초록, 서 있으면 빨강, 주차 통로 안에서는 통로 색, 주차 완료 초록
LED_GREEN, LED_RED, LED_BLUE = (0, 255, 0), (255, 0, 0), (0, 0, 255)
STOPPED_STATES = (IDLE, STOP, BLOCKED, LOST, ESTOP, WAIT_JUNCTION, WAIT_EXIT)
MANEUVERS = (SIGN_APPROACH, SIGN_ADVANCE, SIGN_TURN, SIGN_SEARCH, POCKET_END, PARK_TURN)


def parse_plan(text):
    """'turn:right, straight_right:straight' -> [('turn', 'right'), ('straight_right', 'straight')]"""
    plan = []
    for item in str(text).split(','):
        if item.strip():
            kind, _, action = item.strip().partition(':')
            plan.append((kind.strip(), action.strip() or 'straight'))
    return plan


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

    def flag(self, name, on=True):
        pass

    def others_flag(self, name):
        return False


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
        self.last_offset = None             # 직전 프레임의 차선 중심 (한 프레임 튐 거르기)
        self.side_alert = False             # 옆 물체를 피하는 중
        self.intrude_dir, self.intrude_until = 0, -1.0   # 시연용 끼어들기 (+1 오른쪽, -1 왼쪽)
        self.no_lidar = False
        self.jumps = 0
        self.crossings = 0
        self.in_route = False               # 주차 통로에 들어섰다 (STOP/START 전까지 유지)
        self.t_route = 0.0                  # 통로에 들어선 시각
        self.end_hits = 0                   # 칸 끝 선이 정지 행까지 온 연속 프레임 수
        self.events = []                    # (t, 문자열) 최근 이벤트
        self._reset_role()

    def _reset_role(self):
        """lane_role 맵(표지판 경로 / 초록 칸)의 진행 상태."""
        self.prefer = ''                    # 'left' | 'right' : 그쪽 선만 따라간다 (검출기에 전달)
        self.plan, self.plan_i = [], 0      # 표지판 경로 [(종류, 행동)] 과 다음 순번
        self.plan_name = ''
        self.plan_done = False              # 경로를 다 지났다 (흰 차선으로)
        self.action = ''                    # 지금 표지판에서 할 행동
        self.target = None                  # 다가가는 표지판 (종류, x, 먼 끝, 가까운 끝)
        self.t_target = 0.0
        self.advance = 0.0                  # 표지판 끝에서 더 간 거리 (명령 속도 적분)
        self.junction_held = False          # 유턴 구간 락을 쥐고 있다
        self.pocket_mode = False            # 2차선: 초록 칸으로 들어가는 중
        self.pocket_parked = False          # 2차선: 칸에서 돌아섰다
        self.exiting = False                # 2차선: 칸에서 나오는 중
        self.cleared = False                # 경로를 마치고 구간을 벗어났다 (락 반납, 깃발 내림)
        self.flag_up = False                # 1차선: oncoming 깃발을 올려 두었다
        self.t_decide = None                # 2차선: 직우 표지판 위에서 깃발을 기다리기 시작한 시각
        self.saw_robot = False              # 2차선: 칸에서 상대 로봇을 봤다
        self.t_robot = 0.0
        self.t_mode = 0.0
        self.zone_hits = 0

    @property
    def plan_text(self):
        if not self.plan:
            return ''
        return ' '.join(('[' if i == self.plan_i else '') + f'{k}:{a}' + (']' if i == self.plan_i else '')
                        for i, (k, a) in enumerate(self.plan))

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

    def _oncoming(self):
        """다른 로봇(1차선)이 유턴하러 오는 중이다 (서버 깃발). 1대일 때는 늘 아니다."""
        return bool(self.cfg.use_coordinator and self.lock.others_flag(self.cfg.oncoming_flag))

    def _flag(self, on):
        """oncoming 깃발 올리기(계속 불러야 유지된다)/내리기."""
        if self.cfg.use_coordinator and (on or self.flag_up):
            self.lock.flag(self.cfg.oncoming_flag, on)
        self.flag_up = on

    def _set_plan(self, name, now):
        self.plan, self.plan_i, self.plan_name, self.plan_done = parse_plan(getattr(self.cfg, name)), 0, name, False
        self.events.append((now, f'plan {name}: {self.plan_text}'))

    def _wanted_sign(self, p):
        """경로의 다음 표지판과 같은 종류 중 가장 가까운 것."""
        if self.plan_i >= len(self.plan):
            return None
        kind = self.plan[self.plan_i][0]
        for sign in p.signs:
            if kind in ('any', sign[0]) or sign[0] == 'blue':     # 'blue' = 색으로 찾아 종류를 모른다 -> 가장 가까운 것
                return sign
        return None

    def _approach(self, sign, now, why):
        self.target, self.t_target = sign, now
        self.pid.reset()
        self._go(SIGN_APPROACH, now, f'{sign[0]} ({why})')

    def _clear_step(self, now):
        """경로를 마친 뒤 junction_clear_sec 동안 흰 차선을 따라 구간을 벗어나고, 락을 내주고 깃발을 내린다."""
        if self.junction_held:
            self._junction(True)
        if now - self.t_mode >= self.cfg.junction_clear_sec:
            self.cleared = True
            self._junction(False)
            self._flag(False)
            self.events.append((now, 'junction clear'))

    def _arrived(self, now):
        """표지판 끝에 도착: 할 행동을 정한다. 기다려야 하면 Command."""
        cfg = self.cfg
        kind, action = self.plan[self.plan_i]
        if cfg.lane_role == 2 and self.plan_name == 'plan_lane2' and self.plan_i == 0:
            # 직우 표지판 위: 상대가 오면 우회전해서 칸으로, 아니면 직진(유턴하러)
            if self._oncoming():
                self._junction(True)                  # 칸에 들어설 때까지 쥔다 (못 얻어도 비키는 게 낫다)
                self._set_plan('plan_lane2_pocket', now)
                kind, action = self.plan[0]
                self.events.append((now, 'oncoming -> pocket'))
            else:
                if self.t_decide is None:
                    self.t_decide = now
                if cfg.use_coordinator and now - self.t_decide < cfg.pocket_decide_sec:
                    return Command(0.0, 0.0, WAIT_JUNCTION, 'look for oncoming')
                if not self._junction(True):
                    return Command(0.0, 0.0, WAIT_JUNCTION, 'junction busy')
        self.action, self.advance = action, 0.0
        self._go(SIGN_ADVANCE, now, f'{kind}:{action}')
        return None

    def _maneuver_done(self, now):
        """표지판 하나를 마쳤다 -> 다음 표지판 찾기 / 칸으로 / 경로 끝."""
        self.plan_i += 1
        self.pid.reset()
        self.t_seen, self.t_mode = now, now
        if self.plan_name == 'plan_lane2_pocket':
            self.pocket_mode = True               # 칸 쪽으로 돌았다 -> 초록 선까지 (아래 칸 로직)
            self._go(LANE_FOLLOW, now, 'into pocket')
        elif self.plan_i >= len(self.plan):
            self.plan_done = True
            self.events.append((now, f'{self.plan_name} done'))
            self._go(LANE_FOLLOW, now, 'plan done')
        else:
            self._go(SIGN_SEARCH, now, 'next sign')

    def _role_step(self, p, now, dt=0.0):
        """lane_role 에 따른 판단 (LANE_FOLLOW / SIGN_SEARCH 일 때). 멈춰 기다려야 하면 Command 를, 아니면 None.

        표지판 경로(config plan_*)를 만나는 순서대로 따른다. 표지판이 보이면 그쪽으로 가서, 먼 쪽 끝에 닿으면
        sign_advance_m 더 가서 경로에 적힌 대로 제자리 90도(right/left) 또는 곧장(straight). 경로가 끝나면 흰 차선.
        1차선: 첫 표지판(우회전 양방향)이 보이면 oncoming 깃발을 올리고 구간 락을 얻는다 -> 우 -> 우 -> 직진
               -> junction_clear_sec 뒤 락 반납·깃발 내림.
        2차선: 직우 표지판 끝에서 깃발이 있으면 우회전해 초록 칸 -> 180도 -> 상대가 지나가면(카메라·라이다) 또는
               깃발이 내려가면 나와서 plan_lane2_exit (우 -> 좌 -> 좌). 깃발이 없으면 직진 -> 좌 -> 좌.
        """
        cfg = self.cfg
        if cfg.lane_role == 1:
            if p.signs and not self.flag_up and not self.cleared:
                self.events.append((now, 'oncoming flag up'))
            if (p.signs or self.flag_up) and not self.cleared:
                self._flag(True)
        if self.junction_held and not self.cleared:
            self._junction(True)                      # 하트비트
        if self.plan_done:
            if not self.cleared:
                self._clear_step(now)
            return None
        if self.pocket_mode:
            if self.state == LANE_FOLLOW and now - self.t_mode > cfg.pocket_giveup_sec:
                # 칸을 못 찾았다. '칸 안' 상태로 남으면 횡단보도도 안 보고 락도 계속 쥔다 -> 포기하고 보통 주행
                self.pocket_mode, self.pocket_parked, self.plan_done, self.cleared = False, True, True, True
                self._junction(False)
                self.events.append((now, 'pocket give up'))
                return None
            hit = self.state == LANE_FOLLOW and p.zone_seen and p.zone_y >= cfg.park_line_row
            self.zone_hits = self.zone_hits + 1 if hit else 0
            if self.zone_hits >= 2:
                self.advance = 0.0
                self._go(POCKET_END, now, f'green y={p.zone_y:.2f}')
                return Command(cfg.v_min, 0.0, POCKET_END)
            return None
        if not self.plan and not self.pocket_parked:
            first = 'turn' if cfg.lane_role == 1 else 'straight_right'
            sign = next((s for s in p.signs if s[0] in (first, 'blue')), None)
            if sign is None:
                return None
            if cfg.lane_role == 1 and not self._junction(True):
                return Command(0.0, 0.0, WAIT_JUNCTION, 'junction busy')
            self._set_plan('plan_lane1' if cfg.lane_role == 1 else 'plan_lane2', now)
        sign = self._wanted_sign(p)
        if sign is not None:
            self._approach(sign, now, 'seen')
            return None
        if self.state == SIGN_SEARCH or self.exiting:
            # 다음 표지판이 아직 안 보인다 -> 곧장 간다 (갈림길의 흰 선은 좌우 구분이 틀어진다)
            if now - self.t_mode < cfg.sign_search_sec:
                self.t_seen = now
                return Command(cfg.v_min, 0.0, self.state, 'look for sign')
            self.plan_done, self.exiting, self.cleared = True, False, True
            self._junction(False)
            self._flag(False)
            self.events.append((now, f'sign not found ({self.plan_name}): {self.plan_text}'))
            self._go(LANE_FOLLOW, now, 'give up plan')
        return None

    @property
    def led(self):
        return led_color(self.state, self.in_route, self.cfg.route_color)

    # ----- 외부 명령 -----
    def start(self, now=0.0):
        if self.state in (IDLE, ESTOP, LOST, PARKED):
            self.pid.reset()
            self.last_offset, self.jumps = None, 0
            self.t_seen = None
            if self.state != LOST:
                self.in_route = False
                self._junction(False)
                self._flag(False)
                self._reset_role()
            self._go(LANE_FOLLOW, now, 'start')

    def intrude(self, direction, now=0.0):
        """시연용: intrude_sec 동안 옆 차선 쪽(+1 오른쪽, -1 왼쪽)으로 붙는다. 옆 로봇이 비키는 것을 보여 주려고 넣었다."""
        if self.state == LANE_FOLLOW:
            self.intrude_dir, self.intrude_until = (1 if direction > 0 else -1), now + self.cfg.intrude_sec
            self.events.append((now, f"intrude {'right' if direction > 0 else 'left'}"))

    def stop(self, now=0.0):
        self._release()
        self._junction(False)
        self._flag(False)
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

    def step(self, p: Perception, front_m=None, now=0.0, sides=None, lidar_ok=True) -> Command:
        """sides = (왼쪽 거리, 오른쪽 거리) 라이다 측면 최소값 (없으면 None). lidar_ok=False 면 움직이지 않는다."""
        cfg = self.cfg
        if not lidar_ok and cfg.require_lidar and self.state not in (IDLE, ESTOP, PARKED):
            # 라이다가 끊기면 앞의 장애물·옆 로봇을 못 본다 -> 바퀴를 세운다 (상태와 타이머는 그대로 둔다)
            self.t_last = now
            if not self.no_lidar:
                self.events.append((now, 'no lidar -> hold'))
            self.no_lidar = True
            return Command(0.0, 0.0, self.state, 'no lidar')
        self.no_lidar = False
        cmd = self._step(p, front_m, now)
        return self._side_guard(cmd, sides, now)

    def _side_guard(self, cmd, sides, now):
        """옆구리 침범 막기: 측면 side_slow_m 안에 뭔가 있으면 반대쪽으로 조향하고 속도를 줄인다. side_stop_m 안이면 전진은 멈춘다.
        (2026-10-09: 나란히 달리다 한 대가 차선을 잘못 잡으면 옆 차선으로 밀고 들어온다. 두 대 다 이 규칙으로 서로 비킨다)"""
        cfg = self.cfg
        if not cfg.side_guard or not sides or cmd.state not in (LANE_FOLLOW, APPROACH, CROSSING) or cmd.v <= 0 \
                or self.pocket_mode or self.exiting or now < self.intrude_until:   # 끼어드는 쪽은 안 비킨다 (차선을 잘못 본 로봇 흉내)
            self.side_alert = False
            return cmd
        push, near = 0.0, None
        for dist, away in zip(sides, (-1.0, 1.0)):            # 왼쪽에 있으면 오른쪽(-w)으로, 오른쪽에 있으면 왼쪽(+w)으로
            if dist is not None and dist < cfg.side_slow_m:
                push += away * min(1.0, (cfg.side_slow_m - dist) / max(1e-3, cfg.side_slow_m - cfg.side_stop_m))
                near = dist if near is None else min(near, dist)
        if near is None:
            self.side_alert = False
            return cmd
        if not self.side_alert:
            self.events.append((now, f'side guard L={sides[0]} R={sides[1]}'))
        self.side_alert = True
        w = max(-cfg.w_max, min(cfg.w_max, cmd.w + push * cfg.side_push_w))
        v = 0.0 if near < cfg.side_stop_m else cmd.v * (1.0 - 0.6 * min(1.0, abs(push)))
        self.last_w = w
        side = 'L' if push < 0 else 'R'
        return Command(v, w, cmd.state, f'side {side} {near:.2f}')

    def _step(self, p: Perception, front_m=None, now=0.0) -> Command:
        cfg = self.cfg
        dt = 0.0 if self.t_last is None else max(0.0, now - self.t_last)
        self.t_last = now

        if self.state in (IDLE, ESTOP, PARKED):
            return Command(0.0, 0.0, self.state)
        if self.junction_held and self.state in MANEUVERS:
            self._junction(True)                      # 기동 중에도 락을 계속 쥔다 (하트비트)
        if cfg.lane_role == 1 and self.flag_up and not self.cleared:
            self._flag(True)
        if self.state == POCKET_END:
            self.advance += cfg.v_min * dt
            if self.advance < cfg.zone_advance_m:
                return Command(cfg.v_min, 0.0, POCKET_END)
            self._go(PARK_TURN, now, f'on green +{self.advance:.2f}m')
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
            # 칸 입구 쪽을 보고 서 있다. 상대 로봇이 보였다가 사라지면(카메라·라이다) 또는 상대가 깃발을 내리면(서버) 나간다
            if p.obstacle_y > 0 or (front_m is not None and front_m < cfg.pass_front_m):
                self.saw_robot, self.t_robot = True, now
            passed = self.saw_robot and now - self.t_robot >= cfg.pass_clear_sec
            if now - self.t_state < cfg.exit_wait_sec or not (passed or not self._oncoming()):
                return Command(0.0, 0.0, WAIT_EXIT, 'robot seen' if self.saw_robot else 'wait oncoming')
            self.exiting, self.t_mode, self.t_seen = True, now, now
            self._set_plan('plan_lane2_exit', now)
            self.pid.reset()
            self._go(LANE_FOLLOW, now, 'exit pocket (robot passed)' if passed else 'exit pocket (flag down)')
        if self.state == SIGN_APPROACH:
            # 표지판 가운데를 보고 천천히 간다. 화면 아래로 완전히 사라질 때까지(= 표지판 위에 올라탈 때까지) 간 뒤 꺾는다.
            # (2026-10-09 현장 요청: 보이자마자 꺾으면 차선을 넘는다)
            sign = self._wanted_sign(p)
            if sign is not None:
                self.target, self.t_target = sign, now
            arrived = False
            if sign is None:
                gone = now - self.t_target
                if self.target[3] >= cfg.sign_gone_row:            # 화면 아래로 빠져나갔다
                    arrived = gone >= cfg.sign_gone_sec
                    if not arrived:
                        return Command(cfg.v_min, 0.0, SIGN_APPROACH, 'over sign')
                elif gone > cfg.lost_timeout_sec:                   # 멀리서 놓쳤다 -> 다시 찾기
                    self._go(SIGN_SEARCH if self.plan_i or self.exiting else LANE_FOLLOW, now, 'sign lost')
                    self.t_mode = now
                    return Command(0.0, 0.0, self.state)
            if arrived:
                self.exiting = False
                wait = self._arrived(now)
                if wait is not None:
                    return wait
            elif self.plan_name == 'plan_lane2_exit' and self.plan_i == 0:
                # 칸에서 나올 때: 입구의 직우 표지판은 왼쪽으로 길게 보여 가운데를 보고 가면 칸 벽 선을 넘는다 -> 곧장 나간다
                return Command(cfg.v_min, 0.0, SIGN_APPROACH, 'exit straight')
            else:
                x = self.target[1]
                v, w = self._steer(Perception(ok=True, offset=x), dt, cfg.v_min)
                return Command(cfg.v_min, w, SIGN_APPROACH)
        if self.state == SIGN_ADVANCE:
            self.advance += cfg.v_min * dt
            if self.advance < cfg.sign_advance_m:
                return Command(cfg.v_min, 0.0, SIGN_ADVANCE)
            if self.action in ('right', 'left'):
                self._go(SIGN_TURN, now, self.action)
            else:
                self._maneuver_done(now)
        if self.state == SIGN_TURN:
            if now - self.t_state < math.radians(cfg.sign_turn_deg) / max(0.1, cfg.park_turn_w):
                return Command(0.0, -cfg.park_turn_w if self.action == 'right' else cfg.park_turn_w, SIGN_TURN)
            self._maneuver_done(now)
        if cfg.lane_role and self.state in (LANE_FOLLOW, SIGN_SEARCH):
            wait = self._role_step(p, now, dt)
            if wait is not None:
                return wait
            if self.state == SIGN_APPROACH:
                return Command(cfg.v_min, 0.0, SIGN_APPROACH)
            if cfg.lane_role == 2 and self.pocket_mode and self.state == LANE_FOLLOW and not p.zone_seen \
                    and now - self.t_mode < cfg.pocket_blind_sec:
                self.t_seen = now
                return Command(cfg.v_min, 0.0, LANE_FOLLOW, 'to green')
            if self.plan and not self.plan_done or self.pocket_mode or self.exiting:
                p.crosswalk = False                   # 표지판 구간·칸 안에는 횡단보도가 없다 (가로선 오검출 방지)

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
        in_uturn = bool(self.plan) and not self.cleared          # 표지판 경로를 마치고 구간을 벗어날 때까지
        stop_m = cfg.park_stop_m if self.in_route or in_uturn else cfg.obstacle_stop_m

        # ---- 전방 장애물 (라이다 또는 yolo 'robot') : 어떤 주행 상태보다 우선 ----
        blocked = front_m is not None and front_m < stop_m
        if cfg.robot_stop and p.obstacle_y and p.obstacle_y >= cfg.robot_stop_row:
            blocked = True
        if self.state == BLOCKED:
            cleared = (front_m is None or front_m > cfg.obstacle_stop_m + 0.05) and \
                      not (cfg.robot_stop and p.obstacle_y and p.obstacle_y >= cfg.robot_stop_row - 0.05)
            if cleared:
                self._go(self.resume_state, now, 'clear')
            else:
                # 앞이 막혀도 제자리 회전은 한다. 코너에서 벽을 마주 보고 선 경우, 차선 쪽으로 돌면 앞이 트인다
                # (2026-10-04: 벽 가까운 코너에서 0.15m 에 걸려 그대로 서 있었다)
                w = self._steer(p, dt, 0.0)[1] if p.ok else 0.0
                return Command(0.0, w, BLOCKED, f'front {front_m}')
        elif blocked and self.state in (LANE_FOLLOW, APPROACH, CROSSING, LOST):
            self.resume_state = self.state if self.state != LOST else LANE_FOLLOW
            self._go(BLOCKED, now, f'front={front_m}')
            return Command(0.0, 0.0, BLOCKED)

        if p.ok:
            self.t_seen = now
            # 차선 중심이 갑자기 크게 튀면 2프레임까지는 무시한다. 그 뒤에도 같으면 믿는다.
            # (2026-10-09 pinky1: 코너 꼭짓점에서 양쪽 선이 한 덩어리로 붙어 한 프레임 동안 offset +1.5 -> 오른쪽으로 확 꺾여 차선 이탈)
            if self.last_offset is not None and abs(p.offset - self.last_offset) > cfg.offset_jump and self.jumps < 2:
                self.jumps += 1
                p.offset = self.last_offset
            else:
                self.jumps = 0
            self.last_offset = p.offset
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
                v_target = cfg.v_min if self.pocket_mode else cfg.v_max   # 칸 안에서는 천천히
                if front_m is not None and front_m < cfg.obstacle_slow_m:
                    span = max(1e-3, cfg.obstacle_slow_m - stop_m)
                    v_target *= max(0.3, (front_m - stop_m) / span)
                if now < self.intrude_until:
                    # 시연용 끼어들기: 차선 중심을 옆으로 밀어 본 것처럼 해서 옆 차선 쪽으로 붙는다
                    p.offset += self.intrude_dir * cfg.intrude_offset
                v, w = self._steer(p, dt, v_target)
                return Command(v, w, LANE_FOLLOW, 'intrude' if now < self.intrude_until else '')

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
