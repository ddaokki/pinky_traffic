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
SIGN_HOLD = 'sign_hold'            # 표지판/경로를 잃음: 순번을 보존하고 STOP/START까지 정지
POCKET_END = 'pocket_end'          # 초록 선이 보인 뒤 곧장 더 가서 초록 선 위에 선다
WAIT_EXIT = 'wait_exit'            # 초록 칸에서 돌아선 뒤, 상대 로봇이 지나갈 때까지 기다린다

# LED (r, g, b): 달리는 중 초록, 서 있으면 빨강, 주차 통로 안에서는 통로 색, 주차 완료 초록
LED_GREEN, LED_RED, LED_BLUE = (0, 255, 0), (255, 0, 0), (0, 0, 255)
STOPPED_STATES = (IDLE, STOP, BLOCKED, LOST, ESTOP, WAIT_JUNCTION, WAIT_EXIT, SIGN_HOLD)
MANEUVERS = (SIGN_APPROACH, SIGN_ADVANCE, SIGN_TURN, SIGN_SEARCH, SIGN_HOLD, POCKET_END, PARK_TURN)


def parse_plan(text, with_advance=False):
    """'turn:right, straight_right:straight' -> [('turn', 'right'), ('straight_right', 'straight')]
    세 번째 칸은 그 표지판만의 sign_advance_m ('turn:right:0.12'). with_advance 면 [거리 또는 None] 도 함께 돌려준다."""
    plan, adv = [], []
    for item in str(text).split(','):
        if item.strip():
            parts = [x.strip() for x in item.strip().split(':')]
            plan.append((parts[0], parts[1] if len(parts) > 1 and parts[1] else 'straight'))
            try:
                adv.append(float(parts[2]) if len(parts) > 2 and parts[2] else None)
            except ValueError:
                adv.append(None)
    return (plan, adv) if with_advance else plan


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
        self.cw_hits = 0                    # 정지 행 앞에서 횡단보도가 연속으로 보인 프레임 수
        self.held = False                   # 안전 정지 중 (앞·옆이 너무 가깝다)
        self.lane2_up = False               # 2차선: 진행 중 깃발을 올려 두었다
        self.cw_first_y = 1.0
        self.intrude_dir, self.intrude_until = 0, -1.0   # 시연용 끼어들기 (+1 오른쪽, -1 왼쪽)
        self.no_lidar = False
        self.jumps = 0
        self.crossings = 0
        self.in_route = False               # 주차 통로에 들어섰다 (STOP/START 전까지 유지)
        self.t_route = 0.0                  # 통로에 들어선 시각
        self.end_hits = 0                   # 칸 끝 선이 정지 행까지 온 연속 프레임 수
        self.events = []                    # (t, 문자열) 최근 이벤트
        self.t_go = 0.0                     # 이 시각까지는 출발하지 않는다 (2차선 지연 출발)
        self.yaw = None                     # 오도메트리 방향 (rad)
        self.yaw_start = None
        self.turn_deg = None
        self.t_turn_done = None
        self.park_yaw_t = None
        self._reset_role()

    def _reset_role(self):
        """lane_role 맵(표지판 경로 / 초록 칸)의 진행 상태."""
        self.prefer = ''                    # 'left' | 'right' : 그쪽 선만 따라간다 (검출기에 전달)
        self.plan, self.plan_i = [], 0      # 표지판 경로 [(종류, 행동)] 과 다음 순번
        self.plan_adv = []                  # 표지판마다 따로 정한 sign_advance_m (None = 기본값)
        self.advance_m = 0.0                # 지금 표지판에서 더 갈 거리
        self.t_align = None                 # 표지판 정렬(제자리 회전)을 시작한 시각
        self.aligned = False                # 직우 표지판과 나란히 맞췄다 -> 그 방향으로 곧장
        self.t_yolo = 0.0                   # 쫓던 표지판을 마지막으로 (YOLO 로) 본 시각
        self.after_turn = False             # 돈 뒤 다음 표지판으로 다가가는 중 (먼저 정면 맞추기)
        self.t_walled = None                # 표지판 위에서 앞이 막힌 시각
        self.backoff_run = 0.0              # 회전 직후 너무 가까운 다음 표지판을 다시 보기 위해 후진한 거리
        self.backoff_done = False
        self.t_backoff_end = None
        self.exit_run = 0.0                 # 1차선: R2 뒤 곧장 간 거리
        self.align_shaft = False            # 정면 맞추기: 축 맞추는 단계에 들어갔다
        self.align_off = False              # 정면 맞추기를 건너뛰었다 (그래도 표지판 쪽으로는 간다)
        self.pulse_w, self.pulse_until, self.settle_until, self.ok_hits = 0.0, -1.0, -1.0, 0
        self.at_sign = False                # 표지판 위에 도착했다 (기다리는 중에 표지판이 다시 보여도 다시 다가가지 않는다)
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

    def _flag2(self, on):
        """2차선 진행 중 깃발 (계속 불러야 유지)."""
        if self.cfg.use_coordinator and (on or self.lane2_up):
            self.lock.flag(self.cfg.lane2_flag, on)
        self.lane2_up = on

    def _set_plan(self, name, now):
        (self.plan, self.plan_adv), self.plan_i, self.plan_name, self.plan_done = parse_plan(getattr(self.cfg, name), True), 0, name, False
        self.events.append((now, f'plan {name}: {self.plan_text}'))
        if self.cfg.lane_role == 2 and name == 'plan_lane2':
            self._flag2(True)                     # 직우를 봤다 -> 1차선은 R1 위에서 기다린다

    def _mine(self, sign, searching=False):
        """내 차선 앞의 표지판인가: 화면 가운데 쪽(|x| <= sign_max_x)이고 충분히 가까이(가까운 끝 >= sign_start_row) 왔다.
        (2026-10-09: 멀리서 보자마자 다가가 차선을 벗어났고, 1차선 로봇이 옆 차선의 직우 표지판으로 갔다)
        다음 표지판을 찾으며 곧장 가는 중(searching)에는 조금 더 멀리 있어도 된다."""
        row = self.cfg.sign_search_row if searching else self.cfg.sign_start_row
        # 칸에서 나올 때 입구의 직우는 왼쪽으로 길게 보인다. 표지판을 지나 다음 표지판을 찾을 때도 옆에 보일 수 있다
        # (2026-10-09 pinky2: 직우 직진 뒤 R2 가 옆에 보여 ±0.45 에 걸러져 그냥 지나감)
        max_x = 0.9 if self.exiting else (self.cfg.sign_search_x if searching else self.cfg.sign_max_x)
        return abs(sign[1]) <= max_x and sign[3] >= row

    def _wanted_sign(self, p, tracking=False):
        """경로의 다음 표지판과 같은 종류 중 내 차선 앞에서 가장 가까운 것.
        tracking: 이미 다가가는 중이면 거리·위치 조건 없이, 쫓던 표지판과 가로 위치가 가장 가까운 것(다른 표지판으로 갈아타지 않게)."""
        if self.plan_i >= len(self.plan):
            return None
        kind = self.plan[self.plan_i][0]
        ok = [s for s in p.signs if kind in ('any', s[0]) or s[0] == 'blue']   # 'blue' = 종류를 모른다
        if tracking and self.target is not None:
            # 이미 고른 표지판은 위치로만 따라간다. 모양 분류는 보는 각도에 따라 바뀐다
            # (2026-10-10 pinky1: 물러나서 고른 R2 가 '직우'로 분류돼 다음 프레임에 바로 놓친 것으로 처리 -> sign_hold)
            near = [s for s in p.signs if abs(s[1] - self.target[1]) <= 0.5
                    and s[3] >= self.target[3] - self.cfg.sign_track_back_row
                    and s[2] >= self.target[2] - self.cfg.sign_track_back_row]
            return min(near, key=lambda s: abs(s[1] - self.target[1]) + abs(s[2] - self.target[2])
                       + abs(s[3] - self.target[3])) if near else None
        searching = self.state == SIGN_SEARCH or self.exiting
        if searching and not self.exiting and self.backoff_run > 0:
            # 물러나서 다시 본 바로 앞 표지판이 다음 표지판이다. 옆에서 본 화살표는 길쭉해 모양 분류가 틀리므로 종류를 안 따진다
            # 물러난 뒤에는 표지판이 화면 위로 조금 올라가 있다 (2026-10-10: 가까운 끝 0.85 기준에 못 미쳐 오른쪽 먼 표지판을 골라 그쪽으로 돎)
            # -> 화면 가운데 쪽(|x| <= sign_arrive_x)에서 가장 가까운(아래) 것
            front = [s for s in p.signs if abs(s[1]) <= self.cfg.sign_backoff_x and s[3] >= self.cfg.sign_start_row]
            if front:
                return max(front, key=lambda s: s[3])
        if searching and not self.exiting:
            # 직전 회전 뒤 이미 발밑까지 지난 옆 표지판을 다음 순번으로 잡으면
            # 접근 없이 즉시 또 회전한다. 다음 표지판은 아직 도착선보다 앞에 있어야 한다.
            ok = [s for s in ok if s[2] < self.cfg.sign_arrive_far_row]
        # 여러 개면 가장 가까운 것(화면 아래), 비슷하게 가까우면 가운데에 가까운 것
        # (2026-10-09 pinky1: 둘 다 발밑일 때 오른쪽 다른 파랑으로 가다 벽 / pinky2: 가운데만 보고 고르니 바로 앞 좌회전 표지판 대신
        #  멀리 있는 유턴 구간 표지판을 골라 엉뚱한 곳에서 좌회전)
        mine = [s for s in ok if self._mine(s, searching)]
        return min(mine, key=lambda s: (round(-s[3] / 0.1), abs(s[1]))) if mine else None

    def _approach(self, sign, now, why):
        self.after_turn = self.state == SIGN_SEARCH          # 돈 뒤 다음 표지판: 먼저 제자리에서 정면으로 맞춘다
        self.target, self.t_target, self.t_align, self.aligned, self.at_sign = sign, now, None, False, False
        self.align_off = False
        self.pulse_until = self.settle_until = -1.0
        self.t_walled = None
        self.t_yolo = now
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
        if cfg.lane_role == 1 and self.plan_name == 'plan_lane1' and self.plan_i == 0 and cfg.use_coordinator:
            # R1 위: 2차선 로봇이 아직 칸에 안 들어갔으면 기다린다 (2026-10-09: 1차선이 먼저 와서 우회전하다 칸으로 가던 2차선과 충돌)
            if self.t_decide is None:
                self.t_decide = now
            if self.lock.others_flag(cfg.lane2_flag) and now - self.t_decide < cfg.lane2_wait_max_sec:
                return Command(0.0, 0.0, WAIT_JUNCTION, 'wait lane2 into pocket')
            if not self._junction(True):
                return Command(0.0, 0.0, WAIT_JUNCTION, 'junction busy')
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
        self.action, self.advance, self.at_sign = action, 0.0, False
        own = self.plan_adv[self.plan_i] if self.plan_i < len(self.plan_adv) else None
        self.advance_m = cfg.sign_advance_m if own is None else own
        self._go(SIGN_ADVANCE, now, f'{kind}:{action} +{self.advance_m:.2f}m')
        return None

    def _align_kind(self):
        kinds = [k.strip() for k in str(self.cfg.sign_align_kinds).split(',')]
        plans = [k.strip() for k in str(self.cfg.sign_align_plans).split(',')]
        return self.cfg.sign_align_deg > 0 and not self.exiting and self.plan_i < len(self.plan) \
            and self.plan[self.plan_i][0] in kinds and self.plan_name in plans

    def _square_up(self, sign, p, now, dt):
        """직우 표지판 정면 맞추기 (aligned 가 될 때까지 SIGN_APPROACH 대신). Command 를, 다 맞췄으면 None.
        1) 표지판이 옆에 있으면 제자리에서 돌아 가운데로  2) 다가간다  3) 바로 앞(가까운 끝 sign_align_near_row)에서 멈추고
        긴 축이 정면으로 보일 때까지 제자리 회전  (너무 가까워 축이 안 보이거나 발밑으로 사라졌으면 조금 후진).
        sign_align_sec 이 지나면 포기하고 예전처럼 간다.
        (2026-10-09 영상: 코너를 돌며 표지판을 옆에서 늦게 보고 바로 '도착'해 표지판 중간에서 비스듬히 우회전 -> 칸에 비뚤게 들어가
         180도도 틀어짐. 1차선도 S 를 옆으로 지나쳐 직진 대신 흰 선 따라 좌회전)"""
        cfg = self.cfg
        if self.t_align is None:
            self.t_align, self.align_shaft = now, False
            self.pulse_w, self.pulse_until, self.settle_until, self.ok_hits = 0.0, -1.0, -1.0, 0
            self.events.append((now, 'square up to sign'))
        if now - self.t_align > cfg.sign_align_sec:
            self.align_off = True
            self.events.append((now, 'align timeout'))
            return None
        # 돌면서 찍힌 화면은 흔들려 믿을 수 없다: 조금 돌고 -> 멈춰서 화면이 가라앉은 뒤 다시 잰다 (현장 요청)
        if now < self.pulse_until:
            return Command(0.0, self.pulse_w, SIGN_APPROACH, 'turn step')
        if now < self.settle_until:
            return Command(0.0, 0.0, SIGN_APPROACH, 'settle')
        def pulse(w, size, why):
            """size(0..1) 만큼 짧게 돈다. 그다음 sign_settle_sec 동안 멈춰서 다시 잰다."""
            self.ok_hits = 0
            self.pulse_w = w
            self.pulse_until = now + cfg.sign_step_min_sec + (cfg.sign_step_max_sec - cfg.sign_step_min_sec) * min(1.0, size)
            self.settle_until = self.pulse_until + cfg.sign_settle_sec
            return Command(0.0, w, SIGN_APPROACH, why)
        turn_to = lambda x: -cfg.sign_align_w if x > 0 else cfg.sign_align_w
        def skip(why):
            # 후진은 하지 않는다 (2026-10-09 1차선: S 가 발밑에 있어 '너무 가깝다'며 계속 후진) -> 맞추기를 그만두고 예전처럼 간다.
            # 'aligned'(맞췄으니 곧장)가 아니라 그냥 맞추기만 끈다: 표지판 쪽으로 계속 간다
            # (2026-10-09 pinky2: 건너뛴 뒤 곧장만 가서 오른쪽의 직우를 왼쪽으로 지나치고 옆 차선 표지판으로)
            self.align_off = True
            self.events.append((now, f'align skipped ({why})'))
            return None
        if sign is None:
            # 돌다 보면 표지판이 반대쪽으로 확 넘어가 '같은 표지판'으로 안 이어질 수 있다 -> 보이는 것 중 가운데에 가까운 것을 다시 잡는다
            # (2026-10-09 pinky2: 오른쪽에 보던 직우가 돌면서 왼쪽으로 넘어갔는데 계속 오른쪽으로 찾으며 빙빙 돎)
            seen = [sg for sg in p.signs if sg[3] >= self.target[3] - cfg.sign_track_back_row
                    and sg[2] >= self.target[2] - cfg.sign_track_back_row]
            if seen:
                sign = min(seen, key=lambda sg: abs(sg[1]))
                self.target, self.t_target = sign, now
        if sign is None:
            if now - self.t_target < cfg.sign_gone_sec:
                return Command(0.0, 0.0, SIGN_APPROACH, 'look')       # 잠깐 안 잡힌 것일 수 있다
            if now - self.t_target > cfg.sign_find_sec:
                # 못 찾았다: 도착으로 치지 말고 차선을 따라가다 다시 보이면 처음부터 (경로 순번은 그대로)
                self.events.append((now, 'square up: sign lost'))
                self.t_mode = now
                self._go(SIGN_HOLD, now, 'sign lost during alignment')
                return Command(0.0, 0.0, self.state)
            if abs(self.target[1]) > cfg.sign_face_x:
                return pulse(turn_to(self.target[1]), 0.5, 'find sign')        # 옆으로 빠졌다 -> 그쪽으로 조금씩 돈다
            return skip('sign under robot')
        x = sign[1]
        if abs(x) > (0.85 if self.align_shaft else cfg.sign_face_x):
            return pulse(turn_to(x), abs(x), 'face sign')
        if not self.align_shaft and sign[3] < cfg.sign_align_near_row:
            v, w = self._steer(Perception(ok=True, offset=x), dt, cfg.v_min)
            return Command(cfg.v_min, w, SIGN_APPROACH, 'to sign')
        if sign[2] > cfg.sign_align_far_row or sign[3] >= 0.95:
            return skip('too close')                                  # 너무 가까워 축이 안 보인다
        self.align_shaft = True
        if not p.sign_angles:
            return skip('shaft not visible')
        angles = [a for a in p.sign_angles if abs(a[0] - x) <= cfg.sign_blue_dx
                  and (len(a) < 4 or abs(a[2] - sign[2]) <= cfg.sign_track_back_row)]
        if not angles:
            return skip('no matching shaft')
        err = min(angles, key=lambda a: abs(a[0] - x))[1]
        if abs(err) <= cfg.sign_align_deg:
            # 멈춘 채로 sign_align_confirm 번 연속 맞아야 맞은 것으로 본다
            self.ok_hits += 1
            if self.ok_hits >= cfg.sign_align_confirm:
                self.aligned = True
                self.events.append((now, f'aligned {err:+.0f}deg'))
                return None
            self.settle_until = now + 0.15
            return Command(0.0, 0.0, SIGN_APPROACH, 'check')
        # 축이 오른쪽으로 기울었다 -> 오른쪽으로. 많이 틀어졌으면 길게, 조금이면 짧게
        return pulse(-cfg.sign_align_w if err > 0 else cfg.sign_align_w, abs(err) / 45.0, 'align')

    def _turned(self, deg, elapsed):
        """제자리 회전이 deg 만큼 됐나. 오도메트리가 있으면 각도로 (회전 감속 몫 turn_lead_deg 만큼 일찍 멈춘다),
        없으면 시간으로. 오도메트리가 있어도 시간의 2.5배가 지나면 끝 (바퀴가 헛돌 때)."""
        cfg = self.cfg
        t_need = math.radians(deg) / max(0.1, cfg.park_turn_w)
        if self.yaw is not None and self.yaw_start is not None:
            d = abs(math.degrees(math.atan2(math.sin(self.yaw - self.yaw_start), math.cos(self.yaw - self.yaw_start))))
            self.turn_deg = d
            return d >= deg - cfg.turn_lead_deg or elapsed > 2.5 * t_need
        self.turn_deg = None
        return elapsed >= t_need

    def _turn_text(self):
        return f'{self.turn_deg:.0f}deg (odom)' if self.turn_deg is not None else 'by time'

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
            self.backoff_run, self.backoff_done, self.t_backoff_end = 0.0, False, None
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
        if cfg.lane_role == 2:
            # 2차선 진행 중 깃발: 칸에 들어가 돌아서거나 경로를 마칠 때까지. 1차선 로봇은 R1 에서 이걸 보고 기다린다
            # 직우 표지판을 본 뒤(경로 시작)부터 칸에 들어가 돌아서거나 경로를 마칠 때까지 (현장 요청: 2차선이 직우를 보면 1차선은 R1 에서 대기)
            self._flag2(bool(self.plan) and not (self.pocket_parked or self.plan_done))
        if cfg.lane_role == 1:
            if p.signs and not self.flag_up and not self.cleared:
                self.events.append((now, 'oncoming flag up'))
            if (p.signs or self.flag_up) and not self.cleared:
                self._flag(True)
        if self.junction_held and not self.cleared:
            self._junction(True)                      # 하트비트
        exit_m = cfg.lane1_exit_m if self.plan_name == 'plan_lane1' else cfg.lane2_exit_m
        if self.plan_done and self.plan_name in ('plan_lane1', 'plan_lane2', 'plan_lane2_exit') and self.exit_run < exit_m:
            # 경로의 마지막 표지판을 지난 뒤 정해진 거리만큼 곧장 (표지판·칸 입구 선에 흔들리지 않게, 마지막 회전 직후엔 차선이 안 보인다).
            # 그다음 오른쪽 선만 따라가며 구간을 벗어난다 (2026-10-10 pinky2: 마지막 좌회전 뒤 차선이 안 보여 그 자리에서 lost)
            if self.exit_run == 0.0:
                self.events.append((now, f'straight {exit_m:.2f}m after last sign'))
            self.exit_run += cfg.v_min * dt
            self.t_seen = now
            if self.exit_run >= exit_m:
                self.t_mode = now
                self.events.append((now, 'straight done -> lanes'))
            return Command(cfg.v_min, 0.0, self.state, 'straight out')
        if self.plan_done:
            if not self.cleared:
                # 마지막 회전 직후에는 흰 선 하나만 화면 오른쪽에 보인다. 자동 분류에 맡기면
                # 그 선을 왼쪽 경계로 오인해 벽 쪽으로 꺾으므로 구간을 벗어날 때까지 오른쪽 경계로 고정한다.
                # 2차선은 하지 않는다 (2026-10-10 pinky2: 마지막 좌회전 뒤 왼쪽 선만 보이는데 오른쪽 선을 찾겠다고
                # 제자리에서 200도 돌아 뒤집힘)
                if self.plan_name == 'plan_lane1':
                    self.prefer = 'right'
                self._clear_step(now)
                if self.cleared:
                    self.prefer = ''
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
            sign = next((s for s in p.signs if s[0] in (first, 'blue') and self._mine(s)), None)
            if sign is None:
                return None
            if cfg.lane_role == 1:
                self._junction(True)                  # 줄만 선다. 기다리는 건 R1 위에서 (2026-10-09 영상: R1 을 멀리서 보자마자 서서 기다림)
            self._set_plan('plan_lane1' if cfg.lane_role == 1 else 'plan_lane2', now)
        # 회전 직후 다음 표지판이 카메라 바로 밑에 걸리면 모양 전체를 볼 수 없다.
        # 이때 먼 직우 표지를 다음 순번으로 고르지 말고, 한 번만 짧게 후진해 가까운 표지판을 다시 본다.
        # (2026-10-10 pinky1: 4cm 로는 표지판 먼 끝이 여전히 화면 0.8 에 걸려 '앞에 있는 표지판'으로 안 잡히고, 바로 전진해 밟고 지나감
        #  -> 표지판 전체가 보일 때까지(먼 끝이 sign_backoff_far_row 위로) 물러나고, 멈춰서 sign_backoff_settle_sec 동안 본 뒤 고른다)
        # 직진으로 지난 표지판은 발밑에 남아 있는 게 정상이므로, 제자리 회전을 한 뒤에만 물러난다.
        if self.state == SIGN_SEARCH and not self.exiting and not self.backoff_done and cfg.sign_backoff_m > 0 \
                and self.action in ('right', 'left'):
            # 발밑 표지판이 화면 구석(옆)에 걸려도 물러난다 (2026-10-10: R2 가 오른쪽 아래 구석 x 0.8 이라 후진 없이 전진)
            close = [s for s in p.signs if abs(s[1]) <= cfg.sign_backoff_x and s[2] >= cfg.sign_backoff_far_row]
            if close and self.backoff_run < cfg.sign_backoff_m:
                self.backoff_run += cfg.v_min * dt
                self.t_mode, self.t_backoff_end = now, None
                return Command(-cfg.v_min, 0.0, SIGN_SEARCH, 'back up to see sign')
            if self.backoff_run > 0:
                if self.t_backoff_end is None:
                    self.t_backoff_end = now
                    self.events.append((now, f'backed up {self.backoff_run:.2f}m'))
                if now - self.t_backoff_end < cfg.sign_backoff_settle_sec:
                    self.t_mode = now
                    return Command(0.0, 0.0, SIGN_SEARCH, 'look at sign')
                self.backoff_done = True
                self.t_mode = now
        sign = self._wanted_sign(p)
        if sign is not None:
            self._approach(sign, now, 'seen')
            return None
        if self.state == SIGN_SEARCH or self.exiting:
            # 다음 표지판이 아직 안 보인다 -> 곧장 간다 (갈림길의 흰 선은 좌우 구분이 틀어진다)
            if now - self.t_mode < cfg.sign_search_sec:
                self.t_seen = now
                return Command(cfg.v_min, 0.0, self.state, 'look for sign')
            if cfg.lane_role == 1 and self.plan_i < len(self.plan) and self.plan[self.plan_i][1] == 'straight':
                # 1차선의 마지막 직우(곧장 지나가기만 한다)는 못 찾아도 멈추지 않고 그대로 곧장 나간다 (lane1_exit_m)
                self.events.append((now, 'S not found -> straight out'))
                self.plan_i, self.plan_done, self.t_mode = len(self.plan), True, now
                self._go(LANE_FOLLOW, now, 'plan done (no S)')
                return None
            # 표지판을 못 찾은 채 흰 차선을 따라가면 잘못된 분기에서 이탈한다.
            # 경로와 깃발을 유지하고 정지한다.
            self.events.append((now, f'sign not found ({self.plan_name}): {self.plan_text}'))
            self._go(SIGN_HOLD, now, 'next sign not found')
            return Command(0.0, 0.0, SIGN_HOLD, 'next sign not found')
        return None

    @property
    def led(self):
        return led_color(self.state, self.in_route, self.cfg.route_color)

    # ----- 외부 명령 -----
    def start(self, now=0.0):
        if self.state in (IDLE, ESTOP, LOST, PARKED, SIGN_HOLD):
            self.pid.reset()
            self.last_offset, self.jumps = None, 0
            self.t_seen = None
            if self.state != LOST:
                self.in_route = False
                self._junction(False)
                self._flag(False)
                self._reset_role()
            self.t_go = now + (self.cfg.lane2_start_delay_sec if self.cfg.lane_role == 2 else 0.0)
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
        self._flag2(False)
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
        if self.cfg.use_coordinator and self.cfg.crosswalk_lock:
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

    def step(self, p: Perception, front_m=None, now=0.0, sides=None, lidar_ok=True, yaw=None, diag=None) -> Command:
        """sides = (왼쪽 거리, 오른쪽 거리) 라이다 측면 최소값 (없으면 None). lidar_ok=False 면 움직이지 않는다.
        yaw = 오도메트리 방향(rad, 없으면 None): 제자리 회전 각도를 잰다."""
        cfg = self.cfg
        self.yaw = yaw
        if not lidar_ok and cfg.require_lidar and self.state not in (IDLE, ESTOP, PARKED):
            # 라이다가 끊기면 앞의 장애물·옆 로봇을 못 본다 -> 바퀴를 세운다 (상태와 타이머는 그대로 둔다)
            self.t_last = now
            if not self.no_lidar:
                self.events.append((now, 'no lidar -> hold'))
            self.no_lidar = True
            return Command(0.0, 0.0, self.state, 'no lidar')
        self.no_lidar = False
        cmd = self._step(p, front_m, now)
        cmd = self._side_guard(cmd, sides, now)
        cmd = self._wall_avoid(cmd, diag, front_m)
        return self._safety_hold(cmd, front_m, sides, now)

    def _safety_hold(self, cmd, front_m, sides, now):
        """어떤 상태든 마지막에: 앞이나 옆이 너무 가까우면 전진만 멈춘다 (회전은 그대로 = 기동이 깨지지 않는다).
        (2026-10-09: 표지판 기동 중에는 장애물·옆구리 검사가 아예 안 돌아 두 대가 갈림길에서 부딪힐 뻔했다)"""
        cfg = self.cfg
        if cmd.v <= 0:
            self.held = False
            return cmd
        near_front = front_m is not None and front_m < cfg.hold_front_m
        # 옆은 보통 주행 중에만 본다. 표지판 기동·칸 안은 가벽 사이를 지나가 벽 끝이 옆 5~7cm 로 붙는다
        # (2026-10-09 pinky2: 직우 표지판으로 가다 가벽 끝을 옆 로봇으로 보고 계속 멈춰 표지판 끝까지 못 감)
        lane_states = (LANE_FOLLOW, APPROACH, CROSSING)
        near_side = cfg.side_guard and cmd.state in lane_states and not (self.pocket_mode or self.exiting) and bool(sides) and \
            any(d is not None and d < cfg.side_stop_m for d in sides)
        if not (near_front or near_side):
            self.held = False
            return cmd
        why = f'hold front {front_m:.2f}' if near_front else 'hold side ' + '/'.join('-' if d is None else f'{d:.2f}' for d in sides)
        if not self.held:
            self.events.append((now, why))
        self.held = True
        return Command(0.0, cmd.w, cmd.state, why)

    def _wall_avoid(self, cmd, diag, front_m=None):
        """라이다 앞 대각선(diag = (왼쪽 앞, 오른쪽 앞) 최소 거리)에 벽이 wall_avoid_m 보다 가까우면 반대쪽으로 꺾는다.
        카메라가 흰 가벽을 차선으로 잘못 봐도 벽에 박지 않게 (2026-10-09: 가벽이 차선 가장자리에 서 있어 두 대 모두 벽으로 감).
        제자리 회전 중(v=0)이나 표지판 정면 맞추기 중에는 건드리지 않는다."""
        cfg = self.cfg
        if not diag or cfg.wall_avoid_m <= 0 or cmd.v <= 0 or cmd.state not in (LANE_FOLLOW, APPROACH, CROSSING, SIGN_APPROACH):
            return cmd
        m = cfg.wall_avoid_m
        push = lambda d: 0.0 if d is None else max(0.0, (m - d) / m)
        bias = cfg.wall_avoid_w * (push(diag[1]) - push(diag[0]))       # 오른쪽이 가까우면 + (왼쪽으로)
        if front_m is not None and front_m < cfg.wall_turn_m and cmd.state in (LANE_FOLLOW, CROSSING):
            # 정면이 벽: 앞 대각선이 더 트인 쪽으로 (2026-10-09 pinky2: 횡단보도 뒤 왼쪽으로 꺾이는 곳에서 정면 가벽으로 감)
            left, right = (9.0 if d is None else d for d in diag)
            near = (cfg.wall_turn_m - front_m) / max(0.01, cfg.wall_turn_m - cfg.obstacle_stop_m)
            bias += cfg.wall_avoid_w * (1.0 if left > right else -1.0) * min(1.0, 0.5 + near)
        if bias == 0.0:
            return cmd
        w = max(-cfg.w_max, min(cfg.w_max, cmd.w + bias))
        return Command(cmd.v, w, cmd.state, 'wall avoid')

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
        if now < self.t_go:
            self.t_seen = now
            return Command(0.0, 0.0, self.state, f'start in {self.t_go - now:.0f}s')   # 2차선은 조금 늦게 출발
        if self.junction_held and self.state in MANEUVERS:
            self._junction(True)                      # 기동 중에도 락을 계속 쥔다 (하트비트)
        if cfg.lane_role == 1 and self.flag_up and not self.cleared:
            self._flag(True)
        if cfg.lane_role == 2 and (self.lane2_up or (self.plan and not (self.pocket_parked or self.plan_done))):
            self._flag2(True)                         # 표지판 기동 중에도 '2차선 진행 중' 깃발을 계속 올린다 (안 부르면 4초 뒤 사라진다)
        if self.state == SIGN_HOLD:
            return Command(0.0, 0.0, SIGN_HOLD, 'sign lost: reposition and START')
        if cfg.lane_role == 2 and self.plan_name == 'plan_lane2' and not self.plan_done and self._oncoming():
            # 직우에서 '직진'으로 정한 직후에 1차선 신호가 들어왔다 -> 바로 칸 쪽 우회전으로 바꾼다 (현장 요청: 타이밍이 어긋나 사고)
            late = self.state == SIGN_ADVANCE and self.plan_i == 0 and self.action == 'straight'
            just = self.state == SIGN_SEARCH and self.plan_i == 1 and now - self.t_mode < cfg.pocket_late_sec
            if late or just:
                self._junction(True)
                self._set_plan('plan_lane2_pocket', now)
                self.action = 'right'
                self.events.append((now, 'late oncoming -> pocket'))
                if just:
                    self._go(SIGN_TURN, now, 'right (late)')
        if self.state == POCKET_END:
            self.advance += cfg.v_min * dt
            if self.advance < cfg.zone_advance_m:
                return Command(cfg.v_min, 0.0, POCKET_END)
            self._go(PARK_TURN, now, f'on green +{self.advance:.2f}m')
        if self.state == PARK_TURN:
            # 오도메트리로 각도를 재며 돈다 (없으면 시간: 각도 / 회전 속도)
            if self.park_yaw_t != self.t_state:
                self.park_yaw_t, self.yaw_start = self.t_state, self.yaw
            if self._turned(cfg.park_turn_deg, now - self.t_state):
                self.events.append((now, f'park turned {self._turn_text()}'))
                if cfg.lane_role == 2:
                    self.pocket_mode, self.pocket_parked = False, True
                    self._junction(False)             # 칸 안에 들어왔다 -> 1차선 로봇이 유턴해도 된다
                    self._flag2(False)                # '2차선 진행 중' 깃발을 내린다 -> 1차선 로봇이 R1 에서 출발
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
            # 나가는 건 1차선 깃발이 내려갔을 때 (= 1차선이 구간을 다 벗어났다). 로봇이 잠깐 안 보인 것만으로는 안 나간다
            # (2026-10-09: YOLO 가 1.5초 놓치자 '지나갔다'로 보고 1차선이 아직 앞에 있는데 출발). 너무 오래면 lane2_exit_max_sec 뒤 나간다
            go = not self._oncoming() or now - self.t_state > cfg.lane2_exit_max_sec
            if now - self.t_state < cfg.exit_wait_sec or not go:
                return Command(0.0, 0.0, WAIT_EXIT, 'robot seen' if self.saw_robot else 'wait oncoming')
            self.exiting, self.t_mode, self.t_seen = True, now, now
            self._set_plan('plan_lane2_exit', now)
            self.pid.reset()
            self._go(LANE_FOLLOW, now, 'exit pocket (robot passed)' if passed else 'exit pocket (flag down)')
        if self.state == SIGN_APPROACH and self.at_sign:
            # 이미 표지판 위: 여기서 기다린다. 표지판이 다시 보여도 다시 다가가지 않는다
            # (2026-10-09 pinky1: R1 위에서 2차선을 기다리다 표지판이 옆에 다시 잡혀 다가가다 놓치고 R1 을 건너뜀)
            wait = self._arrived(now)
            if wait is not None:
                return wait
        if self.state == SIGN_APPROACH:
            # 표지판 가운데를 보고 천천히 간다. 먼 끝이 충분히 가까워지면 멈춘다.
            sign = self._wanted_sign(p, tracking=True)
            if sign is not None:
                self.target, self.t_target, self.t_yolo = sign, now, now
            if self.after_turn and not self.exiting and not self._align_kind():
                # 돈 뒤 다음 표지판이 옆에 보이면 조금 돌고 멈춰 다시 보는 식으로 정면에 놓은 뒤 다가간다 (현장 요청)
                if now < self.pulse_until:
                    return Command(0.0, self.pulse_w, SIGN_APPROACH, 'face step')
                if now < self.settle_until:
                    return Command(0.0, 0.0, SIGN_APPROACH, 'settle')
                if sign is not None and abs(sign[1]) > cfg.sign_face_x:
                    self.pulse_w = -cfg.sign_align_w if sign[1] > 0 else cfg.sign_align_w
                    self.pulse_until = now + cfg.sign_step_min_sec + (cfg.sign_step_max_sec - cfg.sign_step_min_sec) * min(1.0, abs(sign[1]))
                    self.settle_until = self.pulse_until + cfg.sign_settle_sec
                    return Command(0.0, self.pulse_w, SIGN_APPROACH, 'face sign')
                if sign is not None:
                    self.after_turn = False
            if self._align_kind() and not self.aligned and not self.align_off:
                cmd = self._square_up(sign, p, now, dt)
                if cmd is not None:
                    return cmd
            if sign is None and self.target is not None and abs(self.target[1]) > cfg.sign_arrive_x and not self.exiting:
                # 옆으로 빠진 표지판을 찾으려고 도는 중: 같은 쪽에 다시 보이면 (추적 조건과 상관없이) 바로 다시 잡는다
                # (2026-10-10 pinky2: 다시 보였는데 못 잡고 3초 내내 돌아 엉뚱한 방향을 보고 sign_hold)
                cand = [s for s in p.signs if s[1] * self.target[1] > 0 and s[3] >= cfg.sign_start_row]
                if cand:
                    sign = max(cand, key=lambda s: s[3])
                    self.target, self.t_target, self.t_yolo = sign, now, now
            arrived = False
            expected_kind = self.plan[self.plan_i][0] if self.plan_i < len(self.plan) else ''
            distance_stop = expected_kind in ('straight_right', 'any')
            # 표지판 위에서 앞이 막혔다 (가벽) = 더 못 간다 -> 여기를 표지판 끝으로 본다
            # (2026-10-09 pinky2: 직우 막대 위에서 앞 가벽이 10cm 안이라 안전 정지에 걸린 채 24초 멈춤)
            walled = front_m is not None and front_m < cfg.hold_front_m + 0.01 and self.target is not None \
                and self.target[3] >= cfg.sign_gone_row
            self.t_walled = (self.t_walled or now) if walled else None
            if distance_stop and sign is not None and not self.exiting \
                    and sign[2] >= cfg.sign_arrive_far_row and abs(sign[1]) <= cfg.sign_arrive_x:
                # 위아래로 긴 직우 표지는 끝까지 밟으면 너무 멀리 간다. 먼 끝이 기준선에 오면 멈춘다.
                self.events.append((now, f'sign end: far edge {sign[2]:.2f}'))
                arrived = True
            elif distance_stop and walled and now - self.t_walled >= cfg.sign_wall_sec:
                self.events.append((now, f'sign end: wall {front_m:.2f}m'))
                sign, arrived = None, True
            elif sign is None:
                gone = now - self.t_target
                side = abs(self.target[1]) > cfg.sign_arrive_x and not self.exiting
                if self.target[3] >= cfg.sign_gone_row and side:
                    # 옆으로 빠졌다 = 표지판 위가 아니라 옆을 지나가는 중 -> 그쪽으로 제자리에서 돌아 다시 찾는다
                    # (2026-10-09 pinky1: R1·R2 가 화면 오른쪽 아래로 빠졌는데 '도착'으로 보고 표지판 옆에서 꺾음)
                    if gone < cfg.sign_gone_sec:
                        return Command(0.0, 0.0, SIGN_APPROACH, 'look')
                    if gone >= cfg.sign_find_sec:
                        self._go(SIGN_HOLD, now, 'sign lost sideways')
                        return Command(0.0, 0.0, SIGN_HOLD, 'sign lost sideways')
                    return Command(0.0, -cfg.sign_align_w if self.target[1] > 0 else cfg.sign_align_w, SIGN_APPROACH, 'find sign')
                # HSV 표지판 자체를 매 프레임 추적하므로, 별도 축 조각을 보고
                # 전진 시간을 연장하지 않는다. 앞쪽 다른 표지판을 쫓는 원인이었다.
                if self.target[3] >= cfg.sign_gone_row and self.target[2] >= min(cfg.sign_arrive_far_row, 0.75) - 0.10:
                    arrived = gone >= cfg.sign_gone_sec
                    if not arrived:
                        return Command(0.0, 0.0, SIGN_APPROACH, 'confirm sign end')
                elif gone > cfg.lost_timeout_sec:                   # 멀리서 놓쳤다 -> 다시 찾기
                    self._go(SIGN_HOLD, now, 'sign lost before arrival')
                    return Command(0.0, 0.0, self.state)
                else:
                    return Command(0.0, 0.0, SIGN_APPROACH, 'sign missing')
            if arrived:
                self.exiting, self.at_sign = False, True
                wait = self._arrived(now)
                if wait is not None:
                    return wait
            elif self.plan_name == 'plan_lane2_exit' and self.plan_i == 0:
                # 칸에서 나올 때: 입구의 직우 표지판은 왼쪽으로 길게 보여 가운데를 보고 가면 칸 벽 선을 넘는다 -> 곧장 나간다
                return Command(cfg.v_min, 0.0, SIGN_APPROACH, 'exit straight')
            else:
                # 차선이 보이면 차선을 따라 곧게 간다 (표지판 가운데를 보고 가면 긴 직우 표지판에서 비스듬히 간다)
                if self.aligned:
                    w = 0.0                         # 표지판에 맞춰 돌았다 -> 차선이 아니라 그 방향으로 곧장
                elif self.target[3] >= cfg.sign_center_row and not self.exiting:
                    # 가까이 왔다: 표지판 한가운데 위로 올라타게 표지판 쪽으로 (차선만 따라가면 옆을 지나친다)
                    v, w = self._steer(Perception(ok=True, offset=self.target[1]), dt, cfg.v_min)
                elif p.ok and not self.exiting:
                    v, w = self._steer(p, dt, cfg.v_min)
                else:
                    v, w = self._steer(Perception(ok=True, offset=self.target[1]), dt, cfg.v_min)
                return Command(cfg.v_min, max(-cfg.sign_steer_w, min(cfg.sign_steer_w, w)), SIGN_APPROACH)
        if self.state == SIGN_ADVANCE:
            self.advance += cfg.v_min * dt
            if self.advance < self.advance_m:
                return Command(cfg.v_min, 0.0, SIGN_ADVANCE)
            if self.action in ('right', 'left'):
                self._go(SIGN_TURN, now, self.action)
            else:
                self._maneuver_done(now)
        if self.state == SIGN_TURN:
            # 멈춤(sign_pause_sec) -> 제자리 90도 (오도메트리로 각도를 잰다, 없으면 시간) -> 멈춤(sign_after_turn_sec)
            # (2026-10-09 현장: 시간으로만 돌면 90도 대신 130도까지 돌았다)
            if now - self.t_state < cfg.sign_pause_sec:
                self.yaw_start, self.t_turn_done = self.yaw, None
                return Command(0.0, 0.0, SIGN_TURN, 'pause')     # 표지판 위에서 완전히 멈춘 뒤 돈다
            if self.t_turn_done is None:
                if not self._turned(cfg.sign_turn_deg, now - self.t_state - cfg.sign_pause_sec):
                    return Command(0.0, -cfg.park_turn_w if self.action == 'right' else cfg.park_turn_w, SIGN_TURN)
                self.t_turn_done = now
                self.events.append((now, f'turned {self._turn_text()}'))
            if now - self.t_turn_done < cfg.sign_after_turn_sec:
                # 돈 뒤에도 멈춰서 다음 표지판을 본다 (2026-10-09: 돌자마자 곧장 가서 다음 표지판이 화면 옆으로 빠졌다)
                return Command(0.0, 0.0, SIGN_TURN, 'look')
            self._maneuver_done(now)
        if cfg.lane_role and any(s[3] >= cfg.sign_cw_block_row for s in p.signs):
            # 표지판이 가까이 보이면 횡단보도가 아니다 (2026-10-09 pinky1: R1 바로 앞의 흰 점들을 횡단보도로 보고
            # 정지·통과하는 동안 표지판 처리를 못 해 그냥 지나가 흰 선 따라 좌회전)
            p.crosswalk = False
            if self.state == APPROACH:
                self._release()
                self._go(LANE_FOLLOW, now, 'sign over crosswalk')
        if cfg.lane_role and self.state in (LANE_FOLLOW, SIGN_SEARCH, LOST):
            # LOST 에서도 표지판은 본다 (차선이 가벽·표지판에 가려 안 보여도 표지판이 바로 앞이면 그리로)
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
            # 횡단보도 락(한 대씩 통과)은 기본으로 끈다: 각자 3초 섰다 간다 (2026-10-10 현장: 다른 한 대를 기다리느라 1차선 진입이 늦어져 U턴 중인 2차선과 겹침)
            granted = self.lock.request(cfg.resource) if cfg.use_coordinator and cfg.crosswalk_lock else True
            waited = now - self.t_state
            if waited >= cfg.crosswalk_stop_sec and granted:
                self._go(CROSSING, now, f'waited {waited:.1f}s')
            else:
                return Command(0.0, 0.0, STOP, 'wait' if waited < cfg.crosswalk_stop_sec else 'wait lock')

        # ---- 차선을 못 볼 때 ----
        if not p.ok:
            since = 1e9 if self.t_seen is None else now - self.t_seen
            if self.state == CROSSING and now - self.t_state < cfg.crossing_sec:
                # 횡단보도 위에서 차선이 안 보이면 곧장 (직전 조향을 이어 가면 벽으로 꺾인다)
                return Command(cfg.v_min, 0.0, CROSSING, 'blind')
            if since <= cfg.lost_grace_sec:
                # 마지막 실주행에서 직전 조향이 포화(+/-1.4)된 채 유지돼 차선 끝에서
                # 한 바퀴 가까이 돌았다. 잠깐 놓쳐도 그 자리에서 기다리면 다시 잡는다.
                return Command(0.0, 0.0, self.state, 'grace stop')
            if self.prefer and since <= cfg.side_search_sec:
                # 따라가던 쪽 선을 놓쳤다 (선이 그쪽으로 급하게 꺾였다) -> 그쪽으로 제자리 회전하며 찾는다
                w = -cfg.side_spin_w if self.prefer == 'right' else cfg.side_spin_w
                return Command(0.0, w, self.state, f'search {self.prefer}')
            if since > cfg.lost_timeout_sec or self.t_seen is None:
                if self.state != LOST:
                    self._release()
                    self._go(LOST, now, 'no lane')
                return Command(0.0, 0.0, LOST)
            return Command(0.0, 0.0, self.state, 'waiting for lane')

        if self.state == LOST:
            self.pid.reset()
            self._go(LANE_FOLLOW, now, 'lane found')

        if self.state == LANE_FOLLOW:
            cooled = now - self.t_cross_done >= cfg.crosswalk_cooldown_sec
            # 진짜 횡단보도는 먼 곳에서 먼저 보이고 점점 다가온다. 처음부터 정지 행보다 가까이(발밑)에서
            # 나타난 것은 코너의 테이프 조각 같은 오검출로 보고 무시한다 (2026-10-04 코너에서 오인 정지)
            # 단 여러 프레임 계속 보이면 진짜다: 비스듬히 다가가 늦게(정지 행 바로 앞에서) 처음 잡힌 경우
            # (2026-10-09 pinky1: 0.69 에서 처음 잡혀 0.68 기준에 1% 차로 무시되고 그냥 지나감)
            self.cw_hits = self.cw_hits + 1 if (p.crosswalk and p.crosswalk_y < cfg.crosswalk_stop_row) else 0
            if self.cw_hits == 1:
                self.cw_first_y = p.crosswalk_y              # 발밑(0.72 아래)에서 처음 나타난 것은 여전히 코너 조각으로 본다
            late_ok = self.cw_hits >= cfg.crosswalk_confirm and self.cw_first_y < cfg.crosswalk_stop_row - 0.08
            if p.crosswalk and cooled and (p.crosswalk_y < cfg.crosswalk_stop_row - 0.12 or late_ok):
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
            if cfg.use_coordinator and cfg.crosswalk_lock:
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
            if cfg.use_coordinator and cfg.crosswalk_lock:
                self.lock.request(cfg.resource)       # 통과 중에도 계속 불러 자리를 유지(하트비트)
            if now - self.t_state >= cfg.crossing_sec:
                self.crossings += 1
                self.t_cross_done = now
                self._release()
                self._go(LANE_FOLLOW, now, f'crossed #{self.crossings}')
            if p.crosswalk:
                # 건너는 동안은 줄무늬 전체의 가운데를 본다. 줄무늬 하나를 차선으로 잘못 잡거나 옆 선이 가벽에 가려도
                # 줄무늬는 차선 폭만큼 깔려 있다 (2026-10-09 pinky2: 오른쪽 줄무늬를 차선 가운데로 보고 가벽으로 꺾음)
                v, w = self._steer(Perception(ok=True, offset=p.crosswalk_x), dt, min(cfg.v_cross, cfg.v_max))
            elif now - self.t_state < cfg.crossing_straight_sec or \
                    (self.t_cw_seen is not None and now - self.t_cw_seen < cfg.crossing_straight_sec):
                v, w = min(cfg.v_cross, cfg.v_max), 0.0          # 줄무늬가 발밑으로 들어갔다 -> 잠깐 곧장
            else:
                v, w = self._steer(p, dt, min(cfg.v_cross, cfg.v_max))
            return Command(v, w, self.state)

        return Command(0.0, 0.0, self.state)
