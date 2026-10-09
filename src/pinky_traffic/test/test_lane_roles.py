"""2026-10-07 표지판 맵: 바닥의 파란 양방향 표지판(우회전 / 직우)을 경로 순서대로 지나간다.
1차선이 오면(깃발) 2차선은 직우 표지판에서 우회전해 초록 칸으로 비켰다가, 나와서 우회전 -> 좌 -> 좌 로 1차선에 간다."""
import cv2
import numpy as np

from pinky_traffic.core.config import Config
from pinky_traffic.core.controller import (LaneController, LANE_FOLLOW, PARK_TURN, WAIT_EXIT, WAIT_JUNCTION,
                                            SIGN_APPROACH, SIGN_ADVANCE, SIGN_TURN, SIGN_SEARCH, POCKET_END, parse_plan)
from pinky_traffic.core.coordinator import LockManager, LocalLock
from pinky_traffic.core.detectors import HsvDetector, blue_signs
from pinky_traffic.core.perception import Perception

W, H = 320, 240
WHITE, BLUE, GREEN, CARPET = (235, 235, 235), (180, 60, 20), (90, 170, 40), (95, 100, 100)


def floor():
    return np.full((H, W, 3), CARPET, np.uint8)


def lanes(img):
    cv2.line(img, (70, H - 1), (130, int(H * 0.42)), WHITE, 10)
    cv2.line(img, (250, H - 1), (190, int(H * 0.42)), WHITE, 10)
    return img


def see(**kw):
    return Perception(ok=True, left_seen=True, right_seen=True, **kw)


def started(lock=None, **cfg):
    c = LaneController(Config(**cfg), lock)
    c.start(0.0)
    return c


# ---------- 인식 ----------

def test_green_bar_is_seen():
    img = lanes(floor())
    cv2.line(img, (95, 190), (225, 190), GREEN, 10)
    p, _, _ = HsvDetector(Config(lane_role=2)).detect(img)
    assert p.zone_seen and 0.75 < p.zone_y < 0.88


def test_prefer_right_uses_only_right_line():
    det = HsvDetector(Config(lane_role=2))
    p0, _, _ = det.detect(lanes(floor()))                                 # 두 선을 보며 차선 폭을 익힌다
    img = floor()                                                          # 갈림길: 왼쪽 선은 곧게, 오른쪽 선은 오른쪽으로 벌어진다
    cv2.line(img, (70, H - 1), (130, int(H * 0.42)), WHITE, 10)
    cv2.line(img, (300, H - 1), (250, int(H * 0.42)), WHITE, 10)
    det.prefer = 'right'
    p, _, _ = det.detect(img)
    assert p.ok and p.right_seen and not p.left_seen and p.offset > p0.offset + 0.15
    det.prefer = 'left'
    p, _, _ = det.detect(img)
    assert p.ok and p.left_seen and abs(p.offset - p0.offset) < 0.1


# ---------- 제어 ----------

def run(c, p, t0, t1, dt=0.1, front=1.0):
    t, cmd = t0, None
    while t < t1 - 1e-9:
        t += dt
        cmd = c.step(p, front, t)
    return cmd, t


def test_crosswalk_first_seen_close_is_ignored():
    c = started(crosswalk_stop_row=0.80)
    for i in range(3):
        assert c.step(see(crosswalk=True, crosswalk_y=0.79), 1.0, 0.1 * (i + 1)).state == LANE_FOLLOW


def test_stripes_taken_as_left_line_do_not_shrink_lane_width():
    det = HsvDetector(Config())
    det.detect(lanes(floor()))
    learned = dict(det.memory.width)
    img = lanes(floor())
    cv2.rectangle(img, (150, 150), (175, 215), WHITE, -1)                 # 오른쪽 선 가까이에 흰 덩어리(줄무늬)
    det.detect(img)
    for i, wdt in det.memory.width.items():
        assert wdt > 0.85 * learned[i]


def test_dashboard_lane_button_sets_role_and_stops():
    from pinky_traffic.core.driver import Driver
    d = Driver(Config(), use_dashboard=False, autostart=True)
    d.process(lanes(floor()), 1.0, 0.1)
    assert d.controller.state == LANE_FOLLOW and d.cfg.lane_role == 0
    d.command('lane2')
    d.process(lanes(floor()), 1.0, 0.2)
    assert d.cfg.lane_role == 2 and d.controller.state == 'idle' and d.state['role'] == 2


def test_in_pocket_robot_heads_for_the_green_line():
    img = lanes(floor())
    cv2.line(img, (95, 150), (225, 150), WHITE, 8)                        # 칸 끝을 막은 흰 선 (ㄷ자)
    cv2.line(img, (190, 162), (240, 162), GREEN, 10)                      # 초록 선이 오른쪽에 보인다
    det = HsvDetector(Config(lane_role=2))
    det.follow_zone = True
    p, _, _ = det.detect(img)
    assert p.ok and p.zone_seen and 0.2 < p.offset < 0.5


def test_blocked_at_corner_still_turns_in_place():
    c = started(obstacle_stop_m=0.15)
    c.step(see(), 1.0, 0.1)
    cmd = c.step(see(offset=-0.5), 0.14, 0.2)
    cmd = c.step(see(offset=-0.5), 0.14, 0.3)
    assert cmd.state == 'blocked' and cmd.v == 0 and cmd.w > 0            # 벽 앞: 전진은 안 하고 차선 쪽(왼쪽)으로 돈다
    assert c.step(see(), 0.5, 0.4).state == LANE_FOLLOW


# ---------- 2026-10-07 교행: 깃발 ----------

class Clock:
    """LockManager 시계 + '다른 로봇' a 가 깃발을 올렸다 내렸다 하는 흉내."""

    def __init__(self):
        self.t = 0.0
        self.mgr = LockManager(lease_sec=4.0, clock=lambda: self.t)

    def oncoming(self, on=True):
        self.mgr.raise_flag('oncoming', 'a', on)


def run_flag(c, clock, p, t1, front=1.0, flag=None, dt=0.1):
    cmd = None
    while clock.t < t1 - 1e-9:
        clock.t += dt
        if flag is not None:
            clock.oncoming(flag)
        cmd = c.step(p, front, clock.t)
    return cmd





# ---------- 2026-10-07 표지판 ----------

def sign(kind, x=0.0, far=0.5, near=0.7):
    return see(signs=[(kind, x, far, near)])


FAR_T, AT_T = sign('turn'), see()            # AT_* = 표지판 위에 올라타 화면에서 사라졌다
FAR_S, AT_S = sign('straight_right'), see()
SIGN = dict(v_min=0.04, park_turn_w=0.8, sign_advance_m=0.12, sign_turn_deg=90.0, sign_gone_row=0.85, sign_gone_sec=0.3,
            sign_search_sec=6.0, junction_clear_sec=8.0)


def test_plan_text_is_parsed():
    assert parse_plan('turn:right, turn:right, straight_right:straight') == \
        [('turn', 'right'), ('turn', 'right'), ('straight_right', 'straight')]


def test_blue_signs_classified_by_shape():
    img = floor()
    cv2.line(img, (60, 200), (90, 110), BLUE, 6)                          # 길쭉한 화살표 -> 직우
    cv2.rectangle(img, (200, 150), (240, 175), BLUE, -1)                  # 뭉툭한 꺾인 화살표 -> 우회전
    cv2.rectangle(img, (200, 175), (215, 200), BLUE, -1)
    p, _, _ = HsvDetector(Config(lane_role=1, sign_shape=True)).detect(img)
    kinds = sorted(s[0] for s in p.signs)
    assert kinds == ['straight_right', 'turn']
    turn = next(s for s in p.signs if s[0] == 'turn')
    assert turn[1] > 0.2 and 0.6 < turn[2] < turn[3] < 0.9                # 오른쪽에, 먼 끝 < 가까운 끝
    p, _, _ = HsvDetector(Config()).detect(img)
    assert p.signs == []                                                  # 역할이 없으면 안 본다


def through(c, near_p, far_p, t, action, clock=None, flag=None):
    """표지판 하나: 보임 -> 다가감 -> 발밑으로 사라짐 -> 더 감 -> (회전). 끝난 시각을 돌려준다."""
    step = (lambda p, t1, front=1.0: run_flag(c, clock, p, t1, front, flag)) if clock else \
        (lambda p, t1, front=1.0: run(c, p, t, t1, front=front)[0])
    cmd = step(far_p, t + 0.1)
    assert cmd.state == SIGN_APPROACH and cmd.v == 0.04
    kind = far_p.signs[0][0]
    cmd = step(sign(kind, far=0.85, near=1.0), t + 0.2)                  # 발밑까지 왔다
    assert cmd.state == SIGN_APPROACH
    cmd = step(near_p, t + 0.4)                                           # 사라진 직후: 아직 곧장 간다
    assert cmd.state == SIGN_APPROACH and cmd.w == 0 and cmd.v > 0
    cmd = step(near_p, t + 0.6)
    assert cmd.state == SIGN_ADVANCE and cmd.w == 0                       # 0.3초 동안 안 보임 = 표지판 위
    cmd = step(near_p, t + 3.6)                                           # 0.12m / 0.04 = 3초
    t = t + 3.6
    if action == 'straight':
        return cmd, t
    assert cmd.state == SIGN_TURN and (cmd.w < 0 if action == 'right' else cmd.w > 0)
    cmd = step(see(), t + 2.0)                                            # 90도 / 0.8 = 1.96초
    return cmd, t + 2.0


def test_lane1_route_right_right_straight_then_lanes():
    c = started(lane_role=1, **SIGN)
    c.step(see(), 1.0, 0.1)
    cmd = c.step(sign('turn', x=0.4), 1.0, 0.2)
    assert cmd.state == SIGN_APPROACH and c.plan_name == 'plan_lane1'
    cmd = c.step(sign('turn', x=0.4), 1.0, 0.3)
    assert cmd.w < 0                                                      # 표지판 쪽(오른쪽)으로 간다
    cmd, t = through(c, AT_T, FAR_T, 0.3, 'right')
    assert cmd.state == SIGN_SEARCH and cmd.v == 0.04 and cmd.w == 0       # 다음 표지판을 찾으며 곧장
    cmd, t = through(c, AT_T, FAR_T, t, 'right')
    cmd, t = through(c, AT_S, FAR_S, t, 'straight')
    assert c.plan_done and cmd.state == LANE_FOLLOW                       # 경로 끝 -> 흰 차선
    run(c, see(), t, t + 8.2)
    assert c.cleared


def test_lane2_ignores_turn_sign_before_straight_right():
    c = started(lane_role=2, **SIGN)
    assert c.step(FAR_T, 1.0, 0.1).state == LANE_FOLLOW and not c.plan  # 2차선의 첫 표지판은 직우
    assert c.step(FAR_S, 1.0, 0.2).state == SIGN_APPROACH and c.plan_name == 'plan_lane2'


def test_lane2_without_oncoming_goes_straight_then_left_left():
    c = started(lane_role=2, **SIGN)                                      # 1대 (서버 없음): 기다리지 않는다
    c.step(see(), 1.0, 0.1)
    cmd, t = through(c, AT_S, FAR_S, 0.1, 'straight')
    assert cmd.state == SIGN_SEARCH and not c.pocket_mode
    cmd = c.step(see(signs=[('straight_right', 0.0, 0.7, 1.0), ('turn', -0.3, 0.4, 0.6)]), 1.0, t + 0.1)
    assert cmd.state == SIGN_APPROACH and c.target[0] == 'turn'           # 발밑에 남은 직우는 무시하고 우회전 양방향으로
    t += 0.1
    cmd, t = through(c, AT_T, FAR_T, t + 0.1, 'left')
    cmd, t = through(c, AT_T, FAR_T, t, 'left')
    assert c.plan_done and cmd.state == LANE_FOLLOW


def test_sign_not_found_gives_up_route():
    c = started(lane_role=1, **SIGN)
    c.step(see(), 1.0, 0.1)
    cmd, t = through(c, AT_T, FAR_T, 0.1, 'right')
    cmd, t = run(c, see(), t, t + 5.0)
    assert cmd.state == SIGN_SEARCH and cmd.v > 0
    cmd, t = run(c, see(), t, t + 1.5)
    assert c.plan_done and c.state == LANE_FOLLOW                         # 6초 안에 못 찾으면 흰 차선으로


POCKET = dict(SIGN, use_coordinator=True, park_line_row=0.80, exit_wait_sec=2.0, pass_clear_sec=1.5, pass_front_m=0.35,
              pocket_decide_sec=1.0)


def into_pocket(c, clock):
    """깃발이 있을 때 직우 표지판 -> 우회전 -> 칸 -> 초록 앞 180도 -> WAIT_EXIT 까지."""
    run_flag(c, clock, see(), 0.1, flag=True)
    cmd, _ = through(c, AT_S, FAR_S, clock.t, 'right', clock, True)
    assert c.pocket_mode and cmd.state == LANE_FOLLOW and cmd.w == 0      # 초록이 보일 때까지 곧장
    cmd = run_flag(c, clock, see(zone_seen=True, zone_y=0.85), clock.t + 0.2, flag=True)
    assert cmd.state == POCKET_END and cmd.v == 0.04                     # 초록 선이 보이면 곧장 더 간다
    cmd = run_flag(c, clock, see(), clock.t + 2.9, flag=True)
    assert cmd.state == POCKET_END                                       # 0.12m / 0.04 = 3초
    run_flag(c, clock, see(), clock.t + 0.2, flag=True)
    assert c.state == PARK_TURN                                          # 초록 선 위에서 180도
    cmd = run_flag(c, clock, see(), clock.t + 3.1416 / 0.8 + 0.2, flag=True)
    assert cmd.state == WAIT_EXIT and not c.junction_held               # 칸 안에 들어왔다 -> 구간을 내준다


def test_lane2_waits_a_moment_on_sign_for_oncoming():
    clock = Clock()
    c = started(LocalLock(clock.mgr, 'b'), lane_role=2, **POCKET)
    run_flag(c, clock, FAR_S, 0.1)
    run_flag(c, clock, sign('straight_right', far=0.85, near=1.0), 0.2)
    cmd = run_flag(c, clock, AT_S, 0.9)
    assert cmd.state == WAIT_JUNCTION and cmd.v == 0                      # 깃발이 올지 잠깐 본다
    cmd = run_flag(c, clock, AT_S, 1.6)
    assert c.state == SIGN_ADVANCE and c.action == 'straight' and c.junction_held


def test_lane2_flag_while_deciding_goes_to_pocket():
    clock = Clock()
    c = started(LocalLock(clock.mgr, 'b'), lane_role=2, **POCKET)
    run_flag(c, clock, FAR_S, 0.1)
    run_flag(c, clock, sign('straight_right', far=0.85, near=1.0), 0.2)
    run_flag(c, clock, AT_S, 0.8)
    run_flag(c, clock, AT_S, 1.0, flag=True)
    assert c.plan_name == 'plan_lane2_pocket' and c.action == 'right'


def test_lane2_pocket_then_exit_right_left_left():
    clock = Clock()
    c = started(LocalLock(clock.mgr, 'b'), lane_role=2, **POCKET)
    into_pocket(c, clock)
    cmd = run_flag(c, clock, see(), clock.t + 3.0, flag=True)
    assert cmd.state == WAIT_EXIT                                        # 깃발이 있고 아무도 안 지나갔다
    cmd = run_flag(c, clock, see(), clock.t + 1.0, front=0.25, flag=True)   # 상대가 칸 앞을 지나간다 (라이다)
    assert cmd.state == WAIT_EXIT and c.saw_robot
    cmd = run_flag(c, clock, see(), clock.t + 1.3, flag=True)
    assert cmd.state == WAIT_EXIT                                        # 아직 1.5초가 안 됐다
    cmd = run_flag(c, clock, see(offset=1.2), clock.t + 0.4, flag=True)
    assert cmd.state == LANE_FOLLOW and c.exiting and cmd.w == 0 and cmd.v == 0.04   # 지나갔으면 깃발이 있어도 나간다 (곧장)
    assert c.plan_name == 'plan_lane2_exit'
    cmd, _ = through(c, AT_S, FAR_S, clock.t, 'right', clock, True)       # 입구의 직우 가지에서 우회전
    cmd, _ = through(c, AT_T, FAR_T, clock.t, 'left', clock, True)
    cmd, _ = through(c, AT_T, FAR_T, clock.t, 'left', clock, True)
    assert c.plan_done and cmd.state == LANE_FOLLOW


def test_lane2_exits_when_flag_goes_down():
    clock = Clock()
    c = started(LocalLock(clock.mgr, 'b'), lane_role=2, **POCKET)
    into_pocket(c, clock)
    assert run_flag(c, clock, see(), clock.t + 3.0, flag=True).state == WAIT_EXIT
    clock.oncoming(False)                                                # 1차선 로봇이 칸 입구를 지나 깃발을 내렸다
    cmd = run_flag(c, clock, see(), clock.t + 0.1)
    assert cmd.state == LANE_FOLLOW and c.exiting


def test_lane1_raises_flag_on_first_sign_and_lowers_after_clear():
    clock = Clock()
    a = started(LocalLock(clock.mgr, 'a'), lane_role=1, **POCKET)
    run_flag(a, clock, see(), 0.1)
    assert clock.mgr.flags_of_others('b') == []
    run_flag(a, clock, FAR_T, 0.2)                                        # 우회전 양방향 표지판이 보인다
    assert clock.mgr.flags_of_others('b') == ['oncoming'] and a.junction_held
    for kind, act in (('turn', 'right'), ('turn', 'right'), ('straight_right', 'straight')):
        near, far = (AT_T, FAR_T) if kind == 'turn' else (AT_S, FAR_S)
        through(a, near, far, clock.t, act, clock)
    assert a.plan_done and clock.mgr.flags_of_others('b') == ['oncoming']   # 구간을 벗어날 때까지 유지
    run_flag(a, clock, see(), clock.t + 8.2)
    assert a.cleared and clock.mgr.flags_of_others('b') == [] and not a.junction_held


def test_lane2_gives_up_pocket_if_green_never_found():
    clock = Clock()
    c = started(LocalLock(clock.mgr, 'b'), lane_role=2, pocket_blind_sec=4.0, pocket_giveup_sec=15.0, **POCKET)
    run_flag(c, clock, see(), 0.1, flag=True)
    through(c, AT_S, FAR_S, clock.t, 'right', clock, True)
    cmd = run_flag(c, clock, see(offset=0.3, zone_seen=True, zone_y=0.5), clock.t + 0.3, flag=True)
    assert cmd.w < 0                                                       # 초록이 보이면 그쪽으로
    run_flag(c, clock, see(), clock.t + 16.0, flag=True)
    assert not c.pocket_mode and not c.junction_held                       # 끝내 못 찾으면 포기
    assert c.step(see(crosswalk=True, crosswalk_y=0.5), 1.0, clock.t + 0.1).state != LANE_FOLLOW   # 횡단보도를 다시 본다


def test_two_robots_pass_each_other_at_pocket():
    now = [0.0]
    mgr = LockManager(lease_sec=4.0, clock=lambda: now[0])
    a = started(LocalLock(mgr, 'a'), lane_role=1, **POCKET)
    b = started(LocalLock(mgr, 'b'), lane_role=2, **POCKET)
    t = 0.0
    NEAR_T, NEAR_S = sign('turn', far=0.85, near=1.0), sign('straight_right', far=0.85, near=1.0)

    def tick(pa, pb, n=1, fb=1.0):
        nonlocal t
        for _ in range(n):
            t += 0.1
            now[0] = t
            ca, cb = a.step(pa, 1.0, t), b.step(pb, fb, t)
        return ca, cb

    ca, cb = tick(FAR_T, FAR_S)                                            # 1차선: 우회전 표지판 -> 깃발, 구간을 쥔다
    assert ca.state == SIGN_APPROACH and a.junction_held and cb.state == SIGN_APPROACH
    ca, cb = tick(NEAR_T, NEAR_S)
    ca, cb = tick(see(), see(), 4)                                          # 둘 다 표지판 위로 올라타 사라졌다
    assert b.plan_name == 'plan_lane2_pocket' and cb.state == SIGN_ADVANCE  # 2차선: 깃발이 있다 -> 칸으로
    assert ca.state == SIGN_ADVANCE
    ca, cb = tick(see(), see(), 52)                                         # 2차선 우회전 끝 / 1차선 첫 우회전 끝
    assert b.pocket_mode and a.plan_i == 1
    ca, cb = tick(FAR_T, see(zone_seen=True, zone_y=0.85), 2)
    ca, cb = tick(NEAR_T, see(), 1)
    ca, cb = tick(see(), see(), 31)                                         # 2차선: 초록 선 위까지 / 1차선: 두 번째 표지판 위
    assert cb.state == PARK_TURN
    ca, cb = tick(see(), see(), 45)                                         # 1차선 두 번째 우회전 / 2차선 180도
    assert cb.state == WAIT_EXIT and a.plan_i == 2
    ca, cb = tick(FAR_S, see(), 1)
    ca, cb = tick(NEAR_S, see(), 1, fb=0.25)
    ca, cb = tick(see(), see(), 34, fb=0.25)                                # 1차선이 직우를 지나며 칸 앞을 지나간다
    assert a.plan_done and cb.state == WAIT_EXIT and b.saw_robot
    ca, cb = tick(see(), see(), 16)                                         # 1.5초 동안 안 보인다 -> 지나갔다
    assert b.exiting and b.plan_name == 'plan_lane2_exit'

def test_color_signs_have_no_kind_and_count_as_next_sign():
    img = floor()
    cv2.rectangle(img, (200, 150), (240, 175), BLUE, -1)
    p, _, _ = HsvDetector(Config(lane_role=1)).detect(img)
    assert [s[0] for s in p.signs] == ['blue']                            # 색으로는 종류를 정하지 않는다
    c = started(lane_role=2, **SIGN)
    assert c.step(p, 1.0, 0.1).state == SIGN_APPROACH and c.plan_name == 'plan_lane2'   # 2차선의 첫 표지판(직우)으로 본다


def test_glare_washed_blue_is_still_blue():
    img = floor()
    cv2.rectangle(img, (200, 150), (240, 175), (230, 215, 190), -1)       # 햇빛에 하얗게 뜬 파랑 (H 약 98, S 약 45, V 230)
    p, _, _ = HsvDetector(Config(lane_role=1)).detect(img)
    assert len(p.signs) == 1


def test_exit_goes_straight_to_first_sign_even_if_it_is_off_center():
    clock = Clock()
    c = started(LocalLock(clock.mgr, 'b'), lane_role=2, **POCKET)
    into_pocket(c, clock)
    clock.oncoming(False)
    run_flag(c, clock, see(), clock.t + 2.2)                              # 칸에서 최소 2초는 기다린다
    assert c.exiting
    cmd = run_flag(c, clock, sign('straight_right', x=-0.5), clock.t + 0.3)
    assert cmd.state == SIGN_APPROACH and cmd.w == 0                      # 왼쪽으로 꺾지 않고 곧장 나간다


def test_sign_lost_far_away_is_not_arrival():
    c = started(lane_role=1, **SIGN)
    c.step(see(), 1.0, 0.1)
    c.step(FAR_T, 1.0, 0.2)                                               # 멀리서 보였다가
    cmd, t = run(c, see(), 0.2, 1.0)
    assert cmd.state == SIGN_APPROACH and cmd.v > 0                       # 잠깐 놓친 동안은 계속 다가간다
    cmd, t = run(c, see(), t, 2.0)
    assert c.state == LANE_FOLLOW and c.plan_i == 0                       # 1.5초 넘게 못 보면 도착이 아니라 놓친 것
