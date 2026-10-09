"""2026-10-07 표지판 맵: 바닥의 파란 양방향 표지판(우회전 / 직우)을 경로 순서대로 지나간다.
1차선이 오면(깃발) 2차선은 직우 표지판에서 우회전해 초록 칸으로 비켰다가, 나와서 우회전 -> 좌 -> 좌 로 1차선에 간다."""
import cv2
import numpy as np
from pathlib import Path

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


def test_crosswalk_first_seen_late_but_persistent_still_stops():
    # 2026-10-09 pinky1: 0.69 에서 처음 잡혀 무시하고 지나감. 연속으로 보이면 진짜다
    c = started(crosswalk_stop_row=0.80, crosswalk_confirm=3)
    c.step(see(crosswalk=True, crosswalk_y=0.69), 1.0, 0.1)
    c.step(see(crosswalk=True, crosswalk_y=0.70), 1.0, 0.2)
    assert c.step(see(crosswalk=True, crosswalk_y=0.71), 1.0, 0.3).state == 'approach_crosswalk'


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
    d = Driver(Config(), use_dashboard=False, autostart=True, model=False)
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
SIGN = dict(sign_align_deg=0.0, sign_arrive_far_row=1.01, plan_lane1='turn:right, turn:right, straight_right:straight', lane1_exit_m=0.0, v_min=0.04, park_turn_w=0.8, sign_advance_m=0.12, sign_turn_deg=90.0, sign_gone_row=0.85, sign_gone_sec=0.3,
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
    cmd = step(far_p if c.state != SIGN_APPROACH else see(signs=[c.target]), t + 0.1)
    assert cmd.state == SIGN_APPROACH and cmd.v == 0.04
    kind = far_p.signs[0][0]
    cmd = step(sign(kind, far=0.85, near=1.0), t + 0.2)                  # 발밑까지 왔다
    assert cmd.state == SIGN_APPROACH
    cmd = step(near_p, t + 0.4)                                           # 사라진 직후: 멈춰서 확인
    assert cmd.state == SIGN_APPROACH and cmd.w == 0 and cmd.v == 0
    cmd = step(near_p, t + 0.6)
    assert cmd.state in (SIGN_ADVANCE, SIGN_TURN) and cmd.w == 0          # 0.3초 안 보임. 추가 거리 0이면 바로 회전 준비
    if cmd.state != SIGN_TURN:
        t_end = t + 0.7 + c.advance_m / c.cfg.v_min                       # 설정된 추가 거리만큼 전진
        cmd = step(near_p, t_end)
        t = t_end
    else:
        t = t + 0.6
    if action == 'straight':
        return cmd, t
    assert cmd.state == SIGN_TURN and cmd.v == 0                          # 제자리 (멈춤 -> 회전)
    cmd = step(see(), t + 0.6)
    assert cmd.state == SIGN_TURN and cmd.v == 0 and (cmd.w < 0 if action == 'right' else cmd.w > 0)
    cmd = step(see(), t + 2.6)                                            # 0.5 + 90도 / 0.8 = 2.46초
    assert cmd.state == SIGN_TURN and cmd.v == 0 and cmd.w == 0           # 돈 뒤에도 멈춰서 다음 표지판을 본다 (0.8초)
    cmd = step(see(), t + 3.4)
    return cmd, t + 3.4


def test_lane1_route_right_right_straight_then_lanes():
    c = started(lane_role=1, **SIGN)
    c.step(see(), 1.0, 0.1)
    cmd = c.step(sign('turn', x=0.4), 1.0, 0.2)
    assert cmd.state == SIGN_APPROACH and c.plan_name == 'plan_lane1'
    cmd = c.step(sign('turn', x=0.4, near=0.6), 1.0, 0.3)
    assert abs(cmd.w) < 0.05                                              # 아직 멀다: 차선을 따라 곧게 (표지판 쪽으로 꺾지 않는다)
    cmd = c.step(sign('turn', x=0.4, near=0.7), 1.0, 0.35)
    assert cmd.w < 0                                                      # 가까이 왔다: 표지판 위에 올라타게 그쪽(오른쪽)으로
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
    assert c.plan_done and cmd.state == LANE_FOLLOW and c.prefer == 'right'


def test_sign_not_found_stops_with_route_preserved():
    c = started(lane_role=1, **SIGN)
    c.step(see(), 1.0, 0.1)
    cmd, t = through(c, AT_T, FAR_T, 0.1, 'right')
    cmd, t = run(c, see(), t, t + 5.0)
    assert cmd.state == SIGN_SEARCH and cmd.v > 0
    cmd, t = run(c, see(), t, t + 1.5)
    assert not c.plan_done and c.state == 'sign_hold' and cmd.v == 0


POCKET = dict(SIGN, plan_lane1='turn:right, turn:right, straight_right:straight', lane1_exit_m=0.0, plan_lane2_pocket='straight_right:right', use_coordinator=True, park_line_row=0.80, exit_wait_sec=2.0, pass_clear_sec=1.5, pass_front_m=0.35,
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
    assert cmd.state == WAIT_EXIT                                        # 로봇이 안 보여도 1차선 깃발이 있으면 안 나간다 (YOLO 가 잠깐 놓친 것일 수 있다)
    clock.t += 5.0                                                       # 1차선이 구간을 벗어나 깃발이 사라졌다
    cmd = run_flag(c, clock, see(offset=1.2), clock.t + 0.2)
    assert cmd.state == LANE_FOLLOW and c.exiting and cmd.w == 0 and cmd.v == 0.04   # 나간다 (곧장)
    assert c.plan_name == 'plan_lane2_exit'
    cmd, _ = through(c, AT_S, FAR_S, clock.t, 'right', clock, True)       # 입구의 직우 가지에서 우회전
    cmd, _ = through(c, AT_T, FAR_T, clock.t, 'left', clock, True)
    cmd, _ = through(c, AT_T, FAR_T, clock.t, 'left', clock, True)
    assert c.plan_done and cmd.state == LANE_FOLLOW and c.prefer == 'right'


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

    tick(see(), see())                                                     # 2차선 출발 -> '2차선 진행 중' 깃발
    ca, cb = tick(FAR_T, FAR_S)                                            # 1차선: 우회전 표지판 -> oncoming 깃발
    ca, cb = tick(NEAR_T, NEAR_S)
    ca, cb = tick(see(), see(), 4)                                          # 둘 다 표지판 위
    assert ca.state == WAIT_JUNCTION and ca.v == 0                         # 1차선은 R1 위에서 기다린다 (2차선이 아직 칸 밖)
    assert b.plan_name == 'plan_lane2_pocket'                              # 2차선은 깃발을 보고 칸으로
    ca, cb = tick(see(), see(), 68)                                         # 2차선: 더 감 + 멈춤 + 우회전 + 멈춤
    assert b.pocket_mode and ca.state == WAIT_JUNCTION
    tick(see(), see(zone_seen=True, zone_y=0.85), 2)
    ca, cb = tick(see(), see(), 32)                                         # 초록 선 위까지
    ca, cb = tick(see(), see(), 45)                                         # 180도 -> 칸 안에서 대기
    assert cb.state == WAIT_EXIT
    ca, cb = tick(see(), see(), 2)
    assert ca.state in (SIGN_ADVANCE, SIGN_TURN)                           # 이제 1차선이 R1 에서 우회전

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
    cv2.rectangle(img, (195, 165), (205, 175), BLUE, -1)                  # 진한 파랑이 조금이라도 붙어 있으면
    p, _, _ = HsvDetector(Config(lane_role=1)).detect(img)
    assert len(p.signs) == 1                                              # 하얗게 뜬 부분까지 한 표지판


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
    assert cmd.state == SIGN_APPROACH and cmd.v == 0                       # 놓친 동안 위치를 확인하려고 선다
    cmd, t = run(c, see(), t, 2.0)
    assert c.state == 'sign_hold' and c.plan_i == 0                       # 1.5초 넘게 못 보면 경로 보존·정지


def test_bluish_white_line_alone_is_not_a_sign():
    # 2026-10-09: 이 조명에서 흰 차선이 살짝 푸르게(H 100, S 25, V 220) 떠 표지판으로 잡혔다
    img = lanes(floor())
    cv2.line(img, (60, H - 1), (120, int(H * 0.45)), (230, 215, 200), 10)  # 푸르스름한 흰 선만
    p, _, _ = HsvDetector(Config(lane_role=1)).detect(img)
    assert p.signs == []
    cv2.rectangle(img, (200, 150), (240, 175), BLUE, -1)                  # 진한 파랑이 붙은 덩어리는 표지판
    cv2.rectangle(img, (240, 150), (270, 175), (230, 215, 190), -1)       # (햇빛에 뜬 부분까지 한 덩어리로)
    p, _, _ = HsvDetector(Config(lane_role=1)).detect(img)
    assert len(p.signs) == 1


def test_bluish_white_line_alone_is_not_a_sign():
    # 2026-10-09: 이 조명에서 흰 차선이 살짝 푸르게(H 100, S 25, V 220) 떠 표지판으로 잡혔다
    img = lanes(floor())
    cv2.line(img, (60, H - 1), (120, int(H * 0.45)), (230, 215, 200), 10)  # 푸르스름한 흰 선만
    p, _, _ = HsvDetector(Config(lane_role=1)).detect(img)
    assert p.signs == []
    cv2.rectangle(img, (200, 150), (240, 175), BLUE, -1)                  # 진한 파랑이 붙은 덩어리는 표지판
    cv2.rectangle(img, (240, 150), (270, 175), (230, 215, 190), -1)       # (햇빛에 뜬 부분까지 한 덩어리로)
    p, _, _ = HsvDetector(Config(lane_role=1)).detect(img)
    assert len(p.signs) == 1


def test_one_frame_offset_jump_is_ignored():
    # 2026-10-09 pinky1: 코너 꼭짓점에서 한 프레임만 offset +1.5 -> 오른쪽으로 확 꺾여 이탈
    c = started(offset_jump=0.9)
    for i in range(5):
        c.step(see(offset=0.0), 1.0, 0.1 * (i + 1))
    cmd = c.step(see(offset=1.5), 1.0, 0.6)
    assert abs(cmd.w) < 0.3                                              # 튄 한 프레임은 무시
    c.step(see(offset=1.5), 1.0, 0.7)
    cmd = c.step(see(offset=1.5), 1.0, 0.8)
    assert cmd.w < -0.5                                                  # 계속 그러면 믿는다


def test_green_blob_is_not_the_zone_bar():
    img = lanes(floor())
    cv2.circle(img, (160, 200), 14, GREEN, -1)                            # LED 초록빛 (둥근 얼룩)
    p, _, _ = HsvDetector(Config(lane_role=2)).detect(img)
    assert not p.zone_seen
    cv2.line(img, (95, 190), (225, 190), GREEN, 10)                       # 가로 띠
    p, _, _ = HsvDetector(Config(lane_role=2)).detect(img)
    assert p.zone_seen


def test_near_sign_wins_over_crosswalk():
    c = started(lane_role=1, **SIGN)
    c.step(see(), 1.0, 0.1)
    p = see(crosswalk=True, crosswalk_y=0.66, signs=[('blue', 0.14, 0.45, 0.62)])
    cmd = c.step(p, 1.0, 0.2)
    assert cmd.state == LANE_FOLLOW                                       # 횡단보도로 서지 않고, 아직 멀어 차선을 따라간다
    p = see(crosswalk=True, crosswalk_y=0.70, signs=[('blue', 0.14, 0.5, 0.68)])
    cmd = c.step(p, 1.0, 0.3)
    assert cmd.state == SIGN_APPROACH                                     # 횡단보도가 아니라 표지판으로


def test_sign_turn_starts_with_full_stop():
    c = started(lane_role=1, sign_pause_sec=0.5, **{k: v for k, v in SIGN.items()})
    c.step(see(), 1.0, 0.1)
    c.step(FAR_T, 1.0, 0.2)
    c.step(sign('turn', far=0.85, near=1.0), 1.0, 0.3)
    run(c, see(), 0.3, 0.7)                                               # 사라지고 0.3초 -> 표지판 위
    cmd, t = run(c, see(), 0.7, 3.8)                                      # 0.12m 더 감
    assert c.state == SIGN_TURN
    t0 = c.t_state
    cmd = c.step(see(), 1.0, t0 + 0.2)
    assert cmd.v == 0 and cmd.w == 0                                      # 0.5초 동안은 완전히 멈춤
    cmd = c.step(see(), 1.0, t0 + 0.7)
    assert cmd.v == 0 and cmd.w < 0                                       # 그다음 제자리 우회전


def test_plan_step_can_have_its_own_advance():
    plan, adv = parse_plan('turn:right, turn:right:0.12, straight_right', True)
    assert plan == [('turn', 'right'), ('turn', 'right'), ('straight_right', 'straight')] and adv == [None, 0.12, None]
    c = started(lane_role=1, **dict(SIGN, sign_advance_m=0.07, plan_lane1='turn:right, turn:right:0.12'))
    c.step(see(), 1.0, 0.1)
    c.step(FAR_T, 1.0, 0.2)
    c.step(sign('turn', far=0.85, near=1.0), 1.0, 0.3)
    run(c, see(), 0.3, 0.7)
    assert c.state == SIGN_ADVANCE and c.advance_m == 0.07               # R1: 기본값
    c.plan_i = 1
    c._arrived(1.0)
    assert c.advance_m == 0.12                                           # R2 (가벽 쪽): 경로에 적은 값


def test_lane2_turns_in_place_until_straight_right_sign_looks_straight():
    # 2026-10-09 pinky2: 코너를 넓게 돌아 직우 표지판에 비스듬히 들어갔다 -> 바로 앞에서 멈추고 축이 정면으로 보일 때까지 돈다
    c = started(lane_role=2, **dict(SIGN, sign_align_deg=8.0, sign_settle_sec=0.0, sign_step_min_sec=0.0, sign_step_max_sec=0.0, sign_align_confirm=1))
    c.step(see(), 1.0, 0.1)
    c.step(FAR_S, 1.0, 0.2)
    assert c.state == SIGN_APPROACH
    far = see(signs=[('blue', -0.3, 0.5, 0.7)], sign_angles=[(-0.3, 30.0)])
    cmd = c.step(far, 1.0, 0.3)
    assert cmd.v > 0 and cmd.w > 0                                       # 아직 멀다: 표지판 쪽으로 (왼쪽)
    close = lambda err: see(signs=[('blue', -0.1, 0.3, 0.9)], sign_angles=[(-0.1, err), (0.8, -40.0)])
    cmd = c.step(close(30.0), 1.0, 0.4)
    assert cmd.v == 0 and cmd.w < 0                                      # 바로 앞: 멈추고, 축이 오른쪽으로 기울었다 -> 오른쪽으로 돈다
    cmd = c.step(close(-25.0), 1.0, 0.5)
    assert cmd.v == 0 and cmd.w > 0
    cmd = c.step(close(4.0), 1.0, 0.6)
    assert cmd.v > 0 and cmd.w == 0 and c.aligned                        # 맞았다 -> 그 방향으로 곧장
    cmd = c.step(close(30.0), 1.0, 0.7)
    assert cmd.v > 0 and cmd.w == 0                                      # 한 번 맞추면 다시 돌지 않는다
    c2 = started(lane_role=2, **dict(SIGN, sign_align_deg=8.0, sign_align_sec=1.0))
    c2.step(see(), 1.0, 0.1)
    c2.step(FAR_S, 1.0, 0.2)
    cmd, _ = run(c2, close(30.0), 0.2, 1.6)
    assert cmd.v > 0                                                     # 끝내 못 맞추면 그냥 간다


def test_waiting_on_sign_does_not_approach_again():
    # 2026-10-09 pinky1: R1 위에서 2차선을 기다리다 표지판이 옆에 다시 잡혀 다가가다 놓치고 R1 을 건너뜀
    clock = Clock()
    a = started(LocalLock(clock.mgr, 'a'), lane_role=1, **POCKET)
    clock.mgr.raise_flag('lane2', 'b', True)
    run_flag(a, clock, see(), 0.1)
    run_flag(a, clock, FAR_T, 0.2)
    run_flag(a, clock, sign('turn', far=0.85, near=1.0), 0.3)
    cmd = run_flag(a, clock, see(), 0.8)
    assert cmd.state == WAIT_JUNCTION and cmd.v == 0
    clock.mgr.raise_flag('lane2', 'b', True)
    cmd = run_flag(a, clock, sign('turn', x=0.7, far=0.6, near=0.95), 1.5)   # 기다리는 동안 표지판이 옆에 다시 보인다
    assert cmd.state == WAIT_JUNCTION and cmd.v == 0 and a.state == SIGN_APPROACH


def test_shaft_angle_from_blue_mask():
    from pinky_traffic.core.detectors import blue_angles
    cfg = Config(sign_horizon_row=0.30)
    m = np.zeros((H, W), np.uint8)
    cv2.line(m, (160, 230), (160, 150), 255, 12)                         # 정면 방향 축
    assert abs(blue_angles(m, cfg)[0][1]) < 3
    m[:] = 0
    cv2.line(m, (120, 230), (220, 150), 255, 12)                         # 오른쪽으로 기운 축
    assert blue_angles(m, cfg)[0][1] > 20


def test_lane1_missing_sign_holds_flag_and_stops():
    clock = Clock()
    a = started(LocalLock(clock.mgr, 'a'), lane_role=1, **POCKET)
    run_flag(a, clock, see(), 0.1)
    run_flag(a, clock, FAR_T, 0.2)
    through(a, AT_T, FAR_T, clock.t, 'right', clock)
    run_flag(a, clock, see(), clock.t + 6.5)
    assert not a.plan_done and a.state == 'sign_hold' and clock.mgr.flags_of_others('b') == ['oncoming']
    run_flag(a, clock, see(), clock.t + 8.2)
    assert not a.cleared and clock.mgr.flags_of_others('b') == ['oncoming']


def test_lane_width_not_learned_on_crosswalk_and_near_not_narrower_than_far():
    # 2026-10-09 pinky2: 횡단보도 줄무늬로 좁은 폭을 배워, 지나간 뒤 오른쪽 선만 보일 때 그 선 위를 달렸다
    from pinky_traffic.core.perception import LaneMemory, lane_from_masks
    cfg = Config()
    mem = LaneMemory()
    left, right, cw = (np.zeros((H, W), np.uint8) for _ in range(3))
    cv2.line(left, (140, H - 1), (150, 100), 255, 8)                     # 줄무늬 두 개 (좁다)
    cv2.line(right, (180, H - 1), (170, 100), 255, 8)
    cw[150:230, 100:220] = 255
    lane_from_masks(left, right, cw, cfg, mem, True)
    assert mem.width == {}                                               # 횡단보도 위에서는 안 배운다
    mem.width = {0: 60.0, 5: 200.0}                                      # 가까운 행이 잘못 좁게 배워졌다
    only_right = np.zeros((H, W), np.uint8)
    cv2.line(only_right, (165, H - 1), (200, 100), 255, 8)
    p = lane_from_masks(np.zeros((H, W), np.uint8), only_right, None, cfg, mem)
    assert p.offset < -0.4                                               # 오른쪽 선의 왼쪽(차선 안쪽)을 목표로


def test_square_up_turns_in_steps_and_checks_while_still():
    # 현장 요청: 돌면서 찍힌 화면은 흔들린다 -> 조금 돌고, 멈춰서 다시 재고, 멈춘 채 3번 연속 맞아야 끝
    c = started(lane_role=2, **dict(SIGN, sign_align_deg=8.0, sign_settle_sec=0.5, sign_step_min_sec=0.15,
                                    sign_step_max_sec=0.6, sign_align_confirm=3))
    c.step(see(), 1.0, 0.1)
    c.step(FAR_S, 1.0, 0.2)
    close = lambda err: see(signs=[('blue', 0.0, 0.3, 0.9)], sign_angles=[(0.0, err)])
    cmd = c.step(close(45.0), 1.0, 0.3)
    assert cmd.v == 0 and cmd.w < 0 and cmd.reason == 'align'           # 많이 틀어짐 -> 0.6초 돈다
    assert c.step(close(45.0), 1.0, 0.8).w < 0                           # 도는 중에는 화면을 안 믿는다
    cmd = c.step(close(-30.0), 1.0, 1.0)
    assert cmd.v == 0 and cmd.w == 0 and cmd.reason == 'settle'          # 멈춰서 가라앉기를 기다린다
    cmd = c.step(close(3.0), 1.0, 1.5)
    assert cmd.w == 0 and not c.aligned                                  # 한 번 맞음: 아직 확인 중
    c.step(close(3.0), 1.0, 1.7)
    cmd = c.step(close(3.0), 1.0, 1.9)
    assert c.aligned and cmd.v > 0 and cmd.w == 0                        # 3번 연속 -> 곧장


def test_lane1_does_not_stop_far_from_r1_and_waits_on_it_while_lane2_busy():
    # 2026-10-09 영상: 1차선이 R1 을 멀리서 보자마자 서서 기다림 -> R1 까지 가서, 2차선이 직우를 본 뒤 칸에 들어갈 때까지 대기
    clock = Clock()
    a = started(LocalLock(clock.mgr, 'a'), lane_role=1, **POCKET)
    b = started(LocalLock(clock.mgr, 'b'), lane_role=2, **POCKET)
    assert clock.mgr.flags_of_others('a') == []                          # 2차선은 직우를 보기 전에는 깃발이 없다
    clock.t = 0.1
    b.step(FAR_S, 1.0, clock.t)
    assert clock.mgr.flags_of_others('a') == ['lane2']                   # 직우를 봤다 -> 1차선은 R1 에서 대기
    clock.mgr.request('junction', 'b')                                   # 2차선이 구간을 쥐고 있다
    cmd = run_flag(a, clock, FAR_T, 0.3)
    assert cmd.state == SIGN_APPROACH and cmd.v > 0                      # 멀리서 서지 않고 R1 까지 간다
    clock.mgr.raise_flag('lane2', 'b', True)
    run_flag(a, clock, sign('turn', far=0.85, near=1.0), 0.4)
    cmd = run_flag(a, clock, see(), 0.9)
    assert cmd.state == WAIT_JUNCTION and cmd.reason == 'wait lane2 into pocket'



def test_sign_leaving_sideways_is_not_arrival():
    # 2026-10-09 pinky1: R1·R2 가 화면 오른쪽 아래로 빠졌는데 '도착'으로 보고 표지판 옆에서 꺾음
    c = started(lane_role=1, **SIGN)
    c.step(see(), 1.0, 0.1)
    c.step(FAR_T, 1.0, 0.2)
    c.step(sign('turn', x=0.4, far=0.7, near=0.95), 1.0, 0.25)
    c.step(sign('turn', x=0.75, far=0.8, near=1.0), 1.0, 0.3)
    cmd, t = run(c, see(), 0.3, 1.0)
    assert c.state == SIGN_APPROACH and cmd.v == 0 and cmd.w < 0         # 오른쪽으로 빠졌다 -> 제자리에서 오른쪽으로 돌아 찾는다
    c.step(sign('turn', x=0.5, far=0.8, near=1.0), 1.0, 1.05)            # 돌면서 다시 보인다
    c.step(sign('turn', x=0.2, far=0.8, near=1.0), 1.0, 1.1)
    run(c, see(), 1.1, 1.6)
    assert c.state == SIGN_ADVANCE                                        # 가운데에서 아래로 사라짐 = 도착


def test_white_wall_face_is_not_lane_or_crosswalk():
    # 2026-10-09 pinky2: 횡단보도 옆에 세운 흰 가벽 밑면을 차선·횡단보도로 봤다
    img = lanes(floor())
    pts = np.array([[200, 100], [319, 100], [319, 239], [260, 239]], np.int32)
    cv2.fillPoly(img, [pts], WHITE)                                      # 오른쪽에 크게 붙은 흰 면
    p, masks, _ = HsvDetector(Config()).detect(img)
    assert not masks['right'][200:, 290:].any() and not masks['crosswalk'][200:, 290:].any()
    img = floor()
    for x in (90, 150, 210):
        cv2.rectangle(img, (x, 150), (x + 30, 215), WHITE, -1)          # 횡단보도 줄무늬는 벽이 아니다
    p, masks, _ = HsvDetector(Config()).detect(img)
    assert masks['crosswalk'].any()


def test_square_up_reacquires_sign_that_jumped_to_other_side():
    # 2026-10-09 pinky2: 오른쪽에 보던 직우가 돌면서 왼쪽으로 넘어갔는데 계속 오른쪽으로 찾으며 빙빙 돎
    c = started(lane_role=2, **dict(SIGN, sign_align_deg=8.0, sign_settle_sec=0.0, sign_step_min_sec=0.0,
                                    sign_step_max_sec=0.0, sign_align_confirm=1))
    c.step(see(), 1.0, 0.1)
    c.step(sign('straight_right', x=0.4, near=0.7), 1.0, 0.2)
    c.step(sign('straight_right', x=0.6, far=0.4, near=1.0), 1.0, 0.3)
    cmd = c.step(see(signs=[('blue', -0.5, 0.43, 1.0)], sign_angles=[(-0.5, -15.0)]), 1.0, 0.4)
    assert cmd.w > 0                                                     # 왼쪽으로 넘어간 표지판을 다시 잡아 왼쪽으로
    cmd, _ = run(c, see(), 0.4, 4.0)
    assert c.state == 'sign_hold' and c.plan_i == 0                      # 끝내 못 찾으면 경로를 보존하고 정지


def test_not_arrived_while_blue_still_ahead():
    # 현재 표지판을 잃었을 때 방향 마스크 한 조각만 보고 맹목적으로 더 가지 않는다.
    c = started(lane_role=2, **SIGN)
    c.step(see(), 1.0, 0.1)
    c.step(FAR_S, 1.0, 0.2)
    c.step(sign('straight_right', far=0.4, near=1.0), 1.0, 0.3)
    shaft = see(sign_angles=[(0.1, 0.0, 0.4, 1.0)])
    cmd, t = run(c, shaft, 0.3, 1.5)
    assert c.state == SIGN_APPROACH and cmd.v == 0
    run(c, see(), 1.5, 2.0)
    assert c.state != SIGN_APPROACH                                       # 파랑이 다 지나가면 도착
    # 우회전 표지판에서는 이 규칙을 안 쓴다 (2026-10-09: 앞쪽 다른 표지판 파랑 때문에 벽 앞까지 감)
    c1 = started(lane_role=1, **SIGN)
    c1.step(see(), 1.0, 0.1)
    c1.step(FAR_T, 1.0, 0.2)
    c1.step(sign('turn', far=0.85, near=1.0), 1.0, 0.3)
    run(c1, shaft, 0.3, 0.8)
    assert c1.state == SIGN_ADVANCE

def test_lost_robot_still_goes_to_sign_in_front():
    # 2026-10-09 pinky2: 차선이 가벽·직우 표지판에 가려 LOST 가 된 채 표지판을 바로 앞에 두고 15초 멈춤
    c = started(lane_role=2, **SIGN)
    c.step(see(), 1.0, 0.1)
    run(c, Perception(), 0.1, 3.0)
    assert c.state == 'lost'
    cmd = c.step(Perception(signs=[('blue', 0.05, 0.3, 1.0)]), 1.0, 3.1)
    assert c.state == SIGN_APPROACH and c.plan_name == 'plan_lane2'


def test_lane1_follows_right_line_past_pocket_entrance_after_s():
    # 2026-10-09: S 직진 뒤 차선 따라가기로 돌아가자 칸 입구 선을 따라 좌회전
    c = started(lane_role=1, **SIGN)
    c.step(see(), 1.0, 0.1)
    cmd, t = through(c, AT_T, FAR_T, 0.1, 'right')
    cmd, t = through(c, AT_T, FAR_T, t, 'right')
    cmd, t = through(c, AT_S, FAR_S, t, 'straight')
    c.step(see(), 1.0, t + 0.1)
    assert c.plan_done and c.prefer == 'right'
    run(c, see(), t + 0.1, t + 8.5)
    assert c.cleared and c.prefer == ''


def test_sign_turn_measures_angle_with_odometry():
    # 2026-10-09 현장: 시간으로만 돌면 90도 대신 130도까지 돌았다 -> 오도메트리로 각도를 재서 멈춘다
    import math
    c = started(lane_role=1, **SIGN)
    c.step(see(), 1.0, 0.1, yaw=0.0)
    c.step(FAR_T, 1.0, 0.2, yaw=0.0)
    c.step(sign('turn', far=0.85, near=1.0), 1.0, 0.3, yaw=0.0)
    t, yaw = 0.3, 0.0
    while c.state != SIGN_TURN:
        t += 0.1
        c.step(see(), 1.0, t, yaw=yaw)
    while True:
        t += 0.05
        cmd = c.step(see(), 1.0, t, yaw=yaw)
        if cmd.w == 0 and cmd.reason == 'look':
            break
        yaw += cmd.w * 0.05 * 1.5                                        # 실제로는 명령보다 1.5배 빨리 돈다
    assert 80 <= abs(math.degrees(yaw)) <= 95                            # 시간으로 돌았으면 135도


def test_wall_ahead_on_sign_counts_as_sign_end():
    # 2026-10-09 pinky2: 직우 막대 위에서 앞 가벽이 10cm 안이라 안전 정지에 걸린 채 24초 멈춤
    c = started(lane_role=2, **dict(SIGN, plan_lane2='straight_right:straight'))
    c.step(see(), 1.0, 0.1)
    c.step(FAR_S, 1.0, 0.2)
    on = sign('straight_right', far=0.5, near=1.0)
    cmd, t = run(c, on, 0.2, 1.5, front=0.09)
    assert c.state in (SIGN_ADVANCE, SIGN_TURN)                           # 막힌 채 1초 -> 표지판 끝


def test_crosswalk_kept_with_two_stripes_once_seen():
    # 2026-10-09 pinky2: 정지 행 직전에 바깥 줄무늬가 화면 옆에 닿아 2개만 남자 횡단보도를 놓침
    from pinky_traffic.core.perception import stripes_are_crosswalk
    cfg = Config()
    two = [(90, 130, 50, 50), (180, 132, 45, 50)]
    assert not stripes_are_crosswalk(two, cfg, 240)
    assert stripes_are_crosswalk(two, cfg, 240, 2)


def test_lidar_wall_on_front_right_steers_left():
    # 2026-10-09: 가벽이 차선 가장자리에 서 있어 카메라가 속아 두 대 모두 벽으로 감 -> 라이다 앞 대각선 벽에서 비킨다
    c = started()
    c.step(see(), 1.0, 0.1)
    cmd = c.step(see(), 1.0, 0.2, diag=(0.6, 0.07))
    assert cmd.w > 0.3 and cmd.reason == 'wall avoid'
    cmd = c.step(see(), 1.0, 0.3, diag=(0.07, 0.6))
    assert cmd.w < -0.3
    assert abs(c.step(see(), 1.0, 0.4, diag=(0.5, 0.5)).w) < 0.05



def test_lane1_goes_straight_after_r2_then_lanes():
    # 현장 요청: 두 번째 표지판(R2)을 지나면 정해진 거리 곧장 간 뒤 차선 (S 표지판 판단에 기대지 않는다)
    c = started(lane_role=1, **dict(SIGN, plan_lane1='turn:right, turn:right', lane1_exit_m=0.20))
    c.step(see(), 1.0, 0.1)
    cmd, t = through(c, AT_T, FAR_T, 0.1, 'right')
    cmd, t = through(c, AT_T, FAR_T, t, 'right')
    cmd, t = run(c, see(offset=0.8), t, t + 3.0)
    assert c.plan_done and cmd.reason == 'straight out' and cmd.w == 0   # 0.2m / 0.04 = 5초 동안 차선 무시하고 곧장
    cmd, t = run(c, see(offset=0.8), t, t + 2.5)
    assert cmd.reason != 'straight out' and cmd.w < 0 and c.prefer == 'right'   # 그다음 오른쪽 선 따라


def test_wall_straight_ahead_turns_to_open_side():
    c = started(obstacle_stop_m=0.10)
    c.step(see(), 1.0, 0.1)
    cmd = c.step(see(), 0.16, 0.2, diag=(0.6, 0.25))
    assert cmd.w > 0.3                                                   # 왼쪽이 트였다 -> 왼쪽으로


def test_after_align_skip_still_steers_to_sign():
    # 2026-10-09 pinky2: 정면 맞추기를 건너뛴 뒤 곧장만 가서 오른쪽의 직우를 왼쪽으로 지나치고 옆 차선 표지판으로 감
    c = started(lane_role=2, **dict(SIGN, sign_align_deg=8.0))
    c.step(see(), 1.0, 0.1)
    c.step(sign('straight_right', x=0.3, far=0.8, near=1.0), 1.0, 0.2)   # 처음부터 너무 가깝다 -> 맞추기 건너뜀
    cmd = c.step(sign('straight_right', x=0.3, far=0.8, near=1.0), 1.0, 0.25)
    assert c.align_off and not c.aligned
    cmd = c.step(sign('straight_right', x=0.4, far=0.8, near=1.0), 1.0, 0.3)
    assert cmd.v > 0 and cmd.w < 0                                       # 그래도 오른쪽의 표지판 쪽으로 꺾는다


def test_nearest_sign_wins_over_centered_far_one():
    # 2026-10-09 pinky2: 첫 좌회전 뒤 바로 앞(오른쪽 아래) 좌회전 표지판 대신 멀리 가운데의 유턴 구간 표지판을 골랐다
    c = started(lane_role=2, **SIGN)
    c.plan, c.plan_adv, c.plan_i, c.plan_name = [('straight_right', 'straight'), ('turn', 'left'), ('turn', 'left')], [None] * 3, 2, 'plan_lane2'
    c._go(SIGN_SEARCH, 0.0)
    p = see(signs=[('blue', 0.18, 0.4, 0.66), ('blue', 0.44, 0.7, 1.0)])
    assert c._wanted_sign(p)[1] == 0.44
    p = see(signs=[('blue', 0.6, 0.5, 1.0), ('blue', -0.1, 0.5, 0.98)])
    assert c._wanted_sign(p)[1] == -0.1                                  # 둘 다 발밑이면 가운데 쪽


def test_next_sign_search_ignores_blue_mark_already_under_robot():
    # 23:01 실주행: R1 회전 직후 발밑의 직우 표지를 R2로 잡고 접근 없이 바로 또 회전했다.
    c = started(lane_role=1, **dict(SIGN, sign_arrive_far_row=0.75))
    c.plan, c.plan_adv, c.plan_i, c.plan_name = [('turn', 'right'), ('turn', 'right')], [None] * 2, 1, 'plan_lane1'
    c._go(SIGN_SEARCH, 0.0)
    p = see(signs=[('blue', 0.05, 0.76, 1.0), ('blue', 0.30, 0.50, 0.72)])
    target = c._wanted_sign(p)
    assert target is not None and target[2] == 0.50


def test_next_sign_too_close_backs_up_before_choosing_far_sign():
    # 1차선 첫 회전 직후 R2가 카메라 바로 밑에 있어 모양이 잘리면, 먼 직우를 고르기 전에 시야를 확보한다.
    c = started(lane_role=1, **dict(SIGN, sign_arrive_far_row=0.75, sign_backoff_m=0.04))
    c.plan, c.plan_adv, c.plan_i, c.plan_name = [('turn', 'right'), ('turn', 'right')], [None] * 2, 1, 'plan_lane1'
    c._go(SIGN_SEARCH, 0.0)
    c.action = 'right'                                                   # 방금 제자리 회전을 했다
    p = see(signs=[('turn', 0.05, 0.76, 1.0), ('straight_right', 0.10, 0.45, 0.70)])
    cmd = c.step(p, 1.0, 0.1)
    assert cmd.state == SIGN_SEARCH and cmd.v < 0 and c.target is None
    cmd, t = run(c, p, 0.1, 1.5)
    assert c.backoff_run >= 0.04 and cmd.v == 0 and cmd.reason == 'look at sign'   # 물러난 뒤 멈춰서 본다
    cmd, t = run(c, p, t, t + 1.0)
    assert c.backoff_done and c.state == SIGN_APPROACH and c.target[1] == 0.05   # 바로 앞 표지판을 고른다


def test_backoff_until_whole_sign_visible_and_kind_ignored():
    # 2026-10-10 pinky1: R1 을 돈 뒤 R2 가 발밑에 잘려 보이고, 옆에서 본 화살표라 모양이 '직우'로 분류돼 놓쳤다
    c = started(lane_role=1, **dict(SIGN, sign_backoff_m=0.12, sign_backoff_far_row=0.65, sign_backoff_settle_sec=0.8))
    c.plan, c.plan_adv, c.plan_i, c.plan_name = [('turn', 'right'), ('turn', 'right')], [None] * 2, 1, 'plan_lane1'
    c._go(SIGN_SEARCH, 0.0)
    c.action = 'right'
    cut = see(signs=[('straight_right', 0.0, 0.80, 1.0)])
    cmd, t = run(c, cut, 0.0, 1.5)
    assert cmd.v < 0 and 0.05 < c.backoff_run < 0.07                     # 4cm 를 넘어 계속 물러난다
    whole = see(signs=[('straight_right', 0.0, 0.60, 0.92)])              # 표지판 전체가 보인다
    cmd, t = run(c, whole, t, t + 0.5)
    assert cmd.v == 0 and cmd.reason == 'look at sign'
    cmd, t = run(c, whole, t, t + 0.6)
    assert c.state == SIGN_APPROACH and c.target[0] == 'straight_right'   # 종류와 상관없이 바로 앞의 것


def test_turn_sign_waits_until_blue_disappears():
    # 짧은 좌·우회전 표지는 거리선에서 돌면 아직 코앞에 파랑이 남는다.
    c = started(lane_role=1, **dict(SIGN, sign_arrive_far_row=0.75))
    c.step(see(), 1.0, 0.1)
    c.step(FAR_T, 1.0, 0.2)
    c.step(sign('turn', far=0.6, near=1.0), 1.0, 0.3)
    assert c.state == SIGN_APPROACH
    c.step(sign('turn', far=0.78, near=1.0), 1.0, 0.4)
    assert c.state == SIGN_APPROACH                                      # 파랑이 보이는 동안 계속 전진
    run(c, see(), 0.4, 0.8)
    assert c.state == SIGN_ADVANCE                                       # 완전히 사라진 뒤 도착


def test_straight_right_sign_still_uses_far_edge_distance():
    c = started(lane_role=2, **dict(SIGN, sign_arrive_far_row=0.75,
                                    plan_lane2='straight_right:straight'))
    c.step(see(), 1.0, 0.1)
    c.step(FAR_S, 1.0, 0.2)
    c.step(sign('straight_right', far=0.78, near=1.0), 1.0, 0.3)
    assert c.state in (WAIT_JUNCTION, SIGN_ADVANCE)


def test_lane2_recorded_signs_and_blue_wall():
    # 22:37 실주행: YOLO가 멀리 있는 표지판만 잡아도 직우 전체와 바로 앞
    # 좌회전 표지판이 색 마스크에서 살아 있어야 한다.
    root = Path(__file__).parent / 'fixtures'
    d = HsvDetector(Config(lane_role=2))
    first, _, _ = d.detect(cv2.imread(str(root / 'lane2_first_sign.jpg')))
    assert first.signs and abs(first.signs[0][1]) < 0.4
    two, _, _ = d.detect(cv2.imread(str(root / 'lane2_two_signs.jpg')))
    assert any(s[3] > 0.9 and abs(s[1]) < 0.5 for s in two.signs)
    assert any(0.4 < s[3] < 0.6 and abs(s[1]) < 0.5 for s in two.signs)
    wall, _, _ = d.detect(cv2.imread(str(root / 'lane2_blue_wall.jpg')))
    assert not wall.signs


def test_tracking_does_not_jump_from_near_sign_to_distant_one():
    c = started(lane_role=2, **SIGN)
    c.plan, c.plan_adv, c.plan_i, c.plan_name = [('turn', 'left')], [None], 0, 'plan_lane2'
    c._go(SIGN_SEARCH, 0.0)
    c.step(see(signs=[('blue', 0.30, 0.58, 1.0)]), 1.0, 0.1)
    assert c.target[3] == 1.0
    p = see(signs=[('blue', 0.28, 0.40, 0.53)])
    assert c._wanted_sign(p, tracking=True) is None
    assert c.step(p, 1.0, 0.2).v == 0


def test_after_backoff_picks_near_front_sign_not_far_side_one():
    # 2026-10-10 pinky1: 9cm 물러난 뒤 R2 가 왼쪽 아래(가까운 끝 0.84)에 보였는데 오른쪽 먼 표지판(turn 분류)을 골라 오른쪽으로 돎
    c = started(lane_role=1, **SIGN)
    c.plan, c.plan_adv, c.plan_i, c.plan_name = [('turn', 'right'), ('turn', 'right')], [None] * 2, 1, 'plan_lane1'
    c._go(SIGN_SEARCH, 0.0)
    c.backoff_run = 0.09
    p = see(signs=[('straight_right', -0.25, 0.64, 0.84), ('turn', 0.87, 0.56, 0.70)])
    assert c._wanted_sign(p)[1] == -0.25
