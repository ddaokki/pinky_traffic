"""2026-10-04 맵: 1차선 = 파란 선 유턴, 2차선 = 초록 칸으로 빠졌다가 나옴, 유턴 구간은 한 대씩."""
import cv2
import numpy as np

from pinky_traffic.core.config import Config
from pinky_traffic.core.controller import (LaneController, LANE_FOLLOW, PARK_TURN, WAIT_EXIT, WAIT_JUNCTION,
                                            POCKET_ADVANCE, POCKET_TURN)
from pinky_traffic.core.coordinator import LockManager, LocalLock
from pinky_traffic.core.detectors import HsvDetector
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

def test_lane1_follows_blue_line_in_the_middle():
    img = floor()
    cv2.line(img, (200, H - 1), (220, int(H * 0.45)), BLUE, 10)          # 파란 선이 오른쪽에 있다
    p, _, _ = HsvDetector(Config(lane_role=1)).detect(img)
    assert p.uturn_near and p.ok and 0.15 < p.offset < 0.5                # 오른쪽으로 가서 선을 가운데에 둔다
    p, _, _ = HsvDetector(Config(lane_role=2)).detect(img)                # 2차선 로봇은 파란 선을 따라가지 않는다
    assert p.uturn_seen and not p.ok


def test_blue_is_ignored_without_role():
    img = floor()
    cv2.line(img, (200, H - 1), (220, int(H * 0.45)), BLUE, 10)
    p, _, _ = HsvDetector(Config()).detect(img)
    assert not p.uturn_seen and not p.ok


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

def test_lane1_uturn_then_follow_right_then_normal():
    c = started(lane_role=1, junction_clear_sec=8.0, uturn_min_deg=0.0)
    assert c.step(see(), 1.0, 0.1).state == LANE_FOLLOW and c.prefer == ''
    c.step(see(uturn_seen=True, uturn_near=True), 1.0, 1.0)
    assert c.uturn_started and not c.uturn_done
    c.step(see(uturn_seen=True, uturn_near=True), 0.18, 2.0)              # 유턴 중에는 벽이 0.18 이어도 안 선다
    assert c.state == LANE_FOLLOW
    c.step(see(), 1.0, 3.0)
    c.step(see(), 1.0, 4.1)                                               # 파란 선이 1초 넘게 안 보인다 = 유턴 끝
    assert c.uturn_done and c.prefer == 'right'
    c.step(see(), 1.0, 12.2)
    assert c.prefer == '' and not c.junction_held


def run(c, p, t0, t1, dt=0.1, front=1.0):
    t, cmd = t0, None
    while t < t1 - 1e-9:
        t += dt
        cmd = c.step(p, front, t)
    return cmd, t


def test_lane2_enters_pocket_turns_at_green_and_exits_left():
    c = started(lane_role=2, v_min=0.04, pocket_advance_m=0.18, exit_advance_m=0.06, pocket_turn_deg=90.0,
                park_turn_w=0.8, park_line_row=0.80, exit_wait_sec=2.0, exit_follow_sec=8.0)
    c.step(see(), 1.0, 0.1)
    c.step(see(uturn_seen=True), 1.0, 0.2)                                # 멀리 보일 때는 그대로 간다
    assert c.state == LANE_FOLLOW
    cmd = c.step(see(uturn_seen=True, uturn_near=True), 1.0, 0.3)         # 파란 화살표가 발밑에: 따라서 조금 더 간다
    assert cmd.state == POCKET_ADVANCE and cmd.v == 0.04 and c.follow_blue
    cmd, t = run(c, see(uturn_seen=True, uturn_near=True), 0.3, 4.0)
    assert cmd.state == POCKET_ADVANCE
    cmd, t = run(c, see(uturn_seen=True, uturn_near=True), t, 5.2)        # 0.18m / 0.04 = 4.5초
    assert cmd.state == POCKET_TURN and cmd.v == 0 and cmd.w == -0.8 and not c.follow_blue   # 오른쪽으로 90도
    cmd, t = run(c, see(), t, t + 2.3)                                    # 90도 / 0.8 = 약 2초
    assert cmd.state == LANE_FOLLOW and cmd.v > 0
    c.step(see(zone_seen=True, zone_y=0.82), 1.0, t + 0.1)
    assert c.step(see(zone_seen=True, zone_y=0.84), 1.0, t + 0.2).state == PARK_TURN
    cmd, t = run(c, see(), t + 0.2, t + 0.2 + 3.1416 / 0.8 + 0.2)
    assert cmd.state == WAIT_EXIT and c.pocket_parked and not c.junction_held
    cmd, t = run(c, see(), t, t + 2.2)
    assert cmd.state == LANE_FOLLOW and c.exiting                          # 칸에서 나간다
    cmd = c.step(see(uturn_seen=True, uturn_near=True), 1.0, t + 0.1)     # 파란 선이 앞을 가로지른다
    assert cmd.state == POCKET_ADVANCE and cmd.w == 0
    cmd, t = run(c, see(uturn_seen=True, uturn_near=True), t + 0.1, t + 2.0)
    assert cmd.state == POCKET_TURN and cmd.w == 0.8                      # 왼쪽으로 90도 (2차선으로)
    cmd, t = run(c, see(uturn_seen=True, uturn_near=True), t, t + 2.4)
    assert cmd.state == LANE_FOLLOW                                        # 파란 선을 또 봐도 다시 돌지 않는다
    cmd, t = run(c, see(), t, t + 8.5)
    assert not c.exiting and not c.junction_held


def test_green_before_blue_is_ignored_by_lane2():
    c = started(lane_role=2)
    for i in range(4):
        assert c.step(see(zone_seen=True, zone_y=0.9), 1.0, 0.1 * (i + 1)).state == LANE_FOLLOW


def test_lost_preferred_line_spins_toward_it():
    c = started(lane_role=1, side_spin_w=0.6, uturn_min_deg=0.0)
    c.step(see(uturn_seen=True, uturn_near=True), 1.0, 0.1)
    c.step(see(), 1.0, 0.2)
    c.step(see(), 1.0, 1.4)                                               # 유턴 끝 -> 오른쪽 선만 따라간다
    assert c.prefer == 'right'
    c.step(see(), 1.0, 1.5)
    cmd = c.step(Perception(ok=False), 1.0, 2.5)
    assert cmd.v == 0 and cmd.w == -0.6                                   # 오른쪽 선을 찾아 오른쪽으로 돈다


def test_crosswalk_first_seen_close_is_ignored():
    c = started(crosswalk_stop_row=0.80)
    for i in range(3):
        assert c.step(see(crosswalk=True, crosswalk_y=0.79), 1.0, 0.1 * (i + 1)).state == LANE_FOLLOW


def test_two_robots_take_turns_at_junction():
    now = [0.0]
    mgr = LockManager(lease_sec=4.0, clock=lambda: now[0])
    cfg = dict(use_coordinator=True, uturn_min_deg=0.0, v_min=0.04, park_turn_w=0.8, exit_wait_sec=2.0, junction_clear_sec=8.0,
               exit_follow_sec=8.0, pocket_advance_m=0.18)
    a = started(LocalLock(mgr, 'a'), lane_role=1, **cfg)
    b = started(LocalLock(mgr, 'b'), lane_role=2, **cfg)
    t = 0.0

    def tick(pa, pb, n=1):
        nonlocal t
        for _ in range(n):
            t += 0.1
            now[0] = t
            ca, cb = a.step(pa, 1.0, t), b.step(pb, 1.0, t)
        return ca, cb

    blue_a, blue_b = see(uturn_seen=True, uturn_near=True), see(uturn_seen=True, uturn_near=True)
    tick(see(), see())                                                    # 2차선 로봇이 출발부터 구간을 쥔다
    ca, cb = tick(blue_a, blue_b)
    assert ca.state == WAIT_JUNCTION and ca.v == 0 and cb.state == POCKET_ADVANCE
    ca, cb = tick(blue_a, blue_b, 70)                                     # 전진 4.5초 + 우회전 2초 (락은 계속 쥔다)
    assert ca.state == WAIT_JUNCTION and cb.state == LANE_FOLLOW
    ca, cb = tick(blue_a, see(zone_seen=True, zone_y=0.85), 2)
    assert cb.state == PARK_TURN and ca.state == WAIT_JUNCTION
    ca, cb = tick(blue_a, see(), 41)                                      # 2차선 로봇이 칸에서 돌아섰다 -> 구간을 내준다
    assert cb.state == WAIT_EXIT
    ca, cb = tick(blue_a, see())
    assert ca.state == LANE_FOLLOW and ca.v > 0 and a.uturn_started        # 이제 1차선 로봇이 유턴
    ca, cb = tick(blue_a, see(), 40)
    assert cb.state == WAIT_EXIT                                           # 지나갈 때까지 칸에서 기다린다
    ca, cb = tick(see(), see(), 12)                                       # 파란 선 끝
    assert a.uturn_done and cb.state == WAIT_EXIT
    ca, cb = tick(see(), see(), 82)                                       # 1차선 로봇이 구간을 내줬다
    assert cb.state == LANE_FOLLOW and b.exiting and not a.junction_held


def test_blue_lost_mid_uturn_spins_right_until_found():
    # 2026-10-04 pinky2: 파란 선의 꺾이는 곳에서 선이 카메라 밑으로 사라져 유턴이 끝난 줄 알고 벽으로 갔다
    c = started(lane_role=1, side_spin_w=0.6, uturn_min_deg=140.0)
    c.step(see(uturn_seen=True, uturn_near=True), 1.0, 0.1)
    assert c.uturn_started and c.follow_blue
    c.step(see(), 1.0, 0.2)
    cmd, t = run(c, see(), 0.2, 2.0)                                      # 아직 덜 돌았는데 파란 선이 안 보인다
    assert cmd.v == 0 and cmd.w == -0.6 and not c.uturn_done
    cmd = c.step(see(uturn_seen=True, uturn_near=True), 1.0, t + 0.1)     # 다시 보이면 따라간다
    assert cmd.v > 0
    cmd, t = run(c, see(), t + 0.1, t + 6.0)                              # 계속 돌아 140도를 넘기면 유턴 끝
    assert c.uturn_done and c.prefer == 'right' and not c.follow_blue


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


def test_lane2_goes_straight_until_green_then_gives_up_if_never_found():
    c = started(lane_role=2, v_min=0.04, pocket_advance_m=0.04, park_turn_w=0.8, pocket_blind_sec=4.0, pocket_giveup_sec=15.0)
    c.step(see(), 1.0, 0.1)
    cmd, t = run(c, see(uturn_seen=True, uturn_near=True), 0.1, 1.5)
    cmd, t = run(c, see(offset=-1.2), t, t + 2.5)                         # 우회전 끝. 흰 선은 왼쪽으로 가라지만
    assert cmd.state == LANE_FOLLOW and cmd.w == 0 and cmd.v == 0.04       # 초록이 보일 때까지 곧장 간다
    cmd, t = run(c, see(offset=0.3, zone_seen=True, zone_y=0.5), t, t + 0.3)
    assert cmd.w < 0                                                       # 초록이 보이면 그쪽으로
    cmd, t = run(c, see(), t, t + 16.0)
    assert not c.pocket_mode and not c.junction_held                       # 끝내 못 찾으면 포기
    assert c.step(see(crosswalk=True, crosswalk_y=0.5), 1.0, t + 0.1).state != LANE_FOLLOW   # 횡단보도를 다시 본다
