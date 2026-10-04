"""2026-10-04 맵: 1차선 = 파란 선 유턴, 2차선 = 초록 칸으로 빠졌다가 나옴, 유턴 구간은 한 대씩."""
import cv2
import numpy as np

from pinky_traffic.core.config import Config
from pinky_traffic.core.controller import LaneController, LANE_FOLLOW, PARK_TURN, WAIT_EXIT, WAIT_JUNCTION
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
    c = started(lane_role=1, junction_clear_sec=8.0)
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


def test_lane2_pockets_at_green_turns_and_exits_left():
    c = started(lane_role=2, park_line_row=0.80, park_turn_w=0.8, exit_wait_sec=2.0, exit_follow_sec=8.0)
    c.step(see(), 1.0, 0.1)
    assert c.prefer == ''
    c.step(see(uturn_seen=True), 1.0, 1.0)                                # 파란 선이 보이면 오른쪽 선을 따라 칸으로
    assert c.pocket_mode and c.prefer == 'right'
    c.step(see(zone_seen=True, zone_y=0.6), 1.0, 2.0)
    c.step(see(zone_seen=True, zone_y=0.82), 1.0, 2.1)
    assert c.step(see(zone_seen=True, zone_y=0.84), 1.0, 2.2).state == PARK_TURN
    cmd = c.step(see(), 1.0, 2.2 + 3.1416 / 0.8 + 0.1)
    assert cmd.state == WAIT_EXIT and cmd.v == 0 and cmd.w == 0
    assert c.step(see(), 1.0, 7.0).state == WAIT_EXIT                     # 최소 2초는 기다린다
    cmd = c.step(see(uturn_seen=True), 1.0, 9.0)
    assert cmd.state == LANE_FOLLOW and cmd.v > 0 and c.prefer == 'left'  # 나올 때는 파란 선을 봐도 다시 안 들어간다
    c.step(see(), 1.0, 17.5)
    assert c.prefer == '' and not c.exiting


def test_green_before_blue_is_ignored_by_lane2():
    c = started(lane_role=2)
    for i in range(4):
        assert c.step(see(zone_seen=True, zone_y=0.9), 1.0, 0.1 * (i + 1)).state == LANE_FOLLOW


def test_lost_preferred_line_spins_toward_it():
    c = started(lane_role=2, side_spin_w=0.6)
    c.step(see(uturn_seen=True), 1.0, 0.1)
    cmd = c.step(Perception(ok=False), 1.0, 1.0)
    assert cmd.v == 0 and cmd.w == -0.6                                   # 오른쪽 선을 찾아 오른쪽으로 돈다


def test_two_robots_take_turns_at_junction():
    now = [0.0]
    mgr = LockManager(lease_sec=4.0, clock=lambda: now[0])
    cfg = dict(use_coordinator=True, park_turn_w=0.8, exit_wait_sec=2.0, junction_clear_sec=8.0, exit_follow_sec=8.0)
    a = started(LocalLock(mgr, 'a'), lane_role=1, **cfg)
    b = started(LocalLock(mgr, 'b'), lane_role=2, **cfg)

    def tick(t, pa, pb):
        now[0] = t
        return a.step(pa, 1.0, t), b.step(pb, 1.0, t)

    tick(0.1, see(), see())                                               # 2차선 로봇이 출발부터 구간을 쥔다
    ca, cb = tick(1.0, see(uturn_seen=True, uturn_near=True), see(uturn_seen=True))
    assert ca.state == WAIT_JUNCTION and ca.v == 0                         # 1차선 로봇은 파란 선 앞에서 기다린다
    tick(2.0, see(uturn_seen=True, uturn_near=True), see(zone_seen=True, zone_y=0.85))
    ca, cb = tick(2.1, see(uturn_seen=True, uturn_near=True), see(zone_seen=True, zone_y=0.85))
    assert cb.state == PARK_TURN and ca.state == WAIT_JUNCTION
    t = 2.1 + 3.1416 / 0.8 + 0.1
    ca, cb = tick(t, see(uturn_seen=True, uturn_near=True), see())        # 2차선 로봇이 칸에서 돌아섰다 -> 구간을 내준다
    assert cb.state == WAIT_EXIT
    ca, cb = tick(t + 0.1, see(uturn_seen=True, uturn_near=True), see())
    assert ca.state == LANE_FOLLOW and ca.v > 0 and a.uturn_started        # 이제 1차선 로봇이 유턴
    ca, cb = tick(t + 3.0, see(uturn_seen=True, uturn_near=True), see())
    assert cb.state == WAIT_EXIT                                           # 유턴이 끝나 지나갈 때까지 칸에서 기다린다
    tick(t + 4.0, see(), see())
    tick(t + 5.2, see(), see())                                           # 파란 선 끝
    assert a.uturn_done and cb.state == WAIT_EXIT
    ca, cb = tick(t + 5.2 + 8.1, see(), see())                            # 1차선 로봇이 구간을 내줬다
    ca, cb = tick(t + 5.2 + 8.2, see(), see())
    assert cb.state == LANE_FOLLOW and b.prefer == 'left'
