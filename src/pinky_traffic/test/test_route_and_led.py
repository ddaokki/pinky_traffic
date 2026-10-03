"""주차 통로(빨강/파랑), 흰 벽 걸러내기, 주차 완료, LED 색 (2026-10-03 현장 맵)."""
import cv2
import numpy as np

from pinky_traffic.core.config import Config
from pinky_traffic.core.controller import (LaneController, LANE_FOLLOW, PARKED, BLOCKED, STOP, IDLE,
                                            led_color, LED_GREEN, LED_RED, LED_BLUE)
from pinky_traffic.core.detectors import HsvDetector
from pinky_traffic.core.perception import Perception

W, H = 320, 240
WHITE, RED, BLUE, CARPET = (235, 235, 235), (40, 40, 200), (180, 60, 20), (95, 100, 100)


def floor():
    return np.full((H, W, 3), CARPET, np.uint8)


def lines(img, color, near=(70, 250), far=(130, 190), thick=10):
    """원근이 있는 두 선: 아래(가까운 곳)는 넓고 위(먼 곳)는 좁다."""
    cv2.line(img, (near[0], H - 1), (far[0], int(H * 0.42)), color, thick)
    cv2.line(img, (near[1], H - 1), (far[1], int(H * 0.42)), color, thick)
    return img


def test_red_route_lines_are_followed():
    p, _, _ = HsvDetector(Config(route_color='red')).detect(lines(floor(), RED))
    assert p.route_seen and p.route_near
    assert p.ok and p.left_seen and p.right_seen and abs(p.offset) < 0.1
    assert not p.crosswalk


def test_red_route_ignored_without_route_color():
    p, _, _ = HsvDetector(Config()).detect(lines(floor(), RED))
    assert not p.route_seen and not p.ok


def test_red_hue_on_both_ends_is_red():
    img = lines(floor(), RED)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 0][hsv[..., 0] < 20] = 175          # H 를 179 쪽 끝으로 옮긴 빨강
    p, _, _ = HsvDetector(Config(route_color='red')).detect(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
    assert p.route_near and p.ok


def test_blue_robot_does_not_follow_red():
    p, _, _ = HsvDetector(Config(route_color='blue')).detect(lines(floor(), RED))
    assert not p.route_seen and not p.ok


def test_white_turns_into_red_far_ahead():
    img = floor()
    cv2.line(img, (70, H - 1), (100, 150), WHITE, 10)
    cv2.line(img, (250, H - 1), (220, 150), WHITE, 10)
    cv2.line(img, (100, 150), (130, int(H * 0.42)), RED, 10)
    cv2.line(img, (220, 150), (190, int(H * 0.42)), RED, 10)
    p, _, _ = HsvDetector(Config(route_color='red')).detect(img)
    assert p.route_seen and not p.route_near        # 아직 흰 차선 위
    assert p.ok and abs(p.offset) < 0.1


def test_white_wall_is_not_a_lane():
    img = floor()
    wall = np.array([[0, int(H * 0.40)], [140, int(H * 0.40)], [0, H - 30]], np.int32)   # 왼쪽 흰 벽(삼각형 면)
    cv2.fillPoly(img, [wall], WHITE)
    p, masks, _ = HsvDetector(Config()).detect(img)
    assert not p.ok and not masks['left'].any() and not masks['right'].any()


def test_white_wall_beside_real_lane_is_dropped():
    img = lines(floor(), WHITE)
    cv2.fillPoly(img, [np.array([[0, int(H * 0.40)], [120, int(H * 0.40)], [0, 200]], np.int32)], WHITE)
    p, _, _ = HsvDetector(Config()).detect(img)
    clean, _, _ = HsvDetector(Config()).detect(lines(floor(), WHITE))
    assert p.ok and abs(p.offset - clean.offset) < 0.1


def test_bay_end_u_shape_keeps_both_lines():
    img = lines(floor(), RED)
    cv2.line(img, (122, 125), (198, 125), RED, 8)     # 칸 끝 가로선이 두 선을 잇는다
    p, _, _ = HsvDetector(Config(route_color='red')).detect(img)
    assert p.ok and p.left_seen and p.right_seen and abs(p.offset) < 0.15


# ---------- 제어 ----------

def lane(route_near=False):
    return Perception(ok=True, left_seen=True, right_seen=True, route_seen=route_near, route_near=route_near)


def started(**cfg):
    c = LaneController(Config(**cfg))
    c.start(0.0)
    return c


def test_parks_at_bay_end_and_stays():
    c = started(route_color='red', park_stop_m=0.15, obstacle_stop_m=0.22)
    assert c.step(lane(True), 0.50, 0.1).state == LANE_FOLLOW and c.in_route
    assert c.step(lane(True), 0.18, 0.2).state == LANE_FOLLOW       # 통로 안에서는 0.22 에서 안 선다
    cmd = c.step(lane(True), 0.14, 0.3)
    assert cmd.state == PARKED and cmd.v == 0 and cmd.w == 0
    assert c.step(lane(True), 1.0, 0.4).state == PARKED             # 벽이 멀어져도 그대로
    c.stop(0.5)
    assert c.state == IDLE and not c.in_route


def test_outside_route_obstacle_still_blocks():
    c = started(route_color='red', park_stop_m=0.15, obstacle_stop_m=0.22)
    assert c.step(lane(False), 0.18, 0.1).state == BLOCKED


def test_led_colors():
    assert led_color(LANE_FOLLOW) == LED_GREEN
    assert led_color(STOP) == LED_RED and led_color(BLOCKED) == LED_RED and led_color(IDLE) == LED_RED
    assert led_color(LANE_FOLLOW, True, 'red') == LED_RED
    assert led_color(LANE_FOLLOW, True, 'blue') == LED_BLUE
    assert led_color(PARKED, True, 'red') == LED_GREEN
    c = started(route_color='blue')
    c.step(lane(True), 0.5, 0.1)
    assert c.led == LED_BLUE


def test_route_in_shadow_at_bay_end():
    # 칸 끝은 벽 그늘: 사진에서 빨강 V 중앙값 105 (1% 57), 파랑 61 (1% 34). 그보다 어둡게 그려 본다.
    dark_floor = np.full((H, W, 3), (40, 42, 42), np.uint8)
    for color, route in [((20, 20, 55), 'red'), ((60, 25, 10), 'blue')]:
        p, _, _ = HsvDetector(Config(route_color=route)).detect(lines(dark_floor.copy(), color))
        assert p.route_near and p.ok, route


def test_reddish_carpet_at_bottom_corner_is_not_route():
    # 2026-10-03 현장: 카메라 아래 오른쪽 구석의 카펫이 붉게 찍혀(46x16 px) 통로로 착각했다
    img = lines(floor(), WHITE)
    cv2.rectangle(img, (274, 224), (319, 239), (30, 30, 70), -1)
    p, _, _ = HsvDetector(Config(route_color='red')).detect(img)
    assert not p.route_seen and not p.route_near and p.ok


def test_dark_red_blob_in_bottom_corner_is_not_route():
    # 현장 기록 24장: 화면 아래 구석(위쪽 끝 0.8h 이하, 높이 29~49 px)에 붉은 얼룩
    img = lines(floor(), WHITE)
    cv2.rectangle(img, (0, 191), (48, 239), (35, 35, 65), -1)
    p, _, _ = HsvDetector(Config(route_color='red')).detect(img)
    assert not p.route_seen and p.ok


def test_wall_base_strip_in_front_is_not_a_lane():
    # 2026-10-03 현장: 정면 흰 벽의 밑단이 ROI 경계 바로 아래에 얇은 가로 띠로 남아 차선으로 잡혔다
    img = lines(floor(), WHITE)
    img[:int(H * 0.44)] = WHITE                       # 화면 위쪽은 벽
    p, _, _ = HsvDetector(Config()).detect(img)
    clean, _, _ = HsvDetector(Config()).detect(lines(floor(), WHITE))
    assert p.ok and abs(p.offset - clean.offset) < 0.1
