"""Driver(검출+제어+명령 처리), 라벨 변환, 라이다 전방거리 테스트."""
import math

import cv2
import numpy as np
import pytest

from pinky_traffic.core.config import Config
from pinky_traffic.core.driver import Driver
from pinky_traffic.sim.track import Track, TrackSpec, Camera, L_LEFT, L_RIGHT, L_CROSSWALK
from pinky_traffic.tools.dataset import mask_to_lines, masks_to_label, DatasetWriter

YELLOW = dict(crosswalk_hsv_lo=[20, 100, 120], crosswalk_hsv_hi=[35, 255, 255])


def test_driver_start_stop_and_param_update():
    track = Track(TrackSpec())
    frame = Camera(track).render(*track.pose_at(0.2))
    driver = Driver(Config(**YELLOW), 'pinky1', use_dashboard=False, model=False)
    assert driver.process(frame, None, 0.0).v == 0            # START 전에는 안 움직인다
    driver.command('start')
    cmd = driver.process(frame, None, 0.1)
    assert cmd.v > 0 and cmd.state == 'lane_follow'
    driver._on_params({'v_max': 0.05, 'unknown_key': 1})       # 대시보드 슬라이더
    assert driver.process(frame, None, 0.2).v <= 0.05
    driver.command('estop')
    assert driver.process(frame, None, 0.3).v == 0
    assert driver.state['state'] == 'estop'
    assert driver.debug.shape[:2] == (240, 320)


def test_config_update_types_and_unknown_keys():
    cfg = Config()
    changed = cfg.update({'v_max': '0.07', 'use_coordinator': 'true', 'imgsz': '416', 'nope': 3,
                          'lane_hsv_lo': [1, 2, 3]})
    assert set(changed) == {'v_max', 'use_coordinator', 'imgsz', 'lane_hsv_lo'}
    assert cfg.v_max == 0.07 and cfg.use_coordinator is True and cfg.imgsz == 416 and cfg.lane_hsv_lo == [1, 2, 3]


def test_field_yaml_loads():
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'field.yaml')
    cfg = Config.load(path)
    assert cfg.weights == 'models/best.pt' and cfg.v_max == 0.08


def test_mask_to_yolo_seg_lines():
    mask = np.zeros((100, 200), np.uint8)
    mask[20:60, 50:150] = 255
    lines = mask_to_lines(mask, 2)
    assert len(lines) == 1
    parts = lines[0].split()
    assert parts[0] == '2' and len(parts) >= 7 and len(parts) % 2 == 1
    values = [float(v) for v in parts[1:]]
    assert all(0 <= v <= 1 for v in values)
    assert min(values[0::2]) == pytest.approx(0.25, abs=0.02) and max(values[1::2]) == pytest.approx(0.59, abs=0.02)


def test_synthetic_labels_match_detector_classes(tmp_path):
    track = Track(TrackSpec())
    camera = Camera(track)
    pose = track.pose_at(0.6)
    labels = camera.render_labels(*pose)
    masks = {'left': np.uint8(labels == L_LEFT) * 255, 'right': np.uint8(labels == L_RIGHT) * 255,
             'crosswalk': np.uint8(labels == L_CROSSWALK) * 255}
    lines = masks_to_label(masks)
    classes = {line.split()[0] for line in lines}
    assert classes == {'0', '1', '2'}                          # left, right, crosswalk 모두 있다
    left_x = np.nonzero(masks['left'])[1].mean()
    right_x = np.nonzero(masks['right'])[1].mean()
    assert left_x < 160 < right_x                              # 왼쪽 선은 화면 왼쪽에
    writer = DatasetWriter(str(tmp_path / 'ds'), val_ratio=0.0)
    writer.add('a', camera.render(*pose), lines)
    yaml_path = writer.finish()
    import yaml
    data = yaml.safe_load(open(yaml_path))
    assert data['names'] == ['left', 'right', 'crosswalk', 'turn', 'straight_right', 'robot'] and data['nc'] == 6   # 앞 3개 번호는 그대로
    assert (tmp_path / 'ds' / 'train' / 'labels' / 'a.txt').read_text().count('\n') == len(lines) - 1


def test_front_range_from_scan():
    pytest.importorskip('rclpy')
    from sensor_msgs.msg import LaserScan
    from pinky_traffic.nodes.lane_driver import front_range
    scan = LaserScan()
    n = 360
    scan.angle_min, scan.angle_increment = -math.pi, 2 * math.pi / n
    scan.range_min, scan.range_max = 0.05, 12.0
    ranges = [2.0] * n
    ranges[n // 2] = 0.3            # 정면(0도)
    ranges[n // 2 + 90] = 0.1       # 왼쪽 90도: 전방 부채꼴 밖
    ranges[n // 2 + 5] = float('inf')
    ranges[n // 2 - 3] = 0.0        # 무효값
    scan.ranges = ranges
    assert front_range(scan, 25.0) == pytest.approx(0.3, abs=1e-5)
    assert front_range(scan, 25.0, yaw_offset_deg=90.0) == pytest.approx(0.1, abs=1e-5)   # 라이다가 돌아가 달린 경우
    scan.ranges = []
    assert front_range(scan, 25.0) is None


def test_robot_mask_from_background_difference():
    from pinky_traffic.tools.robot_autolabel import robot_mask
    bg = np.full((240, 320, 3), (95, 100, 100), np.uint8)
    frame = bg.copy()
    cv2.rectangle(frame, (100, 80), (180, 170), (20, 20, 20), -1)          # 검은 로봇 몸통
    cv2.circle(frame, (120, 170), 15, (200, 180, 40), -1)                  # 하늘색 바퀴
    m = robot_mask(frame, bg)
    ys, xs = np.nonzero(m)
    assert 95 <= xs.min() <= 106 and 174 <= xs.max() <= 186 and ys.max() >= 180
    assert robot_mask(bg, bg) is None                                      # 아무도 없으면 없음


def test_side_ranges_from_scan():
    pytest.importorskip('rclpy')
    from sensor_msgs.msg import LaserScan
    from pinky_traffic.nodes.lane_driver import sector_range
    scan = LaserScan()
    n = 360
    scan.angle_min, scan.angle_increment = -math.pi, 2 * math.pi / n
    scan.range_min, scan.range_max = 0.05, 12.0
    ranges = [2.0] * n
    ranges[n // 2 + 70] = 0.12      # 왼쪽 70도
    ranges[n // 2 - 90] = 0.20      # 오른쪽 90도
    ranges[n // 2 + 170] = 0.08     # 거의 뒤: 측면 밖
    scan.ranges = ranges
    assert sector_range(scan, 35, 110) == pytest.approx(0.12, abs=1e-5)
    assert sector_range(scan, -110, -35) == pytest.approx(0.20, abs=1e-5)
    assert sector_range(scan, 35, 110, yaw_offset_deg=180.0) == pytest.approx(0.20, abs=1e-5)   # 라이다가 뒤로 달린 로봇


def test_side_object_ignores_long_wall_but_sees_robot():
    pytest.importorskip('rclpy')
    from sensor_msgs.msg import LaserScan
    from pinky_traffic.nodes.lane_driver import side_object
    scan = LaserScan()
    n = 360
    scan.angle_min, scan.angle_increment = -math.pi, 2 * math.pi / n
    scan.range_min, scan.range_max = 0.05, 12.0
    ranges = [2.0] * n
    for a in range(-170, -9):                        # 오른쪽에 0.07m 떨어진 곧은 벽 (y = -0.07), 앞뒤로 길게
        ranges[n // 2 + a] = min(2.0, 0.07 / abs(math.sin(math.radians(a))))
    for a in range(60, 76):                          # 왼쪽에 로봇 (짧은 덩어리)
        ranges[n // 2 + a] = 0.09
    scan.ranges = ranges
    assert side_object(scan, -110, -35, near_m=0.2, wall_len=0.30) is None            # 벽은 무시
    assert side_object(scan, 35, 110, near_m=0.2, wall_len=0.30) == pytest.approx(0.09, abs=1e-5)
