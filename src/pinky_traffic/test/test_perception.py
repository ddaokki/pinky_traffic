"""인식 단위 테스트: 시뮬레이터가 그린 영상으로 검출기를 확인한다."""
import numpy as np
import pytest

from pinky_traffic.core.config import Config
from pinky_traffic.core.detectors import HsvDetector
from pinky_traffic.sim.track import Track, TrackSpec, Camera

YELLOW = dict(crosswalk_hsv_lo=[20, 100, 120], crosswalk_hsv_hi=[35, 255, 255])


@pytest.fixture(scope='module')
def world():
    track = Track(TrackSpec())
    return track, Camera(track)


def detect(world, s, lateral=0.0, dtheta=0.0, **cfg):
    track, camera = world
    detector = HsvDetector(Config(**{**YELLOW, **cfg}))
    p, masks, frame = detector.detect(camera.render(*track.pose_at(s, lateral, dtheta)))
    return p


def test_straight_center_sees_both_lines(world):
    p = detect(world, 0.2)
    assert p.ok and p.left_seen and p.right_seen
    assert abs(p.offset) < 0.05
    assert not p.crosswalk


def test_offset_sign_follows_lateral_shift(world):
    # 로봇이 왼쪽으로 치우치면 차선 중심은 화면 오른쪽 -> offset > 0
    left_shift = detect(world, 0.2, lateral=0.05)
    right_shift = detect(world, 0.2, lateral=-0.05)
    assert left_shift.offset > 0.1
    assert right_shift.offset < -0.1


def test_offset_sign_follows_heading(world):
    # 로봇이 왼쪽으로 틀어져 있으면 차선은 화면 오른쪽으로 흐른다
    assert detect(world, 0.2, dtheta=0.3).offset > 0.08
    assert detect(world, 0.2, dtheta=-0.3).offset < -0.08


def test_left_curve_with_one_line_steers_left(world):
    p = detect(world, 1.9)          # 좌회전 곡선 안 (바깥쪽 선만 보인다)
    assert p.ok and p.right_seen and not p.left_seen
    assert p.offset < -0.1


def test_crosswalk_detected_and_gets_closer(world):
    far = detect(world, 0.55)       # 횡단보도(0.9 m) 35 cm 앞
    near = detect(world, 0.75)      # 15 cm 앞
    assert far.crosswalk and near.crosswalk
    assert near.crosswalk_y > far.crosswalk_y
    assert near.crosswalk_y >= 0.8


def test_no_crosswalk_far_from_it(world):
    for s in (0.0, 2.5, 4.0, 5.5):
        assert not detect(world, s).crosswalk


def test_same_color_crosswalk_by_shape():
    spec = TrackSpec()
    spec.crosswalk_bgr = spec.lane_bgr          # 횡단보도도 흰 테이프
    track = Track(spec)
    camera = Camera(track)
    cfg = Config()                               # crosswalk_hsv 없음 -> 모양으로 구분
    near = HsvDetector(cfg).detect(camera.render(*track.pose_at(0.6)))[0]
    plain = HsvDetector(cfg).detect(camera.render(*track.pose_at(0.0)))[0]
    assert near.crosswalk and near.ok
    assert abs(near.offset) < 0.12               # 줄무늬 때문에 차선 중심이 틀어지지 않는다
    assert not plain.crosswalk


def test_blank_floor_is_not_ok():
    frame = np.full((240, 320, 3), 100, np.uint8)
    p = HsvDetector(Config()).detect(frame)[0]
    assert not p.ok and not p.crosswalk


def test_big_frame_is_resized(world):
    track, camera = world
    import cv2
    frame = cv2.resize(camera.render(*track.pose_at(0.2)), (1280, 960))
    p, masks, small = HsvDetector(Config(**YELLOW)).detect(frame)
    assert small.shape[1] == 320 and p.ok and abs(p.offset) < 0.06
