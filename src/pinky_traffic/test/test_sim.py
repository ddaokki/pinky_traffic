"""닫힌 루프 시뮬레이션 테스트: 영상 -> 검출 -> 제어 -> 운동 전체가 맞물려 도는지."""
import pytest

from pinky_traffic.core.config import Config
from pinky_traffic.core.controller import BLOCKED
from pinky_traffic.sim.runner import Simulation
from pinky_traffic.sim.track import Track, TrackSpec

YELLOW = dict(crosswalk_hsv_lo=[20, 100, 120], crosswalk_hsv_hi=[35, 255, 255])
HALF_LANE = TrackSpec().lane_width / 2


@pytest.fixture(scope='module')
def single():
    sim = Simulation(Config(**YELLOW), n_robots=1)
    return sim, sim.run(200)


def test_single_completes_two_laps(single):
    sim, m = single
    r = m['robots']['pinky1']
    assert r['laps'] >= 2.0
    assert r['lost_steps'] == 0
    assert r['final_state'] != 'lost'


def test_single_stays_inside_lane(single):
    r = single[1]['robots']['pinky1']
    assert r['max_abs_lateral_m'] < HALF_LANE - 0.055      # 차체 반폭(5.5cm)을 빼도 선을 안 밟는다
    assert r['rms_lateral_m'] < 0.03


def test_single_stops_before_each_crosswalk(single):
    r = single[1]['robots']['pinky1']
    assert len(r['stops']) >= 2 and r['crossings'] >= 2
    for stop in r['stops']:
        assert 0.03 <= stop['remain_m'] <= 0.30            # 횡단보도 3~30cm 앞
        assert stop['stopped_s'] >= 3.0


def test_same_color_crosswalk_course():
    spec = TrackSpec()
    spec.crosswalk_bgr = spec.lane_bgr
    sim = Simulation(Config(), Track(spec), n_robots=1)
    r = sim.run(110)['robots']['pinky1']
    assert r['laps'] >= 1.0 and r['lost_steps'] == 0
    assert len(r['stops']) >= 1
    assert r['max_abs_lateral_m'] < HALF_LANE


def test_faster_speed_still_holds_lane():
    sim = Simulation(Config(v_max=0.16, **YELLOW), n_robots=1)
    r = sim.run(100)['robots']['pinky1']
    assert r['laps'] >= 1.2 and r['lost_steps'] == 0
    assert r['max_abs_lateral_m'] < HALF_LANE - 0.03


def test_recovers_from_bad_start_pose():
    track = Track(TrackSpec())
    sim = Simulation(Config(**YELLOW), track, n_robots=1)
    x, y, yaw = track.pose_at(3.6, lateral=0.05, dtheta=0.4)     # 직선 구간에서 왼쪽으로 치우치고 23도 틀어짐
    sim.robots[0].x, sim.robots[0].y, sim.robots[0].yaw = x, y, yaw
    sim.run(12)
    tail = [abs(row['lateral']) for row in sim.log['pinky1'][-30:]]
    assert max(tail) < 0.03


@pytest.fixture(scope='module')
def pair():
    sim = Simulation(Config(**YELLOW), n_robots=2, use_coordinator=True)
    return sim, sim.run(200)


def test_two_robots_no_collision(pair):
    sim, m = pair
    assert m['collisions'] == 0
    assert m['min_gap_m'] > 0.15
    for r in m['robots'].values():
        assert r['laps'] >= 1.5 and r['lost_steps'] == 0


def test_two_robots_crosswalk_one_at_a_time(pair):
    sim, m = pair
    assert m['both_in_crossing_steps'] == 0
    assert all(r['crossings'] >= 2 for r in m['robots'].values())
    grants = [h for h in sim.manager.history if h[3] == 'grant']
    releases = [h for h in sim.manager.history if h[3] == 'release']
    assert len(grants) >= 4 and len(releases) >= 4
    assert not any(h[3] == 'expire' for h in sim.manager.history)


def test_follower_stops_behind_stopped_leader():
    sim = Simulation(Config(**YELLOW), n_robots=2, starts=[3.9, 3.4])   # 횡단보도 없는 직선
    sim.controllers[1].start(0.0)        # 뒤차만 출발, 앞차는 서 있다
    for _ in range(int(25 / sim.dt)):
        sim.step()
    assert sim.collisions == 0
    assert sim.controllers[1].state == BLOCKED
    gap = sim.min_gap
    assert 0.15 < gap < 0.40
    sim.controllers[0].start(sim.t)      # 앞차가 출발하면 뒤차도 따라간다
    for _ in range(int(10 / sim.dt)):
        sim.step()
    assert sim.controllers[1].state != BLOCKED
    assert sim.collisions == 0
