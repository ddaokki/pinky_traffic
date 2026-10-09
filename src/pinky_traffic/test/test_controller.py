"""제어/상태머신 단위 테스트: 가짜 인식 결과를 넣고 상태 전이와 출력을 본다."""
from pinky_traffic.core.config import Config
from pinky_traffic.core.controller import (LaneController, PID, IDLE, LANE_FOLLOW, APPROACH, STOP, CROSSING,
                                            BLOCKED, LOST, ESTOP)
from pinky_traffic.core.perception import Perception


def lane(offset=0.0, crosswalk=False, y=0.0, obstacle=0.0):
    return Perception(ok=True, offset=offset, left_seen=True, right_seen=True, crosswalk=crosswalk,
                      crosswalk_y=y, obstacle_y=obstacle)


NOTHING = Perception(ok=False)


def started(cfg=None, lock=None):
    c = LaneController(cfg or Config(), lock)
    c.start(0.0)
    return c


def test_pid_proportional_and_saturation():
    pid = PID(p=2.0, limit=1.0)
    assert pid.update(0.2, 0.1) == 0.4
    assert pid.update(5.0, 0.1) == 1.0
    assert pid.update(-5.0, 0.1) == -1.0


def test_idle_until_start():
    c = LaneController(Config())
    cmd = c.step(lane(), None, 0.0)
    assert cmd.state == IDLE and cmd.v == 0 and cmd.w == 0
    c.start(0.1)
    assert c.step(lane(), None, 0.2).state == LANE_FOLLOW


def test_steering_sign_and_speed():
    c = started()
    straight = c.step(lane(0.0), None, 0.1)
    assert straight.v == Config().v_max and abs(straight.w) < 1e-6
    right = started().step(lane(0.4), None, 0.1)      # 차선 중심이 오른쪽 -> 우회전(음수)
    left = started().step(lane(-0.4), None, 0.1)
    assert right.w < 0 < left.w
    assert right.v < straight.v                        # 치우치면 감속
    assert abs(started().step(lane(1.5), None, 0.1).w) <= Config().w_max


def test_crosswalk_sequence():
    cfg = Config(crosswalk_stop_sec=3.0, crossing_sec=5.0, crosswalk_cooldown_sec=3.0)
    c = started(cfg)
    t = 0.1
    assert c.step(lane(), None, t).state == LANE_FOLLOW
    t += 0.1
    cmd = c.step(lane(crosswalk=True, y=0.6), None, t)          # 멀리 보임 -> 감속 접근
    assert cmd.state == APPROACH and 0 < cmd.v <= cfg.v_approach
    t += 0.1
    cmd = c.step(lane(crosswalk=True, y=0.85), None, t)         # 정지선
    assert cmd.state == STOP and cmd.v == 0
    t_stop = t
    while t < t_stop + 2.9:
        t += 0.1
        assert c.step(lane(crosswalk=True, y=0.9), None, t).v == 0
    t = t_stop + 3.05
    assert c.step(lane(crosswalk=True, y=0.9), None, t).state == CROSSING
    t += 1.0
    cmd = c.step(lane(crosswalk=True, y=0.95), None, t)         # 통과 중에는 다시 서지 않는다
    assert cmd.state == CROSSING and cmd.v > 0
    t += 4.5
    assert c.step(lane(), None, t).state == LANE_FOLLOW
    assert c.crossings == 1
    t += 0.5
    assert c.step(lane(crosswalk=True, y=0.9), None, t).state == LANE_FOLLOW   # 쿨다운 중
    t += 3.0
    assert c.step(lane(crosswalk=True, y=0.6), None, t).state == APPROACH      # 다음 바퀴


def test_crosswalk_false_alarm_returns_to_follow():
    c = started()
    c.step(lane(crosswalk=True, y=0.5), None, 0.1)
    assert c.state == APPROACH
    c.step(lane(), None, 0.5)
    assert c.step(lane(), None, 1.3).state == LANE_FOLLOW


def test_lost_then_recover():
    cfg = Config()
    c = started(cfg)
    c.step(lane(0.2), None, 0.1)
    assert c.step(NOTHING, None, 0.3).v > 0                 # 잠깐은 직전 조향 유지
    cmd = c.step(NOTHING, None, 0.1 + cfg.lost_timeout_sec + 0.1)
    assert cmd.state == LOST and cmd.v == 0
    assert c.step(lane(), None, 3.0).state == LANE_FOLLOW   # 다시 보이면 재개


def test_never_seen_lane_does_not_move():
    c = started()
    cmd = c.step(NOTHING, None, 0.1)
    assert cmd.v == 0 and cmd.state == LOST


def test_blocked_by_lidar_and_resume():
    cfg = Config()
    c = started(cfg)
    c.step(lane(), None, 0.1)
    slow = c.step(lane(), 0.30, 0.2)
    assert slow.state == LANE_FOLLOW and 0 < slow.v < cfg.v_max     # 가까워지면 감속
    cmd = c.step(lane(), 0.15, 0.3)
    assert cmd.state == BLOCKED and cmd.v == 0 and cmd.w == 0
    assert c.step(lane(), 0.24, 0.4).state == BLOCKED               # 히스테리시스 (22+5cm 넘어야 출발)
    assert c.step(lane(), 0.40, 0.5).state == LANE_FOLLOW


def test_blocked_by_yolo_robot_box():
    c = started()
    c.step(lane(), None, 0.1)
    assert c.step(lane(obstacle=0.9), None, 0.2).state == LANE_FOLLOW     # 기본: yolo robot 으로는 안 멈춘다 (라이다만)
    c.cfg.robot_stop = True
    assert c.step(lane(obstacle=0.9), None, 0.3).state == BLOCKED
    assert c.step(lane(obstacle=0.0), None, 0.4).state == LANE_FOLLOW


def test_estop_and_stop_commands():
    c = started()
    c.step(lane(), None, 0.1)
    c.estop(0.2)
    assert c.step(lane(), None, 0.3).state == ESTOP
    assert c.step(lane(), None, 0.4).v == 0
    c.start(0.5)
    assert c.step(lane(), None, 0.6).state == LANE_FOLLOW
    c.stop(0.7)
    assert c.step(lane(), None, 0.8).state == IDLE


class FakeLock:
    def __init__(self):
        self.granted = False
        self.requests = 0
        self.released = 0

    def request(self, resource):
        self.requests += 1
        return self.granted

    def release(self, resource):
        self.released += 1


def test_waits_for_lock_then_releases_after_crossing():
    cfg = Config(use_coordinator=True, crosswalk_stop_sec=1.0, crossing_sec=2.0)
    lock = FakeLock()
    c = started(cfg, lock)
    c.step(lane(), None, 0.1)
    c.step(lane(crosswalk=True, y=0.6), None, 0.2)
    assert lock.requests >= 1                                  # 접근하면서 미리 줄을 선다
    c.step(lane(crosswalk=True, y=0.9), None, 0.3)
    assert c.step(lane(crosswalk=True, y=0.9), None, 5.0).state == STOP    # 시간이 지나도 허가 없으면 대기
    lock.granted = True
    assert c.step(lane(crosswalk=True, y=0.9), None, 5.1).state == CROSSING
    c.step(lane(), None, 7.2)
    assert c.state == LANE_FOLLOW and lock.released >= 1


def test_side_guard_steers_away_and_slows():
    c = started()
    c.step(lane(), None, 0.1)
    free = c.step(lane(), None, 0.2, sides=(0.30, 0.30))
    assert free.reason == '' and free.v > 0                               # 옆 로봇이 멀면 그대로
    cmd = c.step(lane(), None, 0.3, sides=(0.12, 0.30))                   # 왼쪽으로 밀고 들어온다
    assert cmd.w < free.w - 0.3 and 0 < cmd.v < free.v and cmd.reason.startswith('side L')
    cmd = c.step(lane(), None, 0.4, sides=(0.30, 0.07))                   # 오른쪽에 바짝
    assert cmd.v == 0 and cmd.w > 0                                       # 전진은 멈추고 왼쪽으로 비킨다


def test_no_lidar_holds_the_wheels():
    c = started()
    c.step(lane(), None, 0.1)
    cmd = c.step(lane(), None, 0.2, lidar_ok=False)
    assert cmd.v == 0 and cmd.w == 0 and cmd.reason == 'no lidar' and c.state == LANE_FOLLOW
    assert c.step(lane(), None, 0.3).v > 0                                # 돌아오면 다시 달린다
