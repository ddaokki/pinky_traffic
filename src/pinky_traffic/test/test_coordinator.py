"""락(신호수) 로직 테스트."""
from pinky_traffic.core.coordinator import LockManager


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_one_holder_at_a_time_fifo():
    m = LockManager(clock=Clock())
    assert m.request('cw', 'a')
    assert not m.request('cw', 'b')
    assert not m.request('cw', 'c')
    assert m.request('cw', 'a')            # 가진 로봇이 다시 불러도 True
    m.release('cw', 'a')
    assert not m.request('cw', 'c')        # b 가 먼저 줄을 섰다
    assert m.request('cw', 'b')
    m.release('cw', 'b')
    assert m.request('cw', 'c')
    assert m.snapshot()['cw'] == {'holder': 'c', 'queue': []}


def test_release_by_non_holder_does_nothing():
    m = LockManager(clock=Clock())
    m.request('cw', 'a')
    m.release('cw', 'b')
    assert not m.request('cw', 'b')


def test_dead_holder_expires():
    clock = Clock()
    m = LockManager(lease_sec=4.0, clock=clock)
    assert m.request('cw', 'a')
    clock.t = 2.0
    assert not m.request('cw', 'b')
    clock.t = 5.0                           # a 는 2초 이후 소식이 없다... 아직 4초는 안 지남 (마지막 0초 -> 5초 = 만료)
    assert m.request('cw', 'b')
    assert ('cw', 'a', 'expire') in [(r, n, k) for _, r, n, k in m.history]


def test_heartbeat_keeps_lock():
    clock = Clock()
    m = LockManager(lease_sec=4.0, clock=clock)
    m.request('cw', 'a')
    for t in (2, 4, 6, 8):
        clock.t = t
        assert m.request('cw', 'a')
        assert not m.request('cw', 'b')


def test_waiter_that_left_is_skipped():
    clock = Clock()
    m = LockManager(lease_sec=4.0, clock=clock)
    m.request('cw', 'a')
    m.request('cw', 'b')                    # b 줄 섬
    clock.t = 3.0
    m.request('cw', 'a')
    m.request('cw', 'c')                    # c 줄 섬 (b 는 이후 소식 없음)
    clock.t = 6.0
    m.request('cw', 'c')
    m.release('cw', 'a')
    assert m.request('cw', 'c')             # b 는 만료되어 건너뛴다


def test_independent_resources():
    m = LockManager(clock=Clock())
    assert m.request('cw1', 'a')
    assert m.request('cw2', 'b')
