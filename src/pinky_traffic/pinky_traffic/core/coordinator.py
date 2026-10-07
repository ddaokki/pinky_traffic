"""2대 운용용 '신호수'. 한 구간(resource)을 한 번에 한 대만 지나가게 한다.

LockManager      : 순수 로직 (대시보드 서버 안에서 돈다 / 시뮬레이터가 직접 쓴다)
LocalLock        : 시뮬레이터용 어댑터
DashLink         : 실제 노드용. 대시보드 서버와 HTTP 로 상태보고 + 명령수신 + 락 요청을 한 번에 한다.

로봇 두 대는 ROS_DOMAIN_ID 가 달라 서로 토픽이 안 보인다 (수업 19~21번 자료).
그래서 로봇 사이의 약속은 ROS 가 아니라 이 HTTP 서버를 거친다.
"""
import json
import threading
import time
import urllib.request


class LockManager:
    def __init__(self, lease_sec=4.0, clock=time.time):
        self.lease_sec = lease_sec      # 이 시간 동안 연락이 없으면 자리를 뺏는다 (로봇이 죽었을 때 교착 방지)
        self.clock = clock
        self.holder = {}                # resource -> robot
        self.queue = {}                 # resource -> [robot, ...] 도착 순서
        self.seen = {}                  # (resource, robot) -> 마지막 요청 시각
        self.history = []               # (t, resource, robot, 'grant'|'release'|'expire')
        self.flags = {}                 # (flag, robot) -> 마지막으로 올린 시각. 락과 달리 다른 로봇은 보기만 한다
        self._mutex = threading.Lock()

    def _expire(self, resource, now):
        queue = self.queue.setdefault(resource, [])
        for robot in list(queue):
            if now - self.seen.get((resource, robot), now) > self.lease_sec:
                queue.remove(robot)
        holder = self.holder.get(resource)
        if holder and now - self.seen.get((resource, holder), now) > self.lease_sec:
            self.history.append((now, resource, holder, 'expire'))
            self.holder.pop(resource)

    def request(self, resource, robot):
        """줄을 서고, 내 차례면 True. 통과가 끝날 때까지 계속 불러야 한다(=하트비트)."""
        with self._mutex:
            now = self.clock()
            self.seen[(resource, robot)] = now
            self._expire(resource, now)
            queue = self.queue.setdefault(resource, [])
            holder = self.holder.get(resource)
            if holder == robot:
                return True
            if robot not in queue:
                queue.append(robot)
            if holder is None and queue[0] == robot:
                queue.pop(0)
                self.holder[resource] = robot
                self.history.append((now, resource, robot, 'grant'))
                return True
            return False

    def release(self, resource, robot):
        with self._mutex:
            now = self.clock()
            queue = self.queue.setdefault(resource, [])
            if robot in queue:
                queue.remove(robot)
            if self.holder.get(resource) == robot:
                self.holder.pop(resource)
                self.history.append((now, resource, robot, 'release'))

    def raise_flag(self, flag, robot, on=True):
        """깃발 (예: 'oncoming' = 1차선 로봇이 유턴 표시를 봤다). 올린 동안 계속 불러야 한다(=하트비트)."""
        with self._mutex:
            if on:
                self.flags[(flag, robot)] = self.clock()
            else:
                self.flags.pop((flag, robot), None)

    def flags_of_others(self, robot):
        """다른 로봇이 올려 둔 깃발 이름들. lease_sec 동안 소식이 없으면 내려간 것으로 본다."""
        with self._mutex:
            now = self.clock()
            return sorted({f for (f, r), t in self.flags.items() if r != robot and now - t <= self.lease_sec})

    def snapshot(self):
        with self._mutex:
            resources = set(self.holder) | set(self.queue)
            snap = {r: {'holder': self.holder.get(r), 'queue': list(self.queue.get(r, []))} for r in resources}
        now = self.clock()
        for (f, r), t in list(self.flags.items()):
            if now - t <= self.lease_sec:
                snap.setdefault('flag:' + f, {'holder': None, 'queue': []})['queue'].append(r)
        return snap


class LocalLock:
    """LaneController 가 기대하는 request(resource)/release(resource) 모양으로 감싼다."""

    def __init__(self, manager, robot):
        self.manager, self.robot = manager, robot

    def request(self, resource):
        return self.manager.request(resource, self.robot)

    def release(self, resource):
        self.manager.release(resource, self.robot)

    def flag(self, name, on=True):
        self.manager.raise_flag(name, self.robot, on)

    def others_flag(self, name):
        return name in self.manager.flags_of_others(self.robot)


class DashLink:
    """대시보드 서버와의 연결 (백그라운드 스레드, 표준 라이브러리만 사용).

    - report(state): 최신 상태를 올린다. 응답으로 명령/파라미터/락 결과가 온다.
    - frame(jpeg): 디버그 영상을 올린다.
    - request()/release(): 락. 서버가 안 보이면 fail_open 값에 따라 통과/대기.
    - flag()/others_flag(): 깃발. 서버가 안 보이면 다른 로봇의 깃발이 '올라가 있다'고 본다 (조심하는 쪽).
    제어 루프를 막지 않도록 네트워크는 전부 스레드에서 한다.
    """

    def __init__(self, url, robot, on_command=None, on_params=None, hz=8.0, fail_open=False):
        self.url = url.rstrip('/')
        self.robot = robot
        self.on_command = on_command
        self.on_params = on_params
        self.period = 1.0 / hz
        self.fail_open = fail_open
        self.connected = False
        self._state = {}
        self._jpeg = None
        self._want = {}                 # resource -> True(요청중)
        self._release = set()
        self._granted = {}
        self._flags = set()             # 내가 올린 깃발
        self._others = set()            # 다른 로봇이 올린 깃발 (서버 응답)
        self._params_version = -1
        self._mutex = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def report(self, state):
        with self._mutex:
            self._state = dict(state)

    def frame(self, jpeg_bytes):
        with self._mutex:
            self._jpeg = jpeg_bytes

    def request(self, resource):
        with self._mutex:
            self._want[resource] = True
            self._release.discard(resource)
            if not self.connected:
                return self.fail_open
            return self._granted.get(resource, False)

    def release(self, resource):
        with self._mutex:
            if self._want.pop(resource, None) or self._granted.get(resource):
                self._release.add(resource)
            self._granted[resource] = False

    def flag(self, name, on=True):
        with self._mutex:
            (self._flags.add if on else self._flags.discard)(name)

    def others_flag(self, name):
        with self._mutex:
            return name in self._others if self.connected else True

    def close(self):
        self._stop.set()

    def _post(self, path, data, content_type='application/json', timeout=0.6):
        req = urllib.request.Request(self.url + path, data=data, headers={'Content-Type': content_type})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()

    def _loop(self):
        while not self._stop.is_set():
            t0 = time.time()
            with self._mutex:
                body = {'robot': self.robot, 'state': self._state, 'want': list(self._want),
                        'release': list(self._release), 'flags': sorted(self._flags),
                        'params_version': self._params_version}
                self._release.clear()
                jpeg, self._jpeg = self._jpeg, None
            try:
                reply = json.loads(self._post('/api/report', json.dumps(body).encode()))
                if jpeg:
                    self._post(f'/api/frame/{self.robot}', jpeg, 'image/jpeg')
                with self._mutex:
                    self.connected = True
                    self._granted = {r: bool(g) for r, g in reply.get('granted', {}).items()}
                    self._others = set(reply.get('flags', []))
                for cmd in reply.get('commands', []):
                    if self.on_command:
                        self.on_command(cmd)
                if reply.get('params') is not None:
                    self._params_version = reply.get('params_version', self._params_version)
                    if self.on_params:
                        self.on_params(reply['params'])
            except Exception:
                with self._mutex:
                    self.connected = False
                    self._granted = {}
            self._stop.wait(max(0.0, self.period - (time.time() - t0)))
