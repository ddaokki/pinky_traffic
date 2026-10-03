"""관제 대시보드 서버 (표준 라이브러리만 사용, ROS 불필요).

    python3 -m pinky_traffic.dashboard.server --port 8088
    브라우저: http://localhost:8088

하는 일
- 각 로봇의 lane_driver 가 올리는 상태/디버그 영상을 모아 보여준다
- START / STOP / 비상정지 명령과 파라미터 변경을 로봇에게 돌려준다
- 2대 운용 시 횡단보도 구간 락(신호수)을 관리한다
- 상태를 CSV 로, 테스트케이스 결과를 JSON 으로 runs/ 에 남긴다
"""
import argparse
import csv
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..core.config import Config, TUNABLE
from ..core.coordinator import LockManager

HERE = Path(__file__).parent
CSV_FIELDS = ['t', 'robot', 'state', 'v', 'w', 'offset', 'heading', 'ok', 'left', 'right', 'crosswalk',
              'crosswalk_y', 'front', 'fps', 'ms', 'backend']


class Hub:
    """서버의 모든 상태. 핸들러 스레드들이 같이 쓰므로 mutex 로 감싼다."""

    def __init__(self, run_dir, testcases_path=None, lease_sec=4.0):
        self.mutex = threading.Lock()
        self.robots = {}            # name -> {'state': {...}, 't': 수신시각, 'history': [...]}
        self.frames = {}            # name -> jpeg bytes
        self.commands = {}          # name -> [cmd, ...]
        self.params = {}            # 대시보드에서 바꾼 값 (전 로봇 공통)
        self.params_version = 0
        self.locks = LockManager(lease_sec=lease_sec)
        self.events = []
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_files = {}
        self.testcases = self._load_testcases(testcases_path)
        self.results_path = self.run_dir / 'testcase_results.json'
        self.results = json.loads(self.results_path.read_text()) if self.results_path.exists() else {}

    @staticmethod
    def _load_testcases(path):
        path = Path(path) if path else HERE / 'testcases.json'
        return json.loads(path.read_text(encoding='utf-8')) if path.exists() else []

    def event(self, text):
        self.events.append({'t': time.strftime('%H:%M:%S'), 'text': text})
        self.events = self.events[-200:]

    def report(self, body):
        name = body['robot']
        state = body.get('state') or {}
        now = time.time()
        with self.mutex:
            entry = self.robots.setdefault(name, {'history': [], 'state': {}, 't': 0})
            if not entry['t'] or now - entry['t'] > 3:
                self.event(f'{name} 연결됨')
            previous = entry['state'].get('state')
            if state.get('state') and state.get('state') != previous:
                self.event(f"{name}: {previous} → {state['state']}")
            entry['state'], entry['t'] = state, now
            entry['history'].append([round(now, 2), state.get('offset', 0), state.get('v', 0), state.get('w', 0)])
            entry['history'] = entry['history'][-240:]
            self._csv(name, now, state)
            for resource in body.get('release', []):
                self.locks.release(resource, name)
            granted = {r: self.locks.request(r, name) for r in body.get('want', [])}
            reply = {'commands': self.commands.pop(name, []), 'granted': granted,
                     'params_version': self.params_version, 'params': None}
            if body.get('params_version') != self.params_version:
                reply['params'] = dict(self.params)
            return reply

    def _csv(self, name, now, state):
        if not state:
            return
        if name not in self.csv_files:
            path = self.run_dir / f'{name}.csv'
            new = not path.exists()
            f = open(path, 'a', newline='')
            writer = csv.DictWriter(f, CSV_FIELDS, extrasaction='ignore')
            if new:
                writer.writeheader()
            self.csv_files[name] = (f, writer)
        f, writer = self.csv_files[name]
        writer.writerow({**state, 't': round(now, 3), 'robot': name})
        f.flush()

    def snapshot(self):
        now = time.time()
        with self.mutex:
            robots = {n: {'state': e['state'], 'age': round(now - e['t'], 2), 'history': e['history'],
                          'has_frame': n in self.frames} for n, e in self.robots.items()}
            return {'now': now, 'robots': robots, 'locks': self.locks.snapshot(), 'params': self.params,
                    'defaults': Config().to_dict(), 'tunable': TUNABLE, 'events': self.events[-60:],
                    'testcases': self.testcases, 'results': self.results, 'run_dir': str(self.run_dir)}

    def command(self, robot, cmd):
        with self.mutex:
            names = list(self.robots) if robot in ('all', None) else [robot]
            for name in names:
                self.commands.setdefault(name, []).append(cmd)
            self.event(f'명령 {cmd} → {", ".join(names) or "(로봇 없음)"}')

    def set_params(self, values):
        with self.mutex:
            self.params.update(values)
            self.params_version += 1
            self.event('파라미터 ' + ', '.join(f'{k}={v}' for k, v in values.items()))

    def set_result(self, case_id, result, note):
        with self.mutex:
            self.results[case_id] = {'result': result, 'note': note, 't': time.strftime('%Y-%m-%d %H:%M:%S')}
            self.results_path.write_text(json.dumps(self.results, ensure_ascii=False, indent=1), encoding='utf-8')
            self.event(f'테스트 {case_id}: {result} {note}')


class Handler(BaseHTTPRequestHandler):
    hub: Hub = None
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def _send(self, code, body, content_type='application/json'):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        return self.rfile.read(int(self.headers.get('Content-Length', 0) or 0))

    def do_GET(self):
        path = self.path.split('?')[0]
        if path in ('/', '/index.html'):
            self._send(200, (HERE / 'index.html').read_bytes(), 'text/html; charset=utf-8')
        elif path == '/api/state':
            self._send(200, self.hub.snapshot())
        elif path.startswith('/frame/'):
            name = path[len('/frame/'):].replace('.jpg', '')
            jpeg = self.hub.frames.get(name)
            self._send(200 if jpeg else 404, jpeg or b'', 'image/jpeg')
        else:
            self._send(404, {'error': 'not found'})

    def do_POST(self):
        path = self.path.split('?')[0]
        raw = self._body()
        try:
            if path.startswith('/api/frame/'):
                self.hub.frames[path[len('/api/frame/'):]] = raw
                return self._send(200, {'ok': True})
            body = json.loads(raw or b'{}')
            if path == '/api/report':
                return self._send(200, self.hub.report(body))
            if path == '/api/cmd':
                self.hub.command(body.get('robot', 'all'), body['cmd'])
            elif path == '/api/params':
                self.hub.set_params(body)
            elif path == '/api/testcase':
                self.hub.set_result(body['id'], body.get('result', ''), body.get('note', ''))
            else:
                return self._send(404, {'error': 'not found'})
            self._send(200, {'ok': True})
        except Exception as error:      # 잘못된 요청 하나로 서버가 죽지 않게
            self._send(400, {'error': str(error)})


def make_server(port=8088, run_dir=None, host='0.0.0.0', testcases=None):
    run_dir = run_dir or os.path.join('runs', time.strftime('%Y%m%d_%H%M%S'))
    hub = Hub(run_dir, testcases)
    handler = type('BoundHandler', (Handler,), {'hub': hub})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server, hub


def main():
    parser = argparse.ArgumentParser(description='Pinky 교통 관제 대시보드')
    parser.add_argument('--port', type=int, default=8088)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--run-dir', default=None, help='로그 폴더 (기본 runs/날짜_시각)')
    args = parser.parse_args()
    server, hub = make_server(args.port, args.run_dir, args.host)
    print(f'대시보드: http://localhost:{args.port}   로그: {hub.run_dir}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
