"""대시보드 서버 + DashLink 를 실제 HTTP 로 붙여 본다 (ROS 없음)."""
import json
import threading
import time
import urllib.request

import pytest

from pinky_traffic.core.coordinator import DashLink
from pinky_traffic.dashboard.server import make_server


@pytest.fixture()
def server(tmp_path):
    srv, hub = make_server(port=0, run_dir=str(tmp_path / 'run'), host='127.0.0.1')
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{srv.server_address[1]}'
    yield url, hub, tmp_path / 'run'
    srv.shutdown()


def get(url, path):
    with urllib.request.urlopen(url + path, timeout=2) as r:
        return r.read()


def post(url, path, body):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=2) as r:
        return json.loads(r.read())


def wait_for(condition, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if condition():
            return True
        time.sleep(0.02)
    return False


def test_index_page_served(server):
    url, hub, _ = server
    assert 'Pinky 교통 관제'.encode() in get(url, '/')


def test_report_shows_up_and_is_logged(server):
    url, hub, run_dir = server
    link = DashLink(url, 'pinky1', hz=30)
    link.report({'state': 'lane_follow', 'v': 0.1, 'w': 0.0, 'offset': 0.05})
    link.frame(b'\xff\xd8fakejpeg')
    def robot_state():
        return json.loads(get(url, '/api/state'))['robots'].get('pinky1', {}).get('state', {}).get('state')
    assert wait_for(lambda: robot_state() == 'lane_follow')
    assert wait_for(lambda: hub.frames.get('pinky1') == b'\xff\xd8fakejpeg')
    assert get(url, '/frame/pinky1.jpg') == b'\xff\xd8fakejpeg'
    assert wait_for(lambda: (run_dir / 'pinky1.csv').exists())
    assert 'lane_follow' in (run_dir / 'pinky1.csv').read_text()
    assert link.connected
    link.close()


def test_command_and_params_reach_robot(server):
    url, hub, _ = server
    got, params = [], []
    link = DashLink(url, 'pinky1', on_command=got.append, on_params=params.append, hz=30)
    link.report({'state': 'idle'})
    assert wait_for(lambda: link.connected)
    post(url, '/api/cmd', {'robot': 'all', 'cmd': 'start'})
    post(url, '/api/params', {'v_max': 0.07})
    assert wait_for(lambda: 'start' in got)
    assert wait_for(lambda: any(p.get('v_max') == 0.07 for p in params))
    post(url, '/api/cmd', {'robot': 'pinky1', 'cmd': 'estop'})
    assert wait_for(lambda: 'estop' in got)
    link.close()


def test_lock_over_http_is_exclusive(server):
    url, hub, _ = server
    a, b = DashLink(url, 'pinky1', hz=30), DashLink(url, 'pinky2', hz=30)
    assert wait_for(lambda: a.connected and b.connected)
    a.request('crosswalk')
    assert wait_for(lambda: a.request('crosswalk'))
    b.request('crosswalk')
    time.sleep(0.3)
    assert not b.request('crosswalk')
    assert json.loads(get(url, '/api/state'))['locks']['crosswalk'] == {'holder': 'pinky1', 'queue': ['pinky2']}
    a.release('crosswalk')
    assert wait_for(lambda: b.request('crosswalk'))
    a.close()
    b.close()


def test_oncoming_flag_over_http(server):
    url, hub, _ = server
    a, b = DashLink(url, 'pinky1', hz=30), DashLink(url, 'pinky2', hz=30)
    assert wait_for(lambda: a.connected and b.connected)
    assert not b.others_flag('oncoming')
    a.flag('oncoming')
    assert wait_for(lambda: b.others_flag('oncoming'))            # 1차선 로봇이 올린 깃발이 2차선 로봇에게 보인다
    assert not a.others_flag('oncoming')                          # 내 깃발은 나에게는 안 보인다
    assert json.loads(get(url, '/api/state'))['locks']['flag:oncoming']['queue'] == ['pinky1']
    a.flag('oncoming', False)
    assert wait_for(lambda: not b.others_flag('oncoming'))
    a.close()
    b.close()


def test_flag_assumed_up_when_server_unreachable():
    link = DashLink('http://127.0.0.1:9', 'pinky2', hz=30)
    time.sleep(0.2)
    assert link.others_flag('oncoming') is True                   # 모르면 '온다'고 본다 (칸으로 비키는 쪽)
    link.close()


def test_lock_denied_when_server_unreachable():
    link = DashLink('http://127.0.0.1:9', 'pinky1', hz=30)       # 아무도 안 듣는 포트
    time.sleep(0.2)
    assert link.request('crosswalk') is False                     # 연락이 안 되면 통과시키지 않는다
    link.close()


def test_testcase_result_saved(server):
    url, hub, run_dir = server
    state = json.loads(get(url, '/api/state'))
    assert len(state['testcases']) >= 15
    post(url, '/api/testcase', {'id': 'C-03', 'result': 'pass', 'note': '직접 확인'})
    saved = json.loads((run_dir / 'testcase_results.json').read_text(encoding='utf-8'))
    assert saved['C-03']['result'] == 'pass' and saved['C-03']['note'] == '직접 확인'


def test_testcases_judged_automatically_from_reports(server):
    url, hub, run_dir = server
    def rep(robot, state, events=()):
        post(url, '/api/report', {'robot': robot, 'state': state, 'events': list(events), 'want': [], 'release': []})
    rep('pinky1', {'state': 'idle', 'fps': 15.0, 'battery': 7.4, 'role': 1})
    rep('pinky2', {'state': 'idle', 'fps': 14.8, 'battery': 7.3, 'role': 2})
    rep('pinky1', {'state': 'sign_approach', 'fps': 15.0, 'battery': 7.4, 'role': 1}, ['oncoming flag up'])
    rep('pinky2', {'state': 'park_turn', 'fps': 15.0, 'battery': 7.3, 'role': 2}, ['oncoming -> pocket'])
    rep('pinky2', {'state': 'wait_exit', 'fps': 15.0, 'battery': 7.3, 'role': 2})
    rep('pinky1', {'state': 'lane_follow', 'fps': 15.0, 'battery': 7.4, 'role': 1}, ['plan_lane1 done'])
    rep('pinky2', {'state': 'lane_follow', 'fps': 15.0, 'battery': 7.3, 'role': 2},
        ['exit pocket (robot passed)', 'sign not found (plan_lane2_exit): [turn:left]'])
    r = json.loads(get(url, '/api/state'))['results']
    for cid in ('S-01', 'S-02', 'P-02', 'L1-01', 'L1-02', 'L2-01', 'L2-02', 'L2-03', 'P-04'):
        assert r[cid]['result'] == 'pass' and r[cid]['note'].startswith('자동'), cid
    assert r['L2-04']['result'] == 'fail'                          # 탈출 경로에서 표지판을 못 찾음
    rep('pinky2', {'state': 'lane_follow', 'fps': 15.0, 'battery': 7.3, 'role': 2}, ['plan_lane2_exit done'])
    r = json.loads(get(url, '/api/state'))['results']
    assert r['L2-04']['result'] == 'pass' and r['C-04']['result'] == 'pass'   # 다음 시도에서 성공하면 통과로


def test_bad_request_does_not_kill_server(server):
    url, hub, _ = server
    req = urllib.request.Request(url + '/api/cmd', data=b'not json')
    with pytest.raises(Exception):
        urllib.request.urlopen(req, timeout=2)
    assert json.loads(get(url, '/api/state'))['robots'] == {}


def test_start_without_role_warns(server):
    url, hub, run_dir = server
    post(url, '/api/report', {'robot': 'pinky1', 'state': {'state': 'idle', 'role': 0}, 'want': [], 'release': []})
    post(url, '/api/cmd', {'robot': 'all', 'cmd': 'start'})
    assert any('역할 없음' in e['text'] for e in json.loads(get(url, '/api/state'))['events'])
