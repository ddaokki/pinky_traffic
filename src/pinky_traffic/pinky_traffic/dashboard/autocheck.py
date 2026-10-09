"""테스트케이스 자동 판정. 로봇이 올리는 상태(state)와 제어기 이벤트를 보고 통과/실패를 찍는다.

Hub.report() 가 로봇 보고마다 update() 를 부른다. 이미 통과한 항목은 다시 바꾸지 않는다 (사람이 고친 것도 그대로).
"""
import time


class AutoCheck:
    def __init__(self):
        self.robots = {}            # name -> 판정용 기록

    def _r(self, name):
        return self.robots.setdefault(name, {'prev': None, 'moving_since': None, 'lost': False, 'straight': None,
                                             'plan_done_t': None, 'lane1_done': False, 'exit_done': False,
                                             'seen_cross_stop': False})

    def update(self, name, state, events, robots, now=None):
        """-> [(id, 'pass'|'fail', 메모)]"""
        now = time.time() if now is None else now
        out = []
        r = self._r(name)
        st = state.get('state')
        prev, r['prev'] = r['prev'], st
        role = state.get('role')

        # ---- 0. 준비 ----
        volts = [(n, (v.get('state') or {}).get('battery')) for n, v in robots.items()]
        if volts and all(b is not None for _, b in volts):
            if all(b >= 7.2 for _, b in volts):
                out.append(('S-01', 'pass', ' '.join(f'{n} {b:.2f}V' for n, b in volts)))
            elif any(b < 7.0 for _, b in volts):
                out.append(('S-01', 'fail', ' '.join(f'{n} {b:.2f}V' for n, b in volts)))
        fps = [(n, (v.get('state') or {}).get('fps') or 0) for n, v in robots.items()]
        if len(fps) >= 2 and all(f >= 8 for _, f in fps):
            out.append(('S-02', 'pass', ' '.join(f'{n} {f}fps' for n, f in fps)))
        lidar = [(n, (v.get('state') or {}).get('lidar')) for n, v in robots.items()]
        if len(lidar) >= 2 and all(l is True for _, l in lidar) and \
                all((v.get('state') or {}).get('front') is not None for v in robots.values()):
            out.append(('S-04', 'pass', ' '.join(f"{n} front {(v.get('state') or {}).get('front')}" for n, v in robots.items())))
        if state.get('reason') == 'no lidar':
            out.append(('S-04', 'fail', f'{name}: 라이다 끊김'))
        if str(state.get('reason') or '').startswith('side '):
            out.append(('C-05', 'pass', f"{name}: {state.get('reason')}"))
        if st == 'estop' and prev not in (None, 'idle', 'estop') and state.get('v', 1) == 0:
            out.append(('S-03', 'pass', f'{name}: {prev} -> estop'))

        # ---- 1. 인식 ----
        if st == 'lane_follow' and state.get('left') and state.get('right') and abs(state.get('offset') or 9) <= 0.1:
            r['straight'] = r['straight'] or now
            if now - r['straight'] >= 3.0:
                out.append(('P-01', 'pass', f'{name}: 양쪽 선, offset {state.get("offset")}'))
        else:
            r['straight'] = None
        if st == 'sign_approach':
            out.append(('P-02', 'pass', f'{name}: 표지판으로 다가감'))
        if st == 'pocket_end':
            out.append(('P-03', 'pass', f'{name}: 초록 선 위로'))

        # ---- 이벤트 ----
        for text in events:
            if text == 'oncoming flag up':
                out.append(('L1-01', 'pass', name))
            elif text == 'plan_lane1 done':
                out.append(('L1-02', 'pass', name))
                r['plan_done_t'], r['lane1_done'] = now, True
            elif text == 'oncoming -> pocket':
                out.append(('L2-01', 'pass', name))
            elif text.startswith('exit pocket'):
                out.append(('L2-03', 'pass', f'{name}: {text}'))
                if 'robot passed' in text:
                    out.append(('P-04', 'pass', f'{name}: {text}'))
            elif text == 'plan_lane2_exit done':
                out.append(('L2-04', 'pass', name))
                r['plan_done_t'], r['exit_done'] = now, True
            elif text == 'plan_lane2 done':
                out.append(('L2-05', 'pass', name))
                r['plan_done_t'] = now
            elif text.startswith('sign not found'):
                case = {'(plan_lane1)': 'L1-02', '(plan_lane2_exit)': 'L2-04', '(plan_lane2)': 'L2-05'}
                cid = next((c for plan, c in case.items() if plan in text), 'L1-02' if role == 1 else 'L2-04')
                out.append((cid, 'fail', f'{name}: {text}'))
            elif text == 'pocket give up':
                out.append(('L2-02', 'fail', f'{name}: 초록 선을 못 찾음'))
        if st == 'wait_exit' and prev == 'park_turn':
            out.append(('L2-02', 'pass', name))
        if prev == 'stop_at_crosswalk' and st == 'crossing':
            out.append(('C-01', 'pass', name))
        if st == 'blocked' and prev in ('lane_follow', 'approach_crosswalk', 'crossing'):
            out.append(('C-02', 'pass', f'{name}: front {state.get("front")}'))

        # ---- 달리는 동안 ----
        if st in (None, 'idle', 'estop'):
            r['moving_since'], r['lost'], r['plan_done_t'] = None, False, None
            r['lane1_done'] = r['exit_done'] = False
        else:
            r['moving_since'] = r['moving_since'] or now
            if st == 'lost' and not r['lost']:
                r['lost'] = True
                out.append(('C-03', 'fail', f'{name}: lost ({state.get("reason") or ""})'.strip()))
            if not r['lost'] and now - r['moving_since'] >= 30:
                out.append(('C-03', 'pass', f'{name}: 30초 lost 없음'))
            if r['plan_done_t'] and role == 1 and now - r['plan_done_t'] >= 5:
                out.append(('L1-03', 'fail' if r['lost'] else 'pass', name))
        if any(v['lane1_done'] for v in self.robots.values()) and any(v['exit_done'] for v in self.robots.values()):
            out.append(('C-04', 'pass', '1차선 경로 + 2차선 탈출 경로 완주'))
        return out
