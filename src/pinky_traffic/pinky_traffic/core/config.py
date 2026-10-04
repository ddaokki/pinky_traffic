"""설정값 한 곳에 모으기.

기본값은 여기, 현장 값은 config/*.yaml, 실행 중 변경은 대시보드 슬라이더(update()).
"""
from dataclasses import dataclass, field, fields, asdict
from typing import List, Optional

import yaml


@dataclass
class Config:
    # ---------- 인식 (perception) ----------
    backend: str = 'hsv'            # 'hsv' (학습 전/백업) | 'yolo' (best.pt)
    weights: str = 'best.pt'        # yolo 가중치 경로
    conf: float = 0.4               # yolo 신뢰도 임계값
    imgsz: int = 320                # yolo 입력 크기
    proc_width: int = 320           # 처리 전 이 폭으로 줄인다 (속도)

    # HSV 범위 (OpenCV: H 0~179, S/V 0~255). 기본 = 흰색 테이프
    lane_hsv_lo: List[int] = field(default_factory=lambda: [0, 0, 170])
    lane_hsv_hi: List[int] = field(default_factory=lambda: [179, 70, 255])
    # 횡단보도를 다른 색 테이프로 깔았을 때만 채운다. None 이면 '같은 색' 모드(모양으로 구분)
    crosswalk_hsv_lo: Optional[List[int]] = None
    crosswalk_hsv_hi: Optional[List[int]] = None

    roi_top: float = 0.40           # 이 비율 위쪽(먼 곳/벽)은 버린다
    near_row: float = 0.90          # 차선 중심을 재는 가장 가까운 행 (0=위, 1=아래)
    far_row: float = 0.55           # 가장 먼 행
    lookahead_row: float = 0.80     # 조향 기준 행 (클수록 가까운 곳을 본다)
    n_rows: int = 8
    lane_width_near: float = 1.30   # near_row 에서 차선폭 / 영상폭 (한쪽 선만 보일 때 추정용 초기값, 양쪽이 보이면 자동 학습)
    lane_width_far: float = 0.45    # far_row 에서 차선폭 / 영상폭
    min_area: float = 0.0015        # 덩어리 최소 면적 / 영상 면적
    lane_min_height: float = 0.16   # 차선으로 볼 덩어리의 최소 높이 / 영상 높이 (같은 색 모드)
    crosswalk_min_stripes: int = 3  # 같은 색 모드: 줄무늬가 이 개수 이상이면 횡단보도
    crosswalk_min_area: float = 0.004
    crosswalk_stop_row: float = 0.80  # 횡단보도 아래 끝이 이 행까지 내려오면 정지

    # ---------- 제어 (control) ----------
    v_max: float = 0.10             # m/s
    v_min: float = 0.04
    v_approach: float = 0.05        # 횡단보도 접근 속도
    v_cross: float = 0.07           # 횡단보도 통과 속도
    kp: float = 1.6                 # offset -> 각속도
    ki: float = 0.0
    kd: float = 0.25
    k_heading: float = 0.0          # 먼 행과 가까운 행의 차이(곡률) 보정. 시뮬에서는 0 이 가장 정확했다
    w_max: float = 1.4              # rad/s
    slow_gain: float = 0.7          # offset 클수록 감속
    crosswalk_stop_sec: float = 3.0
    crossing_sec: float = 5.0       # 정지 후 횡단보도를 무시하고 지나가는 시간
    crosswalk_cooldown_sec: float = 3.0
    lost_grace_sec: float = 0.4     # 이 시간까지는 직전 조향 유지
    lost_timeout_sec: float = 1.5   # 이후 정지
    obstacle_stop_m: float = 0.22   # 전방 이 거리 안에 뭔가 있으면 정지
    obstacle_slow_m: float = 0.45
    front_angle_deg: float = 25.0   # 라이다 전방 부채꼴 반각
    lidar_yaw_offset_deg: float = 0.0  # 라이다 0도가 정면이 아니면 보정
    robot_stop_row: float = 0.80    # yolo 'robot' 박스 아래끝이 이 행을 넘으면 장애물로 본다

    # ---------- 2대 운용 (coordinator) ----------
    use_coordinator: bool = False   # True: 횡단보도 구간을 대시보드 서버의 락으로 한 대씩만 통과
    resource: str = 'crosswalk'
    dashboard_url: str = 'http://127.0.0.1:8088'

    # ---------- 2026-10-04 맵: 1차선은 파란 선 따라 유턴해 2차선으로 돌아오고, 2차선은 초록 칸으로 빠진다 ----------
    # 0 = 안 씀(기존 동작) | 1 = 1차선(왼쪽) 로봇 | 2 = 2차선(오른쪽) 로봇. 실행할 때 로봇마다 준다 (drive.sh 5번째 값)
    lane_role: int = 0
    # 2026-10-04: 벽 밑 어두운 그늘이 S 80~108, V 65~95 로 초록에 걸렸다 -> 선명하고 밝은 초록만 (로봇 카메라의 파란 테이프는 S 200, V 175)
    green_hsv_lo: List[int] = field(default_factory=lambda: [40, 110, 100])
    green_hsv_hi: List[int] = field(default_factory=lambda: [90, 255, 255])
    junction_resource: str = 'junction'   # 유턴 구간 + 초록 칸 입구. 2대일 때 한 대씩만 지나간다 (use_coordinator)
    junction_clear_sec: float = 8.0  # 1차선 로봇: 파란 선이 끝난 뒤 이 시간 동안 오른쪽 선만 따라가고(칸 입구를 지나침) 그 뒤 구간을 내준다
    exit_wait_sec: float = 2.0       # 2차선 로봇: 칸에서 돌아선 뒤 최소 이만큼 기다렸다가 나간다
    exit_follow_sec: float = 8.0     # 2차선 로봇: 칸에서 나올 때 이 시간 동안 왼쪽 선만 따라간다 (2차선으로 좌회전)
    # 2차선 로봇의 칸 드나들기: 파란 화살표가 칸 입구 위를 지나간다. 파란 선이 발밑에 오면 그 선을 따라
    # pocket_advance_m 만큼 더 간 뒤 제자리에서 오른쪽으로 pocket_turn_deg 돌아 칸으로 들어간다.
    # 나올 때는 파란 선이 발밑에 오면 exit_advance_m 더 간 뒤 왼쪽으로 돈다. (거리는 명령 속도를 더해서 잰다)
    pocket_advance_m: float = 0.18
    exit_advance_m: float = 0.06
    pocket_turn_deg: float = 90.0
    side_spin_w: float = 0.6         # 한쪽 선만 따라가는 중에 그 선을 놓치면 그쪽으로 제자리 회전하며 찾는다 (rad/s)
    side_search_sec: float = 6.0     # 그렇게 찾는 최대 시간

    # ---------- 주차 통로 (흰 차선 끝에서 빨강/파랑 테이프로 이어지는 길) ----------
    route_color: str = ''            # '' (안 씀) | 'red' | 'blue' : 이 로봇이 따라갈 통로 색
    red_hsv_lo: List[int] = field(default_factory=lambda: [0, 70, 40])       # 빨강은 H 가 0 과 179 양 끝에
    red_hsv_hi: List[int] = field(default_factory=lambda: [12, 255, 255])    # 걸쳐 있어서 범위를 두 개 쓴다
    red2_hsv_lo: List[int] = field(default_factory=lambda: [165, 70, 40])   # V 하한이 낮은 건 칸 끝 그늘 때문
    red2_hsv_hi: List[int] = field(default_factory=lambda: [179, 255, 255])
    blue_hsv_lo: List[int] = field(default_factory=lambda: [95, 70, 25])
    blue_hsv_hi: List[int] = field(default_factory=lambda: [135, 255, 255])
    route_min_area: float = 0.003    # 통로 색 면적 / 영상 면적 이 이상이면 '통로 보임'
    route_min_height: float = 0.12   # 통로 색 덩어리는 키가 영상 높이의 이 비율 이상이어야 선으로 본다
                                     # (카메라 아래 구석의 붉은 색 번짐(카펫)은 납작해서 걸러진다)
    route_max_top: float = 0.75      # 통로 색 덩어리의 위쪽 끝이 이 행보다 위(먼 곳)까지 올라와야 한다
                                     # (2026-10-03: 화면 아래 구석 0.8~1.0 행에 붉은 얼룩이 계속 생겼다)
    route_only_row: float = 0.75     # 통로 색이 이 행보다 가까이 오면 흰색은 버리고 통로 색만 따라간다 (흰 벽 회피)
    park_stop_m: float = 0.15        # 통로 안에서 라이다 전방 거리가 이보다 가까우면 주차 (끝 선을 못 봤을 때 대비)
    # 칸 끝을 가로지르는 통로 색 선(끝 선): 이 선이 park_line_row 까지 내려오면 멈추고 제자리에서 돌아 나갈 방향으로 선다
    route_end_width: float = 0.20    # 가로(±24도)로 영상 폭의 이 비율 이상 이어진 통로 색이면 끝 선
    park_line_row: float = 0.80      # 끝 선의 아래 끝이 이 행까지 오면 정지 (클수록 선에 더 가까이 가서 선다)
    park_min_route_sec: float = 3.0  # 통로에 들어선 뒤 이 시간이 지나야 끝 선을 인정한다 (입구의 비스듬한 선 오인 방지)
    park_turn_deg: float = 180.0     # 정지 후 제자리 회전 각도. 0 이면 돌지 않고 바로 주차 완료
    park_turn_w: float = 0.8         # 회전 속도 rad/s. 시간으로 도는 것이라 덜/더 돌면 park_turn_deg 를 조절
    # 흰 벽 걸러내기: 한 행에서 영상 폭의 lane_wall_width 보다 넓은 행이 영상 높이의 lane_wall_rows 이상이면 면(벽)
    lane_wall_width: float = 0.30
    lane_wall_rows: float = 0.12
    lane_wall_base: float = 0.50     # ROI 위 경계 바로 아래 행이 이 폭보다 넓게 희면 벽 밑단으로 보고 지운다
    # 흰 벽이 화면을 채우면 카메라가 어둡게 찍어 테이프 밝기(V)가 170 아래로 떨어진다 (현장 146).
    # 바닥(화면 아래쪽) 밝기 중앙값 + lane_v_margin 까지 V 하한을 내린다. lane_hsv_lo 의 V 보다 올리지는 않는다.
    lane_auto_v: bool = True
    lane_v_margin: int = 60
    lane_v_min: int = 110
    # 그늘 속 차선: 벽 그늘에서는 테이프 밝기가 140 쯤이라 위 기준에 못 미친다 (2026-10-04 출발점 오른쪽 선).
    # 주변 바닥보다 lane_local_margin 이상 밝은 '가는 띠'(폭이 영상의 lane_local_k 보다 좁은 것)도 차선으로 본다.
    # 넓은 흰 벽은 가는 띠가 아니라서 여기에 안 걸린다. 0 이면 끈다.
    lane_local_margin: int = 45
    lane_local_k: float = 0.16

    def update(self, values: dict):
        """알고 있는 키만 형변환해서 반영. 반영된 키 목록을 돌려준다."""
        changed = []
        known = {f.name: f for f in fields(self)}
        for key, value in (values or {}).items():
            if key not in known:
                continue
            current = getattr(self, key)
            try:
                if isinstance(current, bool):
                    value = value if isinstance(value, bool) else str(value).lower() in ('1', 'true', 'yes', 'on')
                elif isinstance(current, int):
                    value = int(value)
                elif isinstance(current, float):
                    value = float(value)
                elif isinstance(current, list) and value is not None:
                    value = [int(v) for v in value]
            except (TypeError, ValueError):
                continue
            if value != current:
                setattr(self, key, value)
                changed.append(key)
        return changed

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path=None, **overrides):
        cfg = cls()
        if path:
            with open(path, encoding='utf-8') as f:
                cfg.update(yaml.safe_load(f) or {})
        cfg.update(overrides)
        return cfg


# 대시보드 슬라이더로 노출할 값: (키, 최소, 최대, 간격)
TUNABLE = [
    ('v_max', 0.03, 0.25, 0.01),
    ('kp', 0.2, 4.0, 0.1),
    ('kd', 0.0, 1.5, 0.05),
    ('k_heading', 0.0, 3.0, 0.1),
    ('slow_gain', 0.0, 1.0, 0.05),
    ('lookahead_row', 0.50, 0.90, 0.01),
    ('crosswalk_stop_row', 0.55, 0.95, 0.01),
    ('crosswalk_stop_sec', 0.0, 10.0, 0.5),
    ('crossing_sec', 1.0, 12.0, 0.5),
    ('obstacle_stop_m', 0.10, 0.60, 0.01),
    ('park_stop_m', 0.05, 0.40, 0.01),
    ('park_line_row', 0.55, 0.95, 0.01),
    ('park_turn_deg', 0.0, 360.0, 5.0),
    ('pocket_advance_m', 0.0, 0.40, 0.01),
    ('exit_advance_m', 0.0, 0.40, 0.01),
    ('pocket_turn_deg', 30.0, 150.0, 5.0),
    ('conf', 0.1, 0.9, 0.05),
]
