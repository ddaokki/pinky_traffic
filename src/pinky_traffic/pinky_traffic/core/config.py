"""설정값 한 곳에 모으기.

기본값은 여기, 현장 값은 config/*.yaml, 실행 중 변경은 대시보드 슬라이더(update()).
"""
from dataclasses import dataclass, field, fields, asdict
from typing import List, Optional

import yaml


@dataclass
class Config:
    # ---------- 인식 (perception) ----------
    # 인식은 한 가지: 차선·횡단보도·초록 선은 색(HSV), 파란 표지판·상대 로봇은 YOLO (detectors.HybridDetector)
    weights: str = 'best.pt'        # yolo 가중치 경로 (표지판·로봇 클래스)
    conf: float = 0.4               # yolo 신뢰도 임계값
    imgsz: int = 320                # yolo 입력 크기
    yolo_threads: int = 2           # yolo 가 쓰는 CPU 스레드 (로봇 2대가 PC 하나를 나눠 쓴다)
    yolo_every: int = 3             # hsv+yolo: yolo 를 이 프레임마다 한 번 (그 사이는 직전 결과)
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
    lane_bottom_height: float = 0.25  # 화면 아래 끝에 닿고 이만큼 길면 멀리까지 안 이어져도 차선 (코너에서 꺾이는 선)
    lane_bottom_side: float = 0.30  # ... 단 발밑 위치가 화면 양 옆 이 비율 안일 때만 (가운데는 횡단보도 줄무늬일 수 있다)
    crosswalk_min_stripes: int = 3  # 같은 색 모드: 줄무늬가 이 개수 이상이면 횡단보도
    crosswalk_min_area: float = 0.004
    crosswalk_stop_row: float = 0.80  # 횡단보도 아래 끝이 이 행까지 내려오면 정지
    crosswalk_confirm: int = 3      # 처음부터 가까이서 보였어도 정지 행 앞에서 이만큼 연속으로 보이면 진짜 횡단보도

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
    offset_jump: float = 0.9        # 한 프레임에 차선 중심이 이보다 크게 바뀌면 그 한 프레임은 무시 (코너에서 선 오인)
    crosswalk_stop_sec: float = 3.0
    crossing_sec: float = 5.0       # 정지 후 횡단보도를 무시하고 지나가는 시간
    crosswalk_cooldown_sec: float = 3.0
    lost_grace_sec: float = 0.4     # 이 시간까지는 직전 조향 유지
    lost_timeout_sec: float = 1.5   # 이후 정지
    obstacle_stop_m: float = 0.22   # 전방 이 거리 안에 뭔가 있으면 정지
    obstacle_slow_m: float = 0.45
    front_angle_deg: float = 25.0   # 라이다 전방 부채꼴 반각
    lidar_yaw_offset_deg: float = 0.0  # 라이다 0도가 정면이 아니면 보정
    robot_stop_row: float = 0.80    # yolo 'robot' 박스 아래끝이 이 행을 넘으면 장애물로 본다 (robot_stop 일 때)
    # 2026-10-09: robot 을 칸 안 시점에서만 학습해 주행 중 카펫·흰 선을 robot(0.5~0.7)으로 보고 멈췄다.
    # 앞 장애물은 라이다로만 멈추고, yolo robot 은 칸 안에서 상대가 지나갔는지 볼 때만 쓴다 (신뢰도 robot_conf 이상만)
    robot_stop: bool = False
    robot_conf: float = 0.8
    # 옆구리 침범 막기: 라이다 측면(side_angle_from~to 도, 좌우) 최소 거리. 나란히 달릴 때 옆 로봇까지는 보통 이보다 멀다
    side_guard: bool = False         # 2026-10-09: 가운데 가벽을 옆 로봇으로 계속 오인해 1차선이 벽 옆에서 멈춤 -> 우선 끔
    side_angle_from: float = 35.0
    side_angle_to: float = 110.0
    # 2026-10-09 현장: 나란히 세우면 옆 로봇까지 0.14~0.15m (라이다는 로봇 가운데, 실제 틈 약 8cm), 바깥 벽도 0.15m
    side_slow_m: float = 0.10        # 이 안이면 반대쪽으로 조향 + 감속 (실제 틈 약 4cm)
    side_stop_m: float = 0.07        # 이 안이면 전진은 멈추고 피하는 회전만 (실제 틈 약 1~2cm)
    side_push_w: float = 0.8         # 피하는 회전 세기 (rad/s, 가까울수록 이만큼까지)
    side_wall_len: float = 0.18      # (2026-10-09 가운데 가벽이 0.30 보다 짧게 잡혀 1차선이 옆구리 회피로 머뭇거림) 그쪽 옆(5~170도) 0.25m 안의 점들이 이 길이 이상 이어지면 벽(무시), 짧으면 로봇 (로봇 폭 약 11cm)
    intrude_offset: float = 0.6      # 시연용 끼어들기: 차선 중심을 이만큼 옆으로 밀어 본다 (약 13cm)
    intrude_sec: float = 4.0
    hold_front_m: float = 0.10       # 어떤 상태든 앞이 이보다 가까우면 전진만 멈춘다 (표지판 기동·칸 안 포함)
    require_lidar: bool = True       # 라이다가 lidar_timeout_sec 넘게 안 오면 바퀴를 세운다
    lidar_timeout_sec: float = 1.5

    # ---------- 2대 운용 (coordinator) ----------
    use_coordinator: bool = False   # True: 횡단보도 구간을 대시보드 서버의 락으로 한 대씩만 통과
    resource: str = 'crosswalk'
    dashboard_url: str = 'http://127.0.0.1:8088'

    # ---------- 2026-10-04 맵: 1차선은 파란 선 따라 유턴해 2차선으로 돌아오고, 2차선은 초록 칸으로 빠진다 ----------
    # 0 = 안 씀(기존 동작) | 1 = 1차선(왼쪽) 로봇 | 2 = 2차선(오른쪽) 로봇. 실행할 때 로봇마다 준다 (drive.sh 5번째 값)
    lane_role: int = 0
    # 2026-10-04: 벽 밑 어두운 그늘이 S 80~108, V 65~95 로 초록에 걸렸다 -> 선명하고 밝은 초록만 (로봇 카메라의 파란 테이프는 S 200, V 175)
    #             멀리 있는 초록 테이프는 옅어서 S 85~100 쯤, 밝기는 160 이상 -> 채도는 낮추고 밝기 기준을 올림
    green_hsv_lo: List[int] = field(default_factory=lambda: [40, 80, 120])
    green_hsv_hi: List[int] = field(default_factory=lambda: [90, 255, 255])
    junction_resource: str = 'junction'   # 유턴 구간 + 초록 칸 입구. 2대일 때 한 대씩만 지나간다 (use_coordinator)
    uturn_min_deg: float = 140.0     # 유턴: 이만큼 돌기 전에 파란 선이 안 보이면(카메라 밑으로 사라짐) 유턴 방향으로 돌며 다시 찾는다
    junction_clear_sec: float = 8.0  # 유턴한 로봇: 파란 선이 끝난 뒤 이 시간 동안 오른쪽 선만 따라가고(칸 입구를 지나침) 그 뒤 구간을 내준다
    exit_wait_sec: float = 2.0       # 2차선 로봇: 칸에서 돌아선 뒤 최소 이만큼 기다렸다가 나간다
    exit_follow_sec: float = 8.0     # (2026-10-07 부터 안 씀) 예전: 칸에서 나올 때 왼쪽 선만 따라 2차선으로 좌회전
    # (2026-10-07 표지판 맵부터 안 씀: uturn_min_deg, exit_follow_sec, pocket_advance_m, exit_advance_m, pocket_turn_deg, exit_blind_sec)
    # 2차선 로봇의 칸 드나들기: 파란 화살표가 칸 입구 위를 지나간다. 파란 선이 발밑에 오면 그 선을 따라
    # pocket_advance_m 만큼 더 간 뒤 제자리에서 오른쪽으로 pocket_turn_deg 돌아 칸으로 들어간다.
    # 나올 때는 파란 선이 발밑에 오면 exit_advance_m 더 간 뒤 오른쪽으로 돌아 파란 선을 따라 유턴한다. (거리는 명령 속도를 더해서 잰다)
    pocket_advance_m: float = 0.22
    zone_roi_top: float = 0.25       # 초록 선은 이 행 아래에서 찾는다 (칸에 막 들어섰을 때는 멀어서 차선 ROI 보다 위에 보인다)
    exit_blind_sec: float = 10.0     # 칸에서 나올 때 파란 선이 발밑에 올 때까지 곧장 가는 최대 시간
    pocket_giveup_sec: float = 15.0  # 칸 쪽으로 돈 뒤 이 시간 안에 초록 앞에 못 서면 칸을 포기하고 보통 주행으로 돌아간다
    pocket_blind_sec: float = 15.0   # 칸 쪽으로 돈 뒤 초록이 아직 안 보이면 이 시간까지는 곧장 간다 (흰 선 좌우 구분을 믿지 않는다)
    exit_advance_m: float = 0.06
    pocket_turn_deg: float = 90.0
    side_spin_w: float = 0.6         # 한쪽 선만 따라가는 중에 그 선을 놓치면 그쪽으로 제자리 회전하며 찾는다 (rad/s)
    side_search_sec: float = 6.0     # 그렇게 찾는 최대 시간
    # 2026-10-07 교행: 1차선 로봇이 파란 유턴 표시를 보면 서버에 oncoming 깃발을 올린다 (칸 입구를 지나 구간을 내줄 때까지).
    # 2차선 로봇은 파란 선 앞에서: 깃발이 있으면 초록 칸으로 비키고, 없으면 파란 선을 거꾸로 따라 유턴해 1차선으로 간다.
    # 칸에서는 상대 로봇(yolo 'robot')이 보였다가 pass_clear_sec 동안 안 보이거나, 깃발이 내려가면 나와서 우회전 -> 유턴.
    oncoming_flag: str = 'oncoming'
    lane2_flag: str = 'lane2'        # 2차선 로봇이 출발해 아직 칸에 안 들어갔다 (1차선 로봇은 R1 에서 기다린다)
    lane2_wait_max_sec: float = 40.0 # 1차선이 R1 에서 기다리는 최대 시간 (2차선이 멈췄을 때 대비)
    sign_after_turn_sec: float = 0.8   # 표지판에서 돈 뒤에도 이만큼 멈춰서 다음 표지판을 찾는다
    sign_pause_sec: float = 0.5      # 표지판 위에서 돌기 전에 완전히 멈추는 시간
    pocket_decide_sec: float = 1.0   # 2차선 로봇: 깃발이 없을 때 파란 선 앞에서 이만큼 서서 한 번 더 기다려 본 뒤 유턴한다
    pass_clear_sec: float = 1.5      # 2차선 로봇: 칸에서 본 상대 로봇이 이 시간 동안 안 보이면 '지나갔다'
    pass_front_m: float = 0.35       # 2차선 로봇: 칸에서 라이다 전방 이 거리 안에 뭔가 지나가도 '상대 로봇을 봤다'로 친다 (칸은 테이프라 벽이 없다)
    # 2026-10-07 표지판 맵: 파란 선 대신 바닥의 파란 양방향 표지판 두 종류를 보고 그 자리에서 돈다.
    #   turn = 우회전 양방향(꺾인 화살표, 오는 방향에 따라 우회전/좌회전), straight_right = 직우 양방향(긴 ←→ 에 칸 쪽 가지)
    # 경로 = "표지판종류:행동" 을 만나는 순서대로 쉼표로. 행동 right|left|straight. 다 지나면 흰 차선을 따라간다.
    plan_lane1: str = 'turn:right, turn:right:0.07'    # 1차선: R1 우회전 -> R2 우회전(=유턴) -> lane1_exit_m 곧장. 세 번째 칸 = 그 표지판만의 sign_advance_m
    plan_lane2: str = 'straight_right:straight, turn:left, turn:left'      # 2차선, 상대가 안 온다: S 직진 -> 좌 -> 좌 (=유턴, 1차선으로)
    plan_lane2_pocket: str = 'straight_right:right:0.0'                        # 2차선, 상대가 온다: S 에서 우회전해 초록 칸으로
    plan_lane2_exit: str = 'any:right, turn:left, turn:left'               # 칸에서 나와: 입구의 표지판(직우 가지)에서 우회전 -> 좌 -> 좌 (=유턴, 1차선으로)
    sign_stop_row: float = 0.80      # (2026-10-09 부터 안 씀) 예전: 표지판 먼 끝이 이 행에 오면 도착
    # 표지판을 따라가다 화면에서 완전히 사라지면 '도착' (가까운 끝이 sign_gone_row 아래까지 왔다가 sign_gone_sec 동안 안 보임)
    sign_gone_row: float = 0.85
    sign_gone_sec: float = 0.3
    sign_advance_m: float = 0.02     # 도착(사라짐) 뒤 곧장 더 가는 거리 (현장: 0.07 은 5cm 쯤 더 가서 돎). R2 는 plan_lane1 에서 0.07 (2026-10-09 현장: 0.05 는 R2 에서 일찍 꺾어 가벽을 봄,
                                     # 0.12 는 R1 에서 너무 가서 돈 뒤 R2 가 화면 오른쪽 끝에 걸려 못 찾음). 카메라 앞 약 10cm 는 안 보이므로 0 이면 표지판 끝 약 10cm 앞에서 돈다
    sign_turn_deg: float = 90.0      # 표지판에서 제자리 회전 각도 (park_turn_w 속도로, 시간으로 잰다)
    sign_search_sec: float = 6.0     # 표지판을 지난 뒤 다음 표지판을 찾으며 곧장 가는 최대 시간. 넘으면 남은 경로를 버리고 흰 차선으로
    sign_min_area: float = 0.002     # 파란 표지판 최소 면적 / 영상 면적
    sign_start_row: float = 0.65     # 표지판 가까운 끝이 이 행 아래로 와야 다가가기 시작 (그 전에는 차선을 따라간다. 2026-10-09: 0.55 는 횡단보도 지나 R1 이 보이자마자 차선을 버리고 감)
    sign_search_row: float = 0.40    # 표지판을 지나 다음 표지판을 찾으며 곧장 가는 중에는 이 행부터
    sign_max_x: float = 0.45         # 화면 가운데에서 이 범위 안(내 차선 앞)의 표지판만 (옆 차선 표지판 무시)
    sign_search_x: float = 0.95      # 표지판을 지나 다음 표지판을 찾는 중에는 이만큼 옆까지 (2026-10-09 pinky1: 돈 뒤 R2 가 x 0.9 에 보였다)
    sign_cw_block_row: float = 0.45  # 표지판 가까운 끝이 이 행 아래로 보이면 횡단보도 검출을 끈다 (표지판 우선)
    sign_long_ratio: float = 3.0     # sign_shape 일 때: 파란 덩어리의 긴 변/짧은 변이 이 이상이면 직우(긴 화살표), 아니면 우회전 양방향
    # 2026-10-09 현장: 카메라가 낮아(6.5cm) 바닥 표지판이 납작하게 보여 우회전 표지판도 길쭉하다 -> 모양 구분이 틀린다.
    # 색으로 찾을 때는 종류를 정하지 않고('blue') 가장 가까운 표지판을 경로의 다음 표지판으로 본다. 종류 구분은 YOLO 가 한다.
    sign_shape: bool = False
    # YOLO 표지판 종류를 경로 판단에 쓸지. False 면 YOLO 가 찾은 표지판도 종류 없이('blue') 가장 가까운 것을 다음 표지판으로 본다
    # (2026-10-09 best_1009.pt: 표지판 라벨을 모양으로 자동 생성해 turn/straight_right 가 뒤바뀌어 학습됨 -> 라벨을 고칠 때까지 False)
    sign_use_kind: bool = False
    # YOLO 표지판은 실제 파랑이 들어 있어야 인정한다 (2026-10-09: 흰 차선 한 토막을 straight_right 0.52 로 보고 다가감)
    sign_conf: float = 0.5           # 이 신뢰도 이상
    sign_blue_frac: float = 0.15     # YOLO 가 칠한 영역 중 이 비율 이상이 파랑(blue_mask)
    # 표지판 정렬: 직우 표지판의 긴 축이 비스듬히 보이면 제자리에서 돌아 정면으로 맞춘 뒤 다가간다
    # (2026-10-09 pinky2: 우회전 코너를 넓게 돌아 직우 표지판에 비스듬히 들어가 칸으로 비뚤게 꺾였다)
    sign_align_deg: float = 8.0      # 축이 이보다 더 틀어져 보이면 돈다 (화면 각도, 0 = 끔)
    sign_align_kinds: str = 'straight_right'   # 경로에서 이 종류 표지판에만
    sign_align_w: float = 0.4        # 제자리 회전 속도 rad/s
    sign_align_sec: float = 15.0     # 정면 맞추기는 이 시간까지만 (못 맞추면 그냥 간다)
    sign_step_min_sec: float = 0.15  # 정면 맞추기: 한 번에 도는 시간 (조금 틀어짐 .. 많이 틀어짐)
    sign_step_max_sec: float = 0.6
    sign_settle_sec: float = 0.5     # 돈 뒤 멈춰서 화면이 가라앉기를 기다렸다 다시 잰다
    sign_align_confirm: int = 3      # 멈춘 채로 이만큼 연속 맞아야 다 맞춘 것
    sign_center_row: float = 0.65    # 표지판 가까운 끝이 이 행 아래로 오면 차선 대신 표지판 가운데를 보고 간다 (위에 올라타게)
    sign_arrive_x: float = 0.5       # 표지판이 이보다 옆에서 사라지면 '도착'이 아니라 옆을 지나친 것 -> 그쪽으로 돌아 다시 찾는다
    sign_find_sec: float = 3.0       # 그렇게 다시 찾는 시간 (넘으면 도착으로 본다)
    sign_blue_row: float = 0.80      # 다가가다 YOLO 가 놓쳐도 이 행 아래에 파랑(색)이 남아 있으면 아직 도착이 아니다
    sign_blue_dx: float = 0.35       # 쫓던 표지판과 가로로 이만큼 안의 파랑만
    sign_blue_sec: float = 4.0       # YOLO 가 이보다 오래 못 보면 색만 남아 있어도 도착으로 본다
    crossing_straight_sec: float = 2.0   # 건너기 시작해서, 또 줄무늬가 발밑으로 사라진 뒤 이만큼은 곧장 (줄무늬·가벽에 차선이 헷갈린다)
    lane2_exit_max_sec: float = 90.0 # 칸에서 1차선 깃발이 이만큼 안 내려가도 나간다
    sign_arrive_far_row: float = 0.75   # 좌/우회전 표지판은 먼 끝이 이 행 아래로 오면 도착 (다 사라질 때까지 안 간다)
    sign_wall_sec: float = 1.0       # 표지판 위에서 앞이 이만큼 막혀 있으면 표지판 끝으로 보고 돈다
    lane1_exit_m: float = 0.40       # 1차선: 두 번째 표지판(R2)을 돈 뒤 이만큼 곧장 가고(S 를 지나) 그다음 흰 차선 (현장 요청)
    wall_turn_m: float = 0.20        # 차선 따라가다 정면 벽이 이보다 가까우면 앞 대각선이 더 트인 쪽으로 꺾는다
    wall_avoid_m: float = 0.15       # 앞 대각선(wall_avoid_from~to 도)에 벽이 이보다 가까우면 반대쪽으로 꺾는다 (0 = 끔)
    wall_avoid_w: float = 1.0        # 아주 붙었을 때 더하는 회전 속도 rad/s
    wall_avoid_from: float = 15.0
    wall_avoid_to: float = 70.0
    turn_lead_deg: float = 5.0       # 오도메트리로 돌 때 목표보다 이만큼 일찍 멈춘다 (멈추는 동안 더 돈다)
    sign_face_x: float = 0.35        # 표지판이 이보다 옆에 보이면 먼저 제자리에서 돌아 가운데로
    sign_align_plans: str = 'plan_lane2'   # 정면 맞추기를 하는 경로 (1차선은 S 를 U턴 직후 발밑에서 보므로 안 한다)
    sign_align_near_row: float = 0.75   # 표지판 가까운 끝이 이 행 아래로 오면 (= 바로 앞) 멈추고 맞춘다
    sign_align_far_row: float = 0.70 # 표지판 먼 끝이 이 행보다 위에 보일 때만 (발밑에 깔리면 축이 안 보인다)
    sign_horizon_row: float = 0.30   # 바닥과 나란한 선이 모이는 소실점 높이 (화면 위에서 / 높이)
    # 초록 칸 끝 선은 가로로 긴 띠만 (2026-10-09: 주행 중 LED 초록빛이 바닥에 비친 것)
    zone_min_width: float = 0.15     # 영상 폭의 이 비율 이상
    zone_max_aspect: float = 0.6     # 높이 / 폭 이 이하
    # 햇빛이 비친 파란 테이프는 S 25 안팎, V 220 으로 하얗게 뜬다 (카펫은 H 55 근처라 색상으로 갈린다). 밝은 곳에서 이 범위도 파랑으로 본다
    # 2026-10-09: 초록 선이 park_line_row(0.80)에 오면 카메라 앞 약 14cm 다 (높이 6.5cm, 8도 숙임).
    # 거기서 바로 돌면 칸 입구에 너무 가까워 지나가는 로봇이 화면을 꽉 채운다 -> 이만큼 더 가서 초록 선 위에서 돈다
    zone_advance_m: float = 0.12
    blue_glare_lo: List[int] = field(default_factory=lambda: [88, 18, 170])
    blue_glare_hi: List[int] = field(default_factory=lambda: [130, 255, 255])

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
    wall_blob_h: float = 0.30        # 흰 덩어리가 이만큼 키가 크고 (횡단보도 줄무늬 0.24~0.28)
    wall_blob_area: float = 0.05     # 영상의 이 비율보다 넓고 (줄무늬 0.03, 가벽 0.06~0.22)
    wall_blob_fill: float = 0.45     # 자기 상자를 이만큼 채우면 벽의 면 (비스듬한 차선 테이프는 0.3 아래)
    wall_from_top: bool = True       # 화면 맨 위부터 이어져 내려오는 흰색(벽 면)은 차선에서 뺀다
    lane_wall_band: float = 0.18     # 그 아래 이만큼(영상 높이 비율) 안에서도 lane_wall_base 보다 넓게 흰 행은 벽 밑단으로 지운다
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
    ('sign_gone_row', 0.50, 1.00, 0.02),
    ('sign_advance_m', 0.0, 0.40, 0.01),
    ('sign_turn_deg', 30.0, 150.0, 5.0),
    ('lane1_exit_m', 0.0, 1.0, 0.05),
    ('sign_arrive_far_row', 0.50, 1.00, 0.05),
    ('sign_align_deg', 0.0, 40.0, 1.0),
    ('zone_advance_m', 0.0, 0.30, 0.01),
    ('side_slow_m', 0.0, 0.30, 0.01),
    ('side_stop_m', 0.0, 0.20, 0.01),
    ('side_wall_len', 0.10, 0.50, 0.01),
    ('conf', 0.1, 0.9, 0.05),
]
