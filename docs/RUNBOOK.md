# 현장 순서 (트랙 깔기 → 1대 → 2대)

원칙: **1대가 테스트케이스 0~2 그룹을 통과한 뒤에 2대로 간다.** 속도는 낮게 시작한다.

## 0. 가져갈 것 / 미리 확인

- 테이프: 차선용 1색 + (가능하면) 횡단보도용 다른 색 1색. 바닥과 대비가 큰 색, 무광.
- 줄자, 가위, 충전기. 로봇 배터리 7.2V 이상 (7V 이하로 떨어지면 충전이 안 될 수 있다).
- PC: `cd ~/pinky_traffic_ws && source scripts/env.sh && cd src/pinky_traffic && python3 -m pytest test -q` 가 통과하는지.

## 1. 트랙 깔기

Pinky 카메라는 바닥에서 6.5cm 높이, 화각 66도다. 바로 앞 10cm 는 못 보고, 차선이 넓으면 양쪽 선이 화면 밖으로 나간다.
시뮬레이터에서 아래 치수로 완주를 확인했다.

| 항목 | 권장 | 이유 |
|---|---|---|
| 차선 폭 (두 테이프 안쪽 사이) | **20~24cm** (시뮬 22cm) | 차체 폭 약 11cm. 25cm 를 넘으면 가까운 곳에서 선이 한 줄만 보인다 |
| 테이프 폭 | 18~24mm (48mm 도 가능) | |
| 곡선 반지름 (차선 중심) | **45cm 이상** | 더 급하면 바깥 선도 화면에서 사라진다 |
| 코스 | 모서리가 둥근 사각형, 2.4m × 1.6m 정도 | 시뮬 기본 코스와 같다 |
| 횡단보도 | **직선 구간**, 곡선이 끝나고 50cm 이상 뒤 | 정지 전에 차선을 똑바로 봐야 한다 |
| 횡단보도 모양 | 진행 방향으로 15cm 짜리 줄 4개를 차선 안에 나란히 | |
| 횡단보도 색 | 가능하면 차선과 다른 색 | 색으로 찾는 모드(hsv)가 훨씬 안정적. 같은 색이어도 동작은 한다 |
| 같은 색(흰색)일 때 | 줄 4개를 **좌우 차선에 붙이지 말고 3~5cm 띄운다**, 줄끼리도 띄운다 | 붙으면 차선과 한 덩어리로 보여 줄무늬(3개 이상)로 못 센다 |
| 주변 | 트랙 근처에 테이프와 같은 색 물건(흰 종이, 케이블) 치우기 | 오검출 |

주행 방향은 한쪽으로 정한다 (반시계면 안쪽 선이 left).

## 2. 로봇 1대 준비

로봇과 PC 가 같은 공유기에 있고 `ROS_DOMAIN_ID` 가 같아야 한다. (`.bashrc` 별칭: `pinky1`=192.168.4.1 · 도메인 24 (2026-10-03 현장 값. .bashrc 별칭이 다르면 고칠 것), `pinky2`=192.168.0.7, `ros23`/`ros24`)

```bash
# PC: 카메라 노드 파일을 로봇에 복사
scripts/robot_install.sh 192.168.4.1

# 로봇 (ssh pinky@192.168.4.1) — 터미널 2개
echo $ROS_DOMAIN_ID                                   # 이 값을 PC 에서도 쓴다
ros2 launch pinky_bringup bringup_robot.launch.xml
python3 ~/camera_pub.py --ros-args -p width:=320 -p height:=240 -p fps:=15.0 -p flip:=true   # pinky1 은 영상이 뒤집혀 나와 flip 필요
```

```bash
# PC: 확인
source scripts/env.sh 24                              # 로봇의 도메인 번호
ros2 topic list                                       # /cmd_vel /odom /scan /camera/image_raw/compressed
ros2 topic hz /camera/image_raw/compressed            # 10~15 Hz
ros2 run rqt_image_view rqt_image_view                # 영상 확인
```

- 영상 색이 이상하면(빨강↔파랑) `-p swap_rb:=true`, 뒤집혀 있으면 `-p flip:=true`.
- `camera_pub` 이 카메라를 못 열면: 주피터 노트북 커널이 카메라를 잡고 있는지 확인 (Shut Down All).
- 여럿이 같은 공유기를 쓰면 느려진다. 수업 자료대로 필요 없는 영상 토픽을 끄고, 그래도 느리면 fps 를 10.0 으로.
- fps 는 꼭 소수점으로 쓴다 (`fps:=15` 처럼 정수로 주면 노드가 죽는다).

## 3. 색 맞추기 (5분)

```bash
python3 -m pinky_traffic.tools.capture --out data/raw      # 로봇을 차선에 놓고 SPACE 로 10장쯤
python3 -m pinky_traffic.tools.hsv_tuner --images data/raw # 테이프만 하얗게 남게 → 출력된 두 줄을
#   src/pinky_traffic/config/field.yaml 의 lane_hsv_lo / lane_hsv_hi 에 붙여 넣기
python3 -m pinky_traffic.tools.hsv_tuner --images data/raw --key crosswalk   # 횡단보도가 다른 색일 때만
python3 -m pinky_traffic.tools.eval_detector --images data/raw --config src/pinky_traffic/config/field.yaml --save out/hsv
```

`lane_ok_%` 가 90 이상이고 `out/hsv` 그림에서 left(파랑)/right(주황)이 맞게 칠해지면 다음으로.

## 4. 1대 주행

```bash
# 터미널 A
scripts/dashboard.sh                 # 브라우저 http://localhost:8088
# 터미널 B
scripts/drive.sh pinky1 24           # 색(HSV) 으로 먼저
```

1. 대시보드에 pinky1 카드와 영상이 뜨는지 본다.
2. **바퀴를 띄운 채** START → 비상정지 (스페이스바도 비상정지). 바퀴가 서는지 확인.
3. 직선 시작점에 놓고 START. `v_max` 0.08 에서 시작.
4. 테스트케이스 표를 위에서부터 채운다 (통과/실패 버튼, 메모). `runs/날짜/` 에 저장된다.

증상별 조정 (대시보드 슬라이더, 바로 적용된다):

| 증상 | 조정 |
|---|---|
| 좌우로 흔들린다 | `kp` ↓ (1.6 → 1.2), `kd` ↑ |
| 곡선에서 바깥으로 밀린다 | `kp` ↑, `v_max` ↓, `slow_gain` ↑ |
| 곡선에서 안쪽 선을 밟는다 | `lookahead_row` ↑ (가까운 곳을 본다) |
| 횡단보도를 지나쳐서 선다 | `crosswalk_stop_row` ↓ (0.80 → 0.70) |
| 너무 멀리서 선다 | `crosswalk_stop_row` ↑ |
| 출발하자마자 같은 횡단보도에서 또 선다 | `crossing_sec` ↑ |
| 앞에 아무것도 없는데 blocked | field.yaml 의 `lidar_yaw_offset_deg` 를 180 으로 (라이다 0도 방향이 뒤일 때) |

잘 맞은 값은 `config/field.yaml` 에 옮겨 적는다 (슬라이더 값은 서버를 끄면 사라진다).

## 5. YOLO 로 바꾸기

[TRAINING.md](TRAINING.md) 대로 `models/best.pt` 를 만든 뒤:

```bash
scripts/drive.sh pinky1 24 yolo
```

HSV 로 먼저 달리게 해 두면, 그 주행 영상을 그대로 학습 데이터로 쓸 수 있다.

## 6. 2대

1대가 그룹 2 까지 통과한 뒤에.

```bash
# 로봇 2 준비는 2번과 같다 (도메인 번호만 pinky1 과 다르게, 예: 23)
scripts/robot_install.sh 192.168.0.7

# PC: 터미널 3개
scripts/dashboard.sh
scripts/drive.sh pinky1 24 yolo "" 1    # 1차선에 놓은 로봇
scripts/drive.sh pinky2 23 yolo "" 2    # 2차선에 놓은 로봇
```
(`scripts/start_all.sh` 로 한 번에 켜도 된다. 그때는 대시보드의 `1차선` / `2차선` 버튼으로 차선을 정한다.)

- 대시보드에서 `횡단보도 한 대씩 통과` 를 켠다 (`use_coordinator`, field.yaml 기본 켜짐). 깃발과 구간 락이 이걸로 돈다.
  락 칸에 "깃발 oncoming: pinky1 (오는 중)", "junction: 통과 중 …" 이 보인다.
- 표지판 경로(우/좌/직진 순서)는 field.yaml 의 `plan_lane1` `plan_lane2` `plan_lane2_pocket` `plan_lane2_exit`.
  표지판에서 서는 위치·회전 각도는 대시보드 슬라이더 `sign_stop_row` `sign_advance_m` `sign_turn_deg`.
- 두 대를 **같이 출발**시킨다. 2차선 로봇이 먼저 직우 표지판에 닿으면 깃발이 없어 바로 유턴한다 (기다리는 시간 `pocket_decide_sec`).
- 두 로봇은 도메인이 달라 ROS 로는 서로 안 보인다. 추돌 방지는 각자의 라이다(`/scan`) 전방 거리다.
  라이다 없는 로봇이면 뒤차의 `v_max` 를 앞차보다 낮게 두고 반 바퀴 간격으로 출발시킨다.
- 그룹 3 테스트케이스를 채운다.

## 끝나고

- `runs/날짜/` 의 CSV, `testcase_results.json` 을 보관.
- 잘 된 값은 `config/field.yaml` 에 반영하고 커밋.
