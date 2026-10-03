# pinky_traffic — Pinky Pro 차선 주행 + 횡단보도 정지

학원 바닥에 테이프로 차선과 횡단보도를 깔고, Pinky Pro 가 카메라로 차선을 따라 돌다가
횡단보도 앞에서 멈췄다 가는 프로젝트. **1대로 먼저 시연**하고, 그다음 2대로 넓힌다.

```
[로봇 Pinky Pro]                         [PC]
 bringup_robot.launch.xml  ── /scan ──▶  lane_driver  ──▶  /cmd_vel ──▶ 로봇 바퀴
 camera_pub.py ── /camera/image_raw/compressed ──▶   │
                                                     │ HTTP (상태·영상·명령·락)
                                                     ▼
                                              dashboard (브라우저 localhost:8088)
```

- 인식: `hsv`(색으로 찾기, 학습 없이 바로) 또는 `yolo`(YOLO11n-seg, 클래스 `left` `right` `crosswalk`)
- 제어: PID 조향 + 상태 머신 (`lane_follow → approach_crosswalk → stop_at_crosswalk → crossing`, `blocked`, `lost`, `estop`)
- 2대: 로봇마다 `ROS_DOMAIN_ID` 가 달라 서로 안 보인다. 횡단보도는 대시보드 서버의 락으로 한 대씩 통과, 추돌은 라이다 전방 거리로 막는다.
- 시뮬레이터: ROS 없이 코스 그림을 카메라 시점으로 투시 변환해서 같은 코드를 돌린다.

## 문서

| 문서 | 내용 |
|---|---|
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | 내일 현장 순서 (트랙 깔기 → 1대 → 2대) |
| [docs/TRAINING.md](docs/TRAINING.md) | 학습 방법 (사진 모으기 → 자동 라벨 → 학습 → 검증) |
| [docs/TESTCASES.md](docs/TESTCASES.md) | 자동 테스트 48개 + 현장 테스트케이스 24개 |
| [docs/PARALLEL.md](docs/PARALLEL.md) | Claude 를 병렬로 돌리는 법, 세션 공유 |
| [docs/PROGRESS.md](docs/PROGRESS.md) | 진행 기록 (노션에 붙여 넣는 용) |

## 지금 바로 (로봇 없이)

```bash
cd ~/pinky_traffic_ws
source scripts/env.sh

# 1) 테스트
cd src/pinky_traffic && python3 -m pytest test -q && cd ~/pinky_traffic_ws

# 2) 시뮬레이션을 창으로 보기 (ESC 종료)
python3 -m pinky_traffic.tools.run_sim
python3 -m pinky_traffic.tools.run_sim --robots 2 --coordinator

# 3) 대시보드로 보기: 터미널 1
scripts/dashboard.sh
#    터미널 2  →  브라우저 http://localhost:8088 에서 START
python3 -m pinky_traffic.tools.run_sim --dashboard --robots 2 --coordinator
```

## 폴더

```
src/pinky_traffic/
  pinky_traffic/core/        config · perception · detectors(hsv/yolo) · controller · coordinator · driver   (ROS 없음)
  pinky_traffic/nodes/       lane_driver(PC) · camera_pub(로봇)
  pinky_traffic/dashboard/   server.py · index.html · testcases.json
  pinky_traffic/sim/         track(코스·카메라) · runner(닫힌 루프)
  pinky_traffic/tools/       capture · hsv_tuner · autolabel · make_synth_dataset · train · eval_detector · run_sim
  config/field.yaml          현장 설정 (★ 표시부터 맞춘다)
  test/                      pytest
scripts/                     env.sh · dashboard.sh · drive.sh · robot_install.sh
colab/train_colab.py         Colab 학습 셀
models/                      best.pt 를 두는 곳
```
