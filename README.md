# pinky_traffic — Pinky Pro 차선 주행 · 횡단보도 정지 · 2대 순서 제어

바닥에 테이프로 깐 차선을 Pinky Pro(ROS2 Jazzy)가 카메라로 보고 따라 달린다.
횡단보도 앞에서는 멈췄다가 지나가고, 앞이 막히면 라이다로 선다.
2대를 함께 달리게 하면 대시보드 서버가 갈림길과 횡단보도를 한 대씩 지나가도록 순서를 정해 준다.

```
[로봇 Pinky Pro ×2]                        [PC]
 bringup_robot.launch.xml  ── /scan ──▶  lane_driver (로봇마다 1개) ──▶ /cmd_vel ──▶ 로봇 바퀴
 camera_pub.py ── /camera/image_raw/compressed ──▶   │          └──▶ /set_led ──▶ 로봇 LED
 led_server                                          │ HTTP (상태·영상·명령·락)
                                                     ▼
                                              dashboard (브라우저 localhost:8088)
```

## 하는 일

- **차선 따라가기**: 색(HSV)으로 테이프를 찾거나 학습한 YOLO11n-seg(`left` `right` `crosswalk` + 표지판 `turn` `straight_right`)로 찾는다. PID 로 조향한다.
- **횡단보도**: 줄무늬를 보고 앞에서 3초 멈췄다가 지나간다.
- **장애물**: 라이다 전방 거리가 가까우면 선다. 막혀 있어도 제자리 회전은 한다.
- **2대 역할 나누기**
  - 갈림길 바닥에 파란 양방향 표지판 두 종류(우회전 양방향 / 직우 양방향)가 있다. 로봇은 표지판을 보면 그 자리까지 가서
    자기 경로대로 우회전·좌회전·직진하고, 경로의 표지판을 다 지나면 다시 흰 차선을 따라간다. 경로는 `config/field.yaml` 의 `plan_*`.
  - 1차선 로봇: 우회전 표지판을 보면 "간다" 깃발을 올리고 우 → 우(유턴) → 직진.
  - 2차선 로봇: 직우 표지판에서 깃발이 있으면 우회전해 초록 칸에 들어가 180도 돌아 기다린다. 상대가 칸 앞을 지나가면(카메라·라이다)
    또는 깃발이 내려가면 나와서 우 → 좌 → 좌로 1차선에 간다. 깃발이 없으면 직진 → 좌 → 좌.
  - 깃발과 유턴 구간 락은 대시보드 서버가 맡는다. 두 로봇은 `ROS_DOMAIN_ID` 가 달라 ROS 로는 서로 안 보인다.

  | 로봇 | 상황 | 표지판에서 하는 일 |
  |---|---|---|
  | 1차선 | 항상 | 우회전 양방향 **우** → 우회전 양방향 **우**(유턴) → 직우 **직진** → 흰 차선 |
  | 2차선 | 상대가 온다 (깃발) | 직우 **우**(초록 칸) → 초록 선 앞 180도, 대기 → 나와서 **우** → **좌** → **좌**(1차선으로) |
  | 2차선 | 상대가 안 온다 | 직우 **직진** → **좌** → **좌**(1차선으로) |
- **LED**: 달리면 초록, 서 있으면 빨강, 통로 안에서는 통로 색.
- **대시보드**: 로봇별 영상·상태, START/STOP/비상정지, 실시간 파라미터 슬라이더, 테스트케이스 기록.
- **시뮬레이터**: ROS 없이 코스 그림을 카메라 시점으로 투시 변환해서 같은 제어 코드를 돌린다.

## 실행

### 로봇 2대 (한 번에)

```bash
scripts/start_all.sh     # 로봇 bringup·카메라·LED → 대시보드 → 주행 노드 2개 → 브라우저
scripts/stop_all.sh      # 끄기
```

- 로봇 주소·도메인·차선 번호는 `scripts/start_all.sh` 맨 위 표에서 바꾼다.
- 대시보드 로봇 카드의 `1차선` / `2차선` 버튼으로 놓은 차선을 정한다 (`역할 없음` 이면 표지판을 보지 않고 흰 차선만 따라간다).
- 처음 한 번은 로봇 ssh 비밀번호를 물어보고, 그다음부터는 키로 접속한다.
- 주행 노드는 대시보드에서 START 를 누르기 전에는 움직이지 않는다.
- 하나씩 켜는 방법과 트랙 치수, 튜닝 표는 [docs/RUNBOOK.md](docs/RUNBOOK.md)에 있다.

### 로봇 없이

```bash
source scripts/env.sh
cd src/pinky_traffic && python3 -m pytest test -q && cd -     # 테스트

python3 -m pinky_traffic.tools.run_sim                        # 시뮬레이션 창 (ESC 종료)
python3 -m pinky_traffic.tools.run_sim --robots 2 --coordinator

scripts/dashboard.sh                                          # 대시보드와 함께: 터미널 1
python3 -m pinky_traffic.tools.run_sim --dashboard --robots 2 --coordinator   # 터미널 2 → localhost:8088 에서 START
```

## 문서

| 문서 | 내용 |
|---|---|
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | 트랙 깔기, 로봇 준비, 색 맞추기, 증상별 튜닝 |
| [docs/TRAINING.md](docs/TRAINING.md) | YOLO 학습 (사진 모으기 → 자동 라벨 → 학습 → 검증) |
| [docs/TESTCASES.md](docs/TESTCASES.md) | 자동 테스트와 현장 테스트케이스 |

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
scripts/                     start_all · stop_all · env · dashboard · drive · robot_install
colab/train_colab.py         Colab 학습 셀
models/                      best.pt 를 두는 곳
```
