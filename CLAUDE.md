# pinky_traffic — 작업 규칙 (Claude 세션 공용)

Pinky Pro(ROS2 Jazzy, Ubuntu 24.04)가 테이프 차선을 따라 돌고 횡단보도 앞에서 멈추는 프로젝트.
**1대 시연이 먼저**, 그다음 2대. 전체 설명은 README.md, 현장 순서는 docs/RUNBOOK.md.

## 꼭 지킬 것

- 사용자에게는 한국어로, 쉬운 말로.
- 코드를 바꾸면 `cd src/pinky_traffic && python3 -m pytest test -q` 를 돌려 통과를 확인한다 (약 80초).
- `core/` 는 ROS 를 import 하지 않는다 (시뮬레이터·테스트가 같은 코드를 쓴다). ROS 는 `nodes/` 에만.
- 로봇을 움직이는 명령(`cmd_vel` 발행, START)은 사용자가 로봇 옆에 있을 때만. 속도 기본값을 올리지 않는다.
- YOLO 학습 시 `fliplr=0` (좌우 뒤집으면 left/right 라벨이 바뀐다).
- 진행 상황은 `docs/PROGRESS.md` 맨 위에 날짜·시각과 함께 한 줄씩 추가한다. 노션 진행 기록 페이지에도 같은 내용을 올린다
  (https://app.notion.com/p/01-3e9a3f519ab8803ab52ec7dbeffe8dd3 — 2026-10-02 기준 Claude 노션 연결에서 접근 불가였음. 접근되면 거기에 기록).
- 커밋: 작업 단위가 끝나고 pytest 가 통과하면 **담당 파일만** `git add <파일>` 로 골라 커밋한다.
  `git add -A` / `git add .` 금지 (다른 세션이 고치던 파일이 섞인다). 메시지는 한국어 한 줄 + 필요하면 본문.
- push 는 사용자가 하라고 할 때만. 원격: https://github.com/ddaokki/pinky_traffic (**공개 레포**)
- 공개 레포이므로 비밀번호·토큰·개인 연락처·수업 PDF 원문을 파일에 넣지 않는다.
  `data/` `runs/` `*.pt` 는 .gitignore 로 빠져 있다 (사진·모델은 git 에 안 올린다).

## 용어

- **색 주행(HSV)**: 카메라 그림에서 테이프 색 범위(`config/field.yaml` 의 HSV 값)에 맞는 픽셀을 차선으로 보고 따라가는 방식. 학습 없이 바로 된다. 기본값.
- **YOLO 주행**: 학습한 `models/best.pt` 가 차선·횡단보도를 찾는 방식. `scripts/drive.sh pinky1 23 yolo`.
- `models/synth_best.pt`: 합성(가짜 그림) 데이터로 미리 학습한 출발점 (mask mAP50 0.979). 실제 트랙용 `best.pt` 는 현장 사진으로 만든다.

## 구조

- `src/pinky_traffic/pinky_traffic/core/` config, perception(마스크→offset), detectors(HsvDetector/YoloDetector),
  controller(PID+상태머신), coordinator(LockManager, DashLink), driver(묶음)
- `nodes/lane_driver.py` PC 에서 실행. 구독 `camera/image_raw/compressed`, `scan` → 발행 `cmd_vel`
- `nodes/camera_pub.py` 로봇에서 실행 (bringup 은 카메라를 발행하지 않는다)
- `dashboard/` 표준 라이브러리 HTTP 서버 + index.html (포트 8088)
- `sim/` 코스 그림을 투시 변환해 카메라 영상으로 만드는 시뮬레이터
- `tools/` capture, hsv_tuner, autolabel, make_synth_dataset, train, eval_detector, run_sim
- `config/field.yaml` 현장 값

## 환경

- `source scripts/env.sh [도메인번호]` — ROS2 + 워크스페이스 + PYTHONPATH
- 시스템 파이썬: opencv 4.6, numpy 1.26, rclpy. YOLO(ultralytics, torch CPU)는 `~/venv/yolo` 에만 있다.
- GPU 없음 (Intel Iris Xe). 학습은 CPU 또는 Colab.
- 로봇: pinky1 = 192.168.0.5, pinky2 = 192.168.0.7 (ssh pinky@IP, 비밀번호는 수업 자료/사용자에게). 도메인 별칭 ros23 / ros24.
- 2대는 ROS_DOMAIN_ID 가 달라 서로 토픽이 안 보인다. 로봇 간 약속은 대시보드 서버(HTTP)로 한다.

## 병렬 작업 시 파일 담당 (겹치지 않게)

| 세션 | 만질 수 있는 곳 |
|---|---|
| 인식·학습 | `core/perception.py` `core/detectors.py` `tools/` `data/` `models/` `docs/TRAINING.md` |
| 주행·튜닝 | `core/controller.py` `core/driver.py` `nodes/` `config/` `docs/RUNBOOK.md` |
| 관제·기록 | `dashboard/` `core/coordinator.py` `docs/TESTCASES.md` `docs/PROGRESS.md` 노션 |
| 공용 | `core/config.py` 는 값 추가만, 기존 이름·기본값 변경은 사용자에게 물어본다 |
