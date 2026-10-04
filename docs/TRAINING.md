# 학습 방법

목표: 로봇 카메라 영상에서 `left`(왼쪽 차선) `right`(오른쪽 차선) `crosswalk`(횡단보도)를 픽셀 단위로 찾는
YOLO11n-seg 모델 `models/best.pt`. 수업 Appendix 5·6 의 흐름(모으기 → 라벨링 → 학습 → 검증)을 그대로 따르되,
라벨링을 손으로 하지 않고 색 검출 결과를 라벨로 쓴다.

## 왜 색 검출이 있는데 YOLO 를 학습하나

색(HSV) 은 조명이 바뀌거나 테이프와 비슷한 색 물건이 보이면 흔들린다. 색으로 만든 라벨을 여러 조명에서 모아 학습하면,
모델은 색이 아니라 모양과 위치로 차선을 배운다. 횡단보도를 차선과 같은 색으로 깔아도 구분한다.

## 1. 사진 모으기 (150~300장)

```bash
python3 -m pinky_traffic.tools.capture --out data/raw --every 0.5
# a = 자동 저장 켜기, 다른 터미널에서 teleop 으로 천천히 몬다
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

HSV 주행이 이미 되면 그냥 달리게 두고 자동 저장을 켜면 된다. 골고루 담을 것:

- 직선 / 곡선 / 횡단보도 앞 30cm·15cm·바로 앞
- 차선 가운데 / 왼쪽·오른쪽으로 치우침 / 비스듬히 틀어짐
- 조명: 전부 켬 / 일부 끔 / 창가 쪽 역광 / 사람 그림자
- 트랙 밖 바닥 20장쯤 (배경, 라벨 없음 → 엉뚱한 것을 안 잡게 된다)

## 2. 자동 라벨링

```bash
python3 -m pinky_traffic.tools.autolabel --images data/raw --out data/lane_ds \
    --config src/pinky_traffic/config/field.yaml --review
```

한 장씩 라벨이 그려져 나온다. `y` 채택, `n` 버림, `b` 배경으로 채택. 버리는 기준: 좌우가 바뀜, 선이 끊겨 칠해짐,
엉뚱한 것이 칠해짐. 조명이 달라 색 검출이 안 되는 사진은 `hsv_tuner` 로 그 조명에 맞는 값을 찾아 `--config` 를
바꿔 한 번 더 돌린다 (같은 `--out` 에 쌓인다).

손으로 고치고 싶으면 `data/lane_ds` 를 Roboflow 에 올려서(Instance Segmentation 프로젝트) 고친 뒤 YOLOv11 형식으로 내려받는다.

## 3. 학습

```bash
source ~/venv/yolo/bin/activate
cd ~/pinky_traffic_ws && export PYTHONPATH=src/pinky_traffic
python3 -m pinky_traffic.tools.train --data data/lane_ds/data.yaml --epochs 60
```

- 이 PC 는 GPU 가 없다. CPU 로 합성 데이터 480장·320px 기준 한 에폭에 약 1분이었다.
- 급하면 Colab T4: `data/lane_ds` 를 드라이브에 올리고 `colab/train_colab.py` 의 셀을 차례로 실행.
- **`fliplr=0` 을 지킬 것.** 좌우 뒤집기 증강을 켜면 left/right 라벨이 뒤바뀐 그림으로 학습한다. (`train.py` 와 Colab 셀에는 이미 들어 있다.)
- 시작 가중치를 `--model models/synth_best.pt` (합성 데이터로 미리 학습한 것)로 주면 적은 사진으로 빨리 수렴한다.

## 4. 검증 (주행 전에 책상에서)

```bash
python3 -m pinky_traffic.tools.eval_detector --images data/raw --config src/pinky_traffic/config/field.yaml \
    --compare models/best.pt --save out/cmp
```

사진별 결과가 필요하면 `--csv out/cmp.csv` 를 붙인다 (어느 사진에서 차선을 놓쳤는지, 횡단보도를 잡았는지 한 줄씩).

| 볼 것 | 기준 |
|---|---|
| 학습 로그 mask mAP50 | 0.9 이상 |
| `lane_ok_%` (yolo) | 95 이상 |
| `mean_abs_offset_diff` (hsv 와 yolo 의 목표점 차이) | 0.05 이하 |
| `crosswalk_agree_%` | 95 이상 |
| `mean_ms` | 70ms 이하 (15fps 를 따라가려면) |
| `out/cmp/yolo/*.jpg` 눈으로 | 좌우 색이 바뀐 그림이 없을 것 |

그다음 시뮬레이터가 아니라 실제로: `scripts/drive.sh pinky1 24 yolo`, 대시보드에서 `conf` 를 0.3~0.5 사이로 조정.

## 5. 더 잘 잡히게 (수업 Appendix 5 의 6-5 와 같은 순서)

1. 못 맞히는 장면을 찍어 더한다 — 주행 중 `lost` 가 뜬 자리, 좌우가 바뀐 자리
2. 데이터를 늘린다
3. 더 오래 학습한다 (`--epochs 150`)
4. 큰 모델 (`--model yolo11s-seg.pt`) — CPU 에서는 느려진다
5. 배경 사진을 넣는다 (전체의 10% 안쪽)

## 선택: 앞 로봇 인식 (`robot` 클래스)

라이다가 없는 로봇으로 2대를 돌릴 때만. Roboflow 에서 4번째 클래스 `robot` 을 추가해 Pinky 뒷모습을 라벨링하고,
`data.yaml` 의 names 를 `[left, right, crosswalk, robot]` 로 한다. `robot` 박스 아래 끝이 화면의 80% 아래로 내려오면 정지한다.

## 합성 데이터 (로봇 없이 파이프라인 확인용)

```bash
python3 -m pinky_traffic.tools.make_synth_dataset --out data/synth_ds --n 600 --tracks 12
python3 -m pinky_traffic.tools.train --data data/synth_ds/data.yaml --epochs 30 --name synth --out models/synth_best.pt
python3 -m pinky_traffic.tools.run_sim --backend yolo --weights models/synth_best.pt
```
