# Google Colab 에서 학습할 때 셀에 차례로 붙여 넣는다 (수업 Appendix 6 과 같은 흐름).
# 준비: data/lane_ds 폴더(train/ valid/ data.yaml)를 구글 드라이브의 MyDrive/pinky/lane_ds 로 올린다.
# 런타임 -> 런타임 유형 변경 -> T4 GPU

# --- 셀 1 ---
from google.colab import drive
drive.mount('/content/drive')

# --- 셀 2 ---
# !pip install ultralytics

# --- 셀 3: data.yaml 의 path 를 드라이브 경로로 바꾼다 ---
import yaml
DATA = '/content/drive/MyDrive/pinky/lane_ds'
cfg = yaml.safe_load(open(f'{DATA}/data.yaml'))
cfg.update(path=DATA, train='train/images', val='valid/images')
yaml.safe_dump(cfg, open(f'{DATA}/data.yaml', 'w'), sort_keys=False)
print(cfg)

# --- 셀 4: 학습 ---
from ultralytics import YOLO
model = YOLO('yolo11n-seg.pt')
# fliplr=0 이 중요하다: 좌우를 뒤집으면 left/right 라벨이 거꾸로 된다
model.train(data=f'{DATA}/data.yaml', epochs=100, imgsz=320, fliplr=0.0, mosaic=0.0,
            degrees=3.0, translate=0.05, scale=0.15, hsv_h=0.02, hsv_s=0.5, hsv_v=0.4, patience=30)

# --- 셀 5: best.pt 를 드라이브로 복사 (세션이 끊기면 /content 는 사라진다) ---
import shutil
shutil.copy('/content/runs/segment/train/weights/best.pt', f'{DATA}/best.pt')
print('드라이브에서 best.pt 를 내려받아 PC 의 ~/pinky_traffic_ws/models/best.pt 로 둔다')
