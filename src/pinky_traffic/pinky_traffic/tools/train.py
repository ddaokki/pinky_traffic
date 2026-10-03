"""YOLO-seg 학습 (수업 Appendix 6 의 Colab 코드와 같은 호출을 내 PC 에서).

  source ~/venv/yolo/bin/activate
  python3 -m pinky_traffic.tools.train --data data/lane_ds/data.yaml --epochs 60

GPU 가 없으면 CPU 로 돈다 (320px, 사진 수백 장, nano 모델이면 수십 분).
급하면 Colab: colab/train_colab.py 내용을 셀에 붙여 넣는다.
끝나면 best.pt 를 models/ 로 복사해 준다.
"""
import argparse
import os
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', required=True, help='data.yaml 경로')
    parser.add_argument('--model', default='yolo11n-seg.pt', help="처음부터: yolo11n-seg.pt / 이어서: 이전 best.pt")
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--imgsz', type=int, default=320)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--name', default='lane')
    parser.add_argument('--out', default='models/best.pt')
    parser.add_argument('--device', default=None, help='cpu / 0 (GPU). 비우면 자동')
    args = parser.parse_args()
    from ultralytics import YOLO

    model = YOLO(args.model)
    # fliplr=0: 좌우를 뒤집으면 left/right 라벨이 거꾸로 되므로 반드시 끈다
    model.train(data=os.path.abspath(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                name=args.name, device=args.device, fliplr=0.0, mosaic=0.0, degrees=3.0, translate=0.05,
                scale=0.15, hsv_h=0.02, hsv_s=0.5, hsv_v=0.4, patience=20, workers=2, exist_ok=True, plots=True)
    best = os.path.join(str(model.trainer.save_dir), 'weights', 'best.pt')
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    shutil.copy(best, args.out)
    metrics = model.val(data=os.path.abspath(args.data), imgsz=args.imgsz, verbose=False)
    print(f'\nbest.pt -> {args.out}')
    print(f'mask mAP50 = {metrics.seg.map50:.3f}   mask mAP50-95 = {metrics.seg.map:.3f}')
    print('0.9 이상이면 주행에 써 볼 만하다. 낮으면 못 맞히는 장면을 더 찍어 다시 학습.')


if __name__ == '__main__':
    main()
