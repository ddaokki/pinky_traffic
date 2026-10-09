"""자동 라벨링: 찍은 사진을 HSV 검출기로 미리 라벨링해서 YOLO-seg 데이터셋으로 만든다.

Roboflow 에서 다각형을 손으로 그리는 대신, 색으로 찾은 영역을 라벨로 쓴다.
(틀린 것만 --review 로 골라내면 된다. 이 과정을 거치면 색이 달라져도 버티는 YOLO 모델이 나온다.)

  python3 -m pinky_traffic.tools.autolabel --images data/raw --out data/lane_ds --config config/field.yaml
  python3 -m pinky_traffic.tools.autolabel --images data/raw --out data/lane_ds --config config/field.yaml --review

--review: 한 장씩 보여준다.  y/SPACE = 채택,  n = 버림,  b = 배경(라벨 없이 채택),  ESC = 그만
left/right 는 '로봇이 차선 안에서 진행방향을 보고 찍은 사진' 이라는 가정으로 위치로 정한다.
"""
import argparse
import glob
import os

import cv2

from ..core.config import Config
import numpy as np

from ..core.detectors import HsvDetector, blue_signs
from .dataset import DatasetWriter, masks_to_label, draw_label


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--images', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--config', default=None, help='lane_hsv_* 가 맞춰진 yaml')
    parser.add_argument('--review', action='store_true')
    parser.add_argument('--val', type=float, default=0.2)
    parser.add_argument('--preview', default=None, help='라벨을 그린 그림을 저장할 폴더')
    args = parser.parse_args()
    cfg = Config.load(args.config)
    cfg.lane_role = 0                        # 역할이 있으면 파란 선 마스크가 차선 자리에 들어간다. 라벨은 따로 만든다
    cfg.sign_shape = True                    # 라벨은 종류가 있어야 한다 (모양 구분은 틀리기 쉬우니 --review 로 고친다)
    files = sorted(sum((glob.glob(os.path.join(args.images, ext)) for ext in ('*.jpg', '*.jpeg', '*.png')), []))
    if not files:
        raise SystemExit(f'사진이 없습니다: {args.images}')
    writer = DatasetWriter(args.out, args.val)
    if args.preview:
        os.makedirs(args.preview, exist_ok=True)
    kept = skipped = empty = 0
    for path in files:
        image = cv2.imread(path)
        if image is None:
            continue
        detector = HsvDetector(cfg)          # 사진마다 새로 (앞 사진의 기억이 섞이지 않게)
        p, masks, small = detector.detect(image)
        if not p.crosswalk:
            masks['crosswalk'] = None
        # 파란 표지판 (turn / straight_right): 색으로 찾은 덩어리를 모양(길쭉한가)으로 나눠 라벨로. --review 로 꼭 확인
        blue = detector.blue_mask(small)
        n, labels = cv2.connectedComponents(blue)
        masks['turn'], masks['straight_right'] = np.zeros_like(blue), np.zeros_like(blue)
        for i in range(1, n):
            part = np.uint8(labels == i) * 255
            found = blue_signs(part, cfg)
            if found:
                masks[found[0][0]] |= part
        lines = masks_to_label(masks)
        view = draw_label(small, lines)
        if args.review:
            cv2.imshow('autolabel (y 채택 / n 버림 / b 배경 / ESC)', cv2.resize(view, None, fx=2, fy=2))
            key = cv2.waitKey(0) & 0xFF
            if key == 27:
                break
            if key == ord('n'):
                skipped += 1
                continue
            if key == ord('b'):
                lines = []
        name = os.path.splitext(os.path.basename(path))[0]
        writer.add(name, small, lines)
        kept += 1
        empty += not lines
        if args.preview:
            cv2.imwrite(os.path.join(args.preview, name + '.jpg'), view)
    yaml_path = writer.finish()
    cv2.destroyAllWindows()
    print(f'채택 {kept} (라벨 없음 {empty}) / 버림 {skipped}  ->  {writer.counts}')
    print(f'data.yaml: {yaml_path}')


if __name__ == '__main__':
    main()
