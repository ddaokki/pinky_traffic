"""상대 로봇 자동 라벨링: 가만히 선 로봇의 카메라로 찍은 사진에서 '빈 배경과 다른 부분'을 robot 으로 라벨링한다.

초록 칸 안에서 입구를 보고 선 로봇(카메라 고정) 앞으로 다른 로봇을 지나가게 하며 찍는다.
배경 사진(아무도 없을 때)과 비교해 크게 달라진 덩어리 하나를 robot 으로 본다. 차선·표지판은 autolabel 과 같이 색으로 라벨링한다.

  python3 -m pinky_traffic.tools.robot_autolabel --images data/robot_raw --background data/robot_raw/bg \\
      --out data/lane_ds --config src/pinky_traffic/config/field.yaml --preview out/robot_preview

--background: 빈 배경 사진 폴더 (여러 장이면 중앙값). 카메라가 움직였으면 배경을 다시 찍어야 한다.
"""
import argparse
import glob
import os

import cv2
import numpy as np

from ..core.config import Config
from ..core.detectors import HsvDetector, blue_signs
from ..core.perception import resize_to
from .dataset import DatasetWriter, masks_to_label, draw_label


def list_images(folder):
    return sorted(sum((glob.glob(os.path.join(folder, ext)) for ext in ('*.jpg', '*.jpeg', '*.png')), []))


def robot_mask(frame, background, thresh=40, min_area=0.01, kernel=5, floor_row=0.45):
    """배경과 다른 부분 중 가장 큰 덩어리 (없으면 None). 그림자처럼 옅은 차이는 thresh 로 거른다."""
    diff = cv2.absdiff(cv2.GaussianBlur(frame, (5, 5), 0), cv2.GaussianBlur(background, (5, 5), 0)).max(axis=2)
    mask = np.uint8(diff >= thresh) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k, iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = mask.shape
    # 바닥까지 내려온 덩어리만 (벽 너머로 지나가는 사람 등은 화면 위쪽에만 있다)
    ok = [i for i in range(1, n) if stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] >= floor_row * h]
    if not ok:
        return None
    i = max(ok, key=lambda j: stats[j, cv2.CC_STAT_AREA])
    if stats[i, cv2.CC_STAT_AREA] < min_area * w * h:
        return None
    out = np.uint8(labels == i) * 255
    contours, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(out)
    cv2.drawContours(filled, contours, -1, 255, -1)     # 바퀴 사이 구멍 메우기
    return filled


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--images', required=True)
    parser.add_argument('--background', required=True, help='빈 배경 사진 폴더')
    parser.add_argument('--out', required=True)
    parser.add_argument('--config', default=None)
    parser.add_argument('--thresh', type=int, default=40, help='배경과 이만큼(0~255) 다르면 robot 후보')
    parser.add_argument('--min-area', type=float, default=0.01, help='robot 덩어리 최소 면적 / 영상 면적')
    parser.add_argument('--val', type=float, default=0.2)
    parser.add_argument('--preview', default=None)
    args = parser.parse_args()
    cfg = Config.load(args.config)
    cfg.lane_role, cfg.sign_shape = 0, True
    bgs = [resize_to(cv2.imread(f), cfg.proc_width) for f in list_images(args.background)]
    if not bgs:
        raise SystemExit(f'배경 사진이 없습니다: {args.background}')
    background = np.median(np.stack(bgs), axis=0).astype(np.uint8)
    writer = DatasetWriter(args.out, args.val)
    if args.preview:
        os.makedirs(args.preview, exist_ok=True)
    found = 0
    files = [f for f in list_images(args.images) if os.path.dirname(f) != os.path.abspath(args.background)]
    for path in files:
        image = cv2.imread(path)
        if image is None:
            continue
        detector = HsvDetector(cfg)
        p, masks, small = detector.detect(image)
        if not p.crosswalk:
            masks['crosswalk'] = None
        robot = robot_mask(small, background, args.thresh, args.min_area)
        if robot is not None:
            found += 1
            for name in ('left', 'right', 'crosswalk'):         # 로봇에 가려진 부분은 차선이 아니다
                if masks.get(name) is not None:
                    masks[name] = masks[name] & ~robot
        masks['robot'] = robot
        blue = detector.blue_mask(small)
        if robot is not None:
            blue &= ~robot                                       # 로봇 바퀴(하늘색)를 표지판으로 보지 않게
        n, labels = cv2.connectedComponents(blue)
        masks['turn'], masks['straight_right'] = np.zeros_like(blue), np.zeros_like(blue)
        for i in range(1, n):
            part = np.uint8(labels == i) * 255
            sign = blue_signs(part, cfg)
            if sign:
                masks[sign[0][0]] |= part
        lines = masks_to_label(masks)
        name = 'robot_' + os.path.splitext(os.path.basename(path))[0]
        writer.add(name, small, lines)
        if args.preview:
            cv2.imwrite(os.path.join(args.preview, name + '.jpg'), draw_label(small, lines))
    print(f'{len(files)}장 중 robot 이 보인 사진 {found}장 -> {writer.counts}')
    print(f'data.yaml: {writer.finish()}')


if __name__ == '__main__':
    main()
