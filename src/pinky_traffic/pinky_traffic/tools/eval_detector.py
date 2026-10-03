"""검출기를 사진 폴더/동영상에 돌려 숫자로 본다 (주행 전에 책상에서 확인).

  python3 -m pinky_traffic.tools.eval_detector --images data/raw --config config/field.yaml
  python3 -m pinky_traffic.tools.eval_detector --images data/raw --backend yolo --weights models/best.pt --save out/
  python3 -m pinky_traffic.tools.eval_detector --images data/raw --compare models/best.pt   # hsv 와 yolo 비교
  python3 -m pinky_traffic.tools.eval_detector --images data/raw --csv out/hsv.csv          # 사진별 결과 (오검출 찾기)

출력: 차선 인식률(ok), 양쪽/한쪽 비율, 횡단보도 검출 수, 평균 처리시간.
--compare: 같은 사진에서 두 방식의 offset 차이 평균 (작을수록 둘이 같은 곳을 본다).
"""
import argparse
import csv
import glob
import os

import cv2
import numpy as np

from ..core.config import Config
from ..core.detectors import make_detector
from ..core.perception import draw_debug


CSV_FIELDS = ['file', 'backend', 'ok', 'left_seen', 'right_seen', 'crosswalk', 'crosswalk_y', 'offset',
              'left_px_%', 'right_px_%', 'crosswalk_px_%', 'ms']


def run(cfg, files, save=None, table=None):
    detector = make_detector(cfg)
    rows = []
    for path in files:
        image = cv2.imread(path)
        if image is None:
            continue
        detector.memory.reset()
        p, masks, small = detector.detect(image)
        rows.append(p)
        if table is not None:
            px = {k: round(100 * cv2.countNonZero(m) / m.size, 2) if m is not None else 0.0
                  for k, m in masks.items()}
            table.append({'file': os.path.basename(path), 'backend': cfg.backend, 'ok': int(p.ok),
                          'left_seen': int(p.left_seen), 'right_seen': int(p.right_seen),
                          'crosswalk': int(p.crosswalk), 'crosswalk_y': round(p.crosswalk_y, 2),
                          'offset': round(p.offset, 3), 'left_px_%': px.get('left', 0.0),
                          'right_px_%': px.get('right', 0.0), 'crosswalk_px_%': px.get('crosswalk', 0.0),
                          'ms': round(p.ms, 1)})
        if save:
            os.makedirs(save, exist_ok=True)
            cv2.imwrite(os.path.join(save, os.path.basename(path)), draw_debug(small, p, masks, f'{cfg.backend} off={p.offset:+.2f}'))
    return rows


def summarize(name, rows):
    n = max(1, len(rows))
    both = sum(p.left_seen and p.right_seen for p in rows)
    one = sum(p.left_seen != p.right_seen for p in rows)
    out = {'backend': name, 'images': len(rows), 'lane_ok_%': round(100 * sum(p.ok for p in rows) / n, 1),
           'both_lines_%': round(100 * both / n, 1), 'one_line_%': round(100 * one / n, 1),
           'crosswalk_images': sum(p.crosswalk for p in rows), 'mean_ms': round(float(np.mean([p.ms for p in rows] or [0])), 1)}
    print(out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--images', required=True)
    parser.add_argument('--config', default=None)
    parser.add_argument('--backend', default=None)
    parser.add_argument('--weights', default=None)
    parser.add_argument('--compare', default=None, help='yolo 가중치 경로: hsv 와 비교')
    parser.add_argument('--save', default=None, help='디버그 그림 저장 폴더')
    parser.add_argument('--csv', default=None, help='사진별 결과를 저장할 csv 경로')
    args = parser.parse_args()
    files = sorted(sum((glob.glob(os.path.join(args.images, ext)) for ext in ('*.jpg', '*.jpeg', '*.png')), []))
    if not files:
        raise SystemExit(f'사진이 없습니다: {args.images}')
    overrides = {k: v for k, v in (('backend', args.backend), ('weights', args.weights)) if v}
    cfg = Config.load(args.config, **overrides)
    table = [] if args.csv else None
    if args.compare:
        hsv = run(Config.load(args.config, backend='hsv'), files, args.save and os.path.join(args.save, 'hsv'), table)
        yolo = run(Config.load(args.config, backend='yolo', weights=args.compare), files,
                   args.save and os.path.join(args.save, 'yolo'), table)
        summarize('hsv', hsv)
        summarize('yolo', yolo)
        pairs = [(a.offset, b.offset) for a, b in zip(hsv, yolo) if a.ok and b.ok]
        if pairs:
            diff = np.abs(np.array(pairs)[:, 0] - np.array(pairs)[:, 1])
            print({'both_ok_images': len(pairs), 'mean_abs_offset_diff': round(float(diff.mean()), 3),
                   'crosswalk_agree_%': round(100 * np.mean([a.crosswalk == b.crosswalk for a, b in zip(hsv, yolo)]), 1)})
    else:
        summarize(cfg.backend, run(cfg, files, args.save, table))
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(table)
        print(f'사진별 결과 -> {args.csv}')


if __name__ == '__main__':
    main()
