"""검출기를 사진 폴더/동영상에 돌려 숫자로 본다 (주행 전에 책상에서 확인).

  python3 -m pinky_traffic.tools.eval_detector --images data/raw --config config/field.yaml
  python3 -m pinky_traffic.tools.eval_detector --images data/raw --backend yolo --weights models/best.pt --save out/
  python3 -m pinky_traffic.tools.eval_detector --images data/raw --compare models/best.pt   # hsv 와 yolo 비교

출력: 차선 인식률(ok), 양쪽/한쪽 비율, 횡단보도 검출 수, 평균 처리시간.
--compare: 같은 사진에서 두 방식의 offset 차이 평균 (작을수록 둘이 같은 곳을 본다).
"""
import argparse
import glob
import os

import cv2
import numpy as np

from ..core.config import Config
from ..core.detectors import make_detector
from ..core.perception import draw_debug


def run(cfg, files, save=None):
    detector = make_detector(cfg)
    rows = []
    for path in files:
        image = cv2.imread(path)
        if image is None:
            continue
        detector.memory.reset()
        p, masks, small = detector.detect(image)
        rows.append(p)
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
    args = parser.parse_args()
    files = sorted(sum((glob.glob(os.path.join(args.images, ext)) for ext in ('*.jpg', '*.jpeg', '*.png')), []))
    if not files:
        raise SystemExit(f'사진이 없습니다: {args.images}')
    overrides = {k: v for k, v in (('backend', args.backend), ('weights', args.weights)) if v}
    cfg = Config.load(args.config, **overrides)
    if args.compare:
        hsv = run(Config.load(args.config, backend='hsv'), files, args.save and os.path.join(args.save, 'hsv'))
        yolo = run(Config.load(args.config, backend='yolo', weights=args.compare), files, args.save and os.path.join(args.save, 'yolo'))
        summarize('hsv', hsv)
        summarize('yolo', yolo)
        pairs = [(a.offset, b.offset) for a, b in zip(hsv, yolo) if a.ok and b.ok]
        if pairs:
            diff = np.abs(np.array(pairs)[:, 0] - np.array(pairs)[:, 1])
            print({'both_ok_images': len(pairs), 'mean_abs_offset_diff': round(float(diff.mean()), 3),
                   'crosswalk_agree_%': round(100 * np.mean([a.crosswalk == b.crosswalk for a, b in zip(hsv, yolo)]), 1)})
    else:
        summarize(cfg.backend, run(cfg, files, args.save))


if __name__ == '__main__':
    main()
