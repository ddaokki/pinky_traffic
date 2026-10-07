"""YOLO-seg 데이터셋 쓰기 공통부.

라벨 한 줄 = "클래스번호 x1 y1 x2 y2 ..." (다각형, 0~1 로 정규화). Roboflow 의 YOLOv11 export 와 같은 형식.
폴더 구조도 Roboflow 와 같게 만든다: <out>/{train,valid}/{images,labels}, <out>/data.yaml
"""
import os
import random

import cv2
import yaml

from ..core.perception import CLASSES


def mask_to_lines(mask, class_id, min_area=40, epsilon=1.0):
    """마스크 -> 라벨 줄 목록. 덩어리마다 한 줄."""
    h, w = mask.shape[:2]
    lines = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        poly = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(poly) < 3:
            continue
        coords = ' '.join(f'{x / w:.5f} {y / h:.5f}' for x, y in poly)
        lines.append(f'{class_id} {coords}')
    return lines


def masks_to_label(masks):
    lines = []
    for class_id, name in enumerate(CLASSES):
        mask = masks.get(name)
        if mask is not None:
            lines += mask_to_lines(mask, class_id)
    return lines


class DatasetWriter:
    def __init__(self, out, val_ratio=0.2, seed=0):
        self.out = os.path.abspath(out)
        self.val_ratio = val_ratio
        self.rng = random.Random(seed)
        self.counts = {'train': 0, 'valid': 0}
        for split in self.counts:
            os.makedirs(os.path.join(self.out, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.out, split, 'labels'), exist_ok=True)

    def add(self, name, image, lines):
        split = 'valid' if self.rng.random() < self.val_ratio else 'train'
        cv2.imwrite(os.path.join(self.out, split, 'images', name + '.jpg'), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(os.path.join(self.out, split, 'labels', name + '.txt'), 'w') as f:
            f.write('\n'.join(lines))
        self.counts[split] += 1
        return split

    def finish(self):
        data = {'path': self.out, 'train': 'train/images', 'val': 'valid/images',
                'nc': len(CLASSES), 'names': list(CLASSES)}
        path = os.path.join(self.out, 'data.yaml')
        with open(path, 'w') as f:
            yaml.safe_dump(data, f, sort_keys=False)
        return path


def draw_label(image, lines):
    """라벨을 그림 위에 그려 눈으로 검수."""
    colors = [(255, 120, 0), (0, 160, 255), (0, 255, 255), (255, 0, 255), (60, 200, 60)]
    out = image.copy()
    h, w = out.shape[:2]
    for line in lines:
        parts = line.split()
        pts = [(int(float(x) * w), int(float(y) * h)) for x, y in zip(parts[1::2], parts[2::2])]
        import numpy as np
        cv2.polylines(out, [np.array(pts, np.int32)], True, colors[int(parts[0]) % len(colors)], 2)
        cv2.putText(out, CLASSES[int(parts[0])], pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[int(parts[0]) % len(colors)], 1)
    return out
