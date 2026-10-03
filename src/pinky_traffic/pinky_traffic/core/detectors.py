"""차선/횡단보도 검출기 두 종류. 둘 다 detect(frame_bgr) -> (Perception, masks dict).

HsvDetector  : 색(inRange)으로 찾는다. 학습 없이 바로 되고, 자동 라벨링에도 쓴다.
YoloDetector : 학습한 YOLO-seg (best.pt). 클래스 left / right / crosswalk (+ robot 선택).
"""
import time

import cv2
import numpy as np

from .perception import (LaneMemory, lane_from_masks, split_lane_mask, stripes_are_crosswalk, resize_to)


class HsvDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.memory = LaneMemory()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def color_mask(self, frame, lo, hi):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask[: int(self.cfg.roi_top * frame.shape[0])] = 0
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)   # 오프닝 = 작은 잡음 제거

    def masks(self, frame):
        """처리 크기 frame -> {'left','right','crosswalk'} 마스크와 횡단보도 판정."""
        cfg = self.cfg
        lane = self.color_mask(frame, cfg.lane_hsv_lo, cfg.lane_hsv_hi)
        if cfg.crosswalk_hsv_lo is not None and cfg.crosswalk_hsv_hi is not None:
            crosswalk = self.color_mask(frame, cfg.crosswalk_hsv_lo, cfg.crosswalk_hsv_hi)
            lane = cv2.bitwise_and(lane, cv2.bitwise_not(crosswalk))
            left, right, _, _ = split_lane_mask(lane, cfg, self.memory, separate_crosswalk=False)
            found = None      # 면적으로 판정
        else:
            left, right, crosswalk, boxes = split_lane_mask(lane, cfg, self.memory, separate_crosswalk=True)
            found = stripes_are_crosswalk(boxes, cfg, frame.shape[0])
        return {'left': left, 'right': right, 'crosswalk': crosswalk}, found

    def detect(self, frame):
        t0 = time.perf_counter()
        frame = resize_to(frame, self.cfg.proc_width)
        masks, found = self.masks(frame)
        p = lane_from_masks(masks['left'], masks['right'], masks['crosswalk'], self.cfg, self.memory, found)
        p.ms = (time.perf_counter() - t0) * 1000
        return p, masks, frame


class YoloDetector:
    def __init__(self, cfg):
        from ultralytics import YOLO   # 여기서만 필요 (hsv 만 쓸 때는 설치 안 해도 된다)
        self.cfg = cfg
        self.memory = LaneMemory()
        self.model = YOLO(cfg.weights)
        self.names = self.model.names            # {0: 'left', ...}
        self.model.predict(np.zeros((cfg.imgsz, cfg.imgsz, 3), np.uint8), imgsz=cfg.imgsz, verbose=False)  # 워밍업

    def detect(self, frame):
        t0 = time.perf_counter()
        cfg = self.cfg
        frame = resize_to(frame, cfg.proc_width)
        h, w = frame.shape[:2]
        r = self.model.predict(frame, conf=cfg.conf, imgsz=cfg.imgsz, verbose=False)[0]
        masks = {name: np.zeros((h, w), np.uint8) for name in ('left', 'right', 'crosswalk', 'lane', 'robot')}
        obstacle_y = 0.0
        if r.boxes is not None and len(r.boxes):
            classes = r.boxes.cls.cpu().numpy().astype(int)
            polys = r.masks.xy if r.masks is not None else [None] * len(classes)
            boxes = r.boxes.xyxy.cpu().numpy()
            for poly, cls_id, box in zip(polys, classes, boxes):
                name = self.names.get(int(cls_id), str(cls_id))
                if name == 'robot':
                    obstacle_y = max(obstacle_y, float(box[3] / (h - 1)))
                if name in masks and poly is not None and len(poly) >= 3:
                    cv2.fillPoly(masks[name], [np.asarray(poly, np.int32)], 255)
        top = int(cfg.roi_top * h)
        for name in ('left', 'right', 'lane'):
            masks[name][:top] = 0
        if masks['lane'].any():
            # 'lane' 한 클래스로만 학습한 모델이면 위치로 좌/우를 나눈다
            left, right, _, _ = split_lane_mask(masks['lane'], cfg, self.memory)
            masks['left'] |= left
            masks['right'] |= right
        p = lane_from_masks(masks['left'], masks['right'], masks['crosswalk'], cfg, self.memory)
        p.obstacle_y = obstacle_y
        p.ms = (time.perf_counter() - t0) * 1000
        del masks['lane']
        return p, masks, frame


def make_detector(cfg):
    if cfg.backend == 'yolo':
        return YoloDetector(cfg)
    if cfg.backend == 'hsv':
        return HsvDetector(cfg)
    raise ValueError(f"backend 는 'hsv' 또는 'yolo' 입니다: {cfg.backend}")
