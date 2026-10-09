"""차선/횡단보도 검출기 두 종류. 둘 다 detect(frame_bgr) -> (Perception, masks dict).

HsvDetector  : 색(inRange)으로 찾는다. 학습 없이 바로 되고, 자동 라벨링에도 쓴다.
YoloDetector : 학습한 YOLO-seg (best.pt). 클래스 left / right / crosswalk (+ turn, straight_right, robot 선택).
               초록 칸 끝 선은 색으로 찾고, 표지판 클래스가 없는 모델이면 파란 표지판도 색으로 찾는다.
"""
import time

import cv2
import numpy as np

from .perception import (LaneMemory, Perception, lane_from_masks, split_lane_mask, stripes_are_crosswalk, resize_to,
                         remove_wall_base, route_end_bar, SIGNS)


def blue_signs(blue, cfg):
    """파란 마스크 -> 표지판 목록 [(종류, x, 먼 끝 행, 가까운 끝 행)] (가까운 것부터).
    종류는 'blue'(아무 표지판). cfg.sign_shape 이면 길쭉한 것(긴 ←→)은 직우 양방향, 아니면 우회전 양방향으로 본다."""
    h, w = blue.shape[:2]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(blue, connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < cfg.sign_min_area * w * h:
            continue
        ys, xs = np.nonzero(labels == i)
        kind = 'blue'
        if cfg.sign_shape:
            (_, _), (a, b), _ = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
            kind = 'straight_right' if max(a, b) >= cfg.sign_long_ratio * max(1.0, min(a, b)) else 'turn'
        out.append((kind, float((xs.mean() - w / 2.0) / (w / 2.0)), float(ys.min() / (h - 1)), float(ys.max() / (h - 1))))
    return sorted(out, key=lambda s: -s[3])


class HsvDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.memory = LaneMemory()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.route_seen = self.route_near = False
        self.route_end_y = 0.0
        self.prefer = ''                 # 'left' | 'right' : 제어기가 정한다. 그쪽 선만 보고 따라간다 (갈림길)
        self.signs = []                  # 파란 표지판 (lane_role 일 때)
        self.follow_zone = False         # 제어기가 정한다: 칸 안에서는 초록 선 가운데를 보고 간다
        self.zone_x = 0.0                # 초록 선 가운데의 가로 위치 (-1 왼쪽 .. 1 오른쪽)
        self.zone_y = 0.0

    def color_mask(self, frame, lo, hi):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask[: int(self.cfg.roi_top * frame.shape[0])] = 0
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)   # 오프닝 = 작은 잡음 제거

    def blue_mask(self, frame):
        """파란 표지판 마스크. 햇빛에 하얗게 뜬 부분(blue_glare_*)도 넣는다."""
        cfg = self.cfg
        return self.color_mask(frame, cfg.blue_hsv_lo, cfg.blue_hsv_hi) | \
            self.color_mask(frame, cfg.blue_glare_lo, cfg.blue_glare_hi)

    def local_bright_mask(self, frame):
        """주변보다 밝은 가는 띠 (그늘 속 흰 테이프). 원본 - 오프닝(가는 밝은 것을 지운 배경) 이 크면 띠."""
        cfg = self.cfg
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v = hsv[..., 2]
        k = int(cfg.lane_local_k * w) | 1
        # 배경은 흐리게 한 그림에서 구한다 (카펫 무늬의 반짝이는 점 때문). 비교는 원본 밝기로 해야 선이 번지지 않는다
        back = cv2.morphologyEx(cv2.blur(v, (5, 5)), cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        mask = ((cv2.subtract(v, back) >= cfg.lane_local_margin) & (v >= cfg.lane_v_min) &
                (hsv[..., 1] <= cfg.lane_hsv_hi[1])).astype(np.uint8) * 255
        mask[: int(cfg.roi_top * h)] = 0
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)

    def route_mask(self, frame):
        """주차 통로 색 마스크 (route_color 가 없으면 None)."""
        cfg = self.cfg
        if cfg.route_color == 'red':
            mask = self.color_mask(frame, cfg.red_hsv_lo, cfg.red_hsv_hi) | \
                   self.color_mask(frame, cfg.red2_hsv_lo, cfg.red2_hsv_hi)
        elif cfg.route_color == 'blue':
            mask = self.color_mask(frame, cfg.blue_hsv_lo, cfg.blue_hsv_hi)
        else:
            return None
        # 선처럼 세로로 긴 덩어리만 남긴다
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        h = frame.shape[0]
        keep = np.flatnonzero((stats[:, cv2.CC_STAT_HEIGHT] >= cfg.route_min_height * h) &
                              (stats[:, cv2.CC_STAT_TOP] <= cfg.route_max_top * h))
        keep = keep[keep > 0]
        return np.where(np.isin(labels, keep), 255, 0).astype(np.uint8)

    def masks(self, frame):
        """처리 크기 frame -> {'left','right','crosswalk'} 마스크와 횡단보도 판정, 통로 상태."""
        cfg = self.cfg
        h, w = frame.shape[:2]
        lane_lo = list(cfg.lane_hsv_lo)
        if cfg.lane_auto_v:
            floor_v = float(np.median(cv2.cvtColor(frame[int(0.6 * h):], cv2.COLOR_BGR2HSV)[..., 2]))
            lane_lo[2] = int(min(lane_lo[2], max(cfg.lane_v_min, floor_v + cfg.lane_v_margin)))
        self.lane_v = lane_lo[2]
        lane = self.color_mask(frame, lane_lo, cfg.lane_hsv_hi)
        if cfg.lane_local_margin > 0:
            lane |= self.local_bright_mask(frame)
        lane = remove_wall_base(lane, cfg)
        route = self.route_mask(frame)
        self.route_seen = self.route_near = False
        self.route_end_y = 0.0
        if route is not None and cv2.countNonZero(route) >= cfg.route_min_area * w * h:
            self.route_seen = True
            self.route_near = np.flatnonzero(route.any(axis=1)).max() >= cfg.route_only_row * (h - 1)
            bar, self.route_end_y = route_end_bar(route, cfg, self.memory.center_near)
            if bar is not None:
                # 끝 선은 따라갈 선이 아니다. 빼 두어야 양쪽 선이 한 덩어리로 붙지 않는다
                route = route & ~cv2.dilate(bar, self.kernel)
        if self.route_near:
            # 통로 안: 통로 색만 따라간다 (옆의 흰 벽·흰 차선을 무시). 통로에는 횡단보도가 없다.
            left, right, _, _ = split_lane_mask(route, cfg, self.memory, separate_crosswalk=False)
            return {'left': left, 'right': right, 'crosswalk': np.zeros_like(lane)}, False
        if self.route_seen:
            lane = lane | route       # 흰 선이 통로 색 선으로 이어지는 구간
        if cfg.crosswalk_hsv_lo is not None and cfg.crosswalk_hsv_hi is not None:
            crosswalk = self.color_mask(frame, cfg.crosswalk_hsv_lo, cfg.crosswalk_hsv_hi)
            lane = cv2.bitwise_and(lane, cv2.bitwise_not(crosswalk))
            left, right, _, _ = split_lane_mask(lane, cfg, self.memory, separate_crosswalk=False)
            found = None      # 면적으로 판정
        else:
            left, right, crosswalk, boxes = split_lane_mask(lane, cfg, self.memory, separate_crosswalk=True)
            found = stripes_are_crosswalk(boxes, cfg, frame.shape[0])
        return {'left': left, 'right': right, 'crosswalk': crosswalk}, found

    def role_marks(self, frame):
        """lane_role 맵의 색 표시: 파란 표지판, 초록 칸 끝 선."""
        cfg = self.cfg
        h, w = frame.shape[:2]
        self.signs, self.zone_y = [], 0.0
        if not cfg.lane_role:
            return
        self.signs = blue_signs(self.blue_mask(frame), cfg)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, np.array(cfg.green_hsv_lo, np.uint8), np.array(cfg.green_hsv_hi, np.uint8))
        green[: int(cfg.zone_roi_top * h)] = 0
        green = cv2.morphologyEx(green, cv2.MORPH_OPEN, self.kernel)
        if cv2.countNonZero(green) >= cfg.route_min_area * w * h:
            self.zone_y = float(np.flatnonzero(green.any(axis=1)).max() / (h - 1))
            self.zone_x = float((np.nonzero(green)[1].mean() - w / 2.0) / (w / 2.0))

    def detect(self, frame):
        t0 = time.perf_counter()
        frame = resize_to(frame, self.cfg.proc_width)
        masks, found = self.masks(frame)
        self.role_marks(frame)
        p, masks = self.perceive(frame, masks, found)
        p.ms = (time.perf_counter() - t0) * 1000
        return p, masks, frame

    def perceive(self, frame, masks, found):
        """차선 마스크 + 역할 표시(파란 표지판/초록 선) + 제어기의 지시(prefer/follow_zone) -> Perception."""
        if self.follow_zone and self.zone_y > 0:
            # 칸 안: 칸 끝은 흰 선이 ㄷ자로 막혀 있어 좌우 선 구분이 틀어진다 (2026-10-04: 초록을 보고도 왼쪽으로 빠져나감).
            # 초록 선 가운데를 향해 간다
            p = Perception(size=(frame.shape[1], frame.shape[0]), ok=True, offset=float(np.clip(self.zone_x, -1.5, 1.5)))
            p.target = (int((self.zone_x + 1) * frame.shape[1] / 2), int(self.zone_y * (frame.shape[0] - 1)))
        elif self.prefer:
            # 갈림길: 한쪽 선만 보고 (기억해 둔 차선 폭의 절반만큼 떨어져) 따라간다
            keep = masks[self.prefer]
            masks = {'left': keep if self.prefer == 'left' else None,
                     'right': keep if self.prefer == 'right' else None, 'crosswalk': None}
            p = lane_from_masks(masks['left'], masks['right'], None, self.cfg, self.memory, False)
        else:
            p = lane_from_masks(masks['left'], masks['right'], masks['crosswalk'], self.cfg, self.memory, found)
        p.signs = list(self.signs)
        p.zone_seen, p.zone_y = self.zone_y > 0, self.zone_y
        p.route_seen, p.route_near = self.route_seen, bool(self.route_near)
        p.route_end, p.route_end_y = self.route_end_y > 0, self.route_end_y
        return p, masks


class YoloDetector:
    def __init__(self, cfg):
        from ultralytics import YOLO   # 여기서만 필요 (hsv 만 쓸 때는 설치 안 해도 된다)
        self.cfg = cfg
        self.memory = LaneMemory()
        self.model = YOLO(cfg.weights)
        self.names = self.model.names            # {0: 'left', ...}
        self.model.predict(np.zeros((cfg.imgsz, cfg.imgsz, 3), np.uint8), imgsz=cfg.imgsz, verbose=False)  # 워밍업
        self.has_signs = any(name in SIGNS for name in self.names.values())
        self.marks = HsvDetector(cfg)            # 역할 표시와 갈림길 판단은 색 검출기와 같은 코드를 쓴다
        self.marks.memory = self.memory
        self.prefer, self.follow_zone = '', False   # 제어기가 정한다 (HsvDetector 와 같다)

    def detect(self, frame):
        t0 = time.perf_counter()
        cfg = self.cfg
        frame = resize_to(frame, cfg.proc_width)
        h, w = frame.shape[:2]
        r = self.model.predict(frame, conf=cfg.conf, imgsz=cfg.imgsz, verbose=False)[0]
        masks = {name: np.zeros((h, w), np.uint8) for name in ('left', 'right', 'crosswalk', 'lane', 'robot')}
        obstacle_y = 0.0
        signs = []
        if r.boxes is not None and len(r.boxes):
            classes = r.boxes.cls.cpu().numpy().astype(int)
            polys = r.masks.xy if r.masks is not None else [None] * len(classes)
            boxes = r.boxes.xyxy.cpu().numpy()
            for poly, cls_id, box in zip(polys, classes, boxes):
                name = self.names.get(int(cls_id), str(cls_id))
                if name == 'robot':
                    obstacle_y = max(obstacle_y, float(box[3] / (h - 1)))
                if name in SIGNS:
                    signs.append((name, float(((box[0] + box[2]) / 2 - w / 2.0) / (w / 2.0)),
                                  float(box[1] / (h - 1)), float(box[3] / (h - 1))))
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
        robot = masks.pop('robot')
        del masks['lane']
        m = self.marks
        m.prefer, m.follow_zone = self.prefer, self.follow_zone
        m.role_marks(frame)                      # 초록 선 (+ 파란 표지판을 색으로)
        if cfg.lane_role and self.has_signs:
            m.signs = sorted(signs, key=lambda s: -s[3])   # 학습한 표지판을 쓴다 (색 대신)
        p, masks = m.perceive(frame, masks, None)
        masks = dict(masks, robot=robot)
        p.obstacle_y = obstacle_y
        p.ms = (time.perf_counter() - t0) * 1000
        return p, masks, frame


def make_detector(cfg):
    if cfg.backend == 'yolo':
        return YoloDetector(cfg)
    if cfg.backend == 'hsv':
        return HsvDetector(cfg)
    raise ValueError(f"backend 는 'hsv' 또는 'yolo' 입니다: {cfg.backend}")
