"""인식기. detect(frame_bgr) -> (Perception, masks dict, frame).

HybridDetector : 주행용. 차선·횡단보도·초록 선은 색(HSV), 파란 표지판·상대 로봇은 YOLO-seg (best.pt).
HsvDetector    : 그 색 부분. 시뮬레이터·테스트·자동 라벨링에서 혼자 쓴다.
"""
import math
import time

import cv2
import numpy as np

from .perception import (LaneMemory, Perception, lane_from_masks, split_lane_mask, stripes_are_crosswalk, resize_to,
                         remove_wall_base, route_end_bar, SIGNS)


def shaft_angle(ys, xs, h, w, horizon_row):
    """파란 덩어리의 긴 축이 '로봇 정면 방향' 에서 몇 도 틀어져 보이는가 (화면 각도, + = 축이 오른쪽으로 기울었다 = 오른쪽으로 돌아야 한다).
    바닥에서 로봇과 나란한 선은 화면에서 소실점(가운데, horizon_row)을 향한다. 그 방향과 축의 차이."""
    if len(xs) < 20:
        return None
    vx, vy, x0, y0 = cv2.fitLine(np.column_stack([xs, ys]).astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    if vy > 0:
        vx, vy = -vx, -vy
    seen = math.degrees(math.atan2(vx, -vy))
    ideal = math.degrees(math.atan2(w / 2.0 - x0, max(1.0, y0 - horizon_row * h)))
    err = seen - ideal
    return float((err + 90.0) % 180.0 - 90.0)       # 선은 방향이 없다: -90..90


def blue_angles(blue, cfg):
    """파란 덩어리마다 [(x -1..1, 축 각도 오차, 먼 끝 행, 가까운 끝 행)] (shaft_angle)."""
    h, w = blue.shape[:2]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(blue, connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < cfg.sign_min_area * w * h:
            continue
        ys, xs = np.nonzero(labels == i)
        a = shaft_angle(ys, xs, h, w, cfg.sign_horizon_row)
        if a is not None:
            out.append((float((xs.mean() - w / 2.0) / (w / 2.0)), a, float(ys.min() / (h - 1)), float(ys.max() / (h - 1))))
    return out


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
        self.angles = []                 # 파란 덩어리 축 각도 [(x, 오차)] (표지판 정렬)
        self.objects = []                # 인식한 물체 [(이름, 신뢰도|None, 마스크)] (발표용 그림)
        self.follow_zone = False         # 제어기가 정한다: 칸 안에서는 초록 선 가운데를 보고 간다
        self.zone_x = 0.0                # 초록 선 가운데의 가로 위치 (-1 왼쪽 .. 1 오른쪽)
        self.zone_y = 0.0

    def color_mask(self, frame, lo, hi):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask[: int(self.cfg.roi_top * frame.shape[0])] = 0
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)   # 오프닝 = 작은 잡음 제거

    def blue_mask(self, frame):
        """파란 표지판 마스크. 햇빛에 하얗게 뜬 부분(blue_glare_*)은 진한 파랑에 붙어 있을 때만 넣는다.
        (2026-10-09: 이 조명에서는 흰 차선도 살짝 푸르게 떠 glare 범위에 들어갔다 -> 흰 선을 표지판으로 보고 차선을 벗어남.
         진짜 표지판은 햇빛을 받아도 진한 파랑이 조금은 남는다)"""
        cfg = self.cfg
        strong = self.color_mask(frame, cfg.blue_hsv_lo, cfg.blue_hsv_hi)
        glare = self.color_mask(frame, cfg.blue_glare_lo, cfg.blue_glare_hi)
        both = strong | glare
        if not glare.any() or not strong.any():
            return strong
        n, labels = cv2.connectedComponents(both, connectivity=8)
        seeded = np.unique(labels[(strong > 0)])
        seeded = seeded[seeded > 0]
        return np.where(np.isin(labels, seeded), 255, 0).astype(np.uint8)

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
        self.signs, self.zone_y, self.objects, self.angles = [], 0.0, [], []
        if not cfg.lane_role:
            return
        blue = self.blue_mask(frame)
        self.signs = blue_signs(blue, cfg)
        self.angles = blue_angles(blue, cfg)
        if self.signs:
            n, labels, stats, _ = cv2.connectedComponentsWithStats(blue, connectivity=8)
            self.objects = [('sign', None, np.uint8(labels == i) * 255) for i in range(1, n)
                            if stats[i, cv2.CC_STAT_AREA] >= cfg.sign_min_area * w * h]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, np.array(cfg.green_hsv_lo, np.uint8), np.array(cfg.green_hsv_hi, np.uint8))
        green[: int(cfg.zone_roi_top * h)] = 0
        green = cv2.morphologyEx(green, cv2.MORPH_OPEN, self.kernel)
        n, labels, st, _ = cv2.connectedComponentsWithStats(green, connectivity=8)
        bars = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= cfg.route_min_area * w * h
                and st[i, cv2.CC_STAT_WIDTH] >= cfg.zone_min_width * w
                and st[i, cv2.CC_STAT_HEIGHT] <= cfg.zone_max_aspect * st[i, cv2.CC_STAT_WIDTH]]
        if bars:
            bar = labels == max(bars, key=lambda i: st[i, cv2.CC_STAT_AREA])
            self.zone_y = float(np.flatnonzero(bar.any(axis=1)).max() / (h - 1))
            self.zone_x = float((np.nonzero(bar)[1].mean() - w / 2.0) / (w / 2.0))

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
        p.sign_angles = list(self.angles)
        p.zone_seen, p.zone_y = self.zone_y > 0, self.zone_y
        p.route_seen, p.route_near = self.route_seen, bool(self.route_near)
        p.route_end, p.route_end_y = self.route_end_y > 0, self.route_end_y
        return p, masks


def load_yolo(cfg):
    """YOLO 모델을 읽는다. 로봇 2대가 PC 하나에서 같이 돌므로 torch 스레드 수를 묶는다
    (2026-10-09: 묶지 않으면 한 프로세스가 CPU 350% 를 써서 두 대 모두 영상을 못 따라갔다)."""
    import torch
    torch.set_num_threads(max(1, int(cfg.yolo_threads)))
    from ultralytics import YOLO   # 여기서만 필요 (시뮬레이터·테스트는 설치 안 해도 된다)
    model = YOLO(cfg.weights)
    model.predict(np.zeros((cfg.imgsz, cfg.imgsz, 3), np.uint8), imgsz=cfg.imgsz, verbose=False)  # 워밍업
    torch.set_num_threads(max(1, int(cfg.yolo_threads)))   # ultralytics 가 불러오면서 다시 늘려 놓는다
    return model


def yolo_objects(r, names, h, w, cfg):
    """YOLO 결과 -> (표지판 목록, 발표용 물체 목록, robot 아래 끝 행, 클래스별 마스크, [(표지판, 물체)] 짝)."""
    masks = {name: np.zeros((h, w), np.uint8) for name in ('left', 'right', 'crosswalk', 'lane', 'robot')}
    obstacle_y = 0.0
    signs, objects, pairs = [], [], []
    if r.boxes is not None and len(r.boxes):
        classes = r.boxes.cls.cpu().numpy().astype(int)
        polys = r.masks.xy if r.masks is not None else [None] * len(classes)
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, 'conf') else [None] * len(classes)
        for poly, cls_id, box, conf in zip(polys, classes, boxes, confs):
            name = names.get(int(cls_id), str(cls_id))
            obj = None
            if (name in SIGNS or name == 'robot') and poly is not None and len(poly) >= 3:
                mask = np.zeros((h, w), np.uint8)
                cv2.fillPoly(mask, [np.asarray(poly, np.int32)], 255)
                obj = (name, None if conf is None else float(conf), mask)
                objects.append(obj)                      # 화면에는 학습한 이름 그대로
            if name == 'robot' and (conf is None or conf >= cfg.robot_conf):
                obstacle_y = max(obstacle_y, float(box[3] / (h - 1)))
            if name in SIGNS:
                sign = (name if cfg.sign_use_kind else 'blue', float(((box[0] + box[2]) / 2 - w / 2.0) / (w / 2.0)),
                        float(box[1] / (h - 1)), float(box[3] / (h - 1)))
                signs.append(sign)
                pairs.append((sign, obj, None if conf is None else float(conf)))
            if name in masks and poly is not None and len(poly) >= 3:
                cv2.fillPoly(masks[name], [np.asarray(poly, np.int32)], 255)
    return sorted(signs, key=lambda s: -s[3]), objects, obstacle_y, masks, pairs


class HybridDetector(HsvDetector):
    """주행 인식기: 차선·초록 선은 색(HSV), 파란 표지판·상대 로봇은 YOLO.
    YOLO 는 yolo_every 프레임마다 한 번 돌리고 그 사이는 직전 결과를 쓴다 (CPU 를 아낀다)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = load_yolo(cfg)
        self.names = self.model.names
        self.has_signs = any(name in SIGNS for name in self.names.values())
        self.n = 0
        self.last = ([], [], 0.0)

    def detect(self, frame):
        t0 = time.perf_counter()
        cfg = self.cfg
        frame = resize_to(frame, cfg.proc_width)
        h, w = frame.shape[:2]
        if self.n % max(1, int(cfg.yolo_every)) == 0:
            r = self.model.predict(frame, conf=cfg.conf, imgsz=cfg.imgsz, verbose=False)[0]
            _, objects, obstacle_y, _, pairs = yolo_objects(r, self.names, h, w, cfg)
            blue = self.blue_mask(frame) > 0
            keep = []
            for sign, obj, conf in pairs:                # 실제 파랑이 들어 있는 표지판만 (흰 선 오인 거르기)
                if obj is None or (conf is not None and conf < cfg.sign_conf):
                    continue
                area = obj[2] > 0
                if float((blue & area).sum()) / max(1, int(area.sum())) >= cfg.sign_blue_frac:
                    keep.append((sign, obj))
            objects = [o for o in objects if o[0] not in SIGNS] + [o for _, o in keep]
            self.last = (sorted([s for s, _ in keep], key=lambda s: -s[3]), objects, obstacle_y)
        self.n += 1
        masks, found = self.masks(frame)
        self.role_marks(frame)                   # 초록 선 (+ 색으로 찾은 표지판)
        signs, objects, obstacle_y = self.last
        if cfg.lane_role and self.has_signs:
            # YOLO 가 못 잡으면 색으로 찾은 파란 표지판을 쓴다 (2026-10-09 pinky2: 직우 막대 위에 올라서니 YOLO 가 거의 못 잡아
            # 표지판을 앞에 두고 15초 멈춤). 색은 진한 파랑에 붙은 부분만 파랑으로 본다 (blue_mask) -> 흰 선 오인 없음
            hsv_signs = self.signs
            self.signs = list(signs) if signs else hsv_signs
            self.objects = list(objects) if signs else self.objects + [o for o in objects if o[0] == 'robot']
        else:
            self.objects = self.objects + [o for o in objects if o[0] == 'robot']
        p, masks = self.perceive(frame, masks, found)
        p.obstacle_y = obstacle_y
        p.ms = (time.perf_counter() - t0) * 1000
        return p, masks, frame


def make_detector(cfg, model=True):
    """주행 인식기: 차선은 색, 표지판·로봇은 YOLO (HybridDetector). 가중치가 없으면 켜지지 않는다.
    model=False 는 로봇 없이 돌리는 시뮬레이터·테스트용 (YOLO 없이 색만: 파란 표지판도 색으로 찾는다)."""
    return HybridDetector(cfg) if model else HsvDetector(cfg)
