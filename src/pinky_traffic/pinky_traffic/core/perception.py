"""인식 공통부: 클래스별 마스크 -> 차선 중심/횡단보도 정보.

HSV 든 YOLO 든 결국 (left, right, crosswalk) 세 장의 마스크를 만들고,
여기 있는 lane_from_masks() 가 조향에 쓸 숫자로 바꾼다.

좌표: 영상 좌상단 (0,0), 오른쪽 +x, 아래 +y.  offset > 0 = 차선 중심이 화면 중앙보다 오른쪽.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

CLASSES = ['left', 'right', 'crosswalk']   # YOLO 클래스 순서 (data.yaml 의 names 와 같아야 한다)


@dataclass
class Perception:
    ok: bool = False                 # 차선 중심을 구했는가
    offset: float = 0.0              # -1..1  (lookahead 행에서 차선중심 - 화면중앙) / (폭/2)
    heading: float = 0.0             # (먼 행 중심 - 가까운 행 중심) / (폭/2)  ≈ 곡률/방향
    left_seen: bool = False
    right_seen: bool = False
    crosswalk: bool = False
    crosswalk_y: float = 0.0         # 횡단보도 아래 끝 행 / 높이 (클수록 가깝다)
    obstacle_y: float = 0.0          # yolo 'robot' 박스 아래 끝 / 높이 (0 = 없음)
    target: Optional[Tuple[int, int]] = None
    centers: List[Tuple[int, int]] = field(default_factory=list)
    left_pts: List[Tuple[int, int]] = field(default_factory=list)
    right_pts: List[Tuple[int, int]] = field(default_factory=list)
    size: Tuple[int, int] = (0, 0)   # (w, h)
    ms: float = 0.0                  # 처리 시간
    route_seen: bool = False         # 주차 통로 색(빨강/파랑)이 보인다
    route_near: bool = False         # 통로 색이 로봇 바로 앞까지 왔다 (= 통로에 들어섰다)


class LaneMemory:
    """프레임 사이에 기억할 것: 행별 차선폭(px), 직전 가까운 행 중심."""

    def __init__(self):
        self.width = {}        # row index -> px
        self.center_near = None

    def reset(self):
        self.width.clear()
        self.center_near = None


def resize_to(frame, proc_width):
    h, w = frame.shape[:2]
    if w == proc_width:
        return frame
    return cv2.resize(frame, (proc_width, int(round(h * proc_width / w))), interpolation=cv2.INTER_AREA)


def _row_x(mask, y, band, pick):
    """y±band 행에서 마스크가 켜진 x 중 pick('max'|'min') 값. 없으면 None."""
    h = mask.shape[0]
    y0, y1 = max(0, y - band), min(h, y + band + 1)
    cols = np.flatnonzero(mask[y0:y1].any(axis=0))
    if cols.size == 0:
        return None
    return int(cols.max() if pick == 'max' else cols.min())


def lane_from_masks(left, right, crosswalk, cfg, memory: LaneMemory, crosswalk_found=None):
    """left/right/crosswalk: uint8 마스크(0/255, 같은 크기). None 허용."""
    ref = left if left is not None else right if right is not None else crosswalk
    h, w = ref.shape[:2]
    p = Perception(size=(w, h))
    half = w / 2.0

    rows = np.linspace(cfg.near_row, cfg.far_row, cfg.n_rows)
    centers = {}
    # 차선폭(px)은 원근 때문에 행에 대해 직선으로 변한다. 두 선이 같이 보였던 행들로 직선을 맞춰 두면
    # 한쪽 선만 보이는 행(가까운 행은 보통 그렇다)의 폭을 추정할 수 있다.
    fit = None
    if len(memory.width) >= 2:
        known = sorted(memory.width)
        fit = np.polyfit([rows[i] for i in known], [memory.width[i] for i in known], 1)
    for i, frac in enumerate(rows):
        y = int(frac * (h - 1))
        lx = _row_x(left, y, 2, 'max') if left is not None else None      # 왼쪽 선의 안쪽(오른쪽) 가장자리
        rx = _row_x(right, y, 2, 'min') if right is not None else None    # 오른쪽 선의 안쪽(왼쪽) 가장자리
        if lx is not None and rx is not None and rx - lx < 0.08 * w:
            # 두 선이 겹쳐 보이면(오검출) 이 행은 버린다
            continue
        t = i / max(1, cfg.n_rows - 1)
        default_w = ((1 - t) * cfg.lane_width_near + t * cfg.lane_width_far) * w
        if fit is not None and i not in memory.width:
            default_w = max(0.1 * w, float(np.polyval(fit, frac)))
        if lx is not None:
            p.left_pts.append((lx, y))
        if rx is not None:
            p.right_pts.append((rx, y))
        if lx is not None and rx is not None:
            width = rx - lx
            memory.width[i] = 0.7 * memory.width.get(i, width) + 0.3 * width
            centers[i] = ((lx + rx) / 2.0, y)
        elif lx is not None:
            centers[i] = (lx + memory.width.get(i, default_w) / 2.0, y)
        elif rx is not None:
            centers[i] = (rx - memory.width.get(i, default_w) / 2.0, y)

    p.left_seen = len(p.left_pts) >= 2
    p.right_seen = len(p.right_pts) >= 2

    if len(centers) >= 2:
        idx = sorted(centers)
        look_y = cfg.lookahead_row * (h - 1)
        best = min(idx, key=lambda i: abs(centers[i][1] - look_y))
        tx, ty = centers[best]
        near_x = centers[idx[0]][0]
        far_x = centers[idx[-1]][0]
        p.ok = True
        p.offset = float(np.clip((tx - half) / half, -1.5, 1.5))
        p.heading = float(np.clip((far_x - near_x) / half, -1.5, 1.5))
        p.target = (int(tx), int(ty))
        p.centers = [(int(centers[i][0]), int(centers[i][1])) for i in idx]
        memory.center_near = near_x

    if crosswalk is not None:
        area = int(cv2.countNonZero(crosswalk))
        found = crosswalk_found if crosswalk_found is not None else area >= cfg.crosswalk_min_area * w * h
        if found and area > 0:
            ys = np.flatnonzero(crosswalk.any(axis=1))
            p.crosswalk = True
            p.crosswalk_y = float(ys.max() / (h - 1))
    return p


def split_lane_mask(lane_mask, cfg, memory: LaneMemory, separate_crosswalk=False):
    """한 색 마스크 -> (left, right, stripes).

    덩어리(connected component)마다 '아래쪽 끝의 x' 를 보고, 직전 차선 중심보다
    왼쪽이면 left, 오른쪽이면 right. separate_crosswalk=True 면 키가 작은 덩어리는
    줄무늬(stripes)로 따로 모은다 (횡단보도를 차선과 같은 색으로 깔았을 때).
    """
    h, w = lane_mask.shape[:2]
    left = np.zeros_like(lane_mask)
    right = np.zeros_like(lane_mask)
    stripes = np.zeros_like(lane_mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(lane_mask, connectivity=8)
    ref = memory.center_near if memory.center_near is not None else w / 2.0
    min_area = cfg.min_area * w * h
    stripe_boxes = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        comp = labels == i
        if is_wall(comp[y:y + bh, x:x + bw], cfg, w, h):
            continue
        touches_side = x <= 1 or x + bw >= w - 1
        # 차선: 화면 옆 가장자리에 닿거나, 키가 크면서 먼 곳(ROI 위쪽)까지 이어진다.
        # 횡단보도 줄무늬: 가까이 오면 키는 커지지만 먼 곳까지 이어지지는 않는다.
        reaches_far = bh >= cfg.lane_min_height * h and y <= (cfg.roi_top + 0.10) * h
        if separate_crosswalk and not reaches_far and not touches_side:
            stripes[comp] = 255
            stripe_boxes.append((x, y, bw, bh))
            continue
        # 아래쪽 6행의 평균 x = 로봇 가까운 쪽 위치
        ys, xs = np.nonzero(comp[y + max(0, bh - 6): y + bh, x: x + bw])
        if _is_u_shape(xs + x, comp[y + max(0, bh - 6): y + bh], ref):
            # 양쪽 선이 앞에서 가로선으로 이어진 'U' (주차칸 끝): 가로선 행을 빼고 좌우로 나눈다
            part = comp.copy()
            part[comp[:, int(ref)]] = False
            cols = np.arange(w)[None, :]
            left[part & (cols < ref)] = 255
            right[part & (cols >= ref)] = 255
            continue
        xb = x + (xs.mean() if xs.size else bw / 2.0)
        (left if xb < ref else right)[comp] = 255
    return left, right, stripes, stripe_boxes


def is_wall(box_mask, cfg, w, h):
    """덩어리가 선이 아니라 넓은 면(흰 벽, 종이)인가: 영상 폭의 lane_wall_width 보다 넓은 행이 많다."""
    wide_rows = int(np.count_nonzero(box_mask.sum(axis=1) > cfg.lane_wall_width * w))
    return wide_rows >= cfg.lane_wall_rows * h


def _is_u_shape(xs, bottom_rows, ref):
    """아래쪽 행에서 기준선(ref) 양쪽에 따로 떨어진 픽셀이 있으면 두 선이 위에서 이어진 덩어리."""
    c = int(ref)
    if xs.size == 0 or not 0 <= c < bottom_rows.shape[1]:
        return False
    return xs.min() < ref - 2 and xs.max() > ref + 2 and not bottom_rows[:, max(0, c - 1):c + 2].any()


def stripes_are_crosswalk(stripe_boxes, cfg, h):
    """같은 색 모드: 비슷한 높이에 줄무늬가 여러 개 나란히 있으면 횡단보도."""
    if len(stripe_boxes) < cfg.crosswalk_min_stripes:
        return False
    mids = sorted(y + bh / 2.0 for _, y, _, bh in stripe_boxes)
    # 중앙값 주변 ±15% 높이 안에 들어오는 줄무늬 수
    med = mids[len(mids) // 2]
    close = sum(1 for m in mids if abs(m - med) <= 0.15 * h)
    return close >= cfg.crosswalk_min_stripes


def draw_debug(frame, p: Perception, masks=None, text=None):
    """대시보드/디버그용 그림. frame 은 처리 크기의 BGR."""
    out = frame.copy()
    h, w = out.shape[:2]
    if masks:
        overlay = out.copy()
        colors = {'left': (255, 120, 0), 'right': (0, 160, 255), 'crosswalk': (0, 255, 255), 'robot': (255, 0, 255)}
        for name, mask in masks.items():
            if mask is not None:
                overlay[mask > 0] = colors.get(name, (0, 255, 0))
        out = cv2.addWeighted(overlay, 0.45, out, 0.55, 0)
    for pt in p.left_pts:
        cv2.circle(out, pt, 2, (255, 120, 0), -1)
    for pt in p.right_pts:
        cv2.circle(out, pt, 2, (0, 160, 255), -1)
    for a, b in zip(p.centers, p.centers[1:]):
        cv2.line(out, a, b, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.line(out, (w // 2, h - 1), (w // 2, int(h * 0.5)), (120, 120, 120), 1)
    if p.target:
        cv2.circle(out, p.target, 5, (0, 0, 255), -1)
        cv2.line(out, (w // 2, h - 1), p.target, (0, 0, 255), 1, cv2.LINE_AA)
    if p.crosswalk:
        y = int(p.crosswalk_y * (h - 1))
        cv2.line(out, (0, y), (w - 1, y), (0, 255, 255), 1)
    if text:
        cv2.rectangle(out, (0, 0), (w, 14), (0, 0, 0), -1)
        cv2.putText(out, text, (3, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1, cv2.LINE_AA)
    return out
