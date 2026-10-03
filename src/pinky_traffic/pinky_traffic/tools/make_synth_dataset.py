"""시뮬레이터로 합성 학습 데이터 만들기 (정답 라벨 자동).

로봇/트랙이 없어도 학습 파이프라인 전체를 돌려 볼 수 있다. 실제 사진과 섞어 학습해도 된다.

  python3 -m pinky_traffic.tools.make_synth_dataset --out data/synth_ds --n 400
"""
import argparse

import cv2
import numpy as np

from ..core.perception import resize_to
from ..sim.track import Track, TrackSpec, Camera, CameraSpec, L_LEFT, L_RIGHT, L_CROSSWALK
from .dataset import DatasetWriter, masks_to_label


def random_spec(rng):
    spec = TrackSpec()
    spec.lane_width = float(rng.uniform(0.19, 0.26))
    spec.tape_width = float(rng.choice([0.018, 0.024, 0.036, 0.048]))
    spec.stripe_width = spec.tape_width
    spec.stripes = int(rng.integers(3, 6))
    floor = int(rng.integers(55, 150))
    spec.floor_bgr = tuple(int(np.clip(floor + rng.integers(-12, 13), 0, 255)) for _ in range(3))
    spec.noise = float(rng.uniform(2, 12))
    palette = [(235, 235, 235), (40, 210, 235), (250, 250, 250), (60, 60, 230), (220, 160, 40)]   # 흰/노랑/흰/빨강/파랑
    spec.lane_bgr = palette[int(rng.integers(0, len(palette)))]
    spec.crosswalk_bgr = spec.lane_bgr if rng.random() < 0.5 else palette[int(rng.integers(0, len(palette)))]
    spec.crosswalk_s = [float(rng.uniform(0.5, 1.2)), float(rng.uniform(4.0, 4.8))]
    return spec


def augment(image, rng):
    out = image.astype(np.float32)
    out = out * rng.uniform(0.6, 1.3) + rng.uniform(-25, 25)             # 밝기/대비
    h, w = out.shape[:2]
    if rng.random() < 0.5:                                               # 한쪽이 어두운 조명
        ramp = np.linspace(rng.uniform(0.6, 1.0), rng.uniform(0.9, 1.2), w, dtype=np.float32)
        out *= ramp[None, :, None]
    out += rng.normal(0, rng.uniform(1, 6), out.shape)                   # 센서 잡음
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < 0.4:
        k = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (k, k), 0)                           # 움직임/초점 흐림
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='data/synth_ds')
    parser.add_argument('--n', type=int, default=400)
    parser.add_argument('--tracks', type=int, default=8, help='서로 다른 코스/색 조합 수')
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    writer = DatasetWriter(args.out, 0.2, args.seed)
    per_track = max(1, args.n // args.tracks)
    count = 0
    for t in range(args.tracks):
        track = Track(random_spec(rng), seed=args.seed + t)
        camera = Camera(track, CameraSpec(tilt_deg=float(rng.uniform(4, 14)), cam_z=float(rng.uniform(0.055, 0.08))))
        for _ in range(per_track):
            s = float(rng.uniform(0, track.total))
            if rng.random() < 0.35:                      # 횡단보도 앞 장면을 넉넉히
                s = float(rng.choice(track.spec.crosswalk_s) - rng.uniform(0.15, 0.6))
            x, y, yaw = track.pose_at(s, lateral=float(rng.uniform(-0.06, 0.06)), dtheta=float(rng.uniform(-0.35, 0.35)))
            image = augment(camera.render(x, y, yaw, wall_bgr=tuple(int(v) for v in rng.integers(60, 220, 3))), rng)
            labels = camera.render_labels(x, y, yaw)
            image = resize_to(image, args.width)
            labels = cv2.resize(labels, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            left, right = L_LEFT, L_RIGHT
            if rng.random() < 0.5:
                # 좌우를 뒤집으면 시계방향 코스(우회전)가 된다. 이때 left/right 라벨도 서로 바꾼다.
                image, labels = cv2.flip(image, 1), cv2.flip(labels, 1)
                left, right = L_RIGHT, L_LEFT
            masks = {'left': np.uint8(labels == left) * 255, 'right': np.uint8(labels == right) * 255,
                     'crosswalk': np.uint8(labels == L_CROSSWALK) * 255}
            writer.add(f'synth_{count:05d}', image, masks_to_label(masks))
            count += 1
    path = writer.finish()
    print(f'{count} 장 -> {writer.counts}\ndata.yaml: {path}')


if __name__ == '__main__':
    main()
