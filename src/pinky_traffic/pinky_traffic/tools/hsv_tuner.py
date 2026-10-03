"""테이프 색의 HSV 범위 찾기 (수업 42번 자료 inRange, 38번 자료 HSV).

  python3 -m pinky_traffic.tools.hsv_tuner --images data/raw          # 찍어 둔 사진으로
  python3 -m pinky_traffic.tools.hsv_tuner --source 0                 # 웹캠으로

슬라이더로 테이프만 하얗게 남게 맞춘다. 요령: H 는 좁게, S/V 는 넉넉히.
  n / p = 다음/이전 사진,  s = 현재 값을 yaml 형식으로 출력,  ESC = 종료
흰 테이프: S 를 낮게(0~70), V 를 높게(170~255).  색 테이프: H 범위를 그 색으로.
"""
import argparse
import glob
import os

import cv2
import numpy as np

NAMES = ['H lo', 'S lo', 'V lo', 'H hi', 'S hi', 'V hi']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--images', default=None)
    parser.add_argument('--source', default=None)
    parser.add_argument('--key', default='lane', choices=['lane', 'crosswalk'], help='출력할 yaml 키 이름')
    parser.add_argument('--init', type=int, nargs=6, default=[0, 0, 170, 179, 70, 255])
    args = parser.parse_args()
    files = sorted(glob.glob(os.path.join(args.images, '*.jpg')) + glob.glob(os.path.join(args.images, '*.png'))) if args.images else []
    cap = None if files else cv2.VideoCapture(int(args.source or 0))
    win = 'hsv tuner'
    cv2.namedWindow(win)
    for name, value in zip(NAMES, args.init):
        cv2.createTrackbar(name, win, value, 179 if name.startswith('H') else 255, lambda v: None)
    index = 0
    while True:
        if files:
            frame = cv2.imread(files[index % len(files)])
        else:
            ok, frame = cap.read()
            if not ok:
                break
        if frame.shape[1] > 480:
            frame = cv2.resize(frame, (480, int(frame.shape[0] * 480 / frame.shape[1])))
        values = [cv2.getTrackbarPos(name, win) for name in NAMES]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(values[:3], np.uint8), np.array(values[3:], np.uint8))
        cv2.imshow(win, np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]))
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break
        if key == ord('n'):
            index += 1
        if key == ord('p'):
            index -= 1
        if key == ord('s'):
            print(f'{args.key}_hsv_lo: {values[:3]}\n{args.key}_hsv_hi: {values[3:]}')
    values = [cv2.getTrackbarPos(name, win) for name in NAMES]
    print(f'# config yaml 에 붙여 넣으세요\n{args.key}_hsv_lo: {values[:3]}\n{args.key}_hsv_hi: {values[3:]}')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
