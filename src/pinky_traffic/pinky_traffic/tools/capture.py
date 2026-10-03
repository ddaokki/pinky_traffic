"""학습용 사진 모으기.

  # 로봇 카메라 토픽에서 (PC 에서 실행, 로봇은 camera_pub 실행 중)
  python3 -m pinky_traffic.tools.capture --out data/raw --every 0.5
  # 웹캠/동영상에서
  python3 -m pinky_traffic.tools.capture --source 0 --out data/raw

창이 뜨면:  SPACE = 한 장 저장,  a = 자동 저장 켜기/끄기(--every 초 간격),  ESC = 종료
teleop 으로 로봇을 천천히 몰면서 자동 저장을 켜 두면 된다.
다양하게: 직선/곡선/횡단보도 앞, 차선 가운데/치우침/비스듬히, 조명 바꿔서.
"""
import argparse
import os
import time

import cv2
import numpy as np


def frames_from_topic(topic):
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage
    rclpy.init()
    node = rclpy.create_node('capture')
    box = {}
    node.create_subscription(CompressedImage, topic, lambda m: box.update(msg=m), qos_profile_sensor_data)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            msg = box.pop('msg', None)
            yield None if msg is None else cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def frames_from_source(source):
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
    cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--topic', default='camera/image_raw/compressed')
    parser.add_argument('--source', default=None, help='웹캠 번호 또는 동영상 파일 (주면 토픽 대신 사용)')
    parser.add_argument('--out', default='data/raw')
    parser.add_argument('--every', type=float, default=0.5, help='자동 저장 간격(초)')
    parser.add_argument('--auto', action='store_true', help='시작부터 자동 저장')
    parser.add_argument('--prefix', default=time.strftime('%m%d_%H%M'))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    frames = frames_from_source(args.source) if args.source is not None else frames_from_topic(args.topic)
    auto, count, t_last, last = args.auto, 0, 0.0, None
    for frame in frames:
        if frame is not None:
            last = frame
        if last is None:
            continue
        save = False
        view = last.copy()
        cv2.putText(view, f'saved {count}  auto={"ON" if auto else "off"}', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 0) if auto else (0, 200, 255), 1, cv2.LINE_AA)
        cv2.imshow('capture (SPACE 저장 / a 자동 / ESC)', view)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord('a'):
            auto = not auto
        if key == ord(' '):
            save = True
        if auto and frame is not None and time.time() - t_last >= args.every:
            save = True
        if save:
            path = os.path.join(args.out, f'{args.prefix}_{count:04d}.jpg')
            cv2.imwrite(path, last)
            count += 1
            t_last = time.time()
    cv2.destroyAllWindows()
    print(f'{count} 장 저장 -> {args.out}')


if __name__ == '__main__':
    main()
