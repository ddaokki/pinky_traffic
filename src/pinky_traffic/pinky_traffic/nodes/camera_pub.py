"""카메라 발행 노드 (로봇에서 실행). 수업 37번 자료 img_pub 의 Pinky 판.

bringup_robot.launch.xml 은 카메라를 발행하지 않는다. 그래서 로봇에서 이 노드를 따로 띄운다.
WiFi 로 보내므로 원본(Image) 대신 JPEG(CompressedImage) 로 보낸다
(320x240 원본 = 230KB/장, JPEG = 10~20KB/장).

  ros2 run pinky_traffic camera_pub --ros-args -p width:=320 -p height:=240 -p fps:=15

워크스페이스 없이 파일 하나만 복사해서도 실행된다:
  python3 camera_pub.py --ros-args -p fps:=15

backend: auto(기본) -> picamera2 -> pinkylib -> opencv 순서로 시도
"""
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage


class Source:
    """카메라 백엔드 3종을 read() -> BGR ndarray 로 통일."""

    def __init__(self, backend, width, height, device, log):
        self.kind = None
        errors = []
        order = ['picamera2', 'pinkylib', 'opencv'] if backend == 'auto' else [backend]
        for kind in order:
            try:
                getattr(self, '_open_' + kind)(width, height, device)
                self.kind = kind
                break
            except Exception as error:
                errors.append(f'{kind}: {error}')
        if self.kind is None:
            raise RuntimeError('카메라를 열 수 없습니다 -> ' + ' | '.join(errors))
        log(f'camera backend = {self.kind}' + (f' (실패: {errors})' if errors else ''))

    def _open_picamera2(self, width, height, device):
        from picamera2 import Picamera2
        self.cam = Picamera2()
        # RGB888 로 요청하면 배열은 BGR 순서로 온다 (OpenCV 와 같다)
        self.cam.configure(self.cam.create_video_configuration(main={'size': (width, height), 'format': 'RGB888'}))
        self.cam.start()

    def _open_pinkylib(self, width, height, device):
        from pinkylib import Camera
        self.cam = Camera()
        self.cam.start()

    def _open_opencv(self, width, height, device):
        self.cam = cv2.VideoCapture(device, cv2.CAP_V4L2)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cam.isOpened():
            raise RuntimeError(f'/dev/video{device} 열기 실패')

    def read(self):
        if self.kind == 'picamera2':
            return self.cam.capture_array()
        if self.kind == 'pinkylib':
            return self.cam.get_frame()
        ok, frame = self.cam.read()
        return frame if ok else None

    def close(self):
        try:
            if self.kind == 'picamera2':
                self.cam.stop()
            elif self.kind == 'pinkylib':
                self.cam.close()
            else:
                self.cam.release()
        except Exception:
            pass


class CameraPub(Node):
    def __init__(self):
        super().__init__('camera_pub')
        self.declare_parameter('width', 320)
        self.declare_parameter('height', 240)
        self.declare_parameter('fps', 15.0)
        self.declare_parameter('quality', 70)
        self.declare_parameter('backend', 'auto')
        self.declare_parameter('device', 0)
        self.declare_parameter('swap_rb', False)     # 색이 파랗게/빨갛게 뒤바뀌어 보이면 True
        self.declare_parameter('flip', False)        # 영상이 뒤집혀 있으면 True (180도)
        self.declare_parameter('topic', 'camera/image_raw/compressed')
        get = lambda name: self.get_parameter(name).value
        self.size = (int(get('width')), int(get('height')))
        self.quality = int(get('quality'))
        self.swap_rb, self.flip = bool(get('swap_rb')), bool(get('flip'))
        self.source = Source(get('backend'), self.size[0], self.size[1], int(get('device')), self.get_logger().info)
        self.pub = self.create_publisher(CompressedImage, get('topic'), qos_profile_sensor_data)
        self.create_timer(1.0 / float(get('fps')), self.tick)
        self.count = 0
        self.get_logger().info(f"발행: {get('topic')} {self.size[0]}x{self.size[1]} @ {get('fps')} fps")

    def tick(self):
        frame = self.source.read()
        if frame is None:
            return
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
        if self.swap_rb:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if self.flip:
            frame = cv2.flip(frame, -1)
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return
        msg = CompressedImage(format='jpeg', data=jpeg.tobytes())
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'front_camera_link'
        self.pub.publish(msg)
        self.count += 1
        if self.count == 1:
            self.get_logger().info(f'첫 프레임 발행 ({len(msg.data) / 1024:.0f} KB)')


def main(args=None):
    rclpy.init(args=args)
    node = CameraPub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.source.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
