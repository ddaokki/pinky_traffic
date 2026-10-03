"""차선 주행 노드 (PC 에서 실행).

구독  camera/image_raw/compressed (sensor_msgs/CompressedImage)  또는  camera/image_raw (Image)
      scan (sensor_msgs/LaserScan, 있으면 전방 장애물 정지)
발행  cmd_vel (geometry_msgs/Twist)
      lane/debug/compressed (rqt image view 로 확인용)
      lane/state (std_msgs/String, JSON)

  ros2 run pinky_traffic lane_driver --ros-args -p robot:=pinky1 -p config:=<yaml> -p backend:=yolo -p weights:=best.pt

안전장치: 영상이 image_timeout 초 넘게 안 오면 0 속도를 낸다. 종료할 때도 0 속도를 낸다.
"""
import json
import math
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CompressedImage, Image, LaserScan
from std_msgs.msg import String

from ..core.config import Config
from ..core.driver import Driver


def front_range(scan: LaserScan, half_angle_deg, yaw_offset_deg=0.0):
    """라이다에서 정면 ±half_angle 안의 최소 거리. 유효값이 없으면 None."""
    ranges = np.asarray(scan.ranges, dtype=np.float32)
    if ranges.size == 0:
        return None
    angles = scan.angle_min + np.arange(ranges.size) * scan.angle_increment - math.radians(yaw_offset_deg)
    angles = np.arctan2(np.sin(angles), np.cos(angles))
    valid = np.isfinite(ranges) & (ranges > max(scan.range_min, 0.03)) & (np.abs(angles) <= math.radians(half_angle_deg))
    return float(ranges[valid].min()) if valid.any() else None


class LaneDriverNode(Node):
    def __init__(self):
        super().__init__('lane_driver')
        self.declare_parameter('robot', 'pinky1')
        self.declare_parameter('config', '')
        self.declare_parameter('backend', '')
        self.declare_parameter('weights', '')
        self.declare_parameter('image_topic', 'camera/image_raw/compressed')
        self.declare_parameter('compressed', True)
        self.declare_parameter('use_scan', True)
        self.declare_parameter('use_dashboard', True)
        self.declare_parameter('dashboard_url', '')
        self.declare_parameter('autostart', False)
        self.declare_parameter('image_timeout', 0.7)
        get = lambda name: self.get_parameter(name).value

        overrides = {k: get(k) for k in ('backend', 'weights', 'dashboard_url') if get(k)}
        self.cfg = Config.load(get('config') or None, **overrides)
        self.driver = Driver(self.cfg, get('robot'), get('use_dashboard'), get('autostart'),
                             log=lambda text: self.get_logger().info(text))
        self.bridge = CvBridge()
        self.front = None
        self.t_scan = 0.0
        self.t_image = 0.0
        self.image_timeout = float(get('image_timeout'))

        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.debug_pub = self.create_publisher(CompressedImage, 'lane/debug/compressed', 2)
        self.state_pub = self.create_publisher(String, 'lane/state', 10)
        if get('compressed'):
            self.create_subscription(CompressedImage, get('image_topic'), self.on_compressed, qos_profile_sensor_data)
        else:
            self.create_subscription(Image, get('image_topic'), self.on_image, qos_profile_sensor_data)
        if get('use_scan'):
            self.create_subscription(LaserScan, 'scan', self.on_scan, qos_profile_sensor_data)
        self.create_timer(0.1, self.watchdog)
        self.get_logger().info(f"lane_driver: robot={get('robot')} backend={self.cfg.backend} "
                               f"image={get('image_topic')} dashboard={self.cfg.dashboard_url if get('use_dashboard') else 'off'}")

    def on_scan(self, msg):
        self.front = front_range(msg, self.cfg.front_angle_deg, self.cfg.lidar_yaw_offset_deg)
        self.t_scan = time.time()

    def on_compressed(self, msg):
        frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            self.on_frame(frame)

    def on_image(self, msg):
        self.on_frame(self.bridge.imgmsg_to_cv2(msg, 'bgr8'))

    def on_frame(self, frame):
        now = time.time()
        self.t_image = now
        front = self.front if now - self.t_scan < 1.0 else None     # 오래된 라이다 값은 쓰지 않는다
        cmd = self.driver.process(frame, front, now)
        self.publish(cmd.v, cmd.w)
        self.state_pub.publish(String(data=json.dumps(self.driver.state)))
        if self.debug_pub.get_subscription_count():
            ok, jpeg = cv2.imencode('.jpg', self.driver.debug)
            if ok:
                out = CompressedImage(format='jpeg', data=jpeg.tobytes())
                out.header.stamp = self.get_clock().now().to_msg()
                self.debug_pub.publish(out)

    def publish(self, v, w):
        twist = Twist()
        twist.linear.x, twist.angular.z = float(v), float(w)
        self.cmd_pub.publish(twist)

    def watchdog(self):
        if time.time() - self.t_image > self.image_timeout:
            self.publish(0.0, 0.0)
            self.driver.idle_report('no image')

    def shutdown(self):
        for _ in range(3):
            self.publish(0.0, 0.0)
            time.sleep(0.05)
        self.driver.close()


def main(args=None):
    # Ctrl+C 때 rclpy 가 먼저 꺼지면 마지막 0 속도를 못 보낸다 -> 신호 처리는 파이썬(KeyboardInterrupt)에 맡긴다
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = LaneDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
