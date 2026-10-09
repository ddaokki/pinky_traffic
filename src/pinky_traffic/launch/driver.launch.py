"""PC 에서 한 로봇 몫의 lane_driver 실행.

  ros2 launch pinky_traffic driver.launch.py robot:=pinky1 config:=/abs/path/field.yaml
  ros2 launch pinky_traffic driver.launch.py robot:=pinky1 weights:=/abs/path/best.pt

로봇이 2대면 터미널 2개에서 ROS_DOMAIN_ID 를 각각 맞추고 robot 이름만 바꿔 실행한다.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    default_config = os.path.join(get_package_share_directory('pinky_traffic'), 'config', 'field.yaml')
    names = ['robot', 'config', 'weights', 'autostart', 'use_scan', 'use_dashboard', 'dashboard_url']
    defaults = ['pinky1', default_config, '', 'false', 'true', 'true', '']
    return LaunchDescription(
        [DeclareLaunchArgument(n, default_value=d) for n, d in zip(names, defaults)] + [
            Node(package='pinky_traffic', executable='lane_driver', output='screen',
                 parameters=[{n: ParameterValue(LaunchConfiguration(n), value_type=bool if d in ('true', 'false') else str)
                              for n, d in zip(names, defaults)}]),
        ])
