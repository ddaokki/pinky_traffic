import glob
import os

from setuptools import find_packages, setup

package_name = 'pinky_traffic'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob.glob(os.path.join('config', '*.yaml'))),
        ('share/' + package_name + '/launch', glob.glob(os.path.join('launch', '*.launch.*'))),
    ],
    package_data={package_name: ['dashboard/*.html', 'dashboard/*.json']},
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jeongmin',
    maintainer_email='jeobgmin0518@gmail.com',
    description='Pinky Pro lane following + crosswalk stop (YOLO-seg), dashboard, simulator',
    license='MIT',
    entry_points={
        'console_scripts': [
            'lane_driver = pinky_traffic.nodes.lane_driver:main',
            'camera_pub = pinky_traffic.nodes.camera_pub:main',
            'dashboard = pinky_traffic.dashboard.server:main',
            'run_sim = pinky_traffic.tools.run_sim:main',
            'capture = pinky_traffic.tools.capture:main',
            'hsv_tuner = pinky_traffic.tools.hsv_tuner:main',
            'autolabel = pinky_traffic.tools.autolabel:main',
            'eval_detector = pinky_traffic.tools.eval_detector:main',
        ],
    },
)
