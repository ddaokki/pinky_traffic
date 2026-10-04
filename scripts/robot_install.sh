#!/bin/bash
# 로봇에 카메라 발행 노드(파일 하나)를 복사한다.
#   scripts/robot_install.sh 192.168.4.1
set -e
IP="${1:?로봇 IP}"
WS="$(cd "$(dirname "$0")/.." && pwd)"
scp "$WS/src/pinky_traffic/pinky_traffic/nodes/camera_pub.py" "pinky@$IP:~/camera_pub.py"
echo
echo "복사 완료. 로봇(ssh pinky@$IP)에서 터미널 2개:"
echo "  1) ros2 launch pinky_bringup bringup_robot.launch.xml"
echo "  2) python3 ~/camera_pub.py --ros-args -p width:=320 -p height:=240 -p fps:=15.0"
echo "     (fps 는 소수점으로. 영상이 뒤집혀 있으면 -p flip:=true 추가 — pinky1 은 필요)"
