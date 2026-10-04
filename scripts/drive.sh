#!/bin/bash
# PC 에서 로봇 한 대 몫의 주행 노드 실행.
#   scripts/drive.sh <로봇이름> <ROS_DOMAIN_ID> [hsv|yolo] [가중치]
#   예) scripts/drive.sh pinky1 24            # 색(HSV)으로 주행
#       scripts/drive.sh pinky1 24 yolo       # models/best.pt 로 주행
set -e
WS="$(cd "$(dirname "$0")/.." && pwd)"
ROBOT="${1:?로봇 이름 (예: pinky1)}"
DOMAIN="${2:?ROS_DOMAIN_ID (로봇과 같은 값)}"
BACKEND="${3:-hsv}"
WEIGHTS="${4:-$WS/models/best.pt}"
source /opt/ros/jazzy/setup.bash
# LED 서비스 정의(pinky_interfaces)는 핑키 워크스페이스에 있다
[ -f "$HOME/pinky/install/setup.bash" ] && source "$HOME/pinky/install/setup.bash"
source "$WS/install/setup.bash"
export ROS_DOMAIN_ID="$DOMAIN"
if [ "$BACKEND" = "yolo" ]; then
  # ultralytics 는 ~/venv/yolo 에 있다
  export PYTHONPATH="$HOME/venv/yolo/lib/python3.12/site-packages:$PYTHONPATH"
fi
cd "$WS"
exec ros2 run pinky_traffic lane_driver --ros-args -p robot:="$ROBOT" -p config:="$WS/src/pinky_traffic/config/field.yaml" \
  -p backend:="$BACKEND" -p weights:="$WEIGHTS"
