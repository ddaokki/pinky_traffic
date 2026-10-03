# source scripts/env.sh [ROS_DOMAIN_ID]
# 터미널마다 한 번. ROS2 + 이 워크스페이스 환경을 부르고, 숫자를 주면 도메인도 맞춘다.
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
[ -f "$WS/install/setup.bash" ] && source "$WS/install/setup.bash"
export PYTHONPATH="$WS/src/pinky_traffic:$PYTHONPATH"
[ -n "$1" ] && export ROS_DOMAIN_ID="$1"
cd "$WS"
echo "pinky_traffic 환경 준비됨 (ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0})"
