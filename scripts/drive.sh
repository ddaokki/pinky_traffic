#!/bin/bash
# PC 에서 로봇 한 대 몫의 주행 노드 실행.
#   scripts/drive.sh <로봇이름> <ROS_DOMAIN_ID> [가중치] [차선]
#   인식: 차선·횡단보도·초록 선은 색(HSV), 파란 표지판·상대 로봇은 YOLO (models/best.pt)
#   예) scripts/drive.sh pinky1 24            # 역할은 대시보드에서
#       scripts/drive.sh pinky1 24 "" 1       # 1차선에 놓은 로봇: 표지판에서 우 -> 우(유턴) -> 직진
#       scripts/drive.sh pinky2 23 "" 2       # 2차선에 놓은 로봇: 상대가 오면 초록 칸으로 비켰다가 우 -> 좌 -> 좌, 아니면 직진 -> 좌 -> 좌
#   LANE=1 scripts/drive.sh pinky1 24  처럼 줘도 된다. 로봇은 어느 쪽에 놓아도 되고, 놓은 차선 번호만 맞게 준다.
set -e
WS="$(cd "$(dirname "$0")/.." && pwd)"
ROBOT="${1:?로봇 이름 (예: pinky1)}"
DOMAIN="${2:?ROS_DOMAIN_ID (로봇과 같은 값)}"
WEIGHTS="${3:-$WS/models/best.pt}"
LANE="${4:-${LANE:-0}}"
[ -f "$WEIGHTS" ] || { echo "YOLO 가중치가 없습니다: $WEIGHTS (docs/TRAINING.md)"; exit 1; }
source /opt/ros/jazzy/setup.bash
# LED 서비스 정의(pinky_interfaces)는 핑키 워크스페이스에 있다
[ -f "$HOME/pinky/install/setup.bash" ] && source "$HOME/pinky/install/setup.bash"
source "$WS/install/setup.bash"
export ROS_DOMAIN_ID="$DOMAIN"
# 학원 와이파이처럼 로봇을 자동으로 못 찾는 망에서는 로봇 주소를 직접 알려 준다 (세미콜론으로 구분).
# 주소가 바뀌면: PINKY_PEERS="주소1;주소2" scripts/drive.sh ...   자동으로 찾는 망이면: PINKY_PEERS="" scripts/drive.sh ...
PINKY_PEERS="${PINKY_PEERS-192.168.129.199;192.168.129.200}"
[ -n "$PINKY_PEERS" ] && export ROS_STATIC_PEERS="$PINKY_PEERS"
# ultralytics 는 ~/venv/yolo 에 있다
export PYTHONPATH="$HOME/venv/yolo/lib/python3.12/site-packages:$PYTHONPATH"
cd "$WS"
exec ros2 run pinky_traffic lane_driver --ros-args -p robot:="$ROBOT" -p config:="$WS/src/pinky_traffic/config/field.yaml" \
  -p weights:="$WEIGHTS" -p lane:="$LANE"
