#!/bin/bash
# 한 번에 켜기: 로봇 2대 bringup + 카메라 + LED 서버 → PC 대시보드 → 주행 노드 2개 → 브라우저.
#   scripts/start_all.sh            # 아래 ROBOTS 표 그대로
#   ROBOTS_ONLY=pinky1 scripts/start_all.sh   # 한 대만
# 주행 노드는 켜져도 대시보드에서 START 를 누르기 전에는 바퀴를 돌리지 않는다 (autostart=False).
# 끄기: scripts/stop_all.sh
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"

# 이름  IP(학원 와이파이 FASTCAMPUS_10F)  기본 도메인  차선  영상 뒤집기
ROBOTS=(
  "pinky1 192.168.129.199 24 0 true"
  "pinky2 192.168.129.200 23 0 true"
)
BACKEND="${BACKEND:-hsv}"     # yolo 로 달리려면: BACKEND=yolo scripts/start_all.sh
SSH_OPTS=(-o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new)

say()  { echo -e "\n\033[1;36m▶ $*\033[0m"; }
fail() { echo -e "\033[1;31m✗ $*\033[0m"; }
pause_exit() { echo; read -rp "엔터를 누르면 창이 닫힙니다..." _; exit "${1:-1}"; }

# 0. SSH 키 (처음 한 번만 만든다)
[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 -q

PEERS=""
DRIVERS=()
for row in "${ROBOTS[@]}"; do
  read -r NAME IP DOMAIN LANE FLIP <<<"$row"
  [ -n "${ROBOTS_ONLY:-}" ] && [[ " $ROBOTS_ONLY " != *" $NAME "* ]] && continue

  say "$NAME ($IP) 연결 확인"
  if ! timeout 4 bash -c "</dev/tcp/$IP/22" 2>/dev/null; then
    fail "$NAME 이 안 보입니다. 로봇 전원과 와이파이(FASTCAMPUS_10F)를 확인하세요. 건너뜁니다."
    continue
  fi
  if ! ssh "${SSH_OPTS[@]}" -o BatchMode=yes "pinky@$IP" true 2>/dev/null; then
    echo "처음 연결이라 열쇠를 로봇에 등록합니다. 로봇 비밀번호를 입력하세요."
    ssh-copy-id "${SSH_OPTS[@]}" "pinky@$IP" || { fail "$NAME 열쇠 등록 실패. 건너뜁니다."; continue; }
  fi

  # 로봇이 PC 에 닿는 PC 쪽 주소 (자동 탐색이 막힌 망이라 로봇에 직접 알려 준다)
  PC_IP="$(ip -4 route get "$IP" | sed -n 's/.* src \([0-9.]*\).*/\1/p')"
  # 로봇 .bashrc 의 도메인을 그대로 쓴다 (없으면 표의 값)
  RD="$(ssh "${SSH_OPTS[@]}" "pinky@$IP" 'bash -ic "echo \$ROS_DOMAIN_ID" 2>/dev/null' | tail -1 | tr -dc 0-9)"
  [ -n "$RD" ] && DOMAIN="$RD"
  echo "도메인 $DOMAIN, PC 주소 $PC_IP"

  say "$NAME 카메라 노드 복사 + bringup/카메라/LED 시작"
  scp -q "${SSH_OPTS[@]}" "$WS/src/pinky_traffic/pinky_traffic/nodes/camera_pub.py" "pinky@$IP:~/camera_pub.py"
  ssh "${SSH_OPTS[@]}" "pinky@$IP" bash -s <<EOF
pkill -f bringup_robot.launch.xml; pkill -f camera_pub.py; pkill -f led_server; sleep 1
export ROS_STATIC_PEERS=$PC_IP
nohup setsid bash -ic 'export ROS_DOMAIN_ID=$DOMAIN ROS_STATIC_PEERS=$PC_IP; ros2 launch pinky_bringup bringup_robot.launch.xml' >~/start_bringup.log 2>&1 </dev/null &
sleep 3
nohup setsid bash -ic 'export ROS_DOMAIN_ID=$DOMAIN ROS_STATIC_PEERS=$PC_IP; python3 ~/camera_pub.py --ros-args -p width:=320 -p height:=240 -p fps:=15.0 -p flip:=$FLIP' >~/start_camera.log 2>&1 </dev/null &
nohup setsid bash -ic 'export ROS_DOMAIN_ID=$DOMAIN ROS_STATIC_PEERS=$PC_IP; ros2 run pinky_led led_server' >~/start_led.log 2>&1 </dev/null &
sleep 3
pgrep -f bringup_robot.launch.xml >/dev/null && echo "bringup 켜짐" || echo "bringup 실패: ~/start_bringup.log 확인"
pgrep -f camera_pub.py >/dev/null && echo "카메라 켜짐" || echo "카메라 실패: ~/start_camera.log 확인"
pgrep -f led_server >/dev/null && echo "LED 서버 켜짐" || echo "LED 서버 실패: ~/start_led.log 확인"
EOF
  PEERS="${PEERS:+$PEERS;}$IP"
  DRIVERS+=("$NAME $DOMAIN $LANE")
done

[ ${#DRIVERS[@]} -eq 0 ] && { fail "켜진 로봇이 없습니다."; pause_exit 1; }

say "대시보드"
if ss -ltn | grep -q ':8088 '; then
  echo "이미 켜져 있습니다."
else
  gnome-terminal --tab --title="dashboard" -- bash -c "'$WS/scripts/dashboard.sh'; exec bash"
  for _ in $(seq 20); do ss -ltn | grep -q ':8088 ' && break; sleep 0.5; done
fi

say "주행 노드 ($BACKEND) — START 를 누르기 전에는 움직이지 않습니다"
for d in "${DRIVERS[@]}"; do
  read -r NAME DOMAIN LANE <<<"$d"
  gnome-terminal --tab --title="$NAME" -- bash -c \
    "PINKY_PEERS='$PEERS' '$WS/scripts/drive.sh' $NAME $DOMAIN $BACKEND '' $LANE; exec bash"
done

xdg-open http://localhost:8088 >/dev/null 2>&1 &
say "완료. 브라우저 대시보드에서 로봇 카드가 뜨는지 보세요 (10초쯤 걸립니다)."
pause_exit 0
