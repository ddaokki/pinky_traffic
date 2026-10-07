#!/bin/bash
# 한 번에 끄기: PC 주행 노드 → 로봇 2대 bringup/카메라/LED. 대시보드는 남겨 둔다 (기록 확인용).
#   scripts/stop_all.sh            # 대시보드까지 끄려면: scripts/stop_all.sh all
pkill -f "lane_driver" && echo "PC 주행 노드 끔"
for IP in 192.168.129.199 192.168.129.200; do
  # 명령을 표준입력으로 보낸다 (명령줄에 넣으면 pkill -f 가 자기 ssh 셸까지 죽인다)
  ssh -o ConnectTimeout=4 -o BatchMode=yes "pinky@$IP" bash -s 2>/dev/null <<'R' || echo "$IP 연결 안 됨 (이미 꺼졌거나 열쇠 미등록)"
pkill -f camera_pub.py; pkill -f led_server; pkill -f bringup_robot.launch.xml
echo "$(hostname -I | cut -d' ' -f1) 로봇 끔"
R
done
[ "$1" = "all" ] && pkill -f "pinky_traffic.dashboard.server" && echo "대시보드 끔"
