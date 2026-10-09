#!/bin/bash
# 주행 노드 2개만 다시 켠다 (코드 바꾼 뒤). 예전 주행 탭과 빈 터미널 탭은 닫고, 대시보드 탭은 남긴다.
#   scripts/restart_drivers.sh            # pinky1=1차선, pinky2=2차선
WS="$(cd "$(dirname "$0")/.." && pwd)"
kill -INT $(pgrep -f "install/pinky_traffic/lib/pinky_traffic/lane_[d]river") 2>/dev/null; sleep 2
for s in $(pgrep -x gnome-terminal-); do   # 이름으로만 찾는다 (-f 는 이 스크립트를 부른 셸까지 잡는다)
  for c in $(pgrep -P "$s"); do
    cmd="$(ps -o cmd= -p "$c")"
    [[ "$cmd" == *dashboard* ]] && continue
    # 주행 탭이거나, 아무것도 안 돌리는 빈 탭이면 닫는다
    if [[ "$cmd" == *drive.sh* ]] || [ -z "$(pgrep -P "$c")" ]; then pkill -9 -P "$c" 2>/dev/null; kill -9 "$c" 2>/dev/null; fi
  done
done
kill $(pgrep -f "ros2 run pinky_traffic lane_[d]river") 2>/dev/null; sleep 1
for r in "pinky1 24 1" "pinky2 23 2"; do set -- $r
  gnome-terminal --tab --title="$1" -- bash -c "cd '$WS'; PINKY_PEERS='192.168.129.199;192.168.129.200' '$WS/scripts/drive.sh' $1 $2 '' $3; exec bash"
done
