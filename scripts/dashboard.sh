#!/bin/bash
# 대시보드 서버 (PC 에서 한 번만). 브라우저: http://localhost:8088
cd "$(dirname "$0")/.." && PYTHONPATH="src/pinky_traffic:$PYTHONPATH" exec python3 -m pinky_traffic.dashboard.server --port "${1:-8088}"
