"""시뮬레이션 실행 (로봇 없이 코드/파라미터 확인).

  python3 -m pinky_traffic.tools.run_sim                          # 1대, 창으로 보기
  python3 -m pinky_traffic.tools.run_sim --robots 2 --coordinator # 2대 + 횡단보도 락
  python3 -m pinky_traffic.tools.run_sim --headless --seconds 150 --video out.mp4
  python3 -m pinky_traffic.tools.run_sim --dashboard              # 대시보드로 보기/조작 (서버 먼저 실행)
  python3 -m pinky_traffic.tools.run_sim --backend yolo --weights best.pt
"""
import argparse
import json
import time

import cv2
import numpy as np

from ..core.config import Config
from ..core.driver import Driver
from ..sim.runner import Simulation, front_distance
from ..sim.track import Track, TrackSpec

SIM_YELLOW = dict(crosswalk_hsv_lo=[20, 100, 120], crosswalk_hsv_hi=[35, 255, 255])   # 시뮬 기본 코스의 노란 횡단보도


def compose(sim):
    top = sim.topview(0.35)
    views = [sim.frames[r.name] for r in sim.robots if r.name in sim.frames]
    if not views:
        return top
    side = np.vstack(views)
    scale = top.shape[0] / side.shape[0]
    side = cv2.resize(side, (int(side.shape[1] * scale), top.shape[0]))
    return np.hstack([top, side])


def run_with_dashboard(args, cfg, track):
    """각 로봇을 실제 노드와 같은 Driver 로 돌린다 -> 대시보드 버튼/슬라이더가 그대로 먹는다."""
    sim = Simulation(cfg, track, n_robots=args.robots, use_coordinator=args.coordinator)
    drivers = []
    for robot, c in zip(sim.robots, sim.cfgs):
        c.dashboard_url = args.url
        drivers.append(Driver(c, robot.name, use_dashboard=True, autostart=args.autostart))
    print(f'대시보드 {args.url} 에서 START 를 누르세요 (Ctrl+C 로 종료)')
    try:
        while True:
            t0 = time.time()
            for robot, driver, c in zip(sim.robots, drivers, sim.cfgs):
                others = [o for o in sim.robots if o is not robot]
                front = front_distance(robot, others, c.front_angle_deg) if others else None
                cmd = driver.process(sim.camera.render(robot.x, robot.y, robot.yaw), front)
                robot.command(cmd.v, cmd.w)
            for robot in sim.robots:
                robot.step(sim.dt)
            time.sleep(max(0.0, sim.dt - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        for driver in drivers:
            driver.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--robots', type=int, default=1)
    parser.add_argument('--seconds', type=float, default=150)
    parser.add_argument('--coordinator', action='store_true', help='횡단보도 락 사용 (2대)')
    parser.add_argument('--config', default=None)
    parser.add_argument('--backend', default=None)
    parser.add_argument('--weights', default=None)
    parser.add_argument('--same-color', action='store_true', help='횡단보도를 차선과 같은 흰색으로')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--video', default=None, help='mp4 저장 경로')
    parser.add_argument('--dashboard', action='store_true')
    parser.add_argument('--autostart', action='store_true')
    parser.add_argument('--url', default='http://127.0.0.1:8088')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    spec = TrackSpec()
    overrides = {k: v for k, v in (('backend', args.backend), ('weights', args.weights)) if v}
    if args.same_color:
        spec.crosswalk_bgr = spec.lane_bgr
    else:
        overrides.update(SIM_YELLOW)
    cfg = Config.load(args.config, **overrides)
    track = Track(spec, seed=args.seed)

    if args.dashboard:
        return run_with_dashboard(args, cfg, track)

    sim = Simulation(cfg, track, n_robots=args.robots, use_coordinator=args.coordinator)
    sim.start()
    writer = None
    show = not args.headless
    for i in range(int(args.seconds / sim.dt)):
        draw = show or (args.video and i % 2 == 0)
        sim.step(draw=bool(draw))
        if draw:
            canvas = compose(sim)
            if args.video:
                if writer is None:
                    h, w = canvas.shape[:2]
                    writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*'mp4v'), 15, (w, h))
                writer.write(canvas)
            if show:
                cv2.imshow('pinky_traffic sim (ESC 종료)', canvas)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(json.dumps(sim.metrics(), indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
