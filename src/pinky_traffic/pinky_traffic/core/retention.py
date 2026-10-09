"""주행 사진 폴더(runs/frames_*) 정리. 주행 노드가 켜질 때 한 번 부른다 (ROS 없음).

- keep_days 보다 오래된 폴더는 지운다.
- 그래도 합계가 max_gb 를 넘으면 오래된 폴더부터 지운다.
- 폴더 안에 KEEP 파일이 있으면 건드리지 않는다 (학습에 쓸 주행 등). 합계에도 넣지 않는다.
- CSV 로그(runs/<날짜>/)·testcase_results.json 은 작아서 지우지 않는다 (frames_ 로 시작하는 폴더만 본다).
"""
import os
import shutil
import time

KEEP_FILE = 'KEEP'


def _size(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def prune_frames(runs_dir='runs', keep_days=7.0, max_gb=20.0, now=None, log=print):
    """지운 폴더 목록을 돌려준다. keep_days/max_gb 가 0 이하이면 그 기준은 끈다."""
    now = time.time() if now is None else now
    if not os.path.isdir(runs_dir):
        return []
    folders = []
    for name in os.listdir(runs_dir):
        path = os.path.join(runs_dir, name)
        if name.startswith('frames_') and os.path.isdir(path) and not os.path.exists(os.path.join(path, KEEP_FILE)):
            folders.append((os.path.getmtime(path), path))
    folders.sort()                                  # 오래된 것부터
    removed = []
    if keep_days > 0:
        for mtime, path in list(folders):
            if now - mtime > keep_days * 86400:
                shutil.rmtree(path, ignore_errors=True)
                removed.append(path)
                folders.remove((mtime, path))
    if max_gb > 0:
        sizes = [(mtime, path, _size(path)) for mtime, path in folders]
        total = sum(s for _, _, s in sizes)
        for mtime, path, size in sizes:
            if total <= max_gb * 1024 ** 3:
                break
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path)
            total -= size
    if removed:
        log(f'오래된 주행 사진 폴더 {len(removed)}개 정리 (기준 {keep_days:g}일 / {max_gb:g}GB, KEEP 파일이 있으면 남김)')
    return removed
