"""주행 사진 폴더 정리: 오래된 것·용량 넘친 것 지우기, KEEP·CSV 는 남기기."""
import os

from pinky_traffic.core.retention import prune_frames

DAY = 86400


def make(root, name, age_days, size_kb=0, keep=False, now=1e9):
    path = root / name
    path.mkdir()
    (path / 'a.jpg').write_bytes(b'x' * size_kb * 1024)
    if keep:
        (path / 'KEEP').write_text('')
    os.utime(path, (now - age_days * DAY, now - age_days * DAY))
    return path


def test_old_folders_removed_but_keep_and_logs_stay(tmp_path):
    now = 1e9
    old = make(tmp_path, 'frames_old', 8)
    kept = make(tmp_path, 'frames_kept', 30, keep=True)
    new = make(tmp_path, 'frames_new', 1)
    log = make(tmp_path, '20261001_100000', 30)                 # CSV 로그 폴더
    removed = prune_frames(str(tmp_path), keep_days=7, max_gb=0, now=now, log=lambda *_: None)
    assert removed == [str(old)]
    assert kept.exists() and new.exists() and log.exists()


def test_over_size_removes_oldest_first(tmp_path):
    now = 1e9
    a = make(tmp_path, 'frames_a', 3, size_kb=600)
    b = make(tmp_path, 'frames_b', 2, size_kb=600)
    c = make(tmp_path, 'frames_c', 1, size_kb=600)
    removed = prune_frames(str(tmp_path), keep_days=0, max_gb=1.3 / 1024, now=now, log=lambda *_: None)   # 1.3MB
    assert removed == [str(a)] and b.exists() and c.exists()


def test_missing_runs_dir_is_fine(tmp_path):
    assert prune_frames(str(tmp_path / 'none')) == []
