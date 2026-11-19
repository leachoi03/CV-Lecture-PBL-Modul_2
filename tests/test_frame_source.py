from pathlib import Path

import cv2
import numpy as np

from src.frame_source import FrameSource, check_sequence


def _write(folder: Path, names, size=(32, 24)):
    folder.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(names):
        img = np.full((size[1], size[0], 3), i * 20, np.uint8)
        ok, buf = cv2.imencode(".jpg", img)
        buf.tofile(str(folder / n))


def test_folder_natural_order_and_korean_path(tmp_path):
    folder = tmp_path / "한글_시퀀스"
    _write(folder, ["f10.jpg", "f2.jpg", "f1.jpg"])
    src = FrameSource(str(folder), fps=5)
    assert src.is_folder and src.fps == 5 and src.frame_count == 3
    assert [p.name for p in src.files] == ["f1.jpg", "f2.jpg", "f10.jpg"]
    frames = []
    while True:
        ok, f = src.read()
        if not ok:
            break
        frames.append(f)
    assert len(frames) == 3 and frames[0].shape == (24, 32, 3)


def test_resizes_mismatched_frames(tmp_path):
    _write(tmp_path, ["a_001.jpg"], size=(32, 24))
    _write(tmp_path, ["a_002.jpg"], size=(64, 48))
    src = FrameSource(str(tmp_path))
    src.read()
    ok, f = src.read()
    assert ok and f.shape == (24, 32, 3)


def test_check_sequence():
    contiguous = [Path(f"94022_2021110318392086_{i:03d}.jpg") for i in range(1, 6)]
    assert check_sequence(contiguous) is None
    sparse = [Path("94022_2021110318392086_012.jpg"), Path("94022_2021110318392086_043.jpg")]
    assert check_sequence(sparse) is not None
    mixed = [Path("a_cam1_001.jpg"), Path("a_cam2_002.jpg")]
    assert check_sequence(mixed) is not None
