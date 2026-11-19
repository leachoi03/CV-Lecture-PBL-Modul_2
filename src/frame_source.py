"""입력 소스 통합: 영상 파일 또는 이미지 폴더(JPG/PNG 시퀀스)를 같은 방식으로 읽는다.

이미지 폴더는 파일 이름의 숫자를 고려한 자연 정렬 순서로 읽으며, fps 정보가 없으므로 --fps 로 지정한다.
한글 경로에서도 열리도록 이미지는 np.fromfile + cv2.imdecode 로 읽는다.
"""
import re
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_IMAGE_FPS = 10.0


def natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def imread_any(path) -> Optional[np.ndarray]:
    """한글 등 비ASCII 경로도 읽을 수 있는 imread."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def list_images(folder: Path, recursive: bool = False) -> List[Path]:
    it = folder.rglob("*") if recursive else folder.iterdir()
    return sorted((p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS), key=natural_key)


def check_sequence(files: List[Path]) -> Optional[str]:
    """AI Hub 형식(카메라_클립시각_프레임번호.jpg)이 연속 프레임이 아니면 경고 문구를 돌려준다."""
    clips, gaps = set(), 0
    prev = None
    for f in files:
        m = re.match(r"(.+)_(\d+)$", f.stem)
        if not m:
            return None  # 다른 이름 규칙이면 판단하지 않음
        clip, num = m.group(1), int(m.group(2))
        clips.add(clip)
        if prev and prev[0] == clip and num - prev[1] > 1:
            gaps += 1
        prev = (clip, num)
    if len(clips) > 1 or gaps:
        return (f"경고: 연속 프레임이 아닙니다 (클립 {len(clips)}개, 프레임 번호 건너뜀 {gaps}회). "
                "추적 ID와 위반 판정(정지선 통과·지속 시간)은 의미가 없고, 프레임 단위 검출 확인용으로만 쓰세요.")
    return None


class FrameSource:
    """read() -> (ok, frame) 인터페이스는 cv2.VideoCapture 와 같다."""

    def __init__(self, source: str, fps: Optional[float] = None, recursive: bool = False):
        self.path = Path(source)
        self.is_folder = self.path.is_dir()
        self.warning = None
        if self.is_folder:
            self.files = list_images(self.path, recursive)
            if not self.files:
                raise SystemExit(f"이미지가 없습니다: {source}")
            self.warning = check_sequence(self.files)
            first = imread_any(self.files[0])
            if first is None:
                raise SystemExit(f"이미지를 읽을 수 없습니다: {self.files[0]}")
            self.height, self.width = first.shape[:2]
            self.fps = fps or DEFAULT_IMAGE_FPS
            self.frame_count = len(self.files)
            self._idx = 0
            self.cap = None
        else:
            self.cap = cv2.VideoCapture(str(self.path))
            if not self.cap.isOpened():
                raise SystemExit(f"영상을 열 수 없습니다: {source}")
            self.fps = fps or self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def name(self) -> str:
        return self.path.name if self.is_folder else self.path.stem

    def read(self):
        if self.cap is not None:
            return self.cap.read()
        while self._idx < len(self.files):
            f = self.files[self._idx]
            self._idx += 1
            img = imread_any(f)
            if img is None:
                print(f"건너뜀 (읽기 실패): {f}")
                continue
            if img.shape[:2] != (self.height, self.width):
                img = cv2.resize(img, (self.width, self.height))  # VideoWriter 는 크기가 같아야 함
            return True, img
        return False, None

    def seek(self, index: int):
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        else:
            self._idx = max(0, min(index, len(self.files) - 1))

    def release(self):
        if self.cap is not None:
            self.cap.release()
