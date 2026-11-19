"""모듈 사이에서 주고받는 공통 데이터 구조."""
from dataclasses import dataclass
from typing import Tuple

BBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2) 픽셀 좌표
Point = Tuple[float, float]


@dataclass
class Track:
    """한 프레임에서 추적기가 내놓은 객체 하나."""

    track_id: int
    cls_id: int
    cls_name: str
    category: str  # "vehicle" | "two_wheeler" | "person" | "other"
    conf: float
    bbox: BBox

    @property
    def foot(self) -> Point:
        """지면 접점(BBox 하단 중앙). 원근 오차가 작아 위치 판정 기준으로 쓴다."""
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)


@dataclass
class Event:
    """위반 이벤트 한 건."""

    frame_idx: int
    time_sec: float
    track_id: int
    cls_name: str
    event_type: str  # "stopline" | "jaywalk"
    bbox: BBox
    detail: str = ""
