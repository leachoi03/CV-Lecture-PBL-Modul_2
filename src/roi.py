"""관심 영역(ROI): 정지선, 차도, 횡단보도 + 기하 판정 함수."""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import cv2
import numpy as np

from .structures import BBox, Point


# ---------------------------------------------------------------- 기하 함수
def cross(o: Point, a: Point, b: Point) -> float:
    """벡터 OA × OB 의 z 성분. 부호로 점 B가 선 OA의 어느 쪽에 있는지 알 수 있다."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def to_poly(points) -> np.ndarray:
    return np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)


def in_polygon(poly: np.ndarray, pt: Point) -> bool:
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0


def box_area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection_over_a(a: BBox, b: BBox) -> float:
    """a 넓이 대비 a∩b 넓이 (IoA)."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = box_area(a)
    return inter / area if area > 0 else 0.0


# ---------------------------------------------------------------- 정지선
@dataclass
class StopLine:
    name: str
    p1: Point
    p2: Point
    approach_point: Point  # 정지선 앞(차가 대기하는 쪽)에 있는 임의의 점
    margin: float = 0.1  # 선분 양 끝을 이 비율만큼 연장해서 판정

    def __post_init__(self):
        s = cross(self.p1, self.p2, self.approach_point)
        if s == 0:
            raise ValueError(f"정지선 '{self.name}': approach_point가 선 위에 있습니다.")
        self._sign = 1 if s > 0 else -1

    def side(self, pt: Point) -> int:
        """+1: 정지선 앞(대기 쪽), -1: 정지선을 넘음, 0: 선분 범위 밖이라 판정 안 함."""
        dx, dy = self.p2[0] - self.p1[0], self.p2[1] - self.p1[1]
        length2 = dx * dx + dy * dy
        t = ((pt[0] - self.p1[0]) * dx + (pt[1] - self.p1[1]) * dy) / length2
        if t < -self.margin or t > 1 + self.margin:
            return 0
        s = cross(self.p1, self.p2, pt) * self._sign
        return 1 if s > 0 else -1


# ---------------------------------------------------------------- ROI 설정 전체
@dataclass
class ROIConfig:
    stop_lines: List[StopLine] = field(default_factory=list)
    road: List[np.ndarray] = field(default_factory=list)
    crosswalks: List[np.ndarray] = field(default_factory=list)
    signal: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "ROIConfig":
        lines = [
            StopLine(
                name=s.get("name", f"line{i}"),
                p1=tuple(s["p1"]),
                p2=tuple(s["p2"]),
                approach_point=tuple(s["approach_point"]),
            )
            for i, s in enumerate(d.get("stop_lines", []))
        ]
        return cls(
            stop_lines=lines,
            road=[to_poly(p) for p in d.get("road", [])],
            crosswalks=[to_poly(p) for p in d.get("crosswalks", [])],
            signal=d.get("signal", {}),
        )

    @classmethod
    def load(cls, path) -> "ROIConfig":
        with open(Path(path), encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def in_road(self, pt: Point) -> bool:
        return any(in_polygon(p, pt) for p in self.road)

    def in_crosswalk(self, pt: Point) -> bool:
        return any(in_polygon(p, pt) for p in self.crosswalks)
