"""결과 시각화: ROI, BBox·ID, 궤적, ALERT 문구, 상태 표시줄.

OpenCV putText는 한글을 그리지 못하므로 화면 문구는 영어로 쓴다.
"""
from typing import Iterable, List, Set, Tuple

import cv2
import numpy as np

from .roi import ROIConfig
from .signal_state import GREEN, RED, YELLOW
from .structures import Event, Track
from .trajectory import TrajectoryStore

# BGR
CATEGORY_COLORS = {
    "vehicle": (255, 160, 0),
    "two_wheeler": (0, 200, 255),
    "person": (0, 220, 0),
    "other": (200, 200, 200),
}
VIOLATION_COLOR = (0, 0, 255)
SIGNAL_COLORS = {RED: (0, 0, 255), YELLOW: (0, 220, 255), GREEN: (0, 200, 0)}
EVENT_LABELS = {
    "stopline": "STOP LINE VIOLATION",
    "jaywalk": "JAYWALKING",
    "multi_lane": "MULTI-LANE CHANGE",
    "left_turn_lane_change": "LANE CHANGE IN LEFT TURN",
    "intersection_lane_change": "LANE CHANGE IN INTERSECTION",
    "right_turn_lane": "RIGHT TURN FROM WRONG LANE",
    "left_turn_lane": "LEFT TURN FROM WRONG LANE",
}
LANE_EVENTS = ("multi_lane", "left_turn_lane_change", "intersection_lane_change", "right_turn_lane", "left_turn_lane")
LANE_COLOR = (255, 200, 120)
LANE_X_COLOR = (200, 120, 255)


def _put_label(img, text, org, color, scale=0.5):
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x, y - h - base - 2), (x + w + 4, y), color, -1)
    cv2.putText(img, text, (x + 2, y - base), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_roi(img, roi: ROIConfig, signal: str) -> None:
    overlay = img.copy()
    for poly in roi.road:
        cv2.fillPoly(overlay, [poly.astype(np.int32)], (90, 90, 90))
    for poly in roi.crosswalks:
        cv2.fillPoly(overlay, [poly.astype(np.int32)], (0, 255, 255))
    cv2.addWeighted(overlay, 0.18, img, 0.82, 0, dst=img)

    for ln in roi.lanes:
        pts = ln.polygon.astype(np.int32)
        c = LANE_X_COLOR if ln.zone == "intersection" else LANE_COLOR
        cv2.polylines(img, [pts], True, c, 1, cv2.LINE_AA)
        cx, cy = pts.reshape(-1, 2).mean(axis=0)
        # 진입로 이름이 한글이면 OpenCV가 못 그리므로 차로 번호와 구역만 쓴다
        cv2.putText(img, f"{'X' if ln.zone == 'intersection' else 'L'}{ln.index}", (int(cx) - 8, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)

    line_color = SIGNAL_COLORS.get(signal, (255, 255, 255))
    for line in roi.stop_lines:
        p1 = tuple(int(v) for v in line.p1)
        p2 = tuple(int(v) for v in line.p2)
        cv2.line(img, p1, p2, line_color, 3, cv2.LINE_AA)
        cv2.putText(img, line.name, (p1[0], p1[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 1, cv2.LINE_AA)


def draw_tracks(img, tracks: Iterable[Track], traj: TrajectoryStore, violators: Set[int], lane_of=None) -> None:
    for t in tracks:
        violated = t.track_id in violators
        color = VIOLATION_COLOR if violated else CATEGORY_COLORS.get(t.category, (200, 200, 200))
        x1, y1, x2, y2 = (int(v) for v in t.bbox)

        pts = traj.get(t.track_id)
        if len(pts) >= 2:
            cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3 if violated else 2)
        cv2.circle(img, (int(t.foot[0]), int(t.foot[1])), 3, color, -1)
        lane = (lane_of or {}).get(t.track_id)
        lane_txt = f" {'X' if lane.zone == 'intersection' else 'L'}{lane.index}" if lane else ""
        _put_label(img, f"{t.cls_name} #{t.track_id} {t.conf:.2f}{lane_txt}", (x1, y1), color)


class AlertBoard:
    """최근 위반 이벤트를 일정 시간 동안 화면 상단에 띄운다."""

    def __init__(self, ttl_frames: int):
        self.ttl = ttl_frames
        self._items: List[Tuple[int, Event]] = []

    def add(self, event: Event) -> None:
        self._items.append((event.frame_idx, event))

    def draw(self, img, frame_idx: int) -> None:
        self._items = [(f, e) for f, e in self._items if frame_idx - f <= self.ttl]
        if not self._items:
            return
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, 36 + 26 * (len(self._items) - 1)), VIOLATION_COLOR, -1)
        for i, (_, e) in enumerate(self._items):
            text = f"ALERT  {EVENT_LABELS.get(e.event_type, e.event_type)}  ID {e.track_id}  t={e.time_sec:.1f}s"
            cv2.putText(img, text, (12, 26 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def draw_hud(img, fps: float, signal: str, counts: dict) -> None:
    h, w = img.shape[:2]
    lane_n = sum(counts.get(k, 0) for k in LANE_EVENTS)
    text = (f"FPS {fps:5.1f} | signal {signal.upper()} | stopline {counts.get('stopline', 0)} | "
            f"jaywalk {counts.get('jaywalk', 0)} | lane {lane_n}")
    cv2.rectangle(img, (0, h - 30), (w, h), (0, 0, 0), -1)
    cv2.putText(img, text, (10, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(img, (w - 20, h - 15), 9, SIGNAL_COLORS.get(signal, (128, 128, 128)), -1)
