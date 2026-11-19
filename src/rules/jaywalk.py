"""보행자 무단횡단 판정.

판정 기준
1. 보행자 지면 접점이 '차도 안 AND 횡단보도 밖' 영역에 있으면 위반 구역에 있는 것으로 본다.
2. 위반 구역에 min_frames 프레임 연속 머무르면 위반 확정 (경계에서 잠깐 걸친 경우 제외).
3. 이륜차 박스(위로 확장)와 rider_overlap 이상 겹치는 사람은 탑승자로 보고 제외.
4. 같은 ID는 한 번만 기록한다.
"""
from typing import Dict, List, Sequence, Set

from ..roi import ROIConfig, intersection_over_a
from ..structures import BBox, Event, Track


def _expand_up(b: BBox, ratio: float = 0.6) -> BBox:
    """이륜차 박스를 위로 늘린다. 탑승자의 상체는 이륜차 박스보다 위에 있기 때문."""
    h = b[3] - b[1]
    return (b[0], b[1] - ratio * h, b[2], b[3])


class JaywalkRule:
    EVENT_TYPE = "jaywalk"

    def __init__(self, roi: ROIConfig, min_frames: int = 30, rider_overlap: float = 0.3):
        self.roi = roi
        self.min_frames = max(1, min_frames)
        self.rider_overlap = rider_overlap
        self._count: Dict[int, int] = {}
        self._reported: Set[int] = set()
        self.violators: Set[int] = set()

    def _is_rider(self, person: Track, riders: Sequence[BBox]) -> bool:
        return any(intersection_over_a(person.bbox, r) >= self.rider_overlap for r in riders)

    def update(self, tracks: Sequence[Track], frame_idx: int, t_sec: float) -> List[Event]:
        riders = [_expand_up(t.bbox) for t in tracks if t.category == "two_wheeler"]
        events = []
        for tr in tracks:
            if tr.category != "person" or tr.track_id in self._reported:
                continue
            if self._is_rider(tr, riders):
                self._count.pop(tr.track_id, None)
                continue

            pt = tr.foot
            in_zone = self.roi.in_road(pt) and not self.roi.in_crosswalk(pt)
            if in_zone:
                self._count[tr.track_id] = self._count.get(tr.track_id, 0) + 1
            else:
                self._count.pop(tr.track_id, None)

            if self._count.get(tr.track_id, 0) >= self.min_frames:
                self._reported.add(tr.track_id)
                self.violators.add(tr.track_id)
                events.append(
                    Event(
                        frame_idx=frame_idx,
                        time_sec=t_sec,
                        track_id=tr.track_id,
                        cls_name=tr.cls_name,
                        event_type=self.EVENT_TYPE,
                        bbox=tr.bbox,
                        detail=f"frames_in_zone={self._count[tr.track_id]}",
                    )
                )
        return events
