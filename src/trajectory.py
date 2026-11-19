"""ID별 이동 궤적(지면 접점 좌표) 저장."""
from collections import deque
from typing import Dict, Iterable, List

from .structures import Point, Track


class TrajectoryStore:
    def __init__(self, maxlen: int = 40, max_age: int = 60):
        self.maxlen = maxlen
        self.max_age = max_age  # 이 프레임 수 이상 안 보인 ID는 삭제
        self._points: Dict[int, deque] = {}
        self._last_seen: Dict[int, int] = {}
        self.all_ids = set()  # 지금까지 등장한 모든 ID (평가용)

    def update(self, tracks: Iterable[Track], frame_idx: int) -> None:
        for t in tracks:
            dq = self._points.setdefault(t.track_id, deque(maxlen=self.maxlen))
            dq.append(t.foot)
            self._last_seen[t.track_id] = frame_idx
            self.all_ids.add(t.track_id)

        stale = [tid for tid, f in self._last_seen.items() if frame_idx - f > self.max_age]
        for tid in stale:
            self._points.pop(tid, None)
            self._last_seen.pop(tid, None)

    def get(self, track_id: int) -> List[Point]:
        return list(self._points.get(track_id, ()))
