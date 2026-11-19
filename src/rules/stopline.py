"""차량 정지선 위반 판정.

판정 기준
1. 적신호(red) 동안, 차량 지면 접점이 정지선 앞(+1)에서 정지선 너머(-1)로 넘어가면 위반 후보.
   - 적신호가 켜지기 전에 이미 넘어가 있던 차량은 '앞→너머' 전환이 없으므로 자동 제외된다.
   - 황색 신호 중 통과는 위반으로 보지 않는다.
2. 넘어간 상태가 confirm_frames 프레임 연속 유지되면 위반 확정 (BBox 흔들림 오탐 방지).
3. 같은 (ID, 정지선)은 한 번만 기록한다.
"""
from typing import Dict, List, Sequence, Set, Tuple

from ..roi import StopLine
from ..signal_state import RED
from ..structures import Event, Track

APPLIES_TO = ("vehicle", "two_wheeler")


class StopLineRule:
    EVENT_TYPE = "stopline"

    def __init__(self, stop_lines: Sequence[StopLine], confirm_frames: int = 3):
        self.stop_lines = list(stop_lines)
        self.confirm_frames = max(1, confirm_frames)
        self._prev_side: Dict[Tuple[int, str], int] = {}
        self._pending: Dict[Tuple[int, str], int] = {}
        self._reported: Set[Tuple[int, str]] = set()
        self.violators: Set[int] = set()  # 화면에서 빨간 박스로 표시할 ID

    def update(self, tracks: Sequence[Track], frame_idx: int, t_sec: float, signal: str) -> List[Event]:
        events = []
        for tr in tracks:
            if tr.category not in APPLIES_TO:
                continue
            for line in self.stop_lines:
                key = (tr.track_id, line.name)
                side = line.side(tr.foot)
                prev = self._prev_side.get(key)
                if side != 0:
                    self._prev_side[key] = side

                if key in self._reported:
                    continue
                if side == 0:
                    self._pending.pop(key, None)
                    continue

                if prev == 1 and side == -1 and signal == RED:
                    self._pending[key] = 1  # 적신호 중 정지선 통과 시작
                elif key in self._pending:
                    if side == -1:
                        self._pending[key] += 1
                    else:
                        del self._pending[key]  # 다시 정지선 앞으로 돌아옴

                if self._pending.get(key, 0) >= self.confirm_frames:
                    del self._pending[key]
                    self._reported.add(key)
                    self.violators.add(tr.track_id)
                    events.append(
                        Event(
                            frame_idx=frame_idx,
                            time_sec=t_sec,
                            track_id=tr.track_id,
                            cls_name=tr.cls_name,
                            event_type=self.EVENT_TYPE,
                            bbox=tr.bbox,
                            detail=f"line={line.name}",
                        )
                    )
        return events
