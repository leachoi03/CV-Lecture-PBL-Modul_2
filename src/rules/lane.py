"""차로 위반 판정 (웹 편집기에서 그린 lanes 사용).

차량마다 지면 접점(bbox 아래 가운데)이 들어 있는 차로를 매 프레임 찾는다.
경계에서 흔들리는 값을 거르려고, 같은 차로가 stable_frames 프레임 연속 나와야 '그 차로에 있음'으로 확정한다.

판정 기준
1. 2차로 이상 한 번에 변경 (multi_lane)
   진입부(zone=approach) 차로가 바뀌어 확정될 때, 같은 진입로에서 최근 window 초 안에 있었던 차로와
   번호 차이가 2 이상이면 위반. 1→2→3 처럼 window 안에 연달아 바꾼 것도 포함된다.
2. 교차로 안 차로 변경 (left_turn_lane_change / intersection_lane_change)
   교차로 안(zone=intersection) 차로에서 같은 진입로의 다른 교차로 안 차로로 옮겨 확정되면 위반.
   두 차로 모두 좌회전 칸이면 left_turn_lane_change, 아니면 intersection_lane_change.
   다른 진입로의 유도선 칸은 교차로 안에서 겹치므로, 차량이 들어온 진입로의 칸만 본다.
3. 회전 차로 위반 (right_turn_lane 등)
   그려 둔 차로를 모두 벗어난 뒤 진행 방향이 진입 때보다 turn_degrees 이상 오른쪽(왼쪽)으로 꺾이면 우(좌)회전으로 본다.
   들어온 진입부 차로의 진행 가능 방향(turns)에 그 회전이 없으면 위반. turns 를 비워 둔 차로는 판정하지 않는다.
   화면 좌표(y 아래쪽)에서 외적 > 0 이면 화면상 시계 방향 = 오른쪽으로 꺾음.
같은 (ID, 위반 종류)는 한 번만 기록한다.
"""
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Set, Tuple

from ..roi import Lane, ROIConfig, in_polygon
from ..structures import Event, Point, Track


@dataclass
class _Segment:
    lane: Lane
    t0: float
    t1: float
    pts: List[Point] = field(default_factory=list)


@dataclass
class _State:
    raw: Optional[str] = None          # 이번 프레임에 찾은 차로 ID (None = 차로 밖)
    raw_count: int = 0
    cur: Optional[Lane] = None         # 확정된 차로
    segments: List[_Segment] = field(default_factory=list)
    entry: Optional[_Segment] = None   # 마지막으로 확정된 진입부 차로 구간
    turn_done: bool = False
    pts: Deque[Tuple[float, Point]] = field(default_factory=lambda: deque(maxlen=120))
    last_frame: int = 0


def _vec(a: Point, b: Point) -> Tuple[float, float]:
    return (b[0] - a[0], b[1] - a[1])


def signed_angle(v1, v2) -> float:
    """v1 → v2 회전 각도(도). 화면 좌표에서 + 는 시계 방향(오른쪽)."""
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    return math.degrees(math.atan2(cross, dot))


class LaneRule:
    def __init__(self, roi: ROIConfig, fps: float, applies_to: Sequence[str] = ("vehicle",),
                 stable_seconds: float = 0.3, multi_window_seconds: float = 3.0,
                 turn_degrees: float = 45.0, heading_seconds: float = 0.7, min_move_px: float = 15.0,
                 turn_checks: Sequence[str] = ("right",), max_age: int = 60):
        self.lanes = list(roi.lanes)
        self.applies_to = set(applies_to)
        self.stable_frames = max(1, round(stable_seconds * fps))
        self.window = multi_window_seconds
        self.turn_degrees = turn_degrees
        self.heading_seconds = heading_seconds
        self.min_move = min_move_px
        self.turn_checks = set(turn_checks)
        self.max_age = max_age
        self._st: Dict[int, _State] = {}
        self._reported: Set[Tuple[int, str]] = set()
        self.violators: Set[int] = set()
        self.current: Dict[int, Lane] = {}  # 화면 표시용: 추적 ID → 확정 차로

    # ------------------------------------------------------------ 차로 찾기
    def lane_of(self, pt: Point, entry_approach: Optional[str]) -> Optional[Lane]:
        hits = [ln for ln in self.lanes if in_polygon(ln.polygon, pt)]
        if entry_approach is not None:
            # 교차로 안에서는 들어온 진입로의 칸만 본다 (다른 진입로 유도선과 겹치기 때문)
            hits = [ln for ln in hits if ln.zone != "intersection" or ln.approach == entry_approach]
        return min(hits, key=lambda ln: ln.area) if hits else None

    # ------------------------------------------------------------ 갱신
    def update(self, tracks: Sequence[Track], frame_idx: int, t_sec: float) -> List[Event]:
        events: List[Event] = []
        if not self.lanes:
            return events
        for tr in tracks:
            if tr.category not in self.applies_to:
                continue
            st = self._st.setdefault(tr.track_id, _State())
            st.last_frame = frame_idx
            pt = tr.foot
            st.pts.append((t_sec, pt))

            lane = self.lane_of(pt, st.entry.lane.approach if st.entry else None)
            lid = lane.id if lane else None
            if lid == st.raw:
                st.raw_count += 1
            else:
                st.raw, st.raw_count = lid, 1

            if st.cur is not None and st.segments and st.segments[-1].lane is st.cur:
                seg = st.segments[-1]
                seg.t1 = t_sec
                seg.pts.append(pt)

            cur_id = st.cur.id if st.cur else None
            if st.raw_count >= self.stable_frames and st.raw != cur_id:
                events += self._commit(tr, st, lane, frame_idx, t_sec)

            events += self._check_turn(tr, st, frame_idx, t_sec)
            if st.cur:
                self.current[tr.track_id] = st.cur
            else:
                self.current.pop(tr.track_id, None)

        stale = [tid for tid, s in self._st.items() if frame_idx - s.last_frame > self.max_age]
        for tid in stale:
            del self._st[tid]
            self.current.pop(tid, None)
        return events

    def _commit(self, tr: Track, st: _State, lane: Optional[Lane], frame_idx: int, t_sec: float) -> List[Event]:
        events = []
        prev_seg = st.segments[-1] if st.segments else None
        st.cur = lane
        if lane is None:
            return events
        seg = _Segment(lane, t_sec, t_sec, [tr.foot])

        if lane.zone == "approach":
            # 1. 2차로 이상 변경: 같은 진입로, 최근 window 초 안에 있던 진입부 차로와 비교
            recent = [s for s in st.segments if s.lane.zone == "approach" and s.lane.approach == lane.approach
                      and s.t1 >= t_sec - self.window and s.lane.id != lane.id]
            if recent:
                far = max(recent, key=lambda s: abs(s.lane.index - lane.index))
                if abs(far.lane.index - lane.index) >= 2:
                    events += self._event(tr, "multi_lane", frame_idx, t_sec,
                                          f"{far.lane.id}->{lane.id} in {t_sec - far.t1:.1f}s")
            st.entry = seg
        elif prev_seg is not None and prev_seg.lane.zone == "intersection" \
                and prev_seg.lane.approach == lane.approach and prev_seg.lane.id != lane.id \
                and prev_seg.t1 >= t_sec - self.window:
            # 2. 교차로 안 차로 변경
            left = "left" in prev_seg.lane.turns and "left" in lane.turns
            kind = "left_turn_lane_change" if left else "intersection_lane_change"
            events += self._event(tr, kind, frame_idx, t_sec, f"{prev_seg.lane.id}->{lane.id}")
        st.segments.append(seg)
        return events

    def _heading(self, pts: Sequence[Point]) -> Optional[Tuple[float, float]]:
        if len(pts) < 2:
            return None
        v = _vec(pts[0], pts[-1])
        return v if math.hypot(*v) >= self.min_move else None

    def _check_turn(self, tr: Track, st: _State, frame_idx: int, t_sec: float) -> List[Event]:
        # 3. 그려 둔 차로를 모두 벗어난 뒤 방향이 크게 바뀌면 회전으로 본다.
        #    교차로 안 유도선 칸 안에서 옆 칸으로 옮기는 움직임은 회전으로 오인하기 쉬워 판정하지 않는다.
        if st.turn_done or st.entry is None or st.cur is not None:
            return []
        v_in = self._heading(st.entry.pts)
        recent = [p for t, p in st.pts if t >= t_sec - self.heading_seconds]
        v_now = self._heading(recent)
        if v_in is None or v_now is None:
            return []
        ang = signed_angle(v_in, v_now)
        if abs(ang) < self.turn_degrees:
            return []
        st.turn_done = True
        turn = "right" if ang > 0 else "left"
        lane = st.entry.lane
        if turn in self.turn_checks and lane.turns and turn not in lane.turns:
            allowed = "/".join(lane.turns)
            return self._event(tr, f"{turn}_turn_lane", frame_idx, t_sec,
                               f"from {lane.id} (allowed {allowed}), turn {ang:+.0f}deg")
        return []

    def _event(self, tr: Track, kind: str, frame_idx: int, t_sec: float, detail: str) -> List[Event]:
        key = (tr.track_id, kind)
        if key in self._reported:
            return []
        self._reported.add(key)
        self.violators.add(tr.track_id)
        return [Event(frame_idx=frame_idx, time_sec=t_sec, track_id=tr.track_id, cls_name=tr.cls_name,
                      event_type=kind, bbox=tr.bbox, detail=detail)]
