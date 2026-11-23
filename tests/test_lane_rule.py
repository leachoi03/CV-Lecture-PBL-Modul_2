"""차로 위반 판정 단위 테스트 (가짜 궤적).

장면: 북쪽 진입로 3개 차로가 세로 띠(y 300~600)이고 차량은 위(y 감소)로 달린다.
1차로가 중앙선 쪽(왼쪽). 그 위(y 100~300)에 좌회전 유도선 칸 2개(X1, X2)가 있다.
"""
from src.roi import ROIConfig
from src.rules.lane import LaneRule, signed_angle
from src.structures import Track

FPS = 10


def rect(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


ROI = ROIConfig.from_dict({"lanes": [
    {"approach": "N", "index": 1, "turns": ["left"], "polygon": rect(0, 300, 100, 600)},
    {"approach": "N", "index": 2, "turns": ["straight"], "polygon": rect(100, 300, 200, 600)},
    {"approach": "N", "index": 3, "turns": ["straight", "right"], "polygon": rect(200, 300, 300, 600)},
    {"approach": "N", "index": 1, "zone": "intersection", "turns": ["left"], "polygon": rect(0, 100, 100, 300)},
    {"approach": "N", "index": 2, "zone": "intersection", "turns": ["left"], "polygon": rect(100, 100, 200, 300)},
    # 다른 진입로의 유도선 칸: 북쪽 차량에게는 무시되어야 한다
    {"approach": "S", "index": 1, "zone": "intersection", "turns": ["left"], "polygon": rect(0, 100, 200, 300)},
]})


def car(x, y, tid=1):
    return Track(tid, 2, "car", "vehicle", 0.9, (x - 20, y - 40, x + 20, y))


def run(path, **kw):
    rule = LaneRule(ROI, FPS, stable_seconds=0.2, **kw)
    events = []
    for i, (x, y) in enumerate(path):
        events += rule.update([car(x, y)], i, i / FPS)
    return events, rule


def line(p0, p1, n):
    return [(p0[0] + (p1[0] - p0[0]) * k / n, p0[1] + (p1[1] - p0[1]) * k / n) for k in range(n + 1)]


def kinds(events):
    return [e.event_type for e in events]


def test_signed_angle_right_is_positive():
    up, right, left = (0, -1), (1, 0), (-1, 0)
    assert signed_angle(up, right) == 90
    assert signed_angle(up, left) == -90


def test_straight_in_lane_no_event():
    ev, rule = run(line((150, 590), (150, 310), 20))
    assert ev == [] and rule.current[1].id == "N-2"


def test_one_lane_change_is_ok():
    ev, _ = run(line((150, 590), (150, 500), 5) + line((150, 500), (250, 400), 5) + line((250, 400), (250, 310), 5))
    assert ev == []


def test_two_lanes_at_once():
    # 1차로 → 3차로를 1초 만에 가로지름 (2차로는 잠깐 지나감)
    ev, _ = run(line((50, 590), (50, 520), 4) + line((50, 520), (250, 450), 10) + line((250, 450), (250, 320), 6))
    assert kinds(ev) == ["multi_lane"]
    assert "N-1->N-3" in ev[0].detail


def test_two_lanes_slowly_is_ok():
    # 한 차로씩, 각 차로에서 4초씩 머문 뒤 바꾸면 위반 아님 (window 3초)
    path = line((50, 590), (50, 580), 40) + line((50, 580), (150, 560), 3) + line((150, 560), (150, 540), 40) \
        + line((150, 540), (250, 520), 3) + line((250, 520), (250, 500), 5)
    ev, _ = run(path)
    assert ev == []


def test_lane_change_in_left_turn():
    # 1차로로 들어와 교차로 안 X1 → X2 로 옮김
    ev, _ = run(line((50, 450), (50, 250), 8) + line((50, 250), (150, 200), 5) + line((150, 200), (150, 120), 4))
    assert kinds(ev) == ["left_turn_lane_change"]
    assert ev[0].detail == "N-X1->N-X2"


def test_left_turn_keep_guide_lane_ok():
    ev, _ = run(line((50, 450), (50, 120), 15))
    assert ev == []


def test_right_turn_from_straight_lane():
    # 2차로(직진 전용)에서 올라가다 오른쪽으로 꺾음
    ev, _ = run(line((150, 590), (150, 320), 10) + line((150, 320), (170, 250), 3) + line((170, 250), (450, 230), 10))
    assert kinds(ev) == ["right_turn_lane"]
    assert "N-2" in ev[0].detail


def test_right_turn_from_right_lane_ok():
    ev, _ = run(line((250, 590), (250, 320), 10) + line((250, 320), (270, 250), 3) + line((270, 250), (550, 230), 10))
    assert ev == []


def test_left_turn_not_checked_by_default():
    # 2차로에서 좌회전: 기본 설정(turn_checks=[right])에서는 판정하지 않음, left 를 켜면 위반
    path = line((150, 590), (150, 80), 15) + line((150, 80), (-200, 60), 10)
    assert run(path)[0] == []
    assert kinds(run(path, turn_checks=["right", "left"])[0]) == ["left_turn_lane"]


def test_boundary_jitter_not_lane_change():
    # 1·2차로 경계(x=100)에서 한 프레임씩 흔들리면 확정되지 않는다
    path = [(95 if k % 2 else 105, 590 - 10 * k) for k in range(25)]
    ev, _ = run([(50, 600)] * 3 + path)
    assert ev == []
