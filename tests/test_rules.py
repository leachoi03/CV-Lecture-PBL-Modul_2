"""위반 판정 로직 단위 테스트 (YOLO 없이 가짜 트랙으로 검증).

    python -m pytest tests -q
"""
from src.roi import ROIConfig, StopLine
from src.rules import JaywalkRule, StopLineRule
from src.structures import Track

# 정지선: y=500 가로선, 차량은 아래(y>500)에서 대기하고 위로 진행
LINE = StopLine("s", (100, 500), (700, 500), approach_point=(400, 600))


def car(tid, foot_y, x=400, cat="vehicle", name="car"):
    return Track(tid, 2, name, cat, 0.9, (x - 40, foot_y - 60, x + 40, foot_y))


def person(tid, x, y):
    return Track(tid, 0, "person", "person", 0.9, (x - 15, y - 80, x + 15, y))


def run_stop(ys, signals, confirm=2):
    rule = StopLineRule([LINE], confirm_frames=confirm)
    events = []
    for i, (y, s) in enumerate(zip(ys, signals)):
        events += rule.update([car(1, y)], i, i / 30, s)
    return events


def test_stopline_side():
    assert LINE.side((400, 600)) == 1
    assert LINE.side((400, 450)) == -1
    assert LINE.side((2000, 450)) == 0  # 선분 범위 밖


def test_red_crossing_is_violation():
    ev = run_stop([560, 520, 490, 470, 450], ["red"] * 5)
    assert len(ev) == 1 and ev[0].event_type == "stopline"


def test_green_crossing_is_not_violation():
    assert run_stop([560, 520, 490, 470, 450], ["green"] * 5) == []


def test_already_past_when_red_starts():
    # 녹색에 넘어간 뒤 적색이 켜져도 위반 아님
    assert run_stop([520, 490, 470, 450, 430], ["green", "green", "red", "red", "red"]) == []


def test_jitter_on_line_not_confirmed():
    # 넘었다가 바로 돌아오면 확정되지 않음
    assert run_stop([520, 498, 503, 520, 530], ["red"] * 5, confirm=2) == []


def test_reported_once():
    ev = run_stop([560, 490, 470, 450, 430, 410], ["red"] * 6)
    assert len(ev) == 1


ROI = ROIConfig.from_dict(
    {
        "road": [[[0, 300], [800, 300], [800, 700], [0, 700]]],
        "crosswalks": [[[300, 300], [500, 300], [500, 700], [300, 700]]],
    }
)


def test_jaywalk_outside_crosswalk():
    rule = JaywalkRule(ROI, min_frames=3)
    ev = []
    for i in range(5):
        ev += rule.update([person(7, 100 + 5 * i, 500)], i, i / 30)
    assert len(ev) == 1 and ev[0].track_id == 7


def test_crosswalk_is_not_violation():
    rule = JaywalkRule(ROI, min_frames=3)
    ev = []
    for i in range(5):
        ev += rule.update([person(7, 400, 400 + 10 * i)], i, i / 30)
    assert ev == []


def test_rider_excluded():
    rule = JaywalkRule(ROI, min_frames=2)
    bike = Track(9, 3, "motorcycle", "two_wheeler", 0.9, (80, 440, 130, 520))
    ev = []
    for i in range(4):
        ev += rule.update([person(8, 105, 480), bike], i, i / 30)
    assert ev == []
