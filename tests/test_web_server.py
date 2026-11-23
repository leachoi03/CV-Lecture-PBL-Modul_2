from src.web_server import merge_roi, parse_range, valid_roi


def test_parse_range():
    assert parse_range("bytes=0-99", 1000) == (0, 99)
    assert parse_range("bytes=900-", 1000) == (900, 999)
    assert parse_range("bytes=-100", 1000) == (900, 999)
    assert parse_range("bytes=500-5000", 1000) == (500, 999)
    assert parse_range("bytes=1000-", 1000) is None
    assert parse_range("items=0-1", 1000) is None


def test_valid_roi():
    assert valid_roi({"road": [[[0, 0], [10, 0], [10, 10]]]}) == ""
    assert valid_roi({"road": [[[0, 0], [10, 0]]]})            # 점 2개
    assert valid_roi({"road": [[[0, 0], [10, 0], ["a", 1]]]})  # 좌표가 숫자가 아님
    assert valid_roi([1, 2])


def test_merge_keeps_other_fields():
    old = {"road": [[[0, 0], [1, 0], [1, 1]]], "stop_lines": [{"name": "s"}], "custom": 1}
    new = {"road": [[[5, 5], [6, 5], [6, 6]]]}
    out = merge_roi(old, new)
    assert out["road"] == new["road"] and out["stop_lines"] == old["stop_lines"] and out["custom"] == 1


def _lane(**kw):
    lane = {"id": "N-1", "approach": "N", "index": 1, "zone": "approach", "turns": ["left"],
            "polygon": [[0, 0], [10, 0], [10, 10]]}
    lane.update(kw)
    return lane


def test_valid_lanes():
    assert valid_roi({"road": [], "lanes": [_lane(), _lane(id="N-2", index=2)]}) == ""
    assert valid_roi({"lanes": [_lane(index=0)]})                       # 번호는 1부터
    assert valid_roi({"lanes": [_lane(index=True)]})                    # bool 은 정수로 안 침
    assert valid_roi({"lanes": [_lane(zone="exit")]})                   # 없는 구역
    assert valid_roi({"lanes": [_lane(turns=["back"])]})                # 없는 방향
    assert valid_roi({"lanes": [_lane(approach=" ")]})                  # 진입로 이름 없음
    assert valid_roi({"lanes": [_lane(polygon=[[0, 0], [1, 1]])]})      # 점 2개
    assert "겹칩니다" in valid_roi({"lanes": [_lane(), _lane()]})        # 같은 ID
