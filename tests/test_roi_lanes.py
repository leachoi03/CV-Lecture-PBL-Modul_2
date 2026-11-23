from src.roi import ROIConfig


def _cfg():
    return ROIConfig.from_dict({
        "road": [[[0, 0], [300, 0], [300, 300], [0, 300]]],
        "lanes": [
            {"approach": "N", "index": 1, "turns": ["left"], "polygon": [[0, 0], [100, 0], [100, 200], [0, 200]]},
            {"approach": "N", "index": 2, "turns": ["straight"], "polygon": [[100, 0], [200, 0], [200, 200], [100, 200]]},
            # 교차로 안 좌회전 유도선 칸 (진입부 1차로와 일부 겹침)
            {"id": "N-L1", "approach": "N", "index": 1, "zone": "intersection", "turns": ["left"],
             "polygon": [[0, 150], [60, 150], [60, 300], [0, 300]]},
        ],
    })


def test_lane_parse_and_default_id():
    cfg = _cfg()
    assert [ln.id for ln in cfg.lanes] == ["N-1", "N-2", "N-L1"]
    assert cfg.lanes[0].zone == "approach" and cfg.lanes[0].turns == ("left",)


def test_lane_at():
    cfg = _cfg()
    assert cfg.lane_at((50, 50)).id == "N-1"
    assert cfg.lane_at((150, 50)).id == "N-2"
    assert cfg.lane_at((30, 180)).id == "N-L1"                       # 겹치면 더 작은 칸
    assert cfg.lane_at((30, 180), zone="approach").id == "N-1"
    assert cfg.lane_at((250, 250)) is None


def test_old_file_without_lanes():
    assert ROIConfig.from_dict({"road": [[[0, 0], [1, 0], [1, 1]]]}).lanes == []
