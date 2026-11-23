import numpy as np

from src.auto_roi import (RoadAccumulator, clean_mask, coverage, group_scenes, mask_to_polygons,
                          refine_with_color, scene_signature, static_boxes)


def _lane_boxes(k):
    """화면 가운데 가로 차로(y 400~600)를 지나가는 차량 박스, 프레임마다 x 위치가 다르다."""
    x = 50 + (k % 13) * 60  # 13프레임 주기로 같은 자리를 다시 지나간다
    return [(x, 450, x + 120, 560)]


def test_moving_vehicles_make_road_polygon():
    acc = RoadAccumulator(1000, 1000, scale=0.25)
    for k in range(40):
        acc.add(_lane_boxes(k))
    assert acc.build(static_frac=0.15) == 0
    mask = clean_mask(acc.mask(2), close_frac=0.03, min_area_frac=0.005)
    polys = mask_to_polygons(mask, acc.scale)
    assert len(polys) == 1
    assert coverage(polys, [(500, 540)]) == 1.0      # 차로 위
    assert coverage(polys, [(500, 100), (500, 900)]) == 0.0  # 차로 밖(보도·건물)


def test_static_box_is_excluded():
    frames = [_lane_boxes(k) + [(800, 100, 900, 200)] for k in range(20)]  # 같은 자리에 계속 있는 박스
    static = static_boxes(frames, static_frac=0.15)
    assert all(bi == 1 for _, bi in static) and len(static) == 20
    acc = RoadAccumulator(1000, 1000, scale=0.25)
    for f in frames:
        acc.add(f)
    assert acc.build(static_frac=0.15) == 20
    polys = mask_to_polygons(clean_mask(acc.mask(2)), acc.scale)
    assert coverage(polys, [(850, 190)]) == 0.0


def test_color_refine_fills_gray_but_not_green():
    bg = np.zeros((200, 200, 3), np.uint8)
    bg[:, :] = (40, 160, 40)            # 잔디
    bg[60:140, :] = (110, 110, 110)     # 아스팔트 띠
    mask = np.zeros((200, 200), np.uint8)
    mask[85:115, 20:180] = 255          # 차량이 지나간 곳은 띠의 가운데뿐
    out = refine_with_color(bg, mask, grow_frac=0.2)
    assert out[65, 100] == 255 and out[130, 100] == 255   # 띠 가장자리까지 보충
    assert out[40, 100] == 0 and out[160, 100] == 0       # 잔디는 제외


def test_group_scenes_splits_different_views():
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 255, (2, 72, 128, 3), dtype=np.uint8)
    frames = [np.clip(a.astype(int) + rng.integers(-10, 10, a.shape), 0, 255).astype(np.uint8) for _ in range(5)]
    frames += [np.clip(b.astype(int) + rng.integers(-10, 10, b.shape), 0, 255).astype(np.uint8) for _ in range(3)]
    groups = group_scenes(np.stack([scene_signature(f) for f in frames]))
    assert [sorted(g) for g in groups] == [[0, 1, 2, 3, 4], [5, 6, 7]]
