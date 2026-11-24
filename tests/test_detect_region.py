import numpy as np

from src.detect_region import PAD_GRAY, DetectRegion
from src.roi import ROIConfig
from src.structures import Track

W, H = 400, 300
# 도로: x 100~300, y 150~280 사각형
ROI = ROIConfig.from_dict({"road": [[[100, 150], [300, 150], [300, 280], [100, 280]]]})
CFG = {"margin_px": 0, "up_frac": 0.1, "foot_tolerance_px": 5}


def car(x, y):
    return Track(1, 2, "car", "vehicle", 0.9, (x - 20, y - 40, x + 20, y))


def test_rect_extends_up_not_down():
    r = DetectRegion.from_roi(ROI, W, H, CFG)
    x0, y0, x1, y1 = r.rect
    assert (x0, x1) == (100, 301)
    assert y0 == 150 - 30          # 위로 화면 높이의 10% (30px) 넓힘
    assert y1 == 281               # 아래로는 넓히지 않음
    assert 0 < r.area_ratio < 0.5


def test_prepare_crops_and_masks():
    r = DetectRegion.from_roi(ROI, W, H, {**CFG, "up_frac": 0})
    frame = np.full((H, W, 3), 255, np.uint8)
    # ROI 가 사각형이 아닌 경우를 보려고 삼각형 도로로 다시 만든다
    tri = ROIConfig.from_dict({"road": [[[100, 280], [300, 280], [300, 150]]]})
    r = DetectRegion.from_roi(tri, W, H, {**CFG, "up_frac": 0})
    img, off = r.prepare(frame)
    assert off == (100, 150) and img.shape[:2] == (131, 201)
    assert (img[0, 0] == PAD_GRAY).all()        # 삼각형 밖(왼쪽 위)은 가려짐
    assert (img[-2, -2] == 255).all()           # 삼각형 안은 그대로
    assert (frame == 255).all()                 # 원본은 바뀌지 않음


def test_crop_only_without_mask():
    r = DetectRegion.from_roi(ROI, W, H, {**CFG, "mask": False})
    frame = np.full((H, W, 3), 255, np.uint8)
    img, _ = r.prepare(frame)
    assert (img == 255).all()


def test_filter_by_foot_point():
    r = DetectRegion.from_roi(ROI, W, H, CFG)
    inside, edge, outside = car(200, 200), car(200, 284), car(50, 200)
    assert r.filter([inside, edge, outside]) == [inside, edge]   # 경계에서 4px 밖은 여유(5px) 안이라 남김


def test_no_polygons_means_full_frame():
    r = DetectRegion.from_roi(ROIConfig.from_dict({}), W, H, CFG)
    frame = np.zeros((H, W, 3), np.uint8)
    img, off = r.prepare(frame)
    assert not r.active and img is frame and off == (0, 0) and r.filter([car(5, 5)]) == [car(5, 5)]


def test_disabled():
    assert not DetectRegion.from_roi(ROI, W, H, {**CFG, "enabled": False}).active


def test_lanes_count_as_roi():
    roi = ROIConfig.from_dict({"lanes": [{"approach": "N", "index": 1, "polygon": [[10, 10], [50, 10], [50, 60]]}]})
    r = DetectRegion.from_roi(roi, W, H, CFG)
    assert r.active and r.rect[0] == 10


def test_imgsz_keep_scale():
    r = DetectRegion.from_roi(ROI, W, H, {**CFG, "keep_scale": True})   # 잘라낸 사각형 긴 변 201 / 원본 400
    assert r.imgsz(640) == 320
    assert DetectRegion.from_roi(ROI, W, H, CFG).imgsz(640) == 640        # 기본값: 확대
    assert DetectRegion.from_roi(ROIConfig.from_dict({}), W, H, CFG).imgsz(640) == 640
