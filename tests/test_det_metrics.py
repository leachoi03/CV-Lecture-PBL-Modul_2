import math

from eval.det_metrics import Accumulator, average_precision, capture_hour, day_night, match_image, parse_cvat

XML = """<?xml version='1.0' encoding='utf-8'?>
<annotations>
  <image id="0" name="94022_2021110320392086_012.jpg" width="1920" height="1080">
    <box label="car" occluded="0" xtl="0" ytl="0" xbr="100" ybr="100"></box>
    <box label="small_truck" occluded="0" xtl="200" ytl="0" xbr="300" ybr="100"></box>
    <box label="bike" occluded="0" xtl="400" ytl="0" xbr="420" ybr="20"></box>
    <box label="unknown" occluded="0" xtl="600" ytl="0" xbr="700" ybr="100"></box>
    <box label="face" occluded="0" xtl="800" ytl="0" xbr="810" ybr="10"></box>
  </image>
</annotations>"""


def test_parse_cvat_maps_labels(tmp_path):
    p = tmp_path / "t.xml"
    p.write_text(XML, encoding="utf-8")
    (g,) = parse_cvat(p)
    assert [c for c, _ in g.boxes] == ["vehicle", "vehicle", "two_wheeler"]
    assert len(g.ignore) == 1  # unknown → 무시 영역, face 는 버림


def test_capture_hour_and_day_night():
    assert capture_hour("94022_2021110320392086_012.jpg") == 20
    assert day_night(20) == "night" and day_night(9) == "day" and day_night(None) == "unknown"


def test_match_greedy_and_ignore():
    gt = [(0, 0, 100, 100)]
    preds = [(0.9, (2, 2, 100, 100)),      # TP
             (0.8, (0, 0, 98, 98)),        # 같은 GT 중복 → FP
             (0.7, (600, 0, 700, 100))]    # 무시 영역 → 제외
    dets, matched = match_image(gt, preds, ignore=[(600, 0, 700, 100)])
    assert dets == [(0.9, True), (0.8, False)] and matched == [True]


def test_average_precision():
    assert average_precision([(0.9, True), (0.8, True)], 2) == 1.0
    assert abs(average_precision([(0.9, True), (0.8, False)], 2) - 0.5) < 1e-9
    assert math.isnan(average_precision([], 0))


def test_accumulator_threshold_and_sizes():
    acc = Accumulator(conf_thr=0.5)
    acc.add([(0, 0, 10, 10), (0, 0, 200, 200)], [(0.9, (0, 0, 200, 200)), (0.3, (0, 0, 10, 10))])
    s = acc.summary()
    assert s["tp"] == 1 and s["fn"] == 1 and s["recall"] == 0.5 and s["precision"] == 1.0
    assert s["ap50"] == 1.0  # 낮은 점수 검출까지 포함하면 둘 다 맞음
    assert s["recall_small"] == 0.0 and s["recall_large"] == 1.0


def test_label_region_filters_outside():
    from eval.det_metrics import GTImage, in_region, label_regions

    g = GTImage("cam1_2021110310000000_001.jpg", 1000, 1000, [("vehicle", (100, 100, 200, 200)), ("vehicle", (300, 300, 400, 400))])
    region = label_regions([g], margin=0.0)["cam1"]
    assert in_region((150, 150, 250, 250), region)
    assert not in_region((800, 800, 900, 900), region)
