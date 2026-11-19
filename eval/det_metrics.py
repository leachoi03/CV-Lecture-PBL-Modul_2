"""프레임 단위 검출 평가용 순수 함수: CVAT XML 파싱, IoU 매칭, AP50, 크기별 재현율."""
import re
from collections import defaultdict
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# AI Hub 094 라벨 → 파이프라인 범주. None 은 '무시 영역'(그 위의 검출은 오검출로 세지 않음)
GT_LABEL_MAP = {
    "car": "vehicle", "small_truck": "vehicle", "truck": "vehicle", "large_truck": "vehicle",
    "bus": "vehicle", "small_bus": "vehicle",
    "bike": "two_wheeler",
    "unknown": None,
}
SIZE_BINS = (("small", 0, 32 ** 2), ("medium", 32 ** 2, 96 ** 2), ("large", 96 ** 2, float("inf")))

Box = Tuple[float, float, float, float]


@dataclass
class GTImage:
    name: str
    width: int
    height: int
    boxes: List[Tuple[str, Box]] = field(default_factory=list)   # (범주, box)
    ignore: List[Box] = field(default_factory=list)


def parse_cvat(path) -> List[GTImage]:
    root = ET.parse(path).getroot()
    out = []
    for im in root.iter("image"):
        g = GTImage(im.get("name"), int(im.get("width")), int(im.get("height")))
        for b in im.iter("box"):
            box = tuple(float(b.get(k)) for k in ("xtl", "ytl", "xbr", "ybr"))
            label = b.get("label")
            if label not in GT_LABEL_MAP:
                continue  # face, car_plate 등 평가 대상이 아닌 라벨
            cat = GT_LABEL_MAP[label]
            (g.ignore if cat is None else g.boxes).append(box if cat is None else (cat, box))
        out.append(g)
    return out


def capture_hour(name: str) -> Optional[int]:
    """'94022_2021110318392086_012.jpg' 처럼 YYYYMMDDHH... 가 들어간 이름에서 촬영 시(hour)."""
    m = re.search(r"_(20\d{2})(\d{2})(\d{2})(\d{2})\d*_", name)
    return int(m.group(4)) if m else None


def day_night(hour: Optional[int], night_from: int = 18, night_until: int = 6) -> str:
    if hour is None:
        return "unknown"
    return "night" if hour >= night_from or hour < night_until else "day"


def camera_id(name: str) -> str:
    """'94022_2021110318392086_012.jpg' → '94022'."""
    return name.split("_", 1)[0]


def label_regions(images: Sequence[GTImage], margin: float = 0.05) -> Dict[str, np.ndarray]:
    """카메라별 '라벨링된 영역' = 그 카메라 모든 GT 박스 꼭짓점의 볼록 껍질 (이미지 크기 비율만큼 여유).

    AI Hub 094 는 화면 일부(주로 진입 차로)의 차량만 라벨링되어 있어, 영역 밖 검출을 FP 로 세면 정밀도가 크게 낮아진다.
    """
    import cv2

    pts = defaultdict(list)
    size = {}
    for g in images:
        cam = camera_id(g.name)
        size[cam] = (g.width, g.height)
        for _, (x1, y1, x2, y2) in g.boxes:
            pts[cam] += [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    out = {}
    for cam, p in pts.items():
        hull = cv2.convexHull(np.array(p, np.float32)).reshape(-1, 2)
        c = hull.mean(axis=0)
        w, h = size[cam]
        d = hull - c
        norm = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        out[cam] = (hull + d / norm * margin * max(w, h)).astype(np.float32)  # 바깥쪽으로 여유
    return out


def in_region(box: Box, region: Optional[np.ndarray]) -> bool:
    if region is None:
        return True
    import cv2

    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return cv2.pointPolygonTest(region.reshape(-1, 1, 2), (float(cx), float(cy)), False) >= 0


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match_image(gt: Sequence[Box], preds: Sequence[Tuple[float, Box]], ignore: Sequence[Box] = (), iou_thr=0.5):
    """한 이미지·한 범주에서 점수 높은 검출부터 GT 에 탐욕 매칭.

    반환: dets [(score, is_tp)] (무시 영역에 걸린 미매칭 검출은 제외), gt_matched [bool]
    """
    order = sorted(range(len(preds)), key=lambda i: -preds[i][0])
    g = np.array(gt, float).reshape(-1, 4)
    p = np.array([preds[i][1] for i in order], float).reshape(-1, 4)
    ious = iou_matrix(p, g)
    ign = iou_matrix(p, np.array(ignore, float).reshape(-1, 4))
    matched = [False] * len(g)
    dets = []
    for k, i in enumerate(order):
        best, j = -1.0, -1
        for jj in range(len(g)):
            if not matched[jj] and ious[k, jj] >= iou_thr and ious[k, jj] > best:
                best, j = ious[k, jj], jj
        if j >= 0:
            matched[j] = True
            dets.append((preds[i][0], True))
        elif ign.shape[1] and ign[k].max() >= iou_thr:
            continue
        else:
            dets.append((preds[i][0], False))
    return dets, matched


def average_precision(dets: List[Tuple[float, bool]], n_gt: int) -> float:
    """VOC 방식(모든 점 보간) AP."""
    if n_gt == 0:
        return float("nan")
    if not dets:
        return 0.0
    dets = sorted(dets, key=lambda d: -d[0])
    tp = np.cumsum([d[1] for d in dets])
    fp = np.cumsum([not d[1] for d in dets])
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def size_bin(box: Box) -> str:
    a = (box[2] - box[0]) * (box[3] - box[1])
    return next(n for n, lo, hi in SIZE_BINS if lo <= a < hi)


class Accumulator:
    """그룹·범주별 누적. conf_thr 는 P/R/F1 을 낼 운영 임계값."""

    def __init__(self, conf_thr: float):
        self.conf_thr = conf_thr
        self.dets: List[Tuple[float, bool]] = []
        self.n_gt = 0
        self.size_total: Dict[str, int] = {n: 0 for n, *_ in SIZE_BINS}
        self.size_hit: Dict[str, int] = {n: 0 for n, *_ in SIZE_BINS}

    def add(self, gt: Sequence[Box], preds, ignore=(), iou_thr=0.5):
        dets, _ = match_image(gt, preds, ignore, iou_thr)
        self.dets += dets
        self.n_gt += len(gt)
        # 크기별 재현율은 운영 임계값 이상의 검출만으로 다시 매칭
        _, matched = match_image(gt, [p for p in preds if p[0] >= self.conf_thr], ignore, iou_thr)
        for b, m in zip(gt, matched):
            s = size_bin(b)
            self.size_total[s] += 1
            self.size_hit[s] += int(m)

    def summary(self) -> dict:
        kept = [d for d in self.dets if d[0] >= self.conf_thr]
        tp = sum(d[1] for d in kept)
        fp = len(kept) - tp
        fn = self.n_gt - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / self.n_gt if self.n_gt else 0.0
        out = {
            "n_gt": self.n_gt, "tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "ap50": average_precision(self.dets, self.n_gt),
        }
        for n, *_ in SIZE_BINS:
            t = self.size_total[n]
            out[f"recall_{n}"] = self.size_hit[n] / t if t else float("nan")
            out[f"n_{n}"] = t
        return out
