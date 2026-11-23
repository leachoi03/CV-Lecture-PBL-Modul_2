"""자동 ROI 1단계: 차량이 지나간 자리를 누적해 차도(road) 다각형을 만든다.

고정 CCTV에서는 차량 바퀴가 닿는 곳이 곧 차도다. 프레임마다 YOLO로 차량·이륜차를 검출하고
BBox 아래쪽 띠(지면에 가까운 부분)를 누적 지도에 더한 뒤, 여러 번 지나간 곳만 남겨
모폴로지 닫기 → 구멍 메우기 → 작은 조각 제거 → 다각형 근사 순서로 차도 영역을 얻는다.
보도·건물·하늘처럼 차량이 지나가지 않는 곳은 자연히 빠진다. 추적이 필요 없어서
AI Hub처럼 띄엄띄엄한 이미지 묶음에서도 같은 카메라 이미지만 모으면 쓸 수 있다.

예)
    python -m src.auto_roi --source data/videos/day01.mp4 --out configs/roi/day01.json --stride 5
    python -m src.auto_roi --source "data/094/.../VS_4.바운딩박스" --camera 84011 --out configs/roi/auto_84011.json

결과: --out JSON 의 "road" 만 바꾸고 정지선·횡단보도·신호 설정은 그대로 둔다.
      같은 폴더에 <이름>_auto_road.jpg(확인용 그림), <이름>_background.jpg(차량을 지운 중앙값 배경) 저장.
"""
import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import load_config
from .frame_source import FrameSource, imread_any, list_images

BBox = Tuple[float, float, float, float]
DEFAULT_SIGNAL = {"mode": "timeline", "cycle": None, "timeline": [{"start": 0.0, "end": 9999.0, "state": "red"}]}


# ---------------------------------------------------------------- 누적 지도 → 다각형
class RoadAccumulator:
    """검출 박스의 지면 쪽 띠를 축소 해상도 지도에 누적한다."""

    def __init__(self, width: int, height: int, scale: float = 0.25, bottom_frac: float = 0.35, shrink_x: float = 0.1):
        self.width, self.height, self.scale = width, height, scale
        self.bottom_frac, self.shrink_x = bottom_frac, shrink_x
        self.hits = np.zeros((max(1, round(height * scale)), max(1, round(width * scale))), np.float32)
        self.n_frames = 0
        self.n_boxes = 0
        self.frames: List[List[BBox]] = []

    def footprint(self, b: BBox) -> Optional[Tuple[int, int, int, int]]:
        x1, y1, x2, y2 = b
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return None
        fx1, fx2 = x1 + w * self.shrink_x, x2 - w * self.shrink_x
        fy1, fy2 = y2 - h * self.bottom_frac, y2
        s = self.scale
        hh, ww = self.hits.shape
        c = lambda v, hi: int(np.clip(round(v * s), 0, hi))
        r = (c(fx1, ww), c(fy1, hh), c(fx2, ww), c(fy2, hh))
        return r if r[2] > r[0] and r[3] > r[1] else None

    def add(self, boxes: Iterable[BBox]):
        self.frames.append([tuple(b) for b in boxes])
        self.n_boxes += len(self.frames[-1])
        self.n_frames += 1

    def build(self, static_frac: float = 0.15, static_iou: float = 0.6) -> int:
        """누적 지도를 만든다. 여러 프레임에서 거의 같은 자리에 계속 있는 박스(주차 차량, 자전거 거치대,
        노면 표시 오검출)는 '지나간' 흔적이 아니므로 뺀다. 제외한 박스 수를 돌려준다."""
        static = static_boxes(self.frames, static_frac, static_iou)
        self.hits[:] = 0
        removed = 0
        for fi, boxes in enumerate(self.frames):
            frame = np.zeros_like(self.hits)
            for bi, b in enumerate(boxes):
                if (fi, bi) in static:
                    removed += 1
                    continue
                r = self.footprint(b)
                if r is not None:
                    frame[r[1]:r[3], r[0]:r[2]] = 1.0  # 한 프레임 안에서 겹친 박스는 1번만 센다
            self.hits += frame
        return removed

    def mask(self, min_hits: int = 2) -> np.ndarray:
        return (self.hits >= min_hits).astype(np.uint8) * 255


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (4,), b: (N,4) → (N,)"""
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[..., 2] - r[..., 0]) * (r[..., 3] - r[..., 1])
    return inter / np.maximum(area(a) + area(b) - inter, 1e-9)


def static_boxes(frames: Sequence[Sequence[BBox]], static_frac: float = 0.15, iou_thr: float = 0.6) -> set:
    """전체 프레임의 static_frac 이상에서 IoU iou_thr 이상으로 같은 자리에 나오는 박스의 (프레임, 박스) 번호."""
    flat = [(fi, bi, b) for fi, bs in enumerate(frames) for bi, b in enumerate(bs)]
    if not flat or static_frac <= 0:
        return set()
    arr = np.asarray([b for _, _, b in flat], np.float32)
    fidx = np.asarray([fi for fi, _, _ in flat])
    need = max(2, int(np.ceil(static_frac * len(frames))))
    out = set()
    for k, (fi, bi, _) in enumerate(flat):
        same = _iou(arr[k], arr) >= iou_thr
        if len(np.unique(fidx[same])) >= need:
            out.add((fi, bi))
    return out


def clean_mask(mask: np.ndarray, close_frac: float = 0.03, min_area_frac: float = 0.005) -> np.ndarray:
    """차선 사이 틈을 닫고, 내부 구멍을 메우고, 작은 조각(오검출 흔적)을 지운다."""
    h, w = mask.shape
    k = max(3, int(round(max(h, w) * close_frac)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    min_area = min_area_frac * h * w
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(out, [c], -1, 255, cv2.FILLED)  # 외곽선 안을 채우면 구멍도 메워진다
    return out


def refine_with_color(bg: np.ndarray, mask: np.ndarray, grow_frac: float = 0.05, pct: float = 90.0) -> np.ndarray:
    """통행 흔적이 드문 차도 가장자리·교차로 안쪽을 배경 색으로 보충한다.

    통행 마스크 안쪽 화소로 아스팔트 색 분포(Lab 평균·공분산)를 구하고, 마스크 주변 grow_frac 거리 안에서
    색이 비슷하고(마할라노비스 거리가 마스크 안쪽 화소의 pct 백분위 이하) 마스크와 이어진 화소만 더한다.
    잔디·나무·적색 자전거도로는 색이 달라 빠지고, 거리 제한으로 멀리 있는 회색 보도까지 번지는 것을 막는다."""
    h, w = mask.shape
    lab = cv2.cvtColor(cv2.resize(bg, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)
    seed = cv2.erode(mask, np.ones((3, 3), np.uint8)) > 0
    if seed.sum() < 50:
        return mask
    x = lab[seed.ravel()]
    mu, cov = x.mean(0), np.cov(x.T) + np.eye(3) * 1e-3
    inv = np.linalg.inv(cov)
    dlt = lab - mu
    d = np.sqrt(np.einsum("ij,jk,ik->i", dlt, inv, dlt)).reshape(h, w)
    thr = np.percentile(d[seed], pct)
    k = max(3, int(round(max(h, w) * grow_frac)) | 1)
    near = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    cand = ((d <= thr) & near) | (mask > 0)
    n, lbl = cv2.connectedComponents(cand.astype(np.uint8))
    keep = np.unique(lbl[mask > 0])
    return (np.isin(lbl, keep[keep > 0]) * 255).astype(np.uint8)


def mask_to_polygons(mask: np.ndarray, scale: float = 1.0, eps_frac: float = 0.005) -> List[List[List[int]]]:
    """마스크 외곽선을 단순화한 다각형 목록(원본 해상도 좌표)으로 바꾼다. 큰 것부터."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    eps = eps_frac * max(mask.shape)
    polys = []
    for c in contours:
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(approx) >= 3:
            polys.append([[int(round(x / scale)), int(round(y / scale))] for x, y in approx])
    return polys


def coverage(polys: Sequence[Sequence[Sequence[int]]], points: Sequence[Tuple[float, float]]) -> float:
    """점(예: 정답 차량의 지면 접점)이 다각형 안에 들어간 비율."""
    if not points:
        return float("nan")
    cs = [np.asarray(p, np.float32).reshape(-1, 1, 2) for p in polys]
    inside = sum(any(cv2.pointPolygonTest(c, (float(x), float(y)), False) >= 0 for c in cs) for x, y in points)
    return inside / len(points)


def scene_signature(img: np.ndarray) -> np.ndarray:
    """장면 비교용 특징: 64x36 회색조 축소 영상의 경계 세기(밝기·조명 변화에 덜 민감), 정규화."""
    g = cv2.cvtColor(cv2.resize(img, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
    e = np.sqrt(cv2.Sobel(g, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(g, cv2.CV_32F, 0, 1) ** 2).ravel()
    e -= e.mean()
    return e / (np.linalg.norm(e) + 1e-9)


def group_scenes(sigs: np.ndarray, thr: float = 0.4) -> List[List[int]]:
    """특징 상관이 thr 를 넘는 프레임끼리 이어 붙인 묶음(단일 연결). 큰 묶음부터.
    AI Hub 는 같은 카메라 번호에 다른 화각(회전·확대)이 섞여 있어서, 묶음 하나만 골라 써야 한다."""
    n = len(sigs)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    corr = sigs @ sigs.T
    for i, j in zip(*np.where(np.triu(corr > thr, 1))):
        parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=len, reverse=True)


# ---------------------------------------------------------------- 입력
def iter_frames(source: str, camera: Optional[str], stride: int, max_frames: int, scene: Optional[int] = 0):
    """(이름, 프레임) 생성. 폴더는 하위 폴더까지 읽고 --camera 로 파일 이름 앞부분을 거른다.
    scene 이 None 이 아니면 폴더 이미지를 장면별로 묶어 scene 번째(0=가장 큰) 장면만 쓴다."""
    src = Path(source)
    if src.is_dir():
        files = list_images(src, recursive=True)
        if camera:
            files = [f for f in files if f.name.startswith(f"{camera}_")]
        if scene is not None and len(files) > 1:
            ok = [(f, imread_any(f)) for f in files]
            ok = [(f, im) for f, im in ok if im is not None]
            groups = group_scenes(np.stack([scene_signature(im) for _, im in ok]))
            print(f"장면 묶음 {len(groups)}개: {[len(g) for g in groups[:8]]} → {scene}번 장면({len(groups[min(scene, len(groups) - 1)])}장) 사용")
            files = [ok[i][0] for i in sorted(groups[min(scene, len(groups) - 1)])]
        files = files[::stride][:max_frames] if max_frames > 0 else files[::stride]
        for f in files:
            img = imread_any(f)
            if img is not None:
                yield f.name, img
        return
    cap = FrameSource(source)
    i = n = 0
    while max_frames <= 0 or n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            yield f"frame{i:06d}", frame
            n += 1
        i += 1
    cap.release()


def median_background(frames: List[np.ndarray]) -> np.ndarray:
    """여러 프레임의 화소별 중앙값: 움직이는 차량이 지워진 빈 도로 영상."""
    h, w = frames[0].shape[:2]
    stack = np.stack([f if f.shape[:2] == (h, w) else cv2.resize(f, (w, h)) for f in frames])
    return np.median(stack, axis=0).astype(np.uint8)


def draw_result(bg: np.ndarray, hits: np.ndarray, polys, cov: Optional[float] = None) -> np.ndarray:
    h, w = bg.shape[:2]
    heat = cv2.resize(hits / max(1.0, hits.max()), (w, h))
    heat = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    vis = cv2.addWeighted(bg, 0.65, heat, 0.35, 0)
    overlay = vis.copy()
    for p in polys:
        cv2.fillPoly(overlay, [np.array(p, np.int32)], (0, 200, 0))
    vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
    for p in polys:
        cv2.polylines(vis, [np.array(p, np.int32)], True, (0, 255, 0), 3)
    text = f"auto road ROI: {len(polys)} polygon(s)" + (f", GT footprint coverage {cov:.1%}" if cov == cov and cov is not None else "")
    cv2.putText(vis, text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return vis


def imwrite_any(path: Path, img: np.ndarray):
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if ok:
        buf.tofile(str(path))


def gt_footprints(labels_root: Path, names: set) -> List[Tuple[float, float]]:
    """AI Hub CVAT XML 에서 해당 이미지들의 차량·이륜차 지면 접점(BBox 하단 중앙)."""
    from eval.det_metrics import parse_cvat

    pts = []
    for xml in labels_root.rglob("*.xml"):
        for g in parse_cvat(xml):
            if g.name in names:
                pts += [((b[0] + b[2]) / 2, b[3]) for _, b in g.boxes]
    return pts


def detect_all(args):
    """프레임마다 차량을 검출해 누적기에 넣는다. (누적기, 프레임 이름 집합, 중앙값 배경) 반환."""
    from ultralytics import YOLO

    cfg = load_config(args.config)
    # 자전거(COCO 1)는 보도·자전거도로로도 다니므로 빼고, 차량과 오토바이(3)만 쓴다
    classes = sorted(set(cfg["categories"]["vehicle"]) | {3})
    model = YOLO(args.model or cfg["model"])
    acc, bg_frames, names = None, [], set()
    for k, (name, frame) in enumerate(iter_frames(args.source, args.camera, max(1, args.stride), args.max_frames,
                                                  None if args.scene < 0 else args.scene)):
        if acc is None:
            acc = RoadAccumulator(frame.shape[1], frame.shape[0], bottom_frac=args.bottom_frac)
        elif frame.shape[:2] != (acc.height, acc.width):
            frame = cv2.resize(frame, (acc.width, acc.height))
        r = model.predict(frame, conf=args.conf, iou=cfg["iou"], imgsz=args.imgsz, classes=classes,
                          device=cfg.get("device"), verbose=False)[0]
        acc.add(r.boxes.xyxy.cpu().tolist())
        names.add(name)
        if len(bg_frames) < 60 and k % 2 == 0:
            bg_frames.append(frame)
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}장 처리, 누적 박스 {acc.n_boxes}")
    if acc is None:
        raise SystemExit("읽은 프레임이 없습니다. --source/--camera 를 확인하세요.")
    return acc, names, median_background(bg_frames)


# ---------------------------------------------------------------- 실행
def main():
    ap = argparse.ArgumentParser(description="차량 통행 누적으로 차도 ROI 자동 생성")
    ap.add_argument("--source", required=True, help="영상 파일 또는 이미지 폴더(하위 폴더 포함)")
    ap.add_argument("--camera", default=None, help="폴더 입력일 때 파일 이름 앞부분(카메라 ID)으로 거르기, 예: 84011")
    ap.add_argument("--out", required=True, help="ROI JSON. 있으면 road 만 바꾼다")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--imgsz", type=int, default=1280, help="먼 곳 작은 차량까지 잡도록 기본 1280")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--scene", type=int, default=0, help="폴더 입력: 화각이 같은 이미지끼리 묶어 N번째(0=가장 큰) 묶음만 사용, -1=묶지 않음")
    ap.add_argument("--stride", type=int, default=1, help="영상에서 N프레임마다 1장 사용")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--min-hits", type=int, default=2, help="이 횟수 이상 차량이 지나간 칸만 차도로 본다 (최솟값)")
    ap.add_argument("--min-hits-frac", type=float, default=0.01, help="최소 통과 횟수를 프레임 수의 이 비율 이상으로")
    ap.add_argument("--static-frac", type=float, default=0.15, help="이 비율 이상 프레임에서 같은 자리에 있는 박스는 정지 물체로 제외 (0=끔)")
    ap.add_argument("--bottom-frac", type=float, default=0.35, help="BBox 아래쪽 몇 %%를 지면 접촉부로 볼지")
    ap.add_argument("--close", type=float, default=0.02, help="모폴로지 닫기 커널 크기(화면 긴 변 대비)")
    ap.add_argument("--min-area", type=float, default=0.005, help="이보다 작은 조각 제거(화면 넓이 대비)")
    ap.add_argument("--no-color", action="store_true", help="배경 색으로 차도를 보충하는 단계를 끈다")
    ap.add_argument("--grow", type=float, default=0.05, help="색 보충을 허용할 거리(화면 긴 변 대비)")
    ap.add_argument("--reuse-dets", action="store_true", help="이전 실행의 <이름>_dets.json 검출 결과를 다시 써서 파라미터만 바꿔 본다")
    ap.add_argument("--labels", default=None, help="(선택) AI Hub CVAT XML 폴더: 정답 차량 지면 접점이 ROI 안에 드는 비율 출력")
    args = ap.parse_args()

    out = Path(args.out)
    dets_path = out.with_name(out.stem + "_dets.json")
    bg_path = out.with_name(out.stem + "_background.jpg")
    if args.reuse_dets:
        with open(dets_path, encoding="utf-8") as f:
            d = json.load(f)
        acc = RoadAccumulator(d["width"], d["height"], bottom_frac=args.bottom_frac)
        for boxes in d["frames"]:
            acc.add(boxes)
        names, bg = set(d["names"]), imread_any(bg_path)
    else:
        acc, names, bg = detect_all(args)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(dets_path, "w", encoding="utf-8") as f:
            json.dump({"width": acc.width, "height": acc.height, "names": sorted(names),
                       "frames": [[[round(v, 1) for v in b] for b in bs] for bs in acc.frames]}, f)
        imwrite_any(bg_path, bg)
    if acc.n_boxes == 0:
        raise SystemExit("차량이 검출된 프레임이 없습니다. --source/--camera 를 확인하세요.")

    removed = acc.build(args.static_frac)
    min_hits = max(args.min_hits, int(np.ceil(args.min_hits_frac * acc.n_frames)))
    mask = clean_mask(acc.mask(min_hits), args.close, args.min_area)
    if not args.no_color and bg is not None:
        mask = clean_mask(refine_with_color(bg, mask, args.grow), args.close, args.min_area)
    polys = mask_to_polygons(mask, acc.scale)
    cov = coverage(polys, gt_footprints(Path(args.labels), names)) if args.labels else None

    data = {"stop_lines": [], "road": [], "crosswalks": [], "signal": dict(DEFAULT_SIGNAL)}
    if out.exists():
        with open(out, encoding="utf-8") as f:
            data.update(json.load(f))
    data["road"] = polys
    data["_auto_road"] = {"method": "vehicle_footprint_accumulation", "source": str(args.source), "camera": args.camera, "scene": args.scene,
                          "frames": acc.n_frames, "boxes": acc.n_boxes, "min_hits": min_hits, "static_removed": removed,
                          "bottom_frac": args.bottom_frac, "color_refine": not args.no_color, "imgsz": args.imgsz, "gt_coverage": cov}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    imwrite_any(out.with_name(out.stem + "_auto_road.jpg"), draw_result(bg, acc.hits, polys, cov))
    road_frac = mask.mean() / 255
    print(f"프레임 {acc.n_frames}장, 차량 박스 {acc.n_boxes}개 (정지 물체로 제외 {removed}개, 최소 통과 {min_hits}회) → 차도 다각형 {len(polys)}개 (화면의 {road_frac:.1%})")
    if cov is not None:
        print(f"정답 차량 지면 접점 중 ROI 안: {cov:.1%}")
    print(f"저장됨: {out}  (확인용 그림: {out.stem}_auto_road.jpg)")


if __name__ == "__main__":
    main()
