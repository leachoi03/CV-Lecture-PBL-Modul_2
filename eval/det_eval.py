"""프레임 단위 검출 평가 (추적 없음): AI Hub 094 VS_4 이미지 + VL_4 CVAT XML 정답.

    python -m eval.det_eval \
        --images "data/094/01.데이터/2.Validation/원천데이터/VS_4.바운딩박스" \
        --labels "data/094/01.데이터/2.Validation/라벨링데이터/VL_4.바운딩박스" \
        --models yolov8n.pt yolov8s.pt --imgsz 640 1280 --per-group 60

범주: vehicle(car·truck·bus 계열 전부), two_wheeler(bike). 'unknown' 라벨은 무시 영역.
영역: 카메라별로 GT 박스가 있는 영역(볼록 껍질) 밖의 검출은 제외 (--region none 으로 끔). 라벨이 화면 일부에만 있기 때문.
그룹: all / 폴더 조건(1.교차로·2.악천후·3.시간대) / 주간·야간(파일 이름의 촬영 시각 기준).
주의: 영역 안에서도 라벨이 빠진 차량이 많아 정밀도·AP 는 실제보다 낮게 나온다(하한). 재현율 위주로 해석한다.
지표: AP50, 운영 임계값(--conf)에서의 P/R/F1, 객체 크기별(small<32², medium<96², large) 재현율, FPS.
"""
import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2

from eval.det_metrics import Accumulator, camera_id, capture_hour, day_night, in_region, label_regions, parse_cvat
from src.config import load_config
from src.frame_source import imread_any
from src.preprocess import apply_clahe

PRED_CONF = 0.05  # AP 계산용으로 낮은 임계값까지 검출을 받아 둔다


def load_dataset(images_root: Path, labels_root: Path):
    jpgs = {}
    for p in images_root.rglob("*.jpg"):
        jpgs.setdefault(p.name, p)
    items, missing, all_gt = [], 0, []
    for xml in sorted(labels_root.rglob("*.xml")):
        for g in parse_cvat(xml):
            all_gt.append(g)
            img = jpgs.get(g.name)
            if img is None:
                missing += 1
                continue
            cond = img.relative_to(images_root).parts[0]
            items.append({"gt": g, "path": img, "condition": cond, "daynight": day_night(capture_hour(g.name))})
    if missing:
        print(f"이미지 파일을 찾지 못한 라벨 {missing}개는 건너뜀")
    return items, label_regions(all_gt)


def sample(items, per_group: int, seed: int):
    if per_group <= 0:
        return items
    rng = random.Random(seed)
    by = defaultdict(list)
    for it in items:
        by[it["condition"]].append(it)
    out = []
    for cond in sorted(by):
        group = by[cond]
        out += group if len(group) <= per_group else rng.sample(group, per_group)
    return out


def draw(img, gt, preds, conf_thr, region=None):
    vis = img.copy()
    if region is not None:
        cv2.polylines(vis, [region.astype("int32")], True, (0, 255, 255), 2)
    for _, b in gt.boxes:
        cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
    for b in gt.ignore:
        cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (128, 128, 128), 1)
    for cat, items in preds.items():
        for s, b in items:
            if s >= conf_thr:
                cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 128, 0), 2)
                cv2.putText(vis, f"{cat[:3]} {s:.2f}", (int(b[0]), int(b[1]) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)
    cv2.putText(vis, "GT green / pred blue / labeled region yellow", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return vis


def eval_one(items, model_path, imgsz, cfg, conf_thr, clahe, regions=None, vis_dir=None, n_vis=0):
    from ultralytics import YOLO

    model = YOLO(model_path)
    cats = {k: v for k, v in cfg["categories"].items() if k in ("vehicle", "two_wheeler")}
    cls2cat = {c: cat for cat, ids in cats.items() for c in ids}
    acc = defaultdict(lambda: Accumulator(conf_thr))  # (group, category) -> Accumulator
    elapsed, n = 0.0, 0
    for k, it in enumerate(items):
        img = imread_any(it["path"])
        if img is None:
            continue
        inp = apply_clahe(img, cfg["preprocess"]["clahe_clip"]) if clahe else img
        t0 = time.perf_counter()
        r = model.predict(inp, conf=PRED_CONF, iou=cfg["iou"], imgsz=imgsz, classes=sorted(cls2cat),
                          device=cfg.get("device"), verbose=False)[0]
        if k >= 3:  # 첫 몇 장은 초기화 시간이 섞여 제외
            elapsed += time.perf_counter() - t0
            n += 1
        preds = {c: [] for c in cats}
        for b, c, s in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.cls.int().cpu().tolist(), r.boxes.conf.cpu().tolist()):
            preds[cls2cat[c]].append((s, tuple(b)))
        g = it["gt"]
        region = regions.get(camera_id(g.name)) if regions else None
        preds = {c: [p for p in ps if in_region(p[1], region)] for c, ps in preds.items()}
        for group in ("all", it["condition"], it["daynight"]):
            for c in cats:
                acc[(group, c)].add([b for cc, b in g.boxes if cc == c], preds[c], g.ignore)
        if vis_dir is not None and k < n_vis:
            ok, buf = cv2.imencode(".jpg", cv2.resize(draw(img, g, preds, conf_thr, region), None, fx=0.5, fy=0.5))
            buf.tofile(str(vis_dir / f"{k:03d}_{it['condition']}_{g.name}"))
    fps = n / elapsed if elapsed else 0.0
    return {key: a.summary() for key, a in acc.items()}, fps


def main():
    ap = argparse.ArgumentParser(description="프레임 단위 검출 평가 (AI Hub CVAT XML 정답)")
    ap.add_argument("--images", required=True, help="원천데이터 VS_4 폴더")
    ap.add_argument("--labels", required=True, help="라벨링데이터 VL_4 폴더")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--models", nargs="+", default=None, help="기본: 설정 파일의 model")
    ap.add_argument("--imgsz", nargs="+", type=int, default=None, help="기본: 설정 파일의 imgsz")
    ap.add_argument("--conf", type=float, default=None, help="P/R/F1 운영 임계값 (기본: 설정 파일의 conf)")
    ap.add_argument("--clahe", action="store_true")
    ap.add_argument("--region", choices=["camera_hull", "none"], default="camera_hull", help="라벨링 영역 밖 검출 제외 방식")
    ap.add_argument("--only", choices=["day", "night"], default=None, help="주간 또는 야간 이미지만 평가")
    ap.add_argument("--per-group", type=int, default=60, help="조건 폴더별 표본 이미지 수 (0=전부)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vis", type=int, default=6, help="GT/예측을 그려 저장할 이미지 수")
    ap.add_argument("--out", default="outputs/det_eval")
    args = ap.parse_args()

    cfg = load_config(args.config)
    models = args.models or [cfg["model"]]
    sizes = args.imgsz or [cfg["imgsz"]]
    conf_thr = args.conf if args.conf is not None else cfg["conf"]

    all_items, regions = load_dataset(Path(args.images), Path(args.labels))
    if args.region == "none":
        regions = None
    if args.only:
        all_items = [it for it in all_items if it["daynight"] == args.only]
    items = sample(all_items, args.per_group, args.seed)
    counts = defaultdict(int)
    for it in items:
        counts[it["condition"]] += 1
        counts[it["daynight"]] += 1
    print(f"평가 이미지 {len(items)}장: {dict(counts)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in models:
        for sz in sizes:
            tag = f"{Path(m).stem}_{sz}{'_clahe' if args.clahe else ''}{'_noregion' if regions is None else ''}"
            vis_dir = out / f"vis_{tag}"
            vis_dir.mkdir(exist_ok=True)
            res, fps = eval_one(items, m, sz, cfg, conf_thr, args.clahe, regions, vis_dir, args.vis)
            print(f"\n== {tag}  {fps:.1f} FPS (추론만)")
            print(f"{'group':12s} {'cat':12s} {'n_gt':>6s} {'AP50':>6s} {'P':>6s} {'R':>6s} {'F1':>6s} {'R_s':>6s} {'R_m':>6s} {'R_l':>6s}")
            for (group, cat), s in sorted(res.items()):
                rows.append({"model": m, "imgsz": sz, "clahe": args.clahe, "region": args.region, "conf": conf_thr, "fps": round(fps, 2),
                             "group": group, "category": cat, **{k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()}})
                print(f"{group:12s} {cat:12s} {s['n_gt']:6d} {s['ap50']:6.3f} {s['precision']:6.3f} {s['recall']:6.3f} "
                      f"{s['f1']:6.3f} {s['recall_small']:6.3f} {s['recall_medium']:6.3f} {s['recall_large']:6.3f}")

    path = out / "det_eval.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # 엑셀에서 한글이 깨지지 않도록 BOM
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out / "det_eval_meta.json", "w", encoding="utf-8") as f:
        json.dump({"images": [str(it["path"]) for it in items], "counts": counts, "conf": conf_thr,
                   "per_group": args.per_group, "region": args.region, "only": args.only, "seed": args.seed}, f, ensure_ascii=False, indent=2)
    print(f"\n저장됨: {path}")


if __name__ == "__main__":
    main()
