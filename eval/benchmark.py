"""모델 크기 × 추적기 조합별 처리 속도와 추적 통계 비교.

    python -m eval.benchmark --source data/videos/day01.mp4 \
        --models yolov8n.pt yolov8s.pt yolov8m.pt --trackers bytetrack.yaml botsort.yaml --frames 300

unique_ids: 같은 구간에서 생성된 ID 개수. 실제 객체 수가 같으므로 값이 클수록 ID가 자주 끊긴 것(ID 전환·조각남)으로 해석할 수 있다.
"""
import argparse
import csv
import time
from pathlib import Path

from src.config import load_config
from src.detector_tracker import DetectorTracker
from src.frame_source import FrameSource

WARMUP = 5


def bench_one(source, model, tracker, cfg, frames, conf, recursive=False):
    dt = DetectorTracker(model, tracker, cfg["categories"], conf, cfg["iou"], cfg["imgsz"], cfg.get("device"))
    cap = FrameSource(source, recursive=recursive)
    ids, n_tracks, elapsed, n = set(), 0, 0.0, 0
    while n < frames + WARMUP:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        tracks = dt.update(frame)
        if n >= WARMUP:  # 첫 몇 프레임은 모델 초기화 시간이 섞이므로 제외
            elapsed += time.perf_counter() - t0
            n_tracks += len(tracks)
            ids.update(t.track_id for t in tracks)
        n += 1
    cap.release()
    measured = max(0, n - WARMUP)
    return {
        "model": model,
        "tracker": tracker,
        "conf": conf,
        "frames": measured,
        "fps": measured / elapsed if elapsed else 0.0,
        "avg_tracks_per_frame": n_tracks / measured if measured else 0.0,
        "unique_ids": len(ids),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="영상 또는 이미지 폴더")
    ap.add_argument("--recursive", action="store_true", help="이미지 폴더의 하위 폴더까지 읽기")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--models", nargs="+", default=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
    ap.add_argument("--trackers", nargs="+", default=["bytetrack.yaml"])
    ap.add_argument("--confs", nargs="+", type=float, default=None)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--out", default="outputs/benchmark")
    args = ap.parse_args()

    cfg = load_config(args.config)
    confs = args.confs or [cfg["conf"]]
    rows = []
    for m in args.models:
        for tr in args.trackers:
            for c in confs:
                r = bench_one(args.source, m, tr, cfg, args.frames, c, args.recursive)
                rows.append(r)
                print(f"{m:12s} {tr:15s} conf={c:.2f}  {r['fps']:6.1f} FPS  "
                      f"tracks/frame={r['avg_tracks_per_frame']:.1f}  unique_ids={r['unique_ids']}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"benchmark_{Path(args.source).name if Path(args.source).is_dir() else Path(args.source).stem}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"저장됨: {path}")


if __name__ == "__main__":
    main()
