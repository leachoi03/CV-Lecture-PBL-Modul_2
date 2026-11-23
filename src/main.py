"""교통 위반 감지 파이프라인 실행.

예)
    python -m src.main --source data/videos/day01.mp4 --roi configs/roi/day01.json --show
    python -m src.main --source data/frames/seq01 --fps 10 --roi configs/roi/seq01.json   # 이미지 폴더
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2

from .config import load_config
from .detector_tracker import DetectorTracker
from .event_logger import EventLogger
from .frame_source import FrameSource
from .preprocess import apply_clahe
from .roi import ROIConfig
from .rules import JaywalkRule, LaneRule, StopLineRule
from .signal_state import SignalProvider
from .trajectory import TrajectoryStore
from .visualize import AlertBoard, draw_hud, draw_roi, draw_tracks


def parse_args():
    p = argparse.ArgumentParser(description="교차로 CCTV 교통 위반 감지")
    p.add_argument("--source", required=True, help="입력 영상 경로 또는 이미지 폴더")
    p.add_argument("--roi", required=True, help="ROI JSON 경로 (웹 편집기 src.web_server 또는 src.roi_tool로 생성)")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--out", default="outputs", help="결과 저장 상위 폴더")
    p.add_argument("--name", default=None, help="결과 폴더 이름 (기본: 영상이름_시각)")
    p.add_argument("--show", action="store_true", help="처리 화면 띄우기 (q로 종료)")
    p.add_argument("--max-frames", type=int, default=0, help="0이면 끝까지")
    p.add_argument("--fps", type=float, default=None, help="이미지 폴더의 프레임 속도 (기본 10, 영상은 파일 값)")
    p.add_argument("--recursive", action="store_true", help="이미지 폴더의 하위 폴더까지 읽기")
    # 실험용 덮어쓰기
    p.add_argument("--model")
    p.add_argument("--tracker")
    p.add_argument("--conf", type=float)
    p.add_argument("--clahe", action="store_true")
    return p.parse_args()


def run(args) -> dict:
    cfg = load_config(args.config)
    for key in ("model", "tracker", "conf"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if args.clahe:
        cfg["preprocess"]["clahe"] = True

    cap = FrameSource(args.source, fps=args.fps, recursive=args.recursive)
    if cap.warning:
        print(cap.warning)
    fps, width, height = cap.fps, cap.width, cap.height

    name = args.name or f"{cap.name}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = Path(args.out) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 구성 요소 준비
    roi = ROIConfig.load(args.roi)
    signal_provider = SignalProvider(roi.signal)
    dt = DetectorTracker(
        model=cfg["model"],
        tracker=cfg["tracker"],
        categories=cfg["categories"],
        conf=cfg["conf"],
        iou=cfg["iou"],
        imgsz=cfg["imgsz"],
        device=cfg.get("device"),
    )
    traj = TrajectoryStore(cfg["trajectory"]["maxlen"], cfg["trajectory"]["max_age"])
    stop_rule = StopLineRule(roi.stop_lines, cfg["stopline"]["confirm_frames"])
    jay_rule = JaywalkRule(
        roi,
        min_frames=int(round(cfg["jaywalk"]["min_seconds"] * fps)),
        rider_overlap=cfg["jaywalk"]["rider_overlap"],
    )
    lane_cfg = cfg.get("lane", {})
    lane_rule = LaneRule(
        roi, fps,
        applies_to=lane_cfg.get("applies_to", ["vehicle"]),
        stable_seconds=lane_cfg.get("stable_seconds", 0.3),
        multi_window_seconds=lane_cfg.get("multi_window_seconds", 3.0),
        turn_degrees=lane_cfg.get("turn_degrees", 45),
        heading_seconds=lane_cfg.get("heading_seconds", 0.7),
        min_move_px=lane_cfg.get("min_move_px", 15),
        turn_checks=lane_cfg.get("turn_checks", ["right"]),
        max_age=cfg["trajectory"]["max_age"],
    )
    if roi.lanes:
        print(f"차로 {len(roi.lanes)}개: " + ", ".join(ln.id for ln in roi.lanes))
    logger = EventLogger(out_dir, cfg["output"]["save_snapshots"])
    alerts = AlertBoard(ttl_frames=int(cfg["output"]["alert_ttl_seconds"] * fps))
    writer = None
    if cfg["output"]["save_video"]:
        writer = cv2.VideoWriter(str(out_dir / "result.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # ---- 프레임 루프
    frame_idx, proc_time, fps_ema = 0, 0.0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and frame_idx >= args.max_frames):
            break
        t_sec = frame_idx / fps
        t0 = time.perf_counter()

        inp = apply_clahe(frame, cfg["preprocess"]["clahe_clip"]) if cfg["preprocess"]["clahe"] else frame
        tracks = dt.update(inp)
        traj.update(tracks, frame_idx)
        signal = signal_provider.get(frame, t_sec)
        events = (stop_rule.update(tracks, frame_idx, t_sec, signal) + jay_rule.update(tracks, frame_idx, t_sec)
                  + lane_rule.update(tracks, frame_idx, t_sec))

        dt_sec = time.perf_counter() - t0  # 검출·추적·판정 시간만 측정 (그리기·저장 제외)
        proc_time += dt_sec
        fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt_sec) if fps_ema else 1.0 / dt_sec

        # ---- 그리기 / 기록
        vis = frame.copy()
        draw_roi(vis, roi, signal)
        draw_tracks(vis, tracks, traj, stop_rule.violators | jay_rule.violators | lane_rule.violators,
                    lane_of=lane_rule.current)
        for e in events:
            alerts.add(e)
        alerts.draw(vis, frame_idx)
        counts = dict(logger.counts)  # 이번 프레임 이벤트까지 포함한 누적 개수
        for e in events:
            counts[e.event_type] = counts.get(e.event_type, 0) + 1
        draw_hud(vis, fps_ema, signal, counts)
        for e in events:
            logger.log(e, vis)
            print(f"[{e.time_sec:7.2f}s] {e.event_type:9s} id={e.track_id} {e.cls_name} {e.detail}")

        if writer is not None:
            writer.write(vis)
        if args.show:
            cv2.imshow("traffic-violation", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    logger.close()
    cv2.destroyAllWindows()

    summary = {
        "source": args.source,
        "roi": args.roi,
        "frames": frame_idx,
        "input_type": "images" if cap.is_folder else "video",
        "video_fps": fps,
        "sequence_warning": cap.warning,
        "processing_fps": frame_idx / proc_time if proc_time else 0.0,
        "avg_ms_per_frame": 1000 * proc_time / frame_idx if frame_idx else 0.0,
        "unique_track_ids": len(traj.all_ids),
        "events": logger.counts,
        "config": cfg,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n완료: {frame_idx} frames, {summary['processing_fps']:.1f} FPS, events={logger.counts}")
    print(f"결과 폴더: {out_dir}")
    return summary


if __name__ == "__main__":
    run(parse_args())
