"""한 프레임 처리 파이프라인: 검출·추적 → 판정 → 결과 화면 그리기.

명령줄 실행(src.main)과 웹 검출 화면(src.web_server)이 같은 코드를 쓴다.
"""
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .detect_region import DetectRegion
from .detector_tracker import DetectorTracker
from .preprocess import apply_clahe
from .roi import ROIConfig
from .rules import JaywalkRule, LaneRule, StopLineRule
from .signal_state import SignalProvider
from .structures import Event, Track
from .trajectory import TrajectoryStore
from .visualize import AlertBoard, draw_hud, draw_roi, draw_tracks


@dataclass
class StepResult:
    vis: np.ndarray          # 결과를 그린 화면
    events: List[Event]      # 이번 프레임에 새로 생긴 위반
    tracks: List[Track]
    signal: str
    proc_sec: float          # 검출·추적·판정 시간 (그리기 제외)


class Pipeline:
    def __init__(self, cfg: dict, roi: ROIConfig, fps: float, width: int, height: int):
        self.cfg, self.roi, self.fps = cfg, roi, fps
        self.signal_provider = SignalProvider(roi.signal)
        self.region = DetectRegion.from_roi(roi, width, height, cfg.get("detect_region"))
        self.dt = DetectorTracker(
            model=cfg["model"], tracker=cfg["tracker"], categories=cfg["categories"],
            conf=cfg["conf"], iou=cfg["iou"], imgsz=cfg["imgsz"], device=cfg.get("device"),
        )
        self.traj = TrajectoryStore(cfg["trajectory"]["maxlen"], cfg["trajectory"]["max_age"])
        self.stop_rule = StopLineRule(roi.stop_lines, cfg["stopline"]["confirm_frames"])
        self.jay_rule = JaywalkRule(
            roi,
            min_frames=int(round(cfg["jaywalk"]["min_seconds"] * fps)),
            rider_overlap=cfg["jaywalk"]["rider_overlap"],
        )
        lane_cfg = cfg.get("lane", {})
        self.lane_rule = LaneRule(
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
        self.alerts = AlertBoard(ttl_frames=int(cfg["output"]["alert_ttl_seconds"] * fps))
        self.counts: Dict[str, int] = {}
        self.fps_ema = 0.0

    def describe_region(self) -> str:
        r = self.region
        if not r.active:
            return "검출 영역: 전체 화면 (ROI 다각형 없음 또는 detect_region.enabled=false)"
        x0, y0, x1, y1 = r.rect
        return (f"검출 영역: ROI 주변 {x1 - x0}x{y1 - y0} (원본의 {100 * r.area_ratio:.0f}%), "
                f"ROI 밖 가리기 {'켬' if r.mask else '끔'}, YOLO 입력 {r.imgsz(self.cfg['imgsz'])}")

    @property
    def violators(self):
        return self.stop_rule.violators | self.jay_rule.violators | self.lane_rule.violators

    def step(self, frame: np.ndarray, frame_idx: int) -> StepResult:
        cfg = self.cfg
        t_sec = frame_idx / self.fps
        t0 = time.perf_counter()

        inp = apply_clahe(frame, cfg["preprocess"]["clahe_clip"]) if cfg["preprocess"]["clahe"] else frame
        det_img, offset = self.region.prepare(inp)
        tracks = self.region.filter(self.dt.update(det_img, offset, self.region.imgsz(cfg["imgsz"])))
        self.traj.update(tracks, frame_idx)
        signal = self.signal_provider.get(frame, t_sec)
        events = (self.stop_rule.update(tracks, frame_idx, t_sec, signal)
                  + self.jay_rule.update(tracks, frame_idx, t_sec)
                  + self.lane_rule.update(tracks, frame_idx, t_sec))

        proc = time.perf_counter() - t0
        self.fps_ema = 0.9 * self.fps_ema + 0.1 / proc if self.fps_ema else 1.0 / proc

        vis = frame.copy()
        draw_roi(vis, self.roi, signal)
        self.region.draw(vis)
        draw_tracks(vis, tracks, self.traj, self.violators, lane_of=self.lane_rule.current)
        for e in events:
            self.alerts.add(e)
            self.counts[e.event_type] = self.counts.get(e.event_type, 0) + 1
        self.alerts.draw(vis, frame_idx)
        draw_hud(vis, self.fps_ema, signal, self.counts)
        return StepResult(vis, events, tracks, signal, proc)
