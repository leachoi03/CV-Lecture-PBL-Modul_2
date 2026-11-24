"""웹 편집기에서 YOLO 검출 결과를 보여 주는 실행 관리.

브라우저가 편집 중인 ROI 와 시작 시각을 보내면(start) 실행 하나를 만든다.
- 결과 화면은 MJPEG(multipart/x-mixed-replace)로 흘려보낸다 → <img src> 하나로 볼 수 있다.
- 진행 상황과 위반 목록은 status 로 따로 받는다.
한 번에 하나만 돌린다: 새 실행을 시작하면 이전 실행은 멈춘다(GPU·CPU를 나눠 쓰지 않도록).
"""
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import cv2

from .config import load_config
from .roi import ROIConfig

BOUNDARY = "frame"
MAX_SHOW_WIDTH = 1280      # 큰 영상은 줄여서 보낸다 (판정은 원본 해상도로)
JPEG_QUALITY = 80
MAX_EVENTS = 500


class DetectRun:
    def __init__(self, video: Path, roi: dict, start_sec: float, cfg: dict, realtime: bool = True):
        self.id = uuid.uuid4().hex[:12]
        self.video, self.roi, self.start_sec, self.cfg, self.realtime = video, roi, max(0.0, start_sec), cfg, realtime
        self.stop_flag = threading.Event()
        self.lock = threading.Lock()
        self.streamed = False
        self.created = time.time()
        self.info = {"state": "ready", "frame": 0, "t": self.start_sec, "duration": 0.0, "proc_fps": 0.0,
                     "region": "", "counts": {}, "error": ""}
        self.events: List[dict] = []

    # ------------------------------------------------------------ 상태
    def status(self) -> dict:
        with self.lock:
            return {**self.info, "counts": dict(self.info["counts"]), "events": list(self.events)}

    def _set(self, **kw):
        with self.lock:
            self.info.update(kw)

    def stop(self):
        self.stop_flag.set()

    # ------------------------------------------------------------ 실행
    def frames(self) -> Iterator[bytes]:
        """결과 화면 JPEG 를 차례로 내놓는다. 끝나거나 stop 되면 멈춘다."""
        cap = cv2.VideoCapture(str(self.video))
        try:
            if not cap.isOpened():
                raise RuntimeError("영상을 열 수 없습니다")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            start = int(round(self.start_sec * fps))
            if start:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            self._set(state="loading", duration=total / fps if total else 0.0)

            from .pipeline import Pipeline  # YOLO(ultralytics) 는 검출할 때만 불러온다
            pipe = Pipeline(self.cfg, ROIConfig.from_dict(self.roi), fps, width, height)
            self._set(state="running", region=pipe.describe_region())
            scale = min(1.0, MAX_SHOW_WIDTH / float(width))
            frame_idx = start
            wall0 = time.perf_counter()
            while not self.stop_flag.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                res = pipe.step(frame, frame_idx)
                with self.lock:
                    for e in res.events:
                        if len(self.events) < MAX_EVENTS:
                            self.events.append({"t": round(e.time_sec, 2), "frame": e.frame_idx, "type": e.event_type,
                                                "id": e.track_id, "cls": e.cls_name, "detail": e.detail})
                    self.info.update(frame=frame_idx, t=frame_idx / fps, proc_fps=pipe.fps_ema,
                                     counts=dict(pipe.counts))
                vis = res.vis if scale >= 1.0 else cv2.resize(res.vis, None, fx=scale, fy=scale,
                                                              interpolation=cv2.INTER_AREA)
                ok, jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ok:
                    yield jpg.tobytes()
                frame_idx += 1
                if self.realtime:  # 처리가 영상보다 빠르면 영상 속도에 맞춘다
                    ahead = (frame_idx - start) / fps - (time.perf_counter() - wall0)
                    if ahead > 0:
                        time.sleep(ahead)
            self._set(state="stopped" if self.stop_flag.is_set() else "done")
        except Exception as ex:  # 모델 로드 실패 등은 화면에 이유를 보여 준다
            self._set(state="error", error=f"{type(ex).__name__}: {ex}")
        finally:
            cap.release()


class DetectManager:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.runs: Dict[str, DetectRun] = {}
        self.lock = threading.Lock()

    def start(self, video: Path, roi: dict, start_sec: float, realtime: bool = True) -> DetectRun:
        cfg = load_config(self.config_path)
        run = DetectRun(video, roi, start_sec, cfg, realtime)
        with self.lock:
            for old in self.runs.values():
                old.stop()
            # 오래된 실행 기록 정리 (최근 5개만 남김)
            for rid in sorted(self.runs, key=lambda r: self.runs[r].created)[:-4]:
                del self.runs[rid]
            self.runs[run.id] = run
        return run

    def get(self, run_id: str) -> Optional[DetectRun]:
        with self.lock:
            return self.runs.get(run_id)
