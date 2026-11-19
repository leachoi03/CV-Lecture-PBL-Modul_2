"""위반 이벤트 CSV 로그 + 장면 스냅샷 저장."""
import csv
from pathlib import Path

import cv2

from .structures import Event

FIELDS = ["frame_idx", "time_sec", "track_id", "cls_name", "event_type", "x1", "y1", "x2", "y2", "detail"]


class EventLogger:
    def __init__(self, out_dir, save_snapshots: bool = True):
        self.out_dir = Path(out_dir)
        self.snap_dir = self.out_dir / "snapshots"
        self.save_snapshots = save_snapshots
        if save_snapshots:
            self.snap_dir.mkdir(parents=True, exist_ok=True)
        self._f = open(self.out_dir / "events.csv", "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow(FIELDS)
        self.counts = {}

    def log(self, e: Event, annotated_frame=None) -> None:
        x1, y1, x2, y2 = (round(v, 1) for v in e.bbox)
        self._w.writerow([e.frame_idx, f"{e.time_sec:.3f}", e.track_id, e.cls_name, e.event_type, x1, y1, x2, y2, e.detail])
        self._f.flush()
        self.counts[e.event_type] = self.counts.get(e.event_type, 0) + 1
        if self.save_snapshots and annotated_frame is not None:
            name = f"{e.frame_idx:06d}_{e.event_type}_id{e.track_id}.jpg"
            cv2.imwrite(str(self.snap_dir / name), annotated_frame)

    def close(self) -> None:
        self._f.close()
