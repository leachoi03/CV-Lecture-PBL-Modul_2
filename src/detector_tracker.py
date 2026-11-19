"""YOLOv8 검출 + 다중 객체 추적(ByteTrack / BoT-SORT).

Ultralytics의 model.track(persist=True)는 프레임마다 검출을 수행한 뒤
내장 추적기로 이전 프레임의 트랙과 매칭해 고유 ID를 붙여 준다.
"""
from typing import Dict, List, Optional

from ultralytics import YOLO

from .structures import Track


class DetectorTracker:
    def __init__(
        self,
        model: str,
        tracker: str,
        categories: Dict[str, List[int]],
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 640,
        device: Optional[str] = None,
    ):
        self.model = YOLO(model)
        self.tracker = tracker
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        # COCO 클래스 ID → 카테고리 이름
        self.cls2cat = {c: cat for cat, ids in categories.items() for c in ids}
        self.classes = sorted(self.cls2cat)
        self.names = self.model.names

    def update(self, frame) -> List[Track]:
        """프레임 한 장을 처리해 ID가 부여된 트랙 목록을 반환한다."""
        result = self.model.track(
            frame,
            persist=True,  # 이전 프레임의 추적 상태 유지
            tracker=self.tracker,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )[0]

        boxes = result.boxes
        # 아직 확정된 트랙이 없으면 id가 None
        if boxes is None or boxes.id is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.int().cpu().tolist()
        cls_ids = boxes.cls.int().cpu().tolist()
        confs = boxes.conf.cpu().tolist()

        tracks = []
        for box, tid, cid, score in zip(xyxy, ids, cls_ids, confs):
            tracks.append(
                Track(
                    track_id=int(tid),
                    cls_id=int(cid),
                    cls_name=self.names[int(cid)],
                    category=self.cls2cat.get(int(cid), "other"),
                    conf=float(score),
                    bbox=tuple(float(v) for v in box),
                )
            )
        return tracks
