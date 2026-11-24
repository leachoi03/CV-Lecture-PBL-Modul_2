"""검출 영역 제한: 지정한 ROI(도로·차로·횡단보도) 부분만 YOLO에 넣는다.

1. 잘라내기(crop): ROI를 감싸는 사각형만 잘라서 검출한다.
   YOLO 는 입력을 항상 imgsz 크기로 맞추므로 두 가지로 쓸 수 있다.
   - keep_scale=true : imgsz 를 잘라낸 비율만큼 줄여 전체 화면 검출과 같은 배율을 유지 → 넓이만큼 빨라진다.
   - keep_scale=false: imgsz 를 그대로 써서 잘라낸 부분을 확대 → 속도 이득은 없지만 작은(먼) 차량이 크게 보인다.
2. 가리기(mask): 사각형 안에서도 ROI 밖 픽셀은 회색(114, YOLO 여백 색)으로 칠해 검출되지 않게 한다.
   차량은 화면에서 위로 서 있으므로(지붕이 도로 다각형 위로 삐져나옴) ROI를 위쪽으로 up_frac 만큼 넓혀서 가린다.
3. 거르기(filter): 추적 결과 중 지면 접점(bbox 아래 가운데)이 ROI 밖인 객체는 버린다.
ROI 다각형이 하나도 없으면 아무것도 하지 않고 전체 화면을 쓴다.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .roi import ROIConfig
from .structures import Track

PAD_GRAY = 114


@dataclass
class DetectRegion:
    width: int
    height: int
    keep_mask: Optional[np.ndarray] = None   # YOLO 에 보여 줄 영역 (위로 넓힌 ROI)
    foot_mask: Optional[np.ndarray] = None   # 지면 접점이 있어야 하는 영역 (ROI + 약간의 여유)
    rect: Tuple[int, int, int, int] = (0, 0, 0, 0)  # 잘라낼 사각형 x0, y0, x1, y1
    crop: bool = True
    mask: bool = True
    keep_scale: bool = False

    @property
    def active(self) -> bool:
        return self.keep_mask is not None

    @classmethod
    def from_roi(cls, roi: ROIConfig, width: int, height: int, cfg: Optional[dict] = None) -> "DetectRegion":
        cfg = cfg or {}
        polys: List[np.ndarray] = list(roi.road) + [ln.polygon for ln in roi.lanes] + list(roi.crosswalks)
        region = cls(width, height, crop=cfg.get("crop", True), mask=cfg.get("mask", True),
                     keep_scale=cfg.get("keep_scale", False))
        if not cfg.get("enabled", True) or not polys:
            return region

        base = np.zeros((height, width), np.uint8)
        cv2.fillPoly(base, [np.round(p).astype(np.int32).reshape(-1, 1, 2) for p in polys], 255)

        tol = int(cfg.get("foot_tolerance_px", 12))
        region.foot_mask = cv2.dilate(base, np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)) if tol > 0 else base

        margin = int(cfg.get("margin_px", 16))
        keep = cv2.dilate(base, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)) if margin > 0 else base.copy()
        up = int(round(cfg.get("up_frac", 0.15) * height))
        if up > 0:
            # 기준점이 맨 위인 세로 커널: 아래쪽에 ROI 가 있으면 켜진다 = ROI 를 위로 up 픽셀 넓힘
            kernel = np.ones((up + 1, 1), np.uint8)
            keep = cv2.dilate(keep, kernel, anchor=(0, 0))
        region.keep_mask = keep

        ys, xs = np.nonzero(keep)
        region.rect = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        return region

    # ------------------------------------------------------------ 검출 전
    def prepare(self, frame) -> Tuple[np.ndarray, Tuple[int, int]]:
        """YOLO 에 넣을 이미지와, 그 이미지 좌표를 원본 좌표로 되돌릴 (dx, dy)."""
        if not self.active:
            return frame, (0, 0)
        x0, y0, x1, y1 = self.rect if self.crop else (0, 0, self.width, self.height)
        img = frame[y0:y1, x0:x1]
        if self.mask:
            img = img.copy()
            img[self.keep_mask[y0:y1, x0:x1] == 0] = PAD_GRAY
        return img, (x0, y0)

    def imgsz(self, base: int) -> int:
        """YOLO 입력 크기. keep_scale 이면 잘라낸 긴 변 비율만큼 줄인다(32의 배수)."""
        if not (self.active and self.crop and self.keep_scale):
            return base
        x0, y0, x1, y1 = self.rect
        ratio = max(x1 - x0, y1 - y0) / float(max(self.width, self.height))
        return max(64, int(round(base * ratio / 32)) * 32)

    # ------------------------------------------------------------ 검출 후
    def contains_foot(self, track: Track) -> bool:
        if not self.active:
            return True
        x, y = int(round(track.foot[0])), int(round(track.foot[1]))
        x, y = min(max(x, 0), self.width - 1), min(max(y, 0), self.height - 1)
        return bool(self.foot_mask[y, x])

    def filter(self, tracks: Sequence[Track]) -> List[Track]:
        return [t for t in tracks if self.contains_foot(t)]

    @property
    def area_ratio(self) -> float:
        """원본 대비 YOLO 에 넣는 사각형 넓이 비율 (속도 비교용)."""
        if not self.active or not self.crop:
            return 1.0
        x0, y0, x1, y1 = self.rect
        return (x1 - x0) * (y1 - y0) / float(self.width * self.height)

    def draw(self, img) -> None:
        """결과 화면에 검출 영역 경계(흰 점선 느낌의 얇은 선)를 그린다."""
        if not self.active:
            return
        contours, _ = cv2.findContours(self.keep_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (220, 220, 220), 1, cv2.LINE_AA)
