"""신호 상태 제공.

mode
- "timeline": 영상 시간(초)별 신호를 설정 파일로 입력 (AI Hub 라벨이 있으면 변환해서 사용)
- "hsv":      화면 속 신호등 영역의 색으로 적/황/녹 판별
- "fixed":    항상 같은 상태 (테스트용)
"""
import cv2
import numpy as np

RED, YELLOW, GREEN, UNKNOWN = "red", "yellow", "green", "unknown"


class SignalProvider:
    def __init__(self, cfg: dict):
        self.cfg = cfg or {}
        self.mode = self.cfg.get("mode", "fixed")

    def get(self, frame, t_sec: float) -> str:
        if self.mode == "timeline":
            return self._from_timeline(t_sec)
        if self.mode == "hsv":
            return self._from_hsv(frame)
        return self.cfg.get("state", UNKNOWN)

    # ------------------------------------------------------------ timeline
    def _from_timeline(self, t_sec: float) -> str:
        cycle = self.cfg.get("cycle")
        t = t_sec % cycle if cycle else t_sec
        for seg in self.cfg.get("timeline", []):
            if seg["start"] <= t < seg["end"]:
                return seg["state"]
        return UNKNOWN

    # ------------------------------------------------------------ hsv
    def _from_hsv(self, frame) -> str:
        x, y, w, h = self.cfg["light_roi"]
        crop = frame[y : y + h, x : x + w]
        if crop.size == 0:
            return UNKNOWN
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # 켜진 등만 보도록 채도·명도가 높은 픽셀만 사용
        s_min, v_min = self.cfg.get("s_min", 100), self.cfg.get("v_min", 150)
        lo = lambda h0: np.array([h0, s_min, v_min])
        hi = lambda h1: np.array([h1, 255, 255])
        red = cv2.inRange(hsv, lo(0), hi(10)) | cv2.inRange(hsv, lo(160), hi(180))
        yellow = cv2.inRange(hsv, lo(15), hi(35))
        green = cv2.inRange(hsv, lo(40), hi(95))

        total = crop.shape[0] * crop.shape[1]
        ratios = {
            RED: cv2.countNonZero(red) / total,
            YELLOW: cv2.countNonZero(yellow) / total,
            GREEN: cv2.countNonZero(green) / total,
        }
        state, ratio = max(ratios.items(), key=lambda kv: kv[1])
        return state if ratio >= self.cfg.get("min_ratio", 0.02) else UNKNOWN
