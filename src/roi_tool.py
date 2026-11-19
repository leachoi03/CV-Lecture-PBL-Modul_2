"""ROI 설정 도구: 영상의 한 프레임 위에 마우스로 정지선·차도·횡단보도·신호등 영역을 찍어 JSON으로 저장.

예)
    python -m src.roi_tool --source data/videos/day01.mp4 --out configs/roi/day01.json --frame 100

키
    s : 정지선      - 양 끝점 2번 클릭 + 정지선 앞(차가 대기하는 쪽) 1번 클릭
    r : 차도 다각형  - 꼭짓점을 여러 번 클릭 후 Enter
    c : 횡단보도 다각형 - 꼭짓점을 여러 번 클릭 후 Enter
    l : 신호등 영역  - 좌상단, 우하단 2번 클릭 (signal.mode 가 hsv 로 바뀜)
    z : 입력 중인 점 하나 취소      x : 마지막으로 완성한 도형 삭제
    w : 저장        q : 종료
신호 타임라인(signal.timeline)은 저장 후 JSON을 직접 편집해 입력한다.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .frame_source import FrameSource

HELP = "s:stopline r:road c:crosswalk l:light | Enter:finish z:undo pt x:del shape | w:save q:quit"
DEFAULT_SIGNAL = {
    "mode": "timeline",
    "cycle": None,
    "timeline": [{"start": 0.0, "end": 9999.0, "state": "red"}],
}


class ROITool:
    def __init__(self, frame, out_path: Path):
        self.base = frame
        self.out_path = out_path
        self.mode = None
        self.cur = []
        self.history = []  # 완성한 도형 종류 순서 (x 키로 되돌리기)
        self.data = {"stop_lines": [], "road": [], "crosswalks": [], "signal": dict(DEFAULT_SIGNAL)}
        if out_path.exists():
            with open(out_path, encoding="utf-8") as f:
                self.data.update(json.load(f))
            print(f"기존 ROI 파일을 불러왔습니다: {out_path}")

    # ------------------------------------------------------------ 입력 처리
    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or self.mode is None:
            return
        self.cur.append([int(x), int(y)])
        if self.mode == "s" and len(self.cur) == 3:
            n = len(self.data["stop_lines"])
            self.data["stop_lines"].append(
                {"name": f"line{n}", "p1": self.cur[0], "p2": self.cur[1], "approach_point": self.cur[2]}
            )
            self._finish("stop_lines")
        elif self.mode == "l" and len(self.cur) == 2:
            (x1, y1), (x2, y2) = self.cur
            self.data["signal"] = {
                "mode": "hsv",
                "light_roi": [min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)],
                "min_ratio": 0.02,
            }
            self._finish("signal")

    def finish_polygon(self):
        if self.mode in ("r", "c") and len(self.cur) >= 3:
            self.data["road" if self.mode == "r" else "crosswalks"].append(self.cur)
            self._finish("road" if self.mode == "r" else "crosswalks")

    def _finish(self, kind):
        self.history.append(kind)
        self.cur = []
        print(f"추가됨: {kind}")

    def delete_last(self):
        if not self.history:
            return
        kind = self.history.pop()
        if kind == "signal":
            self.data["signal"] = dict(DEFAULT_SIGNAL)
        elif self.data[kind]:
            self.data[kind].pop()
        print(f"삭제됨: {kind}")

    def save(self):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        print(f"저장됨: {self.out_path}")

    # ------------------------------------------------------------ 그리기
    def render(self):
        img = self.base.copy()
        for poly in self.data["road"]:
            cv2.polylines(img, [np.array(poly, np.int32)], True, (160, 160, 160), 2)
        for poly in self.data["crosswalks"]:
            cv2.polylines(img, [np.array(poly, np.int32)], True, (0, 255, 255), 2)
        for s in self.data["stop_lines"]:
            cv2.line(img, tuple(s["p1"]), tuple(s["p2"]), (0, 0, 255), 3)
            cv2.circle(img, tuple(s["approach_point"]), 5, (0, 0, 255), -1)
            cv2.putText(img, s["name"], tuple(s["p1"]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if self.data["signal"].get("mode") == "hsv":
            x, y, w, h = self.data["signal"]["light_roi"]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        for p in self.cur:
            cv2.circle(img, tuple(p), 4, (0, 255, 0), -1)
        if len(self.cur) >= 2:
            cv2.polylines(img, [np.array(self.cur, np.int32)], False, (0, 255, 0), 1)

        cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(img, f"[mode {self.mode or '-'}] {HELP}", (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return img


def read_frame(source: str, index: int):
    cap = FrameSource(source)
    cap.seek(index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"프레임을 읽을 수 없습니다: {source} #{index}")
    return frame


def main():
    p = argparse.ArgumentParser(description="ROI 설정 도구")
    p.add_argument("--source", required=True, help="영상 또는 이미지 폴더")
    p.add_argument("--out", required=True)
    p.add_argument("--frame", type=int, default=0, help="기준으로 쓸 프레임 번호")
    args = p.parse_args()

    tool = ROITool(read_frame(args.source, args.frame), Path(args.out))
    win = "roi-tool"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, tool.on_mouse)
    print(__doc__)

    while True:
        cv2.imshow(win, tool.render())
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("s"), ord("r"), ord("c"), ord("l")):
            tool.mode, tool.cur = chr(key), []
        elif key == 13:  # Enter
            tool.finish_polygon()
        elif key == ord("z") and tool.cur:
            tool.cur.pop()
        elif key == ord("x"):
            tool.delete_last()
        elif key == ord("w"):
            tool.save()
        elif key == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
