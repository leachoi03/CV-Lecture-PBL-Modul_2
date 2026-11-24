"""웹 ROI 편집기 서버 (표준 라이브러리만 사용).

    python -m src.web_server                 # http://127.0.0.1:8000 을 브라우저로 연다
    python -m src.web_server --videos data/videos --roi-dir configs/roi --port 8000

- GET  /                 웹 편집기 (web/roi_editor.html)
- GET  /api/videos       영상 폴더의 영상 목록
- GET  /video/<이름>     영상 스트리밍 (Range 요청 지원 → 브라우저에서 앞뒤로 이동 가능)
- GET  /api/roi/<이름>   configs/roi/<이름>.json 읽기
- POST /api/roi/<이름>   같은 파일에 저장 (기존 정지선·신호 등 다른 항목은 유지하고 road·lanes 를 바꾼다)
- POST /api/detect/start          {video, roi, start} → YOLO 검출 실행 하나 만들기 (src.web_detect)
- GET  /api/detect/stream/<실행>  결과 화면 MJPEG
- GET  /api/detect/status/<실행>  진행 상황·위반 목록
- POST /api/detect/stop/<실행>    멈추기
내 PC에만 열리도록 127.0.0.1 에서만 듣는다.
"""
import argparse
import json
import mimetypes
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

from .web_detect import BOUNDARY, DetectManager

ROOT = Path(__file__).resolve().parent.parent
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".m4v", ".ogg", ".mkv", ".avi"}
SAFE_STEM = re.compile(r"^[^/\\:*?\"<>|]{1,200}$")
CHUNK = 1 << 20


LANE_ZONES = {"approach", "intersection"}
LANE_TURNS = {"left", "straight", "right", "uturn"}


def _poly_error(name: str, poly) -> str:
    if not isinstance(poly, list) or len(poly) < 3:
        return f"{name} 는 점이 3개 이상이어야 합니다"
    for pt in poly:
        if not (isinstance(pt, list) and len(pt) == 2
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in pt)):
            return f"{name} 의 점 형식이 잘못되었습니다: {pt!r}"
    return ""


def valid_roi(data) -> str:
    """저장할 ROI 내용 검사. 문제가 있으면 이유 문자열, 없으면 빈 문자열."""
    if not isinstance(data, dict):
        return "JSON 객체가 아닙니다"
    road = data.get("road", [])
    if not isinstance(road, list):
        return "road 는 다각형 목록이어야 합니다"
    for i, poly in enumerate(road):
        err = _poly_error(f"road[{i}]", poly)
        if err:
            return err
    lanes = data.get("lanes", [])
    if not isinstance(lanes, list):
        return "lanes 는 차로 목록이어야 합니다"
    ids = set()
    for i, lane in enumerate(lanes):
        if not isinstance(lane, dict):
            return f"lanes[{i}] 는 객체여야 합니다"
        err = _poly_error(f"lanes[{i}].polygon", lane.get("polygon"))
        if err:
            return err
        if not isinstance(lane.get("approach"), str) or not lane["approach"].strip():
            return f"lanes[{i}] 의 진입로 이름이 없습니다"
        idx = lane.get("index")
        if not isinstance(idx, int) or isinstance(idx, bool) or idx < 1:
            return f"lanes[{i}] 의 차로 번호는 1 이상의 정수여야 합니다"
        if lane.get("zone", "approach") not in LANE_ZONES:
            return f"lanes[{i}] 의 구역은 {sorted(LANE_ZONES)} 중 하나여야 합니다"
        turns = lane.get("turns", [])
        if not isinstance(turns, list) or not set(turns) <= LANE_TURNS:
            return f"lanes[{i}] 의 진행 방향은 {sorted(LANE_TURNS)} 만 쓸 수 있습니다"
        lid = lane.get("id")
        if lid is not None:
            if lid in ids:
                return f"차로 ID '{lid}' 가 겹칩니다"
            ids.add(lid)
    return ""


def merge_roi(old: dict, new: dict) -> dict:
    """기존 파일 내용에 새 내용을 덮어쓴다(편집기가 모르는 항목도 보존)."""
    out = dict(old)
    out.update(new)
    return out


def parse_range(header: str, size: int):
    """'bytes=START-END' → (start, end) 포함 구간. 해석할 수 없으면 None."""
    m = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
    if not m or (not m.group(1) and not m.group(2)):
        return None
    if m.group(1):
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else size - 1
    else:  # 끝에서 N바이트
        start, end = max(0, size - int(m.group(2))), size - 1
    end = min(end, size - 1)
    return (start, end) if start <= end and start < size else None


def make_handler(video_dir: Path, roi_dir: Path, web_dir: Path, detector: Optional[DetectManager] = None):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ROIEditor/1.0"

        def log_message(self, fmt, *args):  # 영상 조각·검출 상태 요청마다 찍히는 로그는 생략
            if not self.path.startswith(("/video/", "/api/detect/status/")):
                super().log_message(fmt, *args)

        # ------------------------------------------------ 응답 도우미
        def send_json(self, obj, status=HTTPStatus.OK):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def error_json(self, status, msg):
            self.send_json({"error": msg}, status)

        def safe_child(self, base: Path, name: str):
            """base 바깥 경로(../ 등)는 거부한다."""
            if not SAFE_STEM.match(name):
                return None
            p = (base / name).resolve()
            return p if p.parent == base.resolve() else None

        def read_json(self):
            """요청 본문 JSON. 문제가 있으면 오류 응답을 보내고 None."""
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > 5 * CHUNK:
                self.error_json(HTTPStatus.BAD_REQUEST, "내용이 없거나 너무 큽니다")
                return None
            try:
                return json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self.error_json(HTTPStatus.BAD_REQUEST, "JSON 형식이 아닙니다")
                return None

        def load_roi_file(self, stem: str) -> dict:
            p = self.safe_child(roi_dir, stem + ".json")
            if p is None or not p.is_file():
                return {}
            with open(p, encoding="utf-8") as f:
                return json.load(f)

        # ------------------------------------------------ 라우팅
        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path.startswith("/api/detect/"):
                return self.detect_get(path[len("/api/detect/"):])
            if path in ("/", "/index.html"):
                return self.send_file(web_dir / "roi_editor.html", "text/html; charset=utf-8")
            if path == "/api/videos":
                vids = sorted(p for p in video_dir.glob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS) \
                    if video_dir.is_dir() else []
                return self.send_json([{"name": p.name, "size": p.stat().st_size,
                                        "has_roi": (roi_dir / f"{p.stem}.json").exists()} for p in vids])
            if path.startswith("/video/"):
                p = self.safe_child(video_dir, path[len("/video/"):])
                if p is None or not p.is_file():
                    return self.error_json(HTTPStatus.NOT_FOUND, "영상을 찾을 수 없습니다")
                return self.send_file(p, mimetypes.guess_type(p.name)[0] or "video/mp4", allow_range=True)
            if path.startswith("/api/roi/"):
                p = self.safe_child(roi_dir, path[len("/api/roi/"):] + ".json")
                if p is None:
                    return self.error_json(HTTPStatus.BAD_REQUEST, "잘못된 이름")
                if not p.is_file():
                    return self.error_json(HTTPStatus.NOT_FOUND, "ROI 파일 없음")
                with open(p, encoding="utf-8") as f:
                    return self.send_json(json.load(f))
            self.error_json(HTTPStatus.NOT_FOUND, "없는 주소")

        def do_POST(self):
            path = unquote(urlparse(self.path).path)
            if path.startswith("/api/detect/"):
                return self.detect_post(path[len("/api/detect/"):])
            if not path.startswith("/api/roi/"):
                return self.error_json(HTTPStatus.NOT_FOUND, "없는 주소")
            p = self.safe_child(roi_dir, path[len("/api/roi/"):] + ".json")
            if p is None:
                return self.error_json(HTTPStatus.BAD_REQUEST, "잘못된 이름")
            data = self.read_json()
            if data is None:
                return
            err = valid_roi(data)
            if err:
                return self.error_json(HTTPStatus.BAD_REQUEST, err)
            old = {}
            if p.is_file():
                with open(p, encoding="utf-8") as f:
                    old = json.load(f)
            roi_dir.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(merge_roi(old, data), f, ensure_ascii=False, indent=2)
            rel = p.relative_to(ROOT) if p.is_relative_to(ROOT) else p
            print(f"ROI 저장: {rel} (도로 다각형 {len(data.get('road', []))}개, 차로 {len(data.get('lanes', []))}개)")
            self.send_json({"ok": True, "path": str(rel).replace("\\", "/")})

        # ------------------------------------------------ YOLO 검출
        def detect_post(self, sub: str):
            if detector is None:
                return self.error_json(HTTPStatus.SERVICE_UNAVAILABLE, "검출 기능이 꺼져 있습니다")
            if sub == "start":
                body = self.read_json()
                if body is None:
                    return
                if not isinstance(body, dict):
                    return self.error_json(HTTPStatus.BAD_REQUEST, "JSON 객체가 아닙니다")
                video = self.safe_child(video_dir, str(body.get("video", "")))
                if video is None or not video.is_file():
                    return self.error_json(HTTPStatus.NOT_FOUND, "영상을 찾을 수 없습니다 (서버 영상만 검출할 수 있습니다)")
                roi = body.get("roi") or {}
                err = valid_roi(roi)
                if err:
                    return self.error_json(HTTPStatus.BAD_REQUEST, err)
                start = body.get("start", 0)
                if not isinstance(start, (int, float)) or isinstance(start, bool):
                    return self.error_json(HTTPStatus.BAD_REQUEST, "start 는 초 단위 숫자여야 합니다")
                # 저장된 파일의 정지선·횡단보도·신호 등은 살리고 편집 중인 도로·차로로 덮어쓴다
                merged = merge_roi(self.load_roi_file(video.stem), roi)
                run = detector.start(video, merged, float(start), bool(body.get("realtime", True)))
                print(f"검출 시작: {video.name} {float(start):.1f}초부터 (실행 {run.id})")
                return self.send_json({"run": run.id})
            if sub.startswith("stop/"):
                run = detector.get(sub[len("stop/"):])
                if run is None:
                    return self.error_json(HTTPStatus.NOT_FOUND, "없는 실행")
                run.stop()
                return self.send_json({"ok": True})
            self.error_json(HTTPStatus.NOT_FOUND, "없는 주소")

        def detect_get(self, sub: str):
            if detector is None:
                return self.error_json(HTTPStatus.SERVICE_UNAVAILABLE, "검출 기능이 꺼져 있습니다")
            kind, _, run_id = sub.partition("/")
            if kind not in ("status", "stream"):
                return self.error_json(HTTPStatus.NOT_FOUND, "없는 주소")
            run = detector.get(run_id)
            if run is None:
                return self.error_json(HTTPStatus.NOT_FOUND, "없는 실행")
            if kind == "status":
                return self.send_json(run.status())
            if run.streamed:
                return self.error_json(HTTPStatus.CONFLICT, "이미 보고 있는 실행입니다")
            run.streamed = True
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                for jpg in run.frames():
                    head = f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpg)}\r\n\r\n"
                    self.wfile.write(head.encode("ascii") + jpg + b"\r\n")
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                run.stop()  # 브라우저가 화면을 닫으면 멈춘다

        # ------------------------------------------------ 파일 전송 (Range 지원)
        def send_file(self, p: Path, ctype: str, allow_range: bool = False):
            if not p.is_file():
                return self.error_json(HTTPStatus.NOT_FOUND, "파일 없음")
            size = p.stat().st_size
            start, end = 0, size - 1
            rng = self.headers.get("Range") if allow_range else None
            if rng:
                r = parse_range(rng, size)
                if r is None:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                start, end = r
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(end - start + 1))
            if allow_range:
                self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                with open(p, "rb") as f:
                    f.seek(start)
                    left = end - start + 1
                    while left > 0:
                        buf = f.read(min(CHUNK, left))
                        if not buf:
                            break
                        self.wfile.write(buf)
                        left -= len(buf)
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                pass  # 브라우저가 이동하면서 요청을 끊는 것은 정상

    return Handler


def main():
    ap = argparse.ArgumentParser(description="웹 ROI 편집기 서버")
    ap.add_argument("--videos", default="data/videos", help="영상 폴더")
    ap.add_argument("--roi-dir", default="configs/roi", help="ROI JSON 저장 폴더")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--config", default="configs/default.yaml", help="YOLO 검출 설정")
    ap.add_argument("--no-detect", action="store_true", help="YOLO 검출 기능 끄기 (편집만)")
    args = ap.parse_args()

    # 상대 경로는 실행 위치가 아니라 프로젝트 폴더 기준으로 푼다
    video_dir, roi_dir = (ROOT / args.videos).resolve(), (ROOT / args.roi_dir).resolve()
    detector = None if args.no_detect else DetectManager((ROOT / args.config).resolve())
    handler = make_handler(video_dir, roi_dir, ROOT / "web", detector)
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    n = len([p for p in video_dir.glob("*") if p.suffix.lower() in VIDEO_EXTS]) if video_dir.is_dir() else 0
    print(f"영상 폴더: {video_dir} ({n}개)\nROI 저장 폴더: {roi_dir}")
    print(f"브라우저에서 http://127.0.0.1:{args.port} 을 여세요. 종료: Ctrl+C")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
