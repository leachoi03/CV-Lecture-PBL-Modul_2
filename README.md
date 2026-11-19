# 도심 교차로 CCTV 교통 위반 감지 (PBL 모듈 2)

YOLOv8으로 차량·이륜차·보행자를 검출하고, ByteTrack으로 고유 ID를 유지하며,
궤적을 분석해 **차량 정지선 위반**과 **보행자 무단횡단**을 감지한다.

```
영상 → [전처리(CLAHE, 선택)] → YOLOv8 검출 → ByteTrack 추적(ID) → 궤적 저장
     → 신호 상태 + ROI → 정지선 / 무단횡단 규칙 → 시각화 · events.csv · 스냅샷 · result.mp4
```

## 설치

Python 3.10 이상 (3.14.5에서 동작 확인). [uv](https://docs.astral.sh/uv/)를 쓰는 경우:

```bash
uv venv .venv --python 3.14
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
```

uv 없이 설치할 때:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

아래 명령은 모두 가상환경을 활성화한 상태(또는 `.venv\Scripts\python.exe`로 직접 실행)에서 실행한다.

NVIDIA GPU가 있으면 [PyTorch 설치 페이지](https://pytorch.org/get-started/locally/)에서 CUDA 버전을 먼저 설치한다.
모델 가중치(`yolov8s.pt` 등)는 처음 실행할 때 자동으로 다운로드된다.

## 사용 순서

1. 영상을 `data/videos/`에 넣는다. (얼굴·번호판이 식별되는 영상은 사용하지 않거나 블러 처리)
2. 영상마다 ROI를 만든다.
   ```bash
   python -m src.roi_tool --source data/videos/day01.mp4 --out configs/roi/day01.json
   ```
   만든 JSON의 `signal.timeline`에 신호 시간표(초 단위)를 입력한다. 신호등이 화면에 보이면 `l` 키로 신호등 영역을 지정해 HSV 자동 판별을 쓸 수 있다.
3. 실행한다.
   ```bash
   python -m src.main --source data/videos/day01.mp4 --roi configs/roi/day01.json --show
   ```
   결과는 `outputs/<영상이름_시각>/`에 저장된다: `result.mp4`, `events.csv`, `snapshots/`, `summary.json`(처리 FPS, 이벤트 수, 사용 설정).
4. 평가한다.
   ```bash
   python -m eval.event_eval --pred outputs/<폴더>/events.csv --gt data/gt/day01.csv --tol 1.5
   python -m eval.benchmark --source data/videos/day01.mp4 --models yolov8n.pt yolov8s.pt --trackers bytetrack.yaml botsort.yaml
   ```
5. 판정 로직 단위 테스트: `python -m pytest tests -q`

실험할 때 자주 바꾸는 값은 명령줄에서 덮어쓸 수 있다: `--model yolov8m.pt --tracker botsort.yaml --conf 0.15 --clahe`.

### 이미지 폴더 입력

`--source`에 영상 대신 JPG/PNG가 든 폴더를 줄 수 있다(`src.main`, `src.roi_tool`, `eval.benchmark` 공통). 파일 이름의 숫자 순서(자연 정렬)로 읽고, fps 정보가 없으므로 `--fps`로 지정한다(기본 10). 하위 폴더까지 읽으려면 `--recursive`. 한글 경로도 된다.

```bash
python -m src.main --source data/frames/seq01 --fps 10 --roi configs/roi/seq01.json
```

**주의 – AI Hub 094 원천데이터**: 파일 이름이 `카메라_클립시각_프레임번호.jpg`이고, 한 폴더에 여러 카메라·클립의 프레임이 2~3장씩 띄엄띄엄 섞여 있다. 연속 영상이 아니므로 추적 ID와 위반 판정은 의미가 없다(실행 시 경고가 뜬다). 연속 프레임에서만 추적기가 객체를 확정하기 때문에, 움직이는 차량은 아예 표시되지 않고 정지한 노면 표시 오검출만 남을 수 있다. 이 데이터는 프레임 단위 검출 성능 확인용으로 쓰고, 위반 감지는 공통 영상과 직접 촬영한 영상으로 평가한다.

### 프레임 단위 검출 평가 (AI Hub 094 VS_4 + CVAT XML)

추적 없이 이미지마다 검출만 해서 정답 박스와 비교한다. 모델 크기·입력 크기·CLAHE 비교 실험용.

```bash
python -m eval.det_eval --images "data/094/01.데이터/2.Validation/원천데이터/VS_4.바운딩박스" --labels "data/094/01.데이터/2.Validation/라벨링데이터/VL_4.바운딩박스" --models yolov8n.pt yolov8s.pt --imgsz 640 1280 --per-group 60
```

- 범주: `vehicle`(car·truck·bus 계열), `two_wheeler`(bike). `unknown` 라벨은 무시 영역. 라벨 목록에 face·car_plate·person 이 있지만 실제 박스는 없다.
- 그룹: 전체 / 폴더 조건(교차로·악천후·시간대) / 주간·야간(파일 이름의 촬영 시각, 18시~6시는 야간).
- 야간만 평가: `--only night` (예: `--only night --per-group 250 --clahe --out outputs/det_eval_night_clahe`).
- 지표: AP50, `--conf` 임계값에서 P/R/F1, 크기별(small<32², medium<96², large) 재현율, 추론 FPS.
- 결과: `outputs/det_eval/det_eval.csv`, 정답(초록)·예측(파랑)·라벨링 영역(노랑)을 그린 `vis_*/` 이미지.
- **해석 주의**: 이 데이터는 화면 일부의 차량만 라벨링되어 있고, 그 영역 안에서도 빠진 차량이 많다. 카메라별 라벨링 영역(GT 박스의 볼록 껍질) 밖 검출은 제외하지만(`--region none`으로 끔), 정밀도·AP는 여전히 실제보다 낮게 나오는 하한값이다. **재현율 위주로 비교**한다.

### VS Code에서 실행

`code .`로 프로젝트 폴더를 열면 `.vscode/settings.json`이 `.venv` 인터프리터를 자동으로 쓴다. 실행 및 디버그(Ctrl+Shift+D)에서 `.vscode/launch.json`의 항목(검출 평가, 야간+CLAHE, 파이프라인, ROI 도구)을 골라 F5로 실행한다. 테스트는 왼쪽 테스트(비커) 아이콘에서 돌린다.

## 폴더 구조

| 경로 | 내용 | 담당(예시) |
| --- | --- | --- |
| `src/frame_source.py` | 영상/이미지 폴더 입력 통합 | ① |
| `src/detector_tracker.py` | YOLOv8 + ByteTrack/BoT-SORT, 트랙 목록 반환 | ① |
| `src/trajectory.py` | ID별 지면 접점 궤적 | ① |
| `src/preprocess.py` | 야간용 CLAHE | ① |
| `src/roi.py` | 정지선·차도·횡단보도, 기하 판정 함수 | ② |
| `src/roi_tool.py` | 마우스로 ROI를 찍는 도구 | ② |
| `src/signal_state.py` | 신호 상태 (타임라인 / HSV / 고정) | ② |
| `src/rules/stopline.py` | 정지선 위반 규칙 | ② |
| `src/rules/jaywalk.py` | 무단횡단 규칙 | ③ |
| `src/visualize.py` | BBox·ID·궤적·ALERT·HUD 그리기 | ③ |
| `src/event_logger.py` | events.csv, 스냅샷 | ③ |
| `src/main.py` | 전체 파이프라인 실행 | ① |
| `eval/` | 이벤트 P/R/F1, 모델·추적기 속도 비교, 프레임 단위 검출 평가 | ④ |
| `tests/` | 규칙 단위 테스트 | 전원 |
| `configs/` | 실행 설정, 영상별 ROI | 전원 |
| `data/gt/` | 사람이 기록한 정답 이벤트 | ④ |
| `docs/ai_log.md` | AI 도구 활용 기록 (평가 항목) | 전원 |

## 위반 판정 기준 (configs/default.yaml 에서 조정)

**정지선 위반**: 적신호 중에 차량·이륜차의 지면 접점(BBox 하단 중앙)이 정지선 앞에서 너머로 넘어가고,
그 상태가 `stopline.confirm_frames` 프레임 동안 유지되면 위반이다. 적신호 전에 이미 넘어가 있던 차량과 황색 신호 중 통과는 제외한다. 같은 ID는 한 번만 기록한다.

**무단횡단**: 보행자의 지면 접점이 '차도 안이면서 횡단보도 밖'에 `jaywalk.min_seconds` 초 이상 연속으로 있으면 위반이다.
이륜차 박스와 `jaywalk.rider_overlap` 이상 겹치는 사람은 탑승자로 보고 제외한다.

## 알려진 한계 (보고서에서 분석할 것)

- 신호 상태는 타임라인 입력이 기본이라, 신호 주기가 바뀌는 영상에서는 시간표를 다시 입력해야 한다.
- 트랙 ID가 바뀌면(가림 후 재등장) 정지선 '앞→너머' 전환을 놓치거나, 무단횡단 연속 프레임 수가 초기화될 수 있다.
- ROI는 카메라가 고정되어 있다는 가정에서만 유효하다.
