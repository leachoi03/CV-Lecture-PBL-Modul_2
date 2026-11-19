"""위반 이벤트 감지 성능 평가 (Precision / Recall / F1).

정답 CSV(사람이 직접 기록): time_sec,event_type[,note]
예측 CSV: src.main 이 만든 events.csv

같은 종류의 이벤트끼리, 시간 차이가 tol 초 이내인 쌍을 가까운 순서대로 1:1 매칭한다.

    python -m eval.event_eval --pred outputs/day01_xxx/events.csv --gt data/gt/day01.csv --tol 1.5
"""
import argparse
import csv
import json
from collections import defaultdict


def read_times(path):
    by_type = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_type[row["event_type"].strip()].append(float(row["time_sec"]))
    return by_type


def match(pred, gt, tol):
    """시간 차이가 작은 쌍부터 그리디 매칭. (TP, FP, FN) 반환."""
    pairs = sorted(
        (abs(p - g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt) if abs(p - g) <= tol
    )
    used_p, used_g = set(), set()
    for _, i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
    tp = len(used_p)
    return tp, len(pred) - tp, len(gt) - tp


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def evaluate(pred_path, gt_path, tol):
    pred, gt = read_times(pred_path), read_times(gt_path)
    results = {}
    total = [0, 0, 0]
    for etype in sorted(set(pred) | set(gt)):
        tp, fp, fn = match(pred.get(etype, []), gt.get(etype, []), tol)
        total = [total[0] + tp, total[1] + fp, total[2] + fn]
        results[etype] = dict(zip(("tp", "fp", "fn", "precision", "recall", "f1"), (tp, fp, fn, *prf(tp, fp, fn))))
    results["all"] = dict(zip(("tp", "fp", "fn", "precision", "recall", "f1"), (*total, *prf(*total))))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--tol", type=float, default=1.5, help="시간 허용 오차(초)")
    ap.add_argument("--save", help="결과 JSON 저장 경로")
    args = ap.parse_args()

    res = evaluate(args.pred, args.gt, args.tol)
    print(f"{'type':10s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'P':>6s} {'R':>6s} {'F1':>6s}")
    for k, v in res.items():
        print(f"{k:10s} {v['tp']:4d} {v['fp']:4d} {v['fn']:4d} {v['precision']:6.3f} {v['recall']:6.3f} {v['f1']:6.3f}")
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
