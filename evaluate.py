"""Compare detector output against ground-truth timestamps.

Usage:
  python evaluate.py <video> <groundtruth.txt>

Ground truth file format (one per line):
  MM:SS kind       # kind = kill (roshan rows are ignored in v0.1)
Or:
  HH:MM:SS kind

v0.1 evaluates kill-only — Roshan detection is parked for v0.2.

Matching: a ground-truth event is counted as recalled if the detector fires
the same kind within ±TOLERANCE_SEC of its timestamp. A detector event is
counted as precise if it maps to some ground-truth event. The precision
denominator includes all detector events; the recall denominator, all GT
events.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

from detectors.gameplay_gate import GameplayGate
from detectors.score_ocr import ScoreDetector


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("evaluate")

ASSETS = Path(__file__).parent / "assets"
TOLERANCE_SEC = 10.0


def parse_ts(token: str) -> float:
    parts = [int(p) for p in token.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"bad timestamp: {token}")


def load_gt(path: Path) -> list[tuple[float, str]]:
    out: list[tuple[float, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        ts_tok, kind = line.split()[:2]
        kind = kind.lower()
        if kind != "kill":
            continue
        out.append((parse_ts(ts_tok), kind))
    out.sort()
    return out


def detect_events(video_path: Path) -> list[tuple[float, str, str]]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sc = ScoreDetector(ASSETS / "digits")
    gate = GameplayGate()
    step = max(1, int(round(fps)))
    events: list[tuple[float, str, str]] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % step:
            continue
        ts = idx / fps
        if not gate.is_in_game(frame):
            continue
        for name, score in sc.detect(frame, now=ts):
            events.append((ts, "kill", f"{name}({score:.2f})"))
    cap.release()
    return events


def evaluate(gt: list[tuple[float, str]], det: list[tuple[float, str, str]]) -> None:
    gt_matched = [False] * len(gt)
    det_matched = [False] * len(det)

    for i, (d_ts, d_kind, d_src) in enumerate(det):
        for j, (g_ts, g_kind) in enumerate(gt):
            if gt_matched[j] or g_kind != d_kind:
                continue
            if abs(d_ts - g_ts) <= TOLERANCE_SEC:
                gt_matched[j] = True
                det_matched[i] = True
                break

    tp = sum(det_matched)
    fp = len(det) - tp
    fn = sum(1 for m in gt_matched if not m)
    precision = tp / max(1, len(det))
    recall = tp / max(1, len(gt))
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    print("\n===== RESULTS =====")
    print(f"ground truth events: {len(gt)}")
    print(f"detector events:     {len(det)}")
    print(f"true positives:      {tp}")
    print(f"false positives:     {fp}")
    print(f"false negatives:     {fn}")
    print(f"precision:           {precision:.2%}")
    print(f"recall:              {recall:.2%}")
    print(f"F1:                  {f1:.2%}")

    gt_k = [j for j, (_, k) in enumerate(gt) if k == "kill"]
    det_k = [i for i, (_, k, _) in enumerate(det) if k == "kill"]
    tp_k = sum(1 for j in gt_k if gt_matched[j])
    fp_k = sum(1 for i in det_k if not det_matched[i])
    fn_k = len(gt_k) - tp_k
    p_k = tp_k / max(1, len(det_k))
    r_k = tp_k / max(1, len(gt_k))
    print(f"\n  kill: gt={len(gt_k)} det={len(det_k)} tp={tp_k} fp={fp_k} fn={fn_k}  P={p_k:.2%} R={r_k:.2%}")

    # Show misses
    misses = [(gt[j][0], gt[j][1]) for j in range(len(gt)) if not gt_matched[j]]
    if misses:
        print(f"\n--- MISSED by detector ({len(misses)}):")
        for ts, kind in misses:
            m, s = int(ts // 60), int(ts % 60)
            print(f"  {m:02d}:{s:02d}  {kind}")

    fps_list = [(det[i][0], det[i][1], det[i][2]) for i in range(len(det)) if not det_matched[i]]
    if fps_list:
        print(f"\n--- FALSE POSITIVES ({len(fps_list)}):")
        for ts, kind, src in fps_list:
            m, s = int(ts // 60), int(ts % 60)
            print(f"  {m:02d}:{s:02d}  {kind:6s}  {src}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("gt")
    args = ap.parse_args()
    vp = Path(args.video)
    gp = Path(args.gt)
    if not vp.exists() or not gp.exists():
        print("missing file")
        sys.exit(1)
    gt = load_gt(gp)
    log.info("ground truth loaded: %d events", len(gt))
    det = detect_events(vp)
    log.info("detector produced: %d events", len(det))
    evaluate(gt, det)


if __name__ == "__main__":
    main()
