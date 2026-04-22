"""Score-based kill detector.

Reads the numeric score at the top-center HUD (Radiant N - M Dire) and emits
one 'kill' event per confirmed integer increment. This is an orthogonal
signal to kill-feed icon matching: the scoreboard is authoritative — if it
ticks up, a kill happened, full stop.

Pipeline per frame:
  1. Crop left and right score ROIs (1920x1080 reference; scaled per frame).
  2. Binarize (bright white glyphs on dark HUD) and segment digit contours.
  3. Classify each digit against multi-template library via cosine-similarity
     on binary masks (templates pre-normalized to 30x24).
  4. Reassemble integer. Require N consecutive identical reads before
     confirming — kills the transient flicker during the +1 animation.
  5. When confirmed value exceeds previous confirmed value, emit that many
     kill events (handles fast double/triple kills in teamfights).

Drops and implausible jumps (>3 per sample) are treated as noise and reset
the baseline without firing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("detectors.score_ocr")

DIGIT_H = 30
DIGIT_W = 24


@dataclass
class ScoreConfig:
    roi_left_px: tuple[int, int, int, int] = (830, 0, 945, 65)
    roi_right_px: tuple[int, int, int, int] = (990, 0, 1105, 65)
    ref_width: int = 1920
    ref_height: int = 1080
    bin_threshold: int = 200
    min_h: int = 10
    max_h: int = 22
    min_w: int = 5
    max_w: int = 18
    max_digits: int = 3
    match_min_conf: float = 0.80
    stable_frames: int = 1
    max_jump: int = 5


@dataclass
class _Side:
    last_confirmed: int | None = None
    pending_value: int | None = None
    pending_count: int = 0


def _normalize(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    if h != DIGIT_H:
        new_w = max(1, int(round(w * DIGIT_H / h)))
        mask = cv2.resize(mask, (new_w, DIGIT_H), interpolation=cv2.INTER_AREA)
    w = mask.shape[1]
    if w >= DIGIT_W:
        start = (w - DIGIT_W) // 2
        return mask[:, start:start + DIGIT_W]
    padded = np.zeros((DIGIT_H, DIGIT_W), dtype=mask.dtype)
    off = (DIGIT_W - w) // 2
    padded[:, off:off + w] = mask
    return padded


class ScoreDetector:
    def __init__(self, templates_dir: Path | str, cfg: ScoreConfig | None = None):
        self.cfg = cfg or ScoreConfig()
        self.templates: dict[str, list[np.ndarray]] = {}
        self.radiant = _Side()
        self.dire = _Side()
        self._load_templates(Path(templates_dir))

    def _load_templates(self, path: Path) -> None:
        if not path.exists():
            log.warning("score: digits dir missing: %s", path)
            return
        for f in sorted(path.glob("*.png")):
            digit = f.stem.split("_")[0]
            if digit not in "0123456789":
                continue
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            _, bin_mask = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
            flat = (_normalize(bin_mask) > 128).astype(np.float32).flatten()
            self.templates.setdefault(digit, []).append(flat)
        total = sum(len(v) for v in self.templates.values())
        log.info("score: loaded %d digit templates across %d digit classes", total, len(self.templates))

    def _scaled_roi(self, frame_bgr: np.ndarray, box_px: tuple[int, int, int, int]) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        sx, sy = w / self.cfg.ref_width, h / self.cfg.ref_height
        x1, y1, x2, y2 = box_px
        return frame_bgr[int(y1 * sy): int(y2 * sy), int(x1 * sx): int(x2 * sx)]

    def _segment(self, roi_bgr: np.ndarray) -> list[np.ndarray]:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self.cfg.bin_threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rh = roi_bgr.shape[0]
        boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if self.cfg.min_h <= h <= self.cfg.max_h and self.cfg.min_w <= w <= self.cfg.max_w:
                boxes.append((x, y, w, h))
        if not boxes:
            return []
        # Score digits live on a single horizontal line. Find the tightest
        # y-cluster (most boxes within ±6 px of some center) and keep only
        # those. Ties broken by preferring the upper cluster (score is at
        # the top of the HUD; any second cluster below is noise).
        ys = [y + h / 2 for x, y, w, h in boxes]
        best_center = min(ys)
        best_count = 0
        for c in ys:
            cnt = sum(1 for y in ys if abs(y - c) <= 6)
            if cnt > best_count or (cnt == best_count and c < best_center):
                best_count = cnt
                best_center = c
        boxes = [b for b, yc in zip(boxes, ys) if abs(yc - best_center) <= 6]
        boxes.sort(key=lambda b: b[0])
        return [mask[y:y + h, x:x + w] for x, y, w, h in boxes]

    def _classify(self, digit_mask: np.ndarray) -> tuple[str | None, float]:
        q = (_normalize(digit_mask) > 128).astype(np.float32).flatten()
        q_norm = float(np.sqrt((q * q).sum()))
        if q_norm == 0:
            return None, 0.0
        best_d: str | None = None
        best_sc = 0.0
        for d, tpls in self.templates.items():
            for t in tpls:
                t_norm = float(np.sqrt((t * t).sum()))
                if t_norm == 0:
                    continue
                sc = float(q @ t) / (q_norm * t_norm)
                if sc > best_sc:
                    best_sc = sc
                    best_d = d
        return best_d, best_sc

    def _read_number(self, roi_bgr: np.ndarray) -> int | None:
        digits = self._segment(roi_bgr)
        if not digits or len(digits) > self.cfg.max_digits:
            return None
        chars: list[str] = []
        for d_mask in digits:
            d, sc = self._classify(d_mask)
            if d is None or sc < self.cfg.match_min_conf:
                return None
            chars.append(d)
        try:
            return int("".join(chars))
        except ValueError:
            return None

    def _update_side(self, side: _Side, value: int | None) -> int:
        if value is None:
            side.pending_value = None
            side.pending_count = 0
            return 0
        if value == side.pending_value:
            side.pending_count += 1
        else:
            side.pending_value = value
            side.pending_count = 1
        if side.pending_count < self.cfg.stable_frames:
            return 0
        if side.last_confirmed is None:
            side.last_confirmed = value
            return 0
        delta = value - side.last_confirmed
        if delta <= 0 or delta > self.cfg.max_jump:
            side.last_confirmed = value
            return 0
        side.last_confirmed = value
        return delta

    def detect(self, frame_bgr: np.ndarray, now: float | None = None) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        left = self._scaled_roi(frame_bgr, self.cfg.roi_left_px)
        right = self._scaled_roi(frame_bgr, self.cfg.roi_right_px)
        rv = self._read_number(left)
        dv = self._read_number(right)
        for _ in range(self._update_side(self.radiant, rv)):
            out.append(("radiant", 1.0))
        for _ in range(self._update_side(self.dire, dv)):
            out.append(("dire", 1.0))
        return out
