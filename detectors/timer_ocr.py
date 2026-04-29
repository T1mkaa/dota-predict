"""Game-clock OCR.

Reads the central HUD timer (M:SS / MM:SS) and reports in-game time in
seconds. Used in two modes:

  - One-shot calibration (calibrate_offset): probe N frames near the start
    of a VOD, take the median (raw_timer - video_time) as a constant
    offset. Then game_time = video_time + offset for the whole run.
    This is the demo-mode default: simple and robust as long as the VOD
    has no in-match pauses (verified true for our reference chunk).

  - Continuous read_raw on a single frame for ad-hoc inspection or
    future live-OCR work.

The state-machine update() with pause/jump detection is left in for
future continuous-OCR work but is NOT used by the demo pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from detectors.score_ocr import _normalize  # reuse digit normalization

log = logging.getLogger("detectors.timer_ocr")

DIGIT_H = 30
DIGIT_W = 24


@dataclass
class TimerConfig:
    roi_px: tuple[int, int, int, int] = (915, 10, 1005, 55)
    ref_width: int = 1920
    ref_height: int = 1080
    bin_threshold: int = 200
    min_h: int = 6
    max_h: int = 14
    min_w: int = 3
    max_w: int = 10
    match_min_conf: float = 0.65
    templates_dir: str = "assets/digits_timer"
    pause_repeat_n: int = 3        # same value N times in a row → paused
    delta_tolerance: int = 1       # |raw_delta - video_delta| must be ≤ this
    new_match_drop: int = 60       # backward jump >= this → treat as new match


@dataclass
class TimerState:
    last_game_time: int | None = None
    last_video_time: float | None = None
    repeat_count: int = 0
    paused: bool = False


class TimerDetector:
    def __init__(self, templates_dir: Path | str, cfg: TimerConfig | None = None):
        self.cfg = cfg or TimerConfig()
        self.templates: dict[str, list[np.ndarray]] = {}
        self.state = TimerState()
        self._load_templates(Path(templates_dir))

    def _load_templates(self, path: Path) -> None:
        if not path.exists():
            log.warning("timer: digits dir missing: %s", path)
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
        log.info("timer: loaded %d digit templates across %d classes", total, len(self.templates))

    def _scaled_roi(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        sx, sy = w / self.cfg.ref_width, h / self.cfg.ref_height
        x1, y1, x2, y2 = self.cfg.roi_px
        return frame_bgr[int(y1 * sy): int(y2 * sy), int(x1 * sx): int(x2 * sx)]

    def _segment(self, roi_bgr: np.ndarray) -> list[tuple[int, np.ndarray]]:
        """Return list of (x, mask) for digit-sized contours, left-to-right."""
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self.cfg.bin_threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if self.cfg.min_h <= h <= self.cfg.max_h and self.cfg.min_w <= w <= self.cfg.max_w:
                boxes.append((x, y, w, h))
        if not boxes:
            return []
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
        return [(x, mask[y:y + h, x:x + w]) for x, y, w, h in boxes]

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

    def read_raw(self, frame_bgr: np.ndarray) -> int | None:
        """Single-frame OCR: return game-time in seconds or None."""
        roi = self._scaled_roi(frame_bgr)
        digits = self._segment(roi)
        if not digits:
            return None
        # Expected: 3 digits (M:SS) or 4 digits (MM:SS). The colon should
        # have been filtered out by the height check.
        if len(digits) not in (3, 4):
            return None
        chars: list[str] = []
        for _, mask in digits:
            d, sc = self._classify(mask)
            if d is None or sc < self.cfg.match_min_conf:
                return None
            chars.append(d)
        s = "".join(chars)
        if len(s) == 3:
            mm, ss = int(s[0]), int(s[1:])
        else:
            mm, ss = int(s[:2]), int(s[2:])
        if ss >= 60:
            return None
        return mm * 60 + ss

    def update(self, frame_bgr: np.ndarray, video_time: float) -> dict:
        """Read timer + update pause/match state.

        Returns dict {game_time, paused, source, ok}.
        source: "ocr" (fresh read accepted), "extrapolate" (interpolated),
                "frozen" (paused), "miss" (no signal).
        """
        raw = self.read_raw(frame_bgr)
        st = self.state

        if raw is None:
            if st.last_game_time is None:
                return {"game_time": None, "paused": False, "source": "miss", "ok": False}
            if st.paused:
                return {"game_time": st.last_game_time, "paused": True, "source": "frozen", "ok": True}
            dv = video_time - (st.last_video_time or video_time)
            return {
                "game_time": int(st.last_game_time + max(0, dv)),
                "paused": False,
                "source": "extrapolate",
                "ok": True,
            }

        # Got a fresh reading. Validate against history.
        prev = st.last_game_time
        prev_vt = st.last_video_time
        if prev is not None and prev_vt is not None:
            game_delta = raw - prev
            video_delta = video_time - prev_vt
            if game_delta <= -self.cfg.new_match_drop:
                # Big backward step → new match started.
                log.info("timer: match boundary detected (%ds → %ds), resetting", prev, raw)
                st.last_game_time = raw
                st.last_video_time = video_time
                st.repeat_count = 1
                st.paused = False
                return {"game_time": raw, "paused": False, "source": "ocr", "ok": True}
            # Plausible if either: roughly tracking video time (active),
            # or game time held still (paused).
            active_match = abs(game_delta - video_delta) <= self.cfg.delta_tolerance
            paused_match = (game_delta == 0 and video_delta > 0)
            if not (active_match or paused_match):
                # Implausible single-frame OCR error. Fall back to extrapolate
                # but DO NOT poison state with the bad reading.
                if st.paused:
                    return {"game_time": prev, "paused": True, "source": "frozen", "ok": True}
                return {
                    "game_time": int(prev + max(0, video_delta)),
                    "paused": False,
                    "source": "extrapolate",
                    "ok": True,
                }

        if prev is not None and raw == prev:
            st.repeat_count += 1
            if st.repeat_count >= self.cfg.pause_repeat_n:
                st.paused = True
        else:
            st.repeat_count = 1
            st.paused = False

        st.last_game_time = raw
        st.last_video_time = video_time
        return {"game_time": raw, "paused": st.paused, "source": "ocr", "ok": True}


def calibrate_offset(
    video_path: str | Path,
    templates_dir: str | Path = "assets/digits_timer",
    probe_start: float = 2.0,
    probe_end: float = 60.0,
    samples: int = 30,
    min_consensus: int = 5,
    tolerance: int = 1,
) -> int | None:
    """Probe N frames between probe_start..probe_end seconds of video.

    For each successful raw read, compute candidate = raw_timer - video_t.
    Cluster candidates within ±tolerance seconds; if the largest cluster
    has ≥ min_consensus members, return its median as the offset.

    Returns int seconds offset such that game_time = video_time + offset,
    or None if calibration cannot agree on a value.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error("calibrate: cannot open %s", video_path)
        return None

    det = TimerDetector(templates_dir)
    candidates: list[int] = []
    step = (probe_end - probe_start) / max(1, samples - 1)
    try:
        for i in range(samples):
            t = probe_start + i * step
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            raw = det.read_raw(frame)
            if raw is None:
                continue
            candidates.append(int(round(raw - t)))
    finally:
        cap.release()

    if not candidates:
        log.warning("calibrate: no successful reads")
        return None

    # Find the largest cluster within ±tolerance.
    candidates.sort()
    best_cluster: list[int] = []
    for i, c in enumerate(candidates):
        cluster = [x for x in candidates if abs(x - c) <= tolerance]
        if len(cluster) > len(best_cluster):
            best_cluster = cluster

    if len(best_cluster) < min_consensus:
        log.warning(
            "calibrate: weak consensus — %d candidates, best cluster size %d (need %d)",
            len(candidates), len(best_cluster), min_consensus,
        )
        return None

    offset = sorted(best_cluster)[len(best_cluster) // 2]
    log.info(
        "calibrate: offset=%ds (cluster %d/%d, video_path=%s)",
        offset, len(best_cluster), len(candidates), video_path,
    )
    return offset
