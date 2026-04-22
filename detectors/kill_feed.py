"""Kill-feed detector via paired edge-based hero-icon template matching.

Observation from ground-truth evaluation (22 real events vs 22 detector events
at F1=18%): single-icon matches are dominated by noise — specific hero
templates (batrider, lion) keep matching random edge patterns that happen
to be geometrically similar at the noise floor.

Key insight: a real kill-feed row shows EXACTLY TWO hero portraits side by
side — killer on the left, victim on the right, on the same vertical row.
Requiring the *pair* structure (two distinct heroes, similar y, distinct x)
is orthogonal to any individual template's noise floor.

Pipeline:
  1. Load icons → Canny-edge templates at multiple scales.
  2. Each frame: crop ROI, compute edge map, for every hero template find
     the best-matching (x, y, score) above threshold.
  3. From all candidates, find the best PAIR — two different heroes whose
     match positions are vertically close (|dy| ≤ max_dy) and horizontally
     separated (|dx| ≥ min_dx).
  4. If a pair is found, fire one kill event tagged with both heroes.

Dedup: a pair of (hero_a, hero_b) is not fired again within dedupe_window.
No more persistence filter — the pair requirement already kills transient
single-template noise without needing multiple frames.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("detectors.kill_feed")


@dataclass
class HeroTemplate:
    name: str
    edges: list[np.ndarray]


@dataclass
class KillFeedConfig:
    roi_rel: tuple[float, float, float, float] = (0.78, 0.08, 0.99, 0.22)
    match_threshold: float = 0.38
    """Minimum per-icon edge-NCC score to be considered a candidate.
    Lower than single-match threshold because pair requirement kills noise downstream."""
    dedupe_window_sec: float = 6.0
    scales: tuple[float, ...] = (0.50, 0.70, 0.90, 1.10, 1.30)
    target_icon_height: int = 40
    pair_max_dy_frac: float = 0.35
    """Max vertical gap between killer and victim, as fraction of ROI height."""
    pair_min_dx_frac: float = 0.15
    """Min horizontal gap between killer and victim — must be distinct icons."""
    canny_low: int = 60
    canny_high: int = 180


@dataclass
class Candidate:
    name: str
    score: float
    x: int
    y: int


class KillFeedDetector:
    def __init__(self, templates_dir: str | Path, cfg: KillFeedConfig | None = None):
        self.cfg = cfg or KillFeedConfig()
        self.templates: list[HeroTemplate] = []
        # key = frozenset({hero_a, hero_b}), value = last fire timestamp
        self._recent_pairs: dict[frozenset, float] = {}
        self._load_templates(Path(templates_dir))

    def _edges(self, gray: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        return cv2.Canny(blurred, self.cfg.canny_low, self.cfg.canny_high)

    def _load_templates(self, path: Path) -> None:
        if not path.exists():
            log.warning("templates dir missing: %s", path)
            return
        for f in sorted(path.glob("*.png")):
            raw = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if raw is None:
                continue
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            variants: list[np.ndarray] = []
            for s in self.cfg.scales:
                h = max(14, int(self.cfg.target_icon_height * s))
                w = max(14, int(gray.shape[1] * (h / gray.shape[0])))
                scaled = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
                edges = self._edges(scaled)
                if edges.sum() < 255 * 30:
                    continue
                variants.append(edges)
            if variants:
                self.templates.append(HeroTemplate(name=f.stem, edges=variants))
        log.info("kill-feed: loaded %d hero templates (edge+pair)", len(self.templates))

    def _crop_roi_gray(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = self.cfg.roi_rel
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return gray[int(h * y1): int(h * y2), int(w * x1): int(w * x2)]

    def detect_pairs(self, frame_bgr: np.ndarray, now: float | None = None) -> list[tuple[str, str, float]]:
        """Return list of (killer, victim, combined_score) pairs fired this frame."""
        now = now or time.time()
        gray = self._crop_roi_gray(frame_bgr)
        if gray.size == 0 or not self.templates:
            return []
        roi_edges = self._edges(gray)
        rh, rw = roi_edges.shape

        candidates: list[Candidate] = []
        for tpl in self.templates:
            best_score = 0.0
            best_pos = (0, 0)
            for tpl_edges in tpl.edges:
                if tpl_edges.shape[0] > rh or tpl_edges.shape[1] > rw:
                    continue
                res = cv2.matchTemplate(roi_edges, tpl_edges, cv2.TM_CCOEFF_NORMED)
                _, score, _, loc = cv2.minMaxLoc(res)
                if score > best_score:
                    best_score = score
                    # center of the matched region
                    best_pos = (loc[0] + tpl_edges.shape[1] // 2,
                                loc[1] + tpl_edges.shape[0] // 2)
            if best_score >= self.cfg.match_threshold:
                candidates.append(Candidate(tpl.name, float(best_score), best_pos[0], best_pos[1]))

        if len(candidates) < 2:
            return []

        max_dy = self.cfg.pair_max_dy_frac * rh
        min_dx = self.cfg.pair_min_dx_frac * rw

        best_pair: tuple[Candidate, Candidate] | None = None
        best_score = 0.0
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                a, b = candidates[i], candidates[j]
                if abs(a.y - b.y) > max_dy:
                    continue
                if abs(a.x - b.x) < min_dx:
                    continue
                pair_score = a.score + b.score
                if pair_score > best_score:
                    best_score = pair_score
                    best_pair = (a, b) if a.x <= b.x else (b, a)

        if best_pair is None:
            return []

        killer, victim = best_pair
        key = frozenset({killer.name, victim.name})
        last = self._recent_pairs.get(key, 0.0)
        if now - last < self.cfg.dedupe_window_sec:
            return []
        self._recent_pairs[key] = now
        return [(killer.name, victim.name, (killer.score + victim.score) / 2)]

    # Backward-compatibility shim: the rest of the codebase still calls .detect()
    # and expects a list of (hero, score). We emit two entries per pair so the
    # caller sees "both heroes were involved in a kill".
    def detect(self, frame_bgr: np.ndarray, now: float | None = None) -> list[tuple[str, float]]:
        pairs = self.detect_pairs(frame_bgr, now)
        out: list[tuple[str, float]] = []
        for killer, victim, sc in pairs:
            out.append((killer, sc))
            out.append((victim, sc))
        return out
