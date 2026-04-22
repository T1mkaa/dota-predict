"""Roshan detector via template matching.

Используем два независимых визуальных сигнала:
  1. Надпись "Roshan has been slain" — ярко-жёлтый текст в центре-верху
     экрана, висит ~3 секунды.
  2. Иконка Aegis of the Immortal, появляющаяся в инвентаре героя-поднявшего.
     Висит до ~5 минут (либо до применения) — это «долгий» сигнал.

Для каждого шаблона ведём свой cooldown. Для Aegis cooldown длинный,
чтобы не триггерить событие всё время пока иконка висит.

Шаблоны (PNG, grayscale на диске не обязателен — читаем как color и
конвертим) кладём в assets/templates/:
  - `roshan_slain.png` — вырезанная из матча надпись
  - `aegis.png`        — иконка артефакта

Если шаблон отсутствует, соответствующий канал просто выключается.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("detectors.roshan")


@dataclass
class RoshanConfig:
    slain_roi_rel: tuple[float, float, float, float] = (0.20, 0.08, 0.80, 0.35)
    aegis_roi_rel: tuple[float, float, float, float] = (0.35, 0.85, 0.75, 0.99)
    slain_threshold: float = 0.78
    aegis_threshold: float = 0.80
    slain_cooldown_sec: float = 180.0
    aegis_cooldown_sec: float = 600.0
    scales: tuple[float, ...] = (0.7, 0.85, 1.0, 1.15)


class RoshanDetector:
    def __init__(self, templates_dir: str | Path, cfg: RoshanConfig | None = None):
        self.cfg = cfg or RoshanConfig()
        self._slain_tpls: list[np.ndarray] = []
        self._aegis_tpls: list[np.ndarray] = []
        self._last_slain: float = float("-inf")
        self._last_aegis: float = float("-inf")

        root = Path(templates_dir)
        self._load("roshan_slain.png", root, self._slain_tpls, base_height=60)
        self._load("aegis.png", root, self._aegis_tpls, base_height=40)
        log.info("roshan: slain variants=%d aegis variants=%d",
                 len(self._slain_tpls), len(self._aegis_tpls))

    def _load(self, fname: str, root: Path, bucket: list[np.ndarray], base_height: int) -> None:
        p = root / fname
        if not p.exists():
            log.warning("roshan: missing template %s", p)
            return
        raw = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if raw is None:
            return
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        for s in self.cfg.scales:
            h = max(16, int(base_height * s))
            w = max(16, int(gray.shape[1] * (h / gray.shape[0])))
            bucket.append(cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA))

    @staticmethod
    def _crop(frame_bgr: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = box
        sub = frame_bgr[int(h * y1): int(h * y2), int(w * x1): int(w * x2)]
        return cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _best_match(roi: np.ndarray, templates: list[np.ndarray]) -> float:
        best = 0.0
        for tpl in templates:
            if tpl.shape[0] > roi.shape[0] or tpl.shape[1] > roi.shape[1]:
                continue
            res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(res)
            if score > best:
                best = score
        return float(best)

    def detect(self, frame_bgr: np.ndarray, now: float | None = None) -> Optional[tuple[str, float]]:
        """Return ('slain'|'aegis', score) or None."""
        now = now or time.time()

        if self._slain_tpls:
            roi = self._crop(frame_bgr, self.cfg.slain_roi_rel)
            if roi.size:
                s = self._best_match(roi, self._slain_tpls)
                if s >= self.cfg.slain_threshold and now - self._last_slain >= self.cfg.slain_cooldown_sec:
                    self._last_slain = now
                    return ("slain", s)

        if self._aegis_tpls:
            roi = self._crop(frame_bgr, self.cfg.aegis_roi_rel)
            if roi.size:
                s = self._best_match(roi, self._aegis_tpls)
                if s >= self.cfg.aegis_threshold and now - self._last_aegis >= self.cfg.aegis_cooldown_sec:
                    self._last_aegis = now
                    return ("aegis", s)

        return None
