"""Top-score detector.

В Dota 2 по центру верха HUD находится счёт команд (Radiant — Dire). Счёт
увеличивается **только** при убийстве героя (денай союзником не засчитывается,
смерть от крипов без участия героя — тоже). Это значит: если сумма цифр
выросла — в игре произошёл хотя бы один килл.

Зона счёта стримеры практически никогда не закрывают (она у самого верха,
а веб-камера обычно в правом верхнем или нижнем углу), поэтому сигнал
служит резервом когда kill-feed перекрыт оверлеями.

Реализация без OCR: для каждой из двух зон (левая цифра, правая цифра)
считается perceptual hash (average hash, 8×8). Если hamming-дистанция
между текущим и предыдущим кадром превышает порог — зона изменилась,
значит счёт обновился. Точное значение мы не восстанавливаем: для игры
важен только факт события и его момент.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import imagehash
import numpy as np
from PIL import Image

log = logging.getLogger("detectors.score")


@dataclass
class ScoreConfig:
    # Верхняя центральная панель. Числа расположены симметрично от центра.
    radiant_roi_rel: tuple[float, float, float, float] = (0.435, 0.005, 0.475, 0.045)
    dire_roi_rel: tuple[float, float, float, float] = (0.525, 0.005, 0.565, 0.045)
    hash_size: int = 8
    hamming_threshold: int = 18
    cooldown_sec: float = 5.0


class ScoreDetector:
    def __init__(self, cfg: ScoreConfig | None = None):
        self.cfg = cfg or ScoreConfig()
        self._prev_radiant: Optional[imagehash.ImageHash] = None
        self._prev_dire: Optional[imagehash.ImageHash] = None
        self._last_fire: float = 0.0

    def _crop(self, frame_bgr: np.ndarray, box: tuple[float, float, float, float]) -> Image.Image:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = box
        roi = frame_bgr[int(h * y1): int(h * y2), int(w * x1): int(w * x2)]
        return Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

    def detect(self, frame_bgr: np.ndarray, now: float | None = None) -> Optional[str]:
        """Return 'radiant' | 'dire' | None depending on which side's digit changed."""
        now = now or time.time()
        r_img = self._crop(frame_bgr, self.cfg.radiant_roi_rel)
        d_img = self._crop(frame_bgr, self.cfg.dire_roi_rel)

        r_hash = imagehash.phash(r_img, hash_size=self.cfg.hash_size)
        d_hash = imagehash.phash(d_img, hash_size=self.cfg.hash_size)

        # Compare against a "stable baseline" (updated only on confirmed change),
        # not against the immediate previous hash — otherwise any frame-to-frame
        # antialiasing jitter above the threshold fires repeatedly.
        changed: Optional[str] = None
        if self._prev_radiant is not None and (r_hash - self._prev_radiant) >= self.cfg.hamming_threshold:
            changed = "radiant"
        elif self._prev_dire is not None and (d_hash - self._prev_dire) >= self.cfg.hamming_threshold:
            changed = "dire"

        # Initialize baselines on first frame.
        if self._prev_radiant is None:
            self._prev_radiant = r_hash
        if self._prev_dire is None:
            self._prev_dire = d_hash

        if changed and (now - self._last_fire) >= self.cfg.cooldown_sec:
            self._last_fire = now
            # Advance the baseline *only* on a confirmed event, so subsequent
            # frames don't keep comparing against the old score.
            self._prev_radiant = r_hash
            self._prev_dire = d_hash
            return changed
        return None
