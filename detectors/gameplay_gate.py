"""In-game gate: decide whether the current frame is actual gameplay.

Два независимых чекера, обе проверки должны пройти:

1) **Минимапа в нижнем левом углу** — сложная мозаика из светло- и тёмно-
   зелёной земли, реки, двух баз разных цветов, точек героев. std цветов
   высокое. На меню-кадрах тот же угол либо равномерно тёмный, либо
   содержит ровный логотип — std низкий.

2) **Отсутствие Dota-логотипа в верхнем левом углу** — в клиентских меню
   (главный экран, профиль, арсенал, экран выбора героя) у самого верха
   слева висит большой красно-оранжевый треугольник логотипа Dota 2.
   В игре его никогда нет. Ищем характерный красно-оранжевый hue в зоне
   верхнего левого — если доля таких пикселей высокая, это меню.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger("detectors.gameplay")


@dataclass
class GameplayGateConfig:
    minimap_roi_rel: tuple[float, float, float, float] = (0.005, 0.78, 0.15, 0.99)
    min_color_std: float = 28.0
    """Standard deviation across BGR channels in the minimap ROI.
    In-game minimaps: ~35-55. Menu/dark screens: <20."""

    logo_roi_rel: tuple[float, float, float, float] = (0.05, 0.00, 0.22, 0.10)
    logo_max_orange_ratio: float = 0.03
    """Fraction of pixels in logo ROI that are bright red-orange.
    The Dota-2 logo lights this up to 0.10-0.30. In-game HUD stays <0.02."""


class GameplayGate:
    def __init__(self, cfg: GameplayGateConfig | None = None):
        self.cfg = cfg or GameplayGateConfig()

    def _minimap_complex(self, frame_bgr: np.ndarray) -> bool:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = self.cfg.minimap_roi_rel
        patch = frame_bgr[int(h * y1): int(h * y2), int(w * x1): int(w * x2)]
        if patch.size == 0:
            return False
        return float(patch.std()) >= self.cfg.min_color_std

    def _no_dota_logo(self, frame_bgr: np.ndarray) -> bool:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = self.cfg.logo_roi_rel
        patch = frame_bgr[int(h * y1): int(h * y2), int(w * x1): int(w * x2)]
        if patch.size == 0:
            return True
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 130, 140), (18, 255, 255))
        ratio = float(mask.sum() / 255) / mask.size
        return ratio < self.cfg.logo_max_orange_ratio

    def is_in_game(self, frame_bgr: np.ndarray) -> bool:
        return self._minimap_complex(frame_bgr) and self._no_dota_logo(frame_bgr)
