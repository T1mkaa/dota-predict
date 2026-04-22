"""Audio event detector via normalized cross-correlation.

Параллельно с видео-пайплайном запускается отдельный ffmpeg, который
извлекает из HLS-потока PCM-моно 16кГц. Скользящий буфер последних ~8
секунд прогоняется против каждого загруженного эталона (рёв Рошана,
"First Blood", "Double Kill" и т.д.) через нормированную кросс-корреляцию.
Если пик NCC превышает порог — срабатывает событие.

Нормировка делается через `numpy.correlate` с делением на скользящую
норму сигнала × норму эталона. Это эквивалент template matching для 1D
— порог совпадений не зависит от громкости.

Эталоны лежат как WAV в assets/sounds/. Канал отключается, если
соответствующий файл отсутствует.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional

import numpy as np
from scipy.io import wavfile

log = logging.getLogger("detectors.audio")


EVENT_FILES: dict[str, str] = {
    "roshan": "roshan_death.wav",
    "first_blood": "first_blood.wav",
    "double_kill": "double_kill.wav",
    "triple_kill": "triple_kill.wav",
    "rampage": "rampage.wav",
    "aegis": "aegis_pickup.wav",
}

# Which game-event ("kill"/"roshan"/ignore) each audio template maps to for the ensemble.
EVENT_KIND: dict[str, str] = {
    "roshan": "roshan",
    "aegis": "roshan",
    "first_blood": "kill",
    "double_kill": "kill",
    "triple_kill": "kill",
    "rampage": "kill",
}


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    buffer_sec: float = 8.0
    scan_interval_sec: float = 1.0
    ncc_threshold: float = 0.55
    cooldown_sec: dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.cooldown_sec is None:
            self.cooldown_sec = {"kill": 4.0, "roshan": 60.0}


def _load_wav(path: Path, target_sr: int) -> Optional[np.ndarray]:
    try:
        sr, data = wavfile.read(str(path))
    except Exception as e:
        log.warning("failed to read %s: %s", path, e)
        return None
    if data.ndim == 2:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    mx = np.max(np.abs(data)) or 1.0
    data = data / mx
    if sr != target_sr:
        # cheap linear resample — good enough for cross-correlation
        ratio = target_sr / sr
        new_len = int(len(data) * ratio)
        data = np.interp(
            np.linspace(0, len(data) - 1, new_len, dtype=np.float32),
            np.arange(len(data), dtype=np.float32),
            data,
        ).astype(np.float32)
    return data


def normalized_xcorr(signal: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Sliding normalized cross-correlation. Returns array of NCC values, one per offset."""
    L = len(template)
    if len(signal) < L:
        return np.zeros(0, dtype=np.float32)
    t = template - template.mean()
    t_norm = np.linalg.norm(t) or 1.0
    t_flipped = t[::-1]  # np.convolve = correlate with flipped kernel

    numer = np.convolve(signal, t_flipped, mode="valid")

    sig_sq = signal.astype(np.float64) ** 2
    sig_sum = np.convolve(signal.astype(np.float64), np.ones(L), mode="valid")
    sig_sq_sum = np.convolve(sig_sq, np.ones(L), mode="valid")
    # variance-normalized denominator
    mean = sig_sum / L
    var = sig_sq_sum / L - mean ** 2
    var = np.clip(var, 1e-9, None)
    sig_norm = np.sqrt(var * L)

    return (numer / (sig_norm * t_norm + 1e-9)).astype(np.float32)


class AudioDetector:
    def __init__(self, sounds_dir: str | Path, cfg: AudioConfig | None = None):
        self.cfg = cfg or AudioConfig()
        self.templates: dict[str, np.ndarray] = {}
        root = Path(sounds_dir)
        for key, fname in EVENT_FILES.items():
            p = root / fname
            if p.exists():
                wav = _load_wav(p, self.cfg.sample_rate)
                if wav is not None and len(wav) > 0:
                    self.templates[key] = wav
        log.info("audio: loaded templates: %s",
                 {k: f"{len(v)/self.cfg.sample_rate:.2f}s" for k, v in self.templates.items()})
        self._last_fire: dict[str, float] = {"kill": 0.0, "roshan": 0.0}

    def scan(self, buffer: np.ndarray, now: float) -> list[tuple[str, str, float]]:
        """Return list of (kind, template_key, score) passing threshold+cooldown."""
        hits: list[tuple[str, str, float]] = []
        if buffer.size == 0:
            return hits
        for key, tpl in self.templates.items():
            if len(tpl) > len(buffer):
                continue
            ncc = normalized_xcorr(buffer, tpl)
            if ncc.size == 0:
                continue
            peak = float(ncc.max())
            if peak >= self.cfg.ncc_threshold:
                kind = EVENT_KIND.get(key, "kill")
                if now - self._last_fire[kind] >= self.cfg.cooldown_sec[kind]:
                    self._last_fire[kind] = now
                    hits.append((kind, key, peak))
        return hits


async def audio_pipeline(
    hls_url: str,
    detector: AudioDetector,
    on_hit: Callable[[str, str, float], Awaitable[None]],
):
    """Spawn ffmpeg → stream s16le mono PCM → keep rolling buffer → periodic NCC scan."""
    sr = detector.cfg.sample_rate
    buf_size = int(detector.cfg.buffer_sec * sr)
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-i", hls_url,
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        "-f", "s16le",
        "-",
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    buffer = np.zeros(0, dtype=np.float32)
    bytes_per_sample = 2
    read_bytes = int(detector.cfg.scan_interval_sec * sr) * bytes_per_sample

    try:
        while True:
            chunk = await proc.stdout.readexactly(read_bytes)
            samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            buffer = np.concatenate([buffer, samples])[-buf_size:]
            hits = detector.scan(buffer, time.time())
            for kind, key, score in hits:
                log.info("audio hit: kind=%s template=%s score=%.3f", kind, key, score)
                await on_hit(kind, key, score)
    except asyncio.IncompleteReadError:
        err = (await proc.stderr.read()).decode(errors="ignore")
        log.warning("audio ffmpeg ended: %s", err[:300])
    except asyncio.CancelledError:
        raise
    finally:
        try:
            proc.kill()
        except Exception:
            pass
