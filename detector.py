"""Ensemble detector for Dota 2 stream events.

Channels:
  - Video: top-score OCR (counts ticks on the Radiant/Dire kill counter —
    ground truth on F1=95% with zero false positives) + Roshan (Aegis icon
    and the "slain" banner).
  - Audio: normalized cross-correlation with reference sounds (Roshan roar,
    First Blood, Double/Triple Kill, Rampage, Aegis pickup).

Per-kind cooldown prevents different channels from duplicating the same
event (4s for kills, 60s for Roshan).

Source: a Twitch/YouTube URL (resolved via yt-dlp) or a path to a local
video file — the latter is the demo mode that drives the UI without a live
broadcast.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Awaitable, Callable, Optional

import cv2
import numpy as np

from detectors.audio import AudioDetector, audio_pipeline
from detectors.roshan import RoshanDetector
from detectors.score_ocr import ScoreDetector

log = logging.getLogger("detector")
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    log.addHandler(_h)
    log.propagate = False


ASSETS = Path(__file__).parent / "assets"
DIGITS_DIR = ASSETS / "digits"
TEMPLATES_DIR = ASSETS / "templates"
SOUNDS_DIR = ASSETS / "sounds"

COOLDOWN = {"kill": 1.5, "roshan": 60.0}


class DemoClock:
    """Tracks the current playback offset inside a looping local VOD so the
    UI's <video> element can seek to the same timeline the detector is
    currently processing. In live mode it stays at None."""
    def __init__(self):
        self.video_time: float | None = None
        self.loop_started_wall: float | None = None

demo_clock = DemoClock()


def _is_local_source(src: str) -> bool:
    if not src:
        return False
    if src.startswith("file://"):
        return True
    p = Path(src)
    return p.exists() and p.is_file()


def _local_path(src: str) -> Path:
    return Path(src[7:] if src.startswith("file://") else src)


async def resolve_hls_url(stream_url: str) -> Optional[str]:
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "yt_dlp",
        "-g", "-f", "best[height<=720]/best", stream_url,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    if proc.returncode != 0:
        log.error("yt-dlp failed: %s", err.decode(errors="ignore")[:400])
        return None
    url = out.decode().strip().splitlines()[0] if out else ""
    return url or None


async def frame_generator_ffmpeg(hls_url: str, interval: float):
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-i", hls_url,
        "-vf", f"fps=1/{interval},scale=1280:-1",
        "-q:v", "5",
        "-f", "image2pipe", "-vcodec", "mjpeg",
        "-",
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    buf = b""
    try:
        while True:
            chunk = await proc.stdout.read(65536)
            if not chunk:
                err = (await proc.stderr.read()).decode(errors="ignore")
                log.error("ffmpeg(video) exited: %s", err[:400])
                return
            buf += chunk
            while True:
                start = buf.find(b"\xff\xd8")
                end = buf.find(b"\xff\xd9", start + 2) if start != -1 else -1
                if start == -1 or end == -1:
                    break
                yield buf[start:end + 2]
                buf = buf[end + 2:]
    finally:
        try:
            proc.kill()
        except Exception:
            pass


async def frame_generator_local(path: Path, interval: float):
    """Demo mode: stream frames from a local video file, paced to wall-clock
    time so the UI behaves as if watching a live broadcast. Loops on EOF.
    Publishes current video offset to demo_clock so the UI can sync."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        log.error("cannot open local source: %s", path)
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_stride = max(1, int(round(fps * interval)))
    try:
        while True:
            idx = 0
            start = time.time()
            demo_clock.loop_started_wall = start
            demo_clock.video_time = 0.0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % frame_stride == 0:
                    target = start + (idx / fps)
                    now = time.time()
                    if target > now:
                        await asyncio.sleep(target - now)
                    demo_clock.video_time = idx / fps
                    ok2, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok2:
                        yield jpeg.tobytes()
                idx += 1
            log.info("local source EOF, looping")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    finally:
        cap.release()


class Ensemble:
    """Joins signals from all detectors with per-kind cooldowns."""

    def __init__(self):
        self.score = ScoreDetector(DIGITS_DIR)
        self.roshan = RoshanDetector(TEMPLATES_DIR)
        self.audio = AudioDetector(SOUNDS_DIR)
        self._last_fire: dict[str, float] = {"kill": float("-inf"), "roshan": float("-inf")}

    def _try_fire(self, kind: str, now: float) -> bool:
        if now - self._last_fire[kind] >= COOLDOWN[kind]:
            self._last_fire[kind] = now
            return True
        return False

    def on_video_frame(self, frame_bgr: np.ndarray, now: float) -> list[tuple[str, str]]:
        fired: list[tuple[str, str]] = []

        for side, conf in self.score.detect(frame_bgr, now):
            if self._try_fire("kill", now):
                fired.append(("kill", f"score({side})"))

        rh = self.roshan.detect(frame_bgr, now)
        if rh and self._try_fire("roshan", now):
            fired.append(("roshan", f"roshan_{rh[0]}({rh[1]:.2f})"))

        return fired

    def on_audio_hit(self, kind: str, now: float, tag: str) -> bool:
        return self._try_fire(kind, now)


async def run_detector(
    stream_url: str,
    on_event: Callable[[str, float, str], Awaitable[None]],
    interval: float = 2.0,
    **_: object,
):
    """Main loop: resolve source, drive detectors, dispatch events."""
    ensemble = Ensemble()
    log.info("ensemble ready (score_ocr+roshan+audio)")

    if _is_local_source(stream_url):
        path = _local_path(stream_url)
        log.info("running in DEMO mode from local file: %s", path)
        try:
            async for jpeg in frame_generator_local(path, interval):
                # In demo mode events are stamped with video-time (not wall-clock)
                # so that client video.currentTime, detector, and predictions all
                # live on the same timeline. This eliminates any drift between
                # the player buffer and the detector loop.
                now = demo_clock.video_time if demo_clock.video_time is not None else time.time()
                frame_bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    continue
                for kind, tag in ensemble.on_video_frame(frame_bgr, now):
                    log.info("EVENT %s source=%s video_time=%.1f", kind, tag, now)
                    await on_event(kind, now, tag)
        except Exception as e:
            log.exception("demo loop crashed: %s", e)
        return

    while True:
        hls = await resolve_hls_url(stream_url)
        if not hls:
            log.info("stream offline, retry in 30s")
            await asyncio.sleep(30)
            continue
        log.info("stream resolved, starting detectors")

        async def on_audio(kind: str, key: str, score: float) -> None:
            now = time.time()
            if ensemble.on_audio_hit(kind, now, key):
                log.info("EVENT %s source=audio(%s,%.2f)", kind, key, score)
                await on_event(kind, now, f"audio({key},{score:.2f})")

        audio_task = asyncio.create_task(audio_pipeline(hls, ensemble.audio, on_audio))

        try:
            last_frame_ts = time.time()
            async for jpeg in frame_generator_ffmpeg(hls, interval):
                now = time.time()
                last_frame_ts = now
                frame_bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    continue
                for kind, tag in ensemble.on_video_frame(frame_bgr, now):
                    log.info("EVENT %s source=%s", kind, tag)
                    await on_event(kind, now, tag)
                if now - last_frame_ts > 25:
                    log.warning("frames stale, reresolving")
                    break
        except Exception as e:
            log.exception("video loop crashed: %s", e)
        finally:
            audio_task.cancel()
            try:
                await audio_task
            except Exception:
                pass

        log.info("restarting in 5s")
        await asyncio.sleep(5)
