import asyncio
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import db
from detector import run_detector, demo_clock

load_dotenv()

BASE = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE / "templates"))


class Broadcaster:
    def __init__(self):
        self.clients: set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.clients.add(ws)

    def disconnect(self, ws: WebSocket):
        self.clients.discard(ws)

    async def broadcast(self, payload: dict):
        dead = []
        for ws in self.clients:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


bus = Broadcaster()


async def on_event_detected(event_type: str, ts: float, source: str = "ai"):
    """Called by the detector pipeline whenever an event is spotted."""
    await db.add_event(event_type, ts, source)
    awarded = await db.match_predictions(event_type, ts)
    await bus.broadcast({"type": "event", "event_type": event_type, "ts": ts, "source": source})
    if awarded:
        board = await db.leaderboard()
        await bus.broadcast({"type": "leaderboard", "board": board, "awarded": awarded})


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    stream_url = os.getenv("STREAM_URL", "")
    interval = float(os.getenv("FRAME_INTERVAL", "2.0"))
    task = None
    if stream_url:
        async def on_event(kind: str, ts: float, source: str):
            await on_event_detected(kind, ts, source)
        task = asyncio.create_task(run_detector(stream_url, on_event, interval))
    yield
    if task:
        task.cancel()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")

_stream_url = os.getenv("STREAM_URL", "")
_local_media: Path | None = None
if _stream_url:
    candidate = Path(_stream_url[7:] if _stream_url.startswith("file://") else _stream_url)
    if candidate.is_file():
        _local_media = candidate.resolve()
        app.mount("/media", StaticFiles(directory=str(_local_media.parent)), name="media")


class PredictBody(BaseModel):
    nickname: str
    event_type: str
    target: str | float    # absolute game time, "MM:SS" or seconds


class AdminEventBody(BaseModel):
    event_type: str
    source: str = "manual"


MIN_LEAD_SECONDS = 10           # cannot bet less than this far ahead
MIN_SPACING_SECONDS = 30        # min gap between successive bets by same user
SCORING_WINDOW = 20.0           # |target - event| must fit in this


def _parse_target(v: str | float) -> int | None:
    if isinstance(v, (int, float)):
        return int(v) if v >= 0 else None
    s = str(v).strip()
    if ":" in s:
        try:
            mm, ss = s.split(":")
            return int(mm) * 60 + int(ss)
        except ValueError:
            return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _current_game_time() -> int | None:
    return demo_clock.game_time()


@app.get("/")
async def index(request: Request):
    stream_url = os.getenv("STREAM_URL", "")
    channel = stream_url.rstrip("/").split("/")[-1] if "twitch.tv" in stream_url else ""
    local_video_url = f"/media/{_local_media.name}" if _local_media else ""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "twitch_channel": channel,
            "local_video_url": local_video_url,
        },
    )


@app.post("/api/predict")
async def predict(body: PredictBody):
    nick = body.nickname.strip()[:24]
    if not nick:
        return JSONResponse({"error": "укажи никнейм"}, status_code=400)
    if body.event_type != "kill":
        return JSONResponse({"error": "в v0.1 поддерживаются только ставки на kill"}, status_code=400)
    target = _parse_target(body.target)
    if target is None:
        return JSONResponse({"error": "неверный формат target (нужно MM:SS или секунды)"}, status_code=400)

    now_game = _current_game_time()
    if now_game is None:
        return JSONResponse(
            {"error": "игровой таймер ещё не инициализирован, попробуй через секунду"},
            status_code=503,
        )

    if target < now_game + MIN_LEAD_SECONDS:
        return JSONResponse(
            {"error": f"ставку надо делать минимум за {MIN_LEAD_SECONDS} сек до целевого времени"},
            status_code=400,
        )

    uid = await db.get_or_create_user(nick)
    last_target = await db.last_unmatched_target(uid)
    if last_target is not None and abs(target - last_target) < MIN_SPACING_SECONDS:
        return JSONResponse(
            {"error": f"минимум {MIN_SPACING_SECONDS} сек между ставками одного игрока"},
            status_code=409,
        )

    pid = await db.add_prediction(uid, body.event_type, float(now_game), float(target))
    await bus.broadcast({
        "type": "prediction",
        "prediction_id": pid,
        "nickname": nick,
        "event_type": body.event_type,
        "submit_game_time": now_game,
        "target_game_time": target,
    })
    return {"id": pid, "now_game": now_game, "target": target}


@app.get("/api/my_predictions")
async def my_predictions(nickname: str):
    nick = nickname.strip()[:24]
    if not nick:
        return []
    uid = await db.get_or_create_user(nick)
    return await db.user_predictions(uid)


@app.get("/api/leaderboard")
async def leaderboard():
    return await db.leaderboard()


@app.get("/api/events")
async def events():
    return await db.recent_events()


@app.get("/api/status")
async def status():
    return {
        "demo_mode": _local_media is not None,
        "video_time": demo_clock.video_time,
        "game_time": demo_clock.game_time(),
        "game_time_offset": demo_clock.game_time_offset,
    }


@app.post("/api/admin/event")
async def admin_event(body: AdminEventBody):
    """Manual trigger used for testing scoring without the AI pipeline."""
    ts = time.time()
    await on_event_detected(body.event_type, ts, body.source)
    return {"ok": True, "ts": ts}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await bus.connect(ws)
    try:
        board = await db.leaderboard()
        await ws.send_json({"type": "leaderboard", "board": board, "awarded": []})
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        bus.disconnect(ws)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
