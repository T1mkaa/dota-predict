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
    delay_seconds: float
    submit_video_time: float | None = None


class AdminEventBody(BaseModel):
    event_type: str
    source: str = "manual"


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
        return JSONResponse({"error": "nickname required"}, status_code=400)
    delay = max(0.5, min(60.0, body.delay_seconds))
    uid = await db.get_or_create_user(nick)
    # Demo mode clients pass their video.currentTime so submit/predicted/event
    # timestamps all live in the same video-time scale.
    submit_ts = body.submit_video_time if body.submit_video_time is not None else time.time()
    predicted_ts = submit_ts + delay
    if await db.has_active_prediction(uid, submit_ts):
        return JSONResponse(
            {"error": "уже есть активное предсказание — дождись результата"},
            status_code=409,
        )
    pid = await db.add_prediction(uid, body.event_type, submit_ts, predicted_ts)
    await bus.broadcast({
        "type": "prediction",
        "prediction_id": pid,
        "nickname": nick,
        "event_type": body.event_type,
        "submit_ts": submit_ts,
        "predicted_ts": predicted_ts,
        "delay": delay,
    })
    return {"id": pid, "submit_ts": submit_ts, "predicted_ts": predicted_ts}


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
