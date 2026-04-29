import aiosqlite
from pathlib import Path

DB_PATH = Path(__file__).parent / "game.db"


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nickname TEXT UNIQUE NOT NULL,
            score INTEGER DEFAULT 0,
            created_at REAL DEFAULT (strftime('%s', 'now'))
        );
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            submit_ts REAL NOT NULL,
            predicted_ts REAL NOT NULL,
            matched INTEGER DEFAULT 0,
            score_awarded INTEGER DEFAULT 0,
            delta REAL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            ts REAL NOT NULL,
            source TEXT DEFAULT 'ai'
        );
        CREATE INDEX IF NOT EXISTS idx_pred_ts ON predictions(predicted_ts);
        CREATE INDEX IF NOT EXISTS idx_event_ts ON events(ts);
        """)
        await db.commit()


async def get_or_create_user(nickname: str) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("INSERT OR IGNORE INTO users(nickname) VALUES(?)", (nickname,))
        await db.commit()
        async with db.execute("SELECT id FROM users WHERE nickname=?", (nickname,)) as cur:
            row = await cur.fetchone()
            return row[0]


async def add_prediction(user_id: int, event_type: str, submit_ts: float, predicted_ts: float) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "INSERT INTO predictions(user_id, event_type, submit_ts, predicted_ts) VALUES(?,?,?,?)",
            (user_id, event_type, submit_ts, predicted_ts),
        )
        await db.commit()
        return cur.lastrowid


async def last_unmatched_target(user_id: int) -> float | None:
    """Latest predicted_ts (game-time) for an unmatched bet by this user.
    Used to enforce a min spacing between successive bets."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            """SELECT MAX(predicted_ts) FROM predictions
               WHERE user_id=? AND matched=0""",
            (user_id,),
        ) as cur:
            row = await cur.fetchone()
            return row[0] if row and row[0] is not None else None


async def user_predictions(user_id: int, limit: int = 20):
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            """SELECT id, event_type, predicted_ts, matched, score_awarded, delta
               FROM predictions WHERE user_id=?
               ORDER BY id DESC LIMIT ?""",
            (user_id, limit),
        ) as cur:
            rows = await cur.fetchall()
            return [
                {
                    "id": r[0],
                    "event_type": r[1],
                    "target": r[2],
                    "matched": r[3],
                    "points": r[4],
                    "delta": r[5],
                }
                for r in rows
            ]


async def add_event(event_type: str, ts: float, source: str = "ai") -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "INSERT INTO events(event_type, ts, source) VALUES(?,?,?)",
            (event_type, ts, source),
        )
        await db.commit()
        return cur.lastrowid


async def match_predictions(event_type: str, event_ts: float, window: float = 20.0):
    """Match unmatched predictions whose predicted_ts is within window of the actual event."""
    awarded = []
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            """SELECT p.id, p.user_id, p.predicted_ts, u.nickname FROM predictions p
               JOIN users u ON u.id = p.user_id
               WHERE p.event_type=? AND p.matched=0
               AND p.predicted_ts BETWEEN ? AND ?""",
            (event_type, event_ts - window, event_ts + window),
        ) as cur:
            rows = await cur.fetchall()

        for pid, uid, pts, nick in rows:
            delta = abs(pts - event_ts)
            base = 1000 if event_type == "kill" else 3000
            # Cubic falloff so precision dominates: delta=0 → base, delta=10 → base/8.
            # Rewards players who actually time the event, not those who spam guesses.
            points = int(base * max(0.0, 1.0 - delta / window) ** 3)
            await db.execute(
                "UPDATE predictions SET matched=1, score_awarded=?, delta=? WHERE id=?",
                (points, delta, pid),
            )
            await db.execute(
                "UPDATE users SET score = score + ? WHERE id=?",
                (points, uid),
            )
            awarded.append({"prediction_id": pid, "user_id": uid, "nickname": nick, "points": points, "delta": round(delta, 2)})
        await db.commit()
    return awarded


async def expire_old_predictions(now_ts: float, max_future: float = 60.0):
    """Mark as matched=2 (expired) predictions whose predicted_ts is way in the past with no event."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE predictions SET matched=2 WHERE matched=0 AND predicted_ts < ?",
            (now_ts - max_future,),
        )
        await db.commit()


async def leaderboard(limit: int = 10):
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT nickname, score FROM users ORDER BY score DESC LIMIT ?", (limit,)
        ) as cur:
            rows = await cur.fetchall()
            return [{"nickname": r[0], "score": r[1]} for r in rows]


async def recent_events(limit: int = 20):
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT event_type, ts, source FROM events ORDER BY ts DESC LIMIT ?", (limit,)
        ) as cur:
            rows = await cur.fetchall()
            return [{"event_type": r[0], "ts": r[1], "source": r[2]} for r in rows]
