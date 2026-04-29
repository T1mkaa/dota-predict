# Dota Predict

Спортивный прогноз по Dota 2: зрители смотрят трансляцию и заранее ставят, **в какой момент игрового времени** случится ближайший килл (`MM:SS` по HUD-таймеру). AI-детектор фиксирует реальное событие, кубическая формула очков отсекает «лудоманский» спам — попадание секунда-в-секунду даёт максимум, край окна почти ноль.

## Точность детектора

Замерено на 10-минутном публичном VOD (21 размеченный килл):

| Событие | Precision | Recall | F1     |
|---------|-----------|--------|--------|
| Килл    | **100%**  | 90%    | **95%**|

Ноль ложных срабатываний — игрок никогда не увидит призрачное событие.

## Архитектура

Два независимых сигнала, ансамбль с кулдауном:

- **Видео-канал (основной).**
  - **Счёт-OCR** (`detectors/score_ocr.py`): читает цифры Radiant/Dire в верхней HUD. Инкремент счёта = килл. Template matching по 33 шаблонам цифр, стабилизация 2 кадра, классификация через cosine similarity на бинарных масках. Это «истина» — счёт растёт только при реальном килле героя.
  - **Игровой таймер** (`detectors/timer_ocr.py` → `calibrate_offset`): на старте demo-режима читает HUD-таймер на ~30 кадрах в начале VOD, медианой определяет offset `video_time → game_time`. Дальше все события и ставки живут в одном пространстве (`MM:SS` по таймеру Dota).
- **Аудио-канал** (`detectors/audio.py`): нормализованная cross-correlation с эталонными звуками киллов (First Blood, Double/Triple Kill, Rampage). Добавляет recall, когда видео перекрыто оверлеем стримера. Live-only — в demo-режиме не активен.
- **In-game gate** (`detectors/gameplay_gate.py`): отсекает меню/хайлайты по дисперсии миникарты и отсутствию логотипа Dota.

Все каналы слетаются в `Ensemble` (`detector.py`), где cooldown 1.5 сек не даёт дублировать один килл из разных каналов.

## Игровая модель

- **Ставки в game-time.** Пользователь пишет ник и вводит целевое время килла `MM:SS`.
- **Дедлайн ставки:** не позднее чем за 10 сек до целевого момента.
- **Анти-спам:** между двумя своими ставками — минимум 30 сек по `target`.
- **Окно матчинга:** ±20 сек от события. Кубическая формула:
  `points = 1000 · max(0, 1 − Δ / window)³`
  Δ=0 → 1000; Δ=10 → 125; Δ ≥ 20 → 0.
  Спам-стратегия не работает — оракул, попадающий в секунду, кратно обгоняет ставящих наугад.

### Почему именно счёт-OCR

Первая версия делала template matching иконок героев в kill-feed. На тестовом VOD получалось **F1=22%** — шаблоны вроде Batrider и Chaos Knight стабильно «магнитили» шум компресс-артефактов и давали массу false positives, даже когда этих героев в матче не было. Переход на чтение счётчика дал **+73 п.п. F1** без потери recall.

## Стек

Python 3.12, FastAPI + WebSockets + aiosqlite + Jinja2 + vanilla JS. CV — чистый OpenCV + scipy (без torch / tesseract / easyocr). Стрим тянется yt-dlp + ffmpeg; демо-режим читает локальный файл через `cv2.VideoCapture`.

## Запуск

```bash
python -m venv venv
./venv/Scripts/activate        # Windows. Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

Создай `.env`:

```env
# Живой Twitch/YouTube стрим:
STREAM_URL=https://www.twitch.tv/<channel>

# ИЛИ демо-режим с локальным VOD:
STREAM_URL=tmp/vod_chunk.mp4

FRAME_INTERVAL=1.0
```

Запусти:

```bash
./venv/Scripts/python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Открой `http://127.0.0.1:8000`.

## Известные ограничения

- **Только килл — Рошан выпилен из v0.1.** Надёжный HUD-detection Рошана требует чистый 1080p replay без студийных оверлеев (RUHUB / EPICENTER MAJOR webcam закрывают центр-верх HUD), а каноничный сигнал — иконка Aegis в инвентаре поднявшего — ненадёжен из-за того что инвентарь конкретного героя не всегда в кадре. Код `detectors/roshan.py` оставлен в репо как кандидат на v0.2 при наличии чистого источника.
- **Live-режим (Twitch URL) пока без ставок.** В demo-режиме offset `video_time → game_time` калибруется один раз на старте — это работает, потому что в эталонном VOD нет внутриматчевых пауз. Для live-стрима с возможной паузой нужен непрерывный OCR таймера; текущий single-frame OCR на нашем VOD даёт ~63% — недостаточно для полного state-машинного режима. План v0.2 — заменить cosine-classifier на маленький CNN.
- **Demo VOD** (`tmp/vod_chunk.mp4`) склеен из конца одного матча и начала следующего. Калибровка использует только первый матч; после loop детектор сбрасывается.

## Замер точности на размеченном VOD

```bash
./venv/Scripts/python.exe evaluate.py tmp/vod_chunk.mp4 path/to/timings.txt
```

Формат `timings.txt`: одна строка — `MM:SS kill`. Roshan-строки игнорируются для чистоты метрики.

## Структура проекта

```
app.py                  FastAPI + WebSocket server, scoring API
detector.py             Ensemble + live-loop + demo-loop с calibrate_offset
detectors/
  score_ocr.py          Основной kill-детектор (HUD score OCR)
  timer_ocr.py          OCR HUD-таймера + calibrate_offset (demo-mode)
  audio.py              Аудио-канал (kill звуки)
  gameplay_gate.py      In-game фильтр
  roshan.py             Парк до v0.2, не импортируется
db.py                   SQLite (users/events/predictions, всё в game-time)
templates/index.html    UI: live game-clock + MM:SS-инпут ставки
static/style.css        UI стили
assets/
  digits/               Шаблоны цифр 0-9 (счёт HUD)
  digits_timer/         Шаблоны цифр для таймера (мельче чем у счёта)
  sounds/               Эталонные звуки киллов
evaluate.py             Оффлайн метрики на размеченном VOD
```
