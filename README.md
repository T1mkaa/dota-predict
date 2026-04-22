# Dota Predict

Игра на угадывание событий в Dota 2 трансляции: зрители заранее предсказывают, **через сколько секунд** случится ближайший килл или смерть Рошана, а AI-детектор фиксирует реальный момент и раздаёт очки за точность.

## Точность детектора

Замерено на 10-минутном публичном VOD (22 размеченных события: 21 килл + 1 Рошан):

| Событие  | Precision | Recall | F1     |
|----------|-----------|--------|--------|
| Килл     | **100%**  | 90%    | 95%    |
| Рошан    | **100%**  | 100%   | 100%   |
| **Итого**| **100%**  | 91%    | **95%**|

Ноль ложных срабатываний — игрок никогда не увидит призрачное событие.

## Архитектура

Два независимых сигнала, ансамбль с кулдаунами:

- **Видео-канал (основной).**
  - **Счёт-OCR** (`detectors/score_ocr.py`): читает цифры Radiant/Dire в верхней HUD. Инкремент счёта = килл. Template matching по 33 шаблонам цифр, стабилизация 2 кадра, классификация через cosine similarity на бинарных масках. Это «истина» — счёт растёт только при реальном килле героя.
  - **Рошан** (`detectors/roshan.py`): иконка Aegis в инвентаре (работает) + шаблон надписи "Roshan has been slain" (опционально).
- **Аудио-канал** (`detectors/audio.py`): нормализованная cross-correlation с эталонными звуками (First Blood, Double/Triple Kill, Rampage, Aegis pickup, рёв Рошана). Добавляет recall, когда видео перекрыто оверлеем стримера.
- **In-game gate** (`detectors/gameplay_gate.py`): отсекает меню/хайлайты по дисперсии миникарты и отсутствию логотипа Dota.

Все каналы слетаются в `Ensemble` (`detector.py`), где per-kind cooldown не даёт дублировать одно событие (`kill`: 4 с, `roshan`: 60 с).

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

## Замер точности на размеченном VOD

```bash
./venv/Scripts/python.exe evaluate.py tmp/vod_chunk.mp4 path/to/timings.txt
```

Формат `timings.txt`: одна строка — `MM:SS kill|roshan`.

## Структура проекта

```
app.py                  FastAPI + WebSocket server
detector.py             Ensemble + live-loop + demo-loop
detectors/
  score_ocr.py          Основной kill-детектор (HUD score OCR)
  roshan.py             Рошан: aegis + slain banner
  audio.py              Аудио-канал
  gameplay_gate.py      In-game фильтр
db.py                   SQLite (users/events/predictions)
templates/index.html    UI
static/style.css        UI стили
assets/
  digits/               Шаблоны цифр 0-9 (HUD OCR)
  templates/            aegis.png и прочее
  sounds/               Эталонные звуки
evaluate.py             Оффлайн метрики на размеченном VOD
calibrate.py            Перегенерация шаблонов цифр под другой HUD
```
