FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py db.py detector.py evaluate.py entrypoint.sh ./
COPY detectors ./detectors
COPY templates ./templates
COPY static ./static
COPY assets ./assets

RUN chmod +x entrypoint.sh && mkdir -p tmp

ENV FRAME_INTERVAL=1.0 \
    STREAM_URL=tmp/vod_chunk.mp4 \
    DEMO_VOD_URL=https://github.com/T1mkaa/dota-predict/releases/download/v0.1/vod_chunk.mp4

EXPOSE 8000
CMD ["./entrypoint.sh"]
