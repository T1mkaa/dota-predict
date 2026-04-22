#!/bin/sh
# Fetch the demo VOD once per container lifetime so the rest of the image stays
# small. Skipped automatically on live-stream deployments where STREAM_URL points
# to a Twitch/YouTube URL.
set -eu

if [ -n "${DEMO_VOD_URL:-}" ] && [ ! -f "tmp/vod_chunk.mp4" ]; then
    case "${STREAM_URL:-}" in
        http*|https*)
            echo "[entrypoint] STREAM_URL is a live feed — skipping VOD download"
            ;;
        *)
            echo "[entrypoint] fetching demo VOD from $DEMO_VOD_URL"
            curl -fL --retry 3 -o tmp/vod_chunk.mp4 "$DEMO_VOD_URL"
            ls -lh tmp/vod_chunk.mp4
            ;;
    esac
fi

exec python -m uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}"
