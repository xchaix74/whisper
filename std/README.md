# std/ — Whisper ringkas (single-user)

Varian **std** — faster-whisper ringkas untuk single-user / dev. Sama enjin dan model dengan [`fat/`](../fat/), tetapi **tanpa** lapisan multi-user.

## Ciri

- **Enjin:** faster-whisper 1.2.1 + CTranslate2 4.8.1
- **Model:** openai/whisper-large-v3-turbo (809M), bundled dalam image
- **VRAM:** ~2.25 GB (float16)
- **Compute:** float16 (sm_120 tidak support INT8 pada CTranslate2)
- **Input:** Opus / m4a / mp3 / WAV / webm — melalui PyAV
- **Output:** JSON `{"text":"...", "language":"...", "language_probability":...}`
- **Endpoint:** `POST /v1/audio/transcriptions` (OpenAI-compatible subset)
- **Health:** `GET /health`
- **Version:** 0.1.0

## Tiada (berbanding fat)

- ❌ Queue depth 32 + HTTP 429
- ❌ GPU lock (serialize)
- ❌ Timeout 120s + HTTP 504
- ❌ Log JSONL
- ❌ VRAM tracking
- ❌ Max audio 600s / max file 100MB

Untuk production multi-user, guna [`fat/`](../fat/).

## Jalankan

```bash
docker run -d --name whisper-std \
  --gpus all \
  -p 8320:8000 \
  -v /path/to/audio:/audio \
  xchaix74/whisper:std
```

## API

### Transkripsi

```bash
curl -X POST http://127.0.0.1:8320/v1/audio/transcriptions \
  -F "file=@audio.m4a" \
  -F "language=ms"
```

**Respons:**

```json
{"text": "...", "language": "ms", "language_probability": 1.0}
```

### Health

```bash
curl http://127.0.0.1:8320/health
# {"status":"ok","version":"0.1.0"}
```

## Prestasi (node1, RTX 5060 Ti)

| Keadaan | Masa |
| --- | --- |
| Request pertama (cold start) | ~6s (pindah model ke GPU + init CUDA) |
| Request seterusnya (warm) | ~0.23s untuk audio 4.2s |

## Lesen

MIT — lihat LICENSE.
