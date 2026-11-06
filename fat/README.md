# Whisper FAT — CikguSayang ASR

Production-ready **faster-whisper** image untuk **CikguSayang (CS)** — mobile AI tutor bahasa percuma, OSS.

Dibina khas untuk **NVIDIA RTX 5060 Ti (Blackwell, sm_120)** dengan **CUDA 12.8** dan **float16**. Tiada INT8 (tidak disokong pada sm_120 dengan CTranslate2).

## Ciri

- **Enjin:** `faster-whisper 1.2.1` + `CTranslate2 4.8.1`
- **Model:** `openai/whisper-large-v3-turbo` (809M), bundled dalam image
- **VRAM:** ~2.25 GB (float16)
- **Input:** Opus / m4a / mp3 / WAV / webm — melalui PyAV (tiada FFmpeg executable)
- **Output:** JSON `{"text":"...", "language":"...", "language_probability":...}`
- **Endpoint:** `POST /v1/audio/transcriptions` (OpenAI-compatible subset)
- **Multilingual:** 13+ bahasa diuji (de, en, es, fr, it, ms, nl, pa, pt, ta, th, yue, zh)
- **VAD:** Aktif — `min_silence_duration_ms=500`
- **Halusinasi:** `condition_on_previous_text=False` — elak gelung
- **Multi-user:** Queue depth 32, GPU lock, timeout 120s, max 600s audio, max 100MB fail
- **Log:** JSONL di `/app/logs/whisper.jsonl` (mount volume)
- **Version:** `1.0.0`

## Prerequisites

- **GPU:** NVIDIA dengan compute capability **sm_120** (RTX 5060 Ti, 5070, 5080, 5090) atau mana-mana GPU yang disokong CUDA 12.8
- **Driver NVIDIA:** ≥ 570 (menyokong CUDA 12.8)
- **Docker:** dengan **NVIDIA Container Toolkit** (`--gpus all`)
- **VRAM:** ~2.25 GB untuk model, + ~1 GB overhead
- **Disk:** ~4.9 GB (compressed di Docker Hub)

## Jalankan

```bash
docker run -d --name whisper \
  --gpus all \
  -p 8319:8000 \
  -v /path/to/audio:/audio \
  -v /path/to/logs:/app/logs \
  xchaix74/whisper:fat
API
Transkripsi
text
POST /v1/audio/transcriptions
Content-Type: multipart/form-data

Fields:
  file    : Opus/m4a/mp3/WAV (wajib)
  language: ISO 639-1 (pilihan; default auto-detect)
            "en" untuk English, "ms" untuk Melayu, dll.
  task    : "transcribe" (default) atau "translate"
Contoh:

bash
curl -X POST http://127.0.0.1:8319/v1/audio/transcriptions \
  -F "file=@audio.m4a" \
  -F "language=en"
Respons:

json
{"text": "...", "language": "en", "language_probability": 0.998}
Health
bash
curl http://127.0.0.1:8319/health
# {"status": "ok", "version": "1.0.0"}
Had
Had	Nilai	Kesan jika melebihi
Saiz fail	100 MB	HTTP 413
Durasi audio	600s (10 min)	HTTP 413
Masa transkripsi	120s	HTTP 504
Queue	32 serentak	HTTP 429 (server busy)
Log
Setiap request ditulis ke /app/logs/whisper.jsonl (satu baris JSON):

json
{
  "ts": "2026-11-05T22:05:47.706598+00:00",
  "version": "1.0.0",
  "request_id": "f3ed2e1dc0e6",
  "file": "10.m4a",
  "size": 51782075,
  "language": "ta",
  "task": "transcribe",
  "status": "error",
  "duration": 120.063,
  "vram_before_mb": 9204.0,
  "vram_after_mb": 9332.0,
  "text_len": 0,
  "error_type": "Timeout",
  "error_msg": "transkripsi melebihi 120.0s"
}
Bahasa disokong (diuji)
Kod	Bahasa
de	Jerman
en	Inggeris
es	Sepanyol
fr	Perancis
it	Itali
ms	Melayu
nl	Belanda
pa	Punjabi
pt	Portugis
ta	Tamil
th	Thai
yue	Kantonis
zh	Mandarin/Cina
Whisper menyokong 99 bahasa secara umum. Senarai di atas adalah yang kami uji.

Known issues
Audio sangat panjang (ta/10.m4a, th/15.m4a) mungkin timeout 120s. Naikkan WHISPER_REQUEST_TIMEOUT atau pecahkan audio.

INT8 tidak disokong pada sm_120 dengan CTranslate2. float16 adalah satu-satunya pilihan stabil.

yue (Kantonis) — faster-whisper terima language=yue, tetapi kualiti mungkin lebih rendah dari zh.

Tiada diarization (siapa bercakap) — bukan keperluan CS.

Tiada word-level timestamps — tidak stabil dalam faster-whisper.

Versioning
Tag Docker: xchaix74/whisper:fat (latest), :fat-1.0.0 (spesifik).

WHISPER_VERSION env var. Semak via /health.

Untuk versi baharu: ubah WHISPER_VERSION dalam Dockerfile, build, tag fat-X.Y.Z dan fat.

Lesen
MIT

Acknowledgements
SYSTRAN/faster-whisper

OpenAI/whisper

OpenNMT/CTranslate2

Whisper FAT — CikguSayang ASR (English)
Production-ready faster-whisper image for CikguSayang (CS) — a free, OSS mobile AI language tutor.

Built specifically for NVIDIA RTX 5060 Ti (Blackwell, sm_120) with CUDA 12.8 and float16. No INT8 (not supported on sm_120 with CTranslate2).

Features
Engine: faster-whisper 1.2.1 + CTranslate2 4.8.1

Model: openai/whisper-large-v3-turbo (809M), bundled in image

VRAM: ~2.25 GB (float16)

Input: Opus / m4a / mp3 / WAV / webm — via PyAV (no FFmpeg executable)

Output: JSON {"text":"...", "language":"...", "language_probability":...}

Endpoint: POST /v1/audio/transcriptions (OpenAI-compatible subset)

Multilingual: 13+ languages tested (de, en, es, fr, it, ms, nl, pa, pt, ta, th, yue, zh)

VAD: Active — min_silence_duration_ms=500

Hallucination fix: condition_on_previous_text=False — stops loops

Multi-user: Queue depth 32, GPU lock, timeout 120s, max 600s audio, max 100MB file

Log: JSONL at /app/logs/whisper.jsonl (mount volume)

Version: 1.0.0

Prerequisites
GPU: NVIDIA with compute capability sm_120 (RTX 5060 Ti, 5070, 5080, 5090) or any GPU supported by CUDA 12.8

NVIDIA driver: ≥ 570 (supports CUDA 12.8)

Docker: with NVIDIA Container Toolkit (--gpus all)

VRAM: ~2.25 GB for model, + ~1 GB overhead

Disk: ~4.9 GB (compressed on Docker Hub)

Run
bash
docker run -d --name whisper \
  --gpus all \
  -p 8319:8000 \
  -v /path/to/audio:/audio \
  -v /path/to/logs:/app/logs \
  xchaix74/whisper:fat
API
Transcription
text
POST /v1/audio/transcriptions
Content-Type: multipart/form-data

Fields:
  file    : Opus/m4a/mp3/WAV (required)
  language: ISO 639-1 (optional; default auto-detect)
            "en" for English, "ms" for Malay, etc.
  task    : "transcribe" (default) or "translate"
Example:

bash
curl -X POST http://127.0.0.1:8319/v1/audio/transcriptions \
  -F "file=@audio.m4a" \
  -F "language=en"
Response:

json
{"text": "...", "language": "en", "language_probability": 0.998}
Health
bash
curl http://127.0.0.1:8319/health
# {"status": "ok", "version": "1.0.0"}
Limits
Limit	Value	Result if exceeded
File size	100 MB	HTTP 413
Audio duration	600s (10 min)	HTTP 413
Transcription time	120s	HTTP 504
Queue	32 concurrent	HTTP 429 (server busy)
Log
Each request is written to /app/logs/whisper.jsonl (one JSON line):

json
{
  "ts": "2026-11-05T22:05:47.706598+00:00",
  "version": "1.0.0",
  "request_id": "f3ed2e1dc0e6",
  "file": "10.m4a",
  "size": 51782075,
  "language": "ta",
  "task": "transcribe",
  "status": "error",
  "duration": 120.063,
  "vram_before_mb": 9204.0,
  "vram_after_mb": 9332.0,
  "text_len": 0,
  "error_type": "Timeout",
  "error_msg": "transcription exceeded 120.0s"
}
Supported languages (tested)
Code	Language
de	German
en	English
es	Spanish
fr	French
it	Italian
ms	Malay
nl	Dutch
pa	Punjabi
pt	Portuguese
ta	Tamil
th	Thai
yue	Cantonese
zh	Mandarin/Chinese
Whisper generally supports 99 languages. The list above is what we tested.

Known issues
Very long audio (ta/10.m4a, th/15.m4a) may hit the 120s timeout. Increase WHISPER_REQUEST_TIMEOUT or split the audio.

INT8 not supported on sm_120 with CTranslate2. float16 is the only stable option.

yue (Cantonese) — faster-whisper accepts language=yue, but quality may be lower than zh.

No diarization (who spoke) — not needed for CS.

No word-level timestamps — unstable in faster-whisper.

Versioning
Docker tags: xchaix74/whisper:fat (latest), :fat-1.0.0 (specific).

WHISPER_VERSION env var. Check via /health.

For new versions: change WHISPER_VERSION in Dockerfile, build, tag fat-X.Y.Z and fat.

License
MIT

Acknowledgements
SYSTRAN/faster-whisper

OpenAI/whisper

OpenNMT/CTranslate2
