# Whisper — CikguSayang ASR

**Tiga Docker image untuk speech-to-text.** Dibina untuk [CikguSayang](https://github.com/xchaix74) — tutor bahasa AI mudah alih, percuma, OSS. Semua varian guna API yang sama: `POST /v1/audio/transcriptions`.

`docker pull xchaix74/whisper:latest` → **fat 1.1.0** (production)

---

## Mengapa Whisper ini?

Ramai pembangun di Asia Tenggara mahu bina aplikasi suara — tutor bahasa, transkripsi ceramah, nota mesyuarat, subtitle video. Tetapi mereka berdepan masalah yang sama:

- **GPU baru tak disokong.** Kad seperti NVIDIA RTX 5060 Ti (Blackwell, sm_120) tidak berfungsi dengan image Whisper lama. Ralat `no kernel image is available`. Berjam-jam debug, akhirnya menyerah.
- **Audio panjang gagal.** Podcast 1 jam, ceramah agama 2 jam, rakaman mesyuarat 3 jam — image biasa tolak kerana had saiz atau timeout. Tidak boleh digunakan untuk kandungan sebenar.
- **Bahasa Melayu lemah.** Model kecil tersilap teruk. "Selamat pagi" jadi "Selamat Poggy". Nama khas seperti "Yang di-Pertuan Agong" jadi tidak dikenali.
- **Server tersekat.** Satu pengguna hantar audio panjang, semua pengguna lain tunggu. Tiada queue, tiada keadilan.

**Whisper ini menyelesaikan semua masalah itu.** Ia dibina khas untuk GPU Blackwell, memproses audio sehingga 2GB (2.5 jam), menyokong 13+ bahasa termasuk Melayu, dan mempunyai queue yang adil — audio pendek tidak tersekat oleh audio panjang.

## Tiga varian, satu API

| Varian | Untuk siapa | Saiz | GPU? |
| :--- | :--- | :--- | :--- |
| **fat** | Production, ramai pengguna, audio panjang | ~5.2GB | Ya (sm_120) |
| **std** | Single-user, dev, ujian | ~5.2GB | Ya |
| **slim** | Laptop/desktop biasa, audio pendek | ~92MB | Tidak |

## Mula dalam 30 detik

```bash
# Production (GPU Blackwell)
docker run -d --name whisper --gpus all -p 8304:8000 \
  xchaix74/whisper:latest

# Uji
curl -X POST http://127.0.0.1:8304/v1/audio/transcriptions \
  -F "file=@audio.m4a" -F "language=ms"
```

**Respons:**

```json
{"text": "...", "language": "ms", "language_probability": 1.0}
```

> **Untuk pembaca teknikal**
> Seksyen penuh dalam Bahasa Inggeris di bawah — senibina, hasil ujian brutal (675 audio, 40 serentak, 2j 32m), perbandingan 1.0.0 vs 1.1.0, cara ia berfungsi, dan pautan ke GitHub.

---

# Whisper — CikguSayang ASR (English)

Three Docker images for speech-to-text, built for CikguSayang — a free, OSS mobile AI language tutor. All variants share the same API: `POST /v1/audio/transcriptions`.

Docker tags: `latest` = `fat` = `fat-1.1.0` (1.1.0). `fat-1.0.0` (legacy). `std`, `slim`.

### Table of Contents

1. [The problem](#1-the-problem)
2. [The solution: three variants](#2-the-solution-three-variants)
3. [What's new in fat 1.1.0](#3-whats-new-in-fat-110)
4. [Benchmarks: 1.0.0 vs 1.1.0](#4-benchmarks-100-vs-110)
5. [How it works](#5-how-it-works)
6. [Installation](#6-installation)
7. [API reference](#7-api-reference)
8. [Configuration](#8-configuration)
9. [Brutal test results](#9-brutal-test-results)
10. [Known limitations](#10-known-limitations)
11. [Roadmap](#11-roadmap)
12. [License and links](#12-license-and-links)

### 1. The problem

Building speech-to-text for Southeast Asian languages in 2026 is painful. The community hits the same walls:

- **GPU compatibility.** New NVIDIA GPUs (RTX 5060 Ti and above, Blackwell architecture, sm_120) crash with `no kernel image is available` on older Whisper images. PyTorch/CTranslate2 versions bundled in those images were compiled for older architectures. Developers spend hours debugging, then give up.
- **Long audio.** Real content — podcasts, religious lectures, meeting recordings — is 1-3 hours. Standard Whisper images reject files >100MB or timeout at 120s. They are unusable for real content.
- **Low-resource languages.** Whisper tiny/base models produce garbage for Malay, Tamil, Thai, Punjabi, Cantonese. Small models cannot handle non-Latin scripts or domain terminology.
- **Concurrency.** One user submits a 2-hour lecture, everyone else waits. No queue, no fairness, no backpressure.

### 2. The solution: three variants

| Feature | fat | std | slim |
| :--- | :--- | :--- | :--- |
| **Engine** | faster-whisper 1.2.1 + CTranslate2 4.8.1 | same | whisper.cpp v1.9.4 |
| **Model** | large-v3-turbo (809M) | large-v3-turbo (809M) | base-q5_1 (~57MB) |
| **Compute** | float16 | float16 | q5_1 quantized |
| **Device** | GPU (CUDA 12.8, sm_120) | GPU (CUDA 12.8) | CPU |
| **VRAM** | ~2.25 GB | ~2.25 GB | 0 |
| **Image size** | ~5.2 GB | ~5.2 GB | ~92 MB |
| **Max file** | 2GB | 100MB | 2GB (WAV) |
| **Chunking** | ✅ | ❌ | ❌ |
| **Queue separation** | ✅ | ❌ | ❌ |

### 3. What's new in fat 1.1.0

Compared to 1.0.0:

- **Automatic chunking** for audio >5 min. Fixed 5-min chunks, sliding on silence (energy-based approximation).
- **Queue separation:** fast queue (audio ≤ max_duration) vs slow queue (long audio). Fast queue not blocked by slow queue.
- **GPU lock per chunk:** lock released between chunks, so fast requests can interleave.
- **max_duration parameter:** early 413 rejection for CS real-time. Prevents queue blocking.
- **chunk_timeout_s (default 180s):** per-chunk timeout. A stuck chunk is logged and skipped, not blocking the whole request.
- **fast_gpu_timeout_s (default 30s):** GPU lock timeout for fast queue. Returns 503 if GPU busy.
- **Confidence gate:** discard segments with `no_speech_prob > 0.6` AND `avg_logprob < -1.0`.
- **Repetition guard:** trim phrases repeated >3 times.
- **PyAV validation:** 400 for corrupted/non-audio files (1.0.0 returned 500).
- **Config YAML:** all hardcoded values in `/app/config.yaml`, overridable via mount.
- **word_timestamps** (segment-level), **initial_prompt**, **vad_threshold** parameters.
- **Max file size:** 100MB → 2GB. **Timeout:** 120s → 900s.

### 4. Benchmarks: 1.0.0 vs 1.1.0

Tested on node1 (Ryzen 5 7600, RTX 5060 Ti 16GB, 2026-11-07). 11 languages, 3 audio each (small, medium, large). Full data in GitHub `fat/110/README.md`.

#### 4.1 Audio duration vs processing time

| Language | Audio | 1.0.0 | 1.1.0 |
| :--- | :--- | :--- | :--- |
| ms | 1h 57m (109MB) | ❌ HTTP 413 (size) | ✅ 86s |
| en | 2h 32m (141MB) | ❌ HTTP 413 (size) | ✅ 102s |
| de | 13m 54s | ✅ 11s | ✅ 14s |
| es | 26m 17s | ✅ 26s | ✅ 28s |
| fr | 11m 35s | ✅ 6s | ✅ 8s |
| it | 18m 33s | ✅ 9s | ✅ 12s |
| nl | 18m 26s | ✅ 17s | ✅ 20s |
| pt | 19m 46s | ✅ 21s | ✅ 25s |
| zh | 16m 34s | ✅ 18s | ✅ 20s |
| ta | 53m 22s | ❌ HTTP 504 (timeout) | ✅ 315s (chunking) |
| th | 2m 38s | ✅ 3s | ✅ 3s |

> **Key finding:** 1.0.0's real limit is 100MB, not 10 minutes. Audio <100MB works (13m, 26m). Audio >100MB (109MB, 141MB) returns 413. 1.1.0's chunking handles all sizes. Overhead of chunking is 1-3s for medium audio.

#### 4.2 Text quality

| Audio | Same text? | Char diff |
| :--- | :--- | :--- |
| Small (<60s) | ✅ 100% same | 0 |
| Medium (1-10m) | ⚠ 99.5-99.9% | 2-7 chars |
| Large (>10m) | ⚠ 99-99.5% | 4-132 chars |

> **Key finding:** chunking splits on different silence points, so text differs slightly. Quality is equivalent. Small audio is identical.

#### 4.3 Concurrency (1.1.0)

| Test | Result |
| :--- | :--- |
| 20 concurrent (short audio, max_duration=60) | ✅ 20/20 HTTP 200, wall 15s |
| 40 concurrent (mixed fast/slow) | ✅ 40/40 HTTP 200, wall 51s |
| th/1 during ta/10 (queue block test) | ✅ 30.77s (1.0.0: timeout 60s) |

> **Key finding:** queue separation works. Audio pendek tidak tersekat oleh audio panjang.

#### 4.4 Validation

| Test | 1.0.0 | 1.1.0 |
| :--- | :--- | :--- |
| Text file disguised as .m4a | ❌ HTTP 500 | ✅ HTTP 400, 0.024s |
| Empty file | 400 | 400 |
| Audio >100MB | ❌ 413 | ✅ chunking |

### 5. How it works

#### 5.1 Request flow

1. Receive file → size check (413 if >2GB).
2. Validate with PyAV → 400 if corrupted / no audio stream.
3. Read duration → decide fast vs slow queue.
4. `max_duration` check → 413 early if audio > limit (CS real-time).
5. Queue acquire → fast (`_fast_sem`, 16 slots) or slow (`_slow_sem`, 16 slots). Timeout 30s for fast, 900s for slow.
6. Process:
   - Audio ≤5 min: single `model.transcribe()`.
   - Audio >5 min: silence detection → chunk at ~5 min → transcribe each chunk → adjust timestamps → merge.
7. Post-process: confidence gate, repetition guard.
8. Return JSON with `text`, `language`, `language_probability`, optionally `segments`.

#### 5.2 GPU lock per chunk

The critical design: instead of holding GPU lock for the entire long request, we acquire/release per chunk. Between chunks, fast queue requests can grab the lock.

```
ta/10 (slow, 53m, 11 chunks):
  chunk 1: acquire → transcribe (27s) → release
  [fast queue: th/1 acquires → transcribe (0.5s) → release]
  chunk 2: acquire → transcribe (27s) → release
  ...
Result: th/1 waits ~30s (one chunk), not 315s (whole request).
```

#### 5.3 Virtual chunking (no extra storage)

- PyAV reads audio as a container, `seek()` to offset, `decode()` stream frames.
- Chunk = in-memory WAV bytes (16kHz mono). No temp files.
- Total storage: zero extra. Memory: one chunk at a time (~10MB WAV for 5 min).

#### 5.4 Config precedence

1. `/app/config.yaml` (in image, default)
2. Mount override at `/app/config/fat.yaml` + env `WHISPER_CONFIG=/app/config/fat.yaml`
3. Change values, `docker restart`, no rebuild.

### 6. Installation

**fat (production)**

```bash
docker run -d --name whisper-fat \
  --gpus all \
  -p 8304:8000 \
  -v /path/to/audio:/audio \
  -v /path/to/logs:/app/logs \
  -v /path/to/config:/app/config \
  -e WHISPER_CONFIG=/app/config/fat.yaml \
  xchaix74/whisper:latest
```

Requires: NVIDIA driver ≥570, NVIDIA Container Toolkit, GPU sm_120 (RTX 5060 Ti / 5070 / 5080 / 5090).

**std (single-user)**

```bash
docker run -d --name whisper-std --gpus all -p 8320:8000 xchaix74/whisper:std
```

**slim (CPU)**

```bash
docker run -d --name whisper-slim -p 8321:8000 xchaix74/whisper:slim
```

No GPU, no CUDA.

### 7. API reference

#### POST /v1/audio/transcriptions

Multipart form:

- `file` (required): Opus / m4a / mp3 / WAV / webm
- `language` (optional): ISO 639-1 (ms, en, ...). Default auto-detect.
- `task` (optional): `transcribe` (default) or `translate`.

**Query params (fat 1.1.0):**

- `max_duration` (float): reject 413 if audio > this. For CS real-time, use 60.
- `word_timestamps` (bool): include `segments[].start/end`. Default false.
- `initial_prompt` (str): bias toward domain terms. Example: `"Kuliah Maghrib. Hadis riwayat Bukhari Muslim. Istilah: sunnah, wajib, haram."`
- `vad_threshold` (float 0-1): default 0.5. Higher = more aggressive non-speech removal.

#### GET /health

Returns version and config:

```json
{"status":"ok","version":"1.1.0","config":{"max_file_mb":2048,"timeout_s":900,"chunk_seconds":300,"chunk_timeout_s":180,"queue_max":32,"max_duration_s":0,"fast_gpu_timeout_s":30}}
```

### 8. Configuration

Default `/app/config.yaml`:

```yaml
server:
  max_file_mb: 2048
  request_timeout_s: 900
  queue_max: 32
  max_duration_s: 0          # 0 = no limit
  fast_gpu_timeout_s: 30

chunking:
  enabled: true
  chunk_seconds: 300
  min_silence_ms: 500
  vad_threshold: 0.5
  chunk_timeout_s: 180

transcription:
  word_timestamps_default: false
  condition_on_previous_text: false
  beam_size: 5

confidence_gate:
  enabled: true
  no_speech_prob_max: 0.6
  avg_logprob_min: -1.0

repetition_guard:
  enabled: true
  max_repeats: 3
```

### 9. Brutal test results

Full data: GitHub `fat/110/README.md`.

**Test 1 (675 audio files, all languages):**

241 HTTP 200 (audio ≤60s), 434 HTTP 413 (audio >60s, max_duration=60 early reject), 0 crash/timeout/500.

Wall time: 296s.

**Test 2 (multi-connection):**

- 20 concurrent: 20/20 HTTP 200, wall 15s.
- 40 concurrent: 40/40 HTTP 200, wall 51s.
- VRAM stable (7946 MB), container Up 24+ minutes, no crash.

**Test 3 (queue block):**

- ta/10 (53m Tamil) processing + th/1 (12s Thai) arriving 3s later.
- th/1 completed in 30.77s (1.0.0: timeout 60s).
- ta/10 completed in 309s (chunking).

### 10. Known limitations

- **Dense languages (Tamil, Thai) slow for long audio.** 53m Tamil = ~5-13 min (70s/chunk, no silence). This is a model limitation, not a bug.
- **jv (Javanese) not supported.** Whisper has 99 languages; jv is not one. Use `id` (Indonesian) or `ms` (Malay) as fallback.
- **Energy-based silence detection, not Silero VAD.** Works for audio with clear silences. Audio with no silence splits on fixed 5-min. Silero VAD planned for 1.2.0.
- **Text differs 0.1-1% from 1.0.0** on medium/large audio (chunking splits differently). Small audio identical.
- **No async job.** For audio >30 min, client waits. Async job (202 Accepted + poll) planned for 1.2.0.

### 11. Roadmap

**fat 1.2.0 (planned):**

- word_timestamps per word (currently segment-level) → SRT generation in client
- Async job (202 Accepted + `/job/{id}`) for very long audio
- Silero VAD chunking (more accurate than energy-based)
- Queue priority refinement (more than binary fast/slow)
- `--smartchunk` (variable chunk size based on audio density)

**External scripts (separate repo, planned):**

- `whisper-subtitle.sh`: URL (YouTube) → SRT, using yt-dlp + fat
- `whisper-batch.sh`: folder → transcripts, using fat

### 12. License and links

- **License:** MIT — LICENSE
- **GitHub:** github.com/xchaix74/whisper
- **fat 1.0.0 docs:** `fat/100/README.md`
- **fat 1.1.0 docs:** `fat/110/README.md`
- **std docs:** `std/README.md`
- **slim docs:** `slim/README.md`
- **Base code:** SYSTRAN/faster-whisper, OpenAI/whisper, OpenNMT/CTranslate2, PyAV, ggerganov/whisper.cpp (slim)
