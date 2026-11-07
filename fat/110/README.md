# fat/110 — Whisper FAT 1.1.0

Varian **fat** versi 1.1.0 — faster-whisper large-v3-turbo, float16, GPU sm_120. Untuk production CS.

**Status:** 🧪 Ujian. Selepas lulus ujian brutal, akan jadi production (tag `fat`).

## Apa yang baharu berbanding 1.0.0

| Ciri | 1.0.0 | 1.1.0 |
|---|---|---|
| Had saiz fail | 100MB | **2GB** |
| Audio ≤5 minit | ✅ | ✅ (sama) |
| Audio 5-30 minit | ✅ (jika <100MB) | ✅ (chunking, sedikit overhed) |
| Audio >100MB (cth 1j 57m, 109MB) | ❌ **HTTP 413** | ✅ **Chunking automatik** |
| Audio >30 minit (bahasa padat: ta, th) | ⚠️ timeout 120s | ⚠️ **lambat (10-15 minit)**, tetapi berjaya |
| Timeout server | 120s | **900s** |
| Confidence gate | ❌ | ✅ (buang segmen low-confidence) |
| Repetition guard | ❌ | ✅ (potong frasa >3×) |
| Validasi audio | ❌ (500 jika rosak) | ✅ **400 jika rosak** |
| Config YAML | ❌ hardcoded | ✅ **boleh ubah tanpa rebuild** |
| `word_timestamps` | ❌ | ✅ (parameter) |
| `initial_prompt` | ❌ | ✅ (parameter) |
| `vad_threshold` | ❌ | ✅ (parameter) |

## Nota penting (dari ujian brutal 2026-11-07)

**1. Had sebenar 1.0.0 bukan "10 minit" — ia "100MB".**
- Audio 13m 54s (13MB) → 1.0.0 berjaya (11s).
- Audio 26m 17s (25MB) → 1.0.0 berjaya (26s).
- Audio 1j 57m (109MB) → 1.0.0 **HTTP 413** (saiz >100MB).
- Jadi 1.0.0 boleh proses audio panjang **selagi <100MB dan masa proses <120s**.

**2. Bahasa padat (Tamil, Thai) lambat.**
- Tamil 53m → 1.1.0 ambil ~12.8 minit (70s/chunk × 11 chunk), kerana VAD tiada senyap, setiap chunk penuh pertuturan.
- Melayu 1j 57m → 1.1.0 ambil 86s (24 chunk, ~3.5s/chunk).
- Ini **had model**, bukan kerosakan. Untuk CS real-time, guna audio <60s.

**3. Queue boleh block.**
- Jika audio panjang (53m) sedang proses, request baru (12s) menunggu dalam queue.
- `queue_max=32` — sehingga 32 request boleh menunggu.
- Untuk production CS, pertimbang `max_duration` parameter (akan datang 1.2.0) atau queue separation.

**4. Teks beza 0.1-1% berbanding 1.0.0.**
- Audio kecil (<60s): teks **sama** 100%.
- Audio sederhana (1-10m): beza 2-7 aksara (0.1-0.5%).
- Audio panjang (>10m): beza 4-132 aksara (0.5-1%).
- Ini kerana chunking potong pada senyap berbeza. Kualiti keseluruhan sama.

## Ciri

- **Enjin:** faster-whisper 1.2.1 + CTranslate2 4.8.1
- **Model:** openai/whisper-large-v3-turbo (809M), bundled
- **VRAM:** ~2.25 GB (float16)
- **Input:** Opus / m4a / mp3 / WAV / webm — melalui PyAV
- **Chunking:** automatik untuk audio >5 minit (fixed 5 min, sliding pada senyap)
- **Had saiz:** 2GB
- **Timeout:** 900s (15 minit)
- **Log:** JSONL di `/app/logs/whisper.jsonl` (tambah field `chunks`, `audio_duration_s`, `gated_segments`, `repetition_trimmed`)

**Nota untuk CS:** Untuk CS real-time, guna audio <60s. Audio panjang (podcast, ceramah) adalah untuk batch/script, bukan CS interaktif. Jika CS hantar audio panjang, ia akan masuk queue dan boleh block request lain.

## Jalankan

```bash
docker run -d --name whisper-fat110 \
  --gpus all \
  -p 8322:8000 \
  -v /path/to/audio:/audio \
  -v /path/to/logs:/app/logs \
  -v /opt/03-ai-heavy/whisper-config:/app/config \
  -e WHISPER_CONFIG=/app/config/fat.yaml \
  xchaix74/whisper:fat-1.1.0-test
Nota:

Port 8322 (bukan 8304) — elak clash dengan production fat/100 di 8304.

-v /opt/03-ai-heavy/whisper-config:/app/config — mount config luar (override).

-e WHISPER_CONFIG=/app/config/fat.yaml — tunjuk config override.

Jika tiada mount, guna config.yaml default dalam image.

API
Transkripsi
http
POST /v1/audio/transcriptions
Content-Type: multipart/form-data
Fields:

file: Opus/m4a/mp3/WAV/webm (wajib)

language: ISO 639-1 (pilihan; default auto-detect)

task: transcribe (default) atau translate

Query params (baharu 1.1.0):

word_timestamps: true/false (default false) — tambah segments[].words[] dengan start/end

initial_prompt: string (pilihan) — bias Whisper ke arah istilah domain

vad_threshold: float 0.0-1.0 (default 0.5) — lebih tinggi = lebih agresif buang bukan-pertuturan

Contoh:

bash
# Asas
curl -X POST http://127.0.0.1:8322/v1/audio/transcriptions \
  -F "file=@audio.m4a" \
  -F "language=ms"

# Dengan word_timestamps (untuk SRT)
curl -X POST "http://127.0.0.1:8322/v1/audio/transcriptions?word_timestamps=true" \
  -F "file=@audio.m4a" -F "language=ms"

# Dengan initial_prompt (ceramah agama)
curl -X POST "http://127.0.0.1:8322/v1/audio/transcriptions?initial_prompt=Kuliah%20Maghrib.%20Hadis%20riwayat%20Bukhari%20Muslim." \
  -F "file=@ceramah.m4a" -F "language=ms"
Respons asas:

json
{"text": "...", "language": "ms", "language_probability": 1.0}
Respons dengan word_timestamps=true:

json
{
  "text": "...",
  "language": "ms",
  "language_probability": 1.0,
  "segments": [
    {"id": 0, "start": 0.0, "end": 2.5, "text": "..."},
    {"id": 1, "start": 2.5, "end": 5.1, "text": "..."}
  ]
}
Health
bash
curl http://127.0.0.1:8322/health
# {"status":"ok","version":"1.1.0","config":{"max_file_mb":2048,"timeout_s":900,"chunk_seconds":300,"chunk_enabled":true,"queue_max":32}}
Config YAML
Default dalam image (/app/config.yaml). Override dengan mount:

yaml
server:
  max_file_mb: 2048          # 2GB
  request_timeout_s: 900     # 15 minit
  queue_max: 32

chunking:
  enabled: true
  chunk_seconds: 300         # 5 minit
  min_silence_ms: 500
  vad_threshold: 0.5

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
Ubah nilai, docker restart whisper-fat110. Tiada rebuild.

Had
Had	Nilai	Kesan jika melebihi
Saiz fail	2GB	HTTP 413
Durasi audio	Tiada had (chunking)	—
Timeout per request	900s	HTTP 504
Queue	32 serentak	HTTP 429
Format audio	Opus/m4a/mp3/WAV/webm	HTTP 400 (PyAV)
Chunking — bagaimana ia berfungsi
Audio diterima, divalidasi dengan PyAV (400 jika rosak).

Jika durasi ≤ 5 minit: transkrip terus.

Jika durasi > 5 minit:

Cari titik senyap (energy-based approximation).

Potong pada senyap terdekat dengan 5 minit (sliding ±20%).

Transkrip setiap chunk.

Adjust timestamp (offset).

Gabung teks.

Confidence gate + repetition guard pada hasil gabung.

Pulang JSON.

Nota: Chunking menggunakan energy-based silence detection, bukan Silero VAD. Ini cukup untuk audio dengan senyap jelas. Jika audio padat (tiada senyap), potong pada 5 minit tepat. Untuk kualiti lebih baik, silero-vad akan ditambah dalam iterasi seterusnya.

Known Issues (1.1.0-test)
Chunking menggunakan energy-based silence, bukan Silero VAD. Mungkin potong ayat pada audio padat.

initial_prompt tidak diuji secara meluas — mungkin mengganggu jika terlalu panjang (>500 aksara).

word_timestamps=true tambah ~10-20% masa proses.

Audio >2 jam mungkin ambil masa 10-15 minit (bergantung GPU).

Lesen
MIT — lihat LICENSE.

Acknowledgements
SYSTRAN/faster-whisper

OpenAI/whisper

OpenNMT/CTranslate2

PyAV (ffmpeg binding)
