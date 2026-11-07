# fat/ — Whisper production (GPU, multi-user)

Varian **fat** — faster-whisper large-v3-turbo, float16, GPU sm_120. Untuk production CS.

Folder ini mengandungi dua versi:

| Versi | Folder | Status | Ciri utama |
|---|---|---|---|
| **1.1.0** | [`110/`](110/) | ✅ **Production (terkini)** | Chunking automatik, queue separation, `max_duration`, confidence gate, validasi PyAV, config YAML. Audio tanpa had durasi (≤2GB). |
| **1.0.0** | [`100/`](100/) | 📦 Legacy (rollback) | Queue 32, GPU lock, timeout 120s, log JSONL. Had saiz 100MB. |

## Pilih versi

- **Production hari ini:** guna `110/` (1.1.0). Lulus ujian brutal: 675 audio, 40 serentak, audio 2j 32m (141MB), queue separation.
- **Rollback:** guna `100/` (1.0.0) jika 1.1.0 ada isu. Stabil, tetapi had saiz 100MB (audio >100MB → 413).

## Perbezaan utama 1.0.0 vs 1.1.0

| Ciri | 1.0.0 | 1.1.0 |
|---|---|---|
| Had saiz | 100MB | **2GB** |
| Audio >100MB | ❌ 413 | ✅ Chunking automatik |
| Queue | Satu queue | **Fast (pendek) + slow (panjang)** |
| GPU lock | Per request | **Per chunk** (elak queue block) |
| `max_duration` | ❌ | ✅ (tolak 413 awal untuk CS) |
| Confidence gate | ❌ | ✅ |
| Repetition guard | ❌ | ✅ |
| Validasi audio | 500 jika rosak | **400** |
| Config YAML | ❌ | ✅ (boleh ubah tanpa rebuild) |
| `word_timestamps` | ❌ | ✅ (segment-level) |
| `initial_prompt` | ❌ | ✅ |
| `vad_threshold` | ❌ | ✅ |

## Docker tags

| Tag Docker | Versi | Folder |
|---|---|---|
| `xchaix74/whisper:fat` | **1.1.0** (latest) | `110/` |
| `xchaix74/whisper:fat-1.1.0` | 1.1.0 | `110/` |
| `xchaix74/whisper:fat-1.0.0` | 1.0.0 (legacy) | `100/` |

## Dokumentasi penuh

- [`110/README.md`](110/README.md) — 1.1.0 (chunking, queue separation, config YAML, API)
- [`100/README.md`](100/README.md) — 1.0.0 (API, had, log, known issues)

## Nota penting (dari ujian brutal 2026-11-07)

1. **Bahasa padat (Tamil, Thai) lambat** untuk audio panjang — 53m Tamil = ~5-13 minit. Ini had model, bukan kerosakan.
2. **Untuk CS real-time, guna `max_duration=60`.** Audio >60s ditolak 413 awal, tidak block queue.
3. **`jv` (Jawa) tidak disokong** Whisper. Guna `id` (Indonesia) atau `ms` (Melayu) sebagai alternatif.
4. **Teks mungkin beza 0.1-1%** berbanding 1.0.0 kerana chunking. Audio kecil (<60s) sama 100%.

## Lesen

MIT — lihat [LICENSE](../LICENSE).
