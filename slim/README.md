# slim/ — Whisper ringan (CPU, audio pendek)

Varian **slim** — whisper.cpp ringkas untuk CPU atau GPU kecil (VRAM ~0.5GB). Image ~252MB, tiada PyTorch/CTranslate2, tiada CUDA.

Sasaran: laptop/desktop biasa, audio pendek, single-user.

## Ciri

- **Enjin:** whisper.cpp v1.9.4 (C++)
- **Model:** base-q5_1 (~57MB), bundled dalam image
- **Device:** CPU lalai (GPU pilihan via tag berasingan, belum dibina)
- **Input:** WAV 16kHz mono **sahaja** (m4a/mp3 ditolak HTTP 400)
- **Output:** JSON `{"text":"...", "language":"...", "language_probability":...}`
- **Endpoint:** `POST /v1/audio/transcriptions` (OpenAI-compatible subset)
- **Health:** `GET /health`
- **Image size:** ~252MB (compressed ~92MB)
- **Version:** 0.1.0

## Prestasi (diukur pada Ryzen 5 7600, CPU sahaja)

| Audio | Masa | Nota |
| --- | --- | --- |
| 4.2s (ujian_ms.wav) | **0.56s** | warm, selepas request pertama |
| 1-3s | 0.6-2s | pantas, sesuai |
| 5-10s | 5-15s | mula lambat |
| 30-60s | 20-56s | tidak praktikal |
| >60s | timeout 120s | gagal |

**Kesimpulan:** `slim` sesuai untuk audio **<10s**. Untuk audio panjang, guna [`fat/`](../fat/) atau [`std/`](../std/).

## Kualiti Transkrip

Diuji pada 143 WAV (13 bahasa). Keputusan jujur:

| Bahasa | Kualiti |
| --- | --- |
| en, de, es, fr, it, nl, pt | Sederhana — boleh difahami, ada kesilapan |
| ms | Sederhana — "Selamat pagi" betul, bahagian lain tersilap |
| zh | Sederhana |
| ta, th, pa, yue | **Teruk** — aksara bercampur, tidak boleh difahami |

Model `base-q5_1` terlalu kecil untuk bahasa bukan-Latin. Untuk bahasa tersebut, guna `fat/` (large-v3-turbo).

## Kestabilan

Diuji 143 WAV berturut-turut (32 minit):

- **140/143 HTTP 200** (97.9%)
- **3 timeout** (ta/10.wav, th/13.wav, th/15.wav — audio panjang)
- Container kekal hidup, tiada crash

Ujian beban: 8 request serentak → semua HTTP 200, wall 5s. `whisper-server` **serialize** request (satu demi satu), bukan serentak. Sesuai untuk single-user, bukan multi-user.

## Jalankan

```bash
docker run -d --name whisper-slim \
  -p 8321:8000 \
  -v /path/to/audio:/audio \
  xchaix74/whisper:slim
```

> Tiada `--gpus` — slim berjalan pada CPU. Jika ada GPU, ia tidak digunakan (melainkan tag GPU berasingan dibina kemudian).

## API

### Transkripsi

```bash
curl -X POST http://127.0.0.1:8321/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "language=ms"
```

**Respons:**

```json
{"text": "...", "language": "ms", "language_probability": 1.0}
```

### Health

```bash
curl http://127.0.0.1:8321/health
# {"status":"ok","version":"0.1.0"}
```

## Had

| Had | Nilai | Kesan |
| --- | --- | --- |
| Format input | WAV 16kHz mono | m4a/mp3 → HTTP 400 "Invalid request" |
| Audio panjang | Tiada had dalaman | Klien perlu -m (curl timeout) |
| Serentak | Serialize (1 demi 1) | Bukan untuk multi-user |
| Bahasa bukan-Latin | Kualiti rendah | Guna fat/ untuk ta/th/pa/yue |

## Tiada (berbanding fat/std)

- ❌ Queue depth + HTTP 429
- ❌ GPU lock
- ❌ Timeout dalaman + HTTP 504
- ❌ Log JSONL
- ❌ VRAM tracking
- ❌ Sokongan m4a/mp3

## Lesen

MIT — lihat LICENSE.
