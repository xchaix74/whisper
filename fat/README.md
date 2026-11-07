# fat/ — Whisper production (GPU, multi-user)

Varian **fat** — faster-whisper large-v3-turbo, float16, GPU sm_120. Untuk production CS.

Folder ini mengandungi dua versi:

| Versi | Folder | Status | Ciri utama |
|---|---|---|---|
| **1.0.0** | [`100/`](100/) | ✅ Production stabil | Queue 32, GPU lock, timeout 120s, log JSONL. Audio ≤10 minit. |
| **1.1.0** | [`110/`](110/) | 🧪 Ujian | Chunking automatik (audio >5 minit), confidence gate, validasi PyAV, config YAML. Audio tanpa had durasi. |

## Pilih versi

- **Production hari ini:** guna `100/` (1.0.0). Stabil, telah lulus ujian 422 fail.
- **Audio panjang (>10 minit):** guna `110/` (1.1.0). Chunking automatik, tiada had durasi. **Uji dahulu** sebelum production.

## Docker tags

| Tag Docker | Versi | Folder |
|---|---|---|
| `xchaix74/whisper:fat-1.0.0` | 1.0.0 | `100/` |
| `xchaix74/whisper:fat` | 1.0.0 (sekarang) | `100/` |
| `xchaix74/whisper:fat-1.1.0` | 1.1.0 | `110/` (selepas ujian lulus) |

Selepas 1.1.0 lulus ujian brutal, tag `fat` akan naik ke 1.1.0.

## Dokumentasi penuh

- [`100/README.md`](100/README.md) — 1.0.0 (API, had, log, known issues)
- [`110/README.md`](110/README.md) — 1.1.0 (ciri baharu, config YAML, chunking)

## Lesen

MIT — lihat [LICENSE](../LICENSE).
