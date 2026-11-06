# Whisper — CikguSayang ASR

Docker images untuk **speech-to-text** yang dibina untuk [CikguSayang (CS)](https://github.com/xchaix74) — tutor bahasa AI mudah alih, percuma, OSS.

Repo ini mengandungi tiga varian Whisper. Setiap varian ialah satu Docker image yang boleh dijalankan sendiri.

| Varian | Status | Enjin | Sasaran |
| --- | --- | --- | --- |
| [`fat/`](fat/) | ✅ Production | faster-whisper 1.2.1 + CTranslate2 4.8.1, large-v3-turbo, float16 | NVIDIA sm_120 (RTX 5060 Ti / 5070 / 5080 / 5090), CUDA 12.8 |
| `std/` | 📋 Planned | faster-whisper tanpa logik tersuai | GPU/CPU generik |
| `slim/` | 🧪 Experimental | whisper.cpp Q5_0, WAV sahaja | CPU / peranti kecil |

## Quick start (fat)

```bash
docker run -d --name whisper \
  --gpus all \
  -p 8304:8000 \
  -v /path/to/audio:/audio \
  -v /path/to/logs:/app/logs \
  xchaix74/whisper:fat
bash
curl -X POST http://127.0.0.1:8304/v1/audio/transcriptions \
  -F "file=@audio.m4a" \
  -F "language=en"
Respons:

json
{"text": "...", "language": "en", "language_probability": 0.998}
Dokumentasi penuh (dwi-bahasa): fat/README.md

Kenapa repo ini wujud
sm_120 (Blackwell) — image Whisper generik yang lama crash dengan no kernel image is available kerana PyTorch/CTranslate2 lama tiada kernel sm_120. Varian fat dibina khas dengan CUDA 12.8 + float16.

INT8 tidak disokong pada sm_120 dengan CTranslate2 — float16 satu-satunya pilihan stabil.

Bundled model — model large-v3-turbo masuk dalam image, tiada muat turun masa runtime.

OpenAI-compatible subset — POST /v1/audio/transcriptions, senang integrate.

Struktur repo
text
whisper/
├── fat/          # Production: faster-whisper large-v3-turbo, float16, sm_120
│   ├── Dockerfile
│   ├── server.py
│   ├── README.md
│   └── test.sh
├── std/          # Planned
├── slim/         # Experimental
├── LICENSE
└── README.md
Lesen
MIT — lihat LICENSE.

Acknowledgements
SYSTRAN/faster-whisper

OpenAI/whisper

OpenNMT/CTranslate2
