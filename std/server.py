"""FastAPI server ringkas untuk Whisper STD (CikguSayang).

Endpoint OpenAI-compatible:
  POST /v1/audio/transcriptions
    form: file (wajib), language (pilihan), task (pilihan)
    response: {"text": "...", "language": "...", "language_probability": ...}

Health:
  GET /health

Tiada queue, tiada GPU lock, tiada log JSONL.
Untuk production multi-user, guna varian fat/.
"""

import ctypes
import io
import os

# --- Preload CUDA libs dari pip (sama macam fat) ---
_preload_dirs = [
    "/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib",
    "/opt/venv/lib/python3.12/site-packages/nvidia/cublas/lib",
    "/usr/local/cuda/targets/x86_64-linux/lib",
]
for _d in _preload_dirs:
    if not os.path.isdir(_d):
        continue
    for _f in sorted(os.listdir(_d)):
        if not (_f.endswith(".so") or ".so." in _f):
            continue
        try:
            ctypes.CDLL(os.path.join(_d, _f), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from faster_whisper import WhisperModel

VERSION = os.environ.get("WHISPER_VERSION", "dev")
MODEL_ID = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")

app = FastAPI(title="Whisper STD", description="faster-whisper ringkas untuk CikguSayang")

model = WhisperModel(MODEL_ID, device="cuda", compute_type=COMPUTE_TYPE)


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(None),
    task: str = Form("transcribe"),
):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="fail kosong")

    def _do_transcribe():
        segs, inf = model.transcribe(
            io.BytesIO(audio_bytes),
            beam_size=5,
            language=language if language else None,
            task=task,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segs), inf

    try:
        text, info = _do_transcribe()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    return {
        "text": text,
        "language": info.language,
        "language_probability": info.language_probability,
    }


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}
