"""FastAPI server untuk Whisper CS (CikguSayang).

Endpoint OpenAI-compatible:
  POST /v1/audio/transcriptions
    form: file (wajib), language (pilihan), task (pilihan)
    response: {"text": "...", "language": "...", "language_probability": ...}

Health:
  GET /health

Strategi multi-user:
- Satu GPU, satu worker. Akses GPU di-serialize dengan asyncio.Lock.
- Queue depth 32. Jika penuh, pulang HTTP 429.
- Timeout per request 120s. Jika lebih, pulang HTTP 504.
- Max audio duration 600s, max file size 100MB. Jika lebih, pulang HTTP 413.

Log:
- JSONL ke /app/logs/whisper.jsonl (mount volume untuk persist).
- Setiap request ada request_id, duration, status, vram_before, vram_after.
"""

import asyncio
import ctypes
import io
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("whisper.server")

# --- Konfigurasi ---
VERSION = os.environ.get("WHISPER_VERSION", "dev")
QUEUE_MAX = int(os.environ.get("WHISPER_QUEUE_MAX", "32"))
REQUEST_TIMEOUT = float(os.environ.get("WHISPER_REQUEST_TIMEOUT", "120"))
MAX_AUDIO_SECONDS = float(os.environ.get("WHISPER_MAX_AUDIO_SECONDS", "600"))
MAX_FILE_BYTES = int(os.environ.get("WHISPER_MAX_FILE_BYTES", str(100 * 1024 * 1024)))
LOG_DIR = os.environ.get("WHISPER_LOG_DIR", "/app/logs")
LOG_FILE = os.path.join(LOG_DIR, "whisper.jsonl")

# --- Preload CUDA libs dari pip ---
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

MODEL_ID = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")

app = FastAPI(title="Whisper CS", description="faster-whisper untuk CikguSayang")

logger.info("Version: %s", VERSION)
logger.info("Loading model %s (compute_type=%s)", MODEL_ID, COMPUTE_TYPE)
model = WhisperModel(
    MODEL_ID,
    device="cuda",
    compute_type=COMPUTE_TYPE,
)
logger.info("Model loaded")

# GPU lock: serialize akses GPU
_gpu_lock = asyncio.Lock()
# Queue semaphore: hadkan bilangan request serentak dalam queue
_queue_sem = asyncio.Semaphore(QUEUE_MAX)

# Pastikan log dir wujud
os.makedirs(LOG_DIR, exist_ok=True)


def _vram_mb() -> float:
    """Baca VRAM terpakai dari nvidia-smi (MB). Fallback: -1 jika gagal."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=2, stderr=subprocess.DEVNULL,
        )
        return float(out.decode().strip().splitlines()[0])
    except Exception:
        return -1.0


def _write_log(entry: dict) -> None:
    """Tulis satu baris JSONL ke log fail."""
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("gagal tulis log: %s", e)


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(None),
    task: str = Form("transcribe"),
):
    req_id = uuid.uuid4().hex[:12]
    t0 = time.time()
    vram_before = _vram_mb()
    size = 0
    status = "error"
    error_type = None
    error_msg = None
    text = ""
    info = None

    # Semak saiz fail (dari header Content-Length jika ada, atau baca)
    try:
        audio_bytes = await file.read()
        size = len(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"gagal baca fail: {e}")

    if size == 0:
        raise HTTPException(status_code=400, detail="fail kosong")
    if size > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"fail terlalu besar: {size} bytes (max {MAX_FILE_BYTES})",
        )

    # Semak queue: jika penuh, pulang 429
    acquired = _queue_sem.locked()
    if acquired:
        # Semaphore tidak boleh cek terus; cuba acquire dengan timeout 0
        got = False
        try:
            got = await asyncio.wait_for(_queue_sem.acquire(), timeout=0.001)
        except (asyncio.TimeoutError, Exception):
            got = False
        if not got:
            _write_log({
                "ts": datetime.now(timezone.utc).isoformat(),
                "version": VERSION,
                "request_id": req_id,
                "file": file.filename,
                "size": size,
                "language": language,
                "task": task,
                "status": "rejected_queue_full",
                "duration": round(time.time() - t0, 3),
                "vram_before_mb": vram_before,
            })
            raise HTTPException(status_code=429, detail="server busy, cuba lagi")
    else:
        await _queue_sem.acquire()

    try:
        # Tunggu GPU lock (serialize)
        try:
            await asyncio.wait_for(_gpu_lock.acquire(), timeout=REQUEST_TIMEOUT)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="timeout menunggu GPU")

        try:
            # Jalankan transkripsi dalam thread (blocking) dengan timeout
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
                txt = " ".join(s.text.strip() for s in segs)
                return txt, inf

            text, info = await asyncio.wait_for(
                asyncio.to_thread(_do_transcribe),
                timeout=REQUEST_TIMEOUT,
            )
            status = "ok"
        except asyncio.TimeoutError:
            error_type = "Timeout"
            error_msg = f"transkripsi melebihi {REQUEST_TIMEOUT}s"
            raise HTTPException(status_code=504, detail=error_msg)
        except RuntimeError as e:
            error_type = "RuntimeError"
            error_msg = str(e)
            raise HTTPException(status_code=500, detail=error_msg)
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            raise HTTPException(status_code=500, detail=error_msg)
        finally:
            _gpu_lock.release()
    finally:
        _queue_sem.release()
        vram_after = _vram_mb()
        _write_log({
            "ts": datetime.now(timezone.utc).isoformat(),
            "version": VERSION,
            "request_id": req_id,
            "file": file.filename,
            "size": size,
            "language": language,
            "task": task,
            "status": status,
            "duration": round(time.time() - t0, 3),
            "vram_before_mb": vram_before,
            "vram_after_mb": vram_after,
            "text_len": len(text),
            "detected_language": getattr(info, "language", None),
            "language_probability": round(getattr(info, "language_probability", 0), 3),
            "error_type": error_type,
            "error_msg": error_msg,
        })

    return {
        "text": text,
        "language": info.language,
        "language_probability": info.language_probability,
    }


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}
