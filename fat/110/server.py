"""FastAPI server untuk Whisper FAT 1.1.0 (CikguSayang).

Ciri baharu berbanding 1.0.0:
  - Chunking automatik untuk audio >5 minit (fixed chunk, sliding pada senyap)
  - Confidence gate (buang segmen low-confidence)
  - Repetition guard (potong frasa >3×)
  - Validasi PyAV (400 untuk fail rosak)
  - Config YAML (boleh ubah tanpa rebuild)
  - word_timestamps + initial_prompt parameter
  - Had saiz: 100MB → 2GB
  - Timeout: 120s → 900s

Endpoint OpenAI-compatible:
  POST /v1/audio/transcriptions
    form: file (wajib), language (pilihan), task (pilihan)
    query: word_timestamps (pilihan), initial_prompt (pilihan), vad_threshold (pilihan)
    response: {"text": "...", "language": "...", "language_probability": ...}

Health:
  GET /health

Log:
  JSONL ke /app/logs/whisper.jsonl (mount volume untuk persist).
"""

import asyncio
import io
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import av
import yaml
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from faster_whisper import WhisperModel

# --- Logging ---
logging.basicConfig(
    level=os.environ.get("WHISPER_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("whisper.server")

# --- Baca config ---
def load_config() -> dict:
    """Baca config dari WHISPER_CONFIG (override) atau /app/config.yaml (default)."""
    default = {
        "server": {"max_file_mb": 2048, "request_timeout_s": 900, "queue_max": 32},
        "chunking": {"enabled": True, "chunk_seconds": 300, "min_silence_ms": 500, "vad_threshold": 0.5},
        "transcription": {"word_timestamps_default": False, "condition_on_previous_text": False, "beam_size": 5},
        "confidence_gate": {"enabled": True, "no_speech_prob_max": 0.6, "avg_logprob_min": -1.0},
        "repetition_guard": {"enabled": True, "max_repeats": 3},
    }
    path = os.environ.get("WHISPER_CONFIG", "/app/config.yaml")
    try:
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        # Deep merge: user override default
        for section, values in user.items():
            if isinstance(values, dict) and section in default:
                default[section].update(values)
            else:
                default[section] = values
        logger.info("Config loaded from %s", path)
    except FileNotFoundError:
        logger.warning("Config %s tidak wujud, guna default", path)
    except Exception as e:
        logger.error("Gagal baca config %s: %s, guna default", path, e)
    return default

CFG = load_config()
VERSION = os.environ.get("WHISPER_VERSION", "1.1.0")
LOG_DIR = os.environ.get("WHISPER_LOG_DIR", "/app/logs")
LOG_FILE = os.path.join(LOG_DIR, "whisper.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)

MAX_FILE_BYTES = int(CFG["server"]["max_file_mb"]) * 1024 * 1024
REQUEST_TIMEOUT = float(CFG["server"]["request_timeout_s"])
QUEUE_MAX = int(CFG["server"]["queue_max"])

CHUNK_SECONDS = float(CFG["chunking"]["chunk_seconds"])
CHUNK_ENABLED = bool(CFG["chunking"]["enabled"])
MIN_SILENCE_MS = int(CFG["chunking"]["min_silence_ms"])
VAD_THRESHOLD = float(CFG["chunking"]["vad_threshold"])

WORD_TS_DEFAULT = bool(CFG["transcription"]["word_timestamps_default"])
COND_PREV_TEXT = bool(CFG["transcription"]["condition_on_previous_text"])
BEAM_SIZE = int(CFG["transcription"]["beam_size"])

GATE_ENABLED = bool(CFG["confidence_gate"]["enabled"])
GATE_NOSPEECH_MAX = float(CFG["confidence_gate"]["no_speech_prob_max"])
GATE_LOGPROB_MIN = float(CFG["confidence_gate"]["avg_logprob_min"])

REP_ENABLED = bool(CFG["repetition_guard"]["enabled"])
REP_MAX = int(CFG["repetition_guard"]["max_repeats"])

logger.info("Version: %s", VERSION)
logger.info("Config: max_file=%dMB, timeout=%ds, chunk=%ds, queue=%d",
            MAX_FILE_BYTES // (1024*1024), int(REQUEST_TIMEOUT), int(CHUNK_SECONDS), QUEUE_MAX)

# --- Model ---
MODEL_ID = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")
logger.info("Loading model %s (compute_type=%s)", MODEL_ID, COMPUTE_TYPE)
model = WhisperModel(MODEL_ID, device="cuda", compute_type=COMPUTE_TYPE)
logger.info("Model loaded")

# --- Locks ---
_gpu_lock = asyncio.Lock()
_queue_sem = asyncio.Semaphore(QUEUE_MAX)

# --- Helpers ---

def _write_log(entry: dict) -> None:
    """Tulis satu baris JSONL ke log fail."""
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("gagal tulis log: %s", e)


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


def validate_audio(audio_bytes: bytes) -> float:
    """Validasi audio dengan PyAV. Pulang durasi (saat). Raise ValueError jika rosak."""
    try:
        with av.open(io.BytesIO(audio_bytes)) as container:
            if not container.streams.audio:
                raise ValueError("tiada audio stream")
            stream = container.streams.audio[0]
            duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
            if duration <= 0:
                raise ValueError("durasi audio 0")
            return duration
    except av.error.FFmpegError as e:
        raise ValueError(f"fail audio rosak: {e}")


def find_silence_points(audio_bytes: bytes, min_silence_ms: int) -> list[float]:
    """Cari titik senyap (saat) menggunakan PyAV energy threshold ringkas.
    Pulang senarai timestamp (saat) di mana senyap berlaku.
    Nota: ini bukan Silero VAD, cuma energy-based approximation.
    """
    points = []
    try:
        with av.open(io.BytesIO(audio_bytes)) as container:
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
            window_frames = int(16000 * 0.05)  # 50ms window
            samples = []
            for frame in container.decode(stream):
                for rframe in resampler.resample(frame):
                    arr = rframe.to_ndarray()
                    samples.extend(arr.flatten().tolist())
            # Detect silence: window with RMS below threshold
            threshold = 200  # s16 amplitude
            for i in range(0, len(samples) - window_frames, window_frames):
                window = samples[i:i+window_frames]
                rms = (sum(s*s for s in window) / len(window)) ** 0.5
                if rms < threshold:
                    points.append(i / 16000.0)
    except Exception as e:
        logger.warning("find_silence_points gagal: %s", e)
    return points


def chunk_audio(audio_bytes: bytes, chunk_seconds: float, silence_points: list[float]) -> list[tuple[float, float]]:
    """Pecah audio kepada chunk (start, end) saat.
    Cari titik senyap terdekat dengan chunk_seconds (sliding ±20%).
    Jika tiada senyap, potong pada chunk_seconds.
    """
    duration = validate_audio(audio_bytes)
    chunks = []
    start = 0.0
    while start < duration:
        target_end = start + chunk_seconds
        if target_end >= duration:
            chunks.append((start, duration))
            break
        # Cari senyap terdekat dalam julat [target_end - 0.2*chunk, target_end + 0.2*chunk]
        lo = target_end - 0.2 * chunk_seconds
        hi = target_end + 0.2 * chunk_seconds
        candidates = [p for p in silence_points if lo <= p <= hi]
        if candidates:
            # Pilih senyap paling hampir dengan target_end
            best = min(candidates, key=lambda p: abs(p - target_end))
            end = best
        else:
            end = target_end
        chunks.append((start, end))
        start = end
    return chunks


def extract_chunk(audio_bytes: bytes, start: float, end: float) -> bytes:
    """Extract chunk dari audio bytes sebagai WAV 16kHz mono."""
    out = io.BytesIO()
    try:
        with av.open(io.BytesIO(audio_bytes)) as container:
            stream = container.streams.audio[0]
            container.seek(int(start / stream.time_base), stream=stream)
            resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
            with av.open(out, mode="w", format="wav") as out_container:
                out_stream = out_container.add_stream("pcm_s16le", rate=16000)
                out_stream.layout = "mono"
                for frame in container.decode(stream):
                    t = float(frame.pts * stream.time_base) if frame.pts else 0.0
                    if t < start:
                        continue
                    if t > end:
                        break
                    for rframe in resampler.resample(frame):
                        for packet in out_stream.encode(rframe):
                            out_container.mux(packet)
                for packet in out_stream.encode(None):
                    out_container.mux(packet)
    except Exception as e:
        logger.warning("extract_chunk gagal (%s-%s): %s", start, end, e)
        return b""
    return out.getvalue()


def apply_confidence_gate(segments: list) -> tuple[list, int]:
    """Buang segmen yang no_speech_prob > max DAN avg_logprob < min."""
    if not GATE_ENABLED:
        return segments, 0
    kept = []
    gated = 0
    for seg in segments:
        no_speech = getattr(seg, "no_speech_prob", 0.0)
        logprob = getattr(seg, "avg_logprob", 0.0)
        if no_speech > GATE_NOSPEECH_MAX and logprob < GATE_LOGPROB_MIN:
            gated += 1
            continue
        kept.append(seg)
    return kept, gated


def apply_repetition_guard(text: str) -> tuple[str, int]:
    """Potong frasa berulang >max_repeats berturut-turut."""
    if not REP_ENABLED:
        return text, 0
    # Cari frasa (1-5 perkataan) berulang >REP_MAX kali berturut
    pattern = re.compile(r'(\b(?:\w+\s+){0,4}\w+\b)(?:\s+\1){' + str(REP_MAX) + r',}', re.IGNORECASE)
    trimmed = 0
    def repl(m):
        nonlocal trimmed
        trimmed += 1
        return m.group(1)
    new_text = pattern.sub(repl, text)
    return new_text, trimmed


def transcribe_one(audio_bytes: bytes, language: str, task: str,
                   word_timestamps: bool, initial_prompt: str, vad_threshold: float):
    """Transkrip satu blob audio (satu chunk atau audio penuh). Pulang (text, info, segments)."""
    segs, info = model.transcribe(
        io.BytesIO(audio_bytes),
        beam_size=BEAM_SIZE,
        language=language if language else None,
        task=task,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=MIN_SILENCE_MS, threshold=vad_threshold),
        condition_on_previous_text=COND_PREV_TEXT,
        word_timestamps=word_timestamps,
        initial_prompt=initial_prompt if initial_prompt else None,
    )
    segs_list = list(segs)  # force evaluation
    text = " ".join(s.text.strip() for s in segs_list)
    return text, info, segs_list


# --- FastAPI ---
app = FastAPI(title="Whisper FAT 1.1.0", description="faster-whisper untuk CikguSayang")


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(None),
    task: str = Form("transcribe"),
    word_timestamps: bool = Query(None),
    initial_prompt: str = Query(None),
    vad_threshold: float = Query(None),
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
    chunks_info = []
    gated_total = 0
    rep_trimmed_total = 0
    audio_duration = 0.0

    # Baca fail
    try:
        audio_bytes = await file.read()
        size = len(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"gagal baca fail: {e}")

    if size == 0:
        raise HTTPException(status_code=400, detail="fail kosong")
    if size > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail=f"fail terlalu besar: {size} bytes (max {MAX_FILE_BYTES})")

    # Validasi audio (PyAV)
    try:
        audio_duration = validate_audio(audio_bytes)
    except ValueError as e:
        _write_log({"ts": datetime.now(timezone.utc).isoformat(), "version": VERSION,
                    "request_id": req_id, "file": file.filename, "size": size,
                    "status": "invalid_audio", "error_type": "Validation", "error_msg": str(e)})
        raise HTTPException(status_code=400, detail=str(e))

    # Parameter
    wt = WORD_TS_DEFAULT if word_timestamps is None else word_timestamps
    vt = VAD_THRESHOLD if vad_threshold is None else vad_threshold

    # Queue
    acquired = _queue_sem.locked()
    if acquired:
        got = False
        try:
            got = await asyncio.wait_for(_queue_sem.acquire(), timeout=0.001)
        except (asyncio.TimeoutError, Exception):
            got = False
        if not got:
            _write_log({"ts": datetime.now(timezone.utc).isoformat(), "version": VERSION,
                        "request_id": req_id, "file": file.filename, "size": size,
                        "status": "rejected_queue_full", "duration": round(time.time() - t0, 3)})
            raise HTTPException(status_code=429, detail="server busy, cuba lagi")
    else:
        await _queue_sem.acquire()

    try:
        await asyncio.wait_for(_gpu_lock.acquire(), timeout=REQUEST_TIMEOUT)
        try:
            # Tentukan chunk
            if CHUNK_ENABLED and audio_duration > CHUNK_SECONDS:
                sil = find_silence_points(audio_bytes, MIN_SILENCE_MS)
                chunk_ranges = chunk_audio(audio_bytes, CHUNK_SECONDS, sil)
                logger.info("Audio %.1fs → %d chunks", audio_duration, len(chunk_ranges))
                all_segs = []
                for idx, (cs, ce) in enumerate(chunk_ranges):
                    ct0 = time.time()
                    chunk_bytes = extract_chunk(audio_bytes, cs, ce)
                    if not chunk_bytes:
                        chunks_info.append({"idx": idx, "start": cs, "end": ce, "status": "skip"})
                        continue
                    ctext, cinfo, csegs = transcribe_one(
                        chunk_bytes, language, task, wt, initial_prompt or "", vt)
                    # Adjust timestamps by offset
                    for s in csegs:
                        s.start += cs
                        s.end += cs
                        if wt and hasattr(s, "words"):
                            for w in s.words:
                                w.start += cs
                                w.end += cs
                    all_segs.extend(csegs)
                    chunks_info.append({"idx": idx, "start": cs, "end": ce,
                                        "status": "ok", "duration": round(time.time() - ct0, 2),
                                        "text_len": len(ctext)})
                info = all_segs[0] if all_segs else None
                # Gabung text (selepas gate & repetition guard di bawah)
                text = " ".join(s.text.strip() for s in all_segs)
                info_obj = cinfo if 'cinfo' in dir() else None
            else:
                text, info, all_segs = transcribe_one(audio_bytes, language, task, wt, initial_prompt or "", vt)
                chunks_info.append({"idx": 0, "start": 0.0, "end": audio_duration, "status": "ok", "text_len": len(text)})

            # Confidence gate
            all_segs, gated_total = apply_confidence_gate(all_segs)
            # Repetition guard
            text, rep_trimmed_total = apply_repetition_guard(text)

            status = "ok"
        except asyncio.TimeoutError:
            error_type = "Timeout"
            error_msg = f"transkripsi melebihi {REQUEST_TIMEOUT}s"
            raise HTTPException(status_code=504, detail=error_msg)
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
            "audio_duration_s": round(audio_duration, 2),
            "chunks": chunks_info,
            "gated_segments": gated_total,
            "repetition_trimmed": rep_trimmed_total,
            "vram_before_mb": vram_before,
            "vram_after_mb": vram_after,
            "text_len": len(text),
            "detected_language": getattr(info, "language", None),
            "language_probability": round(getattr(info, "language_probability", 0), 3) if info else None,
            "error_type": error_type,
            "error_msg": error_msg,
        })

    resp = {
        "text": text,
        "language": getattr(info, "language", language or "unknown"),
        "language_probability": round(getattr(info, "language_probability", 0), 3) if info else 0.0,
    }
    if wt and all_segs:
        resp["segments"] = [
            {"id": i, "start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip()}
            for i, s in enumerate(all_segs)
        ]
    return resp


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": VERSION,
        "config": {
            "max_file_mb": MAX_FILE_BYTES // (1024*1024),
            "timeout_s": int(REQUEST_TIMEOUT),
            "chunk_seconds": int(CHUNK_SECONDS),
            "chunk_enabled": CHUNK_ENABLED,
            "queue_max": QUEUE_MAX,
        },
    }
