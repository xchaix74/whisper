"""FastAPI server untuk Whisper FAT 1.1.0 (CikguSayang).

Ciri:
  - Chunking automatik untuk audio >5 minit (fixed 5 min, sliding pada senyap)
  - Queue separation: fast (pendek) vs slow (panjang), GPU lock per chunk
  - Confidence gate (buang segmen low-confidence)
  - Repetition guard (potong frasa >3×)
  - Validasi PyAV (400 untuk fail rosak)
  - Config YAML (boleh ubah tanpa rebuild)
  - word_timestamps + initial_prompt + vad_threshold + max_duration parameter
  - Had saiz: 2GB, timeout server: 900s
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


# --- Config ---
def load_config() -> dict:
    default = {
        "server": {"max_file_mb": 2048, "request_timeout_s": 900, "queue_max": 32,
                   "max_duration_s": 0, "fast_gpu_timeout_s": 30},
        "chunking": {"enabled": True, "chunk_seconds": 300, "min_silence_ms": 500,
                     "vad_threshold": 0.5, "chunk_timeout_s": 180},
        "transcription": {"word_timestamps_default": False, "condition_on_previous_text": False, "beam_size": 5},
        "confidence_gate": {"enabled": True, "no_speech_prob_max": 0.6, "avg_logprob_min": -1.0},
        "repetition_guard": {"enabled": True, "max_repeats": 3},
    }
    path = os.environ.get("WHISPER_CONFIG", "/app/config.yaml")
    try:
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
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
MAX_DURATION_DEFAULT = float(CFG["server"].get("max_duration_s", 0))
FAST_GPU_TIMEOUT = float(CFG["server"].get("fast_gpu_timeout_s", 30))

CHUNK_ENABLED = bool(CFG["chunking"]["enabled"])
CHUNK_SECONDS = float(CFG["chunking"]["chunk_seconds"])
MIN_SILENCE_MS = int(CFG["chunking"]["min_silence_ms"])
VAD_THRESHOLD = float(CFG["chunking"]["vad_threshold"])
CHUNK_TIMEOUT = float(CFG["chunking"].get("chunk_timeout_s", 180))

WORD_TS_DEFAULT = bool(CFG["transcription"]["word_timestamps_default"])
COND_PREV_TEXT = bool(CFG["transcription"]["condition_on_previous_text"])
BEAM_SIZE = int(CFG["transcription"]["beam_size"])

GATE_ENABLED = bool(CFG["confidence_gate"]["enabled"])
GATE_NOSPEECH_MAX = float(CFG["confidence_gate"]["no_speech_prob_max"])
GATE_LOGPROB_MIN = float(CFG["confidence_gate"]["avg_logprob_min"])

REP_ENABLED = bool(CFG["repetition_guard"]["enabled"])
REP_MAX = int(CFG["repetition_guard"]["max_repeats"])

logger.info("Version: %s", VERSION)
logger.info("Config: max_file=%dMB, timeout=%ds, chunk=%ds, queue=%d, max_dur=%ds, chunk_timeout=%ds",
            MAX_FILE_BYTES // (1024 * 1024), int(REQUEST_TIMEOUT), int(CHUNK_SECONDS),
            QUEUE_MAX, int(MAX_DURATION_DEFAULT), int(CHUNK_TIMEOUT))

# --- Model ---
MODEL_ID = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")
logger.info("Loading model %s (compute_type=%s)", MODEL_ID, COMPUTE_TYPE)
model = WhisperModel(MODEL_ID, device="cuda", compute_type=COMPUTE_TYPE)
logger.info("Model loaded")

# --- Locks ---
_gpu_lock = asyncio.Lock()
_fast_sem = asyncio.Semaphore(max(1, QUEUE_MAX // 2))
_slow_sem = asyncio.Semaphore(max(1, QUEUE_MAX // 2))


# --- Helpers ---
def _write_log(entry: dict) -> None:
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("gagal tulis log: %s", e)


def _vram_mb() -> float:
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


def find_silence_points(audio_bytes: bytes) -> list[float]:
    """Cari titik senyap (saat) guna energy threshold ringkas."""
    points = []
    try:
        with av.open(io.BytesIO(audio_bytes)) as container:
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
            window_frames = int(16000 * 0.05)
            samples = []
            for frame in container.decode(stream):
                for rframe in resampler.resample(frame):
                    arr = rframe.to_ndarray()
                    samples.extend(arr.flatten().tolist())
            threshold = 200
            for i in range(0, len(samples) - window_frames, window_frames):
                window = samples[i:i + window_frames]
                rms = (sum(s * s for s in window) / len(window)) ** 0.5
                if rms < threshold:
                    points.append(i / 16000.0)
    except Exception as e:
        logger.warning("find_silence_points gagal: %s", e)
    return points


def chunk_audio(duration: float, silence_points: list[float]) -> list[tuple[float, float]]:
    """Pecah audio kepada chunk (start, end). Sliding pada senyap ±20%."""
    chunks = []
    start = 0.0
    while start < duration:
        target_end = start + CHUNK_SECONDS
        if target_end >= duration:
            chunks.append((start, duration))
            break
        lo = target_end - 0.2 * CHUNK_SECONDS
        hi = target_end + 0.2 * CHUNK_SECONDS
        candidates = [p for p in silence_points if lo <= p <= hi]
        end = min(candidates, key=lambda p: abs(p - target_end)) if candidates else target_end
        chunks.append((start, end))
        start = end
    return chunks


def extract_chunk(audio_bytes: bytes, start: float, end: float) -> bytes:
    """Extract chunk sebagai WAV 16kHz mono (dalam memori)."""
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
    if not REP_ENABLED:
        return text, 0
    pattern = re.compile(r'(\b(?:\w+\s+){0,4}\w+\b)(?:\s+\1){' + str(REP_MAX) + r',}', re.IGNORECASE)
    trimmed = 0
    def repl(m):
        nonlocal trimmed
        trimmed += 1
        return m.group(1)
    return pattern.sub(repl, text), trimmed


def transcribe_one(audio_bytes: bytes, language: str, task: str,
                   word_timestamps: bool, initial_prompt: str, vad_threshold: float):
    """Transkrip satu blob audio. Pulang (text, info, segments)."""
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
    segs_list = list(segs)
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
    max_duration: float = Query(None),
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
    all_segs = []
    sem = None

    # 1) Baca fail
    try:
        audio_bytes = await file.read()
        size = len(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"gagal baca fail: {e}")
    if size == 0:
        raise HTTPException(status_code=400, detail="fail kosong")
    if size > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail=f"fail terlalu besar: {size} bytes (max {MAX_FILE_BYTES})")

    # 2) Validasi audio
    try:
        audio_duration = validate_audio(audio_bytes)
    except ValueError as e:
        _write_log({"ts": datetime.now(timezone.utc).isoformat(), "version": VERSION,
                    "request_id": req_id, "file": file.filename, "size": size,
                    "status": "invalid_audio", "error_type": "Validation", "error_msg": str(e)})
        raise HTTPException(status_code=400, detail=str(e))

    # 3) Parameter
    wt = WORD_TS_DEFAULT if word_timestamps is None else word_timestamps
    vt = VAD_THRESHOLD if vad_threshold is None else vad_threshold
    md = MAX_DURATION_DEFAULT if max_duration is None else max_duration

    # 4) Semak max_duration (jika diberi): audio panjang ditolak awal
    if md > 0 and audio_duration > md:
        _write_log({"ts": datetime.now(timezone.utc).isoformat(), "version": VERSION,
                    "request_id": req_id, "file": file.filename, "size": size,
                    "status": "too_long", "audio_duration_s": round(audio_duration, 2),
                    "max_duration_s": md})
        raise HTTPException(status_code=413,
                            detail=f"audio too long: {audio_duration:.1f}s > max {md}s")

    # 5) Queue separation
    is_fast = (md > 0 and audio_duration <= md)
    sem = _fast_sem if is_fast else _slow_sem
    gpu_timeout = FAST_GPU_TIMEOUT if is_fast else REQUEST_TIMEOUT

    try:
        if is_fast:
            await asyncio.wait_for(sem.acquire(), timeout=FAST_GPU_TIMEOUT)
        else:
            await sem.acquire()
    except asyncio.TimeoutError:
        _write_log({"ts": datetime.now(timezone.utc).isoformat(), "version": VERSION,
                    "request_id": req_id, "file": file.filename, "size": size,
                    "status": "rejected_queue_full", "duration": round(time.time() - t0, 3)})
        raise HTTPException(status_code=429, detail="server busy, cuba lagi")

    try:
        # 6) Proses: chunking atau terus
        try:
            if CHUNK_ENABLED and audio_duration > CHUNK_SECONDS:
                sil = find_silence_points(audio_bytes)
                chunk_ranges = chunk_audio(audio_duration, sil)
                logger.info("Audio %.1fs → %d chunks", audio_duration, len(chunk_ranges))
                first_info = None
                for idx, (cs, ce) in enumerate(chunk_ranges):
                    ct0 = time.time()
                    chunk_bytes = extract_chunk(audio_bytes, cs, ce)
                    if not chunk_bytes:
                        chunks_info.append({"idx": idx, "start": cs, "end": ce, "status": "skip"})
                        continue
                    # GPU lock per chunk (release antara chunk)
                    acquired = False
                    try:
                        await asyncio.wait_for(_gpu_lock.acquire(), timeout=CHUNK_TIMEOUT)
                        acquired = True
                    except asyncio.TimeoutError:
                        chunks_info.append({"idx": idx, "start": cs, "end": ce,
                                            "status": "gpu_timeout", "duration": round(time.time() - ct0, 2)})
                        logger.warning("Chunk %d: GPU lock timeout", idx)
                        continue
                    try:
                        ctext, cinfo, csegs = await asyncio.wait_for(
                            asyncio.to_thread(transcribe_one, chunk_bytes, language, task, wt,
                                              initial_prompt or "", vt),
                            timeout=CHUNK_TIMEOUT,
                        )
                    except asyncio.TimeoutError:
                        chunks_info.append({"idx": idx, "start": cs, "end": ce,
                                            "status": "timeout", "duration": round(time.time() - ct0, 2)})
                        logger.warning("Chunk %d timeout (%.1fs)", idx, CHUNK_TIMEOUT)
                        continue
                    finally:
                        if acquired and _gpu_lock.locked():
                            _gpu_lock.release()
                    if first_info is None and cinfo is not None:
                        first_info = cinfo
                    for seg in csegs:
                        seg.start += cs
                        seg.end += cs
                        if wt and hasattr(seg, "words"):
                            for w in seg.words:
                                w.start += cs
                                w.end += cs
                    all_segs.extend(csegs)
                    chunks_info.append({"idx": idx, "start": cs, "end": ce,
                                        "status": "ok", "duration": round(time.time() - ct0, 2),
                                        "text_len": len(ctext)})
                info = first_info
                text = " ".join(seg.text.strip() for seg in all_segs)
            else:
                # Audio pendek: GPU lock untuk seluruh request
                try:
                    await asyncio.wait_for(_gpu_lock.acquire(), timeout=gpu_timeout)
                except asyncio.TimeoutError:
                    raise HTTPException(status_code=503, detail="GPU busy, cuba lagi")
                try:
                    text, info, all_segs = transcribe_one(
                        audio_bytes, language, task, wt, initial_prompt or "", vt)
                    chunks_info.append({"idx": 0, "start": 0.0, "end": audio_duration,
                                        "status": "ok", "text_len": len(text)})
                finally:
                    if _gpu_lock.locked():
                        _gpu_lock.release()

            # 7) Post-process
            all_segs, gated_total = apply_confidence_gate(all_segs)
            text, rep_trimmed_total = apply_repetition_guard(text)
            status = "ok"

        except HTTPException:
            raise
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            raise HTTPException(status_code=500, detail=error_msg)

    finally:
        sem.release()
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
            "queue": "fast" if is_fast else "slow",
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
            "max_file_mb": MAX_FILE_BYTES // (1024 * 1024),
            "timeout_s": int(REQUEST_TIMEOUT),
            "chunk_seconds": int(CHUNK_SECONDS),
            "chunk_enabled": CHUNK_ENABLED,
            "chunk_timeout_s": int(CHUNK_TIMEOUT),
            "queue_max": QUEUE_MAX,
            "max_duration_s": int(MAX_DURATION_DEFAULT),
            "fast_gpu_timeout_s": int(FAST_GPU_TIMEOUT),
        },
    }
