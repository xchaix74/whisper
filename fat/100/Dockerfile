# Whisper FAT — untuk CS (CikguSayang)
# Enjin: faster-whisper 1.2.1 + CTranslate2 4.8.1
# Model: openai/whisper-large-v3-turbo (809M), dibina masuk
# Asas: CUDA 12.8 runtime (lebih kecil dari PyTorch base)
# Compute: float16 (sm_120 tidak support INT8 pada CTranslate2)
# Python: venv di /opt/venv (elak PEP 668)
# Default language: None (auto-detect). CS hantar "en" untuk MVP English.
# VAD: aktif (kurang halusinasi pada senyap)
# Halusinasi: condition_on_previous_text=False (elak gelung)
# Multi-user: queue depth 32, GPU lock, timeout 120s, max 600s audio
# Log: /app/logs/whisper.jsonl (JSONL)
# Version: dari env WHISPER_VERSION
# Tag: xchaix74/whisper:fat

FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    DEBIAN_FRONTEND=noninteractive \
    WHISPER_VERSION=1.0.0 \
    WHISPER_QUEUE_MAX=32 \
    WHISPER_REQUEST_TIMEOUT=120 \
    WHISPER_MAX_AUDIO_SECONDS=600 \
    WHISPER_MAX_FILE_BYTES=104857600 \
    WHISPER_LOG_DIR=/app/logs

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install + pre-download + clean SEMUA dalam SATU layer.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        faster-whisper==1.2.1 \
        ctranslate2==4.8.1 \
        "av>=11,<19" \
        nvidia-cudnn-cu12 \
        fastapi==0.115.6 \
        uvicorn==0.34.0 \
        python-multipart==0.0.20 && \
    python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3-turbo', device='cpu', compute_type='float32')" && \
    find / -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true; \
    find / -name '*.pyc' -delete 2>/dev/null || true; \
    find / -name '*.pyo' -delete 2>/dev/null || true; \
    find /opt/venv/lib/python3.12/site-packages -type d -name 'tests' -exec rm -rf {} + 2>/dev/null || true; \
    find /opt/venv/lib/python3.12/site-packages -type d -name 'test' -exec rm -rf {} + 2>/dev/null || true; \
    rm -rf /usr/share/doc/* /usr/share/man/* /usr/share/info/* 2>/dev/null || true; \
    /opt/venv/bin/pip uninstall -y pip setuptools wheel 2>/dev/null || true; \
    rm -rf /root/.cache/* /tmp/* /var/tmp/* 2>/dev/null || true

ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/cuda/targets/x86_64-linux/lib

COPY server.py ./server.py

# Volume untuk log
VOLUME ["/app/logs"]

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
