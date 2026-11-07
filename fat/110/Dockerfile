# Whisper FAT 1.1.0 — untuk CS (CikguSayang)
# Enjin: faster-whisper 1.2.1 + CTranslate2 4.8.1
# Model: openai/whisper-large-v3-turbo (809M), dibina masuk
# Asas: CUDA 12.8 runtime (sama macam 1.0.0)
# Compute: float16 (sm_120 tidak support INT8 pada CTranslate2)
# Python: venv di /opt/venv (elak PEP 668)
# Ciri baharu 1.1.0:
#   - Chunking automatik untuk audio >5 minit
#   - Confidence gate (buang segmen low-confidence)
#   - Repetition guard (potong frasa >3×)
#   - Validasi PyAV (400 untuk fail rosak)
#   - Config YAML (boleh ubah tanpa rebuild)
#   - word_timestamps + initial_prompt parameter
#   - Had saiz: 100MB → 2GB
#   - Timeout: 120s → 900s
# Tag: xchaix74/whisper:fat-1.1.0

FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    DEBIAN_FRONTEND=noninteractive \
    WHISPER_VERSION=1.1.0 \
    WHISPER_CONFIG=/app/config.yaml \
    WHISPER_LOG_DIR=/app/logs

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install + pre-download + clean dalam SATU layer (sama macam 1.0.0, tambah pyyaml)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        faster-whisper==1.2.1 \
        ctranslate2==4.8.1 \
        "av>=11,<19" \
        nvidia-cudnn-cu12 \
        fastapi==0.115.6 \
        uvicorn==0.34.0 \
        python-multipart==0.0.20 \
        pyyaml==6.0.2 && \
    python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3-turbo', device='cpu', compute_type='float32')" && \
    find / -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true; \
    find / -name '*.pyc' -delete 2>/dev/null || true; \
    find / -name '*.pyo' -delete 2>/dev/null || true; \
    rm -rf /usr/share/doc/* /usr/share/man/* /usr/share/info/* 2>/dev/null || true; \
    rm -rf /root/.cache/* /tmp/* /var/tmp/* 2>/dev/null || true

ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/cuda/targets/x86_64-linux/lib

# Salin server.py dan config.yaml (default)
COPY server.py ./server.py
COPY config.yaml ./config.yaml

# Volume untuk log dan config override
VOLUME ["/app/logs"]

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
