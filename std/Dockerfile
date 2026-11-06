# Whisper STD — varian ringkas untuk single-user / dev
# Enjin: faster-whisper 1.2.1 + CTranslate2 4.8.1
# Model: openai/whisper-large-v3-turbo (809M), dibina masuk
# Asas: CUDA 12.8 runtime (sama macam fat)
# Compute: float16 (sm_120 tidak support INT8 pada CTranslate2)
# Python: venv di /opt/venv (elak PEP 668)
# Tiada queue, tiada GPU lock, tiada log JSONL — ringkas.
# Untuk production multi-user, guna fat/.
# Tag: xchaix74/whisper:std

FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    DEBIAN_FRONTEND=noninteractive \
    WHISPER_VERSION=0.1.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install + pre-download + clean dalam SATU layer (corak sama macam fat)
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
    rm -rf /usr/share/doc/* /usr/share/man/* /usr/share/info/* 2>/dev/null || true; \
    rm -rf /root/.cache/* /tmp/* /var/tmp/* 2>/dev/null || true

ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/cuda/targets/x86_64-linux/lib

COPY server.py ./server.py

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
