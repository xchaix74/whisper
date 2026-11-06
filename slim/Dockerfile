# Whisper SLIM — varian ringan untuk CPU / GPU kecil (VRAM ~0.5GB)
# Enjin: whisper.cpp (C++), bukan PyTorch/CTranslate2
# Model: base-q5_1 (~57MB), bundled dalam image
# Device: CPU lalai (GPU pilihan via tag berasingan, tidak dibina di sini)
# Input: WAV 16kHz mono
# Endpoint: /v1/audio/transcriptions (via --inference-path, tiada wrapper Python)
# Saiz sasaran: ~150-250MB
# Tag: xchaix74/whisper:slim

# ============================================================
# Stage 1: Builder — compile whisper.cpp dari source
# ============================================================
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ARG WHISPER_CPP_TAG=v1.9.4

WORKDIR /src
RUN git clone --depth 1 --branch ${WHISPER_CPP_TAG} \
    https://github.com/ggml-org/whisper.cpp.git .

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF && \
    cmake --build build --config Release --target whisper-server -j"$(nproc)"

# ============================================================
# Stage 2: Runtime — ringkas, tiada compiler, tiada Python
# ============================================================
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    WHISPER_MODEL=/models/ggml-base-q5_1.bin \
    WHISPER_THREADS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/build/bin/whisper-server /usr/local/bin/whisper-server

# Muat turun model base-q5_1 (57MB) — wujud secara rasmi di HF
RUN mkdir -p /models && \
    curl -L --fail -o /models/ggml-base-q5_1.bin \
    https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base-q5_1.bin

EXPOSE 8000

# --inference-path menukar /inference kepada /v1/audio/transcriptions
CMD ["whisper-server", \
     "-m", "/models/ggml-base-q5_1.bin", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "-t", "4", \
     "--inference-path", "/v1/audio/transcriptions"]
