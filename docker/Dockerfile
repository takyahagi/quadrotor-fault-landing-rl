FROM python:3.10-slim

# System libs needed by Genesis (mesh processing, offscreen rendering, video export).
# No GPU here: CPU backend only. See docker/Dockerfile.gpu for the training machine.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git \
    libgl1 libglx-mesa0 libegl1 libgles2 libosmesa6 \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY docker/requirements.txt /tmp/requirements.txt
# CPU-only torch wheel (no CUDA) — this image is for logic/verification, not training.
ENV PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /tmp/requirements.txt

CMD ["bash"]
