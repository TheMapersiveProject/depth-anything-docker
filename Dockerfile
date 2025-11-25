# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf_cache \
    TRANSFORMERS_CACHE=/opt/hf_cache \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /app

# Minimal libs + tini
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 git curl ca-certificates tini \
    && rm -rf /var/lib/apt/lists/*

# Copy DA3 source code
COPY pyproject.toml README.md /app/
COPY src/ /app/src/

# Copy app scripts
COPY da3_batch.py entrypoint_da3.py /app/
COPY stitch_depth_equirect_parallel.py /app/

# Install dependencies
# Note: Installing torchvision separately to ensure it matches the torch version in the base image
RUN python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torchvision==0.18.1+cu121 && \
    python3 -m pip install --no-cache-dir \
      "numpy>=1.26,<2" \
      "transformers>=4.40" \
      "timm>=1.0" \
      "huggingface_hub>=0.20" \
      "safetensors>=0.4" \
      "einops>=0.6" \
      "opencv-python-headless>=4.8" \
      "Pillow>=9.5" \
      "boto3>=1.21" \
      "hf-transfer>=0.1.6"

# Install DA3 from source
RUN rm -rf /opt/hf_cache/* && \
    python3 -m pip install --no-cache-dir -e /app

# Writable paths + user setup
RUN useradd -m runner \
 && mkdir -p /source/OpenSfM/data /data /opt/hf_cache \
 && rm -rf /app/data && ln -s /data /app/data \
 && chown -R runner:runner /app /source /data /opt/hf_cache

USER runner

# Pre-download DA3 model (commented out - will download on first run)
# RUN python3 -c "from depth_anything_3.api import DepthAnything3; \
# DepthAnything3.from_pretrained('depth-anything/DA3METRIC-LARGE')"

VOLUME ["/opt/hf_cache", "/data"]

ENTRYPOINT ["/usr/bin/tini","--","python3","/app/entrypoint_da3.py"]
