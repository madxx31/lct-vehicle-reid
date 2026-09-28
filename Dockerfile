# Vehicle re-ID inference image.
#
# Self-contained: the finetuned encoder, the C-RADIOv4 backbone it is built on (code, config, weights) and
# every pinned dependency live in the image, so the container never touches the network.
# torch is the CUDA 12.6 build (see pyproject.toml): the host needs an NVIDIA driver 525+.
# Build needs internet; running does not.
#
#   docker build -t lct-vehicle-reid .
#   docker run --rm --gpus all -v /path/to/data:/data:ro -v "$PWD/output":/output lct-vehicle-reid

# ---------------------------------------------------------------- builder ----
FROM python:3.14.7-slim-bookworm AS builder

# uv itself is pinned; it installs exactly what uv.lock resolves
COPY --from=ghcr.io/astral-sh/uv:0.10.0 /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=/usr/local/bin/python3.14 \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/opt/huggingface

WORKDIR /build
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# bake the backbone into the image's HF cache, at the revision config.yaml pins. The finetuned
# checkpoint overwrites its weights, but RadioEncoder still builds the model from its remote code and
# config. Loading it once, as inference does, caches both the files and transformers' compiled copy
# of that code, so the runtime needs neither the network nor a writable cache.
COPY train/encoder/ ./train/encoder/
RUN /opt/venv/bin/python -c "\
from omegaconf import OmegaConf; from train.encoder.model import RadioEncoder; \
c = OmegaConf.load('train/encoder/config.yaml').model; RadioEncoder(c.hf_repo, c.revision, c.image_size)" \
 && chmod -R a+rX /opt/huggingface

# ---------------------------------------------------------------- runtime ----
FROM python:3.14.7-slim-bookworm

# HOME/cache dirs point at /tmp so the image also runs under `--user $(id -u):$(id -g)`,
# which is how the output files come out owned by the caller instead of by root
ENV PATH=/opt/venv/bin:$PATH \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/huggingface /opt/huggingface

WORKDIR /app
# the finetuned encoder and the confidence model ship inside the image, at the paths
# inference.py reads
COPY models/vehicle_encoder.pt /opt/model/vehicle_encoder.pt
COPY models/confidence_model.cbm /opt/model/confidence_model.cbm
COPY inference/ /app/inference/
COPY train/encoder/ /app/train/encoder/

# bind mounts land here; created so a run without -v fails on the missing CSVs, not on mkdir
RUN mkdir -p /data /output

ENTRYPOINT ["python", "-m", "inference.inference"]
