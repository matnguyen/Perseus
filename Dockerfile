# syntax=docker/dockerfile:1

FROM python:3.12-slim

# Basic runtime hygiene
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

WORKDIR /app

# System packages
# build-essential/gcc are useful because some Python deps may need compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency metadata first for better layer caching
COPY pyproject.toml ./
COPY README.md ./
COPY src ./src

# Install Perseus
# If pyproject already pins torch, this will honor that.
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install .

# Optional: pre-create a writable cache area for NCBI / ETE data
RUN mkdir -p /data /cache

ENV XDG_CACHE_HOME=/cache

# Default working location for mounted input/output
WORKDIR /data

# Default entrypoint to the CLI
ENTRYPOINT ["perseus"]
CMD ["--help"]