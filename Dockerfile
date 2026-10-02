# syntax=docker/dockerfile:1

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    XDG_CACHE_HOME=/cache

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first to avoid pulling CUDA libraries
RUN pip install --upgrade pip setuptools wheel && \
    pip install \
        torch \
        --index-url https://download.pytorch.org/whl/cpu

# Copy package metadata and source
COPY pyproject.toml README.md ./
COPY src ./src

# Install Perseus itself.
# --no-deps avoids reinstalling torch after the CPU build above;
# install the remaining dependencies explicitly.
RUN pip install \
        numpy \
        pandas \
        scikit-learn \
        pyarrow \
        ete3 \
        alive-progress \
        legacy-cgi && \
    pip install --no-deps .

RUN mkdir -p /data /cache

WORKDIR /data

ENTRYPOINT ["perseus"]
CMD ["--help"]