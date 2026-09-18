# IA stack: samples, measurements (CPU parts), GGL chain, IA fits and the paper figures.
# The GPU pair counts (cucount) need a CUDA base image and a GPU at run time; on the
# analysis cluster there is no container runtime and the recipes run on the host with
# IA_PYTHON pointing at a venv built from requirements.txt.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gfortran git libgsl-dev libfftw3-dev cmake pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV IA_PYTHON=python
