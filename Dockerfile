# Stage 1: Build
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt \
    fastapi uvicorn[standard] sqlalchemy click httpx

COPY src/ src/
COPY configs/ configs/
COPY dashboard/ dashboard/
COPY scripts/ scripts/

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

RUN groupadd -r sentinel && useradd -r -g sentinel sentinel

COPY --from=builder /install /usr/local
COPY --from=builder /app .

COPY .env.example .env.example

RUN mkdir -p data/raw/esa_kelvins data/processed data/cache models_saved proofs \
    && chown -R sentinel:sentinel /app

USER sentinel

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "-m", "uvicorn", "orbital_sentinel.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
