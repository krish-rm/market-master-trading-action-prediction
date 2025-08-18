# ---- Builder ----
FROM python:3.10-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# Install build dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \
 && rm -rf /var/lib/apt/lists/*

# Copy and install requirements first (better caching)
COPY requirements.prod.txt .
RUN python -m pip install -U pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.prod.txt

# ---- Runtime ----
FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Copy only Python packages from builder (no build tools)
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code last (changes most frequently)
COPY src/ ./src/
COPY flows/ ./flows/

# FastAPI (Uvicorn)
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
