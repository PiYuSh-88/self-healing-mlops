# ── Stage 1: slim Python base ────────────────────────────────────────
FROM python:3.10-slim

# Prevents Python from writing .pyc files and enables real-time logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ── Install dependencies first (layer caching) ──────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code, model artifacts, and training data ────────
COPY app/ ./app/
COPY models/ ./models/
COPY data/ ./data/
COPY scripts/ ./scripts/

# ── Create logs directory ────────────────────────────────────────────
RUN mkdir -p /app/logs

# ── Expose port and run ─────────────────────────────────────────────
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

