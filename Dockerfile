# Dockerfile optimise pour Render - API FastAPI
FROM python:3.11-slim

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Installer les dependances systeme
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dependances Python (utilise les wheels pre-compiles)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --only-binary :all: pyarrow || pip install pyarrow && \
    pip install -r requirements.txt

# Copier le code source
COPY src/ ./src/
COPY models/ ./models/
COPY data/processed/ ./data/processed/
COPY data/interim/ ./data/interim/
COPY configs/ ./configs/

# Port expose
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Commande de demarrage
CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
