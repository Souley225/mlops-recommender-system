# Dockerfile for Hugging Face Spaces
# Combined API + UI deployment - single container approach
# Port 7860 exposes Streamlit UI, API runs internally on port 8000

FROM python:3.11-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    API_URL=http://localhost:8000 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --only-binary :all: pyarrow || pip install pyarrow && \
    pip install -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY models/ ./models/
COPY data/processed/ ./data/processed/
COPY data/interim/ ./data/interim/
COPY configs/ ./configs/

# Copy startup script
COPY start.sh ./start.sh
RUN chmod +x ./start.sh

# Hugging Face Spaces uses port 7860
EXPOSE 7860

# Health check for Streamlit
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

# Start both API and UI
CMD ["./start.sh"]
