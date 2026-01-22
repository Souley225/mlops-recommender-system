#!/bin/bash
# Startup script for Hugging Face Spaces
# Runs FastAPI backend and Streamlit frontend

echo "====================================="
echo "MLOps Recommender System - Starting"
echo "====================================="

# Start API in background
echo "[1/3] Starting FastAPI API on port 8000..."
python -m uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait longer for API to be ready (model loading takes time)
echo "[2/3] Waiting for API to be ready..."
max_attempts=60
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "API is ready!"
        break
    fi
    attempt=$((attempt + 1))
    echo "  Attempt $attempt/$max_attempts - API not ready yet..."
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo "WARNING: API may not be fully ready, starting UI anyway..."
fi

# Start Streamlit UI (foreground)
echo "[3/3] Starting Streamlit UI on port 7860..."
echo "====================================="
python -m streamlit run src/ui/app.py \
    --server.port=7860 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
