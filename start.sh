#!/bin/bash
# Startup script for Hugging Face Spaces
# Runs FastAPI backend and Streamlit frontend

set -e

echo "Starting MLOps Recommender System..."

# Start API in background
echo "Starting FastAPI API on port 8000..."
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait for API to be ready
echo "Waiting for API to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "API is ready!"
        break
    fi
    sleep 1
done

# Start Streamlit UI (foreground)
echo "Starting Streamlit UI on port 7860..."
exec streamlit run src/ui/app.py \
    --server.port=7860 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
