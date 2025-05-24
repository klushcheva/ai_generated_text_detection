# Use PyTorch base image for GPU support
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Copy model files and source code
COPY app.py .
COPY worker.py .
COPY ai2 /app/model

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    MODEL_PATH="/app/model" \
    REDIS_HOST="redis" \
    WORKERS_PER_CORE=2 \
    MAX_WORKERS=4

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with gunicorn for production
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
