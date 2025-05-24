# Use official Python image with slim-buster
FROM python:3.9-slim-bookworm

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy bot source code
COPY telegram_bot.py .

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    TELEGRAM_TOKEN="" \
    REDIS_HOST="redis" \
    FASTAPI_URL="http://model:8000"

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=2)" || exit 1

CMD ["python", "telegram_bot.py"]