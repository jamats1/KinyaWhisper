# syntax=docker/dockerfile:1

FROM python:3.10-slim AS runtime

# Environment settings to improve Python behaviour and disable bytecode generation
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=5000

# Install system dependencies required by torchaudio/ffmpeg and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Install Python dependencies first for caching benefits
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    grep -Ev "pyobjc|^sklearn==" requirements.txt > requirements-filtered.txt && \
    pip install --no-cache-dir -r requirements-filtered.txt

# Copy rest of application source
COPY . .

# Ensure runtime directory exists for uploaded/generated audio
RUN mkdir -p /app/audio

# Expose API port for Coolify (actual port read from $PORT)
EXPOSE 5000

# Default command
CMD ["python", "app.py"] 