FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY agent ./agent

# CPU-only torch: the embedding server does not need CUDA.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
        "sentence-transformers>=3.0" \
        "numpy>=1.26" \
        "pyyaml>=6.0" \
        "python-dotenv>=1.0"

EXPOSE 8765
CMD ["python", "-m", "agent.embed_server", "--host", "0.0.0.0", "--port", "8765"]
