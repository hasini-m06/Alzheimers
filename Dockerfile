FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# HF Spaces exposes port 7860
EXPOSE 7860

# Env vars (override at runtime)
ENV API_BASE_URL="https://api-inference.huggingface.co/v1"
ENV MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
ENV HF_TOKEN=""

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
