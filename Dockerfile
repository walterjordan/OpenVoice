FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg git curl unzip build-essential pkg-config libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 HF_HOME=/app/.cache/hf TORCH_HOME=/app/.cache/torch NLTK_DATA=/app/.cache/nltk

RUN pip install torch==2.1.2+cpu torchaudio==2.1.2+cpu --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt setup.py README.md ./
COPY openvoice ./openvoice
RUN pip install -e . && pip install git+https://github.com/myshell-ai/MeloTTS.git \
    && pip install fastapi==0.110.0 uvicorn==0.29.0 httpx==0.27.0 soundfile \
    && python -m unidic download

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('myshell-ai/OpenVoiceV2', local_dir='/app/checkpoints_v2')" \
    && ls /app/checkpoints_v2/converter /app/checkpoints_v2/base_speakers/ses

COPY resources ./resources
COPY jab ./jab
RUN python -m jab.warmup

ENV PORT=8080
CMD exec uvicorn jab.server:app --host 0.0.0.0 --port ${PORT} --workers 1
