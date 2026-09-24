# Hugging Face Spaces (Docker SDK) - free CPU tier: 2 vCPU, 16 GB RAM
FROM python:3.12-slim

# System libs used by Docling / OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run containers as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HOST=0.0.0.0 \
    PORT=7860
WORKDIR /home/user/app

# CPU-only torch first (avoids ~4 GB of CUDA wheels), then the rest.
# requirements.txt pins torch==2.14.0, which the +cpu build satisfies.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch==2.14.0 torchvision==0.29.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --user -r requirements.txt

# Bake models into the image so the first upload doesn't download them
RUN docling-tools models download layout tableformer code_formula \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY --chown=user . .
RUN mkdir -p docs images db static

EXPOSE 7860
CMD ["python", "app.py"]
