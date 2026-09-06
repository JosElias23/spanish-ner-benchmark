# Multi-stage so the runtime image carries no build toolchain.
FROM python:3.12-slim AS builder

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only torch. The GPU wheel is roughly 2 GB larger and buys nothing on the
# free tiers this is meant to run on; batch 64 still clears 100 documents per
# second on two vCPUs.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY serve/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


FROM python:3.12-slim

# A non-root user, because a container that can write to its own code is one
# remote-code-execution away from being a persistent foothold.
RUN useradd --create-home --uid 10001 appuser
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/appuser/.cache/huggingface

WORKDIR /app
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser serve/ ./serve/
COPY --chown=appuser:appuser configs/ ./configs/

USER appuser
EXPOSE 8000

# NER_MODEL_PATH accepts either a Hugging Face repo id or a mounted directory,
# so the same image serves a published model or a local checkpoint.
ENV NER_MODEL_PATH=""

# Readiness, not liveness: the container is not healthy until the weights are
# loaded, which takes several seconds.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,json,sys; \
r=json.load(urllib.request.urlopen('http://localhost:8000/ready',timeout=4)); \
sys.exit(0 if r.get('ready') else 1)"

CMD ["uvicorn", "serve.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
