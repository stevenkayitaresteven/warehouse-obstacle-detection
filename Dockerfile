# Warehouse robot obstacle-avoidance demo (Gradio app) in a container.
#
#   docker compose up --build            -> http://localhost:7860
#
# or without compose:
#   docker build -t warehouse-robot-demo .
#   docker run --rm -p 7860:7860 warehouse-robot-demo
#
# Build with the real-image detection tab (YOLO11n + Depth Anything V2,
# CPU-only PyTorch, ~2 GB larger):
#   docker build --build-arg WITH_PERCEPTION=1 -t warehouse-robot-demo:full .

ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

ARG WITH_PERCEPTION=0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRADIO_ANALYTICS_ENABLED=False

WORKDIR /app

# Dependencies first, so code changes don't reinstall packages
COPY requirements.txt requirements-perception.txt ./
RUN pip install -r requirements.txt \
 && if [ "$WITH_PERCEPTION" = "1" ]; then \
        apt-get update \
        && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/* \
        && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
        && pip install -r requirements-perception.txt ; \
    fi

# Run as an unprivileged user
RUN useradd --create-home --uid 1000 app
COPY --chown=app:app . .
USER app

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/', timeout=4)" || exit 1

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860", "--no-browser"]
