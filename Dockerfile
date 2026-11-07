FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

# ML dependencies: fastembed (ONNX Runtime, CPU, no PyTorch) and google-genai.
COPY requirements-ml.txt ./
RUN pip install -r requirements-ml.txt

COPY pyproject.toml ./
COPY src ./src
# Schema files, so `python -m incident_copilot.db_migrate` works inside the container.
COPY db ./db
# Seed questions, for `python -m incident_copilot.evaluation.retrieval_smoke`.
COPY eval ./eval
RUN pip install --no-deps -e .

# /models is where the embedding model is cached. docker-compose mounts a named volume
# there; creating it here first makes the volume owned by appuser.
RUN useradd --create-home appuser && mkdir -p /models && chown appuser:appuser /models
ENV EMBEDDING_CACHE_DIR=/models
USER appuser

EXPOSE 8000
CMD ["uvicorn", "incident_copilot.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
