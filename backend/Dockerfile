# Build context: repository root (see docker-compose.yml)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 \
    MODEL_DIR=/models/latest CORS_ORIGINS=http://localhost:8080
WORKDIR /srv

# tensorflow-cpu keeps the image ~1 GB smaller than the CUDA build; inference is CPU-friendly.
COPY ml/pyproject.toml ml/pyproject.toml
COPY ml/malariascope ml/malariascope
COPY README.md README.md
RUN pip install --no-cache-dir tensorflow-cpu \
 && pip install --no-cache-dir ./ml "fastapi>=0.115" "uvicorn[standard]>=0.30" "python-multipart>=0.0.9" "pydantic>=2.6"

COPY backend/app backend/app
WORKDIR /srv/backend
RUN useradd --create-home app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/health').status==200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
