FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY retail_agent ./retail_agent
RUN pip install ".[postgres]"

COPY scripts ./scripts
COPY data/docs ./data/docs
COPY data/eval ./data/eval

RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "retail_agent.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
