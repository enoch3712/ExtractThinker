FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EXTRACT_THINKER_DOCUMENT_ROOT=/data \
    EXTRACT_THINKER_HOST=0.0.0.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends libmagic1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
WORKDIR /app
COPY pyproject.toml README.md requirements-server.txt ./
COPY extract_thinker ./extract_thinker
RUN pip install --no-cache-dir . -r requirements-server.txt \
    && mkdir /data && chown app:app /data
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"
CMD ["python", "-m", "extract_thinker.mcp_server"]
