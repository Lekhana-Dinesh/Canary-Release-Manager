# Intentionally duplicated in the repository root and server/ so both build
# targets produce the same Hugging Face Space runtime image.
FROM python:3.11-slim

ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/canary_release_env

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.getenv('PORT', '7860')}/health\")"]

EXPOSE ${PORT}

CMD ["python", "-m", "canary_release_env.server.app"]
