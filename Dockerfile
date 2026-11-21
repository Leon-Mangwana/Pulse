# Pulse: one image for the api and dashboard services (docker-compose.yml picks the command).
# Online build:   docker compose build
# Offline build:  see docs/runbook-docker.md (wheels/ folder + PIP_FLAGS="--no-index", or docker load)
FROM python:3.11-slim

# libgomp1 is the one system library LightGBM needs
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ARG PIP_FLAGS=""
# slow links: wait longer per read and retry before giving up (no effect with --no-index)
ENV PIP_DEFAULT_TIMEOUT=180 PIP_RETRIES=10
COPY requirements.txt pyproject.toml ./
COPY wheels/ /wheels/
RUN pip install --no-cache-dir --find-links /wheels ${PIP_FLAGS} -r requirements.txt

COPY src ./src
COPY config ./config
COPY api ./api
COPY dashboard ./dashboard
COPY .streamlit ./.streamlit
COPY docker/entrypoint.sh ./docker/entrypoint.sh
RUN pip install --no-cache-dir --no-deps --no-build-isolation ${PIP_FLAGS} -e . \
    && chmod +x docker/entrypoint.sh && mkdir -p data/raw data/processed models

ENV PULSE_ROOT=/app PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
EXPOSE 8000 8501
ENTRYPOINT ["./docker/entrypoint.sh"]
CMD ["api"]
