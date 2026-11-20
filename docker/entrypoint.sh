#!/bin/sh
# Container entrypoint. "api" serves the FastAPI app; "dashboard" serves Streamlit.
# On a clean checkout data/ is empty: the api service generates the synthetic data (if no raw
# files exist) and runs the full pipeline once before serving, so `docker compose up` works
# from nothing. On hackathon day put the bank's files in data/raw first and only the pipeline runs.
set -e
cd /app
case "$1" in
  api)
    if [ ! -f data/processed/worklist.parquet ]; then
      if [ -z "$(ls -A data/raw 2>/dev/null)" ]; then
        echo "no raw data found: generating synthetic data"
        python -m pulse.cli data
      fi
      echo "no worklist found: running the pipeline"
      python -m pulse.cli pipeline
    fi
    exec uvicorn api.main:app --host 0.0.0.0 --port 8000
    ;;
  dashboard)
    exec streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
    ;;
  *)
    exec "$@"
    ;;
esac
