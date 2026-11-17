# Convenience targets. On Windows without make, run the python command shown
# next to each target instead, e.g.  python -m pulse.cli data
PY ?= .venv/Scripts/python.exe

.PHONY: data ingest features train score pipeline test

data:        ## generate synthetic bank data into data/raw
	$(PY) -m pulse.cli data

ingest:      ## load data/raw through config/schema.yaml into DuckDB and run data-quality checks
	$(PY) -m pulse.cli ingest

features:    ## (phase 1)
	$(PY) -m pulse.cli features

train:       ## (phase 2)
	$(PY) -m pulse.cli train

score:       ## (phase 3+)
	$(PY) -m pulse.cli score

pipeline:    ## run every built step in order and print timings
	$(PY) -m pulse.cli pipeline

test:
	$(PY) -m pytest
