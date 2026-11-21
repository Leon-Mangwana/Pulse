# Docker runbook

One image serves both services; `docker-compose.yml` picks the command.

| service | port | command inside the container |
| --- | --- | --- |
| api | 8000 | `uvicorn api.main:app` (interactive docs at http://localhost:8000/docs) |
| dashboard | 8501 | `streamlit run dashboard/app.py` |

Both mount `./data` and `./models` from the host, so a pipeline run on the host and one in the
container write the same files.

## Everyday use (online)

```powershell
docker compose build          # once, and after requirements.txt changes (about 3 minutes online)
docker compose up             # api on :8000, dashboard on :8501; Ctrl+C stops both
docker compose down           # remove the containers
```

On a clean checkout `data/` is empty. The api container notices there is no worklist, generates
the synthetic data if `data/raw` is empty (80 s), runs the whole pipeline (150 s) and only then
starts serving; the dashboard waits for the api's health check. On hackathon day put the bank's
files in `data/raw` first, so only the pipeline runs.

To rerun the pipeline against new raw files without rebuilding anything:

```powershell
docker compose run --rm api python -m pulse.cli pipeline
Invoke-RestMethod -Method Post http://localhost:8000/reload     # the running api picks up the new outputs
```

Or run it on the host (`.\.venv\Scripts\python.exe -m pulse.cli pipeline`) and call `/reload`.

## The dashboard (Phase 7)

`docker compose up` serves it at http://localhost:8501. Four pages in demo order: Worklist,
Customer card, Insights, ROI simulator. It only talks to the api service (`PULSE_API_URL`),
caches every response for five minutes and shows a waiting page until the api reports a loaded
worklist. After a new pipeline run call `/reload` on the api and press "Refresh data" in the
sidebar (or wait five minutes).

### Editing without rebuilding

`docker-compose.dev.yml` mounts `src/`, `api/`, `dashboard/`, `config/`, `tests/` and
`.streamlit/` into the containers, so a saved file is live on the next request (Streamlit
re-runs on its own):

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

Rebuild (`docker compose build --build-arg PIP_FLAGS="--no-index"`, 20 s with the cached
package layer) before relying on the plain `docker compose up` again.

### Tests that need the containers

Streamlit (and on the build machine scikit-learn, see below) cannot import from the local venv,
so these run inside the containers:

```powershell
# whole suite (dashboard page tests skip here because the api is not reachable from the api container)
docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm --no-deps api python -m pytest -q
# the four Streamlit pages against the live api (starts the api if needed)
docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm dashboard python -m pytest -q tests/test_dashboard_app.py
```

The page tests render each page twice with Streamlit's AppTest and assert the second, cached
render takes under two seconds, that no raw feature name appears, and that the ROI simulator
gives zeros at zero contacts and no uplift when everyone is contacted.

### Screenshots at projector size

`scripts/screenshot_pages.py` drives headless Edge through the DevTools protocol and saves
1280x720 PNGs of the four pages (the `--screenshot` flag alone captures Streamlit's loading
skeleton, because it does not wait for the websocket):

```powershell
.\.venv\Scripts\python.exe scripts\screenshot_pages.py docs\screens http://localhost:8501
```

## Building with no internet

Docker needs the base image and the Python wheels. Either path below is done once while online.

### Path A (simplest): save the built image to a file

```powershell
docker compose build
docker save pulse:latest -o pulse-image.tar          # about 1.8 GB
# at the venue, on any machine with Docker:
docker load -i pulse-image.tar
docker compose up --no-build
```

Copy `pulse-image.tar` to a USB stick with the repository. `docker compose up --no-build` uses the
loaded image and never touches the network.

### Path B: cache the wheels and rebuild offline

```powershell
docker pull python:3.11-slim
.\.venv\Scripts\python.exe -m pip download -r requirements.txt -d wheels `
    --platform manylinux_2_28_x86_64 --platform manylinux_2_17_x86_64 `
    --platform manylinux2014_x86_64 --platform manylinux1_x86_64 `
    --python-version 3.11 --implementation cp --only-binary=:all: --retries 30 --timeout 180
docker compose build --build-arg PIP_FLAGS="--no-index"
```

The four `--platform` tags cover every Linux wheel the pins use (pillow and contourpy ship
`manylinux_2_28` wheels, the rest `manylinux2014`/`2_17`). The download is about 200 MB and
resumes file by file: if it fails on a slow link, run the same command again and it skips the
wheels it already has. With `--no-index` the build never has to resolve versions, so it is also
the fastest and most predictable build: on 19 Nov an online build on a 10 kB/s connection spent
15 minutes backtracking through matplotlib releases and then timed out; the wheel path avoids both.

The Dockerfile always copies `wheels/` into the image and passes `--find-links /wheels` to pip,
so with `--no-index` pip never reaches the network. `wheels/*.whl` is git-ignored; keep the
folder on the USB stick next to the repository. The one apt package (`libgomp1`) is fetched
during the build, so Path B also needs the base layer cached: run `docker compose build` once
online before relying on it, or use Path A.

## Checks

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod "http://localhost:8000/worklist?limit=5"
Invoke-RestMethod http://localhost:8000/customers/<customer_ref>/card
Invoke-RestMethod http://localhost:8000/segments?dimension=segment
Invoke-RestMethod http://localhost:8000/metrics
Invoke-RestMethod -Method Post -ContentType application/json -Body '{"top_n":500}' http://localhost:8000/simulate
```

`customer_ref` is a pseudonymised reference from `/worklist` (for example `P57DB131092`); raw
customer ids are never loaded by the api.

## Memory

The api container needs about 2.5 GB of memory free inside Docker's Linux VM while it runs the
pipeline (measured on 20 Nov 2026: data generation peaks at 1.4 GB, the pipeline at 2.1 GB;
serving afterwards needs well under 0.5 GB). Docker Desktop's default VM is 4 GB, which is
enough on a laptop running nothing else. If other containers are using the VM, the pipeline step
is killed with a bare `Killed` in the api logs and the container exits. Either stop the other
containers for the duration, raise the VM size (Docker Desktop > Settings > Resources, or
`memory=6GB` in `%USERPROFILE%\.wslconfig` followed by `wsl --shutdown`), or run the pipeline on
the host (`.\.venv\Scripts\python.exe -m pulse.cli pipeline`) and let the container only serve.

## Troubleshooting
- `Killed` in the api logs during data generation or the pipeline: out of memory, see above.

- `docker: error during connect`: Docker Desktop is not running. Start it and wait for the whale icon.
- The api logs show the pipeline running for a few minutes on first start: expected on a clean checkout.
- Port 8000 or 8501 in use: change the left-hand number in `docker-compose.yml` (`"8080:8000"`).
- Streamlit cannot run from the local `.venv` on the build machine (its pyarrow dependency is
  blocked by Windows App Control); use the container.
- `DLL load failed ... An Application Control policy has blocked this file` for a scikit-learn
  file: Smart App Control's cloud reputation for that file changed (it blocked `_sgd_fast` on
  20 Nov after working for three days). Nothing in the repo can fix it; run that step in the
  container with the dev override above.
