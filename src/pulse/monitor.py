"""Light monitoring: a run log for every scoring run and a feature drift report (Phase 6).

Run log: one JSON line per scoring run in data/processed/runs.jsonl with the timestamp, the
snapshot scored, row counts, the probability distribution, band counts and total value at risk.

Drift: population stability index (PSI) per feature, latest snapshot against the training block
of the Phase 2 split. PSI compares two histograms built on the same bins: ten quantile bins fixed
on the training data, so each holds about 10% of training rows. For each bin it takes
(latest share - training share) x ln(latest share / training share) and sums. Zero means the two
distributions agree; the usual reading is below 0.10 stable, 0.10 to 0.25 watch, above 0.25
investigate before trusting the scores. Protected attributes are not features and are not here.

    python -m pulse.cli monitor
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pulse.config import PROJECT_ROOT, load_settings, resolve
from pulse.features.build import feature_names
from pulse.io import read_parquet, write_parquet
from pulse.models import churn

RUNS_FILE = "runs.jsonl"
DRIFT_FILE = "drift.parquet"
BAND_ORDER = ["act this week", "act this month", "watch"]


# ----------------------------------------------------------------------------- PSI
def psi(train: np.ndarray, latest: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """Population stability index of `latest` against `train`, quantile bins fixed on `train`."""
    train = np.asarray(train, dtype=float)
    latest = np.asarray(latest, dtype=float)
    train, latest = train[~np.isnan(train)], latest[~np.isnan(latest)]
    if len(train) == 0 or len(latest) == 0:
        return float("nan")
    edges = np.unique(np.quantile(train, np.linspace(0, 1, bins + 1)[1:-1]))
    if len(edges) == 0:                                   # constant in training: one bin, no drift measurable
        return 0.0
    e = np.histogram(train, bins=np.concatenate(([-np.inf], edges, [np.inf])))[0] / len(train)
    a = np.histogram(latest, bins=np.concatenate(([-np.inf], edges, [np.inf])))[0] / len(latest)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_status(value: float, watch: float, alert: float) -> str:
    if np.isnan(value):
        return "n/a"
    return "investigate" if value >= alert else "watch" if value >= watch else "stable"


def drift_report(feats: pd.DataFrame, settings: dict, latest_date=None) -> pd.DataFrame:
    """PSI per feature: latest snapshot vs the training block (snapshot_date <= model.split.train_end)."""
    mcfg = settings["monitor"]
    train_end = pd.Timestamp(settings["model"]["split"]["train_end"])
    if feats["snapshot_date"].nunique() < 2:
        # fallback mode: one snapshot, nothing to compare against yet
        return pd.DataFrame(columns=["feature", "psi", "status", "train_mean", "latest_mean", "train_rows",
                                     "latest_rows", "train_to", "latest_snapshot"])
    latest_date = pd.Timestamp(latest_date) if latest_date is not None else feats["snapshot_date"].max()
    train = feats[feats["snapshot_date"] <= train_end]
    latest = feats[feats["snapshot_date"] == latest_date]
    if train.empty or latest.empty:
        raise ValueError("drift report needs rows in both the training block and the latest snapshot")
    rows = []
    for f in feature_names():
        if f not in feats.columns:
            continue
        v = psi(train[f].to_numpy(), latest[f].to_numpy(), int(mcfg["psi_bins"]))
        rows.append({"feature": f, "psi": round(v, 4),
                     "status": psi_status(v, float(mcfg["psi_watch"]), float(mcfg["psi_alert"])),
                     "train_mean": float(train[f].mean()), "latest_mean": float(latest[f].mean()),
                     "train_rows": len(train), "latest_rows": len(latest),
                     "train_to": str(train_end.date()), "latest_snapshot": str(latest_date.date())})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def drift_summary(drift: pd.DataFrame) -> dict | None:
    """Counts by status and the top features; None when there is no report (one snapshot only)."""
    if drift is None or len(drift) == 0:
        return None
    counts = drift["status"].value_counts().to_dict()
    top = drift.head(5)[["feature", "psi", "status"]].to_dict("records")
    return {"features": int(len(drift)), "stable": int(counts.get("stable", 0)), "watch": int(counts.get("watch", 0)),
            "investigate": int(counts.get("investigate", 0)), "max_psi": float(drift["psi"].max()),
            "top": top, "latest_snapshot": str(drift["latest_snapshot"].iloc[0]), "train_to": str(drift["train_to"].iloc[0])}


# ----------------------------------------------------------------------------- run log
def run_record(worklist: pd.DataFrame, feats: pd.DataFrame, drift: pd.DataFrame | None = None) -> dict:
    p = worklist["p_churn_90d"].to_numpy()
    q = np.quantile(p, [0.5, 0.9, 0.99])
    rec = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "snapshot_date": str(pd.Timestamp(worklist["snapshot_date"].iloc[0]).date()),
        "customers_scored": int(len(worklist)),
        "feature_rows": int(len(feats)),
        "feature_snapshots": int(feats["snapshot_date"].nunique()),
        "p_churn_mean": round(float(p.mean()), 5),
        "p_churn_p50": round(float(q[0]), 5), "p_churn_p90": round(float(q[1]), 5), "p_churn_p99": round(float(q[2]), 5),
        "share_above_10pct": round(float((p >= 0.10).mean()), 5),
        "bands": {b: int((worklist["urgency"] == b).sum()) for b in BAND_ORDER},
        "value_12m_usd": round(float(worklist["value_12m_usd"].sum()), 2),
        "value_at_risk_usd": round(float(worklist["value_at_risk_usd"].sum()), 2),
        "recommendations": worklist["rec_source"].value_counts().to_dict(),
    }
    s = drift_summary(drift)
    if s is not None:
        rec["drift"] = {k: s[k] for k in ("stable", "watch", "investigate", "max_psi")}
    return rec


def append_run(record: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def read_runs(path: Path, last: int | None = None) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    runs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return runs[-last:] if last else runs


# ----------------------------------------------------------------------------- CLI
def drift_markdown(drift: pd.DataFrame, settings: dict, n: int = 20) -> str:
    mcfg, s = settings["monitor"], drift_summary(drift)
    if s is None:
        return ("# Feature drift report" + chr(10) * 2 + "No report: the feature table has a single snapshot "
                "(fallback mode), so there is no training block to compare the latest month against. Drift "
                "becomes measurable once a second extract arrives." + chr(10))
    head = (f"# Feature drift report\n\nLatest snapshot {s['latest_snapshot']} ({int(drift['latest_rows'].iloc[0]):,} rows) "
            f"against the training block to {s['train_to']} ({int(drift['train_rows'].iloc[0]):,} rows), "
            f"{int(mcfg['psi_bins'])} quantile bins fixed on training.\n\n"
            f"PSI below {mcfg['psi_watch']} stable, {mcfg['psi_watch']} to {mcfg['psi_alert']} watch, above {mcfg['psi_alert']} "
            f"investigate. {s['stable']} stable, {s['watch']} watch, {s['investigate']} investigate; max PSI {s['max_psi']:.3f}.\n\n")
    show = drift.head(n)[["feature", "psi", "status", "train_mean", "latest_mean"]].copy()
    show["train_mean"] = show["train_mean"].round(3)
    show["latest_mean"] = show["latest_mean"].round(3)
    return head + f"**Top {len(show)} features by PSI**\n\n" + churn.to_markdown(show) + "\n"


def main(root: Path = PROJECT_ROOT, log=print) -> dict:
    settings = load_settings(root)
    t0 = time.perf_counter()
    processed = resolve(root, settings["data"]["processed_dir"])
    model_dir = resolve(root, churn.MODEL_DIR)
    feats = read_parquet(processed / "features.parquet")
    feats["snapshot_date"] = pd.to_datetime(feats["snapshot_date"])
    worklist = read_parquet(processed / "worklist.parquet")

    drift = drift_report(feats, settings, pd.Timestamp(worklist["snapshot_date"].iloc[0]))
    write_parquet(drift, processed / DRIFT_FILE)
    (model_dir / "drift_report.md").write_text(drift_markdown(drift, settings), encoding="utf-8")
    record = run_record(worklist, feats, drift)
    runs_path = resolve(root, settings["monitor"]["runs_log"])
    append_run(record, runs_path)

    s = drift_summary(drift)
    log(f"run log: appended to {runs_path.name} ({len(read_runs(runs_path))} runs); "
        f"{record['customers_scored']:,} customers at {record['snapshot_date']}, mean p {record['p_churn_mean']:.3%}, "
        f"p90 {record['p_churn_p90']:.3%}, value at risk {record['value_at_risk_usd']:,.0f} USD")
    if s is None:
        log("drift: no report (single snapshot); the run is logged")
    else:
        log(f"drift: {s['stable']} stable, {s['watch']} watch, {s['investigate']} investigate (max PSI {s['max_psi']:.3f})")
        log("top 10 by PSI\n" + drift.head(10)[["feature", "psi", "status", "train_mean", "latest_mean"]].round(3).to_string(index=False))
    log(f"monitored in {time.perf_counter() - t0:.1f}s")
    return {"drift": drift, "run": record}


if __name__ == "__main__":
    main()
