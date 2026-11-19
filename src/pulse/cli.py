"""Command-line entry point. Every pipeline step runs from here.

    python -m pulse.cli data        generate synthetic data into data/raw
    python -m pulse.cli ingest      load data/raw through schema.yaml into DuckDB + quality report
    python -m pulse.cli pipeline    run every built step in order and print timings

    python -m pulse.cli features    build the snapshot table with features and churn label
    python -m pulse.cli train       churn model (split, three models, evaluation, fairness), then the
                                    time-to-churn model and the value-at-risk report; saves models/
    python -m pulse.cli score       worklist at the latest snapshot: probability, value, value at risk,
                                    urgency band, next best product or retention action; writes
                                    data/processed/worklist.parquet
"""
from __future__ import annotations

import argparse
import sys
import time


def _not_built(step: str):
    def run():
        raise SystemExit(f"step '{step}' is not built yet (see docs/PLAN.md for its phase)")
    return run


def step_data():
    from pulse import synth
    synth.main()


def step_ingest():
    from pulse import ingest
    ingest.main()


def step_features():
    from pulse.features import build
    build.main()


def step_train():
    from pulse.models import churn, survival
    churn.main()
    print(chr(10) + "=== time-to-churn and value at risk ===")
    survival.main()
    from pulse import crosssell
    print(chr(10) + "=== cross-sell: propensity models and trigger report ===")
    crosssell.main()


def step_score():
    from pulse import score
    score.main()


def step_explain():
    """Rebuild cards.json, the SHAP summary and the segment view from the saved worklist (score does this too)."""
    from pulse import explain
    explain.main()


STEPS = {
    "data": step_data,
    "ingest": step_ingest,
    "features": step_features,
    "train": step_train,
    "score": step_score,
    "explain": step_explain,
}
# the pipeline never regenerates data: on hackathon day the raw folder holds the bank's files
PIPELINE = ["ingest", "features", "train", "score"]


def run_pipeline():
    timings = []
    t_all = time.perf_counter()
    for name in PIPELINE:
        t = time.perf_counter()
        print(f"\n=== {name} ===")
        STEPS[name]()
        timings.append((name, time.perf_counter() - t))
    print("\nstep timings")
    for name, secs in timings:
        print(f"  {name:<10} {secs:7.1f}s")
    print(f"  {'total':<10} {time.perf_counter() - t_all:7.1f}s")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pulse", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=list(STEPS) + ["pipeline"])
    args = parser.parse_args(argv)
    t = time.perf_counter()
    if args.step == "pipeline":
        run_pipeline()
    else:
        STEPS[args.step]()
    print(f"[{args.step} finished in {time.perf_counter() - t:.1f}s]")


if __name__ == "__main__":
    sys.exit(main())
