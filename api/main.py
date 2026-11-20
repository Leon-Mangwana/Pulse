"""Pulse scoring API: serves the pipeline outputs, never raw data (Phase 6).

    uvicorn api.main:app --reload            (from the project root)

Everything comes from data/processed and models/, which the pipeline writes. Customer references
are the pseudonymised ones the worklist already carries (rule 7); raw ids are not loaded.
PULSE_ROOT points the app at another project root (the Docker image sets it to /app).
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import json
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from pulse.config import PROJECT_ROOT, load_settings, resolve
from pulse.io import read_parquet
from pulse.models import churn
from pulse.monitor import DRIFT_FILE, drift_summary, read_runs

WORKLIST_FIELDS = ["customer_ref", "rank", "segment", "branch", "p_churn_90d", "p_churn_30d", "value_12m_usd",
                   "value_at_risk_usd", "urgency", "recommendation", "rec_source", "rec_reason"]


# ----------------------------------------------------------------------------- response models
class Health(BaseModel):
    status: str
    snapshot_date: str | None
    customers: int
    cards: int
    loaded_at: str
    root: str


class WorklistRow(BaseModel):
    customer_ref: str
    rank: int
    segment: str
    branch: str
    p_churn_90d: float
    p_churn_30d: float
    value_12m_usd: float
    value_at_risk_usd: float
    urgency: str
    recommendation: str
    rec_source: str
    rec_reason: str


class WorklistResponse(BaseModel):
    snapshot_date: str
    total: int = Field(description="customers matching the filters")
    returned: int
    total_value_at_risk_usd: float = Field(description="over the customers matching the filters")
    rows: list[WorklistRow]


class Reason(BaseModel):
    feature: str
    family: str
    family_label: str
    contribution: float
    text: str


class NextBestProduct(BaseModel):
    product: str
    reason: str
    source: str
    score: float


class Card(BaseModel):
    customer_ref: str
    rank: int
    segment: str
    branch: str
    snapshot_date: str
    p_churn_90d: float
    urgency: str
    value_12m_usd: float
    value_at_risk_usd: float
    reasons: list[Reason]
    action_family: str
    action: str
    call_script: str
    next_best_product: NextBestProduct


class SegmentRow(BaseModel):
    dimension: str
    group: str
    customers: int
    churn_rate_pct: float
    expected_churners: float
    value_12m_usd: float
    value_at_risk_usd: float
    share_of_var_pct: float
    var_per_customer_usd: float


class SegmentsResponse(BaseModel):
    snapshot_date: str
    rows: list[SegmentRow]


class DriftSummary(BaseModel):
    features: int
    stable: int
    watch: int
    investigate: int
    max_psi: float
    top: list[dict[str, Any]]
    latest_snapshot: str
    train_to: str


class Metrics(BaseModel):
    run_date: str
    seed: int
    n_features: int
    split: list[dict[str, Any]]
    model_comparison_test: list[dict[str, Any]] = Field(description="rules, logistic and LightGBM on the test block")
    lightgbm_test: dict[str, Any]
    calibration: dict[str, Any]
    warnings: dict[str, Any]
    fairness_test: list[dict[str, Any]]
    survival: dict[str, Any]
    drift: DriftSummary | None
    last_run: dict[str, Any] | None
    runs_logged: int


class SimulateRequest(BaseModel):
    top_n: int = Field(500, ge=1, description="customers contacted, from the top of the value-at-risk ranking")
    cost_per_contact: float = Field(2.0, ge=0, description="USD per customer contacted")
    incentive_cost: float = Field(10.0, ge=0, description="USD per customer actually saved")
    save_rate: float = Field(0.30, ge=0, le=1, description="share of would-be churners a contact retains")


class Outcome(BaseModel):
    contacted: int
    expected_churners_contacted: float
    expected_saves: float
    revenue_protected_usd: float
    contact_cost_usd: float
    incentive_cost_usd: float
    total_cost_usd: float
    net_return_usd: float
    return_per_usd: float | None = Field(description="revenue protected per USD spent; null when nothing is spent")


class SimulateResponse(BaseModel):
    snapshot_date: str
    assumptions: SimulateRequest
    targeted: Outcome = Field(description="top_n by value at risk")
    random: Outcome = Field(description="top_n customers picked at random (expected value)")
    uplift_net_usd: float = Field(description="targeted net return minus random net return")


# ----------------------------------------------------------------------------- data store
class Store:
    """Pipeline outputs loaded once; reload() picks up a new pipeline run."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.settings = load_settings(self.root)
        self.reload()

    def reload(self) -> None:
        processed = resolve(self.root, self.settings["data"]["processed_dir"])
        model_dir = resolve(self.root, churn.MODEL_DIR)
        self.worklist = read_parquet(processed / "worklist.parquet").sort_values("rank").reset_index(drop=True)
        self.snapshot_date = str(pd.Timestamp(self.worklist["snapshot_date"].iloc[0]).date())
        cards = json.loads((processed / "cards.json").read_text(encoding="utf-8"))
        self.cards = {c["customer_ref"]: c for c in cards}
        self.segments = read_parquet(processed / "segments.parquet")
        self.metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
        sm = model_dir / "survival_metrics.json"
        self.survival = json.loads(sm.read_text(encoding="utf-8")) if sm.exists() else {}
        drift_path = processed / DRIFT_FILE
        self.drift = drift_summary(read_parquet(drift_path)) if drift_path.exists() else None
        self.runs = read_runs(resolve(self.root, self.settings["monitor"]["runs_log"]))
        self.loaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")


def simulate(worklist: pd.DataFrame, req: SimulateRequest) -> SimulateResponse:
    """Expected outcome of contacting the top_n by value at risk versus top_n at random.

    Per contacted customer: expected saves = p_churn_90d x save_rate; revenue protected = saves x
    12-month value; cost = cost_per_contact + saves x incentive_cost. Random outreach is the expected
    value of a random sample of the same size, i.e. the worklist averages times top_n.
    """
    n = min(int(req.top_n), len(worklist))
    p_all, v_all = worklist["p_churn_90d"].to_numpy(), worklist["value_12m_usd"].to_numpy()

    def outcome(p: np.ndarray, v: np.ndarray, contacted: int) -> Outcome:
        churners = float(p.sum())
        saves = churners * req.save_rate
        protected = float((p * v).sum()) * req.save_rate
        contact_cost = contacted * req.cost_per_contact
        incentive = saves * req.incentive_cost
        total = contact_cost + incentive
        return Outcome(contacted=contacted, expected_churners_contacted=round(churners, 2), expected_saves=round(saves, 2),
                       revenue_protected_usd=round(protected, 2), contact_cost_usd=round(contact_cost, 2),
                       incentive_cost_usd=round(incentive, 2), total_cost_usd=round(total, 2),
                       net_return_usd=round(protected - total, 2),
                       return_per_usd=round(protected / total, 3) if total > 0 else None)

    top = worklist.head(n)
    targeted = outcome(top["p_churn_90d"].to_numpy(), top["value_12m_usd"].to_numpy(), n)
    # a random sample of n has, in expectation, n x the population mean of p and of p x value
    random = outcome(np.full(n, p_all.mean()), np.full(n, float((p_all * v_all).sum() / p_all.sum() if p_all.sum() > 0 else 0.0)), n)
    return SimulateResponse(snapshot_date=str(pd.Timestamp(worklist["snapshot_date"].iloc[0]).date()),
                            assumptions=req, targeted=targeted, random=random,
                            uplift_net_usd=round(targeted.net_return_usd - random.net_return_usd, 2))


# ----------------------------------------------------------------------------- app
def create_app(root: Path | None = None) -> FastAPI:
    root = Path(root or os.environ.get("PULSE_ROOT", PROJECT_ROOT))
    app = FastAPI(title="Pulse scoring API", version="0.6.0",
                  description="Churn risk, value at risk, next best product and action cards from the Pulse pipeline. "
                              "Customer references are pseudonymised.")
    app.state.root = root
    app.state.store = None

    def get_store(request: Request) -> Store:
        if request.app.state.store is None:
            try:
                request.app.state.store = Store(request.app.state.root)
            except FileNotFoundError as e:
                raise HTTPException(503, f"pipeline outputs not found ({e.filename}); run `python -m pulse.cli pipeline`")
        return request.app.state.store

    @app.get("/health", response_model=Health)
    def health(request: Request):
        try:
            s = get_store(request)
        except HTTPException as e:
            return Health(status=f"waiting: {e.detail}", snapshot_date=None, customers=0, cards=0,
                          loaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), root=str(request.app.state.root))
        return Health(status="ok", snapshot_date=s.snapshot_date, customers=len(s.worklist), cards=len(s.cards),
                      loaded_at=s.loaded_at, root=str(s.root))

    @app.post("/reload", response_model=Health, summary="Reload the pipeline outputs after a new run")
    def reload(request: Request):
        request.app.state.store = None
        return health(request)

    @app.get("/worklist", response_model=WorklistResponse, summary="Customers ranked by value at risk")
    def worklist(request: Request, limit: int | None = Query(None, ge=1), segment: str | None = None,
                 branch: str | None = None, urgency: str | None = None, store: Store = Depends(get_store)):
        acfg = store.settings["api"]
        limit = min(limit or int(acfg["default_limit"]), int(acfg["max_limit"]))
        df = store.worklist
        for col, val in (("segment", segment), ("branch", branch), ("urgency", urgency)):
            if val is not None:
                df = df[df[col].str.lower() == val.lower()]
        rows = df.head(limit)[WORKLIST_FIELDS].to_dict("records")
        return WorklistResponse(snapshot_date=store.snapshot_date, total=len(df), returned=len(rows),
                                total_value_at_risk_usd=round(float(df["value_at_risk_usd"].sum()), 2), rows=rows)

    @app.get("/customers/{customer_ref}/card", response_model=Card, summary="Action card for one pseudonymised customer")
    def card(customer_ref: str, store: Store = Depends(get_store)):
        c = store.cards.get(customer_ref.upper())
        if c is None:
            raise HTTPException(404, "no card for this customer reference at the latest snapshot")
        return c

    @app.get("/segments", response_model=SegmentsResponse, summary="Churn risk and value at risk by branch, segment and product")
    def segments(dimension: Literal["branch", "segment", "product"] | None = None, store: Store = Depends(get_store)):
        df = store.segments if dimension is None else store.segments[store.segments["dimension"] == dimension]
        df = df.rename(columns={"churn_rate_%": "churn_rate_pct", "share_of_var_%": "share_of_var_pct"})
        return SegmentsResponse(snapshot_date=store.snapshot_date, rows=df.to_dict("records"))

    @app.get("/metrics", response_model=Metrics, summary="Model results, fairness table, drift and the last run")
    def metrics(store: Store = Depends(get_store)):
        m, lg = store.metrics, store.metrics["lightgbm"]
        return Metrics(run_date=m["run_date"], seed=m["seed"], n_features=m["n_features"], split=m["split"],
                       model_comparison_test=m["comparison_test"], lightgbm_test=lg["test"], calibration=lg["calibration"],
                       warnings=m["warnings"], fairness_test=m["fairness_test"],
                       survival={k: store.survival.get(k) for k in ("concordance_test", "timing_test", "bands_test") if k in store.survival},
                       drift=store.drift, last_run=store.runs[-1] if store.runs else None, runs_logged=len(store.runs))

    @app.post("/simulate", response_model=SimulateResponse, summary="Return on contacting the top N by value at risk vs random")
    def simulate_endpoint(req: SimulateRequest, store: Store = Depends(get_store)):
        return simulate(store.worklist, req)

    return app


app = create_app()
