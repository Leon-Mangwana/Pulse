"""Time-to-churn: a Cox proportional hazards model, plus the Phase 3 evaluation report.

    python -m pulse.cli train      (runs after the churn model)

The clock starts at the snapshot. The event is churn at `days_to_churn`; a customer still
here at the end of the 90-day window is censored at 90 days: we know they lasted at least
that long, nothing more. The Cox model learns one baseline timing curve from the data and,
for each feature, a multiplier on the rate of leaving. It is fitted on the train and
validation blocks of the Phase 2 split and judged on the same untouched test block.

Under proportional hazards every customer shares the same curve shape, so the 30-day and
90-day probabilities rank customers identically: the urgency band is a risk tier expressed
in days, not a separate prediction of timing. The report says so and checks whether the
higher-risk customers really do leave sooner.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from pulse.config import PROJECT_ROOT, load_settings, resolve
from pulse.models import churn
from pulse.io import read_parquet
from pulse.value import call_list_comparison, compute_value, unknown_products

BANDS = ["act this week", "act this month", "watch"]


class CoxChurn:
    """Clip, standardise, Cox fit; predicts P(churn within t days) for the configured horizons."""

    def __init__(self, cfg: dict, horizon_days: int):
        self.cfg, self.horizon = cfg, int(horizon_days)
        self.features = list(cfg["features"])
        self.horizons = [int(h) for h in cfg["horizons_days"]]

    def durations(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        E = (df["churn_90d"].to_numpy(float) == 1)
        T = np.where(E, df["days_to_churn"].to_numpy(float), float(self.horizon))
        return np.clip(T, 1.0, None), E.astype(int)

    def _prep(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        X = df[self.features].to_numpy(float)
        if fit:
            lo, hi = self.cfg["clip_percentiles"]
            self.lo_, self.hi_ = np.percentile(X, lo, axis=0), np.percentile(X, hi, axis=0)
            Xc = np.clip(X, self.lo_, self.hi_)
            self.mean_, self.std_ = Xc.mean(axis=0), Xc.std(axis=0) + 1e-9
        Xc = np.clip(X, self.lo_, self.hi_)
        return pd.DataFrame((Xc - self.mean_) / self.std_, columns=self.features, index=df.index)

    def fit(self, df: pd.DataFrame) -> "CoxChurn":
        data = self._prep(df, fit=True)
        data["T"], data["E"] = self.durations(df)
        self.cph_ = CoxPHFitter(penalizer=float(self.cfg["penalizer"])).fit(data, duration_col="T", event_col="E")
        return self

    def risk(self, df: pd.DataFrame) -> np.ndarray:
        return self.cph_.predict_partial_hazard(self._prep(df)).to_numpy(float)

    def predict_probs(self, df: pd.DataFrame) -> pd.DataFrame:
        surv = self.cph_.predict_survival_function(self._prep(df), times=self.horizons)   # (times x rows)
        out = pd.DataFrame(1.0 - surv.T.to_numpy(float), index=df.index,
                           columns=[f"p_churn_{h}d" for h in self.horizons])
        return out.clip(0.0, 1.0)

    def concordance(self, df: pd.DataFrame, score: np.ndarray | None = None) -> float:
        T, E = self.durations(df)
        risk = self.risk(df) if score is None else np.asarray(score, float)
        return float(concordance_index(T, -risk, E))

    def summary(self) -> pd.DataFrame:
        s = self.cph_.summary[["coef", "exp(coef)", "p"]].reset_index().rename(columns={"covariate": "feature"})
        s["coef"] = s["coef"].round(3)
        s["exp(coef)"] = s["exp(coef)"].round(3)
        s["p"] = s["p"].map(lambda v: f"{v:.3g}")
        s["reading"] = [f"one standard deviation higher multiplies the rate of leaving by {r:.2f}" for r in s["exp(coef)"]]
        return s.sort_values("exp(coef)", ascending=False).reset_index(drop=True)


def urgency_band(probs: pd.DataFrame, cfg: dict) -> np.ndarray:
    b = cfg["bands"]
    p30, p90 = probs["p_churn_30d"].to_numpy(float), probs["p_churn_90d"].to_numpy(float)
    return np.where(p30 >= float(b["act_this_week_p30"]), BANDS[0],
                    np.where(p90 >= float(b["act_this_month_p90"]), BANDS[1], BANDS[2])).astype(object)


def load_products(root: Path, settings: dict) -> pd.DataFrame:
    con = duckdb.connect(str(resolve(root, settings["data"]["duckdb_path"])), read_only=True)
    try:
        df = con.execute("SELECT customer_id, product, start_date, end_date FROM products").df()
    finally:
        con.close()
    for c in ("start_date", "end_date"):
        df[c] = pd.to_datetime(df[c])
    return df


def train_survival(feats: pd.DataFrame, products: pd.DataFrame, settings: dict, model_dir: Path, log=print) -> dict:
    """Fit the Cox model, evaluate it and the value-at-risk call list on the test block, save."""
    cfg, seed = settings["survival"], int(settings["seed"])
    horizon = int(settings["target"]["horizon_days"])
    model_dir = Path(model_dir)
    split = churn.time_split(feats, settings["model"], horizon)
    fit_rows = pd.concat([split.train, split.valid])
    test = split.test
    md: list[str] = []

    def section(title, obj=None):
        log(f"\n## {title}")
        md.append(f"\n**{title}**\n")
        if obj is not None:
            text = obj.to_string(index=False) if isinstance(obj, pd.DataFrame) else str(obj)
            log(text)
            md.append(churn.to_markdown(obj) if isinstance(obj, pd.DataFrame) else f"```\n{text}\n```")

    t0 = time.perf_counter()
    cox = CoxChurn(cfg, horizon).fit(fit_rows)
    log(f"Cox fitted on {len(fit_rows):,} rows ({int(fit_rows['churn_90d'].sum()):,} events) in {time.perf_counter() - t0:.1f}s")
    section("Cox coefficients (features standardised; exp(coef) is the hazard multiplier per standard deviation)", cox.summary())

    # --- concordance on the test block: Cox versus the Phase 2 churn score
    p_lgb = churn.predict(model_dir, test)
    probs = cox.predict_probs(test)
    conc = pd.DataFrame([
        {"ranking": "Cox partial hazard", "concordance_test": round(cox.concordance(test), 3)},
        {"ranking": "LightGBM calibrated probability", "concordance_test": round(cox.concordance(test, p_lgb), 3)},
    ])
    section("Concordance on the test block (chance = 0.5): who leaves first?", conc)

    # --- do higher-risk customers really leave sooner?
    tc = test[test["churn_90d"] == 1].copy()
    tc["when"] = pd.cut(tc["days_to_churn"], [0, 30, 60, 1000], labels=["within 30 days", "31 to 60 days", "61 to 92 days"])
    tc["p30"] = probs.loc[tc.index, "p_churn_30d"]
    tc["p90"] = probs.loc[tc.index, "p_churn_90d"]
    timing = tc.groupby("when", observed=True).agg(churners=("p30", "size"), mean_p30=("p30", "mean"), mean_p90=("p90", "mean")).reset_index()
    timing[["mean_p30", "mean_p90"]] = (100 * timing[["mean_p30", "mean_p90"]]).round(1)
    timing.columns = ["churned", "churners", "mean_p30_%", "mean_p90_%"]
    section("Test churners by when they left, and the Cox probabilities they were given", timing)

    # --- urgency bands on the test block: size and what actually happened
    band = urgency_band(probs, cfg)
    tb = pd.DataFrame({"band": band, "churn": test["churn_90d"].to_numpy(int),
                       "within30": ((test["churn_90d"] == 1) & (test["days_to_churn"] <= 30)).to_numpy(int)})
    bands = tb.groupby("band").agg(rows=("churn", "size"), churned_90d_pct=("churn", "mean"), churned_30d_pct=("within30", "mean")).reindex(BANDS).reset_index()
    bands["share_of_rows_%"] = (100 * bands["rows"] / len(tb)).round(1)
    bands["churned_90d_pct"] = (100 * bands["churned_90d_pct"]).round(1)
    bands["churned_30d_pct"] = (100 * bands["churned_30d_pct"]).round(1)
    bands = bands.rename(columns={"churned_90d_pct": "churned_within_90d_%", "churned_30d_pct": "churned_within_30d_%"})
    section("Urgency bands on the test block (thresholds in settings.yaml)", bands)

    # --- calibration of the Cox 90-day probability against the truth
    cal = pd.DataFrame([{"mean_cox_p90_%": round(100 * probs["p_churn_90d"].mean(), 2),
                         "mean_lightgbm_calibrated_%": round(100 * p_lgb.mean(), 2),
                         "actual_churn_%": round(100 * test["churn_90d"].mean(), 2)}])
    section("Average 90-day probability versus the actual rate on the test block", cal)

    # --- value and the call-list comparison
    vcfg = settings["value"]
    unknown = unknown_products(products, vcfg)
    if unknown:
        log(f"WARNING: products with no margin in settings.yaml (counted as 0): {unknown}")
    val = compute_value(test, products, vcfg)
    desc = val.describe(percentiles=[0.5, 0.9]).loc[["mean", "50%", "90%", "max"]].round(0).reset_index().rename(columns={"index": "stat"})
    section("12-month value on the test block, USD", desc)
    comparison = call_list_comparison(test["churn_90d"].to_numpy(int), p_lgb, val["value_12m_usd"].to_numpy(),
                                      [float(s) for s in settings["worklist"]["top_shares"]], seed)
    total_at_risk = float(val.loc[test["churn_90d"] == 1, "value_12m_usd"].sum())
    section(f"Call-list comparison on the test block: 12-month value of customers who actually churned "
            f"(total {total_at_risk:,.0f} USD across {int(test['churn_90d'].sum())} churners)", comparison)

    joblib.dump(cox, model_dir / "survival.joblib")
    metrics = {
        "run_date": pd.Timestamp.today().strftime("%Y-%m-%d"), "fit_rows": int(len(fit_rows)),
        "fit_events": int(fit_rows["churn_90d"].sum()), "features": cox.features,
        "coefficients": cox.summary().to_dict(orient="records"),
        "concordance_test": conc.to_dict(orient="records"), "timing_test": timing.to_dict(orient="records"),
        "bands_test": bands.to_dict(orient="records"), "calibration_test": cal.to_dict(orient="records"),
        "value_test": desc.to_dict(orient="records"), "call_list_test": comparison.to_dict(orient="records"),
        "unknown_products": unknown, "markdown": "\n".join(md),
    }
    (model_dir / "survival_metrics.json").write_text(json.dumps({k: v for k, v in metrics.items() if k != "markdown"},
                                                                indent=1, default=str), encoding="utf-8")
    (model_dir / "value_report.md").write_text(metrics["markdown"], encoding="utf-8")
    log(f"\nsaved survival.joblib, survival_metrics.json, value_report.md to {model_dir}")
    return metrics


def main(root: Path = PROJECT_ROOT) -> dict:
    settings = load_settings(root)
    t0 = time.perf_counter()
    feats = read_parquet(resolve(root, settings["data"]["processed_dir"]) / "features.parquet")
    feats["snapshot_date"] = pd.to_datetime(feats["snapshot_date"])
    products = load_products(root, settings)
    metrics = train_survival(feats, products, settings, resolve(root, churn.MODEL_DIR))
    print(f"\nsurvival + value evaluated in {time.perf_counter() - t0:.1f}s")
    return metrics


if __name__ == "__main__":
    main()
