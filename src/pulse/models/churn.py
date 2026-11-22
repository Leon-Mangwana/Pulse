"""Churn model: time split, three models, honest evaluation, fairness report.

    python -m pulse.cli train

Reads data/processed/features.parquet, splits by snapshot date (settings.yaml `model.split`),
trains a rules baseline, a logistic regression and a LightGBM model on the same rows,
reports everything on the untouched test block, and saves the LightGBM model plus its
calibrator, feature list and metrics to models/.

Protected attributes (gender, age_band) are never model inputs: they are joined from
the customers table after scoring, only to build the fairness table.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pulse.config import PROJECT_ROOT, current_mode, load_settings, resolve
from pulse.features.build import FAMILIES, feature_names
from pulse.io import read_parquet

PROTECTED = ["gender", "age_band"]
MODEL_DIR = "models"


# --------------------------------------------------------------------------- split
@dataclass
class Split:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame

    def blocks(self):
        return [("train", self.train), ("valid", self.valid), ("test", self.test)]


def time_split(feats: pd.DataFrame, cfg: dict, horizon_days: int) -> Split:
    """Earliest snapshots train, next block validates, latest block tests, with a gap of
    one label horizon before the test block so no train/valid label window overlaps it."""
    s = cfg["split"]
    train_end, valid_end, test_start = (pd.Timestamp(s[k]) for k in ("train_end", "valid_end", "test_start"))
    if not (train_end < valid_end < test_start):
        raise ValueError("model.split must satisfy train_end < valid_end < test_start")
    h_months = int(round(horizon_days / 30.4))     # the label looks at the next h_months calendar months
    if valid_end + pd.DateOffset(months=h_months) > test_start:
        raise ValueError(f"model.split: test_start must be at least {h_months} months after valid_end")
    lab = feats[feats["labelled"] == 1]
    d = lab["snapshot_date"]
    return Split(train=lab[d <= train_end], valid=lab[(d > train_end) & (d <= valid_end)],
                 test=lab[d >= test_start])


def stratified_split(feats: pd.DataFrame, cfg: dict, seed: int) -> Split:
    """Fallback mode only: one snapshot, so no time axis. A seeded stratified random split keeps
    the churn rate equal in train, validation and test. Optimistic compared with a time split."""
    s = cfg["summary_split"]
    valid_share, test_share = float(s["valid_share"]), float(s["test_share"])
    lab = feats[feats["labelled"] == 1]
    y = lab["churn_90d"].to_numpy(int)
    rest, test = train_test_split(lab, test_size=test_share, stratify=y, random_state=seed)
    train, valid = train_test_split(rest, test_size=valid_share / (1 - test_share),
                                    stratify=rest["churn_90d"].to_numpy(int), random_state=seed)
    return Split(train=train, valid=valid, test=test)


def describe_split(split: Split) -> pd.DataFrame:
    rows = []
    for name, df in split.blocks():
        rows.append({"block": name, "from": df["snapshot_date"].min().date(), "to": df["snapshot_date"].max().date(),
                     "snapshots": df["snapshot_date"].nunique(), "rows": len(df),
                     "churners": int(df["churn_90d"].sum()), "churn_%": round(100 * df["churn_90d"].mean(), 2)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- metrics
def top_k_capture(y: np.ndarray, score: np.ndarray, share: float) -> tuple[float, float, float]:
    """(capture, lift, flag_threshold) for calling the top `share` of rows by score."""
    n = len(y)
    k = max(1, int(np.floor(share * n)))
    order = np.argsort(-score, kind="stable")
    top = order[:k]
    pos = y.sum()
    capture = float(y[top].sum() / pos) if pos > 0 else float("nan")
    return capture, capture / share, float(score[order[k - 1]])


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int) -> pd.DataFrame:
    """Rows sorted by predicted probability and cut into equal-count bins (decile 1 = lowest)."""
    order = np.argsort(p, kind="stable")
    groups = np.array_split(order, bins)
    rows = []
    for i, g in enumerate(groups, 1):
        rows.append({"decile": i, "rows": len(g), "predicted_%": round(100 * p[g].mean(), 2),
                     "actual_%": round(100 * y[g].mean(), 2)})
    return pd.DataFrame(rows)


def evaluate(y: np.ndarray, score: np.ndarray, shares: list[float]) -> dict:
    out = {"pr_auc": float(average_precision_score(y, score)), "roc_auc": float(roc_auc_score(y, score))}
    for s in shares:
        cap, lift, thr = top_k_capture(y, score, s)
        pct = int(round(100 * s))
        out[f"capture_top{pct}"] = cap
        out[f"lift_top{pct}"] = lift
        out[f"threshold_top{pct}"] = thr
    return out


def to_markdown(df: pd.DataFrame) -> str:
    """Markdown table without the tabulate dependency."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in row) + " |")
    return "\n".join(lines)


def expected_calibration_error(table: pd.DataFrame) -> float:
    w = table["rows"] / table["rows"].sum()
    return float((w * (table["predicted_%"] - table["actual_%"]).abs()).sum() / 100)


# --------------------------------------------------------------------------- models
def rules_score(X: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Count of warning flags a branch manager could compute by hand; ties broken by how far
    the balance and activity fell. Higher = riskier."""
    flags = (
        ((X["salary_detected"] == 1) & (X["months_since_salary"] >= cfg["no_salary_months"])).astype(float)
        + (X["bal_change_vs_6m"] < cfg["balance_drop"]).astype(float)
        + (X["txn_count_change"] < cfg["activity_drop"]).astype(float)
        + (X["leak_share_trend"] > cfg["leak_rise"]).astype(float)
    )
    tiebreak = (1 - X["bal_change_vs_6m"].clip(0, 1)) + (1 - X["txn_count_change"].clip(0, 1))
    return (flags + 0.1 * tiebreak).to_numpy()


class LogisticModel:
    """Clip at train percentiles, standardise, logistic regression with balanced class weights."""

    def __init__(self, cfg: dict, seed: int):
        self.cfg, self.seed = cfg, seed

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        lo, hi = self.cfg["clip_percentiles"]
        self.lo_ = np.nanpercentile(X.to_numpy(float), lo, axis=0)
        self.hi_ = np.nanpercentile(X.to_numpy(float), hi, axis=0)
        self.scaler_ = StandardScaler().fit(self._clip(X))
        self.lr_ = LogisticRegression(C=self.cfg["C"], class_weight="balanced", max_iter=5000,
                                      random_state=self.seed)
        self.lr_.fit(self.scaler_.transform(self._clip(X)), y)
        return self

    def _clip(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(X.to_numpy(float), self.lo_, self.hi_)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.lr_.predict_proba(self.scaler_.transform(self._clip(X)))[:, 1]


CANDIDATE_KEYS = ["num_leaves", "min_child_samples", "feature_fraction", "lambda_l2", "learning_rate"]


def lgb_params(cfg: dict, candidate: dict, pos_weight: float, seed: int) -> dict:
    p = {k: v for k, v in cfg.items() if k not in ("candidates", "max_rounds", "early_stopping_rounds")}
    p.update(candidate)
    p.update({"objective": "binary", "metric": "average_precision", "verbosity": -1,
              "scale_pos_weight": pos_weight, "seed": seed, "deterministic": True,
              "force_row_wise": True, "num_threads": 4})
    return p


def fit_lightgbm(Xtr, ytr, Xva, yva, cfg: dict, seed: int) -> tuple[lgb.Booster, pd.DataFrame]:
    """Fit each candidate with early stopping on the validation block; keep the one with the best
    validation PR-AUC. A handful of fixed candidates, not a search."""
    pos_weight = float((ytr == 0).sum() / max(1, ytr.sum()))
    dtr = lgb.Dataset(Xtr, ytr, free_raw_data=False)
    dva = lgb.Dataset(Xva, yva, reference=dtr, free_raw_data=False)
    rows, boosters = [], []
    for i, cand in enumerate(cfg["candidates"], 1):
        booster = lgb.train(lgb_params(cfg, cand, pos_weight, seed), dtr, num_boost_round=cfg["max_rounds"],
                            valid_sets=[dva], valid_names=["valid"],
                            callbacks=[lgb.early_stopping(cfg["early_stopping_rounds"], verbose=False)])
        pv = booster.predict(Xva, num_iteration=booster.best_iteration)
        ptr = booster.predict(Xtr, num_iteration=booster.best_iteration)
        row = {"candidate": i, **{k: cand.get(k, cfg.get(k)) for k in CANDIDATE_KEYS},
               "rounds": booster.best_iteration,
               "train_pr_auc": round(float(average_precision_score(ytr, ptr)), 3),
               "valid_pr_auc": round(float(average_precision_score(yva, pv)), 3)}
        rows.append(row)
        boosters.append(booster)
    table = pd.DataFrame(rows)
    # keep the least overfit candidate among those within `simpler_tolerance` of the best validation score
    tol = float(cfg.get("simpler_tolerance", 0.0))
    ok = table["valid_pr_auc"] >= table["valid_pr_auc"].max() - tol
    gap = (table["train_pr_auc"] - table["valid_pr_auc"]).where(ok, np.inf)
    chosen = int(gap.idxmin())
    table["chosen"] = np.where(table.index == chosen, "*", "")
    return boosters[chosen], table


# --------------------------------------------------------------------------- sanity checks
def feature_direction(X: pd.DataFrame, y: np.ndarray) -> dict[str, str]:
    """'higher' if churners have a higher median than non-churners, else 'lower'."""
    med1, med0 = X[y == 1].median(), X[y == 0].median()
    return {c: ("higher" if med1[c] > med0[c] else "lower" if med1[c] < med0[c] else "similar") for c in X.columns}


def top_features(booster: lgb.Booster, names: list[str], X: pd.DataFrame, y: np.ndarray, n: int = 15) -> pd.DataFrame:
    desc = {k: v for _, mod in FAMILIES for k, v in mod.FEATURES.items()}
    gain = booster.feature_importance(importance_type="gain")
    direction = feature_direction(X, y)
    df = pd.DataFrame({"feature": names, "gain_%": 100 * gain / gain.sum()}).sort_values("gain_%", ascending=False)
    df = df.head(n).reset_index(drop=True)
    df["gain_%"] = df["gain_%"].round(1)
    def describe(f: str) -> str:   # mapped summary-mode columns have no dictionary entry
        if f in desc:
            return desc[f]
        if f.endswith("_missing"):
            return f"1 when {f[:-8].replace('_', ' ')} was empty in the source table"
        return f"{f.replace('_', ' ')}, as supplied in the source table"
    df["reading"] = [f"{describe(f)}; churners have {direction[f]} values" for f in df["feature"]]
    return df


def single_feature_auc(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    rows = []
    for c in X.columns:
        v = X[c].to_numpy(float)
        if np.nanstd(v) == 0:
            auc = 0.5
        else:
            auc = roc_auc_score(y, v)
            auc = max(auc, 1 - auc)
        rows.append({"feature": c, "roc_auc_alone": round(float(auc), 3)})
    return pd.DataFrame(rows).sort_values("roc_auc_alone", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- fairness
def fairness_table(test: pd.DataFrame, score: np.ndarray, threshold: float, attrs: pd.DataFrame) -> pd.DataFrame:
    """Capture and flag rate by protected attribute at the top-10% cut. The attributes are
    joined here, after scoring, and never enter the model."""
    df = test[["customer_id", "churn_90d"]].copy()
    df["flag"] = (score >= threshold).astype(int)
    df = df.merge(attrs, on="customer_id", how="left")
    rows = []
    for attr in PROTECTED:
        for val, g in df.groupby(attr, sort=True):
            pos = g["churn_90d"].sum()
            rows.append({"attribute": attr, "group": val, "rows": len(g),
                         "churn_%": round(100 * g["churn_90d"].mean(), 2),
                         "flag_%": round(100 * g["flag"].mean(), 2),
                         "capture_%": round(100 * g.loc[g["churn_90d"] == 1, "flag"].mean(), 1) if pos else float("nan")})
    return pd.DataFrame(rows)


def load_protected(root: Path, settings: dict) -> pd.DataFrame:
    con = duckdb.connect(str(resolve(root, settings["data"]["duckdb_path"])), read_only=True)
    try:
        return con.execute(f"SELECT customer_id, {', '.join(PROTECTED)} FROM customers").df()
    finally:
        con.close()


# --------------------------------------------------------------------------- train
def train(feats: pd.DataFrame, settings: dict, attrs: pd.DataFrame | None = None, out_dir: Path | None = None,
          log=print, names: list[str] | None = None, split_mode: str = "time") -> dict:
    """Run the whole Phase 2 recipe on a feature table. Returns the metrics dict; saves to out_dir if given.

    `names` defaults to the transaction feature list; fallback mode passes the mapped columns.
    `split_mode` is "time" (always, when there is a time axis) or "stratified" (fallback mode only)."""
    cfg, seed = settings["model"], int(settings["seed"])
    names = list(names) if names is not None else feature_names()
    assert not set(PROTECTED) & set(names), "protected attributes must never be model inputs"
    if split_mode == "time":
        split = time_split(feats, cfg, int(settings["target"]["horizon_days"]))
    elif split_mode == "stratified":
        split = stratified_split(feats, cfg, seed)
    else:
        raise ValueError(f"unknown split_mode {split_mode!r}")
    rules_ok = all(c in names for c in ("salary_detected", "months_since_salary", "bal_change_vs_6m",
                                        "txn_count_change", "leak_share_trend"))
    shares = [float(s) for s in cfg["top_shares"]]
    md: list[str] = []

    def section(title, obj=None):
        log(f"\n## {title}")
        md.append(f"\n**{title}**\n")
        if obj is not None:
            text = obj.to_string(index=False) if isinstance(obj, pd.DataFrame) else str(obj)
            log(text)
            md.append(to_markdown(obj) if isinstance(obj, pd.DataFrame) else f"```\n{text}\n```")

    split_df = describe_split(split)
    section("Time split (labelled snapshots only; the gap before the test block is dropped)" if split_mode == "time"
            else "Stratified random split (fallback mode: no time axis; results are optimistic)", split_df)

    X = {k: df[names] for k, df in split.blocks()}
    y = {k: df["churn_90d"].to_numpy(int) for k, df in split.blocks()}

    t0 = time.perf_counter()
    scores: dict[str, dict[str, np.ndarray]] = {}
    if rules_ok:
        scores["rules"] = {k: rules_score(X[k], cfg["rules"]) for k in X}
    else:
        log("rules baseline skipped: its hand-made flags need transaction features")
    logit = LogisticModel(cfg["logistic"], seed).fit(X["train"], y["train"])
    scores["logistic"] = {k: logit.predict(X[k]) for k in X}
    lgb_cfg = cfg["lightgbm"]
    if split_mode == "stratified":
        # customer-level tables are small (thousands of rows, not tens of thousands of customer-months):
        # scale the leaf-size floor to the data so the trees can split at all
        floor = max(20, int(0.02 * len(X["train"])))
        lgb_cfg = {**lgb_cfg, "candidates": [{**c, "min_child_samples": min(int(c["min_child_samples"]), floor)}
                                             for c in lgb_cfg["candidates"]]}
    booster, candidates = fit_lightgbm(X["train"], y["train"], X["valid"], y["valid"], lgb_cfg, seed)
    raw = {k: booster.predict(X[k], num_iteration=booster.best_iteration) for k in X}
    scores["lightgbm"] = raw
    log(f"\nfit {len(scores)} models in {time.perf_counter() - t0:.1f}s")
    section("LightGBM candidates, chosen on the validation block (test never looked at)", candidates)

    # --- comparison on the test block
    comp_rows = []
    for name, sc in scores.items():
        m = evaluate(y["test"], sc["test"], shares)
        comp_rows.append({"model": name, "PR-AUC": round(m["pr_auc"], 3), "ROC-AUC": round(m["roc_auc"], 3),
                          **{f"capture@{int(round(100*s))}%": round(100 * m[f"capture_top{int(round(100*s))}"], 1) for s in shares},
                          **{f"lift@{int(round(100*s))}%": round(m[f"lift_top{int(round(100*s))}"], 2) for s in shares}})
    comparison = pd.DataFrame(comp_rows)
    section("Test block: three models", comparison)

    # --- calibration before and after isotonic fit on the validation block
    bins = int(cfg["calibration_bins"])
    cal_before = calibration_table(y["test"], raw["test"], bins)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw["valid"], y["valid"])
    calibrated = {k: iso.predict(raw[k]) for k in raw}
    cal_after = calibration_table(y["test"], calibrated["test"], bins)
    cal = cal_before.merge(cal_after, on=["decile", "rows"], suffixes=("_raw", "_calibrated"))
    cal.columns = ["decile", "rows", "raw_pred_%", "actual_%", "calibrated_pred_%", "actual_%_"]
    cal = cal.drop(columns="actual_%_")
    section("LightGBM calibration on the test block (decile 10 = riskiest)", cal)
    cal_stats = {
        "brier_raw": float(brier_score_loss(y["test"], raw["test"])),
        "brier_calibrated": float(brier_score_loss(y["test"], calibrated["test"])),
        "ece_raw": expected_calibration_error(cal_before), "ece_calibrated": expected_calibration_error(cal_after),
        "mean_pred_raw": float(raw["test"].mean()), "mean_pred_calibrated": float(calibrated["test"].mean()),
        "actual_rate": float(y["test"].mean()),
    }
    section("Calibration summary", pd.DataFrame([{k: round(v, 4) for k, v in cal_stats.items()}]))

    # --- sanity checks
    top = top_features(booster, names, X["train"], y["train"])
    section("Top 15 LightGBM features by gain", top)
    sfa = single_feature_auc(X["test"], y["test"])
    warn_auc = float(cfg["leakage_auc_warning"])
    leaky = sfa[sfa["roc_auc_alone"] > warn_auc]
    section(f"Strongest single features on the test block (leakage warning above {warn_auc})", sfa.head(10))
    log("LEAKAGE WARNING: " + ", ".join(leaky["feature"]) if len(leaky) else f"no single feature above {warn_auc} alone")
    m_train = evaluate(y["train"], raw["train"], shares)
    m_test = evaluate(y["test"], raw["test"], shares)
    gap = pd.DataFrame([{"metric": "PR-AUC", "train": round(m_train["pr_auc"], 3), "test": round(m_test["pr_auc"], 3)},
                        {"metric": "ROC-AUC", "train": round(m_train["roc_auc"], 3), "test": round(m_test["roc_auc"], 3)},
                        {"metric": "capture@10%", "train": round(100 * m_train["capture_top10"], 1),
                         "test": round(100 * m_test["capture_top10"], 1)}])
    gap["gap"] = (gap["train"] - gap["test"]).round(3)
    section("LightGBM train vs test (overfitting check)", gap)
    warn_gap = float(cfg["overfit_gap_warning"])
    overfit = (m_train["pr_auc"] - m_test["pr_auc"]) > warn_gap
    log("OVERFIT WARNING: train PR-AUC exceeds test by more than " + str(warn_gap) if overfit else "train/test gap within limit")

    # --- fairness (attributes joined after scoring; never inputs)
    thr10 = m_test["threshold_top10"]
    fair = None
    if attrs is not None:
        fair = fairness_table(split.test, raw["test"], thr10, attrs)
        section("Fairness on the test block at the top-10% cut (protected attributes never used as inputs)", fair)

    metrics = {
        "run_date": pd.Timestamp.today().strftime("%Y-%m-%d"), "seed": seed, "n_features": len(names),
        "mode": "summary" if split_mode == "stratified" else "transactions", "split_mode": split_mode,
        "split": split_df.astype(str).to_dict(orient="records"),
        "comparison_test": comparison.to_dict(orient="records"),
        "lightgbm": {"best_iteration": int(booster.best_iteration), "params": booster.params,
                     "candidates": candidates.to_dict(orient="records"),
                     "test": m_test, "train": m_train, "calibration": cal_stats,
                     "calibration_table_test": cal.to_dict(orient="records"),
                     "threshold_top10_raw": float(thr10),
                     "threshold_top10_calibrated": float(iso.predict([thr10])[0])},
        "top_features": top.to_dict(orient="records"),
        "single_feature_auc_top10": sfa.head(10).to_dict(orient="records"),
        "warnings": {"leakage": leaky["feature"].tolist(), "overfit": bool(overfit)},
        "fairness_test": fair.to_dict(orient="records") if fair is not None else None,
        "markdown": "\n".join(md),
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(out_dir / "churn_lgbm.txt"), num_iteration=booster.best_iteration)
        joblib.dump(iso, out_dir / "calibrator.joblib")
        (out_dir / "feature_list.json").write_text(json.dumps(names, indent=1), encoding="utf-8")
        (out_dir / "metrics.json").write_text(json.dumps({k: v for k, v in metrics.items() if k != "markdown"},
                                                         indent=1, default=str), encoding="utf-8")
        (out_dir / "train_report.md").write_text(metrics["markdown"], encoding="utf-8")
        log(f"\nsaved churn_lgbm.txt, calibrator.joblib, feature_list.json, metrics.json, train_report.md to {out_dir}")
    return metrics


def load_model(model_dir: Path):
    """(booster, calibrator, feature list) as saved by train()."""
    model_dir = Path(model_dir)
    booster = lgb.Booster(model_file=str(model_dir / "churn_lgbm.txt"))
    iso = joblib.load(model_dir / "calibrator.joblib")
    names = json.loads((model_dir / "feature_list.json").read_text(encoding="utf-8"))
    return booster, iso, names


def predict(model_dir: Path, X: pd.DataFrame) -> np.ndarray:
    """Calibrated churn probability for rows of the feature table."""
    booster, iso, names = load_model(model_dir)
    return iso.predict(booster.predict(X[names]))


def main(root: Path = PROJECT_ROOT) -> dict:
    settings = load_settings(root)
    t0 = time.perf_counter()
    feats = read_parquet(resolve(root, settings["data"]["processed_dir"]) / "features.parquet")
    feats["snapshot_date"] = pd.to_datetime(feats["snapshot_date"])
    print(f"loaded {len(feats):,} snapshot rows ({int(feats['labelled'].sum()):,} labelled) in {time.perf_counter() - t0:.1f}s")
    attrs = load_protected(root, settings)
    if attrs[PROTECTED].isna().all().all():
        attrs = None
        print("no protected attributes mapped: fairness table skipped")
    if current_mode(root) == "summary":
        from pulse.summary_mode import feature_names_from
        metrics = train(feats, settings, attrs=attrs, out_dir=resolve(root, MODEL_DIR),
                        names=feature_names_from(feats), split_mode="stratified")
    else:
        metrics = train(feats, settings, attrs=attrs, out_dir=resolve(root, MODEL_DIR))
    print(f"\ntrained in {time.perf_counter() - t0:.1f}s")
    return metrics


if __name__ == "__main__":
    main()
