"""Cross-sell engine: trigger rules from config/crosssell.yaml, one small propensity model per
product where the data allows, one recommendation per customer, and a retention guardrail.

    python -m pulse.cli train      fits the propensity models and writes models/crosssell_report.md
    python -m pulse.cli score      adds recommendation, rec_score, rec_reason, rec_source to the worklist

The uptake label for a product at a snapshot is "the customer started it within `horizon_days`
after the snapshot date" and rows where the customer already holds it are excluded. The
propensity models use the Phase 2 time split (with its gap), class weights, early stopping on the
validation block and isotonic calibration, exactly like the churn model. A product with fewer
than `min_train_positives` uptake rows in the training block gets no model and the rule alone
decides. Every sentence shown to a user is a fixed template from the YAML file.
"""
from __future__ import annotations

import json
import operator
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from pulse.config import PROJECT_ROOT, load_settings, load_yaml, resolve
from pulse.features.build import feature_names
from pulse.io import read_parquet
from pulse.models import churn
from pulse.models.survival import load_products

OPS = {">=": operator.ge, ">": operator.gt, "<=": operator.le, "<": operator.lt, "==": operator.eq, "!=": operator.ne}
SOURCES = ["retention", "rule+model", "rule", "model", "fallback", "none"]
SUBDIR = "crosssell"


# --------------------------------------------------------------------------- rules
def load_rules(root: Path, settings: dict) -> dict:
    rules = load_yaml(resolve(root, settings["crosssell"]["rules_path"]))
    for t in rules["triggers"]:
        for f, op, _ in t["conditions"]:
            if op not in OPS:
                raise ValueError(f"trigger {t['name']}: unknown operator {op!r}")
    return rules


class _Safe(dict):
    def __missing__(self, key):
        return "?"


def fill(template: str, values: dict) -> str:
    """Fixed template + the customer's own numbers; a missing placeholder shows '?' rather than crashing."""
    return template.format_map(_Safe({k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                                      for k, v in values.items()}))


def _row_values(rows: pd.DataFrame, i: int, extra: dict | None = None) -> dict:
    d = rows.iloc[i].to_dict()
    if extra:
        d.update(extra)
    return d


def products_held(rows: pd.DataFrame, products: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Boolean frame (rows x products): held at the snapshot date."""
    key = rows[["customer_id", "snapshot_date"]].copy()
    key["_i"] = np.arange(len(rows))
    m = key.merge(products, on="customer_id", how="inner")
    held = (m["start_date"] <= m["snapshot_date"]) & (m["end_date"].isna() | (m["end_date"] > m["snapshot_date"]))
    out = pd.DataFrame(False, index=np.arange(len(rows)), columns=names)
    m = m[held & m["product"].isin(names)]
    for prod, g in m.groupby("product"):
        out.loc[g["_i"].unique(), prod] = True
    out.index = rows.index
    return out


def uptake_within(rows: pd.DataFrame, products: pd.DataFrame, names: list[str], horizon_days: int) -> pd.DataFrame:
    """Boolean frame (rows x products): started within horizon_days after the snapshot date."""
    key = rows[["customer_id", "snapshot_date"]].copy()
    key["_i"] = np.arange(len(rows))
    m = key.merge(products, on="customer_id", how="inner")
    up = (m["start_date"] > m["snapshot_date"]) & (m["start_date"] <= m["snapshot_date"] + pd.Timedelta(days=horizon_days))
    out = pd.DataFrame(False, index=np.arange(len(rows)), columns=names)
    m = m[up & m["product"].isin(names)]
    for prod, g in m.groupby("product"):
        out.loc[g["_i"].unique(), prod] = True
    out.index = rows.index
    return out


def trigger_mask(rows: pd.DataFrame, trigger: dict) -> np.ndarray:
    mask = np.ones(len(rows), dtype=bool)
    for feature, op, value in trigger["conditions"]:
        mask &= OPS[op](rows[feature].to_numpy(float), float(value))
    return mask


def fire_triggers(rows: pd.DataFrame, rules: dict, held: pd.DataFrame) -> pd.DataFrame:
    """Long table: one line per (row, trigger) that fires and whose product is not already held."""
    out = []
    for t in rules["triggers"]:
        mask = trigger_mask(rows, t)
        if t["product"] in held.columns:
            mask &= ~held[t["product"]].to_numpy()
        for i in np.flatnonzero(mask):
            out.append({"row": i, "trigger": t["name"], "product": t["product"], "priority": int(t.get("priority", 99)),
                        "reason": fill(t["reason"], _row_values(rows, i))})
    return pd.DataFrame(out, columns=["row", "trigger", "product", "priority", "reason"])


# --------------------------------------------------------------------------- propensity models
def product_list(products: pd.DataFrame, settings: dict) -> list[str]:
    skip = set(settings["crosssell"].get("skip_products", []))
    return sorted(set(products["product"].unique()) - skip)


def _lgb_cfg(settings: dict) -> dict:
    cfg = dict(settings["model"]["lightgbm"])
    cfg["candidates"] = [settings["crosssell"]["lightgbm_candidate"]]
    cfg["simpler_tolerance"] = 0.0
    return cfg


def train_propensity(feats: pd.DataFrame, products: pd.DataFrame, settings: dict, model_dir: Path, log=print) -> dict:
    """One LightGBM per product with enough uptakes; rule-only otherwise. Returns and saves the status table."""
    cfg, seed = settings["crosssell"], int(settings["seed"])
    out_dir = Path(model_dir) / SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    names = feature_names()
    horizon = int(cfg["horizon_days"])
    split = churn.time_split(feats, settings["model"], int(settings["target"]["horizon_days"]))
    prods = product_list(products, settings)
    shares = [float(s) for s in settings["model"]["top_shares"]]
    status, rows_out = {}, []
    for prod in prods:
        blocks = {}
        for name, df in split.blocks():
            held = products_held(df, products, [prod])[prod].to_numpy()
            up = uptake_within(df, products, [prod], horizon)[prod].to_numpy()
            keep = ~held
            blocks[name] = (df[names][keep], up[keep].astype(int))
        ytr = blocks["train"][1]
        base_rate = float(ytr.mean()) if len(ytr) else 0.0
        entry = {"train_rows": int(len(ytr)), "train_positives": int(ytr.sum()), "base_rate_train": base_rate}
        if ytr.sum() < int(cfg["min_train_positives"]) or blocks["valid"][1].sum() == 0 or blocks["test"][1].sum() == 0:
            entry["status"] = "rule_only"
            status[prod] = entry
            rows_out.append({"product": prod, **{k: entry[k] for k in ("train_rows", "train_positives")},
                             "status": "rule only (too few uptakes)", "PR-AUC": "", "ROC-AUC": "", "capture@10%": "", "lift@10%": ""})
            continue
        t0 = time.perf_counter()
        booster, _ = churn.fit_lightgbm(blocks["train"][0], ytr, blocks["valid"][0], blocks["valid"][1], _lgb_cfg(settings), seed)
        raw_va = booster.predict(blocks["valid"][0], num_iteration=booster.best_iteration)
        valid_roc = float(roc_auc_score(blocks["valid"][1], raw_va))
        if valid_roc < float(cfg["min_valid_roc_auc"]):          # judged on validation only, never on test
            entry.update({"status": "rule_only", "valid_roc_auc": valid_roc})
            status[prod] = entry
            rows_out.append({"product": prod, "train_rows": entry["train_rows"], "train_positives": entry["train_positives"],
                             "status": f"rule only (no signal: validation ROC-AUC {valid_roc:.2f})",
                             "PR-AUC": "", "ROC-AUC": "", "capture@10%": "", "lift@10%": ""})
            continue
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_va, blocks["valid"][1])
        raw_te = booster.predict(blocks["test"][0], num_iteration=booster.best_iteration)
        m = churn.evaluate(blocks["test"][1], raw_te, shares)
        booster.save_model(str(out_dir / f"{prod}_lgbm.txt"), num_iteration=booster.best_iteration)
        joblib.dump(iso, out_dir / f"{prod}_calibrator.joblib")
        entry.update({"status": "model", "best_iteration": int(booster.best_iteration), "test": m,
                      "test_rows": int(len(blocks["test"][1])), "test_positives": int(blocks["test"][1].sum())})
        status[prod] = entry
        rows_out.append({"product": prod, "train_rows": entry["train_rows"], "train_positives": entry["train_positives"],
                         "status": f"model ({booster.best_iteration} rounds, {time.perf_counter() - t0:.0f}s)",
                         "PR-AUC": round(m["pr_auc"], 3), "ROC-AUC": round(m["roc_auc"], 3),
                         "capture@10%": round(100 * m["capture_top10"], 1), "lift@10%": round(m["lift_top10"], 2)})
    (out_dir / "propensity.json").write_text(json.dumps(status, indent=1, default=str), encoding="utf-8")
    table = pd.DataFrame(rows_out)
    log(table.to_string(index=False))
    return {"status": status, "table": table}


def load_propensity(model_dir: Path) -> dict:
    out_dir = Path(model_dir) / SUBDIR
    status = json.loads((out_dir / "propensity.json").read_text(encoding="utf-8"))
    models = {}
    for prod, entry in status.items():
        if entry["status"] == "model":
            models[prod] = (lgb.Booster(model_file=str(out_dir / f"{prod}_lgbm.txt")),
                            joblib.load(out_dir / f"{prod}_calibrator.joblib"))
    return {"status": status, "models": models}


def predict_propensity(rows: pd.DataFrame, loaded: dict) -> pd.DataFrame:
    names = feature_names()
    out = pd.DataFrame(index=rows.index)
    for prod, (booster, iso) in loaded["models"].items():
        out[prod] = iso.predict(booster.predict(rows[names]))
    return out


# --------------------------------------------------------------------------- one recommendation per customer
def recommend(rows: pd.DataFrame, products: pd.DataFrame, settings: dict, model_dir: Path,
              value_at_risk: np.ndarray | None = None, rules: dict | None = None, root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Columns: recommendation, rec_score, rec_reason, rec_source, rec_trigger. One row per input row."""
    cfg = settings["crosssell"]
    rules = rules or load_rules(root, settings)
    rows = rows.reset_index(drop=True)
    prods = product_list(products, settings)
    held = products_held(rows, products, prods)
    loaded = load_propensity(model_dir)
    probs = predict_propensity(rows, loaded)
    fired = fire_triggers(rows, rules, held)
    base = {p: float(e["base_rate_train"]) for p, e in loaded["status"].items()}
    lift_min = float(cfg["model_lift_min"])

    n = len(rows)
    rec = pd.DataFrame({"recommendation": "none", "rec_score": 0.0, "rec_reason": "", "rec_source": "none",
                        "rec_trigger": ""}, index=range(n))
    # 1. rules first; among several, the higher model probability wins, then priority
    if len(fired):
        fired = fired.copy()
        fired["score"] = [float(probs.at[r, p]) if p in probs.columns else base.get(p, 0.0)
                          for r, p in zip(fired["row"], fired["product"])]
        fired = fired.sort_values(["row", "score", "priority"], ascending=[True, False, True]).drop_duplicates("row")
        for r in fired.itertuples(index=False):
            rec.loc[r.row, ["recommendation", "rec_score", "rec_reason", "rec_trigger"]] = [r.product, r.score, r.reason, r.trigger]
            rec.at[r.row, "rec_source"] = "rule+model" if r.product in probs.columns else "rule"
    # 2. model-only where confident enough
    todo = rec.index[rec["rec_source"] == "none"]
    if len(todo) and len(probs.columns):
        p = probs.loc[todo].copy()
        for prod in p.columns:
            p.loc[held.loc[todo, prod].to_numpy(), prod] = -1.0                 # never offer what is held
            p[prod] = np.where(p[prod] >= lift_min * base.get(prod, 1.0), p[prod], -1.0)
        best = p.idxmax(axis=1)
        best_p = p.max(axis=1)
        for i, prod, pv in zip(todo, best, best_p):
            if pv > 0:
                template = rules.get("model_reasons", {}).get(prod, "Banking pattern resembles customers who took {product} (model: {score:.0%}).")
                rec.loc[i, ["recommendation", "rec_score", "rec_reason", "rec_source"]] = [
                    prod, float(pv), fill(template, _row_values(rows, i, {"score": pv, "product": prod})), "model"]
    # 3. fallback: first product in the list not already held
    todo = rec.index[rec["rec_source"] == "none"]
    for i in todo:
        for fb in rules.get("fallback", []):
            if fb["product"] in held.columns and not held.at[i, fb["product"]]:
                rec.loc[i, ["recommendation", "rec_score", "rec_reason", "rec_source"]] = [
                    fb["product"], 0.0, fill(fb["reason"], _row_values(rows, i)), "fallback"]
                break
    # 4. guardrail: top share of value at risk gets a retention action instead of any offer
    if value_at_risk is not None:
        var = np.asarray(value_at_risk, float)
        k = max(1, int(np.floor(float(cfg["retention_top_share"]) * n)))
        top = np.argsort(-var, kind="stable")[:k]
        for i in top:
            rec.loc[i, ["recommendation", "rec_score", "rec_reason", "rec_source", "rec_trigger"]] = [
                "retention", float(var[i]), fill(rules["retention"]["action"], _row_values(rows, i, {"value_at_risk_usd": var[i]})),
                "retention", ""]
    return rec


# --------------------------------------------------------------------------- report
def trigger_report(latest: pd.DataFrame, test: pd.DataFrame, products: pd.DataFrame, rules: dict, settings: dict) -> pd.DataFrame:
    """Per trigger: customers it fires for at the latest snapshot, and on the test block the 90-day
    uptake rate of its product among triggered versus untriggered customers (neither holding it)."""
    horizon = int(settings["crosssell"]["horizon_days"])
    prods = sorted({t["product"] for t in rules["triggers"]})
    held_l = products_held(latest, products, prods)
    held_t = products_held(test, products, prods)
    up_t = uptake_within(test, products, prods, horizon)
    rows = []
    for t in rules["triggers"]:
        prod = t["product"]
        fire_l = trigger_mask(latest, t) & ~held_l[prod].to_numpy()
        elig = ~held_t[prod].to_numpy()
        fire_t = trigger_mask(test, t) & elig
        up = up_t[prod].to_numpy()
        r_trig = up[fire_t].mean() if fire_t.sum() else float("nan")
        r_untrig = up[elig & ~fire_t].mean() if (elig & ~fire_t).sum() else float("nan")
        rows.append({"trigger": t["name"], "product": prod, "fires_latest": int(fire_l.sum()),
                     "fires_latest_%": round(100 * fire_l.mean(), 1), "test_triggered": int(fire_t.sum()),
                     "uptake_triggered_%": round(100 * r_trig, 2), "uptake_untriggered_%": round(100 * r_untrig, 2),
                     "lift": round(r_trig / r_untrig, 1) if r_untrig else float("nan")})
    return pd.DataFrame(rows)


def main(root: Path = PROJECT_ROOT) -> dict:
    settings = load_settings(root)
    t0 = time.perf_counter()
    feats = read_parquet(resolve(root, settings["data"]["processed_dir"]) / "features.parquet")
    feats["snapshot_date"] = pd.to_datetime(feats["snapshot_date"])
    products = load_products(root, settings)
    rules = load_rules(root, settings)
    model_dir = resolve(root, churn.MODEL_DIR)
    md = []

    print("\n## Propensity models per product (test block; rule only where uptakes are too few)")
    res = train_propensity(feats, products, settings, model_dir, log=print)
    md += ["\n**Propensity models per product (test block; rule only where uptakes are too few)**\n", churn.to_markdown(res["table"])]

    split = churn.time_split(feats, settings["model"], int(settings["target"]["horizon_days"]))
    latest = feats[feats["snapshot_date"] == feats["snapshot_date"].max()].reset_index(drop=True)
    rep = trigger_report(latest, split.test, products, rules, settings)
    print(f"\n## Triggers: fires at the latest snapshot ({latest['snapshot_date'].iloc[0].date()}, {len(latest):,} customers) "
          f"and uptake on the test block\n" + rep.to_string(index=False))
    md += [f"\n**Triggers: fires at the latest snapshot ({latest['snapshot_date'].iloc[0].date()}) and 90-day uptake on the test block**\n",
           churn.to_markdown(rep)]

    (model_dir / "crosssell_report.md").write_text("\n".join(md), encoding="utf-8")
    (model_dir / SUBDIR / "trigger_report.json").write_text(rep.to_json(orient="records", indent=1), encoding="utf-8")
    print(f"\nsaved models/crosssell/ and crosssell_report.md in {time.perf_counter() - t0:.1f}s")
    return {"propensity": res["status"], "triggers": rep}


if __name__ == "__main__":
    main()
