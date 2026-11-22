"""Reason codes, action cards and the segment view (Phase 5).

SHAP values come from LightGBM's built-in TreeSHAP, ``Booster.predict(pred_contrib=True)``. It is
the same algorithm ``shap.TreeExplainer`` runs on a LightGBM model and returns the same numbers;
the ``shap`` package itself cannot load on the build machine (a compiled dependency is blocked by
Windows App Control). For one customer the model's raw score (log-odds) is

    base value + sum of one contribution per feature

so a positive contribution pushed this customer's score towards churn. The three largest positive
contributions that have a template in config/reasons.yaml become the reasons on the card; the
family of the first one picks the action and call script in config/actions.yaml. Every sentence
shown is one of those fixed templates filled with the customer's own numbers (rule 8).

Runs inside ``python -m pulse.cli score``; ``python -m pulse.explain`` rebuilds the cards from the
saved worklist alone.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pulse.config import PROJECT_ROOT, current_mode, load_settings, load_yaml, resolve
from pulse.crosssell import OPS, fill, products_held
from pulse.io import read_parquet, write_parquet
from pulse.models import churn
from pulse.models.survival import load_products
from pulse.pseudo import pseudonymise

# the only derived placeholders templates may use, besides the features themselves (see reasons.yaml)
PRIOR_ALIASES = {
    "wallet_share_prior_3m": ("wallet_share_3m", "wallet_share_trend"),
    "obank_share_prior_3m": ("obank_share_3m", "obank_share_trend"),
    "leak_share_prior_3m": ("leak_share_3m", "leak_share_trend"),
}
CARD_FIELDS = ["customer_ref", "rank", "segment", "branch", "p_churn_90d", "urgency", "value_12m_usd",
               "value_at_risk_usd"]
SEGMENT_DIMS = ["branch", "segment", "product"]


# ----------------------------------------------------------------------------- config
def _check_conditions(where: str, conds: list) -> None:
    for c in conds:
        if len(c) != 3 or c[1] not in OPS:
            raise ValueError(f"{where}: bad condition {c!r} (expect [feature, operator, value])")


def variants(spec: dict) -> list[dict]:
    """A feature's templates in order of preference: explicit `variants`, else the single when/template."""
    if "variants" in spec:
        return list(spec["variants"])
    return [{"when": spec.get("when", []), "template": spec["template"]}]


def load_reasons(root: Path, settings: dict, mode: str = "transactions") -> dict:
    key = "reasons_path_summary" if mode == "summary" else "reasons_path"
    cfg = load_yaml(resolve(root, settings["explain"][key]))
    cfg.setdefault("features", {})
    if cfg.get("generic"):
        if cfg["generic"]["family"] not in cfg["families"] or not cfg["generic"].get("template"):
            raise ValueError("generic reason needs a known family and a template")
    for name, spec in cfg["features"].items():
        if spec["family"] not in cfg["families"]:
            raise ValueError(f"reason {name}: unknown family {spec['family']!r}")
        for v in variants(spec):
            if not v.get("template"):
                raise ValueError(f"reason {name}: a variant has no template")
            _check_conditions(f"reason {name}", v.get("when", []))
    return cfg


def load_actions(root: Path, settings: dict) -> dict:
    cfg = load_yaml(resolve(root, settings["explain"]["actions_path"]))
    for fam, block in cfg["families"].items():
        if not block.get("action") or not block.get("script"):
            raise ValueError(f"action {fam}: needs 'action' and 'script'")
    return cfg


# ----------------------------------------------------------------------------- SHAP
def contributions(model_dir: Path, rows: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-feature SHAP contributions (log-odds) and the base value, one row per input row."""
    booster, _, names = churn.load_model(model_dir)
    c = booster.predict(rows[names], pred_contrib=True)
    return pd.DataFrame(c[:, :-1], columns=names, index=rows.index), c[:, -1]


def raw_score(model_dir: Path, rows: pd.DataFrame) -> np.ndarray:
    booster, _, names = churn.load_model(model_dir)
    return booster.predict(rows[names], raw_score=True)


def global_summary(contrib: pd.DataFrame, reasons: dict, n: int = 15) -> pd.DataFrame:
    """Top features by mean |SHAP| over the rows given, with how often each pushes a score up."""
    fam = {f: s["family"] for f, s in reasons["features"].items()}
    generic = reasons.get("generic")
    if generic:
        fam = {c: fam.get(c, generic["family"]) for c in contrib.columns}
    df = pd.DataFrame({
        "feature": contrib.columns,
        "mean_abs_shap": contrib.abs().mean().to_numpy(),
        "pushes_up_%": 100 * (contrib > 0).mean().to_numpy(),
    })
    df["share_%"] = 100 * df["mean_abs_shap"] / df["mean_abs_shap"].sum()
    df["family"] = [fam.get(f, "(no template)") for f in df["feature"]]
    df = df.sort_values("mean_abs_shap", ascending=False).head(n).reset_index(drop=True)
    return df.round({"mean_abs_shap": 4, "pushes_up_%": 1, "share_%": 1})


# ----------------------------------------------------------------------------- reasons
def aliases(values: dict) -> dict:
    """Add the fixed derived placeholders documented in reasons.yaml."""
    out = dict(values)
    for k, v in values.items():
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            out[f"{k}_pct"] = 100.0 * float(v)
            out[f"{k}_drop_pct"] = 100.0 * (1.0 - float(v))
            out[f"{k}_abs_pct"] = 100.0 * abs(float(v))
    for alias, (share, trend) in PRIOR_ALIASES.items():
        if share in values and trend in values:
            out[alias] = float(values[share]) - float(values[trend])
    return out


def holds(values: dict, conds: list) -> bool:
    return all(f in values and bool(OPS[op](values[f], v)) for f, op, v in conds)


def pick_reasons(values: dict, contrib_row: pd.Series, reasons: dict, n: int = 3,
                 typical: dict | None = None) -> list[dict]:
    """Top-n positive contributions with a template, one per family, conditions satisfied.

    `typical` (fallback mode) maps feature -> median among customers who stayed; it fills the
    {typical} placeholder and lets the `generic` template describe an unmapped column."""
    specs, labels = reasons["features"], reasons["families"]
    generic = reasons.get("generic")
    typical = typical or {}
    out, used = [], set()
    for feature, s in contrib_row.sort_values(ascending=False).items():
        if s <= 0 or len(out) >= n:
            break
        spec = specs.get(feature)
        if spec is None and generic and feature in values and not feature.endswith("_missing"):
            spec = generic
        if spec is None or spec["family"] in used:
            continue
        vals = {**values, "label": feature.replace("_", " ").capitalize(), "value": values.get(feature),
                "typical": typical.get(feature, float("nan"))}
        template = next((v["template"] for v in variants(spec) if holds(vals, v.get("when", []))), None)
        if template is None:
            continue
        out.append({"feature": feature, "family": spec["family"], "family_label": labels[spec["family"]],
                    "contribution": round(float(s), 4), "text": fill(template, vals)})
        used.add(spec["family"])
    return out


# ----------------------------------------------------------------------------- cards
def align_rows(worklist: pd.DataFrame, rows: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Feature rows in worklist order, matched on the pseudonymised reference."""
    pcfg = settings["pseudonymise"]
    rows = rows.copy()
    rows["customer_ref"] = pseudonymise(rows["customer_id"], pcfg["salt"], int(pcfg["length"]))
    rows = rows.drop_duplicates("customer_ref").set_index("customer_ref")
    missing = set(worklist["customer_ref"]) - set(rows.index)
    if missing:
        raise ValueError(f"{len(missing)} worklist customers have no feature row at this snapshot")
    return rows.loc[worklist["customer_ref"]].reset_index()


def typical_values(rows: pd.DataFrame, names: list[str]) -> dict:
    """Median of each model input among customers who stayed (fallback mode card sentences)."""
    stay = rows[rows["churn_90d"] == 0] if "churn_90d" in rows.columns and (rows["churn_90d"] == 0).any() else rows
    return {c: float(stay[c].median()) for c in names if c in stay.columns}


def build_cards(worklist: pd.DataFrame, rows: pd.DataFrame, contrib: pd.DataFrame, reasons: dict,
                actions: dict, n_reasons: int = 3, typical: dict | None = None) -> list[dict]:
    """One card per worklist row; rows and contrib must be aligned with the worklist (same order)."""
    cards = []
    feature_cols = [c for c in rows.columns if c not in ("customer_id", "customer_ref")]
    row_records = rows[feature_cols].to_dict("records")
    work_records = worklist.to_dict("records")
    names, cvals = list(contrib.columns), contrib.to_numpy()
    for i, w in enumerate(work_records):
        values = aliases({**row_records[i],
                          **{c: w[c] for c in ("value_at_risk_usd", "value_12m_usd", "p_churn_90d",
                                               "urgency", "recommendation")}})
        picked = pick_reasons(values, pd.Series(cvals[i], index=names), reasons, n_reasons, typical)
        family = picked[0]["family"] if picked else None
        block = actions["families"].get(family, actions["default"])
        card = {f: (w[f].item() if hasattr(w[f], "item") else w[f]) for f in CARD_FIELDS}
        card["snapshot_date"] = str(pd.Timestamp(w["snapshot_date"]).date())
        card["reasons"] = picked
        card["action_family"] = family or "default"
        card["action"] = fill(block["action"], values)
        card["call_script"] = fill(block["script"], values)
        card["next_best_product"] = {"product": w["recommendation"], "reason": w["rec_reason"],
                                     "source": w["rec_source"], "score": float(w["rec_score"])}
        cards.append(card)
    return cards


# ----------------------------------------------------------------------------- segments
def segment_table(worklist: pd.DataFrame, rows: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Churn risk and value at risk by branch, segment and product held at the snapshot."""
    names = sorted(products["product"].dropna().unique()) if len(products) else []
    held = products_held(rows, products, names) if names else None
    frames = []
    for dim in SEGMENT_DIMS:
        if dim == "product":
            if not names:
                continue
            long = pd.concat([worklist.loc[held[p].to_numpy()].assign(group=p) for p in names], ignore_index=True)
        else:
            long = worklist.assign(group=worklist[dim])
        g = long.groupby("group", sort=True)
        t = pd.DataFrame({
            "dimension": dim,
            "customers": g.size(),
            "churn_rate_%": 100 * g["p_churn_90d"].mean(),
            "expected_churners": g["p_churn_90d"].sum(),
            "value_12m_usd": g["value_12m_usd"].sum(),
            "value_at_risk_usd": g["value_at_risk_usd"].sum(),
        }).reset_index()
        t["share_of_var_%"] = 100 * t["value_at_risk_usd"] / worklist["value_at_risk_usd"].sum()
        t["var_per_customer_usd"] = t["value_at_risk_usd"] / t["customers"]
        frames.append(t)
    out = pd.concat(frames, ignore_index=True)[["dimension", "group", "customers", "churn_rate_%", "expected_churners",
                                                 "value_12m_usd", "value_at_risk_usd", "share_of_var_%",
                                                 "var_per_customer_usd"]]
    return out.round({"churn_rate_%": 2, "expected_churners": 1, "value_12m_usd": 0, "value_at_risk_usd": 0,
                      "share_of_var_%": 1, "var_per_customer_usd": 2})


# ----------------------------------------------------------------------------- outputs
def card_text(card: dict) -> str:
    """Plain-text rendering of one card (used by the CLI printout and the inspection script)."""
    lines = [f"#{card['rank']}  {card['customer_ref']}  {card['segment']} / {card['branch']}  "
             f"risk {card['p_churn_90d']:.0%}  {card['urgency']}  value at risk {card['value_at_risk_usd']:,.0f} USD "
             f"of {card['value_12m_usd']:,.0f}"]
    for j, r in enumerate(card["reasons"], 1):
        lines.append(f"  {j}. [{r['family_label']}] {r['text']}")
    if not card["reasons"]:
        lines.append("  (no reason with a template; default action)")
    lines.append(f"  Action: {card['action']}")
    lines.append(f"  Script: {card['call_script']}")
    nbp = card["next_best_product"]
    lines.append(f"  Next best product: {nbp['product']} ({nbp['source']}) - {nbp['reason']}")
    return "\n".join(lines)


def explain_snapshot(worklist: pd.DataFrame, rows: pd.DataFrame, settings: dict, model_dir: Path,
                     root: Path = PROJECT_ROOT, mode: str = "transactions") -> dict:
    """Cards, contributions and the global summary for one worklist, nothing written."""
    ecfg = settings["explain"]
    reasons, actions = load_reasons(root, settings, mode), load_actions(root, settings)
    rows = align_rows(worklist, rows, settings)
    contrib, base = contributions(Path(model_dir), rows)
    typical = typical_values(rows, list(contrib.columns)) if mode == "summary" else None
    cards = build_cards(worklist, rows, contrib, reasons, actions, int(ecfg["n_reasons"]), typical)
    summary = global_summary(contrib, reasons, int(ecfg["summary_top"]))
    return {"cards": cards, "contrib": contrib, "base": base, "summary": summary, "rows": rows}


def run(worklist: pd.DataFrame, rows: pd.DataFrame, products: pd.DataFrame, settings: dict, model_dir: Path,
        processed_dir: Path, root: Path = PROJECT_ROOT, log=print, mode: str | None = None) -> dict:
    """Build and write cards.json, shap_latest.parquet, segments.parquet/.csv and the SHAP summary."""
    t0 = time.perf_counter()
    model_dir, processed_dir = Path(model_dir), Path(processed_dir)
    mode = mode or current_mode(root)
    res = explain_snapshot(worklist, rows, settings, model_dir, root, mode)
    cards, contrib, base, summary, rows = res["cards"], res["contrib"], res["base"], res["summary"], res["rows"]
    segments = segment_table(worklist, rows, products)

    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "cards.json").write_text(json.dumps(cards, indent=1, ensure_ascii=False), encoding="utf-8")
    shap_out = contrib.copy()
    shap_out.insert(0, "customer_ref", worklist["customer_ref"].to_numpy())
    shap_out.insert(1, "base_value", base)
    write_parquet(shap_out, processed_dir / "shap_latest.parquet")
    write_parquet(segments, processed_dir / "segments.parquet")
    segments.to_csv(processed_dir / "segments.csv", index=False)
    model_dir.mkdir(parents=True, exist_ok=True)
    summary.to_json(model_dir / "shap_summary.json", orient="records", indent=1)
    (model_dir / "shap_summary.md").write_text(
        f"# Global SHAP summary, {len(contrib):,} customers at {cards[0]['snapshot_date']}\n\n"
        f"Base value (log-odds) {base[0]:.3f}. Mean |SHAP| is the average size of a feature's push, in "
        f"either direction; 'pushes up' is the share of customers it pushes towards churn.\n\n"
        + churn.to_markdown(summary) + "\n", encoding="utf-8")

    n_reasons = pd.Series([len(c["reasons"]) for c in cards])
    fam = pd.Series([c["action_family"] for c in cards])
    log(f"cards.json: {len(cards):,} cards; reasons per card: " +
        ", ".join(f"{k}: {v:,}" for k, v in n_reasons.value_counts().sort_index(ascending=False).items()))
    log("action family\n" + fam.value_counts().to_string())
    log("\nglobal SHAP summary (top " + str(len(summary)) + ")\n" + summary.to_string(index=False))
    top = segments.sort_values("value_at_risk_usd", ascending=False).head(8)
    log("\nsegments with the most value at risk\n" + top.to_string(index=False))
    log(f"\nexplained in {time.perf_counter() - t0:.1f}s")
    return {"cards": cards, "summary": summary, "segments": segments, "contrib": contrib, "base": base}


def main(root: Path = PROJECT_ROOT) -> dict:
    settings = load_settings(root)
    processed = resolve(root, settings["data"]["processed_dir"])
    worklist = read_parquet(processed / "worklist.parquet")
    feats = read_parquet(processed / "features.parquet")
    feats["snapshot_date"] = pd.to_datetime(feats["snapshot_date"])
    snap = pd.Timestamp(worklist["snapshot_date"].iloc[0])
    rows = feats[feats["snapshot_date"] == snap]
    products = load_products(root, settings)
    return run(worklist, rows, products, settings, resolve(root, churn.MODEL_DIR), processed, root)


if __name__ == "__main__":
    main()
