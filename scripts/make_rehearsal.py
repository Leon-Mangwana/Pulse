"""Generate the two rehearsal datasets into data/rehearsal/ (git-ignored).

    python scripts/make_rehearsal.py [--out data/rehearsal] [--seed 2026]

DO NOT READ THE REST OF THIS FILE BEFORE THE REHEARSALS. The point of the exercise is to go from
unfamiliar files to a working demo using only scripts/profile_data.py, schema.yaml and the
runbook. Dataset (a) is in data/rehearsal/a/, dataset (b) in data/rehearsal/b/. No answer key
is written next to them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pulse import synth  # noqa: E402
from pulse.config import load_settings  # noqa: E402

# ----------------------------------------------------------------------------- dataset (a)
ACCT_TYPE = {"current": "CUR", "savings": "SAV"}
CCY = {"USD": "USD", "ZiG": "ZWG"}
SEG_CODE = {"salaried": "SAL", "sme": "SME", "informal": "INF", "diaspora": "DIA"}
CHAN = {"mobile app": "APP", "USSD": "USSD", "ZIPIT": "ZIP", "RTGS": "RTGS", "POS": "POS", "ATM": "ATM", "branch": "BR"}
CPTY = {"mobile_wallet": "WALLET", "merchant": "MERCH", "fee": "FEE", "cash": "CASH", "other_bank": "OBANK",
        "supplier": "SUPP", "self": "OWN", "employer": "EMPL", "remittance": "REMIT", "school": "SCHOOL"}
PRODUCT_NAMES = {"transactional_account": "Current Account", "mobile_banking": "Mobile App",
                 "savings_account": "Save Plus", "education_savings": "School Fees Plan", "fx_savings": "USD Saver",
                 "fixed_deposit": "Term Deposit", "personal_loan": "Salary Loan", "sme_package": "Business Bundle"}
TITLES = ["Mr", "Mrs", "Ms", "Dr", ""]


def blank(rng: np.random.Generator, s: pd.Series, share: float) -> pd.Series:
    s = s.copy().astype(object)
    s[rng.random(len(s)) < share] = ""
    return s


def dataset_a(settings: dict, seed: int, out: Path) -> None:
    frames = synth.generate(settings, n_customers=3000, seed=seed)
    rng = np.random.default_rng(seed + 1)
    cust, acct, tx, prods = frames["customers"], frames["accounts"], frames["transactions"], frames["products"]
    end = pd.Timestamp(settings["data"]["end_date"])

    # one row per account, customer fields repeated (merged table)
    m = acct.merge(cust, on="customer_id", suffixes=("", "_cust"))
    clients = pd.DataFrame({
        "ClientNo": m["customer_id"].str.replace("C", "ZB", regex=False),
        "Title": rng.choice(TITLES, len(m)),
        "AcctNo": m["account_id"].str.replace("A", "40", regex=False),
        "AcctType": m["product_type"].map(ACCT_TYPE),
        "Ccy": m["currency"].map(CCY),
        "DateJoined": pd.to_datetime(m["open_date_cust"]).dt.strftime("%d/%m/%Y"),
        "AcctOpened": pd.to_datetime(m["open_date"]).dt.strftime("%d/%m/%Y"),
        "AcctClosed": pd.to_datetime(m["close_date"]).dt.strftime("%d/%m/%Y").fillna(""),
        "BranchName": blank(rng, m["branch"], 0.03),
        "SegmentCode": blank(rng, m["segment"].map(SEG_CODE), 0.02),
        "Sex": m["gender"],
        "AgeGroup": m["age_band"],
        "Mobile": ["07" + "".join(rng.choice(list("0123456789"), 2)) + "***" + "".join(rng.choice(list("0123456789"), 3)) for _ in range(len(m))],
    }).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    t = tx.copy()
    t["txn_date"] = pd.to_datetime(t["txn_date"])
    txns = pd.DataFrame({
        "RefNo": t["txn_id"].str.replace("T", "TX", regex=False),
        "AcctNo": t["account_id"].str.replace("A", "40", regex=False),
        "ClientNo": t["customer_id"].str.replace("C", "ZB", regex=False),
        "PostDate": t["txn_date"].dt.strftime("%Y%m%d"),
        "Amt": t["amount"].map(lambda v: f"{v:,.2f}"),
        "DrCr": t["direction"].map({"credit": "C", "debit": "D"}),
        "Ccy": t["currency"].map(CCY),
        "Chan": t["channel"].map(CHAN),
        "CptyType": t["counterparty_type"].map(CPTY),
        "Narrative": blank(rng, t["description"], 0.01),
        "RunningBal": t["balance_after"].round(2),
    })
    dup = txns.sample(60, random_state=seed)                      # a few exact duplicate postings
    txns = pd.concat([txns, dup], ignore_index=True).sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)

    p = prods.copy()
    holdings = pd.DataFrame({
        "ClientNo": p["customer_id"].str.replace("C", "ZB", regex=False),
        "ProductName": p["product"].map(PRODUCT_NAMES),
        "StartDate": pd.to_datetime(p["start_date"]).dt.strftime("%Y-%m-%d"),
        "EndDate": pd.to_datetime(p["end_date"]).dt.strftime("%Y-%m-%d").fillna(""),
    })
    extra_ids = cust["customer_id"].sample(frac=0.05, random_state=seed)
    extra = pd.DataFrame({
        "ClientNo": extra_ids.str.replace("C", "ZB", regex=False).to_numpy(),
        "ProductName": "Funeral Cover",
        "StartDate": (end - pd.to_timedelta(rng.integers(30, 600, len(extra_ids)), unit="D")).strftime("%Y-%m-%d"),
        "EndDate": "",
    })
    holdings = pd.concat([holdings, extra], ignore_index=True).sample(frac=1.0, random_state=seed + 3).reset_index(drop=True)

    out.mkdir(parents=True, exist_ok=True)
    clients.to_csv(out / "clients.csv", index=False)
    txns.to_csv(out / "txns.csv", index=False)
    holdings.to_csv(out / "holdings.csv", index=False)
    (out / "README.txt").write_text(
        "Extract from the core banking system, three files. Period: Nov 2024 to Oct 2026.\n"
        "clients.csv: one row per account with the client's details. txns.csv: postings.\n"
        "holdings.csv: products held with start and end dates.\n", encoding="utf-8")


# ----------------------------------------------------------------------------- dataset (b)
def dataset_b(settings: dict, seed: int, out: Path) -> None:
    frames = synth.generate(settings, n_customers=4000, seed=seed + 50)
    rng = np.random.default_rng(seed + 51)
    cust, tx, prods, truth = frames["customers"], frames["transactions"], frames["products"], frames["_truth"]
    end = pd.Timestamp(settings["data"]["end_date"])
    tx = tx.copy()
    tx["txn_date"] = pd.to_datetime(tx["txn_date"])
    last12 = tx[tx["txn_date"] > end - pd.DateOffset(months=12)]
    last6 = tx[tx["txn_date"] > end - pd.DateOffset(months=6)]
    last90 = tx[tx["txn_date"] > end - pd.Timedelta(days=90)]

    g = cust.set_index("customer_id")
    avg_bal = last6.groupby("customer_id")["balance_after"].mean()
    n90 = last90.groupby("customer_id").size()
    last_txn = tx.groupby("customer_id")["txn_date"].max()
    digital = last12.assign(d=last12["channel"].isin(["mobile app", "USSD", "ZIPIT", "RTGS"])).groupby("customer_id")["d"].mean()
    fees = last12[last12["counterparty_type"] == "fee"].groupby("customer_id")["amount"].sum()
    held = prods[(pd.to_datetime(prods["start_date"]) <= end) & (prods["end_date"].isna() | (pd.to_datetime(prods["end_date"]) > end))]
    n_prod = held.groupby("customer_id").size()
    churned = truth.set_index("customer_id")["churner"].reindex(g.index).fillna(False).astype(bool)

    df = pd.DataFrame({
        "cust_id": ["ZB-" + c[1:] for c in g.index],
        "as_of_date": end.strftime("%d-%b-%Y"),
        "months_with_bank": ((end - pd.to_datetime(g["open_date"])).dt.days / 30.4).round().astype(int).to_numpy(),
        "avg_balance": avg_bal.reindex(g.index).round(2).to_numpy(),
        "num_products": n_prod.reindex(g.index).fillna(0).astype(int).to_numpy(),
        "is_active": np.where(n90.reindex(g.index).fillna(0).to_numpy() > 0, "Y", "N"),
        "txns_90d": n90.reindex(g.index).fillna(0).astype(int).to_numpy(),
        "days_since_last_txn": (end - last_txn.reindex(g.index)).dt.days.to_numpy(),
        "digital_pct": (100 * digital.reindex(g.index)).round(1).to_numpy(),
        "fees_12m": fees.reindex(g.index).fillna(0).round(2).to_numpy(),
        "complaints_12m": rng.poisson(np.where(churned.to_numpy(), 0.9, 0.15)).astype(float),
        "segment_code": g["segment"].map(SEG_CODE).to_numpy(),
        "branch": g["branch"].to_numpy(),
        "gender": g["gender"].to_numpy(),
        "age_bracket": g["age_band"].to_numpy(),
        "status": np.where(churned.to_numpy(), "Churned", "Active"),
    })
    df.loc[rng.random(len(df)) < 0.04, "avg_balance"] = np.nan
    df.loc[rng.random(len(df)) < 0.03, "complaints_12m"] = np.nan
    df.loc[rng.random(len(df)) < 0.02, "digital_pct"] = np.nan
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "customer_base.csv", index=False)
    (out / "README.txt").write_text(
        "Customer base extract as of month end, one row per customer, with the retention team's status flag.\n"
        "No transaction-level data is available for this exercise.\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="write the two rehearsal datasets")
    ap.add_argument("--out", default="data/rehearsal")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(argv)
    settings = load_settings()
    out = Path(args.out)
    dataset_a(settings, args.seed, out / "a")
    dataset_b(settings, args.seed, out / "b")
    for d in ("a", "b"):
        files = sorted((out / d).glob("*"))
        print(f"{out / d}: " + ", ".join(f"{f.name} ({f.stat().st_size / 1e6:.1f} MB)" for f in files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
