"""Synthetic BancABC-style bank data.

Why synthetic: we have not seen the real data. A generator lets us plant known
behaviours (salary stopping, wallet leakage, school fees) and later check that
the pipeline rediscovers them. The hidden answers go to data/raw/_truth.csv,
which the pipeline must never read.

How it works, in one paragraph: every customer gets a persona (segment, salary,
life events) and a hidden fate (churn or not, with or without warning signs).
From these we build a MONTHLY PLAN: for each customer and month, how much money
comes in, what share goes out, and where it goes. The plan is expanded into
individual dated transactions with statement-style descriptions and a running
balance. Noise is added at every level so no single rule separates churners.

Run:  python -m pulse.cli data
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pulse.config import PROJECT_ROOT, load_settings, resolve

# --------------------------------------------------------------------------
# Reference lists (purely cosmetic: they make statements look real)
# --------------------------------------------------------------------------
SEGMENTS = ["salaried", "sme", "informal", "diaspora"]

BRANCHES = ["Harare CBD", "Borrowdale", "Avondale", "Bulawayo", "Mutare", "Gweru",
            "Masvingo", "Kwekwe", "Chinhoyi", "Victoria Falls"]
BRANCH_W = [0.22, 0.12, 0.10, 0.16, 0.10, 0.08, 0.07, 0.05, 0.05, 0.05]
BRANCH_CITY = ["HARARE", "HARARE", "HARARE", "BULAWAYO", "MUTARE", "GWERU",
               "MASVINGO", "KWEKWE", "CHINHOYI", "VIC FALLS"]

EMPLOYERS = ["DELTA BEVERAGES", "MIN OF PRIMARY EDU", "ECONET WIRELESS", "ZESA HOLDINGS",
             "OK ZIMBABWE", "ZIMPLATS", "CITY OF HARARE", "PSMAS", "NATIONAL FOODS",
             "UNIVERSITY OF ZIM", "SIMBISA BRANDS", "CIMAS", "TELONE", "NRZ",
             "INNSCOR AFRICA", "SEED CO", "MIMOSA MINING", "ZIMRA", "OLD MUTUAL", "ZINARA"]
WALLETS = ["ECOCASH", "ONEMONEY", "INNBUCKS", "OMARI"]
WALLET_W = [0.70, 0.10, 0.12, 0.08]
BANKS = ["CBZ", "FBC", "STANBIC", "NMB", "ZB BANK", "STEWARD", "FIRST CAPITAL", "NEDBANK"]
MERCHANTS = ["OK ZIMBABWE", "PICK N PAY", "TM SUPERMARKET", "SPAR", "ZUVA FUEL",
             "TOTAL ENERGIES", "CHICKEN INN", "EDGARS", "FOOD LOVERS", "GREENWOOD PHARMACY",
             "BON MARCHE", "NANDOS", "PUMA FUEL", "CHOPPIES", "MEIKLES"]
UTILITIES = ["ECONET AIRTIME", "ZESA PREPAID", "NETONE AIRTIME", "DSTV", "ZOL FIBRE", "HARARE WATER"]
SCHOOLS = ["GATEWAY HIGH", "PRINCE EDWARD SCH", "DOMINICAN CONVENT", "UZ FEES", "MSU FEES",
           "HARARE INTL SCH", "CHISIPITE SENIOR", "PETRA HIGH", "ST GEORGES", "NUST FEES"]
SUPPLIERS = ["NATIONAL FOODS", "LOBELS BREAD", "DAIRIBORD", "PROTON BAKERS", "UNITED REFINERIES",
             "CAIRNS FOODS", "DELTA DISTRIBUTION", "MEGA MARKET", "N RICHARDS", "GAIN CASH N CARRY"]
REMITTERS = ["MUKURU", "WESTERN UNION", "WORLDREMIT", "SENDITOO", "MONEYGRAM"]
REMITTER_W = [0.40, 0.20, 0.20, 0.10, 0.10]

AGE_BANDS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
AGE_P = {
    "salaried": [0.08, 0.32, 0.30, 0.18, 0.09, 0.03],
    "sme":      [0.03, 0.22, 0.33, 0.25, 0.12, 0.05],
    "informal": [0.15, 0.32, 0.26, 0.15, 0.08, 0.04],
    "diaspora": [0.10, 0.20, 0.22, 0.20, 0.15, 0.13],
}

# Persona parameters per segment, one row per segment in SEGMENTS order.
#   base_txn      typical transactions per month on the main account
#   leak_share    share of outflows that goes to mobile wallets + other banks
#   p_salary      chance of a regular monthly salary credit
#   salary_median median monthly salary, USD
#   p_usd         chance the main account is in USD (else ZiG)
#   p_remit       chance of regular diaspora remittances
#   p_school      chance of paying school fees each term
#   p_supplier    chance of regular supplier payments
#   p_rising      chance of a rising salary
#   biz_inflow    typical non-salary credits per month, USD (customer receipts, cash-ins)
#   biz_count     typical number of such credits per month
SEG = pd.DataFrame({
    "base_txn":      [8,    16,   10,   6],
    "leak_share":    [0.35, 0.30, 0.50, 0.30],
    "p_salary":      [0.97, 0.15, 0.05, 0.25],
    "salary_median": [450,  900,  300,  400],
    "p_usd":         [0.60, 0.70, 0.75, 0.90],
    "p_remit":       [0.06, 0.05, 0.08, 0.85],
    "p_school":      [0.35, 0.35, 0.20, 0.30],
    "p_supplier":    [0.02, 0.80, 0.20, 0.02],
    "p_rising":      [0.15, 0.05, 0.00, 0.00],
    "biz_inflow":    [40,   1500, 350,  60],
    "biz_count":     [0.5,  6,    5,    0.5],
}, index=SEGMENTS)

# Product uptake: life event -> (product, P(uptake | event), P(uptake | no event))
UPTAKE = {
    "has_remit":     ("fx_savings",        0.35, 0.04),
    "has_school":    ("education_savings", 0.30, 0.03),
    "has_supplier":  ("sme_package",       0.35, 0.03),
    "rising_salary": ("fixed_deposit",     0.30, 0.04),
}


def month_index(dates, start: pd.Timestamp) -> np.ndarray:
    """Months between the window start and each date (negative = before the window)."""
    d = pd.DatetimeIndex(dates)
    return ((d.year - start.year) * 12 + (d.month - start.month)).to_numpy()


# --------------------------------------------------------------------------
# 1. Customers
# --------------------------------------------------------------------------
def _customers(rng, n, start, M, seg_probs) -> pd.DataFrame:
    seg = rng.choice(SEGMENTS, n, p=[seg_probs[s] for s in SEGMENTS])
    age = np.empty(n, dtype=object)
    for s in SEGMENTS:
        mask = seg == s
        age[mask] = rng.choice(AGE_BANDS, mask.sum(), p=AGE_P[s])
    # 75% opened before the window (6 months to 10 years ago), 25% inside the first
    # 18 months of it, so some customers have short histories like in real data.
    before = rng.random(n) < 0.75
    offset = np.where(before, -rng.integers(180, 3650, n), rng.integers(0, 30 * (M - 6), n))
    return pd.DataFrame({
        "customer_id": [f"C{i:06d}" for i in range(1, n + 1)],
        "open_date": (start + pd.to_timedelta(offset, "D")).normalize(),
        "segment": seg,
        "branch": rng.choice(BRANCHES, n, p=BRANCH_W),
        "gender": rng.choice(["F", "M"], n, p=[0.48, 0.52]),
        "age_band": age,
    })


# --------------------------------------------------------------------------
# 2. Hidden truth: who churns, when, with which warning signs; life events
# --------------------------------------------------------------------------
def _truth(rng, cust, cfg, start, M) -> pd.DataFrame:
    n = len(cust)
    seg = cust["segment"].to_numpy()
    P = SEG.loc[seg].reset_index(drop=True)
    open_m = month_index(cust["open_date"], start)

    def r():
        return rng.random(n)

    # churn needs at least 6 months of history before it, so late joiners are less eligible
    first_ok = np.maximum(open_m + 6, 6)
    eligible = first_ok <= M - 1
    p_churn = min(cfg["churn_rate"] / max(eligible.mean(), 1e-9), 0.95)
    churner = (r() < p_churn) & eligible
    churn_month = np.where(churner, rng.integers(np.minimum(first_ok, M - 1), M), -1)
    churn_day = rng.integers(1, 29, n)
    month_starts = pd.date_range(start, periods=M, freq="MS")
    churn_date = pd.to_datetime(np.where(
        churner, month_starts[np.clip(churn_month, 0, M - 1)].to_numpy()
        + pd.to_timedelta(churn_day - 1, "D").to_numpy(), np.datetime64("NaT")))
    churn_type = np.where(churner, np.where(r() < cfg["closed_share_of_churn"], "closed", "silent"), "")
    lead = rng.integers(cfg["lead_months_min"], cfg["lead_months_max"] + 1, n)

    has_salary = r() < P["p_salary"].to_numpy()
    has_remit = r() < P["p_remit"].to_numpy()
    has_income = has_salary | has_remit

    # warning signs: each fires with its own probability, so churners differ from each other
    no_warn = churner & (r() < cfg["no_warning_share_of_churn"])
    warnable = churner & ~no_warn
    s_income = warnable & has_income & (r() < 0.70)
    s_drain = warnable & (r() < 0.70)
    s_leak = warnable & (r() < 0.60)
    s_act = warnable & (r() < 0.75)

    # false alarms: non-churners who show the same signs for 1-3 months and then recover
    fa = ~churner & (r() < cfg["false_alarm_rate"]) & (np.maximum(open_m + 6, 6) <= M - 4)
    fa_low = np.clip(np.maximum(open_m + 6, 6), 0, M - 4)
    fa_start = np.where(fa, rng.integers(fa_low, M - 3), -1)
    fa_len = rng.integers(1, 4, n)
    s_income |= fa & has_income & (r() < 0.60)
    s_drain |= fa & (r() < 0.60)
    s_leak |= fa & (r() < 0.60)
    s_act |= fa & (r() < 0.60)
    no_warn = churner & ~(s_income | s_drain | s_leak | s_act)

    # job change (non-churn): salary payer changes, sometimes with a one-month gap
    job_change = ~churner & has_salary & (r() < cfg["job_change_rate"])
    emp1 = rng.integers(0, len(EMPLOYERS), n)
    emp2 = (emp1 + rng.integers(1, len(EMPLOYERS), n)) % len(EMPLOYERS)

    rising = has_salary & (r() < P["p_rising"].to_numpy())
    has_school = r() < P["p_school"].to_numpy()
    has_supplier = r() < P["p_supplier"].to_numpy()

    return pd.DataFrame({
        "customer_id": cust["customer_id"],
        "segment": seg,
        "churner": churner,
        "churn_type": churn_type,
        "churn_month": churn_month,
        "churn_date": churn_date,
        "lead_months": lead,
        "no_warning": no_warn,
        "sign_income_stop": s_income,
        "sign_balance_drain": s_drain,
        "sign_leakage_rise": s_leak,
        "sign_activity_drop": s_act,
        "stop_variant": np.where(r() < 0.6, "stop", "shrink"),
        "shrink_to": rng.uniform(0.3, 0.6, n).round(2),
        "stop_offset": rng.integers(0, 2, n),
        "false_alarm": fa,
        "fa_start": fa_start,
        "fa_len": fa_len,
        "job_change": job_change,
        "jc_month": rng.integers(3, M - 2, n),
        "jc_gap": r() < 0.5,
        "jc_mult": rng.uniform(0.9, 1.3, n).round(2),
        "employer1": emp1,
        "employer2": emp2,
        "has_salary": has_salary,
        "salary_usd": np.exp(rng.normal(np.log(P["salary_median"].to_numpy()), 0.45)).round(0),
        "salary_day": rng.integers(21, 29, n),
        "has_remit": has_remit,
        "remit_usd": np.exp(rng.normal(np.log(220), 0.4, n)).round(0),
        "remit_day": rng.integers(1, 28, n),
        "remit_reg": rng.uniform(0.7, 0.98, n).round(2),
        "has_school": has_school,
        "school_usd": rng.uniform(120, 800, n).round(0),
        "has_supplier": has_supplier,
        "sup_share": rng.uniform(0.3, 0.5, n).round(2),
        "rising_salary": rising,
        "rise_month": rng.integers(2, M - 6, n),
        "rise_gain": np.where(rising, rng.uniform(0.02, 0.05, n), 0.0).round(3),
        "wallet_frac": rng.uniform(0.5, 0.85, n).round(2),
        "spend_base": rng.uniform(0.75, 1.0, n).round(2),
        "scale": np.exp(rng.normal(0, 0.4, n)).round(2),
        "has_savings": r() < cfg["savings_account_share"],
        "churn_day": churn_day,
    })


# --------------------------------------------------------------------------
# 3. Accounts: one main (current) account each, 30% also hold a savings account
# --------------------------------------------------------------------------
def _accounts(rng, cust, truth, start, end) -> pd.DataFrame:
    n = len(cust)
    P = SEG.loc[cust["segment"].to_numpy()].reset_index(drop=True)
    usd = rng.random(n) < P["p_usd"].to_numpy()
    closed = (truth["churn_type"] == "closed").to_numpy()
    close_date = pd.to_datetime(np.where(closed, truth["churn_date"].to_numpy(), np.datetime64("NaT")))

    main = pd.DataFrame({
        "account_id": [f"A{i:07d}" for i in range(1, n + 1)],
        "customer_id": cust["customer_id"],
        "product_type": "current",
        "currency": np.where(usd, "USD", "ZiG"),
        "open_date": cust["open_date"],
        "close_date": close_date,
    })
    has_sav = truth["has_savings"].to_numpy()
    idx = np.flatnonzero(has_sav)
    sav_open = (cust["open_date"].to_numpy()[idx]
                + pd.to_timedelta(rng.integers(0, 700, idx.size), "D").to_numpy())
    sav_open = np.minimum(sav_open, (end - pd.Timedelta(days=90)).to_datetime64())
    # a savings account cannot open after the customer has closed their relationship
    cd = close_date[idx].to_numpy()
    sav_open = np.where(pd.isna(cd), sav_open, np.minimum(sav_open, cd - np.timedelta64(30, 'D')))
    sav_open = np.maximum(sav_open, cust['open_date'].to_numpy()[idx])
    sav = pd.DataFrame({
        "account_id": [f"A{n + i + 1:07d}" for i in range(idx.size)],
        "customer_id": cust["customer_id"].to_numpy()[idx],
        "product_type": "savings",
        "currency": np.where(rng.random(idx.size) < 0.6, "USD", "ZiG"),
        "open_date": pd.to_datetime(sav_open),
        "close_date": close_date[idx],
    })
    return pd.concat([main, sav], ignore_index=True)


# --------------------------------------------------------------------------
# 4. Monthly plan -> transactions
# --------------------------------------------------------------------------
def _pois(rng, lam):
    return rng.poisson(np.maximum(lam, 0))


def _split(rng, total, count):
    """Split each total into `count` random pieces (gamma weights). Returns (row->cell, amounts)."""
    cells = np.flatnonzero(total > 0.5)
    cnt = np.maximum(count.ravel()[cells], 1)
    rep = np.repeat(cells, cnt)
    w = rng.gamma(2.0, 1.0, rep.size) + 0.05
    sums = np.bincount(rep, w, minlength=total.size)[rep]
    return rep, total.ravel()[rep] * w / sums


def _transactions(rng, cust, truth, acct, cfg, start, M):
    n = len(cust)
    T = truth
    P = SEG.loc[cust["segment"].to_numpy()].reset_index(drop=True)
    m = np.arange(M)[None, :]
    month_starts = pd.date_range(start, periods=M, freq="MS")
    dim = month_starts.days_in_month.to_numpy()
    cal_month = month_starts.month.to_numpy()

    def col(c):
        return T[c].to_numpy()[:, None]

    def pcol(c):
        return P[c].to_numpy()[:, None]

    open_m = month_index(cust["open_date"], start)[:, None]
    churner = col("churner")
    churn_m = col("churn_month")
    closed = col("churn_type") == "closed"
    silent = churner & ~closed
    mtc = churn_m - m
    lead = col("lead_months")
    warn = churner & (mtc > 0) & (mtc <= lead)
    prog = np.where(warn, 1 - (mtc - 1) / lead, 0.0)          # 1/lead ... 1 as churn nears
    fa_on = col("false_alarm") & (m >= col("fa_start")) & (m < col("fa_start") + col("fa_len"))
    prog = np.where(fa_on, (m - col("fa_start") + 1) / col("fa_len"), prog)
    window = warn | fa_on
    window_start = np.where(warn, churn_m - lead, col("fa_start"))
    post = churner & (m >= churn_m)                            # churn month onward
    post_silent = silent & post
    alive = (m >= open_m) & ~(closed & (m > churn_m))

    def sign(c):
        return col(c) & window

    # --- activity level (multiplies transaction counts) ---
    act = rng.lognormal(0, 0.15, (n, M)) * np.where(cal_month[None, :] == 12, 1.2, 1.0)
    act = act * np.where(sign("sign_activity_drop"), 1 - 0.65 * prog, 1.0)
    act = np.where(post_silent, 0.03 * rng.lognormal(0, 0.3, (n, M)), act)
    act = np.where(closed & (m == churn_m), act * col("churn_day") / dim[None, :], act)
    act = act * alive

    # --- income: salary and/or remittance, with growth, job changes, stops and shrinks ---
    growth = (1 + col("rise_gain")) ** np.maximum(m - col("rise_month"), 0)
    jc_after = col("job_change") & (m >= col("jc_month"))
    employer = np.where(jc_after, col("employer2"), col("employer1"))
    inc_mult = np.where(jc_after, col("jc_mult"), 1.0)
    inc_mult = np.where(col("job_change") & col("jc_gap") & (m == col("jc_month")), 0.0, inc_mult)
    stop_m = window_start + col("stop_offset")
    stopped = sign("sign_income_stop") & (m >= stop_m)
    inc_mult = np.where(stopped, np.where(col("stop_variant") == "shrink", col("shrink_to"), 0.0), inc_mult)
    inc_mult = np.where(post, 0.0, inc_mult) * alive
    salary = (col("has_salary") * col("salary_usd") * growth * inc_mult
              * rng.normal(1, 0.02, (n, M)) * (rng.random((n, M)) > 0.03))   # 3% of months a salary is late
    remit = (col("has_remit") * col("remit_usd") * inc_mult * (rng.random((n, M)) < col("remit_reg"))
             * rng.lognormal(0, 0.15, (n, M)))
    biz = pcol("biz_inflow") * col("scale") * rng.lognormal(0, 0.35, (n, M)) * np.minimum(act, 1.5)
    biz = np.where(biz < 3.0, 0.0, biz)
    biz = np.where(post_silent & (rng.random((n, M)) > 0.10), 0.0, biz)   # a gone customer rarely still receives money
    school = (col("has_school") * np.isin(cal_month, [1, 5, 9])[None, :] * (rng.random((n, M)) < 0.9)
              * col("school_usd") * rng.lognormal(0, 0.1, (n, M)) * alive * ~post)

    # --- where the money goes ---
    leak = np.clip(pcol("leak_share") * rng.lognormal(0, 0.2, (n, M)), 0.05, 0.9)
    leak = np.where(sign("sign_leakage_rise"), np.minimum(leak + 0.35 * prog, 0.92), leak)
    spend = col("spend_base") * rng.lognormal(0, 0.1, (n, M))
    spend = np.where(sign("sign_balance_drain"), 1.0 + 0.9 * prog, spend)

    # --- month by month: outflow can never exceed what is in the account ---
    sav_acct = acct[acct["product_type"] == "savings"].set_index("customer_id")
    has_sav = cust["customer_id"].isin(sav_acct.index).to_numpy()
    sav_open_m = np.full(n, 10**6)
    sav_open_m[has_sav] = month_index(sav_acct.loc[cust["customer_id"][has_sav], "open_date"], start)
    typical = (col("has_salary") * col("salary_usd") + col("has_remit") * col("remit_usd")
               + pcol("biz_inflow") * col("scale"))[:, 0]
    in_window = open_m[:, 0] >= 0
    open_plan = np.where(in_window, 0.0, typical * rng.uniform(0.2, 1.5, n))   # opening balance of the main account
    bal = open_plan.copy()
    init_dep = np.where(in_window, typical * rng.uniform(0.1, 0.8, n), 0.0)
    bal = bal + init_dep
    sav_bal = np.where(has_sav & (sav_open_m < 0), typical * rng.uniform(0.2, 2.0, n), 0.0)
    sav_init = sav_bal.copy()
    out = np.zeros((n, M)); dep = np.zeros((n, M)); wd = np.zeros((n, M)); bal_end = np.zeros((n, M)); svc_fee = np.zeros((n, M))
    drain = sign("sign_balance_drain")
    for k in range(M):
        a = alive[:, k]
        sav_live = has_sav & (sav_open_m <= k) & a
        w = np.where(sav_live & (drain[:, k] | (post_silent[:, k] & (k - churn_m[:, 0] <= 1))), 0.6 * sav_bal, 0.0)
        inflow0 = salary[:, k] + remit[:, k] + biz[:, k]
        inflow = (inflow0 + w) * a
        # spending comes out of what was in the account at the start of the month (plus any
        # savings withdrawn), so the running balance can never dip below zero mid-month
        desired = (inflow0 * spend[:, k] + school[:, k] + w
                   + np.where(drain[:, k], (0.3 + 0.5 * prog[:, k]) * bal, 0.0)
                   + np.where(post_silent[:, k], np.select([k - churn_m[:, 0] == 0, k - churn_m[:, 0] == 1], [0.85, 0.5], 0.0) * bal, 0.0))
        cap = np.maximum(0.97 * bal + w - 2 * cfg["monthly_service_fee_usd"], 0.0)
        o = np.minimum(desired, cap) * a
        o = np.where(o < 5.0, 0.0, o)          # nothing happens in a month with under $5 of spending
        d = np.where(sav_live & ~window[:, k] & ~post[:, k] & (k > sav_open_m), 0.08 * o, 0.0)
        out[:, k], dep[:, k], wd[:, k] = o, d, w
        # the service fee leaves the account too; the statement must charge exactly what the plan
        # deducts, otherwise balances drift away from the plan (transfer fees are carved out of the
        # transfer amounts below, so they need no deduction here)
        svc = np.where(a & (bal + inflow - o > 3 * cfg["monthly_service_fee_usd"]), cfg["monthly_service_fee_usd"], 0.0)
        svc_fee[:, k] = svc
        bal = bal + inflow - o - svc
        sav_bal = sav_bal + d - w
        bal_end[:, k] = bal

    school_paid = np.minimum(school, out)
    rest = out - school_paid - dep
    sup = np.where(col("has_supplier"), col("sup_share"), 0.0) * rest
    rest = rest - sup
    leak_amt = leak * rest
    wallet = leak_amt * col("wallet_frac")
    obank = leak_amt - wallet
    rest = rest - leak_amt
    pos, atm, util = 0.5 * rest, 0.3 * rest, 0.2 * rest

    base = pcol("base_txn") * act
    leak_ratio = leak / pcol("leak_share")

    # transaction counts per kind; in quiet months a kind may draw zero, in which case its
    # money is spent through the card instead so the plan still adds up
    n_sup, n_wallet, n_obank = _pois(rng, base * 0.25), _pois(rng, base * 0.2 * leak_ratio), _pois(rng, base * 0.08 * leak_ratio)
    n_atm, n_util, n_pos = _pois(rng, base * 0.15), _pois(rng, base * 0.15), _pois(rng, base * 0.35)
    for amt_arr, cnt_arr, floor in [(sup, n_sup, 0), (atm, n_atm, 0), (util, n_util, 0),
                                    (wallet, n_wallet, 10), (obank, n_obank, 10)]:   # leakage rows survive unless tiny
        zero = (cnt_arr == 0) & (amt_arr > 0) & ((floor == 0) | (amt_arr < floor) | post_silent)
        pos = pos + np.where(zero, amt_arr, 0.0)
        amt_arr[zero] = 0.0

    parts = []
    cust_branch = pd.Index(BRANCHES).get_indexer(cust["branch"])

    def add(total, count, direction, channels, ch_w, cpty, desc_fn, day_fn, account="main"):
        rep, amt = _split(rng, total, count)
        if rep.size == 0:
            return
        ci, mo = rep // M, rep % M
        k = rep.size
        parts.append(pd.DataFrame({
            "ci": ci, "mo": mo, "day": day_fn(ci, mo, k), "amount_usd": amt,
            "direction": direction,
            "channel": rng.choice(channels, k, p=ch_w) if ch_w else channels[0],
            "counterparty_type": cpty,
            "description": desc_fn(ci, mo, k),
            "account": account,
        }))

    def uniform_day(ci, mo, k):
        return rng.integers(0, dim[mo])

    def pick(names, k, w=None):
        return rng.choice(names, k, p=w)

    def ref(k, lo=100000, hi=999999):
        return rng.integers(lo, hi, k).astype(str)

    emp_names = np.array(EMPLOYERS)
    city = np.array(BRANCH_CITY)

    add(salary, np.ones((n, M), int), "credit", ["RTGS"], None, "employer",
        lambda ci, mo, k: np.char.add("SALARY ", emp_names[employer[ci, mo]]),
        lambda ci, mo, k: np.clip(T["salary_day"].to_numpy()[ci] - 1 + rng.integers(-1, 2, k), 0, dim[mo] - 1))
    add(remit, np.ones((n, M), int), "credit", ["RTGS", "mobile app"], [0.7, 0.3], "remittance",
        lambda ci, mo, k: np.char.add(pick(REMITTERS, k, REMITTER_W), np.char.add(" REMITTANCE REF ", ref(k))),
        lambda ci, mo, k: np.clip(T["remit_day"].to_numpy()[ci] - 1 + rng.integers(-3, 4, k), 0, dim[mo] - 1))
    add(biz, _pois(rng, pcol("biz_count") * act), "credit", ["ZIPIT", "mobile app", "POS"], [0.4, 0.4, 0.2], "mobile_wallet",
        lambda ci, mo, k: np.where(rng.random(k) < 0.6,
                                   np.char.add(np.char.add(pick(WALLETS, k, WALLET_W), " CASH IN "), ref(k)),
                                   np.where(rng.random(k) < 0.5, np.char.add("ZIPIT FROM ", pick(BANKS, k)),
                                            np.char.add("POS SETTLEMENT ", ref(k)))),
        uniform_day)
    add(school_paid, np.ones((n, M), int), "debit", ["RTGS", "branch"], [0.7, 0.3], "school",
        lambda ci, mo, k: np.char.add("SCHOOL FEES ", pick(SCHOOLS, k)),
        lambda ci, mo, k: rng.integers(0, 15, k))
    add(sup, n_sup, "debit", ["RTGS", "ZIPIT"], [0.6, 0.4], "supplier",
        lambda ci, mo, k: np.char.add("PAYMENT ", np.char.add(pick(SUPPLIERS, k), np.char.add(" INV ", ref(k, 1000, 9999)))),
        uniform_day)
    add(wallet, n_wallet, "debit", ["USSD", "mobile app"], [0.55, 0.45], "mobile_wallet",
        lambda ci, mo, k: np.char.add(np.char.add(pick(WALLETS, k, WALLET_W), " TRANSFER 07"), ref(k, 10000000, 99999999)),
        uniform_day)
    add(obank, n_obank, "debit", ["ZIPIT", "RTGS"], [0.7, 0.3], "other_bank",
        lambda ci, mo, k: np.char.add("ZIPIT TO ", np.char.add(pick(BANKS, k), np.char.add(" ", ref(k)))),
        uniform_day)
    add(pos, n_pos, "debit", ["POS"], None, "merchant",
        lambda ci, mo, k: np.char.add("POS ", np.char.add(pick(MERCHANTS, k), np.char.add(" ", city[cust_branch[ci]]))),
        uniform_day)
    add(atm, n_atm, "debit", ["ATM"], None, "cash",
        lambda ci, mo, k: np.char.add("ATM WITHDRAWAL ", city[cust_branch[ci]]),
        uniform_day)
    add(util, n_util, "debit", ["USSD", "mobile app"], [0.5, 0.5], "merchant",
        lambda ci, mo, k: pick(UTILITIES, k), uniform_day)
    add(svc_fee, np.ones((n, M), int), "debit",
        ["branch"], None, "fee", lambda ci, mo, k: np.repeat("MONTHLY SERVICE FEE", k),
        lambda ci, mo, k: dim[mo] - 1)
    # internal transfers between the two accounts (a row on each side)
    for total, d_main, d_sav, desc_main, desc_sav in [
        (dep, "debit", "credit", "TRANSFER TO SAVINGS", "TRANSFER FROM CURRENT"),
        (wd, "credit", "debit", "TRANSFER FROM SAVINGS", "TRANSFER TO CURRENT"),
    ]:
        day_fn = uniform_day if d_main == "debit" else (lambda ci, mo, k: rng.integers(0, 3, k))
        add(total, np.ones((n, M), int), d_main, ["mobile app", "branch"], [0.8, 0.2], "self",
            lambda ci, mo, k, s=desc_main: np.repeat(s, k), day_fn, "main")
        add(total, np.ones((n, M), int), d_sav, ["mobile app", "branch"], [0.8, 0.2], "self",
            lambda ci, mo, k, s=desc_sav: np.repeat(s, k), day_fn, "savings")
    # initial deposits for accounts opened inside the window
    add(init_dep[:, None] * (m == open_m), np.ones((n, M), int), "credit", ["branch"], None, "cash",
        lambda ci, mo, k: np.repeat("INITIAL DEPOSIT", k),
        lambda ci, mo, k: cust["open_date"].dt.day.to_numpy()[ci] - 1)
    add(sav_init[:, None] * (m == np.clip(sav_open_m, 0, M)[:, None]) * (sav_open_m >= 0)[:, None],
        np.ones((n, M), int), "credit", ["branch"], None, "cash",
        lambda ci, mo, k: np.repeat("INITIAL DEPOSIT", k), lambda ci, mo, k: np.zeros(k, int), "savings")

    tx = pd.concat(parts, ignore_index=True)
    # the matching side of each savings transfer shares the day of its partner
    for s_main, s_sav in [("TRANSFER TO SAVINGS", "TRANSFER FROM CURRENT"),
                          ("TRANSFER FROM SAVINGS", "TRANSFER TO CURRENT")]:
        a = tx[tx["description"] == s_main].set_index(["ci", "mo"])["day"]
        b = tx["description"] == s_sav
        tx.loc[b, "day"] = a.reindex(pd.MultiIndex.from_arrays([tx.loc[b, "ci"], tx.loc[b, "mo"]])).to_numpy()

    # fee lines for interbank transfers, and a few reversals (friction signals)
    xfer = tx[(tx["channel"].isin(["RTGS", "ZIPIT"])) & (tx["direction"] == "debit")]
    fees = xfer.copy()
    fee_amt = np.minimum(np.minimum(0.01 * xfer["amount_usd"] + 0.5, 5.0), 0.5 * xfer["amount_usd"])
    fees["amount_usd"] = fee_amt
    tx.loc[xfer.index, "amount_usd"] = xfer["amount_usd"] - fee_amt      # fee comes out of the transfer, totals unchanged
    fees["counterparty_type"] = "fee"
    fees["description"] = np.where(xfer["channel"] == "RTGS", "RTGS FEE", "ZIPIT FEE")
    rev_pool = tx[tx["direction"] == "debit"].index.to_numpy()
    rev = tx.loc[rng.choice(rev_pool, int(0.01 * rev_pool.size), replace=False)].copy()
    rev["direction"] = "credit"
    rev["description"] = "REVERSAL " + rev["description"]
    rev["day"] = np.minimum(rev["day"] + rng.integers(0, 2, len(rev)), dim[rev["mo"]] - 1)
    retry = tx.loc[rev.index].copy()                     # the customer pays again after the reversal
    retry["day"] = np.minimum(rev["day"] + 1, dim[rev["mo"]] - 1)
    tx = pd.concat([tx, fees, rev, retry], ignore_index=True)

    # attach accounts, dates, currency
    main_acct = acct[acct["product_type"] == "current"].set_index("customer_id")
    tx["customer_id"] = cust["customer_id"].to_numpy()[tx["ci"]]
    is_sav = tx["account"] == "savings"
    tx["account_id"] = np.where(is_sav, sav_acct["account_id"].reindex(tx["customer_id"]).to_numpy(),
                                main_acct["account_id"].reindex(tx["customer_id"]).to_numpy())
    tx["txn_date"] = month_starts.to_numpy()[tx["mo"]] + pd.to_timedelta(tx["day"], "D")
    acct_cur = acct.set_index("account_id")["currency"]
    tx["currency"] = acct_cur.reindex(tx["account_id"]).to_numpy()
    zig_rate = cfg["zig_per_usd_start"] * (1 + cfg["zig_monthly_drift"]) ** np.arange(M)
    tx["amount"] = np.where(tx["currency"] == "ZiG", tx["amount_usd"] * zig_rate[tx["mo"]], tx["amount_usd"]).round(2)

    # keep only dates inside each account's life
    a_open = acct.set_index("account_id")["open_date"].reindex(tx["account_id"]).to_numpy()
    a_close = acct.set_index("account_id")["close_date"].reindex(tx["account_id"]).to_numpy()
    keep = (tx["txn_date"].to_numpy() >= a_open) & (pd.isna(a_close) | (tx["txn_date"].to_numpy() < a_close))
    tx = tx[keep & (tx["amount"] > 0)].copy()

    # running balance per account; opening balance is raised if the path ever dips below zero
    tx["seq"] = rng.random(len(tx))
    tx = tx.sort_values(["account_id", "txn_date", "seq"]).reset_index(drop=True)
    signed = np.where(tx["direction"] == "credit", tx["amount"], -tx["amount"])
    cum = pd.Series(signed).groupby(tx["account_id"].to_numpy()).cumsum().to_numpy()
    cum_min = pd.Series(cum).groupby(tx["account_id"].to_numpy()).min()
    opening = pd.Series(0.0, index=cum_min.index)
    # the statement must start from the same opening balance the monthly plan started from
    owner_i = pd.Series(np.arange(n), index=cust["customer_id"]).reindex(acct["customer_id"]).to_numpy()
    is_sav_acct = (acct["product_type"] == "savings").to_numpy()
    base_open = np.where(is_sav_acct, np.where(sav_open_m[owner_i] < 0, sav_init[owner_i], 0.0), open_plan[owner_i])
    base_open = pd.Series(base_open, index=acct["account_id"]).reindex(opening.index)
    base_open = base_open * np.where(acct_cur.reindex(opening.index) == "ZiG", zig_rate[0], 1.0)
    base_open = base_open.fillna(0)
    need = (-(cum_min + base_open)).clip(lower=0)        # only if the path would still dip below zero
    opening = (base_open + need).round(2)
    tx["balance_after"] = (cum + opening.reindex(tx["account_id"]).to_numpy()).round(2) + 0.0

    # closure: closed accounts withdraw whatever is left on the close date
    last = tx.groupby("account_id").tail(1).set_index("account_id")
    closing = acct[acct["close_date"].notna()].set_index("account_id")
    closing = closing.join(last[["balance_after"]], how="inner")
    closing = closing[closing["balance_after"] > 0.01]
    close_rows = pd.DataFrame({
        "account_id": closing.index, "customer_id": closing["customer_id"].to_numpy(),
        "txn_date": closing["close_date"].to_numpy(), "amount": closing["balance_after"].to_numpy(),
        "direction": "debit", "currency": closing["currency"].to_numpy(), "channel": "branch",
        "counterparty_type": "cash", "description": "ACCOUNT CLOSURE BALANCE WITHDRAWN",
        "balance_after": 0.0, "seq": 2.0,
    })
    cols = ["account_id", "customer_id", "txn_date", "amount", "direction", "currency",
            "channel", "counterparty_type", "description", "balance_after", "seq"]
    tx = pd.concat([tx[cols], close_rows[cols]], ignore_index=True)
    tx = tx.sort_values(["txn_date", "seq"]).reset_index(drop=True)
    tx.insert(0, "txn_id", [f"T{i:08d}" for i in range(1, len(tx) + 1)])
    return tx.drop(columns="seq")


# --------------------------------------------------------------------------
# 5. Products held, including uptake driven (noisily) by life events
# --------------------------------------------------------------------------
def _products(rng, cust, truth, acct, start, end, M):
    n = len(cust)
    closed_date = pd.to_datetime(np.where(truth["churn_type"] == "closed", truth["churn_date"], np.datetime64("NaT")))
    rows = [pd.DataFrame({"customer_id": cust["customer_id"], "product": "transactional_account",
                          "start_date": cust["open_date"], "end_date": closed_date})]
    mb = rng.random(n) < 0.6
    mb_start = cust["open_date"] + pd.to_timedelta(rng.integers(0, 400, n), "D")
    rows.append(pd.DataFrame({"customer_id": cust["customer_id"][mb], "product": "mobile_banking",
                              "start_date": mb_start[mb], "end_date": closed_date[mb]}))
    sav = acct[acct["product_type"] == "savings"]
    rows.append(pd.DataFrame({"customer_id": sav["customer_id"], "product": "savings_account",
                              "start_date": sav["open_date"], "end_date": sav["close_date"]}))
    loan = rng.random(n) < 0.12
    loan_start = start + pd.to_timedelta(rng.integers(-300, 30 * (M - 3), n), "D")
    loan_end = loan_start + pd.Timedelta(days=365)
    rows.append(pd.DataFrame({"customer_id": cust["customer_id"][loan], "product": "personal_loan",
                              "start_date": loan_start[loan],
                              "end_date": np.where(loan_end[loan] <= end, loan_end[loan], np.datetime64("NaT"))}))

    # life-event driven uptake: a higher chance, with a delay, never a certainty
    month_starts = pd.date_range(start, periods=M, freq="MS")
    churn_m = np.where(truth["churner"], truth["churn_month"], M + 10)
    uptake_product = np.full(n, "", dtype=object)
    uptake_month = np.full(n, -1)
    for event, (product, p_yes, p_no) in UPTAKE.items():
        has = truth[event].to_numpy()
        p = np.where(has, p_yes, p_no) * np.where(truth["churner"], 0.3, 1.0)
        take = (rng.random(n) < p) & (uptake_product == "")
        mon = rng.integers(6, M, n)
        take &= mon < churn_m - 1
        uptake_product[take] = product
        uptake_month[take] = mon[take]
    idx = np.flatnonzero(uptake_product != "")
    up_start = month_starts[uptake_month[idx]] + pd.to_timedelta(rng.integers(0, 28, idx.size), "D")
    rows.append(pd.DataFrame({"customer_id": cust["customer_id"].to_numpy()[idx], "product": uptake_product[idx],
                              "start_date": up_start, "end_date": closed_date[idx]}))
    prods = pd.concat(rows, ignore_index=True)
    prods = prods[prods["start_date"] <= end].sort_values(["customer_id", "start_date"]).reset_index(drop=True)
    truth = truth.assign(uptake_product=uptake_product, uptake_month=uptake_month)
    return prods, truth


# --------------------------------------------------------------------------
# 6. Entry points
# --------------------------------------------------------------------------
def generate(settings: dict, n_customers: int | None = None, seed: int | None = None) -> dict[str, pd.DataFrame]:
    cfg = settings["synth"]
    start, end = pd.Timestamp(settings["data"]["start_date"]), pd.Timestamp(settings["data"]["end_date"])
    M = (end.year - start.year) * 12 + end.month - start.month + 1
    n = n_customers or cfg["n_customers"]
    rng = np.random.default_rng(settings["seed"] if seed is None else seed)

    cust = _customers(rng, n, start, M, cfg["segments"])
    truth = _truth(rng, cust, cfg, start, M)
    acct = _accounts(rng, cust, truth, start, end)
    tx = _transactions(rng, cust, truth, acct, cfg, start, M)
    prods, truth = _products(rng, cust, truth, acct, start, end, M)
    truth = truth.drop(columns=["employer1", "employer2", "stop_offset", "jc_mult", "wallet_frac",
                                "spend_base", "scale", "remit_reg", "sup_share"])
    return {"customers": cust, "accounts": acct, "transactions": tx, "products": prods, "_truth": truth}


def write(frames: dict[str, pd.DataFrame], raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        df.to_csv(raw_dir / f"{name}.csv", index=False, date_format="%Y-%m-%d")


def main(root: Path = PROJECT_ROOT) -> dict[str, pd.DataFrame]:
    settings = load_settings(root)
    frames = generate(settings)
    raw_dir = resolve(root, settings["data"]["raw_dir"])
    write(frames, raw_dir)
    t = frames["_truth"]
    print(f"wrote {len(frames['customers']):,} customers, {len(frames['accounts']):,} accounts, "
          f"{len(frames['transactions']):,} transactions, {len(frames['products']):,} product rows -> {raw_dir}")
    print(f"planted churn: {t['churner'].mean():.1%} of customers "
          f"({(t['churn_type'] == 'closed').sum()} closed, {(t['churn_type'] == 'silent').sum()} silent, "
          f"{t['no_warning'].sum()} with no warning); false alarms: {t['false_alarm'].sum()}")
    return frames


if __name__ == "__main__":
    main()
