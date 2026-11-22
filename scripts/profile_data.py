"""Profile a folder of CSV files so schema.yaml can be filled in quickly.

    python scripts/profile_data.py <folder> [--rows 200000] [--out profile.md]

For every CSV: row count, and per column the inferred type, null rate, distinct count, example
values and, for anything date-like, the format that parses it and the date range. Then likely
join keys (columns whose values overlap across files) and a schema.yaml skeleton with the
right-hand sides left to fill in. Large files are sampled (first --rows rows) so this finishes
in seconds; the row count is still exact.

Needs only pandas. Does not import the pulse package, so it runs on any machine with the venv.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y%m%d", "%d-%m-%Y", "%Y/%m/%d", "%d.%m.%Y",
                "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M", "%d-%b-%Y", "%d %b %Y", "%b %d, %Y", "%Y-%m-%dT%H:%M:%S"]
YES_NO = {"y", "n", "yes", "no", "true", "false", "t", "f", "1", "0"}
ID_WORDS = ("id", "no", "ref", "key", "code", "number", "num", "acct", "cust", "client")
CANONICAL = {
    "customers": ["customer_id", "open_date", "segment", "branch", "gender", "age_band"],
    "accounts": ["account_id", "customer_id", "product_type", "currency", "open_date", "close_date"],
    "transactions": ["txn_id", "account_id", "customer_id", "txn_date", "amount", "direction", "currency",
                     "channel", "counterparty_type", "description", "balance_after"],
    "products": ["customer_id", "product", "start_date", "end_date"],
}


def sniff_delimiter(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        head = fh.read(64 * 1024)
    try:
        return csv.Sniffer().sniff(head, delimiters=",;|\t").delimiter
    except csv.Error:
        return ","


def count_rows(path: Path) -> int:
    with open(path, "rb") as fh:
        return max(0, sum(1 for _ in fh) - 1)


def date_format_for(s: pd.Series) -> tuple[str | None, pd.Timestamp | None, pd.Timestamp | None]:
    vals = s.dropna().astype(str).str.strip()
    vals = vals[vals != ""]
    if vals.empty:
        return None, None, None
    sample = vals.sample(min(500, len(vals)), random_state=0)
    best, best_rate = None, 0.0
    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(sample, format=fmt, errors="coerce")
        rate = parsed.notna().mean()
        if rate > best_rate:
            best, best_rate = fmt, rate
        if rate == 1.0:
            break
    if best is None or best_rate < 0.9:
        return None, None, None
    full = pd.to_datetime(vals, format=best, errors="coerce")
    return best, full.min(), full.max()


def numeric_rate(s: pd.Series) -> float:
    vals = s.dropna().astype(str).str.strip().str.replace(",", "", regex=False)
    vals = vals[vals != ""]
    if vals.empty:
        return 0.0
    return pd.to_numeric(vals, errors="coerce").notna().mean()


def profile_column(name: str, s: pd.Series, n_total: int) -> dict:
    s = s.astype("string")
    s = s.mask(s.str.strip() == "", pd.NA)
    nulls = s.isna().mean()
    nonnull = s.dropna()
    distinct = nonnull.nunique()
    examples = nonnull.drop_duplicates().head(4).tolist()
    kind, note = "text", ""
    fmt, lo, hi = (None, None, None)
    lowered = set(nonnull.str.lower().unique()[:50]) if distinct <= 12 else set()
    if distinct and lowered and lowered <= YES_NO:
        kind, note = "yes/no", "values " + "/".join(sorted(lowered))
    else:
        fmt, lo, hi = date_format_for(nonnull) if distinct > 1 else (None, None, None)
        if fmt:
            kind, note = "date", f"format {fmt}, {lo.date()} .. {hi.date()}"
        elif numeric_rate(nonnull) >= 0.98 and distinct > 1:
            nums = pd.to_numeric(nonnull.str.replace(",", "", regex=False), errors="coerce")
            kind = "integer" if (nums.dropna() % 1 == 0).all() else "number"
            note = f"min {nums.min():,.2f}, median {nums.median():,.2f}, max {nums.max():,.2f}"
            if (nums < 0).any():
                note += f"; {int((nums < 0).sum()):,} negative"
        elif distinct <= 20:
            kind, note = "category", "values " + ", ".join(map(str, nonnull.value_counts().head(8).index))
    uniq_share = distinct / max(1, len(nonnull))
    looks_key = uniq_share > 0.98 and len(nonnull) > 20 and kind in ("text", "integer")
    return {"column": name, "type": kind, "null_%": round(100 * nulls, 1), "distinct": int(distinct),
            "unique_share": round(uniq_share, 3), "looks_like_key": looks_key, "examples": examples, "note": note,
            "date_format": fmt}


def profile_file(path: Path, max_rows: int) -> dict:
    delim = sniff_delimiter(path)
    n = count_rows(path)
    df = pd.read_csv(path, sep=delim, dtype=str, nrows=max_rows, keep_default_na=False, na_values=["", "NA", "N/A", "NULL", "null", "None"],
                     encoding="utf-8", encoding_errors="replace")
    cols = [profile_column(c, df[c], n) for c in df.columns]
    return {"file": path.name, "rows": n, "sampled": len(df), "delimiter": delim, "columns": cols, "frame": df}


def join_keys(profiles: list[dict]) -> list[dict]:
    """Pairs of columns in different files whose values overlap a lot: the joins."""
    cands = []
    for p in profiles:
        for c in p["columns"]:
            if c["type"] in ("text", "integer") and c["distinct"] > 20:
                cands.append((p["file"], c["column"], set(p["frame"][c["column"]].dropna().astype(str).str.strip())))
    out = []
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            fa, ca, va = cands[i]
            fb, cb, vb = cands[j]
            if fa == fb or not va or not vb:
                continue
            inter = len(va & vb)
            share = inter / min(len(va), len(vb))
            if share >= 0.5:
                out.append({"file_a": fa, "column_a": ca, "file_b": fb, "column_b": cb,
                            "overlap_%": round(100 * share, 1), "a_unique": len(va), "b_unique": len(vb)})
    return sorted(out, key=lambda r: -r["overlap_%"])


def guess_table(p: dict) -> str | None:
    """Which canonical table a file most resembles, from its column types and row count."""
    kinds = [c["type"] for c in p["columns"]]
    dates = kinds.count("date")
    nums = kinds.count("number") + kinds.count("integer")
    keyed = any(c["looks_like_key"] for c in p["columns"])
    if nums >= 2 and dates >= 1 and p["rows"] > 20000:
        return "transactions"
    if nums >= 2 and dates <= 1 and p["rows"] < 50000 and any(c["type"] == "yes/no" for c in p["columns"]):
        return "customer_summary"
    if dates >= 2 and nums == 0:
        return "accounts (or customers + accounts merged)" if keyed else "products"
    if dates >= 1 and keyed:
        return "customers"
    return None


def skeleton(profiles: list[dict]) -> str:
    lines = ["# schema.yaml skeleton: replace every ??? with a column from the profile above.",
             "# Delete the tables you do not have. If only a customer-level table exists, keep",
             "# customer_summary alone (fallback mode). Per-table options: date_format, date_formats,",
             "# values, distinct: true, one_row_per_key: true (see config/schema.yaml).", ""]
    guesses = {p["file"]: guess_table(p) for p in profiles}
    for table, cols in CANONICAL.items():
        file = next((f for f, g in guesses.items() if g == table), None) or "???.csv"
        p = next((p for p in profiles if p["file"] == file), None)
        fmt = next((c["date_format"] for c in (p["columns"] if p else []) if c["date_format"]), "%Y-%m-%d")
        lines += [f"{table}:", f"  path: data/raw/{file}", f'  date_format: "{fmt}"', "  columns:"]
        lines += [f"    {c}: ???" for c in cols]
        if table == "transactions":
            lines += ["  values:", "    direction: {???: credit, ???: debit}",
                      "    # channel and counterparty_type values Pulse understands:",
                      "    #   channel: mobile app, USSD, ZIPIT, RTGS, POS, ATM, branch",
                      "    #   counterparty_type: self, fee, mobile_wallet, other_bank, school, supplier, cash, remittance, employer, merchant"]
        lines.append("")
    summary_file = next((f for f, g in guesses.items() if g == "customer_summary"), None)
    if summary_file:
        p = next(p for p in profiles if p["file"] == summary_file)
        lines += ["# fallback mode candidate:", "# customer_summary:", f"#   path: data/raw/{summary_file}", "#   columns:",
                  "#     customer_id: ???", "#     churn_flag: ???", "#     segment: ???", "#     branch: ???",
                  "#     gender: ???", "#     age_band: ???", "#   features:"]
        for c in p["columns"]:
            if c["type"] in ("number", "integer", "yes/no") and not c["looks_like_key"]:
                lines.append(f"#     {c['column'].lower().replace(' ', '_').replace('-', '_')}: {c['column']}")
    return "\n".join(lines)


def render(profiles: list[dict], keys: list[dict]) -> str:
    out = []
    for p in profiles:
        guess = guess_table(p)
        out.append(f"## {p['file']}: {p['rows']:,} rows, {len(p['columns'])} columns, delimiter {p['delimiter']!r}"
                   + (f", profiled on the first {p['sampled']:,} rows" if p["sampled"] < p["rows"] else "")
                   + (f"  -> looks like: {guess}" if guess else ""))
        out.append("")
        out.append("| column | type | null % | distinct | key? | examples | note |")
        out.append("| --- | --- | --- | --- | --- | --- | --- |")
        for c in p["columns"]:
            ex = ", ".join(str(e)[:24] for e in c["examples"])
            out.append(f"| {c['column']} | {c['type']} | {c['null_%']} | {c['distinct']:,} | {'yes' if c['looks_like_key'] else ''} | {ex} | {c['note']} |")
        out.append("")
    out.append("## Likely join keys (value overlap between files)")
    out.append("")
    if keys:
        out.append("| file A | column A | file B | column B | overlap % | unique A | unique B |")
        out.append("| --- | --- | --- | --- | --- | --- | --- |")
        for k in keys[:15]:
            out.append(f"| {k['file_a']} | {k['column_a']} | {k['file_b']} | {k['column_b']} | {k['overlap_%']} | {k['a_unique']:,} | {k['b_unique']:,} |")
    else:
        out.append("none found (one file, or no column values shared across files)")
    out.append("")
    out.append("## schema.yaml skeleton")
    out.append("")
    out.append("```yaml")
    out.append(skeleton(profiles))
    out.append("```")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="folder with CSV files")
    ap.add_argument("--rows", type=int, default=200_000, help="rows to profile per file (row counts stay exact)")
    ap.add_argument("--out", help="also write the report to this markdown file")
    args = ap.parse_args(argv)
    folder = Path(args.folder)
    files = sorted(p for p in folder.glob("*.csv") if not p.name.startswith("_"))
    if not files:
        print(f"no CSV files in {folder}")
        return 1
    profiles = [profile_file(f, args.rows) for f in files]
    keys = join_keys(profiles)
    text = render(profiles, keys)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
