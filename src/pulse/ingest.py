"""Load the raw CSVs through config/schema.yaml into DuckDB and check data quality.

The schema file is the only place where source column names appear. This module
knows only the CANONICAL names below. To use a new dataset, edit the YAML.

The hidden-truth file (data/raw/_truth.csv) is never read here: the pipeline
must discover churn from the transactions alone.

Run:  python -m pulse.cli ingest
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

from pulse.config import PROJECT_ROOT, load_schema, load_settings, resolve

# Canonical tables: column -> type. "DATE" columns are parsed with the file's date_format.
CANONICAL: dict[str, dict[str, str]] = {
    "customers": {
        "customer_id": "VARCHAR", "open_date": "DATE", "segment": "VARCHAR",
        "branch": "VARCHAR", "gender": "VARCHAR", "age_band": "VARCHAR",
    },
    "accounts": {
        "account_id": "VARCHAR", "customer_id": "VARCHAR", "product_type": "VARCHAR",
        "currency": "VARCHAR", "open_date": "DATE", "close_date": "DATE",
    },
    "transactions": {
        "txn_id": "VARCHAR", "account_id": "VARCHAR", "customer_id": "VARCHAR",
        "txn_date": "DATE", "amount": "DOUBLE", "direction": "VARCHAR", "currency": "VARCHAR",
        "channel": "VARCHAR", "counterparty_type": "VARCHAR", "description": "VARCHAR",
        "balance_after": "DOUBLE",
    },
    "products": {
        "customer_id": "VARCHAR", "product": "VARCHAR", "start_date": "DATE", "end_date": "DATE",
    },
}
PRIMARY_KEY = {
    "customers": ["customer_id"],
    "accounts": ["account_id"],
    "transactions": ["txn_id"],
    "products": ["customer_id", "product", "start_date"],
}
# columns that may legitimately be empty
NULLABLE = {("accounts", "close_date"), ("products", "end_date")}
# (child table, child column) -> (parent table, parent column)
FOREIGN_KEYS = {
    ("accounts", "customer_id"): ("customers", "customer_id"),
    ("transactions", "account_id"): ("accounts", "account_id"),
    ("transactions", "customer_id"): ("customers", "customer_id"),
    ("products", "customer_id"): ("customers", "customer_id"),
}
ALLOWED = {("transactions", "direction"): {"credit", "debit"}}


@dataclass
class Report:
    tables: dict[str, dict] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)   # fatal: the data cannot be used as is
    warnings: list[str] = field(default_factory=list)  # worth a look, pipeline continues

    @property
    def ok(self) -> bool:
        return not self.errors

    def text(self) -> str:
        lines = ["DATA QUALITY REPORT", "=" * 60]
        for name, info in self.tables.items():
            lines.append(f"{name}: {info['rows']:,} rows")
            for c, nulls in info["null_rate"].items():
                flag = "   <-- nulls" if nulls > 0 and (name, c) not in NULLABLE else ""
                lines.append(f"    {c:<18} nulls {nulls:6.1%}{flag}")
            for c, (lo, hi) in info["date_range"].items():
                lines.append(f"    {c:<18} {lo} .. {hi}")
            if info["duplicate_keys"]:
                lines.append(f"    duplicate primary keys: {info['duplicate_keys']:,}")
        lines.append("-" * 60)
        for w in self.warnings:
            lines.append(f"WARNING  {w}")
        for e in self.errors:
            lines.append(f"ERROR    {e}")
        lines.append("RESULT   " + ("OK" if self.ok else f"{len(self.errors)} error(s)"))
        return "\n".join(lines)


def _select_sql(table: str, spec: dict, src: str) -> str:
    """Build the SELECT that renames, casts and value-maps one source table."""
    cols = spec["columns"]
    fmt = spec.get("date_format", "%Y-%m-%d")
    values = spec.get("values", {})
    parts = []
    for canon, ctype in CANONICAL[table].items():
        if canon not in cols:
            raise KeyError(f"schema.yaml: table '{table}' has no mapping for canonical column '{canon}'")
        raw = f'"{cols[canon]}"'
        if ctype == "DATE":
            expr = f"CAST(try_strptime(NULLIF(TRIM({raw}), ''), '{fmt}') AS DATE)"
        elif ctype == "DOUBLE":
            expr = f"TRY_CAST(REPLACE({raw}, ',', '') AS DOUBLE)"
        else:
            expr = f"NULLIF(TRIM({raw}), '')"
        if canon in values:
            cases = " ".join(f"WHEN '{k}' THEN '{v}'" for k, v in values[canon].items())
            expr = f"CASE {expr} {cases} ELSE {expr} END"
        parts.append(f"{expr} AS {canon}")
    return f"SELECT {', '.join(parts)} FROM {src}"


def load(root: Path = PROJECT_ROOT, schema_path: Path | None = None,
         db_path: Path | None = None) -> tuple[duckdb.DuckDBPyConnection, Report]:
    """Load all four tables into DuckDB (file at db_path) and run the quality checks."""
    settings = load_settings(root)
    schema = load_schema(root, schema_path)
    db_path = db_path or resolve(root, settings["data"]["duckdb_path"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    report = Report()

    for table in CANONICAL:
        spec = schema[table]
        path = resolve(root, spec["path"])
        if not path.exists():
            report.errors.append(f"{table}: file not found {path}")
            continue
        # read everything as text first, so a bad value becomes a NULL we can count, not a crash
        con.execute(f"CREATE OR REPLACE TABLE raw_{table} AS SELECT * FROM read_csv("
                    f"'{path.as_posix()}', header=true, all_varchar=true)")
        raw_cols = {r[1] for r in con.execute(f"PRAGMA table_info('raw_{table}')").fetchall()}
        missing = [c for c in spec["columns"].values() if c not in raw_cols]
        if missing:
            report.errors.append(f"{table}: columns named in schema.yaml not in file: {missing}")
            continue
        con.execute(f"CREATE OR REPLACE TABLE {table} AS {_select_sql(table, spec, 'raw_' + table)}")
        con.execute(f"DROP TABLE raw_{table}")
        _check_table(con, table, settings, report)

    if report.ok:
        _check_relations(con, report)
    return con, report


def _check_table(con, table, settings, report: Report) -> None:
    n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    info = {"rows": n, "null_rate": {}, "date_range": {}, "duplicate_keys": 0}
    if n == 0:
        report.errors.append(f"{table}: no rows")
    start, end = settings["data"]["start_date"], settings["data"]["end_date"]
    for col, ctype in CANONICAL[table].items():
        nulls = con.execute(f"SELECT COUNT(*) FILTER (WHERE {col} IS NULL) FROM {table}").fetchone()[0]
        info["null_rate"][col] = nulls / n if n else 0.0
        if nulls and (table, col) not in NULLABLE:
            msg = f"{table}.{col}: {nulls:,} nulls (unparseable or empty values)"
            (report.errors if col in PRIMARY_KEY[table] or nulls == n else report.warnings).append(msg)
        if ctype == "DATE" and n:
            lo, hi = con.execute(f"SELECT MIN({col}), MAX({col}) FROM {table}").fetchone()
            info["date_range"][col] = (lo, hi)
            if hi is not None and hi > end:
                report.warnings.append(f"{table}.{col}: dates after settings end_date {end} (max {hi})")
            if table == "transactions" and lo is not None and lo < start:
                report.warnings.append(f"{table}.{col}: dates before settings start_date {start} (min {lo})")
    key = ", ".join(PRIMARY_KEY[table])
    dups = con.execute(f"SELECT COUNT(*) FROM (SELECT {key} FROM {table} GROUP BY {key} HAVING COUNT(*) > 1)").fetchone()[0]
    info["duplicate_keys"] = dups
    if dups:
        report.errors.append(f"{table}: {dups:,} duplicate primary keys ({key})")
    for (t, col), allowed in ALLOWED.items():
        if t == table:
            bad = con.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} NOT IN ({', '.join(repr(a) for a in allowed)})").fetchone()[0]
            if bad:
                report.errors.append(f"{table}.{col}: {bad:,} values outside {sorted(allowed)}; add a values mapping in schema.yaml")
    if table == "transactions":
        bad = con.execute("SELECT COUNT(*) FROM transactions WHERE amount <= 0").fetchone()[0]
        if bad:
            report.warnings.append(f"transactions: {bad:,} rows with amount <= 0")
        neg = con.execute("SELECT COUNT(*) FROM transactions WHERE balance_after < 0").fetchone()[0]
        if neg:
            report.warnings.append(f"transactions: {neg:,} rows with negative balance_after (overdrafts?)")
    if table == "accounts":
        bad = con.execute("SELECT COUNT(*) FROM accounts WHERE close_date < open_date").fetchone()[0]
        if bad:
            report.errors.append(f"accounts: {bad:,} rows close before they open")
    report.tables[table] = info


def _check_relations(con, report: Report) -> None:
    for (child, ccol), (parent, pcol) in FOREIGN_KEYS.items():
        orphans = con.execute(f"SELECT COUNT(*) FROM {child} c WHERE NOT EXISTS "
                              f"(SELECT 1 FROM {parent} p WHERE p.{pcol} = c.{ccol})").fetchone()[0]
        if orphans:
            report.errors.append(f"{child}.{ccol}: {orphans:,} rows with no matching {parent}.{pcol}")
    mismatch = con.execute("SELECT COUNT(*) FROM transactions t JOIN accounts a USING (account_id) "
                           "WHERE t.customer_id <> a.customer_id").fetchone()[0]
    if mismatch:
        report.errors.append(f"transactions: {mismatch:,} rows whose customer does not own the account")
    after_close = con.execute("SELECT COUNT(*) FROM transactions t JOIN accounts a USING (account_id) "
                              "WHERE a.close_date IS NOT NULL AND t.txn_date > a.close_date").fetchone()[0]
    if after_close:
        report.warnings.append(f"transactions: {after_close:,} rows dated after the account closed")
    before_open = con.execute("SELECT COUNT(*) FROM transactions t JOIN accounts a USING (account_id) "
                              "WHERE t.txn_date < a.open_date").fetchone()[0]
    if before_open:
        report.warnings.append(f"transactions: {before_open:,} rows dated before the account opened")
    cur_mismatch = con.execute("SELECT COUNT(*) FROM transactions t JOIN accounts a USING (account_id) "
                               "WHERE t.currency <> a.currency").fetchone()[0]
    if cur_mismatch:
        report.warnings.append(f"transactions: {cur_mismatch:,} rows whose currency differs from the account's")
    silent = con.execute("SELECT COUNT(*) FROM customers c WHERE NOT EXISTS "
                         "(SELECT 1 FROM transactions t WHERE t.customer_id = c.customer_id)").fetchone()[0]
    if silent:
        report.warnings.append(f"customers: {silent:,} customers with no transactions at all")


def summary_frame(con) -> pd.DataFrame:
    """Small per-table summary used by the CLI and docs."""
    rows = []
    for t in CANONICAL:
        rows.append({"table": t, "rows": con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]})
    return pd.DataFrame(rows)


def main(root: Path = PROJECT_ROOT) -> Report:
    con, report = load(root)
    print(report.text())
    out = resolve(root, load_settings(root)["data"]["processed_dir"]) / "dq_report.txt"
    out.write_text(report.text(), encoding="utf-8")
    con.close()
    if not report.ok:
        sys.exit(1)
    return report


if __name__ == "__main__":
    main()
