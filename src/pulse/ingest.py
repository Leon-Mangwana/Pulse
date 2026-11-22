"""Load the raw CSVs through config/schema.yaml into DuckDB and check data quality.

The schema file is the only place where source column names appear. This module
knows only the CANONICAL names below. To use a new dataset, edit the YAML.

The hidden-truth file (data/raw/_truth.csv) is never read here: the pipeline
must discover churn from the transactions alone.

Fallback (summary) mode: when schema.yaml maps a single `customer_summary` table instead of the
four raw tables, that table is loaded with its churn flag and whatever customer-level columns
are mapped under `features:`; empty transaction tables are created so later steps can run.

Per-table options in schema.yaml: `date_format` (one pattern for the file), `date_formats`
(pattern per column, overrides it), `values` (raw -> canonical value maps), `distinct: true`
(drop exact duplicate rows after renaming) and `one_row_per_key: true` (collapse several rows
per primary key into one, keeping the first non-null value of each column; for a customer file
that holds one row per account).

Run:  python -m pulse.cli ingest
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

from pulse.config import PROJECT_ROOT, SUMMARY_TABLE, data_mode, load_schema, load_settings, resolve

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

# summary mode: the one customer-level table. Required: customer_id, churn_flag. The rest is optional.
SUMMARY_CANONICAL = {"customer_id": "VARCHAR", "churn_flag": "DOUBLE", "snapshot_date": "DATE",
                     "segment": "VARCHAR", "branch": "VARCHAR", "gender": "VARCHAR", "age_band": "VARCHAR"}
SUMMARY_REQUIRED = ["customer_id", "churn_flag"]
TRUE_WORDS = {"1", "1.0", "true", "t", "yes", "y", "churned", "closed", "left", "lost", "inactive"}
FALSE_WORDS = {"0", "0.0", "false", "f", "no", "n", "active", "retained", "stayed", "open"}


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


def _col_expr(canon: str, ctype: str, raw_name: str, spec: dict) -> str:
    """SQL for one canonical column: trim, apply the value map, then parse dates or cast numbers."""
    raw = f'"{raw_name}"'
    fmt = (spec.get("date_formats") or {}).get(canon, spec.get("date_format", "%Y-%m-%d"))
    values = (spec.get("values") or {}).get(canon)
    text = f"NULLIF(TRIM({raw}), '')"
    if values:
        cases = " ".join(f"WHEN '{k}' THEN '{v}'" for k, v in values.items())
        text = f"CASE {text} {cases} ELSE {text} END"
    if ctype == "DATE":
        return f"CAST(try_strptime({text}, '{fmt}') AS DATE)"
    if ctype == "DOUBLE":
        return f"TRY_CAST(REPLACE({text}, ',', '') AS DOUBLE)"
    return text


def _select_sql(table: str, spec: dict, src: str) -> str:
    """Build the SELECT that renames, casts and value-maps one source table."""
    cols = spec["columns"]
    canonical = CANONICAL[table]
    parts = []
    for canon, ctype in canonical.items():
        if canon not in cols:
            raise KeyError(f"schema.yaml: table '{table}' has no mapping for canonical column '{canon}'")
        parts.append(f"{_col_expr(canon, ctype, cols[canon], spec)} AS {canon}")
    distinct = "DISTINCT " if spec.get("distinct") else ""
    sql = f"SELECT {distinct}{', '.join(parts)} FROM {src}"
    if spec.get("one_row_per_key"):
        # collapse several rows per primary key into one, keeping the first non-null value of each
        # column (a customer file with one row per account where branch is blank on some rows)
        key = PRIMARY_KEY[table]
        others = [c for c in canonical if c not in key]
        agg = ", ".join(f"any_value({c}) AS {c}" for c in others)
        sql = f"SELECT {', '.join(key)}, {agg} FROM ({sql}) GROUP BY {', '.join(key)}"
    return sql


def _flag_expr(raw_name: str, spec: dict) -> str:
    """Churn flag: value map first, then common yes/no words, then a plain number."""
    raw = f"LOWER(NULLIF(TRIM(\"{raw_name}\"), ''))"
    values = (spec.get("values") or {}).get("churn_flag")
    if values:
        cases = " ".join(f"WHEN '{str(k).lower()}' THEN '{v}'" for k, v in values.items())
        raw = f"CASE {raw} {cases} ELSE {raw} END"
    yes = ", ".join(repr(w) for w in sorted(TRUE_WORDS))
    no = ", ".join(repr(w) for w in sorted(FALSE_WORDS))
    return (f"CASE WHEN {raw} IN ({yes}) THEN 1.0 WHEN {raw} IN ({no}) THEN 0.0 "
            f"ELSE TRY_CAST({raw} AS DOUBLE) END")


def _read_raw(con, table: str, path: Path, spec: dict, report: Report) -> bool:
    """Read a CSV as text into raw_<table>; False (with an error logged) if a mapped column is missing."""
    con.execute(f"CREATE OR REPLACE TABLE raw_{table} AS SELECT * FROM read_csv("
                f"'{path.as_posix()}', header=true, all_varchar=true)")
    raw_cols = {r[1] for r in con.execute(f"PRAGMA table_info('raw_{table}')").fetchall()}
    mapped = list(spec["columns"].values()) + list((spec.get("features") or {}).values())
    missing = [c for c in mapped if c not in raw_cols]
    if missing:
        report.errors.append(f"{table}: columns named in schema.yaml not in file: {missing}")
        return False
    return True


YES_NO = {**{w: "1" for w in ("Y", "y", "Yes", "yes", "YES", "True", "true", "TRUE", "T")},
          **{w: "0" for w in ("N", "n", "No", "no", "NO", "False", "false", "FALSE", "F")}}


def load_summary(con, schema: dict, root: Path, settings: dict, report: Report) -> None:
    """Summary mode: one customer-level table. Creates customer_summary (id, flag, meta and the
    mapped feature columns as DOUBLE), a customers table for the fairness check and the segment
    view, and empty accounts / transactions / products tables."""
    spec = schema[SUMMARY_TABLE]
    path = resolve(root, spec["path"])
    if not path.exists():
        report.errors.append(f"{SUMMARY_TABLE}: file not found {path}")
        return
    if not _read_raw(con, SUMMARY_TABLE, path, spec, report):
        return
    cols = spec["columns"]
    for req in SUMMARY_REQUIRED:
        if req not in cols:
            report.errors.append(f"{SUMMARY_TABLE}: schema.yaml must map '{req}'")
            return
    features = spec.get("features") or {}
    bad = [f for f in features if f in SUMMARY_CANONICAL or not str(f).isidentifier()]
    if bad:
        report.errors.append(f"{SUMMARY_TABLE}: feature names must be plain identifiers, not one of "
                             f"{list(SUMMARY_CANONICAL)}: {bad}")
        return
    parts = []
    for canon, ctype in SUMMARY_CANONICAL.items():
        if canon == "churn_flag":
            parts.append(f"{_flag_expr(cols[canon], spec)} AS churn_flag")
        elif canon in cols:
            parts.append(f"{_col_expr(canon, ctype, cols[canon], spec)} AS {canon}")
        elif canon == "snapshot_date":
            parts.append(f"CAST('{settings['data']['end_date']}' AS DATE) AS snapshot_date")
        elif canon in ("segment", "branch"):
            parts.append(f"'all' AS {canon}")
        else:
            parts.append(f"CAST(NULL AS {ctype}) AS {canon}")
    for name, raw_name in features.items():
        fspec = {**spec, "values": {name: (spec.get("values") or {}).get(name, YES_NO)}}
        parts.append(f"{_col_expr(name, 'DOUBLE', raw_name, fspec)} AS {name}")
    distinct = "DISTINCT " if spec.get("distinct") else ""
    con.execute(f"CREATE OR REPLACE TABLE {SUMMARY_TABLE} AS SELECT {distinct}{', '.join(parts)} FROM raw_{SUMMARY_TABLE}")
    con.execute(f"DROP TABLE raw_{SUMMARY_TABLE}")
    con.execute("CREATE OR REPLACE TABLE customers AS SELECT customer_id, CAST(NULL AS DATE) AS open_date, "
                f"segment, branch, gender, age_band FROM {SUMMARY_TABLE}")
    for table in ("accounts", "transactions", "products"):
        cols_sql = ", ".join(f"{c} {ty}" for c, ty in CANONICAL[table].items())
        con.execute(f"CREATE OR REPLACE TABLE {table} ({cols_sql})")
    _check_summary(con, features, report)


def _check_summary(con, features: dict, report: Report) -> None:
    n = con.execute(f"SELECT COUNT(*) FROM {SUMMARY_TABLE}").fetchone()[0]
    info = {"rows": n, "null_rate": {}, "date_range": {}, "duplicate_keys": 0}
    report.tables[SUMMARY_TABLE] = info
    if n == 0:
        report.errors.append(f"{SUMMARY_TABLE}: no rows")
        return
    for col in list(SUMMARY_CANONICAL) + list(features):
        nulls = con.execute(f"SELECT COUNT(*) FILTER (WHERE {col} IS NULL) FROM {SUMMARY_TABLE}").fetchone()[0]
        info["null_rate"][col] = nulls / n
        if col in SUMMARY_REQUIRED and nulls:
            report.errors.append(f"{SUMMARY_TABLE}.{col}: {nulls:,} nulls (unparseable or empty values)")
        elif col in features and nulls == n:
            report.errors.append(f"{SUMMARY_TABLE}.{col}: every value is null or not numeric; fix the mapping or add a values map")
        elif col in features and nulls:
            report.warnings.append(f"{SUMMARY_TABLE}.{col}: {nulls:,} nulls, filled with the median in the feature step")
    dups = con.execute(f"SELECT COUNT(*) FROM (SELECT customer_id FROM {SUMMARY_TABLE} GROUP BY 1 HAVING COUNT(*) > 1)").fetchone()[0]
    info["duplicate_keys"] = dups
    if dups:
        report.errors.append(f"{SUMMARY_TABLE}: {dups:,} duplicate customer ids (one row per customer is required; "
                             f"'distinct: true' removes exact duplicates)")
    bad = con.execute(f"SELECT COUNT(*) FROM {SUMMARY_TABLE} WHERE churn_flag NOT IN (0, 1)").fetchone()[0]
    if bad:
        report.errors.append(f"{SUMMARY_TABLE}.churn_flag: {bad:,} values that are not 0/1 or yes/no; add a values map")
    rate = con.execute(f"SELECT AVG(churn_flag) FROM {SUMMARY_TABLE}").fetchone()[0]
    if rate is not None:
        info["date_range"]["churn_rate"] = (f"{100 * rate:.1f}%", f"{int(round(rate * n)):,} churners")
        if rate < 0.01 or rate > 0.6:
            report.warnings.append(f"{SUMMARY_TABLE}: churn rate {100 * rate:.1f}% looks unusual; check what the flag means")
    if not features:
        report.errors.append(f"{SUMMARY_TABLE}: no feature columns mapped under 'features:'")
    lo, hi = con.execute(f"SELECT MIN(snapshot_date), MAX(snapshot_date) FROM {SUMMARY_TABLE}").fetchone()
    info["date_range"]["snapshot_date"] = (lo, hi)


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
    mode = data_mode(schema)
    con.execute("CREATE OR REPLACE TABLE pulse_mode AS SELECT ? AS mode", [mode])
    if mode == "summary":
        load_summary(con, schema, root, settings, report)
        return con, report

    for table in CANONICAL:
        spec = schema[table]
        path = resolve(root, spec["path"])
        if not path.exists():
            report.errors.append(f"{table}: file not found {path}")
            continue
        # read everything as text first, so a bad value becomes a NULL we can count, not a crash
        if not _read_raw(con, table, path, spec, report):
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
        report.errors.append(f"{table}: {dups:,} duplicate primary keys ({key}); if the file holds several rows per "
                             f"{key}, set 'distinct: true' (exact duplicates) or 'one_row_per_key: true' "
                             f"(keeps the first non-null value of each column) on this table in schema.yaml")
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


def stored_mode(con) -> str:
    """The mode the database was loaded in ('transactions' or 'summary')."""
    try:
        return con.execute("SELECT mode FROM pulse_mode").fetchone()[0]
    except duckdb.Error:
        return "transactions"


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
