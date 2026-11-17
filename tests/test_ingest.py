"""Ingest must go through schema.yaml only, never touch _truth.csv, and report problems."""
import inspect
import re
import shutil

import pandas as pd
import yaml

from pulse import ingest


def _run(root, schema_path=None):
    con, report = ingest.load(root, schema_path=schema_path, db_path=root / "data" / "processed" / "t.duckdb")
    try:
        counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in report.tables}
    finally:
        con.close()
    return report, counts


def test_clean_data_loads_without_errors(tmp_root, small_frames):
    report, counts = _run(tmp_root)
    assert report.ok, report.text()
    assert counts["transactions"] == len(small_frames["transactions"])
    assert counts["customers"] == 250
    assert "RESULT   OK" in report.text()


def test_ingest_never_reads_the_truth_file(tmp_root):
    src = re.sub(r'"""[\s\S]*?"""', "", inspect.getsource(ingest))          # drop docstrings
    src = chr(10).join(line.split("#")[0] for line in src.splitlines())        # drop comments
    assert "_truth" not in src
    truth = tmp_root / "data" / "raw" / "_truth.csv"
    moved = truth.with_suffix(".moved")
    truth.rename(moved)
    try:
        report, _ = _run(tmp_root)
        assert report.ok
    finally:
        moved.rename(truth)


def test_renamed_column_only_needs_schema_change(tmp_root, tmp_path):
    """Rename a source column and a date format; change schema.yaml only; ingest still works."""
    root = tmp_path / "renamed"
    shutil.copytree(tmp_root, root)
    csv = root / "data" / "raw" / "transactions.csv"
    df = pd.read_csv(csv)
    df = df.rename(columns={"txn_date": "TransactionDate", "amount": "Amt", "direction": "DrCr"})
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"]).dt.strftime("%d/%m/%Y")
    df["DrCr"] = df["DrCr"].map({"credit": "CR", "debit": "DR"})
    df.to_csv(csv, index=False)

    schema_file = root / "config" / "schema.yaml"
    schema = yaml.safe_load(schema_file.read_text(encoding="utf-8"))
    schema["transactions"]["columns"].update({"txn_date": "TransactionDate", "amount": "Amt", "direction": "DrCr"})
    schema["transactions"]["date_format"] = "%d/%m/%Y"
    schema["transactions"]["values"] = {"direction": {"CR": "credit", "DR": "debit"}}
    schema_file.write_text(yaml.safe_dump(schema), encoding="utf-8")

    report, counts = _run(root)
    assert report.ok, report.text()
    assert counts["transactions"] == len(df)


def test_missing_mapped_column_is_an_error(tmp_root, tmp_path):
    root = tmp_path / "broken"
    shutil.copytree(tmp_root, root)
    schema_file = root / "config" / "schema.yaml"
    schema = yaml.safe_load(schema_file.read_text(encoding="utf-8"))
    schema["customers"]["columns"]["segment"] = "no_such_column"
    schema_file.write_text(yaml.safe_dump(schema), encoding="utf-8")
    report, _ = _run(root)
    assert not report.ok
    assert any("no_such_column" in e for e in report.errors)


def test_duplicates_and_orphans_are_reported(tmp_root, tmp_path):
    root = tmp_path / "dirty"
    shutil.copytree(tmp_root, root)
    acc = root / "data" / "raw" / "accounts.csv"
    df = pd.read_csv(acc)
    df = pd.concat([df, df.iloc[[0]]])                   # a duplicate account row
    df.loc[df.index[-1], "account_id"] = df["account_id"].iloc[0]
    df.loc[df.index[1], "customer_id"] = "GHOST"         # an account whose customer does not exist
    df.to_csv(acc, index=False)
    report, _ = _run(root)
    assert not report.ok
    assert any("duplicate" in e for e in report.errors)
