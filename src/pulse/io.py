"""Parquet read/write through DuckDB.

pyarrow's native library is blocked by Windows Application Control on the build
machine, so pandas' own parquet support is unavailable. DuckDB can read and write
parquet from a DataFrame and is already a dependency.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.register("df_to_write", df)
        con.execute(f"COPY df_to_write TO '{path.as_posix()}' (FORMAT PARQUET)")
    finally:
        con.close()


def read_parquet(path: Path) -> pd.DataFrame:
    con = duckdb.connect()
    try:
        return con.execute(f"SELECT * FROM read_parquet('{Path(path).as_posix()}')").df()
    finally:
        con.close()
