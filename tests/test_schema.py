"""schema.yaml is the only place source column names live, and it must be complete."""
import re
from pathlib import Path

from pulse.config import PROJECT_ROOT, load_schema
from pulse.ingest import CANONICAL


def test_schema_covers_every_canonical_column():
    schema = load_schema()
    for table, cols in CANONICAL.items():
        assert table in schema, f"schema.yaml has no section for {table}"
        assert "path" in schema[table]
        missing = set(cols) - set(schema[table]["columns"])
        assert not missing, f"{table}: canonical columns without a mapping: {sorted(missing)}"


def test_no_hardcoded_csv_paths_in_pipeline_code():
    """Code must get file locations from the YAML, not spell out data/raw/x.csv itself."""
    src_dir = PROJECT_ROOT / "src" / "pulse"
    for py in src_dir.rglob("*.py"):
        if py.name == "synth.py":       # the generator writes the files, it is not the pipeline
            continue
        text = py.read_text(encoding="utf-8")
        text = re.sub(r'"""[\s\S]*?"""', "", text)            # ignore docstrings
        text = "\n".join(line.split("#")[0] for line in text.splitlines())
        assert "data/raw/" not in text, f"{py.name} hardcodes a raw data path"
