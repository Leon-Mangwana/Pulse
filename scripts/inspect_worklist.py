"""Print the worklist checks a reviewer does by hand after `python -m pulse.cli score`.

    python scripts/inspect_worklist.py            # all three checks
    python scripts/inspect_worklist.py --rows 5   # more reason examples per source
    python scripts/inspect_worklist.py --cards 20 # print the first 20 action cards as text
    python scripts/inspect_worklist.py --cards 20 --from 100   # cards ranked 100 to 119
    python scripts/inspect_worklist.py --segments # the segment view sorted by value at risk
"""
import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORKLIST = ROOT / "data" / "processed" / "worklist.parquet"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=3, help="reason examples per source")
    args = ap.parse_args()
    if not WORKLIST.exists():
        print(f"{WORKLIST} not found; run `python -m pulse.cli score` first")
        return 1
    wl = duckdb.sql(f"select * from '{WORKLIST.as_posix()}'").df().sort_values("rank")
    pd.set_option("display.width", 200, "display.max_colwidth", 200)

    print("=== 1. recommendations by source ===")
    print(wl["rec_source"].value_counts().to_string())

    print(f"\n=== 2. {args.rows} reasons per source (read them as a relationship manager would) ===")
    for src, grp in wl.groupby("rec_source", sort=False):
        print(f"\n[{src}]")
        for _, r in grp.head(args.rows).iterrows():
            print(f"  rank {int(r['rank']):>5}  {r['customer_ref']}  {r['recommendation']:<18} {r['rec_reason']}")

    print("\n=== 3. top 20 by value at risk must all be 'retention' ===")
    top = wl.head(20)[["rank", "customer_ref", "value_at_risk_usd", "recommendation", "rec_source"]]
    print(top.to_string(index=False))
    bad = (top["rec_source"] != "retention").sum()
    print("\nOK: every top-20 row is a retention action" if bad == 0 else f"PROBLEM: {bad} of the top 20 are sales offers")

    print("\n=== reasons that mention a phrase (use after adding a trigger) ===")
    print("  python scripts/inspect_worklist.py --find \"cash withdrawals\"")
    return 0


def _arg(flag, default):
    return int(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


if __name__ == "__main__":
    if "--cards" in sys.argv:
        import json
        from pulse.explain import card_text
        cards = json.loads((ROOT / "data" / "processed" / "cards.json").read_text(encoding="utf-8"))
        start, n = _arg("--from", 1) - 1, _arg("--cards", 20)
        for c in cards[start:start + n]:
            print(card_text(c)); print()
        sys.exit(0)
    if "--segments" in sys.argv:
        seg = pd.read_csv(ROOT / "data" / "processed" / "segments.csv")
        pd.set_option("display.width", 200)
        print(seg.sort_values("value_at_risk_usd", ascending=False).to_string(index=False))
        sys.exit(0)
    if "--find" in sys.argv:
        i = sys.argv.index("--find")
        phrase = sys.argv[i + 1]
        wl = duckdb.sql(f"select * from '{WORKLIST.as_posix()}'").df()
        hit = wl[wl["rec_reason"].str.contains(phrase, case=False, regex=False)]
        print(f"{len(hit):,} of {len(wl):,} worklist rows have a reason containing '{phrase}'")
        print(hit[["rank", "customer_ref", "recommendation", "rec_source", "rec_reason"]].head(5).to_string(index=False))
        sys.exit(0)
    sys.exit(main())
