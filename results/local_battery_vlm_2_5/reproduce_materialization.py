"""Reapply the vlm.2.5 quotation verifier to the deposited vlm.2.4 reviews.

The verifier compares every quotation with the retained page text of the
corpus, so a prepared corpus work directory (one holding data/page_text/ and
metadata.csv after preprocessing) is required. The script rebuilds the
pre-materialization text of the battery from reviews_raw/ and
reviews_pre_materialization/, applies the verifier shipped in the installed
vlm.2.5 package, and compares the reviews it changes with materialization.json.

    python results/local_battery_vlm_2_5/reproduce_materialization.py --workdir <prepared corpus directory>

Requires the package installed from the repository root (`python -m pip install .`).
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True, type=Path,
                        help="prepared corpus work directory with data/page_text/ and metadata.csv")
    parser.add_argument("--metadata", type=Path, default=None,
                        help="metadata.csv (default: <workdir>/metadata.csv)")
    args = parser.parse_args()
    workdir = args.workdir.resolve()
    os.environ["RRR_PROJECT_ROOT"] = str(workdir)
    metadata = (args.metadata or workdir / "metadata.csv").resolve()
    if not (workdir / "data" / "page_text").is_dir():
        print(f"no page text under {workdir / 'data' / 'page_text'}", file=sys.stderr)
        return 2

    from rrr import writer
    from rrr.render import _build_display_lookup

    with metadata.open(encoding="utf-8-sig", newline="") as handle:
        docs = {row["doc_id"] for row in csv.DictReader(handle)}
    lookup = _build_display_lookup(docs)

    record = json.loads((BASE / "materialization.json").read_text(encoding="utf-8"))
    expected = {c["item"] for c in record["changed_reviews"]}

    changed = []
    checked = 0
    for review in sorted((BASE / "reviews_raw").rglob("*.md")):
        batch, condition, name = review.parts[-3], review.parts[-2], review.stem
        item = f"{batch}/{condition}/{name}"
        before = BASE / "reviews_pre_materialization" / batch / condition / review.name
        source = before if before.is_file() else review
        text = source.read_text(encoding="utf-8")
        out, stats = writer._strip_fabricated_quotes(text, docs, display_lookup=lookup)
        checked += int(stats.get("checked_quotes") or 0)
        if out != text:
            changed.append(item)
            print(f"changed: {item} ({len(stats.get('fabrications') or [])} action(s))")
    print(f"\n{checked} quotations checked; {len(changed)} reviews changed by the verifier.")
    missing = sorted(expected - set(changed))
    extra = sorted(set(changed) - expected)
    if missing or extra:
        print(f"mismatch with materialization.json: missing {missing}, extra {extra}")
        return 1
    print("The changed reviews match materialization.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
