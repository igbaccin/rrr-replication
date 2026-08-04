"""Reproduce the vlm.2.1 citation normalization from the deposited vlm.2.0 reviews.

For each raw review under reviews_raw/, apply the shipped normalizer
rrr._normalize_leaked_doc_id_citations (gated by the run's allowed docs) and
report which reviews change. This reproduces the deterministic surface transform
that reclassified 41 checker-failures to success (see per_run.csv columns
outcome_asrun -> outcome). Requires the vlm.2.1 package installed
from the repository root (`python -m pip install .`).

Full E1-E5 re-scoring additionally needs the corpus (metadata + page text);
this script verifies the surface transform, which is corpus-independent.
"""
import json, glob, os
from rrr.writer import _normalize_leaked_doc_id_citations

BASE = os.path.dirname(__file__)
allowed = json.load(open(os.path.join(BASE, "allowed_docs.json"), encoding="utf-8"))
changed = 0
for md in glob.glob(os.path.join(BASE, "reviews_raw", "**", "*.md"), recursive=True):
    parts = md.replace("\\", "/").split("/")
    key = f"{parts[-3]}/{parts[-2]}/{parts[-1][:-3]}"
    raw = open(md, encoding="utf-8", errors="replace").read()
    norm, n = _normalize_leaked_doc_id_citations(raw, set(allowed.get(key, [])))
    if norm != raw:
        changed += 1
        print(f"normalized ({n} surface edits): {key}")
print(f"\n{changed} reviews changed by the normalizer. "
      f"Cross-check reclassifications in per_run.csv (outcome_asrun -> outcome).")
