#!/usr/bin/env python3
"""Evaluate ingest cascade against the ground-truth metadata.csv.

Runs the v15.8 auto-metadata cascade on every PDF in the corpus, compares the
extracted (doc_id, authors, year, title) against the hand-curated ground truth
row for the same PDF filename, and writes a diff report.

The critical thing to watch is high_confidence_but_wrong — those bypass the
review table and would ship silently. Everything low_confidence gets surfaced
to the user for review, so wrong-low is fine.

Usage:
  python scripts/eval_ingest.py \
      --corpus corpus/ \
      --ground-truth metadata.csv \
      --output-dir runs/ingest_eval/ \
      [--sidecar-bib bibliography.bib] \
      [--no-llm] [--no-crossref] [--no-openalex] \
      [--llm-model mistral-small:24b]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rrr.ingest import ingest_corpus, ExtractedMeta  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pdf_stem_from_any_path(p: str) -> str:
    """Normalise Windows or Unix path → filename stem. Path().stem on Linux
    doesn't split on backslashes, so 'D:\\Corpus\\foo.pdf' returns
    'D:\\Corpus\\foo' — we normalise first.
    """
    if not p:
        return ""
    normalised = str(p).replace("\\", "/")
    name = normalised.rsplit("/", 1)[-1]
    if name.lower().endswith(".pdf"):
        name = name[:-4]
    return name


def load_ground_truth(path: Path) -> dict:
    """Returns dict keyed by filename stem (matches how the cascade sees the PDF).
    Falls back to also indexing by doc_id.
    """
    rows_by_stem = {}
    rows_by_docid = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            pdf_path = r.get("pdf_path", "") or ""
            stem = _pdf_stem_from_any_path(pdf_path) or r.get("doc_id", "")
            if stem:
                rows_by_stem[stem] = r
            if r.get("doc_id"):
                rows_by_docid[r["doc_id"]] = r
    return {"by_stem": rows_by_stem, "by_docid": rows_by_docid}


def normalize_surnames(s: str) -> list:
    """Split on ' & ' or ';' or ',', normalise to lowercase surname tokens."""
    if not s:
        return []
    parts = re.split(r"\s*&\s*|;\s*", s)
    out = []
    for p in parts:
        surname = p.split(",")[0].strip() if "," in p else p.strip().split()[-1] if p.strip() else ""
        surname = surname.lower()
        surname = re.sub(r"[^a-z]", "", surname)
        if surname:
            out.append(surname)
    return out


def normalize_title(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def title_similarity(a: str, b: str) -> float:
    """Jaccard on 4-grams of normalised titles."""
    na = normalize_title(a)
    nb = normalize_title(b)
    if len(na) < 4 or len(nb) < 4:
        return 0.0
    ga = {na[i:i + 4] for i in range(len(na) - 3)}
    gb = {nb[i:i + 4] for i in range(len(nb) - 3)}
    if not (ga | gb):
        return 0.0
    return round(len(ga & gb) / len(ga | gb), 3)


def compare_row(gt: dict, extracted: ExtractedMeta) -> dict:
    m = {}
    m["gt_doc_id"] = gt.get("doc_id", "")
    m["extracted_doc_id"] = extracted.doc_id
    m["doc_id_match"] = m["gt_doc_id"].strip() == (extracted.doc_id or "").strip()
    m["gt_year"] = str(gt.get("year", "")).strip()
    m["extracted_year"] = str(extracted.year or "").strip()
    m["year_match"] = m["gt_year"] == m["extracted_year"]
    gt_surnames = normalize_surnames(gt.get("authors", ""))
    ex_surnames = normalize_surnames(extracted.authors_short)
    m["gt_authors"] = gt.get("authors", "")
    m["extracted_authors"] = extracted.authors_short
    m["first_author_match"] = bool(
        gt_surnames and ex_surnames and gt_surnames[0] == ex_surnames[0]
    )
    m["authors_len_match"] = len(gt_surnames) == len(ex_surnames)
    m["all_authors_match"] = gt_surnames == ex_surnames
    m["title_similarity"] = title_similarity(gt.get("title", ""), extracted.title)
    # v15.8.1: "correct" means the paper was identified correctly. Title
    # similarity is a diagnostic, not a gate — the bib title and the
    # hand-typed metadata title often differ in wording even though they
    # refer to the same paper. What matters for downstream provenance is
    # (first_author, year) resolving to the right doc, which the doc_id
    # column captures. Doc_id mismatch is still a WARNING (surfaces as a
    # separate stat) since it drives on-disk file layout.
    m["overall_correct"] = m["first_author_match"] and m["year_match"]
    return m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", required=True, type=Path, help="Folder with *.pdf")
    ap.add_argument("--ground-truth", required=True, type=Path, help="metadata.csv")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--sidecar-bib", type=Path, default=None)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--no-crossref", action="store_true")
    ap.add_argument("--no-openalex", action="store_true")
    ap.add_argument("--llm-model", default=None)
    args = ap.parse_args()

    if not args.corpus.is_dir():
        raise SystemExit(f"Corpus dir not found: {args.corpus}")
    if not args.ground_truth.is_file():
        raise SystemExit(f"Ground-truth CSV not found: {args.ground_truth}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    gt = load_ground_truth(args.ground_truth)
    print(f"[eval] {len(gt['by_stem'])} ground-truth rows loaded.")

    t0 = time.perf_counter()
    results = ingest_corpus(
        args.corpus,
        output_csv=args.output_dir / "extracted_metadata.csv",
        sidecar_bib=args.sidecar_bib,
        use_llm=not args.no_llm,
        use_crossref=not args.no_crossref,
        use_openalex=not args.no_openalex,
        llm_model=args.llm_model,
    )
    dt = time.perf_counter() - t0
    print(f"[eval] cascade run in {dt:.1f}s ({dt / max(1, len(results)):.1f}s/PDF)")

    # Compare each extracted row to ground truth
    report_rows = []
    for extracted in results:
        stem = _pdf_stem_from_any_path(extracted.pdf_path)
        gt_row = gt["by_stem"].get(stem, {})
        if not gt_row:
            report_rows.append({
                "pdf": stem, "gt_matched": False,
                "confidence": extracted.confidence,
                "source": extracted.source,
                "extracted_doc_id": extracted.doc_id,
                "notes": extracted.notes,
            })
            continue
        m = compare_row(gt_row, extracted)
        m["pdf"] = stem
        m["gt_matched"] = True
        m["confidence"] = extracted.confidence
        m["source"] = extracted.source
        m["notes"] = extracted.notes
        report_rows.append(m)

    # Summary
    total = len(report_rows)
    matched = sum(1 for r in report_rows if r.get("gt_matched"))
    doc_id_correct = sum(1 for r in report_rows if r.get("doc_id_match"))
    year_correct = sum(1 for r in report_rows if r.get("year_match"))
    first_author_correct = sum(1 for r in report_rows if r.get("first_author_match"))
    overall_correct = sum(1 for r in report_rows if r.get("overall_correct"))

    by_source = {}
    by_conf = {"high": 0, "medium": 0, "low": 0, "failed": 0}
    for r in report_rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
        by_conf[r.get("confidence", "failed")] = by_conf.get(r.get("confidence", "failed"), 0) + 1

    # Critical cell: high confidence but overall_correct is False
    high_but_wrong = [
        r for r in report_rows
        if r.get("confidence") == "high" and r.get("gt_matched") and not r.get("overall_correct")
    ]
    low_but_correct = [
        r for r in report_rows
        if r.get("confidence") in ("low", "medium") and r.get("overall_correct")
    ]
    failed = [r for r in report_rows if r.get("confidence") == "failed"]

    title_sims = [r.get("title_similarity", 0.0) for r in report_rows if r.get("gt_matched")]
    title_sim_avg = round(sum(title_sims) / max(1, len(title_sims)), 3)

    summary = {
        "total_pdfs": total,
        "matched_to_ground_truth": matched,
        "duration_s": round(dt, 1),
        "correctness": {
            "doc_id_exact": doc_id_correct,
            "year_correct": year_correct,
            "first_author_correct": first_author_correct,
            "overall_correct": overall_correct,
            "overall_correct_share": round(overall_correct / max(1, matched), 3),
            "avg_title_similarity": title_sim_avg,
        },
        "by_source": by_source,
        "by_confidence": by_conf,
        "critical": {
            "high_confidence_but_wrong": len(high_but_wrong),
            "low_confidence_but_correct": len(low_but_correct),
            "failed_extraction": len(failed),
        },
    }

    with (args.output_dir / "eval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (args.output_dir / "eval_rows.json").open("w", encoding="utf-8") as f:
        json.dump(report_rows, f, indent=2)

    # Print human-readable summary
    print("\n" + "=" * 70)
    print("v15.8 INGEST CASCADE — GROUND-TRUTH EVAL")
    print("=" * 70)
    print(json.dumps(summary, indent=2))
    print()
    if high_but_wrong:
        print(f"CRITICAL: {len(high_but_wrong)} rows shipped as HIGH-confidence but disagree with ground truth:")
        for r in high_but_wrong[:10]:
            print(f"  {r['pdf']}: gt={r['gt_doc_id']!r} extracted={r['extracted_doc_id']!r} "
                  f"src={r['source']} authors={r['first_author_match']} year={r['year_match']} "
                  f"title_sim={r.get('title_similarity', 0):.2f}")
        if len(high_but_wrong) > 10:
            print(f"  ... and {len(high_but_wrong) - 10} more (see eval_rows.json)")
    if failed:
        print(f"\n{len(failed)} rows FAILED completely (needs_ocr / no extractor hit):")
        for r in failed[:10]:
            print(f"  {r['pdf']}: notes={r.get('notes', '')}")
    print(f"\nArtefacts in {args.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
