#!/usr/bin/env python3
"""v16.8: assemble a plain-text corpus for the off-the-shelf Claude arms.

The Claude comparison arms must NOT be handed raw PDFs (Claude Code cannot
ingest a 50-PDF corpus — it errors "PDF too large") and must NOT be handed
RRR's pre-retrieved passages (that would inject RRR's retrieval, defeating
the "off-the-shelf" comparison). Instead they get the SAME extracted text
RRR works from — one file per document, with [p.N] page markers — and the
agent does its own reading/retrieval, exactly as a researcher's agent would.

Reads data/page_text/<doc_id>_page_<N>.txt (produced by Step 1 preprocess)
and writes <out>/<doc_id>.txt with per-page [p.N] markers, so an unenforced
agent can still produce page-anchored `(doc_id: p.N)` citations that
scripts/check_citations.py can validate and quote-verify.

Deterministic: identical page-text in → identical corpus-text out.

Usage:
    python scripts/build_corpus_text.py \
        --page-text data/page_text --out <workspace>/corpus_text
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser(description="Assemble per-document text corpus with page markers")
    ap.add_argument("--page-text", default="data/page_text",
                    help="Directory of <doc_id>_page_<N>.txt files")
    ap.add_argument("--out", required=True, help="Output directory for <doc_id>.txt files")
    args = ap.parse_args()

    if not os.path.isdir(args.page_text):
        sys.exit(f"[build_corpus_text] page-text dir not found: {args.page_text}\n"
                 f"Run Step 1 (preprocess) first.")

    # Group page files by doc_id. Filenames are <doc_id>_page_<N>.txt and the
    # doc_id itself can contain underscores (e.g. AcemogluEtAl_2001), so split
    # on the LAST '_page_'.
    pages_by_doc = defaultdict(list)
    pat = re.compile(r"^(.*)_page_(\d+)\.txt$")
    for fn in os.listdir(args.page_text):
        m = pat.match(fn)
        if not m:
            continue
        pages_by_doc[m.group(1)].append((int(m.group(2)), fn))

    if not pages_by_doc:
        sys.exit(f"[build_corpus_text] no *_page_*.txt files in {args.page_text}")

    os.makedirs(args.out, exist_ok=True)
    n_docs = 0
    n_pages = 0
    for doc_id in sorted(pages_by_doc):
        parts = [f"# {doc_id}\n"]
        for page_no, fn in sorted(pages_by_doc[doc_id]):
            with open(os.path.join(args.page_text, fn), encoding="utf-8", errors="replace") as f:
                text = f.read().strip()
            parts.append(f"\n[p.{page_no}]\n{text}\n")
            n_pages += 1
        with open(os.path.join(args.out, f"{doc_id}.txt"), "w", encoding="utf-8") as f:
            f.write("".join(parts))
        n_docs += 1

    print(f"[build_corpus_text] wrote {n_docs} document(s), {n_pages} page(s) -> {args.out}")


if __name__ == "__main__":
    main()
