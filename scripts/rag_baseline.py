#!/usr/bin/env python3
"""v16.3 Phase B2: the TRUE RAG baseline.

The ablation staircase needs a literal retrieval-augmented-generation rung:
retrieve top pages for the topic, stuff them into ONE writer call, no
planner, no outline, no admission gating, no validation, no post-processing.
This is what "just use RAG" means in practice, and it is the condition the
reviewers' "RAG" framing refers to. (Phase B — the validation-gate ablation
— runs the FULL pipeline minus validation; this runs almost none of it.)

Scored with the same scripts/check_citations.py as Phases A/B, so E1/E2/E3
ARE comparable (unlike Phase C's weak built-in counter): the model is given
real corpus pages and asked to cite them canonically — every failure mode
the contract exists to catch is observable here.

Usage:
    PYTHONPATH=src python scripts/rag_baseline.py \
        --n 100 --topic "..." --metadata metadata.csv \
        --output-dir runs/phase_b2 [--model TAG] [--docs-k 12] [--pages-per-doc 1]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from rrr.llm import install as _install_llm_shim  # timeout + RRR_RUNTIME parity
_install_llm_shim()
from rrr.retrieve import retrieve_breadth
from rrr.validate import load_page_text


def build_prompt(topic, hits, clip_chars=2400):
    blocks = []
    for h in hits:
        text = load_page_text(h["doc_id"], h["page"]) or h.get("text") or ""
        text = re.sub(r"\s+", " ", text).strip()[:clip_chars]
        blocks.append(f"[{h['doc_id']}: p.{h['page']}]\n{text}")
    excerpts = "\n\n".join(blocks)
    system = (
        "You are an academic research assistant specialising in economic history. "
        "Write scholarly prose with in-text citations."
    )
    user = (
        f"Write a literature review of approximately 900-1400 words on the topic:\n\n"
        f"{topic}\n\n"
        f"Use ONLY the excerpts below as your evidence base. Cite with the exact "
        f"format (DocId: p.N) matching the excerpt headers — for example "
        f"({hits[0]['doc_id']}: p.{hits[0]['page']}). Every factual claim needs a "
        f"citation. Do not cite anything not shown below. Do not use outside "
        f"knowledge.\n\n"
        f"EXCERPTS:\n\n{excerpts}\n\n"
        f"Write the review now (prose only, no headings, no bibliography)."
    )
    return system, user


def main():
    ap = argparse.ArgumentParser(description="True RAG baseline (Phase B2)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--topic", required=True)
    ap.add_argument("--metadata", default="metadata.csv")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--docs-k", type=int, default=12,
                    help="Distinct documents to retrieve from")
    ap.add_argument("--pages-per-doc", type=int, default=1)
    ap.add_argument("--phase-label", default="phase_b2")
    args = ap.parse_args()

    model = args.model or os.environ.get(
        "RRR_MODEL_LATIN", os.environ.get("RRR_MODEL", "mistral-small:24b"))
    os.makedirs(args.output_dir, exist_ok=True)
    checker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_citations.py")

    # Retrieval is deterministic (BM25): do it once, reuse across runs. The
    # measured between-run variance is therefore the GENERATION variance —
    # exactly what the condition isolates.
    hits = retrieve_breadth(args.topic, docs_k=args.docs_k,
                            pages_per_doc=args.pages_per_doc)
    if not hits:
        sys.exit("[B2] retrieval returned nothing — is the index built?")
    system, user = build_prompt(args.topic, hits)
    retrieval_manifest = [{"doc_id": h["doc_id"], "page": h["page"],
                           "score": h.get("bm25_score")} for h in hits]
    with open(os.path.join(args.output_dir, "retrieval_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump({"topic": args.topic, "model": model,
                   "docs_k": args.docs_k, "pages_per_doc": args.pages_per_doc,
                   "pages": retrieval_manifest}, f, indent=2)

    print(f"[B2] RAG baseline: {args.n} runs, model={model}, "
          f"{len(hits)} retrieved pages, prompt~{len(user)} chars")

    import ollama
    per_run = []
    for i in range(1, args.n + 1):
        run_dir = os.path.join(args.output_dir, f"run_{i:03d}")
        os.makedirs(run_dir, exist_ok=True)
        t0 = time.time()
        row = {"run": i, "refusal": False}
        try:
            res = ollama.chat(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                options={"temperature": 0.5, "num_ctx": 16384,
                         "num_predict": 1800},
                keep_alive="30m", stream=False,
            )
            text = (res.get("message", {}).get("content") or "").strip()
        except Exception as e:
            print(f"  [B2] run {i}/{args.n} ERROR: {e}")
            row["error"] = str(e)
            per_run.append(row)
            continue
        row["elapsed_s"] = round(time.time() - t0, 1)
        essay = os.path.join(run_dir, "review_composed.md")
        with open(essay, "w", encoding="utf-8") as f:
            f.write(text)
        row["word_count"] = len(re.findall(r"\b\w+\b", text))

        cj = os.path.join(run_dir, "citations.json")
        try:
            subprocess.run([sys.executable, checker, essay,
                            "--metadata", args.metadata, "--json", cj],
                           check=False, capture_output=True, timeout=120)
        except Exception as ce:  # v16.4: a checker timeout must not kill the phase
            row["checker_error"] = str(ce)
        if os.path.isfile(cj):
            try:
                with open(cj, encoding="utf-8") as f:
                    row.update(json.load(f))
            except Exception:
                row["checker_error"] = "unparseable citations.json"
        with open(os.path.join(run_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump({"refusal": False}, f)
        per_run.append(row)
        print(f"  [B2] run {i}/{args.n} — e1={row.get('e1','?')} "
              f"e2={row.get('e2','?')} e3={row.get('e3','?')} "
              f"docs={row.get('n_docs_cited','?')} words={row['word_count']} "
              f"({row['elapsed_s']:.0f}s)")

    completed = [r for r in per_run if "error" not in r]

    def stat(key):
        vals = [r[key] for r in completed if isinstance(r.get(key), (int, float))]
        if not vals:
            return {"mean": 0, "sd": 0, "min": 0, "max": 0}
        return {"mean": round(statistics.mean(vals), 2),
                "sd": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0,
                "min": min(vals), "max": max(vals)}

    # v16.4: scored subset + quote errors e4/e5 in the zero-error definition.
    scored = [r for r in completed if isinstance(r.get("e1"), (int, float))]
    zero_err = sum(1 for r in scored
                   if r.get("e1", 1) == 0 and r.get("e2", 1) == 0
                   and r.get("e3", 1) == 0 and r.get("e4", 1) == 0
                   and r.get("e5", 1) == 0)
    summary = {
        "phase": args.phase_label,
        "n_runs": len(per_run),
        "n_completed": len(completed),
        "n_refused": 0,
        "refusal_rate_pct": 0,
        "model": model,
        "topic": args.topic,
        "retrieval": {"docs_k": args.docs_k, "pages_per_doc": args.pages_per_doc,
                      "n_pages": len(hits)},
        "n_scored": len(scored),
        "n_checker_failed": len(completed) - len(scored),
        "e1": stat("e1"), "e2": stat("e2"), "e3": stat("e3"),
        "e4": stat("e4"), "e5": stat("e5"),
        "n_citations": stat("n_citations"),
        "e1_loose_advisory": stat("e1_loose_advisory_count"),
        "docs_cited": stat("n_docs_cited"),
        "words": stat("word_count"),
        "zero_err_pct": round(100 * zero_err / (len(scored) or 1), 1),
        "note": "True RAG: retrieval -> single generation, no contract. "
                "Scored by check_citations.py (E-comparable to A/B).",
    }
    out = {"summary": summary, "per_run": per_run}
    out_path = os.path.join(args.output_dir, f"{args.phase_label}_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[B2] complete: {len(completed)}/{len(per_run)} | "
          f"E1={summary['e1']['mean']}+/-{summary['e1']['sd']} | "
          f"zero-error={summary['zero_err_pct']}% | {out_path}")


if __name__ == "__main__":
    main()
