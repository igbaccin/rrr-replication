#!/usr/bin/env python3
"""
t3_unrestricted.py — Unrestricted model baseline (Phase C).

Prompts the LLM to write a literature review with citations WITHOUT providing
any corpus artefacts. Measures fabrication rate under zero constraint.

Usage:
    python3 scripts/t3_unrestricted.py --n 20 --topic "..." --metadata metadata.csv
    python3 scripts/t3_unrestricted.py --n 20 --output-dir runs/phase_c

The --metadata flag is for post-hoc checking only: we compare the model's
invented citations against the real corpus to see if any accidentally match.
"""

import os, sys, re, json, argparse, time, statistics

CITE_STRICT = re.compile(r"\(([A-Za-z0-9_&.\-]+):\s*p\.(\d+)\)")
CITE_LOOSE = re.compile(
    r"([A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)*"
    r"(?:\s+et\s+al\.?)?)\s*\((\d{4})\)"
)


def _call_ollama(model, system_prompt, user_prompt, ctx=32768, pred=2000, temp=0.3):
    """Call Ollama chat API."""
    import ollama
    res = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": temp, "num_ctx": ctx, "num_predict": pred},
        keep_alive="30m",
        stream=False,
    )
    return (res.get("message", {}).get("content") or "").strip()


def _load_valid_docs(metadata_path):
    """Load valid doc_ids from corpus metadata."""
    if not os.path.isfile(metadata_path):
        return set()
    import pandas as pd
    df = pd.read_csv(metadata_path)
    return set(str(x) for x in df["doc_id"])


def _check_output(text, valid_docs):
    """
    Parse model output for citation-like patterns and check against corpus.

    Returns metrics dict. E2 is N/A (no corpus page data available to the model).
    """
    # Strict-format citations
    strict = [{"doc_id": m.group(1), "page": int(m.group(2))}
              for m in CITE_STRICT.finditer(text)]

    # Loose citations: Author (Year), Author and Author (Year), etc.
    loose = [{"author": m.group(1), "year": m.group(2)}
             for m in CITE_LOOSE.finditer(text)]

    # E1 strict: doc_id not in corpus
    e1_strict_fab = sum(1 for c in strict if c["doc_id"] not in valid_docs)
    e1_strict_hit = sum(1 for c in strict if c["doc_id"] in valid_docs)

    # Loose corpus matching: check if first surname appears in any doc_id
    valid_lower = {d.lower() for d in valid_docs}
    loose_hits = 0
    for lc in loose:
        first_surname = lc["author"].split()[0].lower()
        if any(first_surname in d for d in valid_lower):
            loose_hits += 1

    n_total = len(strict) + len(loose)
    e1_total = e1_strict_fab + (len(loose) - loose_hits)
    corpus_matches = e1_strict_hit + loose_hits
    word_count = len(re.findall(r"\b\w+\b", text))

    return {
        "n_strict_citations": len(strict),
        "n_loose_citations": len(loose),
        "n_total_citation_like": n_total,
        "e1": e1_total,
        "e1_strict_fabricated": e1_strict_fab,
        "corpus_matches": corpus_matches,
        "e2": "N/A",
        "e3": 0,
        "n_docs_cited": corpus_matches,
        "word_count": word_count,
    }


def main():
    ap = argparse.ArgumentParser(description="Unrestricted LLM baseline (Phase C)")
    ap.add_argument("--n", type=int, default=20, help="Number of runs")
    ap.add_argument("--topic", default="Institutions are the fundamental cause of long-run economic growth")
    ap.add_argument("--metadata", default="metadata.csv",
                    help="Corpus metadata for post-hoc checking")
    ap.add_argument("--output-dir", default="runs/phase_c_unrestricted")
    ap.add_argument("--model", default=None,
                    help="Ollama model (default: RRR_MODEL env or 'mistral')")
    args = ap.parse_args()

    model = args.model or os.environ.get("RRR_MODEL", "mistral")
    os.makedirs(args.output_dir, exist_ok=True)

    valid_docs = _load_valid_docs(args.metadata)

    system_prompt = (
        "You are an academic research assistant specialising in economic history. "
        "Write scholarly prose with in-text citations. Use the format (AuthorName_Year: p.N) "
        "for every factual claim. Include specific page numbers. "
        "Do not fabricate sources — cite only real, published scholarship."
    )
    user_prompt = (
        f"Write a literature review of approximately 800–1200 words on the following topic:\n\n"
        f"{args.topic}\n\n"
        f"Requirements:\n"
        f"- Cite specific authors, years, and page numbers\n"
        f"- Use the citation format (AuthorName_Year: p.N)\n"
        f"- Cover multiple perspectives and debates\n"
        f"- Write in flowing academic prose\n"
    )

    print(f"[Phase C] Unrestricted baseline: {args.n} runs, model={model}")
    print(f"[Phase C] Topic: {args.topic}")
    print(f"[Phase C] Output: {args.output_dir}")

    per_run = []

    for i in range(1, args.n + 1):
        run_dir = os.path.join(args.output_dir, f"run_{i:03d}")
        os.makedirs(run_dir, exist_ok=True)

        t0 = time.time()
        try:
            text = _call_ollama(model, system_prompt, user_prompt)
        except Exception as e:
            print(f"  [C] run {i}/{args.n} — ERROR: {e}")
            per_run.append({"run": i, "error": str(e), "refusal": False})
            continue
        elapsed = time.time() - t0

        # Save raw output
        with open(os.path.join(run_dir, "review_composed.md"), "w", encoding="utf-8") as f:
            f.write(text)

        result = _check_output(text, valid_docs)
        result["run"] = i
        result["elapsed_s"] = round(elapsed, 1)
        result["refusal"] = False

        with open(os.path.join(run_dir, "citations.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        # Status file (for aggregate_comprehensive.py compatibility)
        with open(os.path.join(run_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump({"refusal": False}, f)

        per_run.append(result)
        print(f"  [C] run {i}/{args.n} — "
              f"total_refs={result['n_total_citation_like']} "
              f"e1={result['e1']} "
              f"corpus_matches={result['corpus_matches']} "
              f"words={result['word_count']} ({elapsed:.0f}s)")

    # ── Aggregate ──────────────────────────────────────────────
    completed = [r for r in per_run if "error" not in r]
    n = len(completed)

    if n == 0:
        print("[Phase C] No successful runs.")
        return

    def _stat(key):
        vals = [r[key] for r in completed if isinstance(r.get(key), (int, float))]
        if not vals:
            return {"mean": 0, "sd": 0, "min": 0, "max": 0}
        return {
            "mean": round(statistics.mean(vals), 2),
            "sd": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0,
            "min": min(vals), "max": max(vals),
        }

    summary = {
        "phase": "phase_c",
        "n_runs": n,
        "n_completed": n,
        "n_refused": 0,
        "n_errors": len(per_run) - n,
        "refusal_rate_pct": 0,
        "model": model,
        "topic": args.topic,
        "e1": _stat("e1"),
        "e2": "N/A",
        "e3": {"mean": 0, "sd": 0, "min": 0, "max": 0},
        "n_total_citations": _stat("n_total_citation_like"),
        "words": _stat("word_count"),
        "docs_cited": _stat("n_docs_cited"),
        "zero_fab_pct": round(
            sum(1 for r in completed if r.get("e1", 999) == 0) / n * 100, 1
        ),
    }

    out = {"summary": summary, "per_run": per_run}
    out_path = os.path.join(args.output_dir, "phase_c_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n[Phase C] Complete. {n} runs.")
    print(f"  E1 (fabricated): {summary['e1']['mean']} ± {summary['e1']['sd']}")
    print(f"  Total refs:      {summary['n_total_citations']['mean']} ± {summary['n_total_citations']['sd']}")
    print(f"  Zero-fab rate:   {summary['zero_fab_pct']}%")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
