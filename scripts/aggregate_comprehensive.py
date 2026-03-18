#!/usr/bin/env python3
"""
aggregate_comprehensive.py — Merge all phase results into one JSON.

Reads per-phase result files from a comprehensive battery directory
and produces comprehensive_results.json with:
  - per-phase summaries
  - cross-phase comparison table (Table 4)
  - refusal stress table (Table 5)
  - E2 anatomy table (Table 6)
  - adversarial robustness table (Table 7)
  - LaTeX-ready row strings for each table

Usage:
    python3 scripts/aggregate_comprehensive.py runs/comprehensive_<timestamp>
"""

import os, sys, json, glob


def _load(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _stat_str(stat_dict):
    """Format a stat dict as 'mean ± sd'."""
    if not stat_dict or not isinstance(stat_dict, dict):
        return "N/A"
    return f"{stat_dict.get('mean', 0)} ± {stat_dict.get('sd', 0)}"


def _stat_str_latex(stat_dict):
    """Format as 'mean $\\pm$ sd'."""
    if not stat_dict or not isinstance(stat_dict, dict):
        return "N/A"
    return f"{stat_dict.get('mean', 0)} $\\pm$ {stat_dict.get('sd', 0)}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/aggregate_comprehensive.py <battery_dir>")
        sys.exit(1)

    battery_dir = sys.argv[1]
    if not os.path.isdir(battery_dir):
        print(f"ERROR: {battery_dir} not found")
        sys.exit(1)

    # ── Load all phase results ──────────────────────────────────
    phases = {}

    # Phase A: Main RRR
    p = _load(os.path.join(battery_dir, "phase_a_main", "phase_a_results.json"))
    if p:
        phases["phase_a"] = p.get("summary", {})

    # Phase B: RAG-only
    p = _load(os.path.join(battery_dir, "phase_b_rag_only", "phase_b_results.json"))
    if p:
        phases["phase_b"] = p.get("summary", {})

    # Phase C: Unrestricted
    p = _load(os.path.join(battery_dir, "phase_c_unrestricted", "phase_c_results.json"))
    if p:
        phases["phase_c"] = p.get("summary", {})

    # Phase D1: High threshold
    p = _load(os.path.join(battery_dir, "phase_d1_high_threshold", "phase_d1_results.json"))
    if p:
        phases["phase_d1"] = p.get("summary", {})

    # Phase D2: Narrow prompt
    p = _load(os.path.join(battery_dir, "phase_d2_narrow_prompt", "phase_d2_results.json"))
    if p:
        phases["phase_d2"] = p.get("summary", {})

    # Phase E: E2 anatomy
    p = _load(os.path.join(battery_dir, "phase_e_e2_anatomy", "phase_e_results.json"))
    if p:
        phases["phase_e"] = p

    # Phase F1: Adversarial (institutions irrelevant)
    p = _load(os.path.join(battery_dir, "phase_f1_adversarial_irrelevant", "phase_f1_results.json"))
    if p:
        phases["phase_f1"] = p.get("summary", {})

    # Phase F2: Adversarial (culture dominates)
    p = _load(os.path.join(battery_dir, "phase_f2_adversarial_culture", "phase_f2_results.json"))
    if p:
        phases["phase_f2"] = p.get("summary", {})

    # ── Build comparison tables ─────────────────────────────────

    # TABLE 4: Cross-condition comparison (A vs B vs C)
    table4 = {"columns": ["Condition", "N", "E1", "E2", "E3", "Zero-fab %", "Words", "Docs cited"]}
    table4["rows"] = []
    table4["latex_rows"] = []

    for label, key in [("RRR (full)", "phase_a"),
                        ("RAG-only (no validation)", "phase_b"),
                        ("Unrestricted LLM", "phase_c")]:
        s = phases.get(key, {})
        if not s:
            continue
        row = {
            "condition": label,
            "n": s.get("n_runs", s.get("n_completed", 0)),
            "e1": _stat_str(s.get("e1")),
            "e2": _stat_str(s.get("e2")),
            "e3": _stat_str(s.get("e3")),
            "zero_fab": s.get("zero_fab_pct", "N/A"),
            "words": _stat_str(s.get("words", s.get("word_count"))),
            "docs_cited": _stat_str(s.get("docs_cited", s.get("n_docs_cited"))),
        }
        table4["rows"].append(row)

        # LaTeX
        e1_l = _stat_str_latex(s.get("e1"))
        e2_l = _stat_str_latex(s.get("e2"))
        e3_l = _stat_str_latex(s.get("e3"))
        zf = s.get("zero_fab_pct", "N/A")
        n_val = s.get("n_runs", s.get("n_completed", 0))
        table4["latex_rows"].append(
            f"{label} & {n_val} & {e1_l} & {e2_l} & {e3_l} & {zf}\\% \\\\"
        )

    # TABLE 5: Refusal stress test (D1 + D2)
    table5 = {"columns": ["Condition", "N", "Refusal rate %", "Top reason", "E1 (completed)", "E2 (completed)"]}
    table5["rows"] = []
    table5["latex_rows"] = []

    for label, key in [("High thresholds", "phase_d1"),
                        ("Narrow prompt", "phase_d2")]:
        s = phases.get(key, {})
        if not s:
            continue
        reasons = s.get("refusal_reasons", {})
        top_reason = max(reasons, key=reasons.get) if reasons else "N/A"
        row = {
            "condition": label,
            "n": s.get("n_runs", 0),
            "refusal_rate": s.get("refusal_rate_pct", 0),
            "top_reason": top_reason,
            "e1_completed": _stat_str(s.get("e1")),
            "e2_completed": _stat_str(s.get("e2")),
        }
        table5["rows"].append(row)
        table5["latex_rows"].append(
            f"{label} & {row['n']} & {row['refusal_rate']}\\% & "
            f"\\texttt{{{top_reason}}} & {_stat_str_latex(s.get('e1'))} & "
            f"{_stat_str_latex(s.get('e2'))} \\\\"
        )

    # TABLE 6: E2 anatomy
    table6 = {"columns": ["Run", "doc_id", "Cited page", "Max valid page", "Overshoot", "Diagnosis"]}
    table6["rows"] = []
    table6["latex_rows"] = []

    pe = phases.get("phase_e", {})
    for r in pe.get("rows", []):
        row = {
            "run_id": r.get("run_id", "?"),
            "doc_id": r.get("doc_id", "?"),
            "cited_page": r.get("cited_page", "?"),
            "max_valid_page": r.get("max_valid_page", "?"),
            "overshoot": r.get("overshoot", "?"),
            "diagnosis": r.get("diagnosis", "?"),
        }
        table6["rows"].append(row)
        table6["latex_rows"].append(
            f"{row['run_id']} & \\texttt{{{row['doc_id']}}} & "
            f"{row['cited_page']} & {row['max_valid_page']} & "
            f"+{row['overshoot']} & {row['diagnosis']} \\\\"
        )
    table6["summary"] = {
        "n_events": pe.get("n_e2_events", 0),
        "ref_truncation_count": pe.get("ref_truncation_count", 0),
        "docs_affected": pe.get("docs_affected", []),
    }

    # TABLE 7: Adversarial robustness (F1 + F2 vs A)
    table7 = {"columns": ["Prompt", "N", "Refusal %", "E1", "E2", "Docs cited", "Zero-fab %"]}
    table7["rows"] = []
    table7["latex_rows"] = []

    for label, key in [("Neutral (baseline)", "phase_a"),
                        ("'Institutions irrelevant'", "phase_f1"),
                        ("'Culture dominates'", "phase_f2")]:
        s = phases.get(key, {})
        if not s:
            continue
        row = {
            "prompt": label,
            "n": s.get("n_runs", 0),
            "refusal_rate": s.get("refusal_rate_pct", 0),
            "e1": _stat_str(s.get("e1")),
            "e2": _stat_str(s.get("e2")),
            "docs_cited": _stat_str(s.get("docs_cited")),
            "zero_fab": s.get("zero_fab_pct", "N/A"),
        }
        table7["rows"].append(row)
        table7["latex_rows"].append(
            f"{label} & {row['n']} & {row['refusal_rate']}\\% & "
            f"{_stat_str_latex(s.get('e1'))} & {_stat_str_latex(s.get('e2'))} & "
            f"{_stat_str_latex(s.get('docs_cited'))} & {row['zero_fab']}\\% \\\\"
        )

    # ── Assemble final output ───────────────────────────────────
    comprehensive = {
        "battery_dir": battery_dir,
        "phases": phases,
        "tables": {
            "table4_cross_condition": table4,
            "table5_refusal_stress": table5,
            "table6_e2_anatomy": table6,
            "table7_adversarial": table7,
        },
    }

    # Load config if present
    cfg_path = os.path.join(battery_dir, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            comprehensive["config"] = json.load(f)

    out_path = os.path.join(battery_dir, "comprehensive_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comprehensive, f, indent=2, ensure_ascii=False)

    # ── Print summary ───────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  COMPREHENSIVE RESULTS: {out_path}")
    print(f"{'='*64}")

    print(f"\n  Phases loaded: {', '.join(sorted(phases.keys()))}")

    if table4["rows"]:
        print(f"\n  ── Table 4: Cross-condition comparison ──")
        for r in table4["rows"]:
            print(f"    {r['condition']:<30} E1={r['e1']:<12} E2={r['e2']:<12} "
                  f"zero-fab={r['zero_fab']}%")

    if table5["rows"]:
        print(f"\n  ── Table 5: Refusal stress test ──")
        for r in table5["rows"]:
            print(f"    {r['condition']:<20} refusal={r['refusal_rate']}% "
                  f"reason={r['top_reason']}")

    if table6["rows"]:
        print(f"\n  ── Table 6: E2 anatomy ──")
        print(f"    {table6['summary']['n_events']} events, "
              f"{table6['summary']['ref_truncation_count']} ref-truncation")

    if table7["rows"]:
        print(f"\n  ── Table 7: Adversarial robustness ──")
        for r in table7["rows"]:
            print(f"    {r['prompt']:<30} E1={r['e1']:<12} refusal={r['refusal_rate']}% "
                  f"zero-fab={r['zero_fab']}%")

    print(f"\n  LaTeX rows written to JSON under tables.table<N>.latex_rows")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
