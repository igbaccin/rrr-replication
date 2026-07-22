#!/usr/bin/env python3
"""v16 Phase H scorer: run the field's own citation checker over every
Claude Code arm essay and emit phase_h_results.json in the same schema the
battery aggregator and make_paper_artifacts.py consume.

Usage: python scripts/score_claude_arm.py ARM_WS --metadata metadata.csv
(ARM_WS is the directory run_claude_code_arm.sh created; essays live at
ARM_WS/runs/run_NNN/review_composed.md)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import subprocess
import sys

REFUSAL_MARKERS = (
    "cannot support a substantive review",
    "corpus cannot support",
    "insufficient evidence in the corpus",
    "unable to write the review",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_ws")
    ap.add_argument("--metadata", default="metadata.csv")
    ap.add_argument("--label", default="phase_h",
                    help="Result label/filename stem (phase_h_sonnet, phase_h_opus, phase_h3)")
    ap.add_argument("--arm-model", default="",
                    help="The pinned model id that produced the essays (recorded in the summary)")
    args = ap.parse_args()

    checker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_citations.py")
    per_run = []
    run_dirs = sorted(glob.glob(os.path.join(args.arm_ws, "runs", "run_*")))
    for rd in run_dirs:
        i = int(os.path.basename(rd).split("_")[1])
        essay = os.path.join(rd, "review_composed.md")
        row = {"run": i}
        text = ""
        if os.path.isfile(essay):
            with open(essay, encoding="utf-8", errors="replace") as f:
                text = f.read()
        words = len(re.findall(r"\b\w+\b", text))
        row["word_count"] = words
        # v16.5: the arm drivers record wall-clock per run; surface it so
        # F4 can plot H/H3 timings (elapsed_s is the key F4 reads).
        et = os.path.join(rd, "elapsed_s.txt")
        if os.path.isfile(et):
            try:
                row["elapsed_s"] = float(open(et).read().strip())
            except ValueError:
                pass
        lowered = text.lower()
        # v16.9: separate PRINCIPLED refusals (the contract's escape hatch,
        # signalled by explicit refusal language) from EMISSION FAILURES
        # (`claude -p` printed only a meta-summary — "The review above is my
        # final output..." — and the essay never reached stdout). The old
        # `words < 150` rule lumped both into "refusal", inflating the
        # refusal rate, and a 154-word 0-citation stub could slip past it
        # and score as a zero-error essay. Observed: 4/33 Opus runs.
        refused = any(m in lowered for m in REFUSAL_MARKERS)
        row["refusal"] = bool(refused)
        if refused:
            row["reason"] = "arm_refusal"
            per_run.append(row)
            with open(os.path.join(rd, "status.json"), "w", encoding="utf-8") as f:
                json.dump({"refusal": True, "reason": row["reason"]}, f)
            continue
        cj = os.path.join(rd, "citations.json")
        try:
            subprocess.run([sys.executable, checker, essay,
                            "--metadata", args.metadata, "--json", cj],
                           check=False, capture_output=True, timeout=120)
        except Exception as e:
            row["checker_error"] = str(e)
        if os.path.isfile(cj):
            try:
                with open(cj, encoding="utf-8") as f:
                    row.update(json.load(f))
            except Exception:
                row["checker_error"] = "unparseable citations.json"
        # v16.14 [auditor P1]: check_citations._empty_result returns
        # refusal=True for an empty/missing essay, and row.update() above just
        # pulled that in — clobbering the row["refusal"]=False we set from
        # REFUSAL_MARKERS. A genuine (principled) refusal already `continue`d
        # above, so a refusal flag arriving from the checker HERE really means
        # an empty output. Re-assert False so an empty/failed run is counted
        # ONCE, as an emission failure — not double-counted in both n_refused
        # AND n_emission_failed (which corrupted H/H3 refusal rates).
        row["refusal"] = False
        # A completed essay must actually BE an essay: enough words and at least
        # one citation (the contract demands 900-1500 cited words). Anything
        # shorter without refusal language is a tool failure, not model
        # behaviour — excluded from error stats AND the refusal rate, reported
        # under its own count. v16.14 [auditor P2]: the 400-word floor MUST
        # match ARM_MIN_WORDS in run_claude_code_arm.sh so the runner and scorer
        # agree on "too short".
        emission_failure = words < 400 or not row.get("n_citations")
        row["emission_failure"] = bool(emission_failure)
        status = {"refusal": False, "emission_failure": bool(emission_failure)}
        if emission_failure:
            row["reason"] = "no_output_emitted"
            status["reason"] = row["reason"]
        with open(os.path.join(rd, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f)
        per_run.append(row)

    completed = [r for r in per_run
                 if not r.get("refusal") and not r.get("emission_failure")]
    n_refused = sum(1 for r in per_run if r.get("refusal"))
    n_emission = sum(1 for r in per_run if r.get("emission_failure"))

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
        "phase": args.label,
        "arm_model": args.arm_model,
        "n_runs": len(per_run),
        "n_completed": len(completed),
        "n_refused": n_refused,
        "refusal_rate_pct": round(100 * n_refused / (len(per_run) or 1), 1),
        "n_emission_failed": n_emission,
        "emission_failure_pct": round(100 * n_emission / (len(per_run) or 1), 1),
        "n_scored": len(scored),
        "n_checker_failed": len(completed) - len(scored),
        "e1": stat("e1"), "e2": stat("e2"), "e3": stat("e3"),
        "e4": stat("e4"), "e5": stat("e5"),
        "n_citations": stat("n_citations"),
        "e1_loose_advisory": stat("e1_loose_advisory_count"),
        "docs_cited": stat("n_docs_cited"),
        "words": stat("word_count"),
        "zero_err_pct": round(100 * zero_err / (len(scored) or 1), 1),
        "note": ("RRR-as-skill inside Claude Code; RRR audit artifacts and agent transcript preserved per run"
                 if args.label == "phase_h3" else
                 "Claude Code + contract-as-prompt; scored by scripts/check_citations.py"),
    }
    out = {"summary": summary, "per_run": per_run}
    out_path = os.path.join(args.arm_ws, f"{args.label}_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[H] {len(completed)}/{len(per_run)} completed | "
          f"E1={summary['e1']['mean']}+/-{summary['e1']['sd']} | "
          f"zero-error={summary['zero_err_pct']}%")
    print(f"[H] {out_path}")


if __name__ == "__main__":
    main()
