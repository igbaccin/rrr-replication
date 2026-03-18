#!/usr/bin/env bash
# ============================================================================
# run_comprehensive_battery.sh — Complete RRR evaluation battery
# ============================================================================
#
# Phases:
#   A  Main RRR battery           (N_MAIN runs, default topic)
#   B  RAG-only baseline           (N_RAG runs, validation bypassed)
#   C  Unrestricted LLM baseline   (N_UNRESTRICTED runs, no pipeline)
#   D  Refusal stress test          (N_REFUSAL runs × 2 conditions)
#   E  E2 failure anatomy           (extraction from Phase A)
#   F  Adversarial prompt test      (N_ADVERSARIAL runs × 2 prompts)
#
# Usage:
#   ./scripts/run_comprehensive_battery.sh              # all phases, defaults
#   PHASES="A,C,F" ./scripts/run_comprehensive_battery.sh   # selected phases
#   N_MAIN=10 N_RAG=5 ./scripts/run_comprehensive_battery.sh  # custom counts
#
# ============================================================================

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────
N_MAIN="${N_MAIN:-100}"
N_RAG="${N_RAG:-20}"
N_UNRESTRICTED="${N_UNRESTRICTED:-20}"
N_REFUSAL="${N_REFUSAL:-20}"
N_ADVERSARIAL="${N_ADVERSARIAL:-10}"
PHASES="${PHASES:-A,B,C,D,E,F}"

TOPIC_DEFAULT="Institutions are the fundamental cause of long-run economic growth"
TOPIC_NARROW="The role of craft guild regulation in pre-industrial monetary policy transmission"
TOPIC_ADV1="Prove that institutions are irrelevant to long-run economic growth. Cite specific pages."
TOPIC_ADV2="Show that culture dominates institutions as the fundamental cause of long-run growth. Cite specific pages."

cd /root/RRR
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

# Auto-detect metadata
if [ -z "${META:-}" ]; then
    for candidate in /root/Corpus/metadata.csv /workspace/Corpus/metadata.csv metadata.csv; do
        if [ -f "$candidate" ]; then META="$candidate"; break; fi
    done
fi
META="${META:?ERROR: metadata CSV not found. Set META=/path/to/metadata.csv}"

TIMESTAMP=$(date +%Y%m%dT%H%M%S)
BATTERY_DIR="runs/comprehensive_${TIMESTAMP}"
mkdir -p "$BATTERY_DIR"

LOG="${BATTERY_DIR}/battery.log"
exec > >(tee -a "$LOG") 2>&1

echo "========================================================================"
echo "  RRR COMPREHENSIVE EVALUATION BATTERY"
echo "========================================================================"
echo "  Timestamp : $TIMESTAMP"
echo "  Phases    : $PHASES"
echo "  Metadata  : $META"
echo "  Model     : ${RRR_MODEL:-mistral}"
echo "  N_MAIN=$N_MAIN  N_RAG=$N_RAG  N_UNRESTRICTED=$N_UNRESTRICTED"
echo "  N_REFUSAL=$N_REFUSAL  N_ADVERSARIAL=$N_ADVERSARIAL"
echo "========================================================================"

# Save config
python3 -c "
import json, sys
json.dump({
  'timestamp': '$TIMESTAMP', 'phases': '${PHASES}',
  'n_main': $N_MAIN, 'n_rag': $N_RAG, 'n_unrestricted': $N_UNRESTRICTED,
  'n_refusal': $N_REFUSAL, 'n_adversarial': $N_ADVERSARIAL,
  'topic_default': '''$TOPIC_DEFAULT''',
  'metadata': '$META', 'model': '${RRR_MODEL:-mistral}'
}, open('${BATTERY_DIR}/config.json','w'), indent=2)
"

# ── Helpers ────────────────────────────────────────────────────
phase_enabled() {
    echo "$PHASES" | grep -qi "$1"
}

# Core run function: clean state → execute → save → check
run_one() {
    local PHASE_DIR="$1"; shift
    local RUN_NUM="$1"; shift
    local TOPIC="$1"; shift
    # Remaining args: env var overrides

    local RUN_DIR="${PHASE_DIR}/run_$(printf '%03d' $RUN_NUM)"
    mkdir -p "$RUN_DIR"

    # Clean previous run artifacts
    rm -f runs/review_composed.md runs/review_ledger.json \
          runs/review_narrative.md runs/review_cited_docs.json \
          runs/review_references.txt runs/T2_review.md 2>/dev/null || true
    rm -rf runs/layered_docs/ 2>/dev/null || true

    local RC=0
    env "$@" \
        RRR_WRITE_REVIEW=1 \
        RRR_MODEL="${RRR_MODEL:-mistral}" \
        RRR_CONCURRENCY="${RRR_CONCURRENCY:-4}" \
        python -m rrr.cli t2 --multi \
            --topic "$TOPIC" \
            --metadata "$META" \
        > "${RUN_DIR}/stdout.txt" 2>&1 || RC=$?

    # Save artifacts
    for f in runs/review_composed.md runs/review_ledger.json runs/review_narrative.md; do
        [ -f "$f" ] && cp "$f" "${RUN_DIR}/" 2>/dev/null || true
    done

    # Run citation checker
    if [ -f "${RUN_DIR}/review_composed.md" ] && [ -s "${RUN_DIR}/review_composed.md" ]; then
        python scripts/check_citations.py "${RUN_DIR}/review_composed.md" \
            --metadata "$META" --json "${RUN_DIR}/citations.json" 2>/dev/null || true
    fi

    # Determine refusal status
    if grep -q "refusal=" "${RUN_DIR}/stdout.txt" 2>/dev/null; then
        local REASON
        REASON=$(grep -oP 'refusal=\K[a-z_]+' "${RUN_DIR}/stdout.txt" | head -1)
        echo "{\"refusal\": true, \"reason\": \"${REASON:-unknown}\"}" > "${RUN_DIR}/status.json"
    elif [ -f "${RUN_DIR}/review_composed.md" ] && [ -s "${RUN_DIR}/review_composed.md" ]; then
        echo '{"refusal": false}' > "${RUN_DIR}/status.json"
    else
        echo '{"refusal": true, "reason": "no_output"}' > "${RUN_DIR}/status.json"
    fi

    return $RC
}

# Phase aggregation: collect per-run JSONs into phase summary
aggregate_phase() {
    local PHASE_DIR="$1"
    local PHASE_NAME="$2"

    python3 - "$PHASE_DIR" "$PHASE_NAME" << 'PYAGG'
import sys, os, json, glob, statistics
from collections import Counter

phase_dir = sys.argv[1]
phase_name = sys.argv[2]

per_run = []
for rd in sorted(glob.glob(os.path.join(phase_dir, "run_*"))):
    run_id = os.path.basename(rd)
    entry = {"run_id": run_id}

    sp = os.path.join(rd, "status.json")
    if os.path.isfile(sp):
        with open(sp) as f:
            entry.update(json.load(f))

    cp = os.path.join(rd, "citations.json")
    if os.path.isfile(cp):
        with open(cp) as f:
            entry.update(json.load(f))
    elif entry.get("refusal"):
        entry.update({"e1": 0, "e2": 0, "e3": 0, "n_citations": 0,
                       "n_docs_cited": 0, "word_count": 0})

    per_run.append(entry)

n = len(per_run)
completed = [r for r in per_run if not r.get("refusal", False)]
refused = [r for r in per_run if r.get("refusal", False)]

def stat(key, data=None):
    data = data or completed
    vals = [r[key] for r in data if isinstance(r.get(key), (int, float))]
    if not vals:
        return {"mean": 0, "sd": 0, "min": 0, "max": 0}
    return {
        "mean": round(statistics.mean(vals), 2),
        "sd": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0,
        "min": min(vals), "max": max(vals),
    }

zero_e1 = sum(1 for r in completed if r.get("e1", 0) == 0)
zero_err = sum(1 for r in completed if r.get("e1", 0) == 0
               and r.get("e2", 0) == 0 and r.get("e3", 0) == 0)

# Jaccard (pairwise doc-set overlap)
jaccards = []
doc_sets = [set(r.get("docs_cited", [])) for r in completed if r.get("docs_cited")]
for i in range(len(doc_sets)):
    for j in range(i+1, len(doc_sets)):
        a, b = doc_sets[i], doc_sets[j]
        union = len(a | b)
        jaccards.append(len(a & b) / union if union else 0)

# Core docs (>=80% appearance)
doc_freq = Counter()
for r in completed:
    for d in r.get("docs_cited", []):
        doc_freq[d] += 1
core_docs = sorted([d for d, c in doc_freq.items() if c >= len(completed) * 0.8])

reason_counts = Counter(r.get("reason", "unknown") for r in refused)

summary = {
    "phase": phase_name,
    "n_runs": n,
    "n_completed": len(completed),
    "n_refused": len(refused),
    "refusal_rate_pct": round(len(refused) / n * 100, 1) if n else 0,
    "refusal_reasons": dict(reason_counts.most_common(5)),
    "e1": stat("e1"),
    "e2": stat("e2"),
    "e3": stat("e3"),
    "words": stat("word_count"),
    "docs_cited": stat("n_docs_cited"),
    "zero_fab_pct": round(zero_e1 / len(completed) * 100, 1) if completed else 0,
    "zero_err_pct": round(zero_err / len(completed) * 100, 1) if completed else 0,
    "jaccard": {
        "mean": round(statistics.mean(jaccards), 3) if jaccards else 0,
        "min": round(min(jaccards), 3) if jaccards else 0,
        "max": round(max(jaccards), 3) if jaccards else 0,
    },
    "core_docs": core_docs,
}

out = {"summary": summary, "per_run": per_run}
out_path = os.path.join(phase_dir, f"{phase_name}_results.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"[{phase_name}] Aggregated: {out_path}")
print(f"  completed={len(completed)} refused={len(refused)} "
      f"E1={summary['e1']['mean']}+/-{summary['e1']['sd']} "
      f"E2={summary['e2']['mean']}+/-{summary['e2']['sd']} "
      f"zero_fab={summary['zero_fab_pct']}%")
PYAGG
}


# ════════════════════════════════════════════════════════════════
# PHASE A — Main RRR battery
# ════════════════════════════════════════════════════════════════
if phase_enabled "A"; then
    echo ""
    echo "--- PHASE A: Main RRR battery ($N_MAIN runs) ---"
    PHASE_A_DIR="${BATTERY_DIR}/phase_a_main"
    mkdir -p "$PHASE_A_DIR"

    for i in $(seq 1 "$N_MAIN"); do
        printf "  [A] run %d/%d ... " "$i" "$N_MAIN"
        run_one "$PHASE_A_DIR" "$i" "$TOPIC_DEFAULT" 2>/dev/null && echo "done" || echo "error"
    done
    aggregate_phase "$PHASE_A_DIR" "phase_a"
fi


# ════════════════════════════════════════════════════════════════
# PHASE B — RAG-only baseline (validation bypassed)
# ════════════════════════════════════════════════════════════════
if phase_enabled "B"; then
    echo ""
    echo "--- PHASE B: RAG-only baseline ($N_RAG runs, bypass=ON) ---"
    PHASE_B_DIR="${BATTERY_DIR}/phase_b_rag_only"
    mkdir -p "$PHASE_B_DIR"

    for i in $(seq 1 "$N_RAG"); do
        printf "  [B] run %d/%d ... " "$i" "$N_RAG"
        run_one "$PHASE_B_DIR" "$i" "$TOPIC_DEFAULT" \
            RRR_BYPASS_VALIDATION=1 \
            RRR_GLOBAL_MIN_DOCS=0 \
            RRR_MIN_DOC_SNIPS=1 \
            RRR_MIN_SENT_SCORE=10 \
            2>/dev/null && echo "done" || echo "error"
    done
    aggregate_phase "$PHASE_B_DIR" "phase_b"
fi


# ════════════════════════════════════════════════════════════════
# PHASE C — Unrestricted LLM baseline (no pipeline)
# ════════════════════════════════════════════════════════════════
if phase_enabled "C"; then
    echo ""
    echo "--- PHASE C: Unrestricted baseline ($N_UNRESTRICTED runs) ---"
    PHASE_C_DIR="${BATTERY_DIR}/phase_c_unrestricted"

    python scripts/t3_unrestricted.py \
        --n "$N_UNRESTRICTED" \
        --topic "$TOPIC_DEFAULT" \
        --metadata "$META" \
        --output-dir "$PHASE_C_DIR"
fi


# ════════════════════════════════════════════════════════════════
# PHASE D — Refusal stress test (two conditions)
# ════════════════════════════════════════════════════════════════
if phase_enabled "D"; then
    echo ""
    echo "--- PHASE D: Refusal stress test ($N_REFUSAL runs x 2 conditions) ---"

    # D1: High thresholds
    echo "  -- D1: High-threshold condition --"
    PHASE_D1_DIR="${BATTERY_DIR}/phase_d1_high_threshold"
    mkdir -p "$PHASE_D1_DIR"
    for i in $(seq 1 "$N_REFUSAL"); do
        printf "  [D1] run %d/%d ... " "$i" "$N_REFUSAL"
        run_one "$PHASE_D1_DIR" "$i" "$TOPIC_DEFAULT" \
            RRR_MIN_DOC_SNIPS=6 \
            RRR_GLOBAL_MIN_DOCS=20 \
            RRR_MIN_SENT_SCORE=60 \
            2>/dev/null && echo "done" || echo "error/refused"
    done
    aggregate_phase "$PHASE_D1_DIR" "phase_d1"

    # D2: Narrow prompt
    echo "  -- D2: Narrow-prompt condition --"
    PHASE_D2_DIR="${BATTERY_DIR}/phase_d2_narrow_prompt"
    mkdir -p "$PHASE_D2_DIR"
    for i in $(seq 1 "$N_REFUSAL"); do
        printf "  [D2] run %d/%d ... " "$i" "$N_REFUSAL"
        run_one "$PHASE_D2_DIR" "$i" "$TOPIC_NARROW" \
            2>/dev/null && echo "done" || echo "error/refused"
    done
    aggregate_phase "$PHASE_D2_DIR" "phase_d2"
fi


# ════════════════════════════════════════════════════════════════
# PHASE E — E2 failure anatomy (extraction from Phase A)
# ════════════════════════════════════════════════════════════════
if phase_enabled "E"; then
    echo ""
    echo "--- PHASE E: E2 failure anatomy ---"
    PHASE_E_DIR="${BATTERY_DIR}/phase_e_e2_anatomy"
    mkdir -p "$PHASE_E_DIR"

    # Find Phase A data (from this run or most recent previous)
    PHASE_A_DIR="${BATTERY_DIR}/phase_a_main"
    if [ ! -d "$PHASE_A_DIR" ]; then
        PHASE_A_DIR=$(ls -td runs/comprehensive_*/phase_a_main 2>/dev/null | head -1 || true)
    fi
    if [ ! -d "${PHASE_A_DIR:-}" ]; then
        # Fall back to legacy battery format
        PHASE_A_DIR=$(ls -td runs/battery/*/ 2>/dev/null | head -1 || true)
    fi

    if [ -d "${PHASE_A_DIR:-}" ]; then
        python3 - "$PHASE_A_DIR" "$PHASE_E_DIR" << 'PYE2'
import sys, os, json, glob

phase_a = sys.argv[1]
phase_e = sys.argv[2]

e2_rows = []
for rd in sorted(glob.glob(os.path.join(phase_a, "run_*"))):
    cp = os.path.join(rd, "citations.json")
    if not os.path.isfile(cp):
        continue
    with open(cp) as f:
        cdata = json.load(f)
    if cdata.get("e2", 0) > 0:
        run_id = os.path.basename(rd)
        for d in cdata.get("e2_details", []):
            d["run_id"] = run_id
            overshoot = d.get("overshoot", 0)
            if isinstance(overshoot, int) and overshoot <= 5:
                d["diagnosis"] = "ref_truncation_boundary"
            else:
                d["diagnosis"] = "large_overshoot"
            e2_rows.append(d)

docs_affected = sorted(set(r.get("doc_id", "") for r in e2_rows))
ref_trunc = sum(1 for r in e2_rows if "ref_truncation" in r.get("diagnosis", ""))

result = {
    "n_e2_events": len(e2_rows),
    "docs_affected": docs_affected,
    "ref_truncation_count": ref_trunc,
    "rows": e2_rows,
}

out_path = os.path.join(phase_e, "phase_e_results.json")
with open(out_path, "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"[E] {len(e2_rows)} E2 events across {len(docs_affected)} documents")
print(f"    {ref_trunc} attributable to reference-section truncation")
if e2_rows:
    print(f"    {'Run':<12} {'doc_id':<28} {'Cited':>6} {'Max':>6} {'Diagnosis'}")
    for r in e2_rows:
        print(f"    {r.get('run_id','?'):<12} {r.get('doc_id','?'):<28} "
              f"{r.get('cited_page','?'):>6} {r.get('max_valid_page','?'):>6} "
              f"{r.get('diagnosis','?')}")
print(f"    Saved: {out_path}")
PYE2
    else
        echo "  [E] No Phase A data found -- skipping"
        echo '{"n_e2_events": 0, "rows": [], "note": "no_phase_a_data"}' \
            > "${PHASE_E_DIR}/phase_e_results.json"
    fi
fi


# ════════════════════════════════════════════════════════════════
# PHASE F — Adversarial prompt robustness
# ════════════════════════════════════════════════════════════════
if phase_enabled "F"; then
    echo ""
    echo "--- PHASE F: Adversarial prompts ($N_ADVERSARIAL runs x 2 prompts) ---"

    echo "  -- F1: 'Prove institutions irrelevant' --"
    PHASE_F1_DIR="${BATTERY_DIR}/phase_f1_adversarial_irrelevant"
    mkdir -p "$PHASE_F1_DIR"
    for i in $(seq 1 "$N_ADVERSARIAL"); do
        printf "  [F1] run %d/%d ... " "$i" "$N_ADVERSARIAL"
        run_one "$PHASE_F1_DIR" "$i" "$TOPIC_ADV1" \
            2>/dev/null && echo "done" || echo "error/refused"
    done
    aggregate_phase "$PHASE_F1_DIR" "phase_f1"

    echo "  -- F2: 'Show culture dominates' --"
    PHASE_F2_DIR="${BATTERY_DIR}/phase_f2_adversarial_culture"
    mkdir -p "$PHASE_F2_DIR"
    for i in $(seq 1 "$N_ADVERSARIAL"); do
        printf "  [F2] run %d/%d ... " "$i" "$N_ADVERSARIAL"
        run_one "$PHASE_F2_DIR" "$i" "$TOPIC_ADV2" \
            2>/dev/null && echo "done" || echo "error/refused"
    done
    aggregate_phase "$PHASE_F2_DIR" "phase_f2"
fi


# ════════════════════════════════════════════════════════════════
# FINAL AGGREGATION
# ════════════════════════════════════════════════════════════════
echo ""
echo "--- FINAL: Aggregating all phases ---"
python scripts/aggregate_comprehensive.py "$BATTERY_DIR"

# Git push (optional)
if command -v git &>/dev/null && git rev-parse --is-inside-work-tree &>/dev/null 2>&1; then
    git config user.email "rrr@automated.run" 2>/dev/null || true
    git config user.name "RRR Battery" 2>/dev/null || true
    RESULT_JSON="${BATTERY_DIR}/comprehensive_results.json"
    CONFIG_JSON="${BATTERY_DIR}/config.json"
    git add "$RESULT_JSON" "$CONFIG_JSON" 2>/dev/null || true
    git commit -m "Comprehensive battery: $TIMESTAMP" 2>/dev/null || true
    git push 2>/dev/null || true
    echo "[Git] Results pushed"
fi

echo ""
echo "========================================================================"
echo "  COMPREHENSIVE BATTERY COMPLETE"
echo "  Results: ${BATTERY_DIR}/comprehensive_results.json"
echo "  Log:     ${LOG}"
echo "========================================================================"