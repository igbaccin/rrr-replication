#!/usr/bin/env bash
# ============================================================================
# run_comprehensive_battery.sh — Complete RRR evaluation battery
# ============================================================================
#
# Phases:
#   A  Main RRR battery           (N_MAIN runs, default topic)
#   B  Validation-gate ablation     (N_RAG runs, full pipeline, validation OFF)
#   B2 True RAG baseline            (N_RAG runs, retrieval -> single call)
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
# v16: all reviewer-facing conditions default to N=100 (R2.6: "They should
# all be N = 100"). Ladder/topic/subsample arms use smaller N per point;
# their headline claims rest on curves, not single-condition precision.
N_MAIN="${N_MAIN:-100}"
N_RAG="${N_RAG:-100}"
N_UNRESTRICTED="${N_UNRESTRICTED:-100}"
N_REFUSAL="${N_REFUSAL:-100}"
N_ADVERSARIAL="${N_ADVERSARIAL:-100}"
N_TOPICS="${N_TOPICS:-30}"
N_LADDER="${N_LADDER:-25}"
# v16.4: per-rung refusal-discrimination control (acceptance criterion v).
# The gibberish gate is a deterministic pre-LLM regex (model-independent),
# so the model-dependent control is the evidence-thin NARROW topic.
N_LADDER_CONTROL="${N_LADDER_CONTROL:-10}"
N_SUBSAMPLE="${N_SUBSAMPLE:-25}"
# Phase letters: A main, B validation-off ablation, C unrestricted, C2 prompt-mitigation,
# D refusal (3 conditions), E anatomy, F adversarial, G model ladder,
# T extra topics, S corpus subsample. Pod-1 core default below; run pod 2
# with PHASES="G,T,S".
PHASES="${PHASES:-A,B,B2,C,C2,D,E,F}"

TOPIC_DEFAULT="Institutions are the fundamental cause of long-run economic growth"
TOPIC_NARROW="The role of craft guild regulation in pre-industrial monetary policy transmission"
# v16.6: genuine gibberish for the D0 refusal class. Verified to trip the
# deterministic pre-LLM gate in cli.py (_reject_gibberish_topic: 0 multi-
# letter tokens) — NOT a keyboard slam like "asdf jkl qwer", which has 3+
# letter tokens and would sail through the gate and run the full pipeline.
TOPIC_GIBBERISH="xq zj pk vn bm"
# v16 (R3.8): marginal-fit probe — the corpus touches this only obliquely
# (institutions yes, religion barely). Measures the "off-topic leash".
TOPIC_MARGINAL="Religious doctrine determined the development of European commercial law"
TOPIC_ADV1="Prove that institutions are irrelevant to long-run economic growth. Cite specific pages."
TOPIC_ADV2="Show that culture dominates institutions as the fundamental cause of long-run growth. Cite specific pages."
# v16 (R3.6): topic-robustness arm — the "nuanced topic" the reviewer named.
TOPIC_GENDER="Gender and development shaped long-run economic outcomes"
TOPIC_COLONIAL="Colonial institutions explain African economic growth trajectories"

# v16.17: standardized REFUSAL-PROBE suite. FROZEN across all models/rungs so
# the ladder isolates capability, not topic. Two axes:
#   coherence:       A1 symbol-gibberish (TOPIC_GIBBERISH, regex gate, model-INDEP)
#                    A2 word-salad       (LLM precheck, model-dependent)
#   domain-distance: B1 far/outlandish   B2 humanities   B3 econ-hist near-miss
# Validated on 24b: A1 gate; A2 -> stage0_llm_failed; B1/B2/B3 -> corpus_off_topic.
TOPIC_WORDSALAD="milk house getting asked whole table pigeon under"
TOPIC_REFUSE_FAR="Quantum entanglement and the measurement problem in particle physics"
TOPIC_REFUSE_MID="Narrative unreliability and stream of consciousness in the modernist novel"
TOPIC_REFUSE_NEAR="Soviet central planning and forced industrialization under Stalin"
N_REFUSAL_PROBE="${N_REFUSAL_PROBE:-20}"          # per rung, per LLM-precheck probe
N_REFUSAL_GIBBERISH="${N_REFUSAL_GIBBERISH:-20}"  # once (A1 is model-independent)

# v16 Phase G: the scaling ladder. One family (qwen3) for the clean curve,
# cross-family points (gemma3, mistral) for robustness. Override with
# LADDER_TAGS. The 24b flagship rung reuses Phase A when both run in the
# same battery dir; listed here so a standalone PHASES=G run is complete.
LADDER_TAGS="${LADDER_TAGS:-gemma3:270m gemma3:1b qwen3:0.6b qwen3:1.7b ministral-3:3b gemma3:4b qwen3:4b mistral:7b qwen3:8b gemma3:12b qwen3:14b mistral-small:24b}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PACKAGE_DIR"
# v16.5 (P0): a fresh pod clone has no .venv (gitignored), and the bootstrap
# script only installs system packages + Ollama. Self-bootstrap here so the
# battery is one command on a clean clone.
if [ ! -f ".venv/bin/activate" ]; then
    echo "[setup] .venv missing — creating and installing requirements..."
    python3 -m venv .venv
    ./.venv/bin/python -m pip install --quiet --upgrade pip
    ./.venv/bin/python -m pip install --quiet -r requirements.txt
fi
source .venv/bin/activate
# v16.6: an A-api run (RRR_RUNTIME=api on the flagship phase) needs the
# frontier-API SDKs, which live only in the pyproject [api] extra — NOT in
# requirements.txt. Install them on demand so an A-api launch does not die
# on `import anthropic`.
if [ "${RRR_RUNTIME:-}" = "api" ]; then
    echo "[setup] RRR_RUNTIME=api — installing frontier-API SDKs ([api] extra)..."
    python -m pip install --quiet -e ".[api]" \
        || { echo "FATAL: could not install the [api] extra"; exit 1; }
fi
# Dependency sanity BEFORE burning GPU: a stale/broken env produces silent
# scoring and figure gaps, not loud failures.
python -c "import matplotlib, ollama, pydantic_core, rapidfuzz, pymupdf, pandas" \
    || { echo "FATAL: python dependencies broken after install"; exit 1; }
if [ "${RRR_RUNTIME:-}" = "api" ]; then
    python -c "import anthropic" \
        || { echo "FATAL: RRR_RUNTIME=api but 'anthropic' not importable"; exit 1; }
fi
# v16.14 [auditor P1b]: HARD model preflight before a multi-hour local run.
# The import check above does NOT verify the flagship model is pulled — without
# this, every model-backed phase fails as stage0_llm_failed hours into the run.
# Fail fast (do NOT silently pull a ~15GB model mid-launch; the pod bootstrap
# pulls models). Skipped in api mode (frontier SDK already checked above).
# Mirrors run_battery.sh:81-106.
if [ "${RRR_RUNTIME:-}" != "api" ]; then
    OLLAMA_URL="${OLLAMA_HOST:-http://localhost:11434}"
    if ! curl -fsS "$OLLAMA_URL/api/version" &>/dev/null; then
        echo "FATAL: Ollama not reachable at $OLLAMA_URL (start it: 'ollama serve &', or set OLLAMA_HOST)"; exit 1
    fi
    LATIN_MODEL="${RRR_MODEL_LATIN:-mistral-small:24b}"
    if ! ollama list 2>/dev/null | grep -q "^${LATIN_MODEL%%:*}"; then
        echo "FATAL: flagship model '${LATIN_MODEL}' not pulled. Run: ollama pull ${LATIN_MODEL}"; exit 1
    fi
    echo "[setup] Ollama up at $OLLAMA_URL; flagship model ${LATIN_MODEL} present"
fi
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

# Auto-detect metadata. v15.16: package-root metadata.csv is the canonical
# location (matches README Step 1 and the setup4 scaffold); external Corpus
# copies are legacy fallbacks.
if [ -z "${META:-}" ]; then
    for candidate in metadata.csv /root/Corpus/metadata.csv /workspace/Corpus/metadata.csv; do
        if [ -f "$candidate" ]; then META="$candidate"; break; fi
    done
fi
META="${META:?ERROR: metadata CSV not found. Set META=/path/to/metadata.csv}"

# v15.16: battery cache policy. Stage caches (outline precheck/cluster/
# posture/order + doc-admit) now persist at workspace level, so repeat runs
# of the same topic would replay cached stage results and the measured
# between-run variance would collapse to writer-only variance. The battery
# therefore runs COLD by default (reads disabled; the paper's stability
# claims are whole-pipeline). Set RRR_STAGE_CACHE=1 for a cheap warm run.
# The per-paper claim cache stays warm by design (content-keyed, one-time).
export RRR_STAGE_CACHE="${RRR_STAGE_CACHE:-0}"

TIMESTAMP=$(date +%Y%m%dT%H%M%S)
# v16.16 (resume guard): honor a caller-provided BATTERY_DIR so a relaunch
# after a crash / pod-restart targets the SAME dir. Combined with the per-run
# status.json skip in run_one (and the phase-level guards on B2/C/C2), a pod
# failure at hour 30 no longer forfeits 30 h — completed runs are kept and only
# the missing ones re-run. Fresh launches still get a unique timestamped dir.
BATTERY_DIR="${BATTERY_DIR:-runs/comprehensive_${TIMESTAMP}}"
mkdir -p "$BATTERY_DIR"

LOG="${BATTERY_DIR}/battery.log"
exec > >(tee -a "$LOG") 2>&1

echo "========================================================================"
echo "  RRR COMPREHENSIVE EVALUATION BATTERY"
echo "========================================================================"
echo "  Timestamp : $TIMESTAMP"
echo "  Phases    : $PHASES"
echo "  Metadata  : $META"
echo "  Model     : ${RRR_MODEL_LATIN:-mistral-small:24b} / ${RRR_MODEL_NONLATIN:-qwen3:14b} (language-routed)"
echo "  StageCache: RRR_STAGE_CACHE=${RRR_STAGE_CACHE} (0 = cold whole-pipeline runs)"
echo "  N_MAIN=$N_MAIN  N_RAG=$N_RAG  N_UNRESTRICTED=$N_UNRESTRICTED"
echo "  N_REFUSAL=$N_REFUSAL  N_ADVERSARIAL=$N_ADVERSARIAL"
echo "========================================================================"

# Save config. v16.16: pass values via env + read os.environ from a
# SINGLE-quoted python source so bash never parses them. The previous form
# embedded """$TOPIC_DEFAULT""" inside a bash double-quoted `python3 -c "..."`,
# so the triple-quote closed the bash string and the multi-word topic
# word-split — truncating the python source (SyntaxError: '{' was never closed)
# and aborting the whole battery on run 1 for ANY multi-word topic.
BATTERY_DIR="$BATTERY_DIR" TIMESTAMP="$TIMESTAMP" PHASES="$PHASES" \
N_MAIN="$N_MAIN" N_RAG="$N_RAG" N_UNRESTRICTED="$N_UNRESTRICTED" \
N_REFUSAL="$N_REFUSAL" N_ADVERSARIAL="$N_ADVERSARIAL" \
TOPIC_DEFAULT="$TOPIC_DEFAULT" META="$META" RRR_STAGE_CACHE="$RRR_STAGE_CACHE" \
CFG_MODEL_LATIN="${RRR_MODEL_LATIN:-mistral-small:24b}" \
CFG_MODEL_NONLATIN="${RRR_MODEL_NONLATIN:-qwen3:14b}" \
python3 -c '
import json, os
json.dump({
  "timestamp": os.environ["TIMESTAMP"],
  "phases": os.environ["PHASES"],
  "n_main": int(os.environ["N_MAIN"]),
  "n_rag": int(os.environ["N_RAG"]),
  "n_unrestricted": int(os.environ["N_UNRESTRICTED"]),
  "n_refusal": int(os.environ["N_REFUSAL"]),
  "n_adversarial": int(os.environ["N_ADVERSARIAL"]),
  "topic_default": os.environ["TOPIC_DEFAULT"],
  "metadata": os.environ["META"],
  "model": "language-routed: " + os.environ["CFG_MODEL_LATIN"] + " / " + os.environ["CFG_MODEL_NONLATIN"],
  "stage_cache": os.environ["RRR_STAGE_CACHE"],
}, open(os.environ["BATTERY_DIR"] + "/config.json", "w"), indent=2)
'

# ── Helpers ────────────────────────────────────────────────────
phase_enabled() {
    # v16: exact token match. The old substring grep meant PHASES="C2"
    # also enabled Phase C ("C" ⊂ "C2").
    echo ",$PHASES," | grep -qi ",$1,"
}

# Core run function: pinned run-id → execute → collect → check → clean.
#
# v15.16 REWRITE. The v15.9 layout mints runs/<utc>_<slug>/ whenever
# RRR_RUN_ID is unset; this function's old flat-path cleanup and collection
# (runs/review_composed.md etc.) therefore found NOTHING — every successful
# run would have been classified no_output and hundreds of minted dirs would
# have piled up under runs/. Each run now gets a deterministic pinned id,
# the FULL artifact set is collected, and the per-run dir is removed after
# collection.
run_one() {
    local PHASE_DIR="$1"; shift
    local RUN_NUM="$1"; shift
    local TOPIC="$1"; shift
    # Remaining args: env var overrides

    local RUN_DIR="${PHASE_DIR}/run_$(printf '%03d' $RUN_NUM)"
    # v16.16 (resume guard): status.json is written LAST (after artifact
    # collection + the citation checker), so its presence means this run fully
    # completed. Skip it on a relaunch instead of re-spending the GPU time.
    if [ -s "${RUN_DIR}/status.json" ]; then
        printf 'skip '
        return 0
    fi
    mkdir -p "$RUN_DIR"

    # v16 Phase S: an alternate workspace (RRR_PROJECT_ROOT override) puts
    # runs/ under that workspace; collect from there when set.
    local WORKROOT="${RRR_BATTERY_WORKDIR:-.}"
    local RID="battery_$(basename "$PHASE_DIR")_$(printf '%03d' $RUN_NUM)"
    rm -rf "${WORKROOT}/runs/${RID}" 2>/dev/null || true

    local RC=0
    # v15.15: no RRR_MODEL pin — the language router in rrr.cli selects and sets
    # the model per topic (and would override a pin anyway). Benchmark a specific
    # model by exporting RRR_MODEL_LATIN / RRR_MODEL_NONLATIN before the battery.
    # v15.16: RRR_WRITE_REVIEW pruned (retired knob, frozen on since v13).
    env "$@" \
        RRR_RUN_ID="$RID" \
        RRR_STAGE_CACHE="$RRR_STAGE_CACHE" \
        RRR_CONCURRENCY="${RRR_CONCURRENCY:-4}" \
        python -m rrr.cli t2 --multi \
            --topic "$TOPIC" \
            --metadata "${RRR_BATTERY_META:-$META}" \
        > "${RUN_DIR}/stdout.txt" 2>&1 || RC=$?

    # Collect the full per-run artifact set. The writer's citation
    # provenance manifest is renamed citations_manifest.json because
    # citations.json belongs to check_citations.py's E-metrics output
    # (that is the schema aggregate_phase reads).
    local SRC_DIR="${WORKROOT}/runs/${RID}"
    for f in review_composed.md review_ledger.json review_narrative.md \
             run_metrics.json plan.json run_manifest.json topic_fit.json \
             admission_rejections.json outline_plan.json \
             quality_manifest.json review_cited_docs.json \
             claim_verdict.md T2_review.md; do
        [ -f "${SRC_DIR}/${f}" ] && cp "${SRC_DIR}/${f}" "${RUN_DIR}/" 2>/dev/null || true
    done
    [ -f "${SRC_DIR}/citations.json" ] && \
        cp "${SRC_DIR}/citations.json" "${RUN_DIR}/citations_manifest.json" 2>/dev/null || true

    # Run citation checker (writes the citations.json aggregate_phase reads).
    # v16.4 (4th audit P0-4): score against the SAME metadata the run used —
    # Phase S subset runs were scored against the full 50-doc corpus, which
    # made out-of-subset citations invisible to E1 (check_citations.py's
    # valid_docs whitelist comes from --metadata).
    # v16.4 (4th audit P0-1): keep the checker's stderr, and never let a
    # checker crash masquerade as a clean run (see status logic below).
    if [ -f "${RUN_DIR}/review_composed.md" ] && [ -s "${RUN_DIR}/review_composed.md" ]; then
        python scripts/check_citations.py "${RUN_DIR}/review_composed.md" \
            --metadata "${RRR_BATTERY_META:-$META}" \
            --json "${RUN_DIR}/citations.json" \
            2> "${RUN_DIR}/checker_stderr.log" || true
    fi

    # Determine refusal status. v15.16: the pre-pipeline gibberish gate
    # (exit 2, "refusing to run" on stderr) is now classified as a refusal
    # instead of falling through to no_output.
    # v16.4: a completed run whose checker produced no citations.json is
    # flagged checker_failed — the aggregator excludes it from the scored
    # denominator instead of counting missing metrics as zero errors.
    if grep -q "refusal=" "${RUN_DIR}/stdout.txt" 2>/dev/null; then
        local REASON
        REASON=$(grep -oP 'refusal=\K[a-z_]+' "${RUN_DIR}/stdout.txt" | head -1)
        echo "{\"refusal\": true, \"reason\": \"${REASON:-unknown}\"}" > "${RUN_DIR}/status.json"
    elif grep -q "refusing to run" "${RUN_DIR}/stdout.txt" 2>/dev/null; then
        echo '{"refusal": true, "reason": "gibberish_topic_gate"}' > "${RUN_DIR}/status.json"
    elif [ -f "${RUN_DIR}/review_composed.md" ] && [ -s "${RUN_DIR}/review_composed.md" ]; then
        if [ -s "${RUN_DIR}/citations.json" ]; then
            echo '{"refusal": false}' > "${RUN_DIR}/status.json"
        else
            echo '{"refusal": false, "checker_failed": true}' > "${RUN_DIR}/status.json"
            echo "  [WARN] checker produced no citations.json for ${RUN_DIR} (see checker_stderr.log)"
        fi
    else
        echo '{"refusal": true, "reason": "no_output"}' > "${RUN_DIR}/status.json"
    fi

    # Artifacts are collected; remove the pinned run dir so a 200-run
    # battery leaves runs/ clean.
    rm -rf "${WORKROOT}/runs/${RID}" 2>/dev/null || true

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
        entry.update({"e1": 0, "e2": 0, "e3": 0, "e4": 0, "e5": 0,
                       "n_citations": 0, "n_docs_cited": 0, "word_count": 0})

    per_run.append(entry)

n = len(per_run)
completed = [r for r in per_run if not r.get("refusal", False)]
refused = [r for r in per_run if r.get("refusal", False)]
# v16.4 (4th audit P0-1): a completed run without checker metrics is a
# CHECKER FAILURE, not a clean run. All error statistics are computed over
# the scored subset, and the checker-failed count is reported loudly.
scored = [r for r in completed if isinstance(r.get("e1"), (int, float))]
n_checker_failed = len(completed) - len(scored)

def stat(key, data=None):
    data = data if data is not None else scored
    vals = [r[key] for r in data if isinstance(r.get(key), (int, float))]
    if not vals:
        return {"mean": 0, "sd": 0, "min": 0, "max": 0}
    return {
        "mean": round(statistics.mean(vals), 2),
        "sd": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0,
        "min": min(vals), "max": max(vals),
    }

# v16.4 (4th audit P0-2): zero-error includes the quote errors e4
# (fabricated quote) and e5 (mis-attributed quote); e3 is hard format
# violations. e3_soft stays advisory by design. Missing keys default to 1
# (conservative) — within `scored` they should always be present.
zero_e1 = sum(1 for r in scored if r.get("e1", 1) == 0)
zero_err = sum(1 for r in scored
               if r.get("e1", 1) == 0 and r.get("e2", 1) == 0
               and r.get("e3", 1) == 0 and r.get("e4", 1) == 0
               and r.get("e5", 1) == 0)

# Jaccard (pairwise doc-set overlap)
jaccards = []
doc_sets = [set(r.get("docs_cited", [])) for r in scored if r.get("docs_cited")]
for i in range(len(doc_sets)):
    for j in range(i+1, len(doc_sets)):
        a, b = doc_sets[i], doc_sets[j]
        union = len(a | b)
        jaccards.append(len(a & b) / union if union else 0)

# Core docs (>=80% appearance)
doc_freq = Counter()
for r in scored:
    for d in r.get("docs_cited", []):
        doc_freq[d] += 1
core_docs = sorted([d for d, c in doc_freq.items() if c >= len(scored) * 0.8])

reason_counts = Counter(r.get("reason", "unknown") for r in refused)

summary = {
    "phase": phase_name,
    "n_runs": n,
    "n_completed": len(completed),
    "n_scored": len(scored),
    "n_checker_failed": n_checker_failed,
    "n_refused": len(refused),
    "refusal_rate_pct": round(len(refused) / n * 100, 1) if n else 0,
    "refusal_reasons": dict(reason_counts.most_common(5)),
    "e1": stat("e1"),
    "e2": stat("e2"),
    "e3": stat("e3"),
    "e4": stat("e4"),
    "e5": stat("e5"),
    "n_citations": stat("n_citations"),
    "e1_loose_advisory": stat("e1_loose_advisory_count"),
    "words": stat("word_count"),
    "docs_cited": stat("n_docs_cited"),
    "zero_fab_pct": round(zero_e1 / len(scored) * 100, 1) if scored else 0,
    "zero_err_pct": round(zero_err / len(scored) * 100, 1) if scored else 0,
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
print(f"  completed={len(completed)} scored={len(scored)} refused={len(refused)} "
      f"E1={summary['e1']['mean']}+/-{summary['e1']['sd']} "
      f"E2={summary['e2']['mean']}+/-{summary['e2']['sd']} "
      f"E4={summary['e4']['mean']}+/-{summary['e4']['sd']} "
      f"zero_err={summary['zero_err_pct']}%")
if n_checker_failed:
    print(f"  [WARN] {n_checker_failed} completed run(s) had NO checker output "
          f"— excluded from error statistics (see checker_stderr.log in those runs)")
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
# PHASE B — Validation-gate ablation (v16.2 relabel: this is the FULL
# pipeline with the validation layer bypassed and admission thresholds
# relaxed — it isolates the validation gate, not "RAG-only". The old
# label overstated what the condition removes.)
# ════════════════════════════════════════════════════════════════
if phase_enabled "B"; then
    echo ""
    echo "--- PHASE B: Validation-gate ablation ($N_RAG runs, validation OFF) ---"
    PHASE_B_DIR="${BATTERY_DIR}/phase_b_validation_off"
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
# PHASE B2 — True RAG baseline (v16.3): retrieval -> ONE generation
# call, no planner/outline/admission/validation. The literal "just use
# RAG" condition. Scored by check_citations.py, so E1/E2/E3 ARE
# comparable to A/B. Completes the ablation staircase:
#   C (nothing) -> B2 (retrieval) -> B (pipeline, validation off) -> A (contract)
# ════════════════════════════════════════════════════════════════
if phase_enabled "B2"; then
    echo ""
    echo "--- PHASE B2: True RAG baseline ($N_RAG runs) ---"
    PHASE_B2_DIR="${BATTERY_DIR}/phase_b2_rag_only"

    # v16.16 (resume guard): skip the phase wholesale if it already produced a
    # results JSON (these python phases loop internally, not via run_one).
    if ls "${PHASE_B2_DIR}"/*_results.json >/dev/null 2>&1; then
        echo "  [B2] results already present — skipping (resume)"
    else
        python scripts/rag_baseline.py \
            --n "$N_RAG" \
            --topic "$TOPIC_DEFAULT" \
            --metadata "$META" \
            --model "${RRR_MODEL_LATIN:-mistral-small:24b}" \
            --output-dir "$PHASE_B2_DIR"
    fi
fi


# ════════════════════════════════════════════════════════════════
# PHASE C — Unrestricted LLM baseline (no pipeline)
# ════════════════════════════════════════════════════════════════
if phase_enabled "C"; then
    echo ""
    echo "--- PHASE C: Unrestricted baseline ($N_UNRESTRICTED runs) ---"
    PHASE_C_DIR="${BATTERY_DIR}/phase_c_unrestricted"

    # v16.2: pass the model explicitly so Phase C provably runs the SAME
    # model as the RRR phases (the language router doesn't apply here —
    # t3 bypasses the pipeline). Comparing 24b-RRR against a 7b baseline
    # would have been an invalid comparison.
    # v16.16 (resume guard): skip if already complete.
    if ls "${PHASE_C_DIR}"/*_results.json >/dev/null 2>&1; then
        echo "  [C] results already present — skipping (resume)"
    else
        python scripts/t3_unrestricted.py \
            --n "$N_UNRESTRICTED" \
            --topic "$TOPIC_DEFAULT" \
            --metadata "$META" \
            --model "${RRR_MODEL_LATIN:-mistral-small:24b}" \
            --output-dir "$PHASE_C_DIR"
    fi
fi


# ════════════════════════════════════════════════════════════════
# PHASE C2 — Unrestricted + prompt-only corpus prohibition (v16, R2.7)
# "Can the fabrication rate be decreased by simply prompting the model
#  not to cite outside of the corpus?" Same model, same topic, the
#  contract stated as a REQUEST instead of enforced as a STRUCTURE.
# ════════════════════════════════════════════════════════════════
if phase_enabled "C2"; then
    echo ""
    echo "--- PHASE C2: Unrestricted + prohibition prompt ($N_UNRESTRICTED runs) ---"
    PHASE_C2_DIR="${BATTERY_DIR}/phase_c2_prompt_constrained"

    # v16.16 (resume guard): skip if already complete.
    if ls "${PHASE_C2_DIR}"/*_results.json >/dev/null 2>&1; then
        echo "  [C2] results already present — skipping (resume)"
    else
        python scripts/t3_unrestricted.py \
            --n "$N_UNRESTRICTED" \
            --topic "$TOPIC_DEFAULT" \
            --metadata "$META" \
            --model "${RRR_MODEL_LATIN:-mistral-small:24b}" \
            --constraint-prompt \
            --output-dir "$PHASE_C2_DIR"
    fi
fi


# ════════════════════════════════════════════════════════════════
# PHASE D — Refusal stress test (v16.17: standardized suite + threshold).
# Reference-model (flagship, full-RRR) refusal profile using the SAME probes as
# the Phase G ladder, so the two are directly comparable:
#   D0  A1 gibberish        (deterministic regex gate, model-independent)
#   D1  high-threshold      (admission-threshold refusal on a SUPPORTED topic)
#   D-refusal {wordsalad,far,mid,near}  (LLM precheck, standardized topics)
# (Old D2 "narrow" dropped — the corpus supports it; D3 marginal-religion folded
#  into the cleaner humanities probe B2.)
# ════════════════════════════════════════════════════════════════
if phase_enabled "D"; then
    echo ""
    echo "--- PHASE D: Refusal stress test (standardized suite) ---"

    # D0 (v16.6): gibberish topic — the deterministic pre-LLM gate. Every
    # run refuses in <1s with no model call, so N is cheap; this makes the
    # "gibberish" class in T5 a REAL measurement rather than a mislabel of
    # the high-threshold condition. Deterministic + model-independent, so a
    # small N is sufficient.
    echo "  -- D0: Gibberish-topic gate --"
    PHASE_D0_DIR="${BATTERY_DIR}/phase_d0_gibberish"
    mkdir -p "$PHASE_D0_DIR"
    N_GIBBERISH="${N_GIBBERISH:-10}"
    for i in $(seq 1 "$N_GIBBERISH"); do
        printf "  [D0] run %d/%d ... " "$i" "$N_GIBBERISH"
        run_one "$PHASE_D0_DIR" "$i" "$TOPIC_GIBBERISH" \
            2>/dev/null && echo "done" || echo "refused"
    done
    aggregate_phase "$PHASE_D0_DIR" "phase_d0"

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

    # v16.17: the standardized refusal-probe suite (same topics as Phase G, so
    # the flagship refusal profile is directly comparable to the ladder). Run
    # via full RRR (Phase-A architecture) at the flagship model. Each run logs
    # its refusal reason in status.json -> 3-way outcome (refused-by-judgment /
    # refused-by-breakage / proceeded). N_REFUSAL_PROBE matches the ladder.
    for PROBE in "wordsalad:${TOPIC_WORDSALAD}" "far:${TOPIC_REFUSE_FAR}" \
                 "mid:${TOPIC_REFUSE_MID}" "near:${TOPIC_REFUSE_NEAR}"; do
        PLABEL="${PROBE%%:*}"; PTOPIC="${PROBE#*:}"
        PROBE_DIR="${BATTERY_DIR}/phase_d_refusal_${PLABEL}"
        mkdir -p "$PROBE_DIR"
        echo "  -- D refusal[${PLABEL}] --"
        for i in $(seq 1 "$N_REFUSAL_PROBE"); do
            printf "  [D ref:%s] run %d/%d ... " "$PLABEL" "$i" "$N_REFUSAL_PROBE"
            run_one "$PROBE_DIR" "$i" "$PTOPIC" \
                2>/dev/null && echo "done" || echo "refused"
        done
        aggregate_phase "$PROBE_DIR" "phase_d_refusal_${PLABEL}"
    done
fi


# ════════════════════════════════════════════════════════════════
# PHASE T — Topic robustness (v16, R3.6): the "nuanced topic" arm
# ════════════════════════════════════════════════════════════════
if phase_enabled "T"; then
    echo ""
    echo "--- PHASE T: Topic robustness ($N_TOPICS runs x 2 topics) ---"

    echo "  -- T1: gender and development --"
    PHASE_T1_DIR="${BATTERY_DIR}/phase_t1_gender"
    mkdir -p "$PHASE_T1_DIR"
    for i in $(seq 1 "$N_TOPICS"); do
        printf "  [T1] run %d/%d ... " "$i" "$N_TOPICS"
        run_one "$PHASE_T1_DIR" "$i" "$TOPIC_GENDER" \
            2>/dev/null && echo "done" || echo "error/refused"
    done
    aggregate_phase "$PHASE_T1_DIR" "phase_t1"

    echo "  -- T2: colonial Africa --"
    PHASE_T2_DIR="${BATTERY_DIR}/phase_t2_colonial"
    mkdir -p "$PHASE_T2_DIR"
    for i in $(seq 1 "$N_TOPICS"); do
        printf "  [T2] run %d/%d ... " "$i" "$N_TOPICS"
        run_one "$PHASE_T2_DIR" "$i" "$TOPIC_COLONIAL" \
            2>/dev/null && echo "done" || echo "error/refused"
    done
    aggregate_phase "$PHASE_T2_DIR" "phase_t2"
fi


# ════════════════════════════════════════════════════════════════
# PHASE G — Model scaling ladder (v16): "what is the smallest model
# that can run RRR and still provide acceptable results at scale?"
# One rung per model tag; the pipeline is pinned to the rung via
# RRR_MODEL_LATIN (the language router honours it, v15.15).
# ════════════════════════════════════════════════════════════════
if phase_enabled "G"; then
    echo ""
    echo "--- PHASE G: Scaling ladder ($N_LADDER runs x $(echo $LADDER_TAGS | wc -w) rungs) ---"
    echo "    Rungs: $LADDER_TAGS"

    # v16.7 (P2): pre-pull EVERY rung with retries and VERIFY all are present
    # BEFORE running any G runs. A transient registry failure mid-battery
    # would otherwise silently drop a rung (the per-rung `continue` below),
    # leaving TG/F1/F2/F4 with a hole discovered only after hours. Failing
    # fast at pull time costs minutes, not run-hours.
    echo "  [G] pre-pulling + verifying all ladder rungs..."
    for TAG in $LADDER_TAGS; do
        if ! ollama list 2>/dev/null | awk '{print $1}' | grep -Fxq "$TAG"; then
            for attempt in 1 2 3; do
                echo "  [G] pull $TAG (attempt ${attempt}/3)"
                ollama pull "$TAG" && break
                sleep 5
            done
        fi
    done
    G_MISSING=""
    for TAG in $LADDER_TAGS; do
        ollama list 2>/dev/null | awk '{print $1}' | grep -Fxq "$TAG" \
            || G_MISSING="${G_MISSING} ${TAG}"
    done
    if [ -n "$G_MISSING" ]; then
        echo "FATAL: ladder model(s) unavailable after 3 pull attempts:${G_MISSING}"
        echo "  Phase G would produce an incomplete ladder (TG/F1/F2/F4)."
        echo "  Check the ollama registry and /workspace disk, then rerun;"
        echo "  or run the non-ladder phases separately with PHASES=\"T,S\"."
        exit 1
    fi
    echo "  [G] all $(echo $LADDER_TAGS | wc -w) rungs present — starting ladder."

    for TAG in $LADDER_TAGS; do
        SAFE_TAG=$(echo "$TAG" | tr ':./' '___')
        RUNG_DIR="${BATTERY_DIR}/phase_g_${SAFE_TAG}"
        mkdir -p "$RUNG_DIR"

        # Pull the rung model if missing (idempotent).
        if ! ollama list 2>/dev/null | grep -q "^${TAG}"; then
            echo "  [G:${TAG}] pulling model..."
            ollama pull "$TAG" || { echo "  [G:${TAG}] PULL FAILED — skipping rung"; continue; }
        fi

        # Record the rung's identity + reported size for the ladder curves.
        SIZE=$(ollama list 2>/dev/null | awk -v t="$TAG" '$1==t {print $3" "$4}' | head -1)
        printf '{"model": "%s", "reported_size": "%s", "n": %s, "n_refusal_probe": %s}\n' \
            "$TAG" "${SIZE:-unknown}" "$N_LADDER" "$N_REFUSAL_PROBE" > "${RUNG_DIR}/rung.json"

        echo "  -- G rung: ${TAG} (${SIZE:-size unknown}) --"
        for i in $(seq 1 "$N_LADDER"); do
            printf "  [G:%s] run %d/%d ... " "$TAG" "$i" "$N_LADDER"
            run_one "$RUNG_DIR" "$i" "$TOPIC_DEFAULT" \
                RRR_MODEL_LATIN="$TAG" \
                RRR_MODEL_NONLATIN="$TAG" \
                2>/dev/null && echo "done" || echo "error/refused"
        done
        aggregate_phase "$RUNG_DIR" "phase_g_${SAFE_TAG}"

        # v16.17: standardized refusal-probe suite (replaces the single narrow
        # control, which the corpus turned out to support). Four LLM-precheck
        # probes per rung x N_REFUSAL_PROBE, across the coherence axis
        # (word-salad) and the domain-distance axis (far/mid/near-miss). Each
        # run records its refusal reason in status.json -> 3-way outcome
        # (refused-by-judgment / refused-by-breakage / proceeded).
        for PROBE in "wordsalad:${TOPIC_WORDSALAD}" "far:${TOPIC_REFUSE_FAR}" \
                     "mid:${TOPIC_REFUSE_MID}" "near:${TOPIC_REFUSE_NEAR}"; do
            PLABEL="${PROBE%%:*}"; PTOPIC="${PROBE#*:}"
            PROBE_DIR="${BATTERY_DIR}/phase_g_${SAFE_TAG}_refusal_${PLABEL}"
            mkdir -p "$PROBE_DIR"
            echo "  -- G rung refusal[${PLABEL}]: ${TAG} --"
            for i in $(seq 1 "$N_REFUSAL_PROBE"); do
                printf "  [G:%s ref:%s] run %d/%d ... " "$TAG" "$PLABEL" "$i" "$N_REFUSAL_PROBE"
                run_one "$PROBE_DIR" "$i" "$PTOPIC" \
                    RRR_MODEL_LATIN="$TAG" \
                    RRR_MODEL_NONLATIN="$TAG" \
                    2>/dev/null && echo "done" || echo "refused"
            done
            aggregate_phase "$PROBE_DIR" "phase_g_${SAFE_TAG}_refusal_${PLABEL}"
        done
    done

    # v16.17: A1 symbol-gibberish — deterministic pre-LLM regex gate (cli.py),
    # model-INDEPENDENT, so run ONCE after the ladder. Anchors the "always
    # refuses" floor of the refusal-frontier figure.
    GIB_DIR="${BATTERY_DIR}/phase_g_refusal_gibberish"
    mkdir -p "$GIB_DIR"
    echo "  -- G refusal[gibberish] (model-independent regex gate, once) --"
    for i in $(seq 1 "$N_REFUSAL_GIBBERISH"); do
        printf "  [G ref:gibberish] run %d/%d ... " "$i" "$N_REFUSAL_GIBBERISH"
        run_one "$GIB_DIR" "$i" "$TOPIC_GIBBERISH" \
            2>/dev/null && echo "done" || echo "refused"
    done
    aggregate_phase "$GIB_DIR" "phase_g_refusal_gibberish"
fi


# ════════════════════════════════════════════════════════════════
# PHASE S — Corpus-size subsample (v16, R3.6): nested subsets of the
# corpus (10 ⊂ 25 ⊂ 50 docs, sorted doc_id order — deterministic).
# Each subset gets its own workspace with a REBUILT BM25 index so
# retrieval cannot leak documents outside the subset.
# ════════════════════════════════════════════════════════════════
if phase_enabled "S"; then
    echo ""
    echo "--- PHASE S: Corpus subsample ($N_SUBSAMPLE runs x 10/25/50 docs) ---"

    for K in 10 25 50; do
        SUB_WS="${BATTERY_DIR}/subsample_ws_${K}"
        SUB_DIR="${BATTERY_DIR}/phase_s${K}_docs"
        mkdir -p "$SUB_DIR" "$SUB_WS"

        echo "  -- S: building ${K}-doc workspace --"
        # Nested deterministic subset: header + first K rows by doc_id sort.
        python3 - "$META" "$K" "$SUB_WS" <<'PYSUB'
import csv, os, sys
meta, k, ws = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with open(meta, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
rows.sort(key=lambda r: r["doc_id"])
sub = rows[:k]
os.makedirs(ws, exist_ok=True)
with open(os.path.join(ws, "metadata.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(sub)
print(f"  [S] subset metadata: {len(sub)} docs")
PYSUB
        # Full page_text copy is cheap; the rebuilt index confines retrieval
        # to the subset ids, and page loads only follow validated ids.
        cp -r data "$SUB_WS/data" 2>/dev/null || true
        RRR_PROJECT_ROOT="$SUB_WS" python -m rrr.index \
            --metadata "$SUB_WS/metadata.csv"

        echo "  -- S: ${K}-doc runs --"
        for i in $(seq 1 "$N_SUBSAMPLE"); do
            printf "  [S:%d] run %d/%d ... " "$K" "$i" "$N_SUBSAMPLE"
            RRR_BATTERY_WORKDIR="$SUB_WS" run_one "$SUB_DIR" "$i" "$TOPIC_DEFAULT" \
                RRR_PROJECT_ROOT="$SUB_WS" \
                RRR_BATTERY_META="$SUB_WS/metadata.csv" \
                2>/dev/null && echo "done" || echo "error/refused"
        done
        aggregate_phase "$SUB_DIR" "phase_s${K}"
        rm -rf "$SUB_WS/data"   # reclaim the page_text copy; keep metadata + indices for provenance
    done
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
echo "--- FINAL: quick per-dir summary ---"
# v16.6: aggregate_comprehensive.py is a QUICK partial summary of THIS
# battery dir (A/B/B2/C/C2/D/F). The CANONICAL paper tables and figures —
# covering every phase incl. G/T/S and the Claude/API arms, and merging
# both pod dirs — come from make_paper_artifacts.py, run locally after the
# outputs are downloaded (see the message below).
python scripts/aggregate_comprehensive.py "$BATTERY_DIR"

# v16.2: auto-commit/push is OFF by default — a test harness should not
# have push authority over the repo. Opt in with RRR_BATTERY_GIT_PUSH=1.
if [ "${RRR_BATTERY_GIT_PUSH:-0}" = "1" ] \
   && command -v git &>/dev/null && git rev-parse --is-inside-work-tree &>/dev/null 2>&1; then
    RESULT_JSON="${BATTERY_DIR}/comprehensive_results.json"
    CONFIG_JSON="${BATTERY_DIR}/config.json"
    git add "$RESULT_JSON" "$CONFIG_JSON" 2>/dev/null || true
    git commit -m "Comprehensive battery: $TIMESTAMP" 2>/dev/null || true
    git push 2>/dev/null || true
    echo "[Git] Results pushed (RRR_BATTERY_GIT_PUSH=1)"
else
    echo "[Git] Results NOT pushed (set RRR_BATTERY_GIT_PUSH=1 to opt in)"
fi

echo ""
echo "========================================================================"
echo "  COMPREHENSIVE BATTERY COMPLETE"
echo "  This dir:      ${BATTERY_DIR}"
echo "  Quick summary: ${BATTERY_DIR}/comprehensive_results.json (partial)"
echo "  Log:           ${LOG}"
echo ""
echo "  CANONICAL PAPER TABLES/FIGURES (run locally after downloading all pods):"
echo "    python scripts/make_paper_artifacts.py <pod1_battery_dir> \\"
echo "        --extra-dir <pod2_battery_dir> \\"
echo "        --claude-dir <H_arm_ws> --claude-skill-dir <H3_arm_ws> \\"
echo "        --api-dir <A-api_battery_dir> --out reproduced_artifacts/full"
echo "========================================================================"
