#!/usr/bin/env bash
# v15.10 gibberish smoke — validates the four new refusal-hardening changes:
#   #1 CLI topic-sanity gate
#   #2 Early Stage 0 precheck (before per_document_sweep)
#   #3 Split refusal reasons (no_admitted_docs / stage0_llm_failed /
#      corpus_off_topic / stage1_clustering_failed)
#   #4 Stage 0 prompt tightened against gibberish word-salad
#
# Runs three cases:
#   A. keyboard slam ("09p<GYHKLGCH")   — expected: CLI-gate reject, exit 2
#   B. incoherent English               — expected: corpus_off_topic
#   C. control (well-formed topic)      — expected: PROCEED to review
#
# Emits durations + final refusal_reason from run_metrics.json for each.
set -uo pipefail

METADATA="${METADATA:-metadata.csv}"
OUT="${OUT:-runs_v15_10_gibberish_smoke}"
mkdir -p "$OUT"

run_case() {
    local label="$1"
    local topic="$2"
    local expect="$3"

    echo
    echo "========================================================================"
    echo "CASE: $label"
    echo "TOPIC: $topic"
    echo "EXPECT: $expect"
    echo "========================================================================"

    export RRR_RUN_ID="v15_10_${label}"
    local t0=$(date +%s)
    set +e
    python -m rrr.cli t2 --metadata "$METADATA" --topic "$topic" --multi \
        > "$OUT/${label}.stdout.log" 2> "$OUT/${label}.stderr.log"
    local rc=$?
    set -e
    local t1=$(date +%s)
    local dur=$((t1 - t0))

    local refusal_reason="(none/success)"
    local metrics_file="runs/${RRR_RUN_ID}/run_metrics.json"
    if [[ -f "$metrics_file" ]]; then
        # v15.10: metrics.set() writes to the 'values' sub-dict, not top-level.
        refusal_reason=$(python -c "import json,sys; d=json.load(open(r'$metrics_file')); v=d.get('values') or {}; print(v.get('refusal_reason') or ('SUCCESS' if not v.get('refusal') else 'refusal_but_no_reason'))" 2>/dev/null || echo "(parse-fail)")
        cp "$metrics_file" "$OUT/${label}.run_metrics.json"
    fi
    if [[ -f "runs/${RRR_RUN_ID}/outline_plan.json" ]]; then
        cp "runs/${RRR_RUN_ID}/outline_plan.json" "$OUT/${label}.outline_plan.json"
    fi

    printf "RESULT: rc=%d duration=%ds refusal_reason=%s\n" "$rc" "$dur" "$refusal_reason"
    printf "%s,%s,%d,%d,%s\n" "$label" "$expect" "$rc" "$dur" "$refusal_reason" >> "$OUT/summary.csv"
}

echo "label,expected,rc,duration_s,refusal_reason" > "$OUT/summary.csv"

run_case "keyboard_slam"     "09p<GYHKLGCH"                            "CLI_reject_exit_2"
run_case "incoherent_english" "milk house getting asked whole table"    "corpus_off_topic"
run_case "control_ok"         "Institutions are the fundamental cause of long-run economic growth" "PROCEED_or_review"

echo
echo "========================================================================"
echo "SUMMARY"
echo "========================================================================"
cat "$OUT/summary.csv"
