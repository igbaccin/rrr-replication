#!/usr/bin/env bash
# ============================================================================
# run_claude_code_arm.sh — Phase H: off-the-shelf agentic baseline (v16.8)
# Answers R2.3 / R1.4 / editor: "do the 100 run exercise with similar
# prompts with Claude Code and see how the procedures stack up."
#
# NOT a port of RRR. The scholarly contract is stated as INSTRUCTIONS in
# CLAUDE.md; nothing enforces it. The agent is given the SAME extracted text
# RRR works from (one file per document, [p.N] page markers) and does its
# OWN reading/retrieval — exactly what a researcher's agent would do. Raw
# PDFs are deliberately NOT used: Claude Code cannot ingest 50 PDFs (it
# errors "PDF too large"), and that is a separate one-line finding, not the
# comparison. Every essay is scored by the field's own checker
# (scripts/check_citations.py) against the same metadata as every other
# condition.
#
# REVIEWER REPRODUCTION: the only requirement is a working `claude` on PATH
# (Claude Code installed and logged in) plus a completed Step 1 (preprocess,
# which produced data/page_text/). See docs/reproduction/claude_arms.md.
#
# Usage:
#   ./scripts/run_claude_code_arm.sh                # N=100, default topic
#   N_ARM=3 ./scripts/run_claude_code_arm.sh        # smoke first!
#   ARM_MODEL=claude-sonnet-5 ./scripts/run_claude_code_arm.sh
# ============================================================================
set -euo pipefail

N_ARM="${N_ARM:-100}"
TOPIC="${ARM_TOPIC:-Institutions are the fundamental cause of long-run economic growth}"
# v16.8: PIN the model. "Account default" is not reproducible — Anthropic
# changes it over time, so a reviewer could silently get a different model.
# Default = Sonnet 4.5 (the current stock Claude Code default, empirically
# confirmed). Run the Opus arm with:
#   ARM_MODEL=claude-opus-4-8 ./scripts/run_claude_code_arm.sh
# NB: the bare alias "opus" resolves to 4.1, NOT 4.8 — always pin the id.
ARM_MODEL="${ARM_MODEL:-claude-sonnet-4-5-20250929}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-20}" # seconds; be polite to rate limits

# v16.9: graceful handling of Max/API usage limits. When a limit is reached,
# `claude -p` fails (non-zero rc, little/no stdout). Without a guard the loop
# would grind through every remaining run emitting empty essays that score as
# refusals — inflating the refusal rate and burning wall-clock. Instead we
# retry a failed run a few times with backoff (rides out short rolling
# windows) and, if failures persist, STOP cleanly: keep the completed essays,
# discard the trailing failed runs, and score what we have. Re-run later
# (limits reset) to top up N — each invocation is its own timestamped
# workspace. These defaults are inert on a healthy run (they only trigger on
# real consecutive infrastructure failures).
ARM_MIN_WORDS="${ARM_MIN_WORDS:-400}"    # below this = failed/no-output. v16.14 [auditor P2]: MUST match the emission-failure floor in score_claude_arm.py (was 150; runner kept 150-399w outputs the scorer then dropped as emissions)
ARM_MAX_RETRY="${ARM_MAX_RETRY:-2}"      # retries per run when a usage/rate limit is detected
ARM_ABORT_AFTER="${ARM_ABORT_AFTER:-4}"  # consecutive infra-failed runs -> stop the batch cleanly

# 0 (match) if a run's stderr looks like a usage/rate limit rather than a
# model refusal — used to decide whether to retry and whether to abort.
_looks_like_limit() {
    grep -qiE 'usage limit|rate limit|too many requests|overloaded|reset[s]? (at|in)|\b(429|529)\b' "$1" 2>/dev/null
}

# v16.10.1: 0 (match) if a run's output looks like an AUTH failure. The CLI
# prints these to STDOUT (they end up in review_composed.md), e.g.
#   API Error: 401 {"type":"authentication_error",...} · Please run /login
# Backoff can never heal an expired login, so abort IMMEDIATELY with an
# operator-facing message instead of burning retries.
_looks_like_auth() {
    grep -qiE 'authentication_error|invalid authentication|\b401\b|please run /login' "$1" "$2" 2>/dev/null
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
META="${META:-$PACKAGE_DIR/metadata.csv}"
PAGE_TEXT_DIR="${PAGE_TEXT_DIR:-$PACKAGE_DIR/data/page_text}"

# v16.8: label the arm by model tier so the Sonnet and Opus arms write
# distinct result files (phase_h_sonnet / phase_h_opus) that
# make_paper_artifacts.py reports as separate rows.
case "$ARM_MODEL" in
    *opus*)   ARM_LABEL="${ARM_LABEL:-phase_h_opus}" ;;
    *sonnet*) ARM_LABEL="${ARM_LABEL:-phase_h_sonnet}" ;;
    *haiku*)  ARM_LABEL="${ARM_LABEL:-phase_h_haiku}" ;;
    *)        ARM_LABEL="${ARM_LABEL:-phase_h}" ;;
esac

# ── Preconditions (fail loud, no environment archaeology) ──────────────────
if ! command -v claude >/dev/null 2>&1; then
    echo "ERROR: 'claude' is not on your PATH."
    echo "  Install Claude Code (https://claude.com/claude-code), run 'claude'"
    echo "  once to log in, then re-run this script from the same shell."
    exit 1
fi
if ! ls "$PAGE_TEXT_DIR"/*_page_*.txt >/dev/null 2>&1; then
    echo "ERROR: no extracted page-text at $PAGE_TEXT_DIR"
    echo "  Run Step 1 (preprocess) first — it produces data/page_text/."
    exit 1
fi

ARM_WS="${ARM_WS:-$PACKAGE_DIR/arm_outputs/${ARM_LABEL}_$(date +%Y%m%dT%H%M%S)}"
WS="$ARM_WS/workspace"
mkdir -p "$WS/corpus_text" "$ARM_WS/runs"

# Assemble the text corpus the agent will read (deterministic, from Step 1).
python "$SCRIPT_DIR/build_corpus_text.py" --page-text "$PAGE_TEXT_DIR" --out "$WS/corpus_text"
cp "$SCRIPT_DIR/claude_arm/CLAUDE.md" "$WS/CLAUDE.md"
N_DOC=$(ls "$WS/corpus_text"/*.txt 2>/dev/null | wc -l)
echo "[H] workspace: $WS ($N_DOC document text files)"
echo "[H] model: $ARM_MODEL  label: $ARM_LABEL"
[ "$N_DOC" -gt 0 ] || { echo "[H] ERROR: corpus_text is empty"; exit 1; }

TASK="Write a literature review on the following topic, under the contract in CLAUDE.md, using only the documents in ./corpus_text: ${TOPIC}"

completed=0
consec_fail=0
declare -a fail_dirs=()
for i in $(seq 1 "$N_ARM"); do
    RUN_DIR="$ARM_WS/runs/run_$(printf '%03d' "$i")"
    mkdir -p "$RUN_DIR"
    printf "[H] run %d/%d ... " "$i" "$N_ARM"
    T0=$(date +%s)
    attempt=0
    ok=0
    while : ; do
        attempt=$(( attempt + 1 ))
        set +e
        # ARM_MODEL is always pinned (v16.8) — reproducible by construction.
        (cd "$WS" && claude -p "$TASK" --model "$ARM_MODEL") \
            > "$RUN_DIR/review_composed.md" 2> "$RUN_DIR/stderr.log"
        RC=$?
        set -e
        WORDS=$(wc -w < "$RUN_DIR/review_composed.md" 2>/dev/null || echo 0)
        if [ "$RC" -eq 0 ] && [ "$WORDS" -ge "$ARM_MIN_WORDS" ]; then
            ok=1
            break
        fi
        # v16.10.1: expired login — retrying is pointless; stop the whole
        # batch NOW with a clear operator message. Completed essays stay.
        if _looks_like_auth "$RUN_DIR/review_composed.md" "$RUN_DIR/stderr.log"; then
            echo ""
            echo "[H] AUTH FAILURE: the Claude CLI login has expired."
            echo "[H]   $(head -c 200 "$RUN_DIR/review_composed.md" 2>/dev/null)"
            echo "[H]   Re-login (run 'claude' in your terminal, then /login),"
            echo "[H]   then re-run this script; kept $completed completed essay(s)."
            rm -rf "$RUN_DIR"
            exit 3
        fi
        # Failed run that looks like a usage/rate limit, with retries left:
        # back off and re-run the SAME run (each attempt truncates the essay).
        if _looks_like_limit "$RUN_DIR/stderr.log" && [ "$attempt" -le "$ARM_MAX_RETRY" ]; then
            backoff=$(( attempt * 60 ))
            printf "limit? backoff %ds (retry %d/%d) ... " "$backoff" "$attempt" "$ARM_MAX_RETRY"
            sleep "$backoff"
            continue
        fi
        break
    done
    echo "$RC" > "$RUN_DIR/returncode.txt"
    echo "$(( $(date +%s) - T0 ))" > "$RUN_DIR/elapsed_s.txt"
    if [ "$ok" -eq 1 ]; then
        echo "rc=$RC words=$WORDS"
        completed=$(( completed + 1 ))
        consec_fail=0
        fail_dirs=()
    elif [ "$RC" -eq 0 ] && ! _looks_like_limit "$RUN_DIR/stderr.log"; then
        # rc=0 but short: a genuine model refusal / short output, NOT a limit.
        # Keep it (the scorer records it as a refusal) and do not count it as
        # an infrastructure failure.
        echo "rc=$RC words=$WORDS  [short output — kept as refusal]"
        consec_fail=0
        fail_dirs=()
    else
        # Infrastructure failure (non-zero rc or a limit signature): count it
        # toward the abort streak; discard these runs if we bail out.
        echo "rc=$RC words=$WORDS  [FAILED — infra/limit]"
        consec_fail=$(( consec_fail + 1 ))
        fail_dirs+=("$RUN_DIR")
        if [ "$consec_fail" -ge "$ARM_ABORT_AFTER" ]; then
            echo "[H] aborting: $consec_fail consecutive failed runs (usage limit likely)."
            echo "[H] kept $completed completed essay(s); discarding $consec_fail trailing failed run(s)."
            # v16.10.1: preserve the failure evidence BEFORE discarding — the
            # 2026-07-04 401 incident was undiagnosable because the guard
            # deleted the only copies of the error text.
            for d in "${fail_dirs[@]}"; do
                {
                    echo "===== $(basename "$d") rc=$(cat "$d/returncode.txt" 2>/dev/null) ====="
                    echo "--- stdout (review_composed.md head) ---"
                    head -c 400 "$d/review_composed.md" 2>/dev/null; echo
                    echo "--- stderr ---"
                    tail -c 400 "$d/stderr.log" 2>/dev/null; echo
                } >> "$ARM_WS/failed_runs.log"
            done
            for d in "${fail_dirs[@]}"; do rm -rf "$d"; done
            echo "[H] failure evidence preserved at $ARM_WS/failed_runs.log"
            echo "[H] limits reset over time — re-run this script later to collect more essays."
            break
        fi
    fi
    sleep "$SLEEP_BETWEEN"
done

echo "[H] scoring with the field's own checker..."
python "$SCRIPT_DIR/score_claude_arm.py" "$ARM_WS" --metadata "$META" --label "$ARM_LABEL" --arm-model "$ARM_MODEL"
echo "[H] done: $ARM_WS/${ARM_LABEL}_results.json"
