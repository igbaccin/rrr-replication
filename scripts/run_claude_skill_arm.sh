#!/usr/bin/env bash
# ============================================================================
# run_claude_skill_arm.sh, corrected Phase H3 skill condition
#
# Counterpart to Phase H. There, Claude Code was ASKED to follow the
# scholarly contract (CLAUDE.md instructions, nothing enforced). Here,
# Claude Code carries the field-owned tool itself: the rrr skill
# orchestrates the actual RRR pipeline, every validator intact. Together
# Together they test whether the RRR contract can operate through the
# commercial agentic interface.
#
# Each released review is scored by check_citations.py. The runner also
# preserves the review, ledger, manifests, metrics, checker output, and agent
# transcript.
#
# Usage:
#   N_ARM=10 ./scripts/run_claude_skill_arm.sh        # smoke with N_ARM=1 first
#   ARM_RUNTIME=api ANTHROPIC_API_KEY=... ./scripts/run_claude_skill_arm.sh
#   ARM_RUNTIME=local ./scripts/run_claude_skill_arm.sh   # needs local ollama
# ============================================================================
set -euo pipefail

N_ARM="${N_ARM:-10}"
TOPIC="${ARM_TOPIC:-Institutions are the fundamental cause of long-run economic growth}"
ARM_RUNTIME="${ARM_RUNTIME:-api}"    # api | local
SLEEP_BETWEEN="${SLEEP_BETWEEN:-20}"
# v16.14 [auditor P3]: PIN the orchestration model (the Claude that DRIVES the
# skill), same as Phase H. "Account default" drifts, so an unpinned H3 is not
# reproducible. This is separate from RRR's inner runtime (ARM_RUNTIME):
# RRR writes the review; this only fixes which model runs the skill harness.
ARM_MODEL="${ARM_MODEL:-claude-opus-4-8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CORPUS_SRC="${CORPUS_SRC:-$PACKAGE_DIR/corpus}"

ARM_WS="${ARM_WS:-$PACKAGE_DIR/arm_outputs/claude_skill_$(date +%Y%m%dT%H%M%S)}"
WS="$ARM_WS/workspace"
mkdir -p "$WS/corpus" "$WS/.claude/skills/rrr" "$ARM_WS/runs"

# Isolate ALL RRR state (data, page_text, indices, claim_cache, runs) UNDER the
# arm workspace. Without this, rrr.paths.project_root() falls back to the
# installed package root, so the pipeline can write into and overwrite the
# repo/install tree, and the collector below (which globs "$WS"/runs) finds
# nothing ("NO PIPELINE OUTPUT"). A Windows venv needs a native path (cygpath);
# Linux uses the POSIX path unchanged.
export RRR_PROJECT_ROOT="$(cygpath -w "$WS" 2>/dev/null || echo "$WS")"

# The agent's workspace: PDFs + the skill. v16.5: the AUDITED local checkout
# is preinstalled (editable) so the skill's "install" step resolves to this
# exact code, not whatever a network install would fetch. v16.6: in API mode
# pull the [api] extra (anthropic/openai live only there, not in
# requirements.txt); in local mode clear any inherited RRR_RUNTIME=api so
# RRR's inner calls actually use local Ollama.
if [ "$ARM_RUNTIME" = "api" ]; then
    PIP_EXTRA="[api]"
else
    PIP_EXTRA=""
    unset RRR_RUNTIME
fi
# Install editable using a RELATIVE target from the package dir. A Windows-venv
# pip (Git Bash) can't parse the absolute POSIX path "$PACKAGE_DIR[api]"
# ("/d/RRR[api]" is not a valid editable requirement); "." from inside the dir
# is equivalent on Linux and works on Windows too.
( cd "$PACKAGE_DIR" && python -m pip install --quiet -e ".${PIP_EXTRA}" ) \
    || { echo "[H3] ERROR: editable install of .${PIP_EXTRA} from $PACKAGE_DIR failed"; exit 1; }
SKILL_SOURCE="$PACKAGE_DIR/skills/rrr/SKILL.md"
if [ ! -f "$SKILL_SOURCE" ]; then
    SKILL_SOURCE="$SCRIPT_DIR/claude_skill/rrr/SKILL.md"
fi
[ -f "$SKILL_SOURCE" ] || { echo "[H3] ERROR: rrr skill file not found"; exit 1; }
cp "$SKILL_SOURCE" "$WS/.claude/skills/rrr/SKILL.md"
cp "$PACKAGE_DIR/bibliography.bib" "$WS/" 2>/dev/null || true
find -L "$CORPUS_SRC" -maxdepth 1 -iname '*.pdf' -exec cp {} "$WS/corpus/" \;
N_PDF=$(ls "$WS/corpus" | wc -l)
echo "[H3] workspace: $WS ($N_PDF PDFs, runtime=$ARM_RUNTIME)"
[ "$N_PDF" -gt 0 ] || { echo "[H3] ERROR: no PDFs at $CORPUS_SRC"; exit 1; }

if [ "$ARM_RUNTIME" = "api" ]; then
    : "${ANTHROPIC_API_KEY:?ARM_RUNTIME=api requires ANTHROPIC_API_KEY}"
    export RRR_RUNTIME=api
fi

TASK="Use the rrr skill to produce a literature review on this topic from the PDFs in ./corpus (bibliography.bib is the sidecar): ${TOPIC}. If ingest reports pending rows, accept them explicitly. I approve corpus rows for this benchmark run."

for i in $(seq 1 "$N_ARM"); do
    RUN_DIR="$ARM_WS/runs/run_$(printf '%03d' "$i")"
    mkdir -p "$RUN_DIR"
    printf "[H3] run %d/%d ... " "$i" "$N_ARM"
    T0=$(date +%s)
    set +e
    (cd "$WS" && claude -p "$TASK" --model "$ARM_MODEL") > "$RUN_DIR/agent_transcript.md" 2> "$RUN_DIR/stderr.log"
    RC=$?
    set -e
    echo "$RC" > "$RUN_DIR/returncode.txt"
    echo "$(( $(date +%s) - T0 ))" > "$RUN_DIR/elapsed_s.txt"

    # Collect the PIPELINE's output (not the agent's words): newest run dir
    # the RRR pipeline wrote inside the workspace.
    NEWEST=$(ls -td "$WS"/runs/*/ 2>/dev/null | head -1 || true)
    if [ -n "$NEWEST" ] && [ -f "${NEWEST}review_composed.md" ]; then
        for f in review_composed.md review_ledger.json run_manifest.json \
                 quality_manifest.json citations.json topic_fit.json run_metrics.json; do
            [ -f "${NEWEST}${f}" ] && cp "${NEWEST}${f}" "$RUN_DIR/" 2>/dev/null || true
        done
        # keep the checker's namespace clean (same collision rule as the battery)
        [ -f "$RUN_DIR/citations.json" ] && mv "$RUN_DIR/citations.json" "$RUN_DIR/citations_manifest.json"
        WORDS=$(wc -w < "$RUN_DIR/review_composed.md")
        echo "rc=$RC words=$WORDS (pipeline artifacts collected)"
    else
        echo "rc=$RC NO PIPELINE OUTPUT (agent transcript kept)"
    fi
    rm -rf "$WS"/runs/*/ 2>/dev/null || true
    sleep "$SLEEP_BETWEEN"
done

echo "[H3] scoring with the field's own checker..."
python "$SCRIPT_DIR/score_claude_arm.py" "$ARM_WS" \
    --metadata "$PACKAGE_DIR/metadata.csv" \
    --label phase_h3 \
    --arm-model "$ARM_MODEL"
echo "[H3] done: $ARM_WS/phase_h3_results.json"
