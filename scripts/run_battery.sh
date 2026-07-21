#!/usr/bin/env bash
# =============================================================================
# run_battery.sh -- Step 2 of the RRR replication package
#
# Runs the comprehensive evaluation battery over preprocessed artifacts.
# Produces a timestamped results directory under runs/.
#
# Usage:
#   ./scripts/run_battery.sh
#
# Prerequisites:
#   - Step 1 completed (metadata.csv, data/page_text/, indices/ all present)
#   - Ollama installed and running (ollama serve)
#   - Default model pulled (ollama pull mistral-small:24b; the preflight below
#     pulls it automatically). Non-Latin topics also need qwen3:14b.
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Resolve paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo ""
echo "=============================================="
echo "  RRR Replication -- Step 2: Evaluation Battery"
echo "=============================================="
echo ""
echo "  Package directory : $PACKAGE_DIR"
echo ""

cd "$PACKAGE_DIR"

# ---------------------------------------------------------------------------
# 1. Check preprocessing artifacts
# ---------------------------------------------------------------------------
echo "[1/6] Checking preprocessing artifacts..."

ERRORS=0

if [ ! -f "metadata.csv" ]; then
    echo "       ERROR: metadata.csv not found."
    echo "       Run Step 1 first: ./scripts/preprocess.sh"
    ERRORS=$((ERRORS + 1))
fi

if [ ! -f "indices/bm25.pkl" ]; then
    echo "       ERROR: indices/bm25.pkl not found."
    echo "       Run Step 1 first: ./scripts/preprocess.sh"
    ERRORS=$((ERRORS + 1))
fi

if [ ! -d "data/page_text" ]; then
    echo "       ERROR: data/page_text/ not found."
    echo "       Run Step 1 first: ./scripts/preprocess.sh"
    ERRORS=$((ERRORS + 1))
fi

if [ "$ERRORS" -gt 0 ]; then
    echo ""
    echo "  Preprocessing artifacts are missing. Cannot proceed."
    echo "  Complete Step 1 first, then re-run this script."
    exit 1
fi

PAGE_COUNT=$(find data/page_text -name "*.txt" | wc -l | tr -d ' ')
echo "       metadata.csv: found"
echo "       indices/bm25.pkl: found"
echo "       data/page_text/: $PAGE_COUNT page-text files"
echo "       Artifacts OK."
echo ""

# ---------------------------------------------------------------------------
# 2. Check Ollama
# ---------------------------------------------------------------------------
echo "[2/6] Checking Ollama..."

OLLAMA_URL="${OLLAMA_HOST:-http://127.0.0.1:11434}"

if ! curl -fsS "$OLLAMA_URL/api/version" &>/dev/null; then
    echo ""
    echo "       ERROR: Ollama is not reachable at $OLLAMA_URL"
    echo ""
    echo "       Start Ollama with:"
    echo ""
    echo "         ollama serve &"
    echo ""
    echo "       Then re-run this script."
    echo ""
    echo "       If Ollama is running on a different host or port, set:"
    echo "         export OLLAMA_HOST=http://your-host:port"
    exit 1
fi
echo "       Ollama is running at $OLLAMA_URL"

# Check for the default Latin-tier model (the English battery topics route to
# it via language.py). v15.15: the default is mistral-small:24b, not mistral 7B.
# To benchmark another model (Mistral 7B, a Gemma tier, etc.) set
# RRR_MODEL_LATIN before running — the preflight and the pipeline both honour it.
LATIN_MODEL="${RRR_MODEL_LATIN:-mistral-small:24b}"
if ! ollama list 2>/dev/null | grep -q "^${LATIN_MODEL%%:*}"; then
    echo "       ${LATIN_MODEL} not found. Pulling now (this may take a while)..."
    ollama pull "$LATIN_MODEL"
fi
echo "       Latin-tier model: ${LATIN_MODEL} available"
echo ""

# ---------------------------------------------------------------------------
# 3. Activate virtual environment
# ---------------------------------------------------------------------------
echo "[3/6] Activating Python environment..."

VENV_DIR="$PACKAGE_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "       ERROR: Virtual environment not found at $VENV_DIR"
    echo "       Run Step 1 first: ./scripts/preprocess.sh"
    exit 1
fi

source "$VENV_DIR/bin/activate"
echo "       Python: $(which python)"
echo ""

# ---------------------------------------------------------------------------
# 4. Set environment variables (paper's battery configuration)
# ---------------------------------------------------------------------------
echo "[4/6] Setting battery configuration..."
echo ""
echo "       These values match Appendix B (Table A2) of the paper:"
echo ""

# Pipeline parameters
export RRR_PER_DOC_TOPK=8
export RRR_MAX_SENTS_PAGE=10
export RRR_MIN_DOC_SNIPS=2
export RRR_GLOBAL_MIN_DOCS=8
export RRR_MIN_SENT_SCORE=40

# Model and concurrency
# v15.15: no RRR_MODEL pin. The language router (language.py) selects the model
# per topic — mistral-small:24b for Latin scripts, qwen3:14b for non-Latin —
# and would override an RRR_MODEL pin anyway. To benchmark a specific model,
# export RRR_MODEL_LATIN / RRR_MODEL_NONLATIN before running this script.
export RRR_CONCURRENCY=2
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=1

# v15.16: the "Writer settings" block that used to live here exported five
# knobs retired in v13 (RRR_WRITE_REVIEW, RRR_WRITER_CTX, RRR_WRITER_PRED,
# RRR_WRITER_TOPM_MECH, RRR_WRITER_TOPL_PER_MECH) — no-ops that the run
# manifest then recorded as if they configured the run. Writer tuning is
# frozen in src/rrr/writer.py; see README Appendix B.

# Paths
export PYTHONPATH="$PACKAGE_DIR/src"

echo "       RRR_PER_DOC_TOPK       = $RRR_PER_DOC_TOPK"
echo "       RRR_MAX_SENTS_PAGE     = $RRR_MAX_SENTS_PAGE"
echo "       RRR_MIN_DOC_SNIPS      = $RRR_MIN_DOC_SNIPS"
echo "       RRR_GLOBAL_MIN_DOCS    = $RRR_GLOBAL_MIN_DOCS"
echo "       RRR_MIN_SENT_SCORE     = $RRR_MIN_SENT_SCORE"
echo "       Model (Latin/non-Latin)= ${RRR_MODEL_LATIN:-mistral-small:24b} / ${RRR_MODEL_NONLATIN:-qwen3:14b} (language-routed)"
echo "       RRR_CONCURRENCY        = $RRR_CONCURRENCY"
echo "       Stage caches           = cold per run (battery default RRR_STAGE_CACHE=0)"
echo ""

# ---------------------------------------------------------------------------
# 5. Run battery
# ---------------------------------------------------------------------------
echo "[5/6] Preparing and running the comprehensive battery..."
echo ""
echo "       This core block executes 1,010 attempts across A, B, B2,"
echo "       C, C2, D, and F. Phase E is derived from completed outputs."
echo "       Runtime depends on model tier, hardware, and provider limits."
echo "       Use reduced N_* values for a smoke run before the full block."
echo ""
echo "       Progress will be printed below."
echo "       -----------------------------------------------"
echo ""

# Phase B uses RRR_BYPASS_VALIDATION=1 at runtime. No source patching is needed.

# Make the inner battery script executable
chmod +x scripts/run_comprehensive_battery.sh

# Record start time
START_TIME=$(date +%s)

# Run the battery
./scripts/run_comprehensive_battery.sh

END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo "       -----------------------------------------------"
echo "       Battery completed in approximately $ELAPSED minutes."
echo ""

# ---------------------------------------------------------------------------
# 6. Locate and summarize results
# ---------------------------------------------------------------------------
echo "[6/6] Locating results..."

# Find the most recent comprehensive results directory
RESULTS_DIR=$(ls -dt runs/comprehensive_* 2>/dev/null | head -1)

if [ -z "$RESULTS_DIR" ]; then
    echo "       WARNING: No results directory found under runs/"
    echo "       Check the output above for errors."
    exit 1
fi

RESULTS_FILE="$RESULTS_DIR/comprehensive_results.json"

if [ ! -f "$RESULTS_FILE" ]; then
    echo "       WARNING: comprehensive_results.json not found in $RESULTS_DIR"
    exit 1
fi

echo "       Results saved to: $RESULTS_FILE"
echo ""

# Print the output and deposited-reference locations.
echo "=============================================="
echo "  Quick comparison with reference results"
echo "=============================================="
echo ""

python -c "
import os
print('  See the full results in:')
print(f'    {os.path.abspath(\"$RESULTS_FILE\")}')
print()
print('  Compare against the deposited tables under:')
print(f'    {os.path.abspath(\"results/corrected/tables\")}')
"

echo ""
echo "=============================================="
echo "  Replication complete."
echo "=============================================="
echo ""
echo "  Your results are in: $RESULTS_FILE"
echo "  Reference results:   results/corrected/analysis_source/conditions/"
echo ""
echo "  To inspect individual run artifacts, look inside:"
echo "    $RESULTS_DIR/"
echo ""
